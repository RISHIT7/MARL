import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import numpy as np
from environment.parameters import *

class MaritimeTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.zones = ['z_dummy', 'z1', 'z2', 'z3', 'z4']
        self.Z = len(self.zones)
        self.H = HORIZON
        
        # Valid transitions (chain + reverse)
        self.valid_transitions = [
            ('z_dummy', 'z1'), ('z1', 'z2'), ('z2', 'z3'), ('z3', 'z4'),
            ('z4', 'z3'), ('z3', 'z2'), ('z2', 'z1'), ('z1', 'z_dummy')
        ]
        
        # State space dimensions
        self.state_size = 2 * (self.Z**2 * self.H) + 2 * (self.Z**2) + self.Z
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(len(self.valid_transitions),), dtype=np.float32
        )
        
        # Transit time bounds
        self.t_min = {trans: 1 for trans in self.valid_transitions}
        self.t_max = {trans: 3 for trans in self.valid_transitions}
        
        # Zone capacities for reward computation
        self.capacities = {z: 5 for z in self.zones}
        self.w_r = W_R
        self.w_d = W_D
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize count vectors
        self.n_txn = defaultdict(int)  # (z,z',τ)
        self.n_arr = defaultdict(int)  # z
        self.n_nxt = defaultdict(int)  # (z,z')
        self.n_tilda = defaultdict(int)  # (z,z',τ) 
        
        self.t = 0
        
        # Deterministic arrival schedule: P(⟨z_d,z1,1⟩) = 3
        self.arrival_schedule = {1: {'z1': 3}, 3: {'z1': 2}, 5: {'z1': 1}}
        
        # Alpha probabilities (equal for simplicity)
        self.alpha = self._compute_alpha()
        
        return self._get_state_vector(), {}

    def _compute_alpha(self):
        """Compute α(z'|z) transition probabilities"""
        alpha = {}
        for z in self.zones:
            valid_dests = [dst for src, dst in self.valid_transitions if src == z]
            if valid_dests:
                prob = 1.0 / len(valid_dests)
                alpha[z] = {dest: prob for dest in valid_dests}
            else:
                alpha[z] = {}
        return alpha

    def _get_state_vector(self):
        """Convert count dictionaries to state vector"""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0
        
        # n_txn: transit counts
        for z in self.zones:
            for z_p in self.zones:
                for tau in range(1, self.H + 1):
                    vec[idx] = self.n_txn.get((z, z_p, tau), 0)
                    idx += 1
        
        # n_arr: arrival counts
        for z in self.zones:
            vec[idx] = self.n_arr.get(z, 0)
            idx += 1
            
        # n_nxt: next destination counts
        for z in self.zones:
            for z_p in self.zones:
                vec[idx] = self.n_nxt.get((z, z_p), 0)
                idx += 1
                
        # n_tilda: committed route counts
        for z in self.zones:
            for z_p in self.zones:
                for tau in range(1, self.H + 1):
                    vec[idx] = self.n_tilda.get((z, z_p, tau), 0)
                    idx += 1
                    
        return vec

    def step(self, action):
        self.t += 1
        
        # Transform action to β values using sigmoid: β ∈ [0,1]
        beta_vals = 1.0 / (1.0 + np.exp(-action))
        beta = {trans: beta_vals[i] for i, trans in enumerate(self.valid_transitions)}
        
        # 1. Process arrivals from schedule
        arrivals = self.arrival_schedule.get(self.t, {})
        for z_src, count in arrivals.items():
            self.n_arr[z_src] += count
            
        # 2. Route decisions using α(z'|z) - multinomial distribution
        for z_src, arr_count in arrivals.items():
            if arr_count > 0 and z_src in self.alpha:
                destinations = list(self.alpha[z_src].keys())
                probabilities = list(self.alpha[z_src].values())
                
                if len(destinations) > 0:
                    counts = np.random.multinomial(arr_count, probabilities)
                    for dest, count in zip(destinations, counts):
                        self.n_nxt[(z_src, dest)] += count

        # 3. Transit time sampling using p_nav with β - binomial distribution
        for (z_src, z_dest), nxt_count in list(self.n_nxt.items()):
            if nxt_count > 0 and (z_src, z_dest) in self.valid_transitions:
                t_min = self.t_min[(z_src, z_dest)]
                t_max = self.t_max[(z_src, z_dest)]
                n_trials = t_max - t_min
                p_success = beta[(z_src, z_dest)]
                
                # Sample transit times for count
                for _ in range(nxt_count):
                    if n_trials > 0:
                        delta_tilda = np.random.binomial(n_trials, p_success)
                    else:
                        delta_tilda = 0
                    
                    tau = self.t + t_min + delta_tilda
                    if tau <= self.H:
                        self.n_tilda[(z_src, z_dest, tau)] += 1
                
                # Clear processed next destinations
                self.n_nxt[(z_src, z_dest)] = 0

        # 4. Process transit completions
        completed_transits = []
        for (z_src, z_dest, tau), count in list(self.n_tilda.items()):
            if tau == self.t and count > 0:
                completed_transits.append((z_src, z_dest, tau, count))
                
        for z_src, z_dest, tau, count in completed_transits:
            # Remove from n_tilda
            self.n_tilda[(z_src, z_dest, tau)] -= count
            if self.n_tilda[(z_src, z_dest, tau)] == 0:
                del self.n_tilda[(z_src, z_dest, tau)]
                
            # Add to destination arrivals (unless terminal zone z4)
            if z_dest != 'z4':
                self.n_arr[z_dest] += count

        # 5. Update transit counts (currently moving vessels)
        self.n_txn.clear()
        for (z_src, z_dest, tau), count in self.n_tilda.items():
            if self.t < tau <= self.H:
                self.n_txn[(z_src, z_dest, tau)] = count

        # 6. Compute reward using equation 3
        reward = self._compute_reward()
        
        # 7. Check termination
        done = self.t >= self.H
        
        return self._get_state_vector(), reward, done, False, {}

    def _compute_reward(self):
        """Compute reward using equation 3: C(z,n) = w_r * max(n-cap, 0) + w_r"""
        total_penalty = 0.0
        
        # Compute total vessels per zone
        n_tot = defaultdict(int)
        
        # Count arrivals
        for z, count in self.n_arr.items():
            n_tot[z] += count
            
        # Count vessels in transit (will arrive later)
        for (z_src, z_dest, tau), count in self.n_tilda.items():
            if tau > self.t:
                n_tot[z_dest] += count
                
        # Apply congestion penalty
        for z in self.zones:
            if z != 'z4':  # Terminal zone doesn't have congestion
                excess = max(n_tot[z] - self.capacities[z], 0)
                total_penalty += self.w_r * excess + self.w_d
                
        return -total_penalty
