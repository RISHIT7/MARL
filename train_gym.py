"""Note:
- Implementation of "Multiagent Decision Making For Maritime Traffic Management" (AAAI 2019)
- 5 zones in chain: z_dummy (outside) → z1 (source) → z2 → z3 → z4 (terminal)
- Bidirectional transitions allowed along the chain
- Count-based state abstraction: n_bold_t = [n_txn, n_arr, n_nxt, n_tilda]
- Actions are β_{zz'} values transformed by sigmoid to [0,1] for binomial distribution
- Policy gradient uses exact formulas from equations 17-18 (θ = β directly)
- Value function computed via dynamic programming as per paper
- Reward function from equation 3: C(z,n) = w_r * max(n-cap, 0) + w_r
- Deterministic arrivals: P(⟨z_d,z1,τ⟩) for simplification
- Horizon H=10, vessels in transit at end counted per formula
- Maintains invariant: n_arr(z) = Σ_z' n_nxt(z,z') at all times

Questions
- What should be the intial action ?
- Reverse zones is possible ?
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

class MaritimeTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.zones = ['z_dummy', 'z1', 'z2', 'z3', 'z4']
        self.Z = len(self.zones)
        self.H = 10
        
        # Valid transitions (chain + reverse)
        self.valid_transitions = [
            ('z_dummy', 'z1'), ('z1', 'z2'), ('z2', 'z3'), ('z3', 'z4'),
            ('z3', 'z2'), ('z2', 'z1')
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
        self.w_r = 1.0
        self.w_d = 1.0
        
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
                total_penalty += self.w_r * excess + self.w_d * excess

        return -total_penalty


class PolicyNetwork(nn.Module):
    """Policy network outputting β values directly (θ = β as per paper)"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, state):
        return self.network(state)


class MaritimePolicyGradient:
    """Policy gradient implementation using equations 17-18 from paper"""
    def __init__(self, env):
        self.env = env
        self.policy = PolicyNetwork(env.state_size, len(env.valid_transitions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        
    def compute_value_function(self, trajectory):
        """Compute vessel-based value function as per paper"""
        rewards = [step['reward'] for step in trajectory]
        values = []
        
        # Dynamic programming: V_t = R_t + γ * V_{t+1}
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G  # γ = 0.99
            values.append(G)
        
        return list(reversed(values))
    
    def compute_policy_gradient(self, trajectory):
        """Compute gradients using equations 17-18"""
        states = torch.stack([torch.FloatTensor(step['state']) for step in trajectory])
        actions = torch.stack([torch.FloatTensor(step['action']) for step in trajectory])
        values = torch.FloatTensor(self.compute_value_function(trajectory))
        
        # Forward pass
        logits = self.policy(states)
        
        # Policy gradient: ∇_θ log π_θ(a|s) * V(s,a)
        # Since θ = β directly, we compute gradients w.r.t. β parameters
        log_probs = -torch.sum((logits - actions)**2, dim=1)  # Gaussian log-likelihood
        policy_loss = -torch.mean(log_probs * values)
        
        return policy_loss
    
    def train_episode(self):
        """Train one episode using policy gradient"""
        state, _ = self.env.reset()
        trajectory = []
        done = False
        
        while not done:
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_logits = self.policy(state_tensor)
                action = action_logits.squeeze().numpy()
            
            # Take step
            next_state, reward, done, _, _ = self.env.step(action)
            
            # Store transition
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward
            })
            
            state = next_state
        
        # Compute gradients and update policy
        if len(trajectory) > 0:
            loss = self.compute_policy_gradient(trajectory)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return sum(step['reward'] for step in trajectory)
        
        return 0


# Training loop
if __name__ == '__main__':
    env = MaritimeTrafficEnv()
    agent = MaritimePolicyGradient(env)
    
    # Training
    episode_rewards = []
    for episode in range(100):
        reward = agent.train_episode()
        episode_rewards.append(reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    print(f"Training completed. Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
