import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time

class MaritimeTrafficEnv:
    """
    Final implementation of simplified maritime traffic management
    Based on Singh et al. paper with corrections for:
    - Count-based sufficient statistics
    - Proper time dynamics  
    - Bidirectional movement with α(z'|z) distributions
    - Dummy zone handling (no vessel storage)
    """
    
    def __init__(self, n_zones=4, max_vessels_per_zone=10, arrival_rate=0.3, 
                 time_horizon=100, congestion_penalty=-2.0, throughput_reward=5.0):
        # Zone setup
        self.n_zones = n_zones
        self.dummy_zone = 0  # zd - represents "outside system"
        self.zones = list(range(1, n_zones + 1))
        self.source_zone = 1
        self.terminal_zone = n_zones
        
        # Environment parameters
        self.max_vessels_per_zone = max_vessels_per_zone
        self.arrival_rate = arrival_rate
        self.time_horizon = time_horizon
        self.congestion_penalty = congestion_penalty
        self.throughput_reward = throughput_reward
        
        # Navigation time parameters
        self.t_min = 1
        self.t_max = 5
        
        # Random input probabilities from zd to z1
        np.random.seed(42)  # For reproducibility
        self.zd_to_z1_probabilities = np.random.uniform(0.1, 0.9, size=time_horizon)
        
        # α(z'|z) probability distributions (bidirectional movement)
        self.alpha = {
            1: {0: 0, 2: 1},  # z1 -> zd(exit) or z2(forward)
            2: {1: 0.3, 3: 0.7},  # z2 -> z1(back) or z3(forward)  
            3: {2: 0.3, 4: 0.7},  # z3 -> z2(back) or z4(forward)
            4: {3: 0.2, 0: 0.8}   # z4 -> z3(back) or zd(exit)
        }
        
        # Valid zone pairs for actions
        self.valid_zone_pairs = []
        for from_z in self.zones:
            for to_z in self.alpha[from_z].keys():
                self.valid_zone_pairs.append((from_z, to_z))
        
        self.reset()
    
    def reset(self):
        """Reset environment with proper initialization"""
        self.timestep = 0
        
        # Count-based state (only for navigable zones)
        self.newly_arrived_counts = defaultdict(int)  # Only zones 1-4
        self.in_transit_counts = defaultdict(int)     # Zone pairs only
        self.zone_occupancy = defaultdict(int)        # Only zones 1-4
        
        # In-transit vessel tracking
        self.in_transit_vessels = []  # (from_zone, to_zone, arrival_time)
        
        # Performance tracking
        self.total_arrivals = 0
        self.total_departures = 0
        
        # # Initialize with some vessels (realistic start)
        # initial_vessels = [1, 2, 1, 0]  # Vessels in z1, z2, z3, z4
        # for i, count in enumerate(initial_vessels):
        #     zone = i + 1
        #     self.newly_arrived_counts[zone] = count
        #     self.zone_occupancy[zone] = count
        
        return self._get_state()
    
    def _get_state(self):
        """Get count-based sufficient statistics as state"""
        state = []
        
        # Newly arrived counts (zones 1-4 only)
        for zone in self.zones:
            state.append(self.newly_arrived_counts[zone])
        
        # In-transit counts for valid zone pairs
        for zone_pair in self.valid_zone_pairs:
            state.append(self.in_transit_counts[zone_pair])
        
        # Zone occupancy (zones 1-4 only)
        for zone in self.zones:
            state.append(self.zone_occupancy[zone])
        
        return np.array(state, dtype=np.float32)
    
    def _sample_navigation_time(self, beta):
        """Sample navigation time using binomial distribution (paper's Eq. 8)"""
        trials = self.t_max - self.t_min
        successes = np.random.binomial(trials, beta)
        return self.t_min + successes
    
    def step(self, action):
        """
        Execute one timestep with proper sequencing:
        1. New arrivals at source zone using random probabilities
        2. In-transit vessels complete journeys  
        3. Newly arrived vessels make decisions
        4. Calculate rewards
        """
        action = np.clip(action, 0.1, 0.9)
        
        # Map actions to zone pairs
        zone_pair_actions = {}
        for i, zone_pair in enumerate(self.valid_zone_pairs):
            zone_pair_actions[zone_pair] = action[i]
        
        reward = 0.0
        
        # Step 1: Generate arrivals at source zone using random probabilities
        # Use time-varying probability from zd to z1
        arrival_prob = self.zd_to_z1_probabilities[self.timestep % len(self.zd_to_z1_probabilities)]
        
        # Sample arrivals based on both base rate and random probability
        expected_arrivals = self.arrival_rate * arrival_prob
        new_arrivals = np.random.poisson(expected_arrivals)
        
        if new_arrivals > 0:
            self.newly_arrived_counts[self.source_zone] += new_arrivals
            self.zone_occupancy[self.source_zone] += new_arrivals
            self.total_arrivals += new_arrivals
        
        # Step 2: Process in-transit completions
        reward += self._process_in_transit_completions()
        
        # Step 3: Process vessel decisions using α(z'|z)
        reward += self._process_vessel_decisions(zone_pair_actions)
        
        # Step 4: Calculate penalties and rewards
        reward += self._calculate_congestion_penalty()
        reward -= 0.1  # Step penalty for efficiency
        
        self.timestep += 1
        done = self.timestep >= self.time_horizon
        
        info = {
            'timestep': self.timestep,
            'total_vessels_in_system': sum(self.zone_occupancy.values()),
            'congestion_violations': self._count_violations(),
            'total_arrivals': self.total_arrivals,
            'total_departures': self.total_departures,
            'in_transit': len(self.in_transit_vessels),
            'current_arrival_prob': arrival_prob  # Add this for monitoring
        }
        
        return self._get_state(), reward, done, info
    
    def _process_in_transit_completions(self):
        """Process vessels completing their navigation"""
        reward = 0.0
        completed = []
        
        for i, (from_z, to_z, arrival_time) in enumerate(self.in_transit_vessels):
            if arrival_time == self.timestep:
                completed.append(i)
                
                # Update counts
                self.in_transit_counts[(from_z, to_z)] -= 1
                
                if to_z == self.dummy_zone:
                    # Vessel exiting system (z4 -> zd)
                    reward += self.throughput_reward
                    self.total_departures += 1
                else:
                    # Vessel arriving at navigable zone
                    self.newly_arrived_counts[to_z] += 1
                    self.zone_occupancy[to_z] += 1
        
        # Remove completed vessels
        for i in reversed(completed):
            del self.in_transit_vessels[i]
        
        return reward
    
    def _process_vessel_decisions(self, zone_pair_actions):
        """Process vessel decisions using α(z'|z) distributions"""
        reward = 0.0
        vessel_decisions = []  # Track individual vessel decisions
        
        for zone in list(self.newly_arrived_counts.keys()):
            vessel_count = self.newly_arrived_counts[zone]
            
            if vessel_count > 0 and zone in self.alpha:
                destinations = self.alpha[zone]
                
                for vessel_id in range(vessel_count):
                    # Sample destination using α(z'|z)
                    dest_zones = list(destinations.keys())
                    dest_probs = list(destinations.values())
                    chosen_dest = np.random.choice(dest_zones, p=dest_probs)
                    
                    # Get speed recommendation (beta parameter)
                    zone_pair = (zone, chosen_dest)
                    beta = zone_pair_actions.get(zone_pair, 0.5)
                    
                    # Sample navigation time
                    nav_time = self._sample_navigation_time(beta)
                    arrival_time = self.timestep + nav_time
                    
                    vessel_decisions.append({
                        'zone_pair': zone_pair,
                        'beta': beta,
                        'nav_time': nav_time,
                        'departure_time': self.timestep,
                        'arrival_time': arrival_time
                    })
                    
                    # Update counts
                    self.newly_arrived_counts[zone] -= 1
                    self.zone_occupancy[zone] -= 1
                    self.in_transit_counts[zone_pair] += 1
                    
                    # Track vessel
                    self.in_transit_vessels.append((zone, chosen_dest, arrival_time))
        
        # Store vessel decisions for policy updates
        self.last_vessel_decisions = vessel_decisions
        return reward
    
    def _calculate_congestion_penalty(self):
        """Calculate congestion penalties"""
        penalty = 0.0
        for zone in self.zones:
            if self.zone_occupancy[zone] > self.max_vessels_per_zone:
                excess = self.zone_occupancy[zone] - self.max_vessels_per_zone
                penalty += self.congestion_penalty * excess
        return penalty
    
    def _count_violations(self):
        """Count congestion violations"""
        return sum(1 for zone in self.zones 
                  if self.zone_occupancy[zone] > self.max_vessels_per_zone)

class VesselBasedPolicyGradientAgent:
    """
    vessel-based policy gradient agent implementing the paper's approach
    """
    
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.zone_pairs = env.valid_zone_pairs
        
        # Policy parameters: θ_zz' for each zone pair (paper's Eq. 11)
        self.theta_params = {zp: np.random.normal(0, 0.1, size=3) for zp in self.zone_pairs}
        
        self.state_dim = len(env._get_state())
        self.vessel_value_weights = {zp: np.random.normal(0, 0.1, self.state_dim) 
                                   for zp in self.zone_pairs}
        
        # Baseline value function (not in paper but helps with variance)
        self.baseline_weights = np.random.normal(0, 0.1, self.state_dim)
    
    def compute_vessel_policy(self, state, zone_pair):
        """
        Compute vessel policy π_θ(a|s,z,z') from paper Eq. 11
        Returns beta (speed parameter) for vessels transitioning from z to z'
        """
        theta = self.theta_params[zone_pair]
        
        # Linear policy: β = sigmoid(θ^T * φ(s))
        # Use first 3 state features for policy (can be adjusted)
        state_features = state[:len(theta)]
        logit = np.dot(theta, state_features)
        beta = 1.0 / (1.0 + np.exp(-logit))  # Sigmoid to keep β ∈ (0,1)
        return np.clip(beta, 0.1, 0.9)
    
    def select_action(self, state):
        """
        Generate actions based on current state and policy
        """
        actions = []
        for zone_pair in self.zone_pairs:
            beta = self.compute_vessel_policy(state, zone_pair)
            actions.append(beta)
        return np.array(actions)
    
    def compute_vessel_value(self, state, zone_pair):
        """
        Vessel-based value function V^π_zz'(s) from paper Eq. 12
        """
        return np.dot(self.vessel_value_weights[zone_pair], state)
    
    def compute_baseline_value(self, state):
        """Baseline value function for variance reduction"""
        return np.dot(self.baseline_weights, state)
    
    def update_policy(self, vessel_trajectories):
        """
        Individual vessel-based policy gradient updates
        vessel_trajectories: List of (zone_pair, state_sequence, action_sequence, reward_sequence)
        """
        
        # Update each zone pair's policy based on vessels that used it
        for zone_pair in self.zone_pairs:
            
            # Get trajectories for vessels that used this zone pair
            zp_trajectories = [traj for traj in vessel_trajectories if traj[0] == zone_pair]
            
            if not zp_trajectories:
                continue
            
            policy_gradient = np.zeros_like(self.theta_params[zone_pair])
            value_gradient = np.zeros_like(self.vessel_value_weights[zone_pair])
            
            for _, states, actions, rewards in zp_trajectories:
                # Compute returns
                returns = []
                G = 0
                for reward in reversed(rewards):
                    G = reward + self.gamma * G
                    returns.insert(0, G)
                
                # Compute advantages using vessel-specific value function
                advantages = []
                for i, state in enumerate(states):
                    vessel_value = self.compute_vessel_value(state, zone_pair)
                    advantage = returns[i] - vessel_value
                    advantages.append(advantage)
                
                # Policy gradient for this vessel's trajectory
                for state, action, advantage in zip(states, actions, advantages):
                    # Get current policy
                    current_beta = self.compute_vessel_policy(state, zone_pair)
                    
                    # Gradient of log π_θ(a|s) w.r.t. θ
                    # For sigmoid policy: ∇log π = (a - β) * φ(s)
                    state_features = state[:len(self.theta_params[zone_pair])]
                    policy_grad = (action - current_beta) * state_features * advantage
                    policy_gradient += policy_grad
                
                # Value function gradient
                for state, return_val in zip(states, returns):
                    vessel_value = self.compute_vessel_value(state, zone_pair)
                    value_error = return_val - vessel_value
                    value_grad = value_error * state
                    value_gradient += value_grad
            
            # Update parameters
            if len(zp_trajectories) > 0:
                self.theta_params[zone_pair] += self.learning_rate * policy_gradient / len(zp_trajectories)
                self.vessel_value_weights[zone_pair] += self.learning_rate * value_gradient / len(zp_trajectories)

class MaritimeTrafficVisualizer:
    """Enhanced text-based visualization with metrics"""
    
    def __init__(self, env):
        self.env = env
        self.episode_rewards = []
        self.episode_violations = []
    
    def render(self, reward=None):
        """Render current state with detailed information"""
        print(f"\n{'='*70}")
        print(f"TIMESTEP: {self.env.timestep} | Total Arrivals: {self.env.total_arrivals} | Total Departures: {self.env.total_departures}")
        print(f"{'='*70}")
        
        # Zone status
        print("ZONE STATUS:")
        for zone in self.env.zones:
            occupancy = self.env.zone_occupancy[zone]
            newly_arrived = self.env.newly_arrived_counts[zone]
            capacity = self.env.max_vessels_per_zone
            
            status = ""
            if zone == self.env.source_zone:
                status += " (SOURCE)"
            if zone == self.env.terminal_zone:
                status += " (TERMINAL)"
            if occupancy > capacity:
                status += f" [CONGESTED +{occupancy - capacity}]"
            
            print(f"  z{zone}{status}: {occupancy}/{capacity} total, +{newly_arrived} new")
        
        # In-transit vessels
        print(f"\nIN-TRANSIT VESSELS: {len(self.env.in_transit_vessels)}")
        transit_summary = defaultdict(list)
        for from_z, to_z, arrival_time in self.env.in_transit_vessels:
            eta = arrival_time - self.env.timestep
            transit_summary[(from_z, to_z)].append(eta)
        
        for (from_z, to_z), etas in sorted(transit_summary.items()):
            direction = ""
            if from_z < to_z and to_z != 0:
                direction = " →"
            elif from_z > to_z and from_z != 0:
                direction = " ←"
            elif to_z == 0:
                direction = " ↗ (EXIT)"
            
            avg_eta = np.mean(etas)
            count = len(etas)
            alpha_prob = self.env.alpha.get(from_z, {}).get(to_z, 0.0)
            
            from_name = f"z{from_z}" if from_z != 0 else "zd"
            to_name = f"z{to_z}" if to_z != 0 else "zd"
            
            print(f"  {from_name} → {to_name}{direction}: {count} vessels (ETA: {avg_eta:.1f}) [α={alpha_prob:.2f}]")
        
        # Performance metrics
        total_in_system = sum(self.env.zone_occupancy.values()) + len(self.env.in_transit_vessels)
        throughput_rate = self.env.total_departures / max(1, self.env.timestep)
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Total vessels in system: {total_in_system}")
        print(f"  Congestion violations: {self.env._count_violations()}")
        print(f"  System throughput rate: {throughput_rate:.3f} vessels/timestep")
        
        if reward is not None:
            print(f"  Current reward: {reward:.2f}")
        
        print("-" * 70)
    
    def plot_training_progress(self, episode_rewards, episode_violations):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        ax2.plot(episode_violations)
        ax2.set_title('Congestion Violations')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Violations')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def train_maritime_agent(episodes=500, visualize_every=50):
    """Train the maritime traffic agent with visualization"""
    env = MaritimeTrafficEnv()
    agent = VesselBasedPolicyGradientAgent(env)
    visualizer = MaritimeTrafficVisualizer(env)
    
    episode_rewards = []
    episode_violations = []
    
    print("Training Maritime Traffic Management Agent")
    print("Using Vessel-Based Policy Gradient (No Neural Networks)")
    print("=" * 70)
    
    for episode in range(episodes):
        state = env.reset()
        trajectory = []
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            trajectory.append((state, action, reward))
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy using collective gradient
        agent.update_policy(trajectory)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_violations.append(info['congestion_violations'])
        
        # Visualize progress
        if episode % visualize_every == 0:
            print(f"\nEPISODE {episode}")
            avg_reward = np.mean(episode_rewards[-10:])
            avg_violations = np.mean(episode_violations[-10:])
            print(f"Avg Reward (last 10): {avg_reward:.2f}")
            print(f"Avg Violations (last 10): {avg_violations:.2f}")
            
            if episode % (visualize_every * 2) == 0:
                # Show detailed state occasionally
                visualizer.render(reward=episode_reward)
    
    # Plot final results
    visualizer.plot_training_progress(episode_rewards, episode_violations)
    
    return agent, episode_rewards, episode_violations

if __name__ == "__main__":
    # Test the complete system
    print("Final Maritime Traffic Management Implementation")
    print("=" * 70)
    
    # Quick environment test
    env = MaritimeTrafficEnv()
    visualizer = MaritimeTrafficVisualizer(env)
    
    print("Testing environment...")
    state = env.reset()
    print(f"State dimension: {len(state)}")
    print(f"Action dimension: {len(env.valid_zone_pairs)}")
    print(f"Valid zone pairs: {env.valid_zone_pairs}")
    
    # Run a few steps
    for step in range(150):
        action = np.random.uniform(0.1, 0.9, len(env.valid_zone_pairs))
        state, reward, done, info = env.step(action)
        if step == 0:
            visualizer.render(reward)
    
    print("\nStarting training...")
    trained_agent, rewards, violations = train_maritime_agent(episodes=200, visualize_every=25)
    
    print("Training completed successfully!")
    print(f"Final beta parameters:")
    for zp, beta in trained_agent.beta_params.items():
        print(f"  Zone pair {zp}: β = {beta:.3f}")
