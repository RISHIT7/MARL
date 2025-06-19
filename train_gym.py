import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import matplotlib.pyplot as plt
import time

class MaritimeTrafficEnvGym(gym.Env):
    """
    Gymnasium-compatible version of the Maritime Traffic Environment.
    
    This environment simulates maritime traffic management across several zones.
    The agent's goal is to control vessel speed (via a 'beta' parameter)
    to maximize throughput while minimizing congestion.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, n_zones=4, max_vessels_per_zone=10, arrival_rate=0.3, 
                 time_horizon=100, congestion_penalty=-2.0, throughput_reward=5.0):
        super().__init__()
        
        # Environment parameters
        self.n_zones = n_zones
        self.dummy_zone = 0  # Represents "outside system"
        self.zones = list(range(1, n_zones + 1))
        self.source_zone = 1
        self.terminal_zone = n_zones
        self.max_vessels_per_zone = max_vessels_per_zone
        self.arrival_rate = arrival_rate
        self.time_horizon = time_horizon
        self.congestion_penalty = congestion_penalty
        self.throughput_reward = throughput_reward
        
        # Navigation time parameters
        self.t_min = 1
        self.t_max = 5
        
        # Random input probabilities from zd to z1
        self.zd_to_z1_probabilities = np.random.uniform(0.1, 0.9, size=time_horizon)
        
        # α(z'|z) probability distributions for vessel movement
        self.alpha = {
            1: {0: 0, 2: 1},
            2: {1: 0.3, 3: 0.7},
            3: {2: 0.3, 4: 0.7},
            4: {3: 0.2, 0: 0.8}
        }
        
        # Define valid zone pairs for actions
        self.valid_zone_pairs = []
        for from_z in self.zones:
            for to_z in self.alpha[from_z].keys():
                self.valid_zone_pairs.append((from_z, to_z))
        
        # === Gymnasium Spaces ===
        # Action space: Beta parameter for each zone pair, clipped between 0.1 and 0.9
        self.action_space = spaces.Box(
            low=0.1, high=0.9, shape=(len(self.valid_zone_pairs),), dtype=np.float32
        )
        
        # Observation space: Count-based state vector
        obs_dim = (
            len(self.zones)              # newly_arrived_counts
            + len(self.valid_zone_pairs) # in_transit_counts
            + len(self.zones)              # zone_occupancy
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
    def reset(self, *, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        self.timestep = 0
        self.newly_arrived_counts = defaultdict(int)
        self.in_transit_counts = defaultdict(int)
        self.zone_occupancy = defaultdict(int)
        self.in_transit_vessels = []
        self.total_arrivals = 0
        self.total_departures = 0
        
        # The info dict is returned by reset in Gymnasium
        info = {}
        
        return self._get_state(), info

    def step(self, action):
        """Execute one time step within the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        zone_pair_actions = {
            pair: action[i] for i, pair in enumerate(self.valid_zone_pairs)
        }
        
        reward = 0.0
        
        # 1. New arrivals at source zone
        arrival_prob = self.zd_to_z1_probabilities[self.timestep % len(self.zd_to_z1_probabilities)]
        expected_arrivals = self.arrival_rate * arrival_prob
        new_arrivals = self.np_random.poisson(expected_arrivals)
        
        if new_arrivals > 0:
            self.newly_arrived_counts[self.source_zone] += new_arrivals
            self.zone_occupancy[self.source_zone] += new_arrivals
            self.total_arrivals += new_arrivals

        # 2. Process in-transit completions
        reward += self._process_in_transit_completions()
        
        # 3. Process vessel decisions
        self._process_vessel_decisions(zone_pair_actions)
        
        # 4. Calculate penalties and step cost
        reward += self._calculate_congestion_penalty()
        reward -= 0.1  # Step penalty
        
        self.timestep += 1
        
        # Gymnasium termination conditions
        terminated = self.timestep >= self.time_horizon
        truncated = False # No other truncation condition
        
        info = {
            'timestep': self.timestep,
            'total_vessels_in_system': sum(self.zone_occupancy.values()),
            'congestion_violations': self._count_violations(),
            'total_arrivals': self.total_arrivals,
            'total_departures': self.total_departures
        }
        
        return self._get_state(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment's state."""
        if mode == "human":
            print(f"\n{'='*70}")
            print(f"TIMESTEP: {self.timestep} | Arrivals: {self.total_arrivals} | Departures: {self.total_departures}")
            print(f"{'-'*70}")
            print("ZONE STATUS:")
            for zone in self.zones:
                occ = self.zone_occupancy[zone]
                new = self.newly_arrived_counts[zone]
                cap = self.max_vessels_per_zone
                status = " (SOURCE)" if zone == self.source_zone else " (TERMINAL)" if zone == self.terminal_zone else ""
                if occ > cap:
                    status += f" [CONGESTED +{occ - cap}]"
                print(f"  z{zone}{status}: {occ}/{cap} total, +{new} new")
            
            print(f"\nIN-TRANSIT VESSELS: {len(self.in_transit_vessels)}")
            print(f"{'='*70}")

    def _get_state(self):
        """Constructs the state vector from environment counts."""
        state = []
        state.extend(self.newly_arrived_counts[z] for z in self.zones)
        state.extend(self.in_transit_counts[pair] for pair in self.valid_zone_pairs)
        state.extend(self.zone_occupancy[z] for z in self.zones)
        return np.array(state, dtype=np.float32)

    def _sample_navigation_time(self, beta):
        """Sample navigation time using binomial distribution."""
        trials = self.t_max - self.t_min
        successes = self.np_random.binomial(trials, beta)
        return self.t_min + successes

    def _process_in_transit_completions(self):
        """Process vessels completing their navigation."""
        reward = 0.0
        completed_indices = [i for i, (_, _, arrival_time) in enumerate(self.in_transit_vessels) if arrival_time == self.timestep]
        
        for i in reversed(completed_indices):
            from_z, to_z, _ = self.in_transit_vessels.pop(i)
            self.in_transit_counts[(from_z, to_z)] -= 1
            if to_z == self.dummy_zone:
                reward += self.throughput_reward
                self.total_departures += 1
            else:
                self.newly_arrived_counts[to_z] += 1
                self.zone_occupancy[to_z] += 1
        return reward

    def _process_vessel_decisions(self, zone_pair_actions):
        """Process decisions for newly arrived vessels."""
        for zone in list(self.newly_arrived_counts.keys()):
            if self.newly_arrived_counts[zone] > 0 and zone in self.alpha:
                for _ in range(self.newly_arrived_counts[zone]):
                    # Sample destination using α(z'|z)
                    dest_zones, dest_probs = zip(*self.alpha[zone].items())
                    chosen_dest = self.np_random.choice(dest_zones, p=dest_probs)
                    
                    zone_pair = (zone, chosen_dest)
                    beta = zone_pair_actions.get(zone_pair, 0.5) # Default beta
                    
                    arrival_time = self.timestep + self._sample_navigation_time(beta)
                    
                    # Update counts
                    self.zone_occupancy[zone] -= 1
                    self.in_transit_counts[zone_pair] += 1
                    self.in_transit_vessels.append((zone, chosen_dest, arrival_time))
                
                self.newly_arrived_counts[zone] = 0

    def _calculate_congestion_penalty(self):
        """Calculate penalties for exceeding zone capacity."""
        penalty = 0.0
        for zone in self.zones:
            if self.zone_occupancy[zone] > self.max_vessels_per_zone:
                excess = self.zone_occupancy[zone] - self.max_vessels_per_zone
                penalty += self.congestion_penalty * excess
        return penalty

    def _count_violations(self):
        """Count number of zones currently under congestion."""
        return sum(1 for z in self.zones if self.zone_occupancy[z] > self.max_vessels_per_zone)


# Note: The agent and training logic below are kept from the original file
# to maintain similar functionality. The training loop is adapted for the Gym API.

class VesselBasedPolicyGradientAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.zone_pairs = env.valid_zone_pairs
        self.theta_params = {zp: np.random.normal(0, 0.1, size=3) for zp in self.zone_pairs}

    def compute_vessel_policy(self, state, zone_pair):
        theta = self.theta_params[zone_pair]
        state_features = state[:len(theta)]
        logit = np.dot(theta, state_features)
        beta = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(beta, 0.1, 0.9)

    def select_action(self, state):
        actions = [self.compute_vessel_policy(state, zp) for zp in self.zone_pairs]
        return np.array(actions)

    def update_policy(self, trajectory):
        # This update logic is based on the original file's implementation
        policy_gradient = {zp: np.zeros_like(self.theta_params[zp]) for zp in self.zone_pairs}
        # Simplified update: This part may need to be adapted based on the paper's specific algorithm
        for state, action, reward in trajectory:
            for i, zone_pair in enumerate(self.zone_pairs):
                current_beta = self.compute_vessel_policy(state, zone_pair)
                state_features = state[:len(self.theta_params[zone_pair])]
                grad = (action[i] - current_beta) * state_features * reward
                policy_gradient[zone_pair] += grad
        
        for zp in self.zone_pairs:
            self.theta_params[zp] += self.learning_rate * policy_gradient[zp] / len(trajectory)


def train_maritime_agent(episodes=500, visualize_every=50):
    env = MaritimeTrafficEnvGym()
    agent = VesselBasedPolicyGradientAgent(env)
    
    episode_rewards = []
    episode_violations = []

    for episode in range(episodes):
        state, _ = env.reset()
        trajectory = []
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory.append((state, action, reward))
            episode_reward += reward
            state = next_state
        
        agent.update_policy(trajectory)
        episode_rewards.append(episode_reward)
        episode_violations.append(info.get('congestion_violations', 0))
        
        if episode % visualize_every == 0:
            print(f"\nEPISODE {episode}")
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Avg Reward (last 10): {avg_reward:.2f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax2.plot(episode_violations)
    ax2.set_title('Congestion Violations')
    plt.show()


if __name__ == "__main__":
    print("Testing Gymnasium Environment with a Random Agent...")
    env = MaritimeTrafficEnvGym()
    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample()  # Use a random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    print("\nStarting Training with Policy Gradient Agent...")
    train_maritime_agent(episodes=200, visualize_every=25)
    print("\nTraining complete.")