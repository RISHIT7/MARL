import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import numpy as np
from environment.parameters import *
from math import comb

# Temporary
SOURCE_ROAD_CAPACITY = CAPACITIES
ROAD_CAPACITY = CAPACITIES
JUNCTION_CAPACITY = CAPACITIES

class RoadTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print(f"Number of Vehicles {M}")
        self.np_random = None
        
        # Environment parameters - Junctions instead of zones
        self.junctions = ['j_source']  # Source junction for vehicle arrivals
        for i in range(NODES):
            self.junctions.append(f"j_{i+1}")
        self.J = len(self.junctions)
        self.H = HORIZON
        
        # Create directed roads from edges
        self.roads = []
        self.road_to_index = {}
        
        # Roads from source to arrival junctions
        for i in ARRIVAL_NODES:
            road_id = f"road_source_to_{i}"
            self.roads.append(road_id)
            self.road_to_index[road_id] = len(self.roads) - 1
        
        # Create bidirectional roads as two separate unidirectional roads
        for (i, j) in EDGES:
            # Road from junction i to junction j
            road_ij = f"road_{i}_to_{j}"
            self.roads.append(road_ij)
            self.road_to_index[road_ij] = len(self.roads) - 1
            
            # Road from junction j to junction i (reverse direction)
            road_ji = f"road_{j}_to_{i}"
            self.roads.append(road_ji)
            self.road_to_index[road_ji] = len(self.roads) - 1
        
        self.R = len(self.roads)
        
        # Valid transitions now use road IDs
        self.valid_transitions = []
        for road in self.roads:
            self.valid_transitions.append(road)
        
        # Junction classifications
        self.arrival_junctions = [f"j_{i}" for i in ARRIVAL_NODES]
        self.terminal_junctions = [f"j_{i}" for i in TERMINAL_NODES]
        
        # Road capacity mapping - each road has individual capacity
        self.road_capacities = {}
        for road in self.roads:
            if road.startswith("road_source"):
                self.road_capacities[road] = SOURCE_ROAD_CAPACITY  # Different capacity for source roads
            else:
                self.road_capacities[road] = ROAD_CAPACITY  # Standard road capacity
        
        # Junction capacity (for vehicles waiting at junctions)
        self.junction_capacities = {j: JUNCTION_CAPACITY for j in self.junctions}
        
        print("Junctions:", self.junctions)
        print("Roads:", self.roads)
        print("Road Capacities:", self.road_capacities)
        
        # State space dimensions - now tracking vehicles on roads and at junctions
        # n_road_txn: vehicles in transit on roads (road, τ)
        # n_junction_arr: vehicles arrived at junctions
        # n_road_nxt: vehicles choosing next road
        # n_road_tilda: vehicles committed to future road usage
        self.state_size = (self.R * self.H) + self.J + (self.R) + (self.R * self.H)
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        # Action space: beta values for each road
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.roads),), dtype=np.float32
        )
        
        # Transit time bounds for roads
        self.t_min = {road: T_MIN for road in self.roads}
        self.t_max = {road: T_MAX for road in self.roads}
        
        self.w_r = W_R
        self.w_d = W_D
        self._last_obs = None
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize count vectors for road-based system
        self.n_road_txn = defaultdict(int)  # (road, τ) - vehicles in transit on road
        self.n_junction_arr = defaultdict(int)  # junction - vehicles at junction
        self.n_road_nxt = defaultdict(int)  # road - vehicles choosing road
        self.n_road_tilda = defaultdict(int)  # (road, τ) - committed road usage
        self.n_road_tot = defaultdict(int)  # road - total vehicles on road
        self.n_junction_tot = defaultdict(int)  # junction - total vehicles at junction
        
        # Start all vehicles at source junction
        self.n_junction_tot['j_source'] = M
        
        self.t = 0
        
        # Generate arrival schedule for source to arrival junctions
        self.arrival_schedule = self.generate_arrival_schedule()
        
        # Alpha probabilities for road selection from each junction
        self.alpha = self.equal_alpha()
        
        self._last_obs = self._get_state_vector()
        return self._last_obs, {}

    def render(self):
        pass
		# if self.render_mode != "human":
		# 	return

		# if self.renderer is not None:
		# 	self.renderer.update(node_values=self.n_tot, edge_weights=self.edge_weight)

		
    def close(self):
        pass
		# if self.renderer is not None:
		# 	self.renderer.stop_render_loop()
		# 	pygame.quit()
		# 	self.renderer = None

    def step(self, action):
        self.t += 1
        
        # Transform action to β values for roads
        beta_vals = 1.0 / (1.0 + np.exp(-action))
        beta = {road: beta_vals[i] for i, road in enumerate(self.roads)}

        # Reset junction arrivals (except terminals)
        self.n_junction_arr = {
            j: self.n_junction_arr[j] if (j in self.terminal_junctions and j in self.n_junction_arr.keys()) else 0
            for j in self.junctions
        }

        # 1. Update n_junction_arr - vehicles completing road transit
        for (road, tau), count in list(self.n_road_tilda.items()):
            if tau == self.t and count > 0:
                dest_junction = self._get_road_destination(road)
                self.n_junction_arr[dest_junction] += count

        for (road, tau), count in list(self.n_road_txn.items()):
            if tau == self.t:
                dest_junction = self._get_road_destination(road)
                self.n_junction_arr[dest_junction] += count

        # Clean up zero entries
        self.n_junction_arr = {j: count for j, count in self.n_junction_arr.items() if count > 0}

        # 2. Update n_road_nxt - vehicles at junctions choosing roads
        self.n_road_nxt = defaultdict(int)
        for j in self.junctions:
            if j in self.terminal_junctions:
                continue

            n_arr_j = self.n_junction_arr.get(j, 0)
            if n_arr_j > 0:
                available_roads = self._get_roads_from_junction(j)
                if available_roads and j in self.alpha:
                    probs = np.array([self.alpha[j].get(road, 0) for road in available_roads])
                    if probs.sum() > 0:
                        probs = probs / probs.sum()  # Normalize
                        sampled_counts = self.np_random.multinomial(n=n_arr_j, pvals=probs)
                        for i, road in enumerate(available_roads):
                            self.n_road_nxt[road] += sampled_counts[i]

        # 3. Update n_road_txn - vehicles committed to roads
        new_n_road_txn = defaultdict(int)
        for road in self.roads:
            for tau in range(self.t + 1, self.H + 1):
                prev_txn = self.n_road_txn.get((road, tau), 0)
                new_commit = self.n_road_tilda.get((road, tau), 0)
                new_n_road_txn[(road, tau)] = prev_txn + new_commit
        self.n_road_txn = new_n_road_txn

        # 4. Update n_road_tilda - future road commitments
        self.n_road_tilda = defaultdict(int)
        for road in self.roads:
            count = self.n_road_nxt.get(road, 0)
            if count <= 0:
                continue

            t_min = self.t_min[road]
            t_max = self.t_max[road]
            n_trials = t_max - t_min
            p_success = beta.get(road, 0.5)

            deltas = np.arange(n_trials + 1)
            pmf = np.array([
                comb(n_trials, d) * (p_success**d) * ((1-p_success)**(n_trials-d))
                for d in deltas
            ], dtype=np.float64)
            pmf /= pmf.sum()

            sampled = self.np_random.multinomial(n=count, pvals=pmf)
            for delta, vehicles in zip(deltas, sampled):
                if vehicles != 0:
                    tau = self.t + t_min + delta
                    self.n_road_tilda[(road, int(tau))] += vehicles

        # 5. Compute reward based on road and junction capacities
        reward = self._compute_reward()
        
        # 6. Check termination
        done = self.t >= self.H

        # Update total counts
        self._update_totals()
        
        self._last_obs = self._get_state_vector()
        return self._last_obs, reward, done, False, {}

    def _get_road_destination(self, road):
        """Extract destination junction from road name"""
        if road.startswith("road_source_to_"):
            return f"j_{road.split('_')[-1]}"
        elif "road_" in road and "_to_" in road:
            parts = road.split("_to_")
            return f"j_{parts[1]}"
        return None

    def _get_roads_from_junction(self, junction):
        """Get all roads that start from given junction"""
        available_roads = []
        j_id = junction.replace('j_', '')
        
        if junction == 'j_source':
            available_roads = [road for road in self.roads if road.startswith("road_source_to_")]
        else:
            available_roads = [road for road in self.roads if road.startswith(f"road_{j_id}_to_")]
        
        return available_roads

    def _update_totals(self):
        """Update total vehicle counts for roads and junctions"""
        # Reset totals
        self.n_road_tot = defaultdict(int)
        self.n_junction_tot = defaultdict(int)
        
        # Count vehicles at junctions
        for j, count in self.n_junction_arr.items():
            self.n_junction_tot[j] += count
        
        # Count vehicles on roads (in transit)
        for (road, tau), count in self.n_road_txn.items():
            if tau > self.t:
                self.n_road_tot[road] += count

    def _compute_reward(self):
        """Compute reward based on road and junction capacity violations"""
        total_penalty = 0.0
        
        # Road capacity penalties
        for road in self.roads:
            road_load = self.n_road_tot.get(road, 0)
            road_capacity = self.road_capacities[road]
            excess = max(road_load - road_capacity, 0)
            if excess > 0:
                total_penalty += self.w_r * excess + self.w_d
        
        # Junction capacity penalties (excluding terminals)
        for j in self.junctions:
            if j not in self.terminal_junctions:
                junction_load = self.n_junction_tot.get(j, 0)
                junction_capacity = self.junction_capacities[j]
                excess = max(junction_load - junction_capacity, 0)
                if excess > 0:
                    total_penalty += self.w_r * excess + self.w_d
        
        return -total_penalty

    def equal_alpha(self):
        """Compute α(road|junction) transition probabilities"""
        alpha = {}
        for j in self.junctions:
            available_roads = self._get_roads_from_junction(j)
            if available_roads:
                prob = 1.0 / len(available_roads)
                alpha[j] = {road: prob for road in available_roads}
            else:
                alpha[j] = {}
        return alpha

    def generate_arrival_schedule(self):
        """Generate arrival schedule from source to arrival junctions"""
        elements = list(ARRIVAL_DIST.keys())
        probs = np.array(list(ARRIVAL_DIST.values()))

        for _ in range(M):
            idx = self.np_random.choice(len(elements), p=probs)
            dest, tau = elements[idx]
            road = f"road_source_to_{dest}"
            self.n_road_txn[(road, tau)] += 1

    def _get_state_vector(self):
        """Convert road and junction counts to state vector"""
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0
        
        # n_road_txn: vehicles in transit on roads
        for road in self.roads:
            for tau in range(1, self.H + 1):
                vec[idx] = self.n_road_txn.get((road, tau), 0)
                idx += 1
        
        # n_junction_arr: vehicles at junctions
        for j in self.junctions:
            vec[idx] = self.n_junction_arr.get(j, 0)
            idx += 1
            
        # n_road_nxt: vehicles choosing roads
        for road in self.roads:
            vec[idx] = self.n_road_nxt.get(road, 0)
            idx += 1
            
        # n_road_tilda: committed road usage
        for road in self.roads:
            for tau in range(1, self.H + 1):
                vec[idx] = self.n_road_tilda.get((road, tau), 0)
                idx += 1
                
        return vec
