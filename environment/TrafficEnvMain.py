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
        self.n_road_nxt = defaultdict(int)  # road - vehicles choosing road
        self.n_road_tilda = defaultdict(int)  # (road, τ) - committed road usage
        self.n_road_tot = defaultdict(int)
        
        self.t = 0
        
        # Generate initial arrival schedule directly on roads
        self.generate_arrival_schedule()
        
        # Alpha probabilities for road selection
        self.alpha = self.equal_alpha()
        
        # Initial update of totals
        self._update_totals()
        
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

        # 1. Reset n_road_nxt for this timestep
        self.n_road_nxt = defaultdict(int)
        
        # 2. Process vehicles completing transit (from n_txn)
        completed_vehicles = defaultdict(int)
        new_n_road_txn = defaultdict(int)
        
        for (road, tau), count in self.n_road_txn.items():
            if tau == self.t and count > 0:
                # Vehicles complete this road
                dest_junction = self._get_road_destination(road)
                completed_vehicles[dest_junction] += count
            elif tau > self.t:
                # Keep future transits
                new_n_road_txn[(road, tau)] = count
    
        self.n_road_txn = new_n_road_txn

        # 3. Process vehicles completing commitments (from n_road_tilda) 
        new_n_road_tilda = defaultdict(int)
        
        for (road, tau), count in self.n_road_tilda.items():
            if tau == self.t and count > 0:
                # Move committed vehicles to transit
                self.n_road_txn[(road, tau)] = self.n_road_txn.get((road, tau), 0) + count
                # These vehicles also complete transit immediately
                dest_junction = self._get_road_destination(road)
                completed_vehicles[dest_junction] += count
            elif tau > self.t:
                # Keep future commitments
                new_n_road_tilda[(road, tau)] = count
    
        self.n_road_tilda = new_n_road_tilda

        # 4. Route completed vehicles to next roads
        for junction, vehicle_count in completed_vehicles.items():
            if junction in self.terminal_junctions or vehicle_count <= 0:
                continue
                
            available_roads = self._get_roads_from_junction(junction)
            if available_roads:
                # Equal distribution for now
                per_road = vehicle_count // len(available_roads)
                remainder = vehicle_count % len(available_roads)
                
                for i, road in enumerate(available_roads):
                    self.n_road_nxt[road] += per_road
                    if i < remainder:  # Distribute remainder
                        self.n_road_nxt[road] += 1

        # 5. Create new commitments from road choices
        for road, count in self.n_road_nxt.items():
            if count <= 0:
                continue

            t_min = self.t_min[road]
            t_max = self.t_max[road]
            n_trials = t_max - t_min
            p_success = beta.get(road, 0.5)

            if n_trials == 0:
                # Deterministic travel time
                tau = self.t + t_min
                self.n_road_tilda[(road, int(tau))] += count
            else:
                # Stochastic travel time
                deltas = np.arange(n_trials + 1)
                pmf = np.array([
                    comb(n_trials, d) * (p_success**d) * ((1-p_success)**(n_trials-d))
                    for d in deltas
                ], dtype=np.float64)
                pmf /= pmf.sum()

                sampled = self.np_random.multinomial(n=count, pvals=pmf)
                for delta, vehicles in zip(deltas, sampled):
                    if vehicles > 0:
                        tau = self.t + t_min + delta
                        self.n_road_tilda[(road, int(tau))] += vehicles

        # 6. Update totals and compute reward
        self._update_totals()
        reward = self._compute_reward()
        
        # 7. Check termination
        done = self.t >= self.H
        
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
        """Update total vehicle counts for roads only"""
        self.n_road_tot = defaultdict(int)
        
        # Count vehicles in transit on roads (future arrivals)
        for (road, tau), count in self.n_road_txn.items():
            if tau > self.t:
                self.n_road_tot[road] += count
    
        # Count vehicles committed to roads (not yet started transit)
        for (road, tau), count in self.n_road_tilda.items():
            if tau > self.t:
                self.n_road_tot[road] += count

    def _compute_reward(self):
        """Compute reward based on road capacity violations only"""
        total_penalty = 0.0
        
        # Road capacity penalties only
        for road in self.roads:
            road_load = self.n_road_tot.get(road, 0)
            road_capacity = self.road_capacities[road]
            excess = max(road_load - road_capacity, 0)
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
        """Generate arrival schedule directly on roads from source"""
        if not hasattr(self, 'np_random') or self.np_random is None:
            return
            
        elements = list(ARRIVAL_DIST.keys())
        probs = np.array(list(ARRIVAL_DIST.values()))

        for _ in range(M):
            idx = self.np_random.choice(len(elements), p=probs)
            dest, tau = elements[idx]
            road = f"road_source_to_{dest}"
            if road in self.roads:  # Ensure road exists
                self.n_road_txn[(road, tau)] += 1

    def _get_state_vector(self):
        """Convert road counts to state vector (simplified)"""
        # Simplified state: only road transit and commitments
        self.state_size = (self.R * self.H) + (self.R * self.H)
        
        vec = np.zeros(self.state_size, dtype=np.float32)
        idx = 0
        
        # n_road_txn: vehicles in transit on roads
        for road in self.roads:
            for tau in range(1, self.H + 1):
                vec[idx] = self.n_road_txn.get((road, tau), 0)
                idx += 1
    
        # n_road_tilda: committed road usage
        for road in self.roads:
            for tau in range(1, self.H + 1):
                vec[idx] = self.n_road_tilda.get((road, tau), 0)
                idx += 1
            
        return vec
