import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import numpy as np
from environment.parameters import *
from math import comb
from Visualization.graph_renderer import MaritimeTrafficGraph
import dash
from dash import dcc, html, Input, Output, State
from dash import ctx

class MaritimeTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print(f"Number of Vessels {M}")
        self.np_random = None
        # Environment parameters
        self.zones= ['z_dummy']
        for i in range(NODES):
            self.zones.append(f"z_{i+1}")
        self.Z = len(self.zones)
        self.H = HORIZON
        
        self.valid_transitions = []
        # Valid transitions 
        for i in ARRIVAL_NODES:
            self.valid_transitions.append(("z_dummy",f"z_{i}"))
        for (i,j) in EDGES:
            self.valid_transitions.append((f"z_{i}",f"z_{j}"))

        self.arrival_nodes = []
        for i in ARRIVAL_NODES:
            self.arrival_nodes.append(f"z_{i}")
        self.terminal_nodes = []
        for i in TERMINAL_NODES:
            self.terminal_nodes.append(f"z_{i}")
        print(self.zones)
        print(self.valid_transitions)
        print(self.arrival_nodes)
        print(self.terminal_nodes)
        
        # State space dimensions
        self.state_size = 2 * (self.Z**2 * self.H) + 2 * (self.Z**2) + self.Z
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        # Values of beta 
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.valid_transitions),), dtype=np.float32
        )
        
        # Transit time bounds
        self.t_min = {trans: T_MIN for trans in self.valid_transitions}
        self.t_max = {trans: T_MAX for trans in self.valid_transitions}
        
        # Zone capacities for reward computation
        self.capacities = {z: CAPACITIES for z in self.zones}
        self.w_r = W_R
        self.w_d = W_D
        # self.graph = MaritimeTrafficGraph(
        #     zones=self.zones,
        #     valid_transitions=self.valid_transitions,
        #     node_vmax=50,  # Tune based on expected traffic
        #     edge_vmax=50
        # )
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize count vectors
        self.n_txn = defaultdict(int)  # (z,z',τ)
        self.n_arr = defaultdict(int)  # z
        self.n_nxt = defaultdict(int)  # (z,z')
        self.n_tilda = defaultdict(int)  # (z,z',τ) 
        self.n_tot = defaultdict(int)
        self.edge_weight = defaultdict(int)
        self.n_tot['z_dummy'] = M
        
        self.t = 0
        
        # Variable Arrival Schedule
        self.arrival_schedule = self.generate_arrival_schedule() 
        
        # Alpha probabilities (equal for simplicity)
        self.alpha = self.equal_alpha()
        # print(self.alpha)
        
        
        return self._get_state_vector(), {}

    def render(self):
        pass
        # Compute current node values (arrivals + in-transit)
        # n_total = defaultdict(int)
        # for z, count in self.n_arr.items():
        #     n_total[z] += count
        # for (z_src, z_dest, tau), count in self.n_txn.items():
        #     if tau > self.t:
        #         n_total[z_src] += count

        # # Compute current edge values (future commitments this step)
        # edge_weights = defaultdict(int)
        # for (z, z_p), count in self.n_nxt.items():
        #     edge_weights[(z, z_p)] = count

        # self.graph.update(n_total, edge_weights)
        # self.graph.show()
    
    def close(self):
        pass

    def step(self, action):
        self.t += 1
        
        # Transform action to β values using sigmoid: β ∈ [0,1]
        beta_vals = 1.0 / (1.0 + np.exp(-action))
        beta = {trans: beta_vals[i] for i, trans in enumerate(self.valid_transitions)}

        # reset value of n_arr to zero
        # retain the value at terminal nodes
        self.n_arr = {
            z: self.n_arr[z] if ((z in self.terminal_nodes) and (z in self.n_arr.keys())) else 0
            for z in self.zones
        }

        # 1. Update n_arr
        for (z_src, z_dest, tau), count in list(self.n_tilda.items()):
            if tau == self.t and count > 0:
                self.n_arr[z_dest] += count
                # completed_transits.append((z_src, z_dest, tau, count))

        for (z_src,z_dest,tau),count in list(self.n_txn.items()):
            if tau == self.t:
                self.n_arr[z_dest] += count
 
        self.n_arr = {
        z: count for z, count in self.n_arr.items() if count > 0
    }

        self.n_nxt = defaultdict(int)
        # 2. Update n_nxt 
        for z in self.zones:
            if z in self.terminal_nodes:
                continue

            n_arr_z = self.n_arr.get(z, 0)
            if n_arr_z > 0 and self.alpha[z]: 
                dests = list(self.alpha[z].keys())
                probs = np.array([self.alpha[z][z_p] for z_p in dests])
                
                # Multinomial draw: how many of the n_arr_z vessels go to each z'
                sampled_counts = self.np_random.multinomial(n=n_arr_z, pvals=probs)
                
                for i, z_p in enumerate(dests):
                    self.n_nxt[(z, z_p)] += sampled_counts[i]

        # 3.  Update n_txn
        new_n_txn = defaultdict(int)

        for z in self.zones:
            if z in self.terminal_nodes:
                continue
            for z_p in self.zones:
                for tau in range(self.t + 1, self.H + 1):
                    prev_txn = self.n_txn.get((z, z_p, tau), 0)
                    new_commit = self.n_tilda.get((z, z_p, tau), 0)
                    new_n_txn[(z, z_p, tau)] = prev_txn + new_commit

        self.n_txn = new_n_txn
        
        # 4. Update n_tilde
        # Previously it was taking O(Z*Z*M) time which doesnt scalem, this takes O(Z*Z*H).
        # Sample future arrival times Δ ~ Binomial(n_trials, p=β)
        # clear n_tilda

        self.n_tilda = defaultdict(int)
        for z in self.zones:
            if z in self.terminal_nodes:
                continue
            for z_p in self.zones:
                count = self.n_nxt.get((z, z_p), 0)
                if count <= 0 or (z, z_p) not in self.t_min:
                    continue

                # hard bounds
                t_min = self.t_min[(z, z_p)]
                t_max = self.t_max[(z, z_p)]
                n_trials = t_max - t_min   # number of Bernoulli trials
                
                # fetch β for this leg
                p_success = beta.get((z, z_p), 0.5)

                # support for Δ = 0 .. n_trials
                deltas = np.arange(n_trials + 1)

                # compute binomial PMF:  C(n_trials,Δ) · β^Δ · (1−β)^(n_trials−Δ)
                pmf = np.array([
                    comb(n_trials, d) * (p_success**d) * ((1-p_success)**(n_trials-d))
                    for d in deltas
                ], dtype=np.float64)

                # sanity: should sum to 1
                pmf /= pmf.sum()

                # multinomial: how many of the `count` ships pick each Δ
                sampled = self.np_random.multinomial(n=count, pvals=pmf)

                # commit them in n_tilda at the proper τ = t + t_min + Δ
                for delta, ships in zip(deltas, sampled):
                    if ships != 0:
                        tau = self.t + t_min + delta
                        self.n_tilda[(z, z_p, tau)] += ships
        

        # 6. Compute reward using equation 3
        reward = self._compute_reward()
        
        # 7. Check termination
        done = self.t >= self.H

        # Compute total vessels per zone
        self.n_tot = defaultdict(int)
        
        # Count arrivals
        for z, count in self.n_arr.items():
            self.n_tot[z] += count
            
        # Count vessels in transit (going to another port but not reached yet)
        for (z_src, z_dest, tau), count in self.n_txn.items():
            if tau > self.t:
                self.n_tot[z_src] += count
        # Compute edge_weight = n_nxt + sum(n_txn over all tau)
        self.edge_weight = defaultdict(int)

        for (z, z_p), val in self.n_nxt.items():
            self.edge_weight[(z, z_p)] += val

        for (z, z_p, tau), val in self.n_txn.items():
            self.edge_weight[(z, z_p)] += val

            
        # Ensure all zones are represented (even if 0)
        for z in self.zones:
            _ = self.n_tot[z]  # forces default 0 if missing
        

        info = {}
        # Build stats text
        # print(f"Timestep: {self.t}")
        # print("Zone-wise Ship Count:")
        # for z in self.zones:
        #     print(f"  {z}: {self.n_tot.get(z, 0)}")
        # # print(f"Time step: {env.t}")
        # print(f"Action: {action}")
        # print("Arrived ships per zone (n_arr):")
        # for zone, total in self.n_arr.items():
        #     print(f"  {zone}: {total}")

        # print("\nn_tilda (future transit commitments):")
        # for key, val in sorted(self.n_tilda.items()):
        #     print(f"  {key}: {val}")
        
        # print("\nn_nxt (routing decisions this step):")
        # for key, val in sorted(self.n_nxt.items()):
        #     print(f"  {key}: {val}")
        # print("\nn_txn (committed future transits):")
        # for key, val in sorted(self.n_txn.items()):
        #     if val != 0:
        #         print(f"  {key}: {val}")
        return self._get_state_vector(), reward, done, False, info

    def generate_arrival_schedule(self):    
        elements = list(ARRIVAL_DIST.keys())
        # print(elements)
        probs = np.array(list(ARRIVAL_DIST.values()))
        # print(probs)

        for _ in range(M):
            idx = self.np_random.choice(len(elements), p=probs)
            src,tau = elements[idx]
            self.n_txn[('z_dummy',f'z_{src}',tau)] += 1
            self.edge_weight[('z_dummy',f'z_{src}')] += 1

        print(self.n_txn)

    def equal_alpha(self):
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

    def _compute_reward(self):
        """Compute reward using equation 3: C(z,n) = w_r * max(n-cap, 0) + w_d"""
        total_penalty = 0.0
        
        # Compute total vessels per zone
        # n_tot = defaultdict(int)
        
        # # Count arrivals
        # for z, count in self.n_arr.items():
        #     n_tot[z] += count
            
        # # Count vessels in transit (going to another port but not reached yet)
        # for (z_src, z_dest, tau), count in self.n_txn.items():
        #     if tau > self.t:
        #         n_tot[z_src] += count
                
        # Apply congestion penalty
        for z in self.zones:
            if z not in self.terminal_nodes:  # Terminal zone doesn't have congestion
                excess = max(self.n_tot[z] - self.capacities[z], 0)
                total_penalty += self.w_r * excess + self.w_d
                
        return -total_penalty  

    def initialize_renderer(self):
        self.graph = MaritimeTrafficGraph(
            zones=self.zones,
            valid_transitions=self.valid_transitions,
            node_vmax=M,
            edge_vmax=M
        )

    def get_graph_figure(self):
        # state_info = reconstruct(self._get_state_vector(), self.zones, self.H, self.t)
        # n_total = self.n_tot
        # edge_weights = self.edge_weight
        self.graph.update(self.n_tot, self.edge_weight)
        return self.graph.create_figure(self.t)

    def get_stats_text(self):
        # state_info = reconstruct(self._get_state_vector(), self.zones, self.H, self.t)
        lines = [f"Time step: {self.t}"]
        lines.append("Zone-wise Ship Count:")
        for z in self.zones:
            lines.append(f"  {z}: {self.n_tot.get(z, 0)}")

        lines.append(f"Action: {np.zeros(self.action_space.shape)}")
        lines.append("Total ships per zone (n_total):")
        for zone, total in self.n_tot.items():
            lines.append(f"  {zone}: {total}")

        lines.append("Arrived ships per zone (n_arr):")
        for zone, total in self.n_arr.items():
            lines.append(f"  {zone}: {total}")

        lines.append("\nn_tilda (future transit commitments):")
        for key, val in sorted(self.n_tilda.items()):
            lines.append(f"  {key}: {val}")
        
        lines.append("\nn_nxt (routing decisions this step):")
        for key, val in sorted(self.n_nxt.items()):
            lines.append(f"  {key}: {val}")

        lines.append("\nn_txn (committed future transits):")
        for key, val in sorted(self.n_txn.items()):
            if(val!=0):
                lines.append(f"  {key}: {val}")
        
        return "\n".join(lines)

    def step_forward(self):
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return self.get_graph_figure(), self.get_stats_text()

    # def render_dash(self, port=8050):
    #     self.initialize_renderer()

    #     app = dash.Dash(__name__)
    #     app.title = "Maritime Traffic Simulation"

    #     app.layout = html.Div([
    #         html.H1("Maritime Traffic Simulation", style={"textAlign": "center"}),

    #         html.Div([
    #             dcc.Graph(
    #                 id="traffic-graph",
    #                 figure=self.get_graph_figure(),
    #                 style={"height": "600px"}
    #             ),
    #             html.Div(id="stats", style={"whiteSpace": "pre-line", "padding": "10px"})
    #         ], style={"display": "flex", "justifyContent": "space-between"}),

    #         html.Div([
    #             html.Button("Next Timestep", id="step-button", n_clicks=0)
    #         ], style={"textAlign": "center", "margin": "20px"})
    #     ])

    #     @app.callback(
    #         [Output("traffic-graph", "figure"),
    #          Output("stats", "children")],
    #         Input("step-button", "n_clicks"),
    #         prevent_initial_call=True
    #     )
    #     def update(n_clicks):
    #         return self.step_forward()

    #     app.run(debug=True, port=port)
    def render_dash(self, port=8050):
        self.initialize_renderer()

        app = dash.Dash(__name__)
        app.title = "Maritime Traffic Simulation"

        app.layout = html.Div([
            html.H1("Maritime Traffic Simulation", style={"textAlign": "center"}),

            html.Div([
                dcc.Graph(
                    id="traffic-graph",
                    figure=self.get_graph_figure(),
                    style={"height": "600px"}
                ),
                html.Div(id="stats", style={"whiteSpace": "pre-line", "padding": "10px"})
            ], style={"display": "flex", "justifyContent": "space-between"}),

            html.Div([
                html.Button("Next Timestep", id="step-button", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("Auto Run", id="auto-button", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("Reset", id="reset-button", n_clicks=0)
            ], style={"textAlign": "center", "margin": "20px"}),

            dcc.Interval(
                id="interval-timer",
                interval=2000,  # 2 seconds between timesteps
                n_intervals=0,
                disabled=True
            )
        ])

        # Step forward when Next Timestep or Interval triggers
        @app.callback(
            [Output("traffic-graph", "figure"),
            Output("stats", "children"),
            Output("interval-timer", "disabled")],
            [Input("step-button", "n_clicks"),
            Input("interval-timer", "n_intervals"),
            Input("auto-button", "n_clicks"),
            Input("reset-button", "n_clicks")],
            prevent_initial_call=True
        )
        def unified_callback(step_clicks, auto_intervals, auto_clicks, reset_clicks):
            trigger_id = ctx.triggered_id

            if trigger_id == "reset-button":
                self.reset(seed=1)
                fig = self.get_graph_figure()
                stats = self.get_stats_text()
                self.graph.reset() 
                return fig, stats, True  # Disable interval

            elif trigger_id == "auto-button":
                # Toggle: disable becomes False if odd number of clicks
                disabled = auto_clicks % 2 == 0
                return dash.no_update, dash.no_update, disabled

            elif trigger_id in {"step-button", "interval-timer"}:
                fig, stats = self.step_forward()
                return fig, stats, dash.no_update

            # Fallback (shouldn’t happen)
            return dash.no_update, dash.no_update, dash.no_update

        def reset_env(n_clicks):
            self.reset(seed=1)
            return self.get_graph_figure(), self.get_stats_text(), True  # Reset auto-run to off

        app.run(debug=True, port=port)

def reconstruct(state_vec, zones, H,t):
    idx = 0
    n_txn = defaultdict(int)
    n_arr = defaultdict(int)
    n_nxt = defaultdict(int)
    n_tilda = defaultdict(int)

    # n_txn
    for z in zones:
        for z_p in zones:
            for tau in range(1, H + 1):
                val = state_vec[idx]
                if val > 0:
                    n_txn[(z, z_p, tau)] = val
                idx += 1

    # n_arr
    for z in zones:
        val = state_vec[idx]
        if val > 0:
            n_arr[z] = val
        idx += 1

    # n_nxt
    for z in zones:
        for z_p in zones:
            val = state_vec[idx]
            if val > 0:
                n_nxt[(z, z_p)] = val
            idx += 1

    # n_tilda
    for z in zones:
        for z_p in zones:
            for tau in range(1, H + 1):
                val = state_vec[idx]
                if val > 0:
                    n_tilda[(z, z_p, tau)] = val
                idx += 1

    # Compute total vessels per zone
    n_tot = defaultdict(int)
    
    # Count arrivals
    for z, count in n_arr.items():
        n_tot[z] += count
        
    # Count vessels in transit (going to another port but not reached yet)
    for (z_src, z_dest, tau), count in n_txn.items():
        if tau > t:
            n_tot[z_src] += count
    
    
    # Ensure all zones are represented (even if 0)
    for z in zones:
        _ = n_tot[z]  # forces default 0 if missing
        
    # Compute edge_weight = n_nxt + sum(n_txn over all tau)
    edge_weight = defaultdict(int)

    for (z, z_p), val in n_nxt.items():
        edge_weight[(z, z_p)] += val

    for (z, z_p, tau), val in n_txn.items():
        edge_weight[(z, z_p)] += val

    return {
        'n_txn': n_txn,
        'n_nxt': n_nxt,
        'n_arr': n_arr,
        'n_tilda': n_tilda,
        'n_total': n_tot,
        'edge_weight': edge_weight
    }

# if __name__ == '__main__':
#     env = MaritimeTrafficEnv()
#     env.reset(seed=1)
#     env.render_dash(port=8050)
    # env = MaritimeTrafficEnv()
    # obs,info = env.reset(seed = 1)
    # action = np.full(env.action_space.shape, -np.inf, dtype=np.float32)
    # terminated = False
    
    # while not terminated:
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     info = reconstruct(obs,env.zones,env.H,env.t)
    #     # env.render()
        
    #     print(f"Time step: {env.t}")
    #     print(f"Action: {action}")
    #     print("Total ships per zone (n_total):")
    #     for zone, total in info['n_total'].items():
    #         print(f"  {zone}: {total}")

    #     print("Arrived ships per zone (n_total):")
    #     for zone, total in info['n_arr'].items():
    #         print(f"  {zone}: {total}")

    #     print("\nn_tilda (future transit commitments):")
    #     for key, val in sorted(info['n_tilda'].items()):
    #         print(f"  {key}: {val}")
        
    #     print("\nn_nxt (routing decisions this step):")
    #     for key, val in sorted(info['n_nxt'].items()):
    #         print(f"  {key}: {val}")
    #     print("\nn_txn (committed future transits):")
    #     for key, val in sorted(info['n_txn'].items()):
    #         if val != 0:
    #             print(f"  {key}: {val}")

    #     print("-" * 60)
    #     time.sleep(1) 
        
    # print("Finished")