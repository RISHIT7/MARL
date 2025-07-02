# Multiagent Decision Making for Maritime Traffic Management

This project implements the environment and policy gradient algorithm described in the 2019 paper: **"Multiagent Decision Making For Maritime Traffic Management"**.

## Project Structure

```
MARL/
│
├── environment/
│   ├── env_dash.py         # Dash-based Gym environment (visual/interactive)
│   ├── parameters.py       # Environment parameters (HORIZON, W_R, W_D, etc.)
│   ├── TrafficEnv.py       # Main Gymnasium environment logic
│
├── policy/
│   └── policynetwork.py    # PolicyNetwork: PyTorch neural network for policy
│
├── Verification/
│   ├── run_tests.py        # Test runner (empty)
│   ├── test1.py            # Test cases
│   ├── test2.py
│   └── test3.py
│
├── Visualization/
│   └── graph_renderer.py   # Visualization utilities for traffic graphs
│
├── main.py                 # Minimal Gym environment runner (uses TrafficEnv)
├── main_dash.py            # Dash-based interactive environment runner
├── requirements.txt        # Python dependencies
├── aai19.pdf               # Reference paper
└── README.MD               # Project documentation
```

## Overview

- **Environment:** Simulates vessel traffic through a chain of maritime zones, allowing bidirectional transitions.
- **State Representation:** Count-based abstraction tracking vessels in transit, arrivals, next destinations, and committed routes.
- **Actions:** Policy outputs β values (after sigmoid transformation) that control transition probabilities for vessels between zones.
- **Reward:** Penalizes congestion in zones based on capacity constraints.
## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the standard Gym environment:**
   ```bash
   python main.py
   ```
   - Runs a random agent in the maritime environment and renders each step.

3. **Run the Dash-based interactive environment:**
   ```bash
   python main_dash.py
   ```
   - Launches a web-based dashboard for interactive visualization.

## Key Modules

- `environment/env_dash.py`: Dash-based Gym environment for interactive visualization.
- `environment/parameters.py`: Stores environment constants and parameters.
- `environment/TrafficEnv.py`: Main Gymnasium environment logic.
- `policy/policynetwork.py`: Defines the neural network policy.
- `Visualization/graph_renderer.py`: Visualization utilities for traffic graphs.
- `Verification/`: Contains test scripts and a test runner (empty).
- `main.py`: Minimal runner for the Gym environment.
- `main_dash.py`: Runner for the Dash-based environment.

## Reference

- [Multiagent Decision Making For Maritime Traffic Management (AAAI 2019)](https://jamesarambam.github.io/files/aaai19.pdf)
