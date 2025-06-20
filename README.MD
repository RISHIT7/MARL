# Multiagent Decision Making for Maritime Traffic Management

This project implements the environment and policy gradient algorithm described in the 2019 paper: **"Multiagent Decision Making For Maritime Traffic Management"**.

## Overview

- **Environment:** Simulates vessel traffic through a chain of 5 maritime zones, allowing bidirectional transitions.
- **State Representation:** Count-based abstraction tracking vessels in transit, arrivals, next destinations, and committed routes.
- **Actions:** Policy outputs β values (after sigmoid transformation) that control transition probabilities for vessels between zones.
- **Reward:** Penalizes congestion in zones based on capacity constraints.
- **Learning:** Uses policy gradient (REINFORCE) with exact formulas from the paper, and value function computed via dynamic programming.

## Files

- `train_gym.py`: Main environment and training loop implementation.
- `requirements.txt`: Python dependencies.
- `aaai.pdf`: Reference paper.
- `train.py`: (Not described here; see file for details.)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the agent:**
   ```bash
   python train_gym.py
   ```

3. **Output:**  
   The script prints average rewards every 10 episodes and a final average reward after training.

## Reference

- [Multiagent Decision Making For Maritime Traffic Management (AAAI 2019)](https://jamesarambam.github.io/files/aaai19.pdf)