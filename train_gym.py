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
import torch.optim as optim
import torch
from environment.env import MaritimeTrafficEnv
from policy.policynetwork import PolicyNetwork

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
