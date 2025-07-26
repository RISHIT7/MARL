import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BetaPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Sigmoid()  # Outputs in (0,1)
        )

    def forward(self, obs):
        return self.net(obs)

class ReinforcePolicy:
    def __init__(self, env, lr=1e-3):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = BetaPolicyNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Stores [(log_beta, reward), ...]
        self.episode_memory = []

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        beta = self.policy(obs_tensor)  # Shape: [act_dim]
        log_beta = torch.log(beta + 1e-6)  # Prevent log(0)
        action = beta.detach().cpu().numpy()
        self.last_log_beta = log_beta  # Store for update
        return action

    def store_reward(self, reward):
        # Store log_beta from previous step + reward
        self.episode_memory.append((self.last_log_beta, reward))

    def update_policy(self):
        if not self.episode_memory:
            return
        log_betas, rewards = zip(*self.episode_memory)

        returns = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Compute total loss
        loss = 0
        for log_beta, R in zip(log_betas, returns):
            loss -= (log_beta.sum()) * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.episode_memory = []  # Reset for next episode
