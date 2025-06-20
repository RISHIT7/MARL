import torch.nn as nn

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
