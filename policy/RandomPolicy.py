from environment.TrafficEnv import MaritimeTrafficEnv
import numpy as np

class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
    
    def select_action(self):
        # Generate random actions for each road
        return self.action_space.sample()