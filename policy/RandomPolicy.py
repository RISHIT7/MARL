from environment.TrafficEnv import MaritimeTrafficEnv
import numpy as np

class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.action_shape = env.action_space.shape

    def select_action(self, observation=None):
        return np.random.uniform(low=-1e4, high=1e4, size=self.action_shape).astype(np.float32)