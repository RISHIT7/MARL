from environment.TrafficEnv import MaritimeTrafficEnv

import numpy as np

class MinPolicy:
    def __init__(self, env):
        self.env = env
        self.action_shape = env.action_space.shape

    def select_action(self, observation=None):
        return np.full(self.action_shape,-np.inf, dtype=np.float32)