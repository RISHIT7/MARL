import numpy as np

class MaxPolicy:
    def __init__(self, env):
        self.env = env
        self.action_shape = env.action_space.shape

    def select_action(self, observation=None):
        # MaxPolicy always sets beta to 1, which corresponds to maximum travel time.
        # This introduces the largest possible delay through the policy.
        return np.full(self.action_shape,np.inf, dtype=np.float32)