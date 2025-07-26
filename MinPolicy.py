from environment.TrafficEnv import MaritimeTrafficEnv
from policy.MinPolicy import MinPolicy

import time
# Proper Gymnasium environment can be registered with gym.
if __name__ == "__main__":
	env = MaritimeTrafficEnv(render_mode="human")
	agent = MinPolicy(env)
	env.reset(seed=1)
	terminated = False
	truncated = False
	# env.render()
	while not terminated and not truncated:
		action = agent.select_action()
		# random selection
		# action = env.action_space.sample()

		obs, reward, terminated, truncated, info = env.step(action)
		# env.render()
		time.sleep(5)

	env.close()
