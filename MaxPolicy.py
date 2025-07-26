from environment.TrafficEnv import MaritimeTrafficEnv
from policy.ReinforcePolicy import ReinforcePolicy
from policy.MaxPolicy import MaxPolicy

import time
# Proper Gymnasium environment can be registered with gym.
if __name__ == "__main__":
	env = MaritimeTrafficEnv(render_mode="human")
	agent = MaxPolicy(env)
	env.reset(seed=1)
	terminated = False
	truncated = False
	# env.render()
	while not terminated and not truncated:
		action = agent.select_action()
		# random selection
		# action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		print(env.t,reward)
		# if(terminated): env.render()
		time.sleep(2)

	env.close()
