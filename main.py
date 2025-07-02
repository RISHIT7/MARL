from environment.TrafficEnv import MaritimeTrafficEnv
import time
# Proper Gymnasium environment can be registered with gym.
if __name__ == "__main__":
	env = MaritimeTrafficEnv(render_mode="human")
	env.reset(seed=1)
	terminated = False
	truncated = False
	env.render()
	while not terminated and not truncated:
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		env.render()
		time.sleep(1)

	env.close()
