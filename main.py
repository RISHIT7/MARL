from environment.TrafficEnvMain import MaritimeTrafficEnv
from policy.RandomPolicy import RandomPolicy
import pygame
from renderer.PygameGraphRenderer import PygameGraphRenderer  # Your custom file

import time

if __name__ == "__main__":
    env = MaritimeTrafficEnv()
    agent = RandomPolicy(env)
    renderer = PygameGraphRenderer(env.zones, env.valid_transitions)

    obs, _ = env.reset(seed=1)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break

        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)

        # Provide state to external renderer
        renderer.update(env.n_tot, env.edge_weight)

        time.sleep(5)

    renderer.close()
    env.close()
