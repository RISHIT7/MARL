from environment.TrafficEnv import MaritimeTrafficEnv
from policy.ReinforcePolicy import ReinforcePolicy
import time

if __name__ == "__main__":
    env = MaritimeTrafficEnv(render_mode="human")
    agent = ReinforcePolicy(env)


    

    # num_episodes = 200

    # for episode in range(num_episodes):
    #     obs, _ = env.reset(seed=episode)
    #     done = False
    #     traj = []

    #     while not done:
    #         action, log_prob = agent.select_action(obs)
    #         next_obs, reward, done, _, _ = env.step(action)
    #         traj.append((log_prob, reward))
    #         obs = next_obs

    #     agent.update_policy(traj)

    #     if episode % 10 == 0:
    #         print(f"Episode {episode} done")

    # # Final rollout to see learned policy
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     action, _ = agent.select_action(obs)
    #     obs, _, done, _, _ = env.step(action)
    #     env.render()
    #     time.sleep(1)

    # env.close()
