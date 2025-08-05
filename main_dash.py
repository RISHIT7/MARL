from environment.maritime_env_dash import MaritimeTrafficEnv
from policy.RandomPolicy import RandomPolicy
import time

if __name__ == "__main__":
    env = MaritimeTrafficEnv()
    agent = RandomPolicy(env)
    
    print("Starting Maritime Traffic Environment with Dash...")
    print("Open your browser to view the visualization")
    
    obs, _ = env.reset(seed=1)
    terminated = False
    truncated = False
    
    step_count = 0
    max_steps = 100
    
    while not terminated and not truncated and step_count < max_steps:
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        print(f"Step {step_count}: Reward = {reward}")
        time.sleep(0.5)  # Slow down for better visualization
    
    print(f"Simulation ended after {step_count} steps")
    if hasattr(env, 'close'):
        env.close()