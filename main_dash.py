from environment.env_dash import MaritimeTrafficEnv

if __name__ == '__main__':
    env = MaritimeTrafficEnv()
    env.reset(seed=1)
    env.render_dash(port=8050)