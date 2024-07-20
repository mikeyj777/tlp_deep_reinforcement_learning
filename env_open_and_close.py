import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='rgb_array')
observation = env.reset()[0]
env.step(0)
env.close()

print('closed!')