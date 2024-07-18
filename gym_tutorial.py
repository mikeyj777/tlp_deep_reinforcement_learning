import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = 'human')

trials = 50
for _ in range(trials):
    i = 0
    env.reset()
    done = False
    while not done:
        i += 1
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

    print(i)

apple = 1