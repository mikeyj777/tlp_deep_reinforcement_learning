import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_action(s, w):
    # 0 - left; 1 - right
    act = 0
    res = s.dot(w)
    if res > 0:
        act = 1

    return act

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

    return t

def play_multiple_episodes(env, T, params):
    episode_lengths = []

    for _ in range(T):
        length = play_one_episode(env, params)
        episode_lengths.append(length)
    
    ep_lens_df = pd.DataFrame(episode_lengths, columns = 'episode_length')
    print(f'stats on episod lengths: {ep_lens_df.describe()}')
    return ep_lens_df['episode_length'].mean()


def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    
    return episode_lengths, params

env = gym.make('CartPole-v1', render_mode='human')
episode_lengths, params = random_search(env)
plt.plot(episode_lengths)
plt.show()

print('*** Final run with final weights***')
play_multiple_episodes(env, 100, params)
