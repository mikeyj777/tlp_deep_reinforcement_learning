import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

render_mode = 'rgb_array'

def get_stored_params():
    params = np.genfromtxt('data/cartpole_weights_random_search.csv', delimiter=',')

    return params

def get_action(s, w):
    # 0 - left; 1 - right
    act = 0
    res = s.dot(w)
    if res > 0:
        act = 1

    return act

def play_one_episode(env, params, render_mode = None):
    observation = env.reset()[0]
    done = False
    t = 0

    while not done and t < 10000:
        # if render_mode is not None:
        #     env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

    return t



env = gym.make('CartPole-v1', render_mode=render_mode)
env = RecordVideo(env, 'data')
params = get_stored_params()
final_len = play_one_episode(env, params, render_mode=render_mode)
print(f'final len: {final_len}')

