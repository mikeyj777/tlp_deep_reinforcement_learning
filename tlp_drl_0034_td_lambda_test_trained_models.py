import pickle

import gymnasium as gym
import numpy as np

from tlp_drl_0034_td_lambda import Model, BaseModel, FeatureTransformer

def load_pickled_object(file_nm):
    with open(file_nm, 'rb') as f:
        obj = pickle.load(f)

    return obj


def play_one(model:Model, env):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        env.render()
        action = np.argmax(model.predict(observation))
        observation, reward, done, truncated, info = env.step(action)

        # update model
        
        if iters % 100 == 0:
            print(f'total reward so far in this episode: {totalreward}')

        iters += 1
    return totalreward

def main():
    env = gym.make('MountainCar-v0', render_mode='human')
    ft = FeatureTransformer(env)
    model = Model(env=env, feature_transformer=ft)
    fit_models = load_pickled_object(file_nm='data/models_array_mountain_car_td_lambda_2024_07_24_2251.pickle')
    model.models = fit_models
    play_one(model=model, env=env)

main()