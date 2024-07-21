import gymnasium as gym
import pickle
import numpy as np

from tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent import SGDRegressor, FeatureTransformer, Model

def load_pickled_object(file_nm):
    with open(file_nm, 'rb') as f:
        obj = pickle.load(f)

    return obj

def play_one(model, env, eps, gamma):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(observation, eps) # rps set to zero.  force model to find argmax action.
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

        if iters % 100 == 0:
            print(totalreward)
        
        iters += 1
    
    return totalreward

env = gym.make('CartPole-v1', render_mode = 'human').env
ft = FeatureTransformer(env, n_components=500)
models = load_pickled_object('data/mountain_car_fit_model_n_components_500_2024_07_21_1441.pickle')
model = Model(env = env, feature_transformer=ft, models=models)

rew = play_one(model, env, eps = 0, gamma=0.99)

apple = 1