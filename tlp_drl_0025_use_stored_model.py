import gymnasium as gym
import pickle
import numpy as np

# from tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent import SGDRegressor, FeatureTransformer

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
        action = model.sample_action(observation, eps) # force exploit.
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

        if iters % 100 == 0:
            print(totalreward)
        
        if done:
            reward = -300
            G = reward
        else:
            Q_next = model.predict(observation)
            G = reward + gamma * np.max(Q_next[0])
            
        model.update(prev_observation, action, G)
        totalreward += reward
        iters += 1
    
    return totalreward

model = load_pickled_object('data/mountain_car_fit_model_n_components_500_2024_07_21_1314.pickle')

env = gym.make('CartPole-v1', render_mode = 'rgb_array').env

play_one(model, env, eps = 0, gamma=0.99)

