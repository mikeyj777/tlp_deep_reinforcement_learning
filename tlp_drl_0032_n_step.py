import gymnasium as gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import wrappers
from datetime import datetime

import tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent as q_learning
from tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent import plot_cost_to_go, FeatureTransformer, Model, plot_running_average

monitor = True

class SGDRegressor:

    def __init__(self, **kwargs):
        self.w = None
        self.lr = 0.01

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)
    
    def predict(self, X):
        return X.dot(self.w)
    
q_learning.SGDRegressor = SGDRegressor

def play_one(model, env, eps, gamma, n=5):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0

    multiplier = np.array([gamma]*n)**np.arange(n)
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)

        if len(rewards) >= n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)
        
        totalreward += reward
        iters += 1
    
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]
    
    if observation[0] >= 0.5: # car at end
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        while len(rewards) > 0:
            # assume next n steps return -1 (assume won't hit goal in next n steps)
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    
    return totalreward

def main():
    env = gym.make('MountainCar-v0').env
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    if monitor:
        dir_name = f'data/mountain_car_n_step_{datetime.now():%Y_%m_%d_%H%M}'
        env = wrappers.RecordVideo(env, dir_name)
    
    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print(f'episode: {n} | total reward: {totalreward}')
    print(f'avg reward for last 100 eps:  {totalrewards[-100].mean()}')
    print(f'total step: {-totalrewards.sum()}')
    