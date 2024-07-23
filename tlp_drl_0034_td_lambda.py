import gymnasium as gym

import pickle
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import wrappers
from datetime import datetime

from tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent import plot_cost_to_go, FeatureTransformer, plot_running_average

monitor = True

class BaseModel: # similar to SGDRegressor, but utilizes "eligibility vector"
    def __init__(self, D):
        self.w = np.random.random(D) / np.sqrt(D)
    
    def partial_fit(self, input_, target, eligibility, lr = 0.01):
        self.w += lr*(target - input_.dot(self.w))*eligibility
    
    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
            model = BaseModel(D)
            self.models.append(model)
        
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X) for m in self.models])
    
    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        self.eligibilities *= gamma*lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])
    
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        
        return np.argmax(self.predict(s))

    def pickle_the_models(self):
        f_name = f'data/models_array_mountain_car_td_lambda_{datetime.now():%Y_%m_%d_%H%M}.pickle'
        with open(f_name, 'wb') as f:
            pickle.dump(self.models, f, pickle.DEFAULT_PROTOCOL)
    
def play_one(model:Model, env, eps, gamma, lambda_):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)

        # update model
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward += reward

        if iters % 100 == 0:
            print(f'total reward so far in this episode: {totalreward}')

        iters += 1
    return totalreward

def main():
    env = gym.make('MountainCar-v0', render_mode = 'rgb_array').env
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_ = 0.7

    if monitor:
        monitor_dir = f'data/td_lambda_{datetime.now():%Y_%m_%d_%H%M}'
        env = wrappers.RecordVideo(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        print(f'starting episode {n}')
        totalreward = play_one(model, env, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        print(f'episode: {n} | total reward: {totalreward}')
    print(f'average reward for last 100 episodes: {totalrewards[-100:].mean()}')

    try:
        model.pickle_the_models()

        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_average(totalrewards)

        # plot the optimal state-value function
        plot_cost_to_go(env, model)

    except Exception as e:
        print(f'error: {e}')

main()

apple = 1
