import gymnasium as gym
from gymnasium import wrappers

import pickle
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

monitor = True
pickle_models_when_done = True

class SGDRegressor:
    def __init__(self, D, lr = 0.1):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1
    
    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)
    

class FeatureTransformer:

    def __init__(self, env, n_components=500):
        self.set_observations_and_quantiles()
        observation_examples_df = self.observations_df.sample(n=10000)
        observation_examples = observation_examples_df.values
        observation_classifications = np.apply_along_axis(self.get_state_classifications, axis = 1, arr = observation_examples)
        state_space_size = len(self.quantile_df)

        scaler = StandardScaler()
        scaler.fit(observation_classifications)
        self.n_components = n_components
        
        rbfs = []
        gamma = 5
        for i in range(state_space_size):
            rbfs.append((f'rbf{i+1}', RBFSampler(gamma=gamma, n_components=n_components)))
            gamma /= 2

        # states to features
        featurizer = FeatureUnion(rbfs)

        example_features = featurizer.fit_transform(scaler.transform(observation_classifications))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def set_observations_and_quantiles(self):
    # stored game observations
        observations_df = pd.read_csv('data/cartpole_observations_from_random_params.csv')
        num_state_buckets = 10
        n = num_state_buckets
        quantile_fractions = np.linspace(0, 1, n+1, endpoint=True)

        quantile_list = []
        for q in quantile_fractions:
            quantile = observations_df.quantile(q)
            quantile = quantile.to_dict()
            quantile_list.append(quantile)
        quantile_df = pd.DataFrame(quantile_list)
        quantile_df['quantile_fractions'] = quantile_fractions
        self.quantile_df = quantile_df
        self.observations_df = observations_df

    def get_state_classifications(self, state):
        out_state = []
        observation_classes = self.quantile_df.columns
        for obs, state_obs in zip(observation_classes, state):
            quantiles_np = self.quantile_df[obs].values
            s_class = 0
            for q in quantiles_np:
                if state_obs <= q:
                    break
                if s_class < quantiles_np.shape[0]:
                    s_class += 1
            out_state.append(s_class)
    
        return out_state

    def transform(self, observations):
        obs_classified = []
        for obs in observations:
            one_obs_classified = self.get_state_classifications(obs)
            obs_classified.append(one_obs_classified)
        scaled = self.scaler.transform(obs_classified)
        ans = self.featurizer.transform(scaled)
        return ans

class Model:

    def __init__(self, env, feature_transformer, learning_rate = 0.1, models = None):
        self.env = env
        self.models = []            
        self.feature_transformer = feature_transformer
        if models is not None:
            self.models = models
            return
        
        for i in range(env.action_space.n):
            model = SGDRegressor(D = feature_transformer.dimensions, lr=0.1)
            model.partial_fit(feature_transformer.transform( [env.reset()[0]]), [0])
            self.models.append(model)
    
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        assert(len(result.shape) == 2)        
        return result
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        # call the model for the action taken, 'a'
        self.models[a].partial_fit(X, [G])
    
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
    
    def pickle_the_models(self):
        f_name = f'data/mountain_car_fit_model_n_components_{self.feature_transformer.n_components}_{datetime.now():%Y_%m_%d_%H%M}.pickle'
        with open(f_name, 'wb') as f:
            pickle.dump(self.models, f, pickle.DEFAULT_PROTOCOL)

    

def play_one(model, env, eps, gamma):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
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

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num = num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num = num_tiles)
    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z,
                rstride = 1, cstride=1, cmap = matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-to-go == -V(s)')
    ax.set_title('Cost-to-go Function')
    fig.colorbar(surf)
    plt.show()

def plot_running_average(totalrewards):
    N = len(totalrewards)
    running_ave = np.empty(N)
    for t in range(N):
        running_ave[t] = totalrewards[max(0, t-100):(t+1)].mean()
    
    plt.plot(running_ave)
    plt.title('Running Average')
    plt.show()

def main(n_components=500, show_plots = True):

    env = gym.make('CartPole-v1', render_mode = 'rgb_array').env
    ft = FeatureTransformer(env, n_components=n_components)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    if monitor:
        dir_name = f'data/mountain_car_q_learning_rbf_n_components_{n_components}_{datetime.now():%Y_%m_%d_%H%M}'
        env = wrappers.RecordVideo(env, dir_name)

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/(0.1*n+1)
        if n == 199:
            print(f'epsilon: {eps}')
        totalreward = play_one(model, env, eps, gamma)
        totalrewards[n] = totalreward
        if (n+1) % 10 == 0:
            print(f'n_components: {n_components}.  episode {n}.  total reward:  {totalreward}')
    if show_plots:
        plt.plot(totalrewards)
        plt.title('Rewards')
        plt.show()

        plot_running_average(totalrewards)

    if pickle_models_when_done:
        model.pickle_the_models()
    print(f'n_components: {n_components}.  average reward of last 100 episodes: {totalrewards[-100:].mean()}')
    total_steps = -totalrewards.sum()
    print(f'n_components: {n_components}.  total steps: {total_steps}') # -1 reward for each step that doesn't result in "done"

    env.close()

    return total_steps

n = 500
# steps = main(n_components=n, show_plots=True)


apple = 1
