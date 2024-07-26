import gymnasium as gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import wrappers
from datetime import datetime

# code we already wrote
from tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent import plot_cost_to_go, FeatureTransformer, plot_running_average


class BaseModel:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)

  def partial_fit(self, input_, target, eligibility, lr=1e-2):
    self.w += lr*(target - input_.dot(self.w))*eligibility

  def predict(self, X):
    X = np.array(X)
    return X.dot(self.w)


# Holds one BaseModel for each action
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
    assert(len(X.shape) == 2)
    result = np.stack([m.predict(X) for m in self.models]).T
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G, gamma, lambda_):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    self.eligibilities *= gamma*lambda_
    self.eligibilities[a] += X[0]
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, lambda_):
  observation = env.reset()[0]
  done = False
  totalreward = 0
  iters = 0
  # while not done and iters < 200:
  while not done and iters < 10000:
    # debug xxx apple
    eps = -1
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, truncated, info = env.step(action)

    # update the model
    Qnext = model.predict(observation)
    assert(Qnext.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(Qnext[0])
    model.update(prev_observation, action, G, gamma, lambda_)

    totalreward += reward
    iters += 1

  return totalreward


if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft)
  gamma = 0.9999
  lambda_ = 0.7

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 300
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    # eps = 0.5/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma, lambda_)
    totalrewards[n] = totalreward
    print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_average(totalrewards)

  # plot the optimal state-value function
  plot_cost_to_go(env, model)