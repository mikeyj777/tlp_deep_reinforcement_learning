import gymnasium as gym
from gymnasium import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

f_name = 'data/cartpole_observations_from_random_params.csv'
monitor = True

# parameters
rand_seed = 41
lr = 0.01
test_size = 0.2
epochs = 100
# output_every_x_epochs = epochs // 10
output_every_x_epochs = 10

# set your seed
torch.manual_seed(rand_seed)

class Classifier:

    def __init__(self, f_name = None):
        if f_name is None:
            return
        self.f_name = f_name
        self.set_observations_and_quantiles()

    def set_observations_and_quantiles(self, f_name = None):
        # stored game observations
        if self.f_name is None:
            self.f_name = f_name
        observations_df = pd.read_csv(self.f_name)
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

# create model class that inherits nn.Module
class Model(nn.Module):
    # input layer (cart pole game state space - 4 variables) ->
    # hidden layer (h1) -> 
    # hidden layer (h2) ->
    # output (2 actions)
    
    def __init__(self, in_features=4, h1 = 8, h2 = 9, out_features = 2, classifier = None) -> None:
        # in_features = 4 - 4 flower features (sepal length & width, pedal len & width)
        # h1 = 8, h2 = 9 - num neurons in each hidden layer
        # out_features - images classified into one of 3 classes
        
        super().__init__() # instantiate super class (nn.Module)
        self.classifier = classifier
        if self.classifier is None:
            self.classifier = Classifier(f_name=f_name)

        self.fc1 = nn.Linear(in_features=in_features, out_features=h1) # fc1 = "fully connected layer 1".  connects from input to h1
        self.fc2 = nn.Linear(in_features=h1, out_features=h2)
        self.out = nn.Linear(in_features=h2, out_features=out_features)
        
    
    def transform(self, x):
        x = self.classifier.get_state_classifications(x)
        return torch.FloatTensor(x)

    def forward(self, x):
        x = self.transform(x)
        x = F.relu(self.fc1(x)) # relu - value above zero.  zero otherwise
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x_np = x.detach().numpy()
        return np.argmax(x_np)
    
    def sample_action(self, s, eps, env):
        if np.random.random() < eps:
            return env.action_space.sample()
        else:
            return np.argmax(self.forward(s))
    
def play_one(model, env, eps, gamma):
    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0
    # set criterion to measure error, how far off predictions are from 
    criterion = nn.CrossEntropyLoss()

    # choose Adam optimizer, set learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    while not done and iters < 2000:
        action = model.sample_action(observation, eps, env)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

        if iters % 100 == 0:
            print(totalreward)
        
        if done:
            reward = -300
            G = reward
            break
        
        # measure loss
        Q_next = model.forward(observation)
        action_tensor = torch.FloatTensor([action])
        Q_next_tensor = torch.FloatTensor([Q_next])
        loss = criterion(action_tensor, Q_next_tensor)
        loss.requires_grad = True
        loss_np = loss.detach().numpy()
        loss_val = loss_np
        losses.append(loss_val)
        
        # G = reward + gamma * np.max(Q_next[0])

        if iters % output_every_x_epochs == 0:
            print(f'epoch: {iters}. loss: {loss}')

        # back prop - take error rate in forward prop, feed back thru network to fine tune weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalreward += reward
        iters += 1
    
    return totalreward

def main(show_plots = True):

    env = gym.make('CartPole-v1', render_mode = 'rgb_array').env
    classifier = Classifier(f_name=f_name)
    model = Model(classifier=classifier)
    gamma = 0.99

    if monitor:
        dir_name = f'data/mountain_car_q_learning_torch_{datetime.now():%Y_%m_%d_%H%M}'
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
            print(f'episode {n}.  total reward:  {totalreward}')
    if show_plots:
        plt.plot(totalrewards)
        plt.title('Rewards')
        plt.show()

        plot_running_average(totalrewards)

    print(f'average reward of last 100 episodes: {totalrewards[-100:].mean()}')
    total_steps = -totalrewards.sum()
    print(f'total steps: {total_steps}') # -1 reward for each step that doesn't result in "done"

    env.close()

    return total_steps

def plot_running_average(totalrewards):
    N = len(totalrewards)
    running_ave = np.empty(N)
    for t in range(N):
        running_ave[t] = totalrewards[max(0, t-100):(t+1)].mean()
    
    plt.plot(running_ave)
    plt.title('Running Average')
    plt.show()

main()