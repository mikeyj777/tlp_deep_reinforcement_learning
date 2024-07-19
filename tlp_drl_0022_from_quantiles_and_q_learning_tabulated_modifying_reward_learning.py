import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym


render_mode = None

observations_df = pd.read_csv('data/cartpole_observations_from_random_params.csv')

num_state_buckets = 10
n = num_state_buckets
quantile_fractions = np.linspace(1/n,(n-1)/n, n-1, endpoint=True)

observation_classes = list(observations_df.columns)
i = 0
while i < len(observation_classes):
    obs = observation_classes[i]
    if 'Unnamed' in obs:
        del observation_classes[i]
        i = -1
    i += 1
    

quantile_list = []
for q in quantile_fractions:
    quantile = observations_df.quantile(q)
    quantile = quantile.to_dict()
    quantile_list.append(quantile)

quantile_df = pd.DataFrame(quantile_list)

quantile_df['quantile_fractions'] = quantile_fractions

def get_state_classifications_return_as_tuple(state):
    out_state = []
    for obs, state_obs in zip(observation_classes, state):
        quantiles_np = quantile_df[obs].values
        s_class = 0
        for q in quantiles_np:
            if state_obs < q:
                break
            s_class += 1
        out_state.append(s_class)
    
    return tuple(out_state)

env = gym.make('CartPole-v1', render_mode=render_mode)

action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]

# 'a' is the array which will hold the reward of each action that is chosen.  this will be broadcast to each spot in the state space
# the state space is comprised of the number of buckets across each dimension that is varied across all states.
# in the case of cartpole, there are 4 varied objects:  cart velocity, cart position, pole angle and pole tip veloc.
# the input data of state space is divided into buckets like a histogram, depending on where it falls.
# if there are N buckets, then the state space would be (N, N, N, N).  so a 4-D array each of size N.
# the q_table is of shape (N, N, N, N, action_space_size)
a = np.zeros(action_space_size)

# the broadcasting is a trick to build the appropriate size array.  the copy is to assign the broadcast to memory.
# without the copy, it is just a view.

q_table = np.broadcast_to(a, (num_state_buckets,) * state_space_size + a.shape).copy()

# tunable parameters
num_episodes = 8000
max_steps_per_episode = 200

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-Learning Algo
for episode in range(num_episodes):
    state = env.reset()[0]
    state = get_state_classifications_return_as_tuple(state)
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        if render_mode is not None:
            env.render()
        #Exploration-Exploitation 
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # reward = 0
        if done:
            reward = -300

        # update q-table
        
        target_location = state + (action,)
        new_state_tuple = get_state_classifications_return_as_tuple(new_state)
        q_table[target_location] = q_table[target_location] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state_tuple]))
        
        state = new_state_tuple
        rewards_current_episode += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

    if episode % 500 == 0:
            print(f'***********\n\nepisode: {episode} | step: {step}\n\n**********')
            # print(f'{q_table}\n**********\n\n')

rewards_all_episodes = np.array(rewards_all_episodes)
reward_block_size = 100
ave_rewards = []
i = 0
while i < len(rewards_all_episodes) - 1 + reward_block_size:
    ave_rewards.append(rewards_all_episodes[i:i+reward_block_size].mean())
    i += reward_block_size

plt.plot(ave_rewards)
plt.show()

# method from - https://stackoverflow.com/a/52145217/3825495.  reference for reading data.
import csv
fil_name = f'data/cartpole_q_table_modified_reward_state_quantiles_{num_state_buckets}.csv'
q_list = q_table.tolist()
with open(fil_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(q_list)

# test reading data
with open(fil_name, 'r') as f:
  reader = csv.reader(f)
  q_list_loaded = list(reader)

nw_q_list_loaded = []
for row in q_list_loaded:
    nwrow = []
    for r in row:
        nwrow.append(eval(r))
    nw_q_list_loaded.append(nwrow)

q_table_loaded = np.array(nw_q_list_loaded, dtype=float)


apple = 1