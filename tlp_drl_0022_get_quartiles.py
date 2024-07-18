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

def get_state_classifications(state):
    out_state = []
    for obs, state_obs in zip(observation_classes, state):
        quantiles_np = quantile_df[obs].values
        s_class = 0
        for q in quantiles_np:
            if state_obs < q:
                break
            s_class += 1
        out_state.append(s_class)
    
    return out_state

env = gym.make('CartPole-v1')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
# print(q_table)

# tunable parameters
num_episodes = 10000
max_steps_per_episode = 100

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

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        #Exploration-Exploitation 
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated


        if reward > 0:
            apple = 1

        # update q-table
        state = get_state_classifications(state)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
        state = new_state
        rewards_current_episode += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

    # print(f'***********\n\nepisode: {episode} | step: {step}\n\n**********')
    # print(f'{q_table}\n**********\n\n')

np.savetxt(fname='data/cartpole_q_table.csv', X=q_table, delimiter=',')