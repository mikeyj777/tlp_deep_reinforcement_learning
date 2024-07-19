import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym


render_mode = 'human'

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


fil_name = 'data/cartpole_q_table_modified_reward.csv'

with open(fil_name, 'r') as f:
  reader = csv.reader(f)
  q_list_loaded = list(reader)

nw_q_list_loaded = []
for row in q_list_loaded:
    nwrow = []
    for r in row:
        nwrow.append(eval(r))
    nw_q_list_loaded.append(nwrow)

q_table = np.array(nw_q_list_loaded, dtype=float)
# tunable parameters
max_steps_per_episode = 200


# Q-Learning Algo
state = env.reset()[0]
state = get_state_classifications_return_as_tuple(state)
done = False
rewards_current_episode = 0

for step in range(max_steps_per_episode):
    if render_mode is not None:
        env.render()

    action = np.argmax(q_table[state])
    
    new_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated


    state = get_state_classifications_return_as_tuple(new_state)

    if done:
        break

apple = 1