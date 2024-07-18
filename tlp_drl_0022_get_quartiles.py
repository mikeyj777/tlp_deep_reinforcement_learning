import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    for i in range(len(observation_classes)):
        obs = observation_classes[i]
        quantiles_np= quantile_df[obs].values
        s_class = 0
        state_obs = state[i]
        out_state = []
        for q in quantiles_np:
            if state_obs < q:
                break
            s_class += 1
        out_state.append(s_class)




apple = 1