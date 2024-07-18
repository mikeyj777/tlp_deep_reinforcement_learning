import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

observations_df = pd.read_csv('data/cartpole_observations_from_random_params.csv')

num_state_buckets = 10
n = num_state_buckets
quantile_fractions = np.linspace(1/n,(n-1)/n, n-1, endpoint=True)

quantile_list = []
for q in quantile_fractions:
    quantile = observations_df.quantile(q)
    quantile = quantile.to_dict()
    quantile_list.append(quantile)

quantile_df = pd.DataFrame(quantile_list)

quantile_df['quantile_fractions'] = quantile_fractions



apple = 1