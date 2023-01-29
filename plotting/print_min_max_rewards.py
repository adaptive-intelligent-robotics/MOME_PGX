import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_palette("muted")

from typing import List


def print_env_min_max_rewards(
    env_names: List[str],
    experiment_metrics_list: List[pd.DataFrame]
)-> None:

    for env_num, env in enumerate(env_names):

        print("------------------------------")
        print(f"          ENV: {env}             ")
        print("------------------------------")
        print("\n")

        print_min_max_rewards(experiment_metrics_list[env_num])


def print_min_max_rewards(
    experiment_metrics_list: List[pd.DataFrame]
):

    mins_1 = []
    mins_2 = []
    maxs_1 = []
    maxs_2 = []

    for exp_metrics in experiment_metrics_list:
        # Sort each replication into columns
        exp_min_metrics = exp_metrics.obj.groupby(level=0).agg(list)["min_scores"].apply(pd.Series)
        exp_max_metrics = exp_metrics.obj.groupby(level=0).agg(list)["max_scores"].apply(pd.Series)

        # find min and max of each score for each replication
        for col in exp_min_metrics.columns:
            exp_min_scores = pd.DataFrame(exp_min_metrics[col].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' ')).to_list(), columns=['score1', 'score2'])
            exp_max_scores = pd.DataFrame(exp_max_metrics[col].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' ')).to_list(), columns=['score1', 'score2'])

            min_score1 = exp_min_scores["score1"].min()
            min_score2 = exp_min_scores["score2"].min()
            max_score1 = exp_max_scores["score1"].max()
            max_score2 = exp_max_scores["score2"].max()

            mins_1.append(min_score1)
            mins_2.append(min_score2)
            maxs_1.append(max_score1)
            maxs_2.append(max_score2)

    print(f"MIN SCORES ACROSS ALL EXPERIMENTS: [{np.min(mins_1)}, {np.min(mins_2)}]",)
    print(f"MAX SCORES ACROSS ALL EXPERIMENTS: [{np.max(maxs_1)}, {np.max(maxs_2)}]", )
