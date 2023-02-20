import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

sns.set_palette("muted")

from typing import List, Tuple


def get_metrics(dirname: str, experiment_name: str) -> pd.DataFrame:
    """
    Read in specific metrics for given experiment
    """

    experiment_metrics_list = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "checkpoints/metrics_history.csv"))
        experiment_metrics_list.append(metrics_df)

    experiment_metrics_concat = pd.concat(experiment_metrics_list)
    experiment_metrics = experiment_metrics_concat.groupby(experiment_metrics_concat.index)
    return experiment_metrics


def calculate_quartile_metrics(parent_dirname: str,
    env_names: List[str],
    experiment_names: List[str],
)-> Tuple[List, List, List, List]:

    """
    Calculate quartile metrics across all experiment replications, in all envrionments
    """
    print("\n")
    print("---------------------------------------------------------------------------")
    print("  Calculating quartile metrics for all experiments, in all environments")
    print("---------------------------------------------------------------------------")

    all_metrics = []
    all_medians = []
    all_lqs = []
    all_uqs = []

    for env in env_names:

        print("\n")
        print(f" ENV: {env}")

        dirname = os.path.join(parent_dirname, env)
        _analysis_dir = os.path.join(dirname, "analysis/")
        _plots_dir = os.path.join(_analysis_dir, "plots/")
        _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")
        _median_metrics_dir = os.path.join(_analysis_dir, "median_metrics/")

        os.makedirs(_analysis_dir, exist_ok=True)
        os.makedirs(_plots_dir, exist_ok=True)
        os.makedirs(_emitter_plots_dir, exist_ok=True)
        os.makedirs(_median_metrics_dir, exist_ok=True)

        metrics_list = []
        median_metrics_list = []
        lq_metrics_list = []
        uq_metrics_list = []
    
        for experiment_name in experiment_names:

            experiment_metrics = get_metrics(dirname, experiment_name)
            median_metrics = experiment_metrics.median(numeric_only=True)
            median_metrics.to_csv(f"{_median_metrics_dir}{experiment_name}_median_metrics")
            lq_metrics = experiment_metrics.apply(lambda x: x.quantile(0.25))
            uq_metrics =  experiment_metrics.apply(lambda x: x.quantile(0.75))
            metrics_list.append(experiment_metrics)
            median_metrics_list.append(median_metrics)
            lq_metrics_list.append(lq_metrics)
            uq_metrics_list.append(uq_metrics)

        all_metrics.append(metrics_list)
        all_medians.append(median_metrics_list)
        all_lqs.append(lq_metrics_list)
        all_uqs.append(uq_metrics_list)

    return all_metrics, all_medians, all_lqs, all_uqs


def get_final_metrics(dirname: str, 
    experiment_name: str,
    metric: str) -> np.array:
    """
    Load in final score of experiment across all replications for given metric
    """

    experiment_final_scores = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "checkpoints/metrics_history.csv"))
        final_score = np.array(metrics_df[metric])[-1]
        experiment_final_scores.append(final_score)

    return np.array(experiment_final_scores)