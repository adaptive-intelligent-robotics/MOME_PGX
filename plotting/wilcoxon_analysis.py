
import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
from plotting.load_datasets import get_metrics


def wilcoxon_analysis(
    parent_dirname,
    env_names,
    experiment_names,
    metrics_list,
    num_replications
):
    _analysis_dir = os.path.join(parent_dirname, "analysis/")
    _wilcoxon_dir = os.path.join(_analysis_dir, "wilcoxon_tests/")

    os.makedirs(_analysis_dir, exist_ok=True)
    os.makedirs(_wilcoxon_dir, exist_ok=True)

    for env in env_names:

        print("------------------------------")
        print(f"        ENV: {env}             ")
        print("------------------------------")
        
        dirname = os.path.join(parent_dirname, env)
        
        for metric in metrics_list:

            print("------------------------------")
            print(f"         {metric}             ")
            print("------------------------------")
            print("\n")

            all_final_metrics = {}

            for experiment in experiment_names:
                experiment_final_scores = get_final_metrics(dirname, experiment, metric)
                all_final_metrics[experiment] = experiment_final_scores[:num_replications]

            pvalue_df = pairwise_wilcoxon_analysis(all_final_metrics)
            
            pvalue_df.to_csv(f"{_wilcoxon_dir}/{env}_{metric}.csv")




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


def pairwise_wilcoxon_analysis(all_final_metrics: dict)-> None:
    """
    Calculate p-values for pairwise comparison of experiments for given metric
    """
    experiment_names = all_final_metrics.keys()
    p_value_df = pd.DataFrame(columns=experiment_names, index=experiment_names)

    for experiment_1 in experiment_names:
        experiment_1_p_values = []
        for experiment_2 in experiment_names:
            if experiment_1 == experiment_2:
                experiment_1_p_values.append(np.nan)
            else:
                res = wilcoxon(all_final_metrics[experiment_1], all_final_metrics[experiment_2])
                experiment_1_p_values.append(res.pvalue)
        p_value_df.loc[experiment_1] = experiment_1_p_values
    
    return p_value_df
