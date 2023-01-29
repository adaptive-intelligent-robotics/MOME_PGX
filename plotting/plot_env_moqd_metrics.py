import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

sns.set_palette("muted")

from typing import List, Dict, Tuple

from plotting.plot_scores_evolution import plot_scores_evolution


def plot_envs_scores_evolution(parent_dirname: str,
    env_names: List[str],
    env_labels: List[str],
    experiment_names: List[str],
    experiment_labels: List[str],
    emitter_names: Dict,
    emitter_labels: Dict,
    metrics_list: List,
    median_metrics_list: List,
    lq_metrics_list: List,
    uq_metrics_list: List,
):

    for env_num, env in enumerate(env_names):

        print("------------------------------")
        print(f"          ENV: {env}             ")
        print("------------------------------")
        print("\n")

        dirname = os.path.join(parent_dirname, env)

        _analysis_dir = os.path.join(dirname, "analysis/")
        _plots_dir = os.path.join(_analysis_dir, "plots/")
        _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")
        _median_metrics_dir = os.path.join(_analysis_dir, "median_metrics/")

        #print_min_max_rewards(metrics_list[env_num])

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "max_sum_scores",
                        plot_ylabel = "Maximum Sum Scores",
                        plot_title = "Maximum Sum of Scores evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "max_hypervolume",
                        plot_ylabel = "Maximum hypervolume",
                        plot_title = "Maximum hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_max_hypervolume",
                        plot_ylabel = "Maximum normalised hypervolume",
                        plot_title = "Maximum normalised hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "moqd_score",
                        plot_ylabel = "MOQD Score",
                        plot_title = "MOQD Score evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        )   
        
        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_moqd_score",
                        plot_ylabel = "Normalised MOQD Score",
                        plot_title = "Normalised MOQD Score evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "global_hypervolume",
                        plot_ylabel = "Global Hypervolume",
                        plot_title = "Global hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_global_hypervolume",
                        plot_ylabel = "Normalised Global Hypervolume",
                        plot_title = "Normalised Global hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,

        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "coverage",
                        plot_ylabel = "Coverage",
                        plot_title = "Repertoire Coverage evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir
        )