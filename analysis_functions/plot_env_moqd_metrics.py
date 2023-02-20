import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

sns.set_palette("muted")

from typing import List, Dict, Tuple





def plot_envs_scores_evolution(parent_dirname: str,
    env_names: List[str],
    env_labels: Dict,
    experiment_names: List[str],
    experiment_labels: Dict,
    emitter_names: Dict,
    emitter_labels: Dict,
    metrics_list: List,
    median_metrics_list: List,
    lq_metrics_list: List,
    uq_metrics_list: List,
    num_iterations: int,
    episode_length: int,
    batch_size: int,
    x_axis_evaluations: bool,
):
    print("\n")
    print("-------------------------------------------------------------------------")
    print(" Plotting scores across different metrics in each of the environments")
    print("-------------------------------------------------------------------------")
    experiment_labels = experiment_labels.values()

    for env_num, env in enumerate(env_names):

        print("\n")
        print(f"     ENV: {env}             ")

        dirname = os.path.join(parent_dirname, env)

        _analysis_dir = os.path.join(dirname, "analysis/")
        _plots_dir = os.path.join(_analysis_dir, "plots/")
        _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")
        _median_metrics_dir = os.path.join(_analysis_dir, "median_metrics/")

        env_name = env_labels[env]

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "max_sum_scores",
                        plot_ylabel = "Maximum Sum Scores",
                        plot_title = f"{env_name}: Maximum Sum of Scores evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "max_hypervolume",
                        plot_ylabel = "Maximum hypervolume",
                        plot_title = f"{env_name}: Maximum hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_max_hypervolume",
                        plot_ylabel = "Maximum normalised hypervolume",
                        plot_title = f"{env_name}: Maximum normalised hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        ) 

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "moqd_score",
                        plot_ylabel = "MOQD Score",
                        plot_title = f"{env_name}: MOQD Score evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        )   
        
        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_moqd_score",
                        plot_ylabel = "Normalised MOQD Score",
                        plot_title = f"{env_name}: Normalised MOQD Score evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "global_hypervolume",
                        plot_ylabel = "Global Hypervolume",
                        plot_title = f"{env_name}: Global hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "normalised_global_hypervolume",
                        plot_ylabel = "Normalised Global Hypervolume",
                        plot_title = f"{env_name}: Normalised Global hypervolume evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        )   

        plot_scores_evolution(median_metrics_list[env_num], 
                        lq_metrics_list[env_num],
                        uq_metrics_list[env_num],
                        metrics_label = "coverage",
                        plot_ylabel = "Coverage",
                        plot_title = f"{env_name}: Repertoire Coverage evolution during training",
                        experiment_labels = experiment_labels, 
                        save_dir = _plots_dir,
                        num_iterations = num_iterations,
                        episode_length = episode_length,
                        batch_size = batch_size,
                        x_axis_evaluations = x_axis_evaluations,
        )




def plot_scores_evolution(
    median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    metrics_label: str,
    plot_ylabel: str,
    plot_title: str,
    experiment_labels: Dict,
    save_dir: str,
    num_iterations: int,
    episode_length: int,
    batch_size: int,
    x_axis_evaluations: bool=True
):

    # Visualize the training evolution and final repertoire

    if x_axis_evaluations:
        x_range = np.arange(num_iterations + 1)  * batch_size
        x_label = "Number of evaluations"

    else:
        x_range = np.arange(num_iterations + 1) * episode_length * batch_size
        x_label = "Environment steps"

    fig = plt.figure()
    ax = fig.add_subplot(111)  

    for exp_num, exp_name in enumerate(experiment_labels):
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_ylabel)
        ax.set_title(plot_title)
        ax.plot(x_range, median_metrics[exp_num][metrics_label], label=exp_name)
    
    for exp_num, exp_name in enumerate(experiment_labels):
        ax.fill_between(x_range, 
            lq_metrics[exp_num][metrics_label], 
            uq_metrics[exp_num][metrics_label], 
            alpha=0.2)

    plt.figlegend(loc = 'lower center', labels=experiment_labels)

    fig.tight_layout(pad=5.0)

    plt.savefig(os.path.join(save_dir, metrics_label))
    plt.close()
