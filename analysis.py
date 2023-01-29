import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

sns.set_palette("muted")

from typing import List, Dict, Tuple

from plotting.load_datasets import calculate_quartile_metrics
from plotting.pairwise_coverage_analysis import print_pairwise_coverage
from plotting.plot_env_moqd_metrics import plot_envs_scores_evolution
from plotting.plot_emitter_counts import plot_env_emitter_counts
from plotting.plot_grid import plot_experiments_grid
from plotting.print_min_max_rewards import print_env_min_max_rewards

def run_analysis(parent_dirname: str,
    env_names: List[str],
    env_labels: List[str],
    experiment_names: List[str],
    experiment_labels: List[str],
    emitter_names: Dict,
    emitter_labels: Dict,
    grid_plot_metrics_list: List[str],
    grid_plot_metrics_labels: List[str],
    grid_plot_linestyles: Dict,
    plot_envs: bool=True,
    plot_emitter_counts: bool=True,
    plot_grid: bool=True,
    print_min_max_scores: bool=True,
    pairwise_coverage_analysis: bool=True,
) -> None:

    all_metrics, all_medians, all_lqs, all_uqs = calculate_quartile_metrics(parent_dirname,
        env_names,
        experiment_names,
    )

    if plot_envs:
        plot_envs_scores_evolution(parent_dirname,
            env_names,
            env_labels,
            experiment_names,
            experiment_labels,
            emitter_names,
            emitter_labels,
            all_metrics,
            all_medians,
            all_lqs, 
            all_uqs,
        )
    
    if plot_emitter_counts:
        plot_env_emitter_counts(parent_dirname,
            env_names,
            experiment_names,
            experiment_labels,
            emitter_names,
            emitter_labels,
            all_metrics,
        )

    if plot_grid:
        plot_experiments_grid(parent_dirname,
            env_names,
            env_labels,
            experiment_names,
            experiment_labels,
            grid_plot_metrics_list,
            grid_plot_metrics_labels,
            grid_plot_linestyles,
            all_medians, 
            all_lqs, 
            all_uqs
        )
    
    if print_min_max_scores:
        print_env_min_max_rewards(
            env_names,
            all_metrics
        )    

    if pairwise_coverage_analysis:
        print_pairwise_coverage(parent_dirname, 
            env_names,
            experiment_names,
        )



if __name__ == '__main__':

    # Parent directory of results
    parent_dirname = "results/paper_results/"


    # Directory names of experiments
    experiment_names = [
        "biased_mopga",
        "mopga", 
        "mopga_only_energy",
        "mopga_only_forward",
        "mome", 
        "pga",
        "nsga2",
        "spea2",
    ]

    # Names of experiments (in same order as above) for legends/titles
    experiment_labels = [
        "Biased MOPGA",
        "MOPGA", 
        "MOPGA (Only Energy)",
        "MOPGA (Only Forward)",
        "MOME", 
        "PGA",
        "NSGA-II",
        "SPEA2",
    ]


    # Dictionary of emitter names for each experiment
    emitter_names = {"mome": ["emitter_mutation_count:", "emitter_variation_count:"], 
            "mopga": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "biased_mopga": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "mopga_only_forward": ["emitter_1_count:", "emitter_2_count:"],
            "mopga_only_energy": ["emitter_1_count:", "emitter_2_count:"],
            "nsga2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "spea2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "pga": ["emitter_1_count:", "emitter_2_count:"],
            "bandit_mopga": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
    }

    # Legend names of emitters for each experiment
    emitter_labels = {"mome": ["Mutation Emitter", "Variation Emitter"], 
                "mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "biased_mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "mopga_only_forward": ["Forward Reward Emitter", "GA Emitter"],
                "mopga_only_energy": ["Forward Reward Emitter",  "GA Emitter"],
                "nsga2": ["Mutation Emitter", "Variation Emitter"],
                "spea2": ["Mutation Emitter", "Variation Emitter"],
                "pga": ["Gradient Emitter", "GA Emitter"],
                "bandit_mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
    }

    # Directory names of environments
    env_names=["ant_multi",
        "halfcheetah_multi",
        "hopper_multi",
        "walker2d_multi",
        "humanoid_multi"
    ]

    # Legend/Title names of environments
    env_labels=["Ant",
        "HalfCheetah",
        "Hopper",
        "Walker2d",
        "Humanoid"
    ]
    
    # Metrics to plot in grid plot
    grid_plot_metrics_list = ["moqd_score", "global_hypervolume", "max_sum_scores", "coverage"]
    grid_plot_metrics_labels = ["MOQD Score", "Global Hypervolume", "Max Sum Scores", "Coverage"]

    # Linestyles for experiments in grid plot
    grid_plot_linestyles = {"mome": 'solid',
            "mopga": 'dashed',
            "biased_mopga": 'solid',
            "mopga_only_forward": 'dashed',
            "mopga_only_energy": 'dashed',
            "nsga2": 'solid',
            "spea2": 'solid',
            "pga": 'solid',
            "biased_mome": 'dashed',
    }



    run_analysis(parent_dirname, 
        env_names,
        env_labels,
        experiment_names,
        experiment_labels, 
        emitter_names,
        emitter_labels,
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
        grid_plot_linestyles,
        plot_envs=False,
        plot_emitter_counts=False,
        plot_grid=True,
        print_min_max_scores=False,
        pairwise_coverage_analysis=False,
    )