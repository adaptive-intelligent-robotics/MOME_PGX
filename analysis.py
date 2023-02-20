import os

from typing import List, Dict, Tuple

from analysis_functions.compute_data_efficiency import compute_data_efficiency
from analysis_functions.load_datasets import calculate_quartile_metrics
from analysis_functions.pairwise_coverage_analysis import print_pairwise_coverage
from analysis_functions.plot_env_moqd_metrics import plot_envs_scores_evolution
from analysis_functions.plot_emitter_counts import plot_env_emitter_counts
from analysis_functions.plot_final_pfs import plot_pfs
from analysis_functions.plot_grid import plot_experiments_grid
from analysis_functions.print_min_max_rewards import print_env_min_max_rewards
from analysis_functions.wilcoxon_analysis import wilcoxon_analysis
from analysis_functions.appendix_results import plot_coverage_appendix_results


def run_analysis(parent_dirname: str,
    env_names: List[str],
    env_labels: Dict,
    experiment_names: List[str],
    experiment_labels: Dict,
    emitter_names: Dict,
    emitter_labels: Dict,
    grid_plot_metrics_list: List[str],
    grid_plot_metrics_labels: Dict,
    grid_plot_linestyles: Dict,
    p_value_metrics_list: List[str],
    plot_envs: bool=True,
    plot_emitter_counts: bool=True,
    plot_grid: bool=True,
    print_min_max_scores: bool=True,
    pairwise_coverage_analysis: bool=True,
    plot_fronts: bool=True,
    reward1_label: str="Reward 1",
    reward2_label: str="Reward 2",
    num_replications: int=20,
    calculate_p_values: bool=True,
    analyse_data_efficiency: bool=True,
    data_efficiency_params: Dict={},
    num_iterations: int=4000,
    episode_length: int=1000,
    batch_size: int=256,
    x_axis_evaluations: bool=True
) -> None:

    if plot_fronts:
        plot_pfs(parent_dirname,
            env_names,
            env_labels,
            experiment_names,
            experiment_labels,
            num_replications,
            reward1_label,
            reward2_label,
        )

    if pairwise_coverage_analysis:
        print_pairwise_coverage(parent_dirname, 
            env_names,
            experiment_names,
        )
    
    if calculate_p_values:
        wilcoxon_analysis(parent_dirname,
            env_names,
            experiment_names,
            p_value_metrics_list,
            num_replications,
        )
    
    all_metrics, all_medians, all_lqs, all_uqs = calculate_quartile_metrics(parent_dirname,
        env_names,
        experiment_names,
    )

    if analyse_data_efficiency:
        compute_data_efficiency(
            all_medians,
            env_names,
            experiment_names,
            data_efficiency_params
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
            num_iterations,
            episode_length,
            batch_size,
            x_axis_evaluations
        )
    
    if plot_emitter_counts:
        plot_env_emitter_counts(parent_dirname,
            env_names,
            experiment_names,
            experiment_labels,
            emitter_names,
            emitter_labels,
            all_metrics,
            num_iterations,
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
            all_uqs,
            num_iterations,
            episode_length,
            batch_size,
            x_axis_evaluations
        )
    
    if print_min_max_scores:
        print_env_min_max_rewards(
            env_names,
            all_metrics
        )    


    


if __name__ == '__main__':

    # Parent directory of results
    parent_dirname = "results/"


    # Directory names of experiments
    experiment_names = [
        # main
        "mome_pgx",

        # baselines
        #"pga",
        #"nsga2",
        #"spea2",

        # ablations
        "mome", 
        "biased_mome",
        "mopga", 
        "mopga_only_energy",
        "mopga_only_forward",

    ]

    # Directory names of environments
    env_names=["ant_multi",
        "halfcheetah_multi",
        "hopper_multi",
        "walker2d_multi",
    ]


    # Names of experiments (in same order as above) for legends/titles
    experiment_labels = {
        # main
        "mome_pgx": "MOME-PGX",

        # baselines 
        #"pga": "PGA",
        #"nsga2": "NSGA-II",
        #"spea2": SPEA2",

        # ablations
        "mome": "MOME", 
        "biased_mome": "MOME + Crowding",
        "mopga": "MO-PGA", 
        "mopga_only_energy": "MO-PGA (Only Energy)",
        "mopga_only_forward": "MO-PGA (Only Forward)",

    }


    # Dictionary of emitter names for each experiment
    emitter_names = {"mome": ["emitter_mutation_count:", "emitter_variation_count:"], 
            "mopga": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "mome_pgx": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "mopga_only_forward": ["emitter_1_count:", "emitter_2_count:"],
            "mopga_only_energy": ["emitter_1_count:", "emitter_2_count:"],
            "nsga2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "spea2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "pga": ["emitter_1_count:", "emitter_2_count:"],
            "biased_mome":  ["emitter_mutation_count:", "emitter_variation_count:"], 
    }

    # Legend names of emitters for each experiment
    emitter_labels = {"mome": ["Mutation Emitter", "Variation Emitter"], 
                "mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "mome_pgx": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "mopga_only_forward": ["Forward Reward Emitter", "GA Emitter"],
                "mopga_only_energy": ["Forward Reward Emitter",  "GA Emitter"],
                "nsga2": ["Mutation Emitter", "Variation Emitter"],
                "spea2": ["Mutation Emitter", "Variation Emitter"],
                "pga": ["Gradient Emitter", "GA Emitter"],
                "biased_mome": ["Mutation Emitter", "Variation Emitter"]
    }



    # Legend/Title names of environments
    env_labels={"ant_multi": "Ant",
        "halfcheetah_multi": "HalfCheetah",
        "hopper_multi": "Hopper",
        "walker2d_multi": "Walker2d",
    }
    
    # Metrics to plot in grid plot
    grid_plot_metrics_list = [
        "moqd_score", 
        "global_hypervolume", 
        "max_sum_scores",
        #"coverage"
    ]

    grid_plot_metrics_labels = {
        "moqd_score":"MOQD Score", 
        "global_hypervolume": "Global Hypervolume", 
        "max_sum_scores": "Max Sum Scores",
        #"coverage": "Coverage"
    }

    # Linestyles for experiments in grid plot
    grid_plot_linestyles = {
            # main
            "mome_pgx": 'solid',

            # baselines
            "mome": 'dashed',
            "pga": (0, (3, 1, 1, 1, 1, 1)),
            "nsga2": 'dotted',
            "spea2": (5, (10, 3)),

            # ablations
            "mopga": (0, (3, 1, 1, 1, 1, 1)),
            "biased_mome": 'dashdot',
            "mopga_only_forward": 'dotted',
            "mopga_only_energy": (5, (10, 3)),
    }


    # List of metrics to calculate p-values for
    p_value_metrics_list = [
        "global_hypervolume", 
        "max_sum_scores", 
        "moqd_score",
    ]

    # Which algorithms to compare data-efficiency and which metric for comparison
    data_efficiency_params={
        "test_algs": ["mome_pgx"],
        "baseline_algs": ["mome"],
        "metrics": ["moqd_score"],
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
        p_value_metrics_list,
        plot_envs=False,
        plot_emitter_counts=False,
        plot_grid=False,
        print_min_max_scores=True,
        pairwise_coverage_analysis=False,
        plot_fronts=False,
        reward1_label="Forward Velocity",
        reward2_label="Energy Consumption",
        num_replications=15,
        calculate_p_values=False,
        analyse_data_efficiency=False,
        data_efficiency_params=data_efficiency_params,
        num_iterations=4000,
        episode_length=1000,
        batch_size=256,
        x_axis_evaluations=True
    )