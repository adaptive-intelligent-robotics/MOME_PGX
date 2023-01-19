import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from typing import List, Dict
from qdax.core.containers.mome_repertoire import MOMERepertoire

from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_max_scores_evolution,
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution,
)


def run_analysis(dirname: str,
    experiment_names: List[str],
    experiment_labels: List[str],
    emitter_names: Dict,
    emitter_labels: Dict,
    plot_max_scores: bool=True,
) -> None:

    metrics_list = []
    median_metrics_list = []
    lq_metrics_list = []
    uq_metrics_list = []
    repertoires_list = []

    _analysis_dir = os.path.join(dirname, "analysis/")
    _plots_dir = os.path.join(_analysis_dir, "plots/")
    _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")
    _median_metrics_dir = os.path.join(_analysis_dir, "median_metrics/")

    os.makedirs(_analysis_dir, exist_ok=True)
    os.makedirs(_plots_dir, exist_ok=True)
    os.makedirs(_emitter_plots_dir, exist_ok=True)
    os.makedirs(_median_metrics_dir, exist_ok=True)

    for experiment_name in experiment_names:
        experiment_metrics = get_metrics(dirname, experiment_name)
        median_metrics = experiment_metrics.median()
        median_metrics.to_csv(f"{_median_metrics_dir}{experiment_name}_median_metrics")
        lq_metrics = experiment_metrics.quantile(q=0.25)
        uq_metrics =  experiment_metrics.quantile(q=0.75)

        metrics_list.append(experiment_metrics)
        median_metrics_list.append(median_metrics)
        lq_metrics_list.append(lq_metrics)
        uq_metrics_list.append(uq_metrics)

    print_min_max_rewards(metrics_list)

    """     
    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "max_sum_scores",
                    plot_ylabel = "Maximum Sum Scores",
                    plot_title = "Maximum Sum of Scores evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    ) 

    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "max_hypervolume",
                    plot_ylabel = "Maximum hypervolume",
                    plot_title = "Maximum hypervolume evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    ) 

    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "normalised_max_hypervolume",
                    plot_ylabel = "Maximum normalised hypervolume",
                    plot_title = "Maximum normalised hypervolume evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    ) 


    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "moqd_score",
                    plot_ylabel = "MOQD Score",
                    plot_title = "MOQD Score evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    )   

    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "normalised_moqd_score",
                    plot_ylabel = "Normalised MOQD Score",
                    plot_title = "Normalised MOQD Score evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    )   

    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "global_hypervolume",
                    plot_ylabel = "Global Hypervolume",
                    plot_title = "Global hypervolume evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    )   

    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "normalised_global_hypervolume",
                    plot_ylabel = "Normalised Global Hypervolume",
                    plot_title = "Normalised Global hypervolume evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir,

    )   


    plot_scores_evolution(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    metrics_label = "coverage",
                    plot_ylabel = "Coverage",
                    plot_title = "Repertoire Coverage evolution during training",
                    experiment_labels = experiment_labels, 
                    save_dir = _plots_dir
    )

    plot_emitter_counts(metrics_list,
                    emitter_names,
                    emitter_labels,
                    experiment_names,
                    experiment_labels,
                    _emitter_plots_dir
    )

    """

    return


def get_metrics(dirname: str, experiment_name: str) -> pd.DataFrame:

    #dtype_dict = {"min_scores": np.float64}

    experiment_metrics_list = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "checkpoints/metrics_history.csv"))
        experiment_metrics_list.append(metrics_df)

    experiment_metrics_concat = pd.concat(experiment_metrics_list)
    experiment_metrics = experiment_metrics_concat.groupby(experiment_metrics_concat.index)
    return experiment_metrics


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


def plot_scores_evolution(
    median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    metrics_label: str,
    plot_ylabel: str,
    plot_title: str,
    experiment_labels: List[str],
    save_dir: str,
    num_iterations: int=4000,
    episode_length: int=1000,
    batch_size: int=256,
):

    # Visualize the training evolution and final repertoire
    x_range = np.arange(num_iterations + 1) * episode_length * batch_size

    if episode_length != 1:
        x_label = "Environment steps"

    else:
        x_label = "Number of evaluations"


    fig = plt.figure()
    ax = fig.add_subplot(111)  

    for exp_num, exp_name in enumerate(experiment_labels):
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_ylabel)
        ax.set_title(plot_title)
        ax.plot(x_range, median_metrics[exp_num][metrics_label], label=exp_name)
        ax.fill_between(x_range, 
            lq_metrics[exp_num][metrics_label], 
            uq_metrics[exp_num][metrics_label], 
            alpha=0.2)
        ax.legend()

    fig.tight_layout(pad=5.0)

    plt.savefig(os.path.join(save_dir, metrics_label))
    plt.close()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w    

def plot_emitter_counts(metrics_list: List[pd.DataFrame],
    emitter_names: Dict,
    emitter_labels: Dict,
    experiment_names: List[str],
    experiment_labels: List[str],
    save_dir: str,
    rolling_window_size: int=50,
    num_iterations: int=4000,
) -> None:

    # For each experiment
    for exp_num, exp_name in enumerate(experiment_names):
        # Convert pd dfs into list of df with columns for each replication
        emitter_counts_df_list = []
        for emitter in emitter_names[exp_name]:
            emitter_count_df = metrics_list[exp_num].obj.groupby(level=0).agg(list)[emitter].apply(pd.Series)
            emitter_counts_df_list.append(emitter_count_df)

        # For each replication
        for rep, _ in emitter_counts_df_list[0].iteritems():
            emitter_counts = []

            # get moving average counts of each emitter (for this replication)
            for emitter_num in range(len(emitter_names[exp_name])):
                emitter_count = emitter_counts_df_list[emitter_num][rep]
                average_emitter_count = moving_average(emitter_count, rolling_window_size)
                emitter_counts.append(average_emitter_count)

            # Visualize the cumulative emitter counts for this experiment and replication
            x_range = num_iterations - rolling_window_size + 1
            x = np.arange(x_range + 1)
            zeros = np.zeros(x_range + 1)
            ys = np.row_stack([zeros, emitter_counts]) 
            y_stack = np.cumsum(ys, axis=0)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            
            for emitter_num in range(len(emitter_names[exp_name])):
                ax1.fill_between(x, y_stack[emitter_num, :], y_stack[emitter_num+1,:], alpha=0.7, label=emitter_labels[exp_name][emitter_num])
                ax1.legend()
            
            plt.title(f"Emitter Counts for {experiment_labels[exp_num]} Experiment")
            plt.savefig(os.path.join(save_dir, f"{exp_name}_emitter_counts_rep_{rep}"))
            plt.close()



if __name__ == '__main__':
    dirname = "results/ro_2023-01-17_100023_530cdd5bddb018153e73c6405d695030762f5475/walker2d_multi/"

    experiment_names = [
        "mome", 
        "mopga", 
        "mopga_only_forward", 
        "mopga_only_energy",
        "nsga2",
        "spea2",
        #"pga"]
    ]

    experiment_labels = [
        "MOME", 
        "MOPGA", 
        "MOPGA (Only Forward Emitter)", 
        "MOPGA (Only Energy Emitter)",
        "NSGA-II",
        "SPEA2",
        #"PGA"]
    ]


    emitter_names = {"mome": ["emitter_mutation_count:", "emitter_variation_count:"], 
            "mopga": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "mopga_only_forward": ["emitter_1_count:", "emitter_2_count:"],
            "mopga_only_energy": ["emitter_1_count:", "emitter_2_count:"],
            "nsga2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "spea2": ["emitter_mutation_count:", "emitter_variation_count:"],
            #"pga": ["emitter_1_count", "emitter_2_count"],
    }

    emitter_labels = {"mome": ["Mutation Emitter", "Variation Emitter"], 
                "mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "mopga_only_forward": ["Forward Reward Emitter", "GA Emitter"],
                "mopga_only_energy": ["Forward Reward Emitter",  "GA Emitter"],
                "nsga2": ["Mutation Emitter", "Variation Emitter"],
                "spea2": ["Mutation Emitter", "Variation Emitter"],
                "pga": ["emitter_1_count", "emitter_2_count"],
    }
    
    run_analysis(dirname, 
        experiment_names,
        experiment_labels, 
        emitter_names,
        emitter_labels)