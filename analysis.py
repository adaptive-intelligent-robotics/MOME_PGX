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



    plot_max_scores_evolution(median_metrics_list, 
                            lq_metrics_list,
                            uq_metrics_list,
                            experiment_labels, 
                            _plots_dir
    )
     
    plot_hypervolumes(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    experiment_labels, 
                    _plots_dir
    )   

    plot_emitter_counts(metrics_list,
                    emitter_names,
                    emitter_labels,
                    experiment_names,
                    experiment_labels,
                    _emitter_plots_dir
    )

    plot_coverage_scores(median_metrics_list, 
                    lq_metrics_list,
                    uq_metrics_list,
                    experiment_labels, 
                    _plots_dir
    )

    return


def get_metrics(dirname: str, experiment_name: str) -> pd.DataFrame:
    experiment_metrics_list = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        metrics_df = pd.read_csv(os.path.join(experiment_replication, "checkpoints/metrics_history.csv"))
        experiment_metrics_list.append(metrics_df)

    experiment_metrics_concat = pd.concat(experiment_metrics_list)
    experiment_metrics = experiment_metrics_concat.groupby(experiment_metrics_concat.index)
    return experiment_metrics


def plot_hypervolumes(median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    experiment_labels: List[str],
    save_dir: str,
) -> None:

    num_iterations = 4000
    episode_length = 1000
    batch_size = 256

    # Visualize the training evolution and final repertoire
    x_range = np.arange(num_iterations + 1) * episode_length * batch_size

    if episode_length != 1:
        x_label = "Environment steps"

    else:
        x_label = "Number of evaluations"


    fig, ax = plt.subplots(figsize=(18, 6), ncols=2)

    for exp_num, exp_name in enumerate(experiment_labels):
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel("MOQD Score")
        ax[0].set_title("MOQD Score evolution during training")
        ax[0].set_aspect(0.95 / ax[0].get_data_ratio(), adjustable="box")
        ax[0].plot(x_range, median_metrics[exp_num]["moqd_score"], label=exp_name)
        ax[0].fill_between(x_range, 
            lq_metrics[exp_num]["moqd_score"], 
            uq_metrics[exp_num]["moqd_score"], 
            alpha=0.2)
        ax[0].legend()


        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel("Global hypervolume")
        ax[1].set_title("Global hypervolume evolution during training")
        ax[1].set_aspect(0.95 / ax[1].get_data_ratio(), adjustable="box")
        ax[1].plot(x_range, median_metrics[exp_num]["global_hypervolume"], label=exp_name)
        ax[1].fill_between(x_range, 
            lq_metrics[exp_num]["global_hypervolume"], 
            uq_metrics[exp_num]["global_hypervolume"], 
            alpha=0.2)
        ax[1].legend()


    fig.tight_layout(pad=5.0)

    plt.savefig(os.path.join(save_dir, f"hypervolume_scores_evolution"))
    plt.close()


def plot_max_scores_evolution(median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    experiment_labels: List[str],
    save_dir: str,
) -> None:

    num_iterations = 4000
    episode_length = 1000
    batch_size = 256

    # Visualize the training evolution and final repertoire
    x_range = np.arange(num_iterations + 1) * episode_length * batch_size

    if episode_length != 1:
        x_label = "Environment steps"

    else:
        x_label = "Number of evaluations"


    fig, ax = plt.subplots(figsize=(18, 6), ncols=2)

    for exp_num, exp_name in enumerate(experiment_labels):
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel("Maximum hypervolume")
        ax[0].set_title("Maximum hypervolume evolution during training")
        ax[0].set_aspect(0.95 / ax[0].get_data_ratio(), adjustable="box")
        ax[0].plot(x_range, median_metrics[exp_num]["max_hypervolume"], label=exp_name)
        ax[0].fill_between(x_range, 
            lq_metrics[exp_num]["max_hypervolume"], 
            uq_metrics[exp_num]["max_hypervolume"], 
            alpha=0.2)
        ax[0].legend()


        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel("Max Sum Scores")
        ax[1].set_title("Max Sum Score evolution during training")
        ax[1].set_aspect(0.95 / ax[1].get_data_ratio(), adjustable="box")   
        ax[1].plot(x_range, median_metrics[exp_num]["max_sum_scores"], label=exp_name)
        ax[1].fill_between(x_range, 
            lq_metrics[exp_num]["max_sum_scores"], 
            uq_metrics[exp_num]["max_sum_scores"], 
            alpha=0.2)
        ax[1].legend()

    fig.tight_layout(pad=5.0)

    plt.savefig(os.path.join(save_dir, f"max_scores_evolution"))
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


def plot_coverage_scores(median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    experiment_labels: List[str],
    save_dir: str,
) -> None:

    num_iterations = 4000
    episode_length = 1000
    batch_size = 256

    # Visualize the training evolution and final repertoire
    x_range = np.arange(num_iterations + 1) * episode_length * batch_size

    if episode_length != 1:
        x_label = "Environment steps"

    else:
        x_label = "Number of evaluations"


    fig = plt.figure()
    ax1 = fig.add_subplot(111)  

    for exp_num, exp_name in enumerate(experiment_labels):
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Coverage")
        ax1.set_title("Coverage evolution during training")
        ax1.plot(x_range, median_metrics[exp_num]["coverage"], label=exp_name)
        ax1.fill_between(x_range, 
            lq_metrics[exp_num]["coverage"], 
            uq_metrics[exp_num]["coverage"], 
            alpha=0.2)
        ax1.legend()

    plt.title("Coverage Scores")
    plt.savefig(os.path.join(save_dir, "coverage_scores"))
    plt.close()

"""
def load_repertoire(dirname:str,
    experiment_name: List[str],
) -> List[MOMERepertoire]:

    pareto_front_max_length = 50

    repertoires = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        centroids = np.load(os.path.join(experiment_replication, "final/repertoire/centroids.npy"))
        descriptors = np.load(os.path.join(experiment_replication, "final/repertoire/descriptors.npy"))
        fitnesses = np.load(os.path.join(experiment_replication, "final/repertoire/fitnesses.npy"))
        genotypes = np.load(os.path.join(experiment_replication, "final/repertoire/genotypes.npy"))

        repertoire = MOMERepertoire.init(
            genotypes,
            fitnesses,
            descriptors,
            centroids,
            pareto_front_max_length)

        repertoires.append(repertoire)

    return repertoires   

def plot_repertoires(repertoires_list: List[MOMERepertoire],
    metrics_list: List[pd.DataFrame],
    save_dir: str,
    experiment_names: List[str],
) -> None:

    minval = 0.
    maxval = 1.

    fig, axes = plt.subplots(figsize=(18, 6), nrows=len(experiment_names), ncols=3)


    for exp_num, exp_name in enumerate(experiment_names):

        for replication_num, repertoire in enumerate(repertoires_list[exp_num]):
        
            centroids = repertoire.centroids
            metrics = metrics_list[exp_num][replication_num]

            # plot pareto fronts
            axes = plot_mome_pareto_fronts(
                centroids,
                repertoire,
                minval=minval,
                maxval=maxval,
                color_style='spectral',
                axes=axes[exp_num],
                with_global=True
            )

            # add map elites plot on last axes
            fig, axes = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=metrics["hypervolumes"][-1],
                minval=minval,
                maxval=maxval,
                ax=axes[exp_num][2]
            )

        plt.savefig(os.path.join(save_dir, f"repertoires_replication_{replication_num}"))
        plt.close()


def get_configs(output_path) -> List[Dict]:
    experiment_configs_list = []

    for experiment_replication in os.scandir(dirname):
        with open(os.path.join(experiment_replication, ".hydra", "config.yaml")) as config_file:
            configs = yaml.load(config_file, Loader=yaml.loader.SafeLoader)

    return experiment_configs_list
"""


if __name__ == '__main__':
    dirname = "results/ro_2023-01-05_093551_31c3158b10ec634f8babbaf1211654f34af4eae1/halfcheetah_multi/"
    experiment_names = ["brax_mome", "brax_mopga_normal", "brax_mopga_only_forward", "brax_mopga_only_energy"]
    experiment_labels = ["MOME", "MOPGA", "MOPGA (Only Forward Emitter)", "MOPGA (Only Energy Emitter)"]

    emitter_names = {"brax_mome": ["emitter_mutation_count:", "emitter_variation_count:"], 
            "brax_mopga_normal": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "brax_mopga_only_forward": ["emitter_1_count:", "emitter_2_count:"],
            "brax_mopga_only_energy": ["emitter_1_count:", "emitter_2_count:"]
    }

    emitter_labels = {"brax_mome": ["Mutation Emitter", "Variation Emitter"], 
                "brax_mopga_normal": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "brax_mopga_only_forward": ["Forward Reward Emitter", "GA Emitter"],
                "brax_mopga_only_energy": ["Forward Reward Emitter",  "GA Emitter"],
    }
    
    run_analysis(dirname, 
        experiment_names,
        experiment_labels, 
        emitter_names,
        emitter_labels)