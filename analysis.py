import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

sns.set_palette("muted")

from typing import List, Dict, Tuple
from qdax.core.containers.mome_repertoire import MOMERepertoire

from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_max_scores_evolution,
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution,
)


def run_analysis(parent_dirname: str,
    env_names: List[str],
    env_labels: List[str],
    experiment_names: List[str],
    experiment_labels: List[str],
    emitter_names: Dict,
    emitter_labels: Dict,
    grid_plot_metrics_list: List[str],
    grid_plot_metrics_labels: List[str],
    plot_individual_envs: bool=True,
    plot_experiments_grid: bool=True,
) -> None:

    all_metrics, all_medians, all_lqs, all_uqs = calculate_quartile_metrics(parent_dirname,
        env_names,
        experiment_names,
    )

    if plot_individual_envs:
        plot_individual_envs(parent_dirname,
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

    if plot_experiments_grid:
        plot_experiments_grid(parent_dirname,
            env_names,
            env_labels,
            experiment_names,
            experiment_labels,
            grid_plot_metrics_list,
            grid_plot_metrics_labels,
            all_medians, 
            all_lqs, 
            all_uqs
        )


def calculate_quartile_metrics(parent_dirname: str,
    env_names: List[str],
    experiment_names: List[str],
)-> Tuple[List, List, List, List]:

    print("--------- Calculating Quartile Metrics -----------")

    all_metrics = []
    all_medians = []
    all_lqs = []
    all_uqs = []

    for env in env_names:

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
            median_metrics = experiment_metrics.median()
            median_metrics.to_csv(f"{_median_metrics_dir}{experiment_name}_median_metrics")
            lq_metrics = experiment_metrics.quantile(q=0.25)
            uq_metrics =  experiment_metrics.quantile(q=0.75)

            metrics_list.append(experiment_metrics)
            median_metrics_list.append(median_metrics)
            lq_metrics_list.append(lq_metrics)
            uq_metrics_list.append(uq_metrics)

        all_metrics.append(metrics_list)
        all_medians.append(median_metrics_list)
        all_lqs.append(lq_metrics_list)
        all_uqs.append(uq_metrics_list)

    return all_metrics, all_medians, all_lqs, all_uqs


def plot_individual_envs(parent_dirname: str,
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

        """
        plot_emitter_counts(metrics_list[env_num],
                        emitter_names,
                        emitter_labels,
                        experiment_names,
                        experiment_labels,
                        _emitter_plots_dir
        )
        """



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

    print(f"------ Plotting {plot_ylabel} ------")

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
    
    for exp_num, exp_name in enumerate(experiment_labels):
        ax.fill_between(x_range, 
            lq_metrics[exp_num][metrics_label], 
            uq_metrics[exp_num][metrics_label], 
            alpha=0.2)

    plt.figlegend(loc = 'lower center', labels=experiment_labels)

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
    print("------ Plotting Emitter Counts ------")

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


def plot_experiments_grid(parent_dirname,
    env_names,
    env_labels,
    experiment_names,
    experiment_labels,
    grid_plot_metrics_list,
    grid_plot_metrics_labels,
    medians,
    lqs, 
    uqs,
 ) -> None:

    num_rows = len(grid_plot_metrics_list)
    num_cols = len(env_names)

    """
    GRID_AXES_FONT_SIZE = 30
    GRID_AXES_PAD = 0.5
    SUBPLOT_TICK_SIZE = 20

    
    params = {
        'figure.dpi': 200,
        'axes.labelpad': 0.8*num_rows,
        # title params
        'axes.titlesize': GRID_AXES_FONT_SIZE,
        'axes.titlepad': GRID_AXES_PAD,
        #'axes.linewidth': 0.1*num_rows,
        'xtick.labelsize': SUBPLOT_TICK_SIZE,
        'ytick.labelsize': SUBPLOT_TICK_SIZE,
        #'axes.xmargin': 0.1,
        #'axes.ymargin': 0.1,
        'font.family': 'serif',
    }


    plt.rcParams.update(params)
    """

    fig, ax = plt.subplots(
        figsize=(20, 20),
        nrows=num_rows, 
        ncols=num_cols,
        #constrained_layout=True
    )

    for row, metric in enumerate(grid_plot_metrics_list):
        for col, env in enumerate(env_names):
            fig_num = row*num_cols + col
            ax.ravel()[fig_num] = plot_grid_square(ax.ravel()[fig_num],
                median_metrics = medians[col],
                lq_metrics = lqs[col],
                uq_metrics = uqs[col],
                metrics_label=metric,
                experiment_labels = experiment_labels,
            )

            if row == 0:
                ax.ravel()[fig_num].set_title(env_labels[col])
            
            if col == 0:
                ax.ravel()[fig_num].set_ylabel(grid_plot_metrics_labels[row], 
                    #fontsize=GRID_AXES_FONT_SIZE
                )
    
    handles, labels = ax.ravel()[-1].get_legend_handles_labels()
    

    plt.figlegend(experiment_labels, loc = 'lower center', ncol=len(experiment_labels))
    
    plt.subplots_adjust(
        left  = 0.1,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.1,   # the bottom of the subplots of the figure
        top = 0.9,      # the top of the subplots of the figure
        wspace = 0.2,  # the amount of width reserved for blank space between subplots
        hspace = 0.15, 
    )


    
    plt.savefig(os.path.join(parent_dirname, f"grid_plot"), dpi=200, bbox_inches='tight')
    plt.close()

def plot_grid_square(
    ax: plt.Axes,
    median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    metrics_label: str,
    experiment_labels: List[str],
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

    # Do this second so that legend labels are correct
    for exp_num, exp_name in enumerate(experiment_labels):
        ax.plot(x_range, median_metrics[exp_num][metrics_label], label=exp_name)
        ax.set_xlabel(x_label)

    for exp_num, exp_name in enumerate(experiment_labels):
        ax.fill_between(x_range, 
            lq_metrics[exp_num][metrics_label], 
            uq_metrics[exp_num][metrics_label], 
            alpha=0.2)

    return ax

if __name__ == '__main__':
    parent_dirname = "results/paper_results/"

    """
    experiment_names = [
        "bandit_mopga",
        "mopga", 
        "mopga_only_forward", 
        "mopga_only_energy",
        "mome", 
        "nsga2",
        "spea2",
        "pga",
    ]

    experiment_labels = [
        "Bandit MOPGA",
        "MOPGA", 
        "MOPGA (Only Forward Emitter)", 
        "MOPGA (Only Energy Emitter)",
        "MOME", 
        "NSGA-II",
        "SPEA2",
        "PGA"
    ]
    """
    """
    experiment_names = [
        "bandit_mopga",
        "bandit_mopga_0.5",  
        "bandit_mopga_2",
        "bandit_mopga_5",
        "bandit_mopga_10",
        "mopga_only_forward",
        "pga",
    ]

    experiment_labels = [
        "bandit_mopga",
        "bandit_mopga_0.5",  
        "bandit_mopga_2",
        "bandit_mopga_5",
        "bandit_mopga_10",
        "mopga_only_forward",
        "pga",
    ]
    """

    experiment_names = [
        "mopga", 
        "mome", 
        "pga",
    ]

    experiment_labels = [
        "MOPGA", 
        "MOME", 
        "PGA"
    ]


    
    emitter_names = {"mome": ["emitter_mutation_count:", "emitter_variation_count:"], 
            "mopga": ["emitter_1_count:", "emitter_2_count:", "emitter_3_count:"],
            "mopga_only_forward": ["emitter_1_count:", "emitter_2_count:"],
            "mopga_only_energy": ["emitter_1_count:", "emitter_2_count:"],
            "nsga2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "spea2": ["emitter_mutation_count:", "emitter_variation_count:"],
            "pga": ["emitter_1_count:", "emitter_2_count:"],
            "bandit_mopga": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
            "bandit_mopga_0.5": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
            "bandit_mopga_2": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
            "bandit_mopga_5": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
            "bandit_mopga_10": ["emitter_1_added_count:", "emitter_2_added_count:", "emitter_3_added_count:"],
    }

    emitter_labels = {"mome": ["Mutation Emitter", "Variation Emitter"], 
                "mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "mopga_only_forward": ["Forward Reward Emitter", "GA Emitter"],
                "mopga_only_energy": ["Forward Reward Emitter",  "GA Emitter"],
                "nsga2": ["Mutation Emitter", "Variation Emitter"],
                "spea2": ["Mutation Emitter", "Variation Emitter"],
                "pga": ["Gradient Emitter", "GA Emitter"],
                "bandit_mopga": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "bandit_mopga_0.5": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "bandit_mopga_2": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "bandit_mopga_5": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],
                "bandit_mopga_10": ["Forward Reward Emitter", "Energy Cost Emitter", "GA Emitter"],

    }


    env_names=["ant_multi",
        "halfcheetah_multi",
        "hopper_multi",
        "walker2d_multi",
        
    ]

    env_labels=["Ant",
        "HalfCheetah",
        "Hopper",
        "Walker2d",
    ]

    """
    env_names=["walker2d_multi"]
    env_labels=["Walker2d"]
    """  
    
    grid_plot_metrics_list = ["moqd_score", "global_hypervolume", "max_sum_scores", "coverage"]
    grid_plot_metrics_labels = ["MOQD Score", "Global Hypervolume", "Max Sum Scores", "Coverage"]

    run_analysis(parent_dirname, 
        env_names,
        env_labels,
        experiment_names,
        experiment_labels, 
        emitter_names,
        emitter_labels,
        grid_plot_metrics_list,
        grid_plot_metrics_labels)