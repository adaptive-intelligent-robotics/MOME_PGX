import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict


def plot_env_emitter_counts(parent_dirname: str,
    env_names: List[str],
    experiment_names: List[str],
    experiment_labels: List[str],
    emitter_names: Dict,
    emitter_labels: Dict,
    metrics_list: List[pd.DataFrame],
    num_iterations: int,
)-> None:

    print("\n")
    print("-------------------------------------------------------------------------")
    print("     Plotting emitter counts for each experiment in each enivronment     ")
    print("-------------------------------------------------------------------------")

    for env_num, env in enumerate(env_names):

        print("\n")
        print(f"     ENV: {env}             ")

        dirname = os.path.join(parent_dirname, env)

        _analysis_dir = os.path.join(dirname, "analysis/")
        _plots_dir = os.path.join(_analysis_dir, "plots/")
        _emitter_plots_dir = os.path.join(_plots_dir, "emitters/")


        plot_emitter_counts(metrics_list[env_num],
                            emitter_names,
                            emitter_labels,
                            experiment_names,
                            experiment_labels,
                            num_iterations,
                            _emitter_plots_dir)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w    


def plot_emitter_counts(metrics_list: List[pd.DataFrame],
    emitter_names: Dict,
    emitter_labels: Dict,
    experiment_names: List[str],
    experiment_labels: Dict,
    num_iterations: int,
    save_dir: str,
    rolling_window_size: int=50,
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
            
            plt.title(f"Emitter Counts for {experiment_labels[exp_name]} Experiment")
            plt.savefig(os.path.join(save_dir, f"{exp_name}_emitter_counts_rep_{rep}"))
            plt.close()
