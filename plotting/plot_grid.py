import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_palette("muted")

from typing import List


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