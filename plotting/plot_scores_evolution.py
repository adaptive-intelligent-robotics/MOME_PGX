import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set_palette("muted")

from typing import List

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