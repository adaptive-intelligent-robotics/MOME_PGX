import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Any, Dict
from plotting.pairwise_coverage_analysis import get_global_pareto_front
from envs_setup import get_env_metrics
from functools import partial


from qdax.utils.pareto_front import compute_hypervolume

# CHANGE THESE TO ADJUST APPEARANCE OF PLOT
FIG_WIDTH = 12
FIG_HEIGHT = 10
FIGURE_DPI = 200

# ---- layout of plot ---
NUM_ROWS = 2
NUM_COLS = 2

# ---- font sizes and weights ------
BIG_GRID_FONT_SIZE  = 14 # size of labels for environment
TITLE_FONT_WEIGHT = 'bold' # Can be: ['normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
GLOBAL_TITLE_SIZE = 18 # size of big title
LEGEND_FONT_SIZE = 'x-large'

SCATTER_DOT_SIZE = 4
LEGEND_DOT_SIZE = 4

# ----- colour palettes ------
COLOUR_PALETTE = "colorblind"

# ---- spacing ----
RIGHTSPACING = 0.9   # the right side of the subplots of the figure
BOTTOMSPACING = 0.1  # the bottom of the subplots of the figure





def plot_global_pfs(parent_dirname: str,
    env_names: List[str],
    env_labels: List[str],
    experiment_names: List[str],
    experiment_labels: List[str],
    num_replications: int=10,
) -> None:

    print("Plotting Global PFs for each env, for each experiment")
    
    _global_pfs_dir = os.path.join(parent_dirname, "global_pfs/")
    _max_pfs_dir = os.path.join(parent_dirname, "max_pfs/")
    _global_max_pfs_dir = os.path.join(parent_dirname, "global_and_max_pfs/")

    os.makedirs(_global_pfs_dir, exist_ok=True)
    os.makedirs(_max_pfs_dir, exist_ok=True)
    os.makedirs(_global_max_pfs_dir, exist_ok=True)

    # Calculate coverage scores for each environment
    for replication in range(num_replications):
        print("------------------------------")
        print(f"          REP: {replication+1}             ")
        print("------------------------------")
        print("\n")
        
        replication_global_pfs = []
        replication_max_pfs = []

        for env in env_names:
            print("\n")
            print(f"          ENV: {env}             ")
            print("\n")

            env_dirname = os.path.join(parent_dirname, f"{env}/")


            # Calculate coverage scores for each experiment
            env_global_pfs = []
            env_max_pfs = []

            for experiment in experiment_names:
                replication_name = os.listdir(os.path.join(env_dirname, experiment))[replication]
                replication_dir = os.path.join(env_dirname, experiment, replication_name)
                fitnesses = jnp.load(os.path.join(replication_dir, "final/repertoire/fitnesses.npy"))
                exp_rep_global_pf = get_global_pareto_front(fitnesses)
                exp_rep_max_pf = get_max_pareto_front(fitnesses, env)
                env_global_pfs.append(exp_rep_global_pf)
                env_max_pfs.append(exp_rep_max_pf)

            replication_global_pfs.append(env_global_pfs)
            replication_max_pfs.append(env_max_pfs)
                
        plot_experiments_pfs_grid(
            env_labels,
            experiment_labels,
            replication_global_pfs,
            suptitle="Global Pareto Fronts",
            replication=replication,
            save_dir=_global_pfs_dir
        )
            
        plot_experiments_pfs_grid(
            env_labels,
            experiment_labels,
            replication_max_pfs,
            suptitle="Max Pareto Fronts",
            replication=replication,
            save_dir=_max_pfs_dir
        )

        plot_max_and_global_pfs(
            env_labels,
            experiment_labels,
            replication_global_pfs,
            replication_max_pfs,
            replication=replication,
            save_dir=_global_max_pfs_dir,        
        )

def plot_experiments_pfs_grid(
    env_labels,
    experiment_labels,
    replication_pfs,
    suptitle,
    replication,
    save_dir,
) -> None:

    num_envs = len(replication_pfs)
    num_exps = len(replication_pfs[0])

    # Create color palette
    experiment_colours = sns.color_palette(COLOUR_PALETTE, len(experiment_labels))
    colour_frame = pd.DataFrame(data={"Label": experiment_labels, "Colour": experiment_colours})

    params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': BIG_GRID_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'figure.dpi': FIGURE_DPI,
    }

    plt.rcParams.update(params)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        nrows=NUM_ROWS, 
        ncols=NUM_COLS,
    )

    for env_num, env_fig in enumerate(ax.ravel()):
        env_fig = plot_grid_square(env_fig,
            env = env_labels[env_num],
            experiment_labels = experiment_labels,
            env_pfs = replication_pfs[env_num],
            colour_frame = colour_frame,
        )
        env_fig.spines["top"].set_visible(False)
        env_fig.spines["right"].set_visible(False)

    handles, labels = ax.ravel()[-1].get_legend_handles_labels()
    
    plt.figlegend(experiment_labels, 
        loc = 'lower center',
        ncol=len(experiment_labels), 
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_DOT_SIZE,
    )

    plt.suptitle(suptitle, 
        fontsize=GLOBAL_TITLE_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
    )


    plt.subplots_adjust(
        bottom = BOTTOMSPACING,
    )   
    plt.savefig(os.path.join(save_dir, f"pfs_rep_{replication+1}"), bbox_inches='tight')
    plt.close()



def plot_max_and_global_pfs(
    env_labels,
    experiment_labels,
    replication_global_pfs,
    replication_max_pfs,
    replication,
    save_dir,
) -> None:

    num_envs = len(env_labels)

    # Create color palette
    experiment_colours = sns.color_palette(COLOUR_PALETTE, len(experiment_labels))
    colour_frame = pd.DataFrame(data={"Label": experiment_labels, "Colour": experiment_colours})

    params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': BIG_GRID_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'figure.dpi': FIGURE_DPI,
    }

    plt.rcParams.update(params)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH*num_envs/2, FIG_HEIGHT),
        nrows=2, 
        ncols=num_envs,
    )

    for row in range(2):
        for env_num, env_name in enumerate(env_labels):
            fig_num = row*num_envs + env_num
            # Plot global hypervolumes on first row
            if row == 0:
                ax.ravel()[fig_num] = plot_grid_square(ax.ravel()[fig_num],
                    env = env_name,
                    experiment_labels = experiment_labels,
                    env_pfs = replication_global_pfs[env_num],
                    colour_frame = colour_frame,
                )
                if env_num == 0:
                    ax.ravel()[fig_num].set_ylabel("Global Pareto Fronts", 
                        fontsize=BIG_GRID_FONT_SIZE,
                        fontweight=TITLE_FONT_WEIGHT
                    )    

            else:
            # Plot max hypervolumes on second row
                ax.ravel()[fig_num] = plot_grid_square(ax.ravel()[fig_num],
                    env = None, # dont use title on second row
                    experiment_labels = experiment_labels,
                    env_pfs = replication_max_pfs[env_num],
                    colour_frame = colour_frame,
                )       


                if env_num == 0:
                    ax.ravel()[fig_num].set_ylabel("Max Pareto Fronts", 
                        fontsize=BIG_GRID_FONT_SIZE,
                        fontweight=TITLE_FONT_WEIGHT
                    )          


    handles, labels = ax.ravel()[-1].get_legend_handles_labels()
    
    plt.figlegend(experiment_labels, 
        loc = 'lower center',
        ncol=len(env_labels), 
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_DOT_SIZE,
    )

    plt.subplots_adjust(
        right = RIGHTSPACING,    
    )    

    plt.savefig(os.path.join(save_dir, f"pfs_rep_{replication+1}"), bbox_inches='tight')
    plt.close()


def plot_grid_square(
    env_ax: plt.Axes,
    env: str,
    env_pfs: List[jnp.array],
    experiment_labels: List[str],
    colour_frame: pd.DataFrame,
):
    """
    Plots one subplot of grid
    """

    # Getting the correct color palette
    exp_palette = colour_frame["Colour"].values
    sns.set_palette(exp_palette)

    for exp_num, exp_name in enumerate(experiment_labels):
        env_ax.scatter(
            env_pfs[exp_num][:,0], # first fitnesses
            env_pfs[exp_num][:,1], # second fitnesses
            label=exp_name,
            s=SCATTER_DOT_SIZE,
            c=exp_palette[exp_num]
        )
        if env:
            env_ax.set_title(env)

    return env_ax


def get_max_pareto_front(fitnesses,
    env_name,
):

    num_objectives = fitnesses.shape[0]
    # get env reference point
    env_metrics = get_env_metrics(env_name)
    reference_point = env_metrics["min_rewards"]

    # recompute hypervolumes
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    hypervolumes = jax.vmap(hypervolume_function)(fitnesses)  # num centroids

    # mask empty hypervolumes
    repertoire_empty = fitnesses == -jnp.inf # num centroids x pareto-front length x num criteria
    repertoire_not_empty = jnp.any(~jnp.all(repertoire_empty, axis=-1), axis=-1) # num centroids x pareto-front length
    hypervolumes = jnp.where(repertoire_not_empty, hypervolumes, -jnp.inf) # num_centroids

    # find max hypervolume
    max_hypervolume = jnp.max(hypervolumes)

    # get mask for centroid with max hypervolume
    max_hypervolume_mask = hypervolumes == max_hypervolume

    # get cell fitnesses with max hypervolume
    max_cell_fitnesses = jnp.take(fitnesses, jnp.argwhere(max_hypervolume_mask), axis=0).squeeze(axis=(0,1))

    # create mask for -inf fitnesses
    non_empty_indices = jnp.argwhere(jnp.all(max_cell_fitnesses != -jnp.inf, axis=-1))
    pareto_front = jnp.take(max_cell_fitnesses, non_empty_indices, axis=0).squeeze()
    
    return pareto_front