import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Any, Dict


# CHANGE THESE TO ADJUST APPEARANCE OF PLOT
FIG_WIDTH = 20
FIG_HEIGHT = 10
FIGURE_DPI = 200

# ---- font sizes and weights ------
BIG_GRID_FONT_SIZE  = 14
SMALL_GRID_FONT_SIZE = 14
TITLE_FONT_WEIGHT = 'bold' # Can be: ['normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
LEGEND_FONT_SIZE = 'x-large'

# ----- colour palettes ------
COLOUR_PALETTE = "colorblind"

# ----  spacing -----
LEFTSPACING = 0.13   # the left side of the subplots of the figure
RIGHTSPACING = 0.9   # the right side of the subplots of the figure
BOTTOMSPACING = 0.1  # the bottom of the subplots of the figure
TOPSPACING = 0.87   # the top of the subplots of the figure
WIDTHSPACING = 0.1  # the proportion of width reserved for blank space between subplots
HEIGHTSPACING = 0.1  # the proportion of height reserved for blank space between subplots



def customize_axis(ax: Any) -> Any:
    """
    Customise axis for plots.
    This do appearance changes to make the plot more
    beautiful (removes spine and simplify axis).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.get_xaxis().tick_bottom() # move ticks and labels to bottom of axis
    ax.tick_params(axis="y", length=0) # set y tick length to zero

    # remove x and y labels and ticks
    ax.tick_params(labelbottom = False, bottom = False)
    ax.tick_params(labelleft = False, left = False)


    # offset the spines
    #for spine in ax.spines.values():
    #    spine.set_position(("outward", 5))

    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.85", linestyle="-", linewidth=1.2)
    return ax


def plot_experiments_grid(parent_dirname: str,
    env_names: List[str],
    env_labels: Dict,
    experiment_names: List[str],
    experiment_labels: Dict,
    grid_plot_metrics_list: List[str],
    grid_plot_metrics_labels: Dict,
    grid_plot_linestyles: Dict,
    medians: List[pd.DataFrame],
    lqs: List[pd.DataFrame], 
    uqs:  List[pd.DataFrame],
    num_iterations: int,
    episode_length: int,
    batch_size: int,
    x_axis_evaluations: bool=True
 ) -> None:

    print("\n")
    print("-------------------------------------------------------------------------")
    print("                   Plotting grid plot of experiments                     ")
    print("-------------------------------------------------------------------------")

    _analysis_dir = os.path.join(parent_dirname, "analysis/")
    os.makedirs(_analysis_dir, exist_ok=True)

    num_rows = len(grid_plot_metrics_list)
    num_cols = len(env_names)

    # Create color palette
    experiment_colours = sns.color_palette(COLOUR_PALETTE, len(experiment_names))
    colour_frame = pd.DataFrame(data={"Label": experiment_names, "Colour": experiment_colours})

    params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': BIG_GRID_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'figure.dpi': FIGURE_DPI,
    }

    plt.rcParams.update(params)
    
    if x_axis_evaluations:
        x_range = np.arange(num_iterations + 1)  * batch_size
        x_label = "Number of evaluations"

    else:
        x_range = np.arange(num_iterations + 1) * episode_length * batch_size
        x_label = "Environment steps"

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        nrows=num_rows, 
        ncols=num_cols,
    )

    for row, metric in enumerate(grid_plot_metrics_list):
        for col, env in enumerate(env_names):
            fig_num = row*num_cols + col
            ax.ravel()[fig_num] = plot_grid_square(ax.ravel()[fig_num],
                median_metrics = medians[col],
                lq_metrics = lqs[col],
                uq_metrics = uqs[col],
                x_range = x_range,
                metrics_label=metric,
                experiment_names=experiment_names,
                experiment_labels = experiment_labels,
                experiment_linestyles = grid_plot_linestyles,
                colour_frame = colour_frame,
            )

            if row == 0:
                ax.ravel()[fig_num].set_title(env_labels[env])
            
            if row + 1 == num_rows:
                ax.ravel()[fig_num].set_xlabel(x_label, fontsize=SMALL_GRID_FONT_SIZE)
                ax.ravel()[fig_num].spines["bottom"].set_visible(True)
                ax.ravel()[fig_num].tick_params(labelbottom = True, bottom = True)


            if col == 0:
                ax.ravel()[fig_num].set_ylabel(f"{grid_plot_metrics_labels[metric]}"+ " (%)", 
                    fontsize=BIG_GRID_FONT_SIZE,
                    fontweight=TITLE_FONT_WEIGHT
                )
                #ax.ravel()[fig_num].spines["left"].set_visible(True)
                ax.ravel()[fig_num].tick_params(labelleft = True, left = True)

    handles, labels = ax.ravel()[-1].get_legend_handles_labels()

    fig.align_ylabels()
    
    plt.figlegend(experiment_labels.values(), 
        loc = 'lower center',
        ncol=int(len(experiment_labels.values())), 
        fontsize=LEGEND_FONT_SIZE,
    )
    
    plt.subplots_adjust(
        left  = LEFTSPACING,    
        right = RIGHTSPACING,    
        bottom = BOTTOMSPACING,
        top = TOPSPACING,      
        wspace = WIDTHSPACING,  
        hspace = HEIGHTSPACING,  
    )

    plt.savefig(os.path.join(_analysis_dir, f"grid_plot"), bbox_inches='tight')
    plt.close()



def plot_grid_square(
    ax: plt.Axes,
    median_metrics: List[pd.DataFrame],
    lq_metrics: List[pd.DataFrame],
    uq_metrics: List[pd.DataFrame],
    x_range: int,
    metrics_label: str,
    experiment_names: List[str],
    experiment_labels: Dict,
    experiment_linestyles: Dict,
    colour_frame: pd.DataFrame,
):
    """
    Plots one subplot of grid, normalising scores
    """


    # Find the maximum uq of all experiments in order to scale metrics
    y_max = 0
    for exp_num, exp_name in enumerate(experiment_labels):
        final_score = np.array(uq_metrics[exp_num][metrics_label])[-1]
        if final_score > y_max:
            y_max = final_score

    # Getting the correct color palette
    exp_palette = colour_frame["Colour"].values
    sns.set_palette(exp_palette)

    for exp_num, exp_name in enumerate(experiment_labels):
        medians = np.array(median_metrics[exp_num][metrics_label])
        if metrics_label == "coverage":
            scaled_medians = medians
        else:
            scaled_medians = medians*100/y_max
        ax.plot(x_range, 
            scaled_medians,
            label=exp_name,
            linestyle=experiment_linestyles[experiment_names[exp_num]],
            color=exp_palette[exp_num]
        )
        # set all scales to be same
        ax.set_ylim([0, 101])

    # Do this second so that legend labels are correct
    for exp_num, exp_name in enumerate(experiment_labels):
        lqs = np.array(lq_metrics[exp_num][metrics_label])
        uqs = np.array(uq_metrics[exp_num][metrics_label])

        if metrics_label == "coverage":
            scaled_lqs = lqs
            scaled_uqs = uqs
        else:
            scaled_lqs = lqs*100/y_max
            scaled_uqs = uqs*100/y_max 

        ax.fill_between(x_range, 
            scaled_lqs,
            scaled_uqs,
            alpha=0.2,
            color=exp_palette[exp_num]
        )

    customize_axis(ax)

    return ax


