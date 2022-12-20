import matplotlib.pyplot as plt
import os
from typing import  Dict
from qdax.core.mome import MOMERepertoire
from qdax.types import Fitness, Descriptor, RNGKey, Genotype, Centroid
from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_max_scores_evolution,
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution,
)


class Plotter:

    def __init__(self,
        minval: float,
        maxval: float,
        pareto_front_max_length: int,
        batch_size: int,
        num_iterations: int,
        episode_length: int,
    ):
        self.minval=minval
        self.maxval=maxval
        self.pareto_front_max_length=pareto_front_max_length
        self.batch_size=batch_size
        self.num_iterations=num_iterations
        self.episode_length=episode_length
    
    def plot_num_solutions(
        self,
        centroids: Centroid,
        metrics: Dict,
        save_dir: str="./",
        save_name: str="",
    ) -> None:
            
        fig = plt.figure()
        axes = fig.add_subplot(111) 

        # add map elites plot on last axes
        fig, axes = plot_2d_map_elites_repertoire(
            centroids=centroids,
            repertoire_fitnesses=metrics["num_solutions"][-1],
            minval=self.minval,
            maxval=self.maxval,
            vmin=0,
            vmax=self.pareto_front_max_length,
            ax=axes
        )

        plt.savefig(os.path.join(save_dir, f"num_solutions_{save_name}"))
        plt.close()

    def plot_repertoire(
        self,
        repertoire: MOMERepertoire,
        centroids: Centroid,
        metrics: Dict,
        save_dir: str="./",
        save_name: str="",
    ):

        fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

        # plot pareto fronts
        axes = plot_mome_pareto_fronts(
            centroids,
            repertoire,
            minval=self.minval,
            maxval=self.maxval,
            color_style='spectral',
            axes=axes,
            with_global=True
        )

        # add map elites plot on last axes
        fig, axes = plot_2d_map_elites_repertoire(
            centroids=centroids,
            repertoire_fitnesses=metrics["hypervolumes"][-1],
            minval=self.minval,
            maxval=self.maxval,
            ax=axes[2]
        )

        plt.savefig(os.path.join(save_dir, f"repertoire_{save_name}"))
        plt.close()

        
    def plot_scores_evolution(
        self,
        metrics_history: Dict,
        save_dir: str="./",
    ) -> None:
        
        fig, axes = plt.subplots(figsize=(18, 6), ncols=2)

        axes = plot_mome_scores_evolution(
            metrics_history=metrics_history,
            ax=axes,
            fig=fig,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            episode_length=self.episode_length,
        )

        plt.savefig(os.path.join(save_dir, "scores_evolution"))
        plt.close()

    def plot_max_scores_evolution(
        self,
        metrics_history: Dict,
        save_dir: str="./",
    ) -> None:

        fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

        axes = plot_mome_max_scores_evolution(
            metrics_history=metrics_history,
            ax=axes,
            fig=fig,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            episode_length=self.episode_length,
        )

        plt.savefig(os.path.join(save_dir, f"max_scores_evolution"))
        plt.close()