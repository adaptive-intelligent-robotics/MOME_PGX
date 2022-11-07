import chex
import flax
import hydra
import jax.numpy as jnp
import jax
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from dataclasses import dataclass
from functools import partial
from hydra.core.config_store import ConfigStore
from typing import Callable, Dict, Optional, Tuple
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.emitters.mutation_operators import (
    polynomial_mutation, 
    polynomial_crossover, 
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mome import MOME, MOMERepertoire
from qdax.types import Fitness, Descriptor, RNGKey, ExtraScores
from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_max_scores_evolution,
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution,
)
from qdax.utils.metrics import CSVLogger, default_moqd_metrics


class RunMOME:

    """
    Args:
        pareto_front_max_length: max number of solutions per cell/in each pareto front
        num_param_dimensions: dimensionality of solution genotype
        num_descriptor_dimensions: dimensionality of solution BD
        minval: minimum value of BD
        maxval: maximum value of BD
        num_iterations: number of MAP-Elites iterations to run
        num_centroids: number of cells in grid
        num_init_cvt_samples: number of sampled points to be used for clustering to determine centroids 
        init_batch_size: initial population size
        batch_size: number of solutions to select and mutate in each MAP-Elites iteration
        scoring_fn: function that returns fitness and descriptor of a solution
        emitter: function determining variation mechanism of solutions
        metrics_fn: function for calculating moqd metrics 
    """

    def __init__(self,
                 pareto_front_max_length: int,
                 num_param_dimensions: int,
                 num_descriptor_dimensions: int,
                 minval: int,
                 maxval: int,
                 num_iterations: int, 
                 num_centroids: int,
                 num_init_cvt_samples: int,
                 init_batch_size: int,
                 batch_size: int, 
                 scoring_fn: Callable,
                 emitter: Emitter,
                 metrics_fn: Callable,
                 metrics_log_period: int
                 ):

        self.pareto_front_max_length = pareto_front_max_length
        self.num_param_dimensions = num_param_dimensions
        self.num_descriptor_dimensions = num_descriptor_dimensions
        self.minval = minval
        self.maxval = maxval
        self.num_iterations =  num_iterations
        self.num_centroids = num_centroids
        self.num_init_cvt_samples = num_init_cvt_samples
        self.init_batch_size = init_batch_size
        self.batch_size =  batch_size
        self.scoring_fn = scoring_fn
        self.emitter = emitter
        self.metrics_fn = metrics_fn
        self.metrics_log_period = metrics_log_period



    def run(self,
            random_key: RNGKey,
            ) -> Tuple[MOMERepertoire, jnp.ndarray, RNGKey]:
            
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().handlers[0].setLevel(logging.INFO)
        logger = logging.getLogger(f"{__name__}")
        output_dir = "./" #get_output_dir()

        # initial population
        random_key, subkey = jax.random.split(random_key)

        init_genotypes = jax.random.uniform(
            subkey, 
            (self.init_batch_size, self.num_param_dimensions), 
            minval=self.minval, 
            maxval=self.maxval, 
            dtype=jnp.float32
        )

        # Compute the centroids
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=self.num_descriptor_dimensions, 
            num_init_cvt_samples=self.num_init_cvt_samples, 
            num_centroids=self.num_centroids, 
            minval=self.minval, 
            maxval=self.maxval,
            random_key=random_key,
        )


        # Instantiate MOME
        mome = MOME(
            scoring_function=self.scoring_fn,
            emitter=self.emitter,
            metrics_function=self.metrics_fn,
        )

        # Initializes repertoire and emitter state
        repertoire, emitter_state, random_key = mome.init(
            init_genotypes,
            centroids,
            self.pareto_front_max_length,
            random_key
        )


        # Set up logging functions
        num_loops = self.num_iterations // self.metrics_log_period

        csv_logger = CSVLogger(
            output_dir + "mome-logs.csv",
            
            header = [
                "loop", 
                "iteration",
                "moqd_score", 
                "max_scores",  
                "max_sum_scores", 
                "max_hypervolume", 
                "global_hypervolume", 
                "coverage", 
                "number_solutions", 
                "time"
            ]
        )


        metrics_history = {}

        # Run the algorithm
        for i in range(num_loops):

            start_time = time.time()

            (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
                mome.scan_update,
                (repertoire, emitter_state, random_key),
                (),
                length=self.metrics_log_period,
            )

            timelapse = time.time() - start_time

            logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": 1 + i*self.metrics_log_period}

            for key, value in metrics.items():

                # take last value
                logged_metrics[key] = value[-1]

                # take all values
                if key in metrics_history.keys():
                    metrics_history[key] = jnp.concatenate([metrics_history[key], value])
                else:
                    metrics_history[key] = value

            csv_logger.log(logged_metrics)

        return repertoire, centroids, random_key, metrics, metrics_history


    def plot_final_repertoire(
        self,
        repertoire: MOMERepertoire,
        centroids: jnp.ndarray,
        metrics: Dict,
    ) -> None:
        
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
            repertoire_fitnesses=metrics["moqd_score"][-1],
            minval=self.minval,
            maxval=self.maxval,
            ax=axes[2]
        )

        plt.show()

    
    def plot_scores(
        self,
        metrics_history: Dict,
        episode_length: int = 1,
    ) -> None:
        
        fig, axes = plt.subplots(figsize=(18, 6), ncols=2)

        axes = plot_mome_scores_evolution(
            metrics_history=metrics_history,
            ax=axes,
            fig=fig,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            episode_length=episode_length,
        )

        plt.show()

    def plot_max_scores(
        self,
        metrics_history: Dict,
        episode_length: int = 1,
    ):

        fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

        axes = plot_mome_max_scores_evolution(
            metrics_history=metrics_history,
            ax=axes,
            fig=fig,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            episode_length=episode_length,
        )

        plt.show()