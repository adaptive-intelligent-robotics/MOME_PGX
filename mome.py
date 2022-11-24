import chex
import flax
import hydra
import jax.numpy as jnp
import jax
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import time
import visu_brax

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
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Fitness, Descriptor, RNGKey, ExtraScores, Genotype, Centroid
from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_max_scores_evolution,
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution,
)


class RunMOME:

    """
    Args:
        pareto_front_max_length: max number of solutions per cell/in each pareto front
        num_descriptor_dimensions: dimensionality of solution BD
        minval: minimum value of BD
        maxval: maximum value of BD
        num_iterations: number of MAP-Elites iterations to run
        num_centroids: number of cells in grid
        num_init_cvt_samples: number of sampled points to be used for clustering to determine centroids 
        batch_size: number of solutions to select and mutate in each MAP-Elites iteration
        scoring_fn: function that returns fitness and descriptor of a solution
        emitter: function determining variation mechanism of solutions
        metrics_fn: function for calculating moqd metrics 
    """

    def __init__(self,
                pareto_front_max_length: int,
                num_descriptor_dimensions: int,
                minval: int,
                maxval: int,
                num_iterations: int, 
                num_centroids: int,
                num_init_cvt_samples: int,
                batch_size: int, 
                episode_length: int,
                scoring_fn: Callable,
                emitter: Emitter,
                metrics_fn: Callable,
                num_objective_functions: int,
                metrics_log_period: int,
                plot_repertoire_period: int,
                checkpoint_period: int,
                save_checkpoint_visualisations: bool,
                save_final_visualisations: bool,
                num_save_visualisations: int,
        ):

        self.pareto_front_max_length = pareto_front_max_length
        self.num_descriptor_dimensions = num_descriptor_dimensions
        self.minval = minval
        self.maxval = maxval
        self.num_iterations =  num_iterations
        self.num_centroids = num_centroids
        self.num_init_cvt_samples = num_init_cvt_samples
        self.batch_size =  batch_size
        self.episode_length = episode_length
        self.scoring_fn = scoring_fn
        self.emitter = emitter
        self.metrics_fn = metrics_fn
        self.num_objective_functions = num_objective_functions
        self.metrics_log_period = metrics_log_period
        self.plot_repertoire_period = plot_repertoire_period
        self.checkpoint_period = checkpoint_period
        self.save_checkpoint_visualisations = save_checkpoint_visualisations
        self.save_final_visualisations = save_final_visualisations
        self.num_save_visualisations = num_save_visualisations


    def run(self,
            random_key: RNGKey,
            init_genotypes: Genotype,
            env: Optional[QDEnv]=None,
            policy_network: Optional[MLP]=None,
    ) -> Tuple[MOMERepertoire, Genotype, RNGKey]:
            
        # Set up logging functions 
        num_loops = self.num_iterations // self.metrics_log_period
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().handlers[0].setLevel(logging.INFO)
        logger = logging.getLogger(f"{__name__}")
        output_dir = "./" 

        # Name save directories
        _repertoire_plots_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "plots")
        _metrics_dir = os.path.join(output_dir, "checkpoints")
        _final_metrics_dir = os.path.join(output_dir, "final", "metrics")
        _final_plots_dir = os.path.join(output_dir, "final", "plots")
        _final_repertoire_dir = os.path.join(output_dir, "final", "repertoire/")

        # Create save directories
        os.makedirs(_repertoire_plots_save_dir, exist_ok=True)
        os.makedirs(_metrics_dir, exist_ok=True)
        os.makedirs(_final_metrics_dir, exist_ok=True)
        os.makedirs(_final_plots_dir, exist_ok=True)
        os.makedirs(_final_repertoire_dir, exist_ok=True)

        # Create visualisation directories
        if self.save_checkpoint_visualisations:
            _visualisations_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "visualisations")
            os.makedirs(_visualisations_save_dir)
            
        if self.save_final_visualisations:
            _final_visualisation_dir = os.path.join(output_dir, "final", "visualisations")
            os.makedirs(_final_visualisation_dir)

        # Instantiate MOME
        mome = MOME(
            scoring_function=self.scoring_fn,
            emitter=self.emitter,
            metrics_function=self.metrics_fn,
        )

        # Compute the centroids
        logger.warning("--- Computing the CVT centroids ---")

        # Start timing the algorithm
        init_time = time.time()
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=self.num_descriptor_dimensions, 
            num_init_cvt_samples=self.num_init_cvt_samples, 
            num_centroids=self.num_centroids, 
            minval=self.minval, 
            maxval=self.maxval,
            random_key=random_key,
        )


        centroids_init_time = time.time() - init_time
        logger.warning(f"--- Duration for CVT centroids computation : {centroids_init_time:.2f}s ---")

        logger.warning("--- Algorithm initialisation ---")
        total_algorithm_duration = 0.0
        algorithm_start_time = time.time()

        # Initialize repertoire and emitter state
        repertoire, emitter_state, random_key = mome.init(
            init_genotypes,
            centroids,
            self.pareto_front_max_length,
            random_key
        )

        initial_repertoire_time = time.time() - algorithm_start_time
        total_algorithm_duration += initial_repertoire_time
        logger.warning("--- Initialised initial repertoire ---")
        logger.warning("--- Starting the algorithm main process ---")

        timings = {"initial_repertoire_time": initial_repertoire_time,
                    "centroids_init_time": centroids_init_time,
                    "runtime_logs": jnp.zeros([(num_loops)+1, 1]),
                    "avg_iteration_time": 0.0,
                    "avg_evalps": 0.0}


        metrics_history = {
                "hypervolumes": jnp.zeros((1, self.num_centroids)),
                "moqd_score": jnp.array([0.0]), 
                "max_hypervolume": jnp.array([0.0]), 
                "max_scores": jnp.zeros((1, self.num_objective_functions)),
                "max_sum_scores": jnp.array([0.0]), 
                "coverage": jnp.array([0.0]), 
                "number_solutions": jnp.array([0.0]), 
                "global_hypervolume": jnp.array([0.0]), 
        }

        logger.warning(f"--- Running MOME for {num_loops} loops ---")

        # Run the algorithm
        for i in range(num_loops):
            iteration = (i+1) * self.metrics_log_period
            logger.warning(f"------ Iteration {iteration} out of {self.num_iterations} ------")

            start_time = time.time()

            # 'Log period' number of QD itertions
            (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
                mome.scan_update,
                (repertoire, emitter_state, random_key),
                (),
                length=self.metrics_log_period,
            )

            timelapse = time.time() - start_time
            total_algorithm_duration += timelapse
            metrics_history = {key: jnp.concatenate((metrics_history[key], metrics[key]), axis=0) for key in metrics}

            logger.warning(f"--- MOQD Score: {metrics['moqd_score'][-1]:.2f}")
            logger.warning(f"--- Coverage: {metrics['coverage'][-1]:.2f}%")
            logger.warning("--- Max Fitnesses:" +  str(metrics['max_scores'][-1]))

            timings["avg_iteration_time"] = (timings["avg_iteration_time"]*(i*self.metrics_log_period) + timelapse) / ((i+1)*self.metrics_log_period)
            timings["avg_evalps"] = (timings["avg_evalps"]*(i*self.metrics_log_period) + ((self.batch_size*self.metrics_log_period)/timelapse)) / ((i+1)*self.metrics_log_period)
            timings["runtime_logs"] = timings["runtime_logs"].at[i, 0].set(total_algorithm_duration)

           
            # Save plot of repertoire every plot_repertoire_period iterations
            if iteration % self.plot_repertoire_period == 0:
                self.plot_repertoire(
                    repertoire,
                    centroids,
                    metrics,
                    save_dir=_repertoire_plots_save_dir,
                    save_name=f"{iteration}",
                )
            
            # Save latest repertoire and metrics every 'checkpoint_period' iterations
            if iteration % self.checkpoint_period == 0:
                repertoire.save(path=_final_repertoire_dir)
                    
                with open(os.path.join(_metrics_dir, "metrics_history.pkl"), 'wb') as f:
                    pickle.dump(metrics_history, f)

                with open(os.path.join(_metrics_dir, "timings.pkl"), 'wb') as f:
                    pickle.dump(timings, f)
                
                if self.save_checkpoint_visualisations:
                    random_key, subkey = jax.random.split(random_key)
                    visu_brax.save_mo_samples(
                        env,
                        policy_network,
                        subkey,
                        repertoire, 
                        self.num_save_visualisations,
                        iteration,
                        save_dir=_visualisations_save_dir,
                    )

        total_duration = time.time() - init_time

        logger.warning("--- FINAL METRICS ---")
        logger.warning(f"Total duration: {total_duration:.2f}s")
        logger.warning(f"Main algorithm duration: {total_algorithm_duration:.2f}s")
        logger.warning(f"MOQD Score: {metrics['moqd_score'][-1]:.2f}")
        logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning("Max Fitnesses:" + str(metrics['max_scores'][-1]))

        # Save metrics
        with open(os.path.join(_metrics_dir, "metrics_history.pkl"), 'wb') as f:
            pickle.dump(metrics_history, f)

        with open(os.path.join(_metrics_dir, "timings.pkl"), 'wb') as f:
            pickle.dump(timings, f)

        with open(os.path.join(_final_metrics_dir, "final_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)

        # Save final repertoire
        repertoire.save(path=_final_repertoire_dir)

        # Save visualisation of best repertoire
        if self.save_final_visualisations:
            random_key, subkey = jax.random.split(random_key)
            
            visu_brax.save_mo_samples(
                env,                       
                policy_network,
                subkey,
                repertoire, 
                self.num_save_visualisations,
                save_dir=_final_visualisation_dir,
            )

        # Save final plots
        self.plot_repertoire(
            repertoire,
            centroids,
            metrics,
            save_dir=_final_plots_dir,
            save_name="final",
        )

        self.plot_scores_evolution(
            metrics_history,
            save_dir=_final_plots_dir

        )

        self.plot_max_scores_evolution(
            metrics_history,
            save_dir=_final_plots_dir

        )

        return repertoire, centroids, random_key, metrics, metrics_history


    def plot_repertoire(
        self,
        repertoire: MOMERepertoire,
        centroids: Centroid,
        metrics: Dict,
        save_dir: str="./",
        save_name: str="",
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