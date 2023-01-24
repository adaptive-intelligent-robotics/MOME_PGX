import csv
import hydra
import jax.numpy as jnp
import jax
import logging
import os
import pandas as pd
import time
import visu_brax

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from plotting_fns import Plotter
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.emitter import Emitter
from qdax.core.mome import MOME, MOMERepertoire
from qdax.core.bandit_multi_emitter_mome import BanditMultiEmitterMOME
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Fitness, Descriptor, RNGKey, Genotype, Centroid
from qdax.utils.metrics import CSVLogger


class RunBanditMOPGA:

    """
    Args:
        pareto_front_max_length: max number of solutions per cell/in each pareto front
        num_descriptor_dimensions: dimensionality of solution BD
        minval: minimum value of BD
        maxval: maximum value of BD
        num_iteration s: number of MAP-Elites iteration to run
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
                bias_sampling: bool,
                minval: int,
                maxval: int,
                num_evaluations: int, 
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
        self.bias_sampling = bias_sampling
        self.minval = minval
        self.maxval = maxval
        self.num_evaluations =  num_evaluations
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
        self.num_iterations = self.num_evaluations // self.batch_size
        assert(self.num_iterations%self.metrics_log_period == 0, 
            "Make sure num_iterations % metrics_log_period == 0 to ensure correct number of evaluations")

        num_loops = int(self.num_iterations/self.metrics_log_period)

        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().handlers[0].setLevel(logging.INFO)
        logger = logging.getLogger(f"{__name__}")
        output_dir = "./" 

        # Name save directories
        _repertoire_plots_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "plots")
        _repertoire_num_sols_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "num_sols")
        _normalised_repertoire_save_dir = os.path.join(output_dir, "checkpoints", "repertoires", "normalised_hypervolume")
        _metrics_dir = os.path.join(output_dir, "checkpoints")
        _final_metrics_dir = os.path.join(output_dir, "final", "metrics")
        _final_plots_dir = os.path.join(output_dir, "final", "plots")
        _final_repertoire_dir = os.path.join(output_dir, "final", "repertoire/")

        # Create save directories
        os.makedirs(_repertoire_plots_save_dir, exist_ok=True)
        os.makedirs(_repertoire_num_sols_save_dir, exist_ok=True)
        os.makedirs(_normalised_repertoire_save_dir, exist_ok=True)
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


        # Instantiate plotter 
        plotter = Plotter(
            minval=self.minval,
            maxval=self.maxval,
            pareto_front_max_length=self.pareto_front_max_length,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            episode_length=self.episode_length,
        )

        # Instantiate MOME
        bandit_mome = BanditMultiEmitterMOME(
            scoring_function=self.scoring_fn,
            emitter=self.emitter,
            metrics_function=self.metrics_fn,
            bias_sampling=self.bias_sampling,
        )
        logger.warning("\n")
        logger.warning(f"----  Using bandit emitter ----")
        logger.warning(f"---- Bias Sampling: {self.bias_sampling} ----")
        logger.warning("\n")

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
        repertoire, init_metrics, emitter_state, random_key = bandit_mome.init(
            init_genotypes,
            centroids,
            self.pareto_front_max_length,
            random_key
        )

        initial_repertoire_time = time.time() - algorithm_start_time
        total_algorithm_duration += initial_repertoire_time
        logger.warning("--- Initialised initial repertoire ---")


        # Store initial repertoire metrics and convert to jnp.arrays
        metrics_history = init_metrics.copy()
        for k, v in metrics_history.items():
            metrics_history[k] = jnp.expand_dims(jnp.array(v), axis=0)

        logger.warning(f"------ Initial Repertoire Metrics ------")
        logger.warning(f"--- MOQD Score: {init_metrics['moqd_score']:.2f}")
        logger.warning(f"--- Coverage: {init_metrics['coverage']:.2f}%")
        logger.warning("--- Max Fitnesses:" +  str(init_metrics['max_scores']))

        logger_header = [k for k,_ in metrics_history.items()]
        logger_header.insert(0, "iteration")
        logger_header.append("time")

        csv_logger = CSVLogger(
            "checkpoint-metrics-logs.csv",
            header=logger_header
        )

        logger.warning("--- Starting the algorithm main process ---")
       
        bandit_mome_scan_fn = bandit_mome.scan_update

        # Run the algorithm
        for iteration in range(num_loops):
            start_time = time.time()

            # 'Log period' number of QD itertions
            (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
                bandit_mome_scan_fn,
                (repertoire, emitter_state, random_key),
                (),
                length=self.metrics_log_period,
            )

            timelapse = time.time() - start_time
            total_algorithm_duration += timelapse

            # log metrics
            metrics_history = {key: jnp.concatenate((metrics_history[key], metrics[key]), axis=0) for key in metrics}
            logged_metrics = {"iteration": (iteration+ 1)*self.metrics_log_period,  "time": timelapse}
            for key, value in metrics.items():
                # take last value
                logged_metrics[key] = value[-1]
            
            # Print metrics
            logger.warning(f"------ Iteration: {(iteration+1)*self.metrics_log_period} out of {self.num_iterations} ------")
            logger.warning(f"--- MOQD Score: {metrics['moqd_score'][-1]:.2f}")
            logger.warning(f"--- Coverage: {metrics['coverage'][-1]:.2f}%")
            logger.warning("--- Max Fitnesses:" +  str(metrics['max_scores'][-1]))

            csv_logger.log(logged_metrics)

            # Save plot of repertoire every plot_repertoire_period
            if (iteration+1)*self.metrics_log_period % self.plot_repertoire_period == 0:
                if self.num_descriptor_dimensions == 2:
                    plotter.plot_repertoire(
                        repertoire,
                        centroids,
                        metrics,
                        save_dir=_repertoire_plots_save_dir,
                        save_name=f"{(iteration+1)*self.metrics_log_period }",
                    )
                    plotter.plot_num_solutions(
                        centroids,
                        metrics,
                        save_dir=_repertoire_num_sols_save_dir,
                        save_name=f"{(iteration+1)*self.metrics_log_period}",
                    )
                    
                    plotter.plot_normalised_repertoire(
                        centroids,
                        metrics,
                        save_dir=_normalised_repertoire_save_dir,
                        save_name=f"{(iteration+1)*self.metrics_log_period}",
                    )
                
                if self.save_checkpoint_visualisations:
                    random_key, subkey = jax.random.split(random_key)
                    visu_brax.save_mo_samples(
                        env,
                        policy_network,
                        subkey,
                        repertoire, 
                        self.num_save_visualisations,
                        (iteration+1)*self.metrics_log_period,
                        save_dir=_visualisations_save_dir,
                    )
            
            # Save latest repertoire and metrics every 'checkpoint_period'
            if (iteration+1)*self.metrics_log_period  % self.checkpoint_period == 0:
                repertoire.save(path=_final_repertoire_dir)
                metrics_history_df = pd.DataFrame.from_dict(metrics_history,orient='index').transpose()
                metrics_history_df.to_csv(os.path.join(_metrics_dir, "metrics_history.csv"), index=False)


        total_duration = time.time() - init_time

        logger.warning("--- FINAL METRICS ---")
        logger.warning(f"Total duration: {total_duration:.2f}s")
        logger.warning(f"Main algorithm duration: {total_algorithm_duration:.2f}s")
        logger.warning(f"MOQD Score: {metrics['moqd_score'][-1]:.2f}")
        logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning("Max Fitnesses:" + str(metrics['max_scores'][-1]))

        # Save metrics

        metrics_history_df = pd.DataFrame.from_dict(metrics_history,orient='index').transpose()
        metrics_history_df.to_csv(os.path.join(_metrics_dir, "metrics_history.csv"), index=False)

        metrics_df = pd.DataFrame.from_dict(metrics,orient='index').transpose()
        metrics_df.to_csv(os.path.join(_final_metrics_dir, "final_metrics.csv"), index=False)

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
        if self.num_descriptor_dimensions == 2:
            plotter.plot_repertoire(
                repertoire,
                centroids,
                metrics,
                save_dir=_final_plots_dir,
                save_name="final",
            )

            plotter.plot_num_solutions(
                        centroids,
                        metrics,
                        save_dir=_final_plots_dir,
                        save_name=f"final",
            )

            plotter.plot_normalised_repertoire(
                centroids,
                metrics,
                save_dir=_final_plots_dir,
                save_name=f"final",
            )

        plotter.plot_scores_evolution(
            metrics_history,
            save_dir=_final_plots_dir

        )

        plotter.plot_max_scores_evolution(
            metrics_history,
            save_dir=_final_plots_dir

        )

        return repertoire, centroids, random_key, metrics, metrics_history