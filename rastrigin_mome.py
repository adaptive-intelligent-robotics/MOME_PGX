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
from mome import RunMOME
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
    plot_mome_pareto_fronts, 
    plot_mome_scores_evolution
)
from qdax.utils.metrics import CSVLogger, default_moqd_metrics



@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    alg_name: str
    seed: int
    env_name: str
    episode_length: int

    # MOME parameters
    pareto_front_max_length: int
    init_batch_size: int
    batch_size: int 
    num_evaluations: int
    num_centroids: int
    num_init_cvt_samples: int
    num_objective_functions: int

    # Rastrigin parameters
    num_param_dimensions: int
    num_descriptor_dimensions: int
    minval: int
    maxval: int
    lag: float
    base_lag: float
    reference_point: Tuple[int, ...]

    # Emitter parameters
    proportion_to_mutate: float
    eta: float
    proportion_var_to_change: float
    crossover_percentage: float

    # Logging parameters
    metrics_log_period: int
    plot_repertoire_period: int
    checkpoint_period: int
    save_checkpoint_visualisations: bool
    save_final_visualisations: bool
    num_save_visualisations: int

def rastrigin_scorer(
    genotypes: jnp.ndarray, 
    base_lag: float, 
    lag: float
) -> Tuple[Fitness, Descriptor]:
    """
    Rastrigin Scorer with first two dimensions as descriptors and two Rastrigin objectives with extrema shifted
    """
    descriptors = genotypes[:, :2]

    f1 = -(
        10 * genotypes.shape[1]
        + jnp.sum(
            (genotypes - base_lag) ** 2
            - 10 * jnp.cos(2 * jnp.pi * (genotypes - base_lag)),
            axis=1,
        )
    )

    f2 = -(
        10 * genotypes.shape[1]
        + jnp.sum(
            (genotypes - lag) ** 2 - 10 * jnp.cos(2 * jnp.pi * (genotypes - lag)),
            axis=1,
        )
    )
    scores = jnp.stack([f1, f2], axis=-1)

    return scores, descriptors




@hydra.main(config_path="configs/rastrigin/", config_name="rastrigin_mome")
def main(config: ExperimentConfig) -> None:

    num_iterations = config.num_evaluations // config.batch_size

    # Initialise random key
    random_key = jax.random.PRNGKey(config.seed)
    random_key, subkey = jax.random.split(random_key)

    # Initialise genotypes
    init_genotypes = jax.random.uniform(
    subkey, 
    (config.batch_size, config.num_param_dimensions), 
    minval=config.minval, 
    maxval=config.maxval, 
    dtype=jnp.float32
    )

    # crossover function
    crossover_function = partial(
        polynomial_crossover,
        proportion_var_to_change=config.proportion_var_to_change
    )

    # mutation function
    mutation_function = partial(
        polynomial_mutation,
        eta=config.eta,
        minval=config.minval,
        maxval=config.maxval,
        proportion_to_mutate=config.proportion_to_mutate
    )

    # Define emitter
    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_function, 
        variation_fn=crossover_function, 
        variation_percentage=config.crossover_percentage, 
        batch_size=config.batch_size
    )


    # Default metrics: moqd_score, max_hypervolumes, max_scores, max_sum_scores, coverage, number_solutions, 
    # global_hypervolume
    metrics_function = partial(
        default_moqd_metrics,
        reference_point=jnp.array(config.reference_point)
    )

    scoring_function = partial(rastrigin_scorer, base_lag=config.base_lag, lag=config.lag)


    def scoring_fn(
        genotypes: jnp.ndarray, 
        random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

        fitnesses, descriptors = scoring_function(genotypes)
        return fitnesses, descriptors, {}, random_key

    # Instantiate MOME
    mome = RunMOME(
                pareto_front_max_length=config.pareto_front_max_length,
                num_descriptor_dimensions=config.num_descriptor_dimensions,
                minval=config.minval,
                maxval=config.maxval,
                num_iterations=num_iterations, 
                num_centroids=config.num_centroids,
                num_init_cvt_samples=config.num_init_cvt_samples,
                batch_size=config.batch_size, 
                episode_length=config.episode_length,
                scoring_fn=scoring_fn,
                emitter=mixing_emitter,
                metrics_fn=metrics_function,
                num_objective_functions=config.num_objective_functions,
                metrics_log_period=config.metrics_log_period,
                plot_repertoire_period=config.plot_repertoire_period,
                checkpoint_period=config.checkpoint_period,
                save_checkpoint_visualisations=config.save_checkpoint_visualisations,
                save_final_visualisations=config.save_final_visualisations,
                num_save_visualisations=config.num_save_visualisations,
    ) 

    # Run MOME and plot results
    mome.run(random_key, init_genotypes)




if __name__ == '__main__':
    main()