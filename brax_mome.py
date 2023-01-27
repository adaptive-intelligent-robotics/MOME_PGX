import hydra
import jax.numpy as jnp
import jax

from brax_step_functions import play_mo_step_fn
from dataclasses import dataclass
from envs_setup import get_env_metrics
from functools import partial
from typing import Tuple
from run_mome import RunMOME
from qdax import environments
from qdax.core.emitters.mutation_operators import (
    polynomial_mutation, 
    isoline_variation, 
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.mdp_utils import scoring_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import default_moqd_metrics

@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    seed: int
    env_name: str
    fixed_init_state: bool
    episode_length: int

    # MOME parameters
    pareto_front_max_length: int
    num_objective_functions: int
    bias_sampling: bool

    # Initialisation parameters
    batch_size: int 
    num_evaluations: int
    num_init_cvt_samples: int
    num_centroids: int

    # Brax parameters
    num_descriptor_dimensions: int
    policy_hidden_layer_sizes: Tuple[int,...]
    minval: float 
    maxval: float

    # Emitter parameters
    iso_sigma: float
    line_sigma: float 
    proportion_to_mutate: float
    eta: float
    crossover_percentage: float

    # Logging parameters
    metrics_log_period: int
    plot_repertoire_period: int
    checkpoint_period: int
    save_checkpoint_visualisations: bool
    save_final_visualisations: bool
    num_save_visualisations: int




@hydra.main(config_path="configs/brax/", config_name="brax_mome")
def main(config: ExperimentConfig) -> None:

    # Init environment
    env = environments.create(config.env_name, 
        episode_length=config.episode_length, 
        fixed_init_state=config.fixed_init_state)
    
    env_metrics = get_env_metrics(config.env_name,
        episode_length=config.episode_length
    )

    reference_point = env_metrics["min_rewards"]
    max_rewards = env_metrics["max_rewards"]
    
    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states (same initial state for each individual in batch)
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # TO DO: save init_state

    # Define the function to play a step with the policy in the environment
    play_step_fn = partial(
        play_mo_step_fn,
        policy_network=policy_network,
        env=env,
    )  

    # Define a metrics function
    metrics_function = partial(
        default_moqd_metrics,
        reference_point=jnp.array(reference_point),
        max_rewards=jnp.array(max_rewards)
    )

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    brax_scoring_function = partial(
        scoring_function,
        init_states=init_states,
        episode_length=config.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        num_objective_functions=config.num_objective_functions,
    )


    # crossover function
    crossover_function = partial(
        isoline_variation, 
        iso_sigma=config.iso_sigma, 
        line_sigma=config.line_sigma
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
    
    mome = RunMOME(
        pareto_front_max_length=config.pareto_front_max_length,
        num_descriptor_dimensions=env.behavior_descriptor_length,
        bias_sampling=config.bias_sampling,
        minval=config.minval,
        maxval=config.maxval,
        num_evaluations=config.num_evaluations, 
        num_centroids=config.num_centroids,
        num_init_cvt_samples=config.num_init_cvt_samples,
        batch_size=config.batch_size, 
        episode_length=config.episode_length,
        scoring_fn=brax_scoring_function,
        emitter=mixing_emitter,
        metrics_fn=metrics_function,
        metrics_log_period=config.metrics_log_period,
        plot_repertoire_period=config.plot_repertoire_period,
        checkpoint_period=config.checkpoint_period,
        save_checkpoint_visualisations=config.save_checkpoint_visualisations,
        save_final_visualisations=config.save_final_visualisations,
        num_save_visualisations=config.num_save_visualisations,
    )

    repertoire = mome.run(
        random_key, 
        init_genotypes,
        env,
        policy_network
    )

                                                      
if __name__ == '__main__':
    main()
