import hydra
import jax.numpy as jnp
import jax

from dataclasses import dataclass
from functools import partial
from typing import Tuple
from run_mome import RunMOME
from qdax import environments
from qdax.core.emitters.mutation_operators import (
    polynomial_mutation, 
    isoline_variation, 
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.mdp_utils import mo_scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
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
    num_objective_functions: int
    reference_point: Tuple[int,...]

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

    num_iterations = config.num_evaluations // config.batch_size

    # Init environment
    env = environments.create(config.env_name, episode_length=config.episode_length)
    
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
    def play_step_fn(
        env_state,
        policy_params,
        random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """
        actions = policy_network.apply(policy_params, env_state.obs)
        
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Define a metrics function
    metrics_function = partial(
        default_moqd_metrics,
        reference_point=jnp.array(config.reference_point)
    )

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    brax_scoring_function = partial(
        mo_scoring_function,
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
        minval=config.minval,
        maxval=config.maxval,
        num_iterations=num_iterations, 
        num_centroids=config.num_centroids,
        num_init_cvt_samples=config.num_init_cvt_samples,
        batch_size=config.batch_size, 
        episode_length=config.episode_length,
        scoring_fn=brax_scoring_function,
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

    repertoire = mome.run(
        random_key, 
        init_genotypes,
        env,
        policy_network
    )

                                                      
if __name__ == '__main__':
    main()
