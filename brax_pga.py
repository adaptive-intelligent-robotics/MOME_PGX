import hydra
import jax.numpy as jnp
import jax

from dataclasses import dataclass
from envs_setup import get_env_metrics
from functools import partial
from typing import Tuple
from run_pga import RunPGA
from qdax import environments
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.neuroevolution.mdp_utils import scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.metrics import CSVLogger, default_qd_metrics, default_moqd_metrics
from brax_step_functions import play_mo_step_fn

@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    seed: int
    env_name: str
    fixed_init_state: bool
    episode_length: int

    # Initialisation parameters
    num_evaluations: int
    num_init_cvt_samples: int
    num_centroids: int

    # Brax parameters
    num_objective_functions: int
    num_descriptor_dimensions: int
    policy_hidden_layer_sizes: Tuple[int,...]
    pareto_front_max_length: int
    minval: float 
    maxval: float

    # Emitter parameters
    iso_sigma: float
    line_sigma: float 

    # Logging parameters
    metrics_log_period: int
    plot_repertoire_period: int
    checkpoint_period: int
    save_checkpoint_visualisations: bool
    save_final_visualisations: bool
    num_save_visualisations: int

    # TD3 params
    num_objective_emitters: int
    mutation_ga_batch_size: int
    mutation_qpg_batch_size: int    
    replay_buffer_size: int
    critic_hidden_layer_size: Tuple[int,...]
    critic_learning_rate: float
    greedy_learning_rate: float
    policy_learning_rate: float
    noise_clip: float 
    policy_noise: float 
    discount: float 
    reward_scaling: Tuple[float,...]
    transitions_batch_size: int 
    soft_tau_update: float 
    num_critic_training_steps: int 
    num_pg_training_steps: int 




@hydra.main(config_path="configs/brax/", config_name="brax_pga")
def main(config: ExperimentConfig) -> None:
    batch_size = config.mutation_qpg_batch_size + config.mutation_ga_batch_size

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
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states (same initial state for each individual in env_batch)
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # TO DO: save init_state
    play_step_fn = partial(
        play_mo_step_fn,
        policy_network=policy_network,
        env=env,
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

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[config.env_name]
    reward_offset = jnp.sum(jnp.array(reward_offset))

    # Define a metrics function
    metrics_function = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.episode_length,
    )

    moqd_metrics_function = partial(
        default_moqd_metrics,
        reference_point=jnp.array(reference_point),
        max_rewards=jnp.array(max_rewards)
    )

    # Define the PG-emitter config
    pga_emitter_config = PGAMEConfig(
        num_objective_functions=config.num_objective_functions,
        mutation_ga_batch_size=config.mutation_ga_batch_size,
        mutation_qpg_batch_size=config.mutation_qpg_batch_size,        
        batch_size=config.transitions_batch_size,
        critic_hidden_layer_size=config.critic_hidden_layer_size,
        critic_learning_rate=config.critic_learning_rate,
        greedy_learning_rate=config.greedy_learning_rate,
        policy_learning_rate=config.policy_learning_rate,
        noise_clip=config.noise_clip,
        policy_noise=config.policy_noise,
        discount=config.discount,
        reward_scaling=config.reward_scaling,
        replay_buffer_size=config.replay_buffer_size,
        soft_tau_update=config.soft_tau_update,
        num_critic_training_steps=config.num_critic_training_steps,
        num_pg_training_steps=config.num_pg_training_steps
    )


    # Get the emitter
    variation_function = partial(
        isoline_variation, 
        iso_sigma=config.iso_sigma, 
        line_sigma=config.line_sigma
    )

    pg_emitter = PGAMEEmitter(
        config=pga_emitter_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_function,
    )

    pga = RunPGA(
        num_evaluations=config.num_evaluations,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        num_descriptor_dimensions=env.behavior_descriptor_length,
        minval=config.minval,
        maxval=config.maxval,
        scoring_fn=brax_scoring_function,
        pg_emitter=pg_emitter,
        episode_length=config.episode_length,
        batch_size=batch_size,
        metrics_fn=metrics_function,
        moqd_metrics_fn=moqd_metrics_function,
        pareto_front_max_length=config.pareto_front_max_length,
        metrics_log_period=config.metrics_log_period,
        plot_repertoire_period=config.plot_repertoire_period,
        checkpoint_period=config.checkpoint_period,
        save_checkpoint_visualisations=config.save_checkpoint_visualisations,
        save_final_visualisations=config.save_final_visualisations,
        num_save_visualisations=config.num_save_visualisations,
    )


    repertoire = pga.run(
        random_key, 
        init_genotypes,
        env,
        policy_network
    )



                                                      
if __name__ == '__main__':
    main()