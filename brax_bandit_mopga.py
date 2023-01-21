import hydra
import jax.numpy as jnp
import jax

from brax_step_functions import play_mo_step_fn
from dataclasses import dataclass
from envs_setup import get_env_metrics
from functools import partial
from typing import Tuple
from run_bandit_mome import RunBanditMOPGA
from qdax import environments
from qdax.core.emitters.bandit_mopga_emitter import (
    BanditMOPGAConfig, 
    BanditMOPGAEmitter, 
    DynamicBanditMOPGAConfig, 
    DynamicBanditMOPGAEmitter
)
from qdax.core.neuroevolution.mdp_utils import scoring_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.metrics import default_moqd_metrics


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    seed: int
    env_name: str
    fixed_init_state: bool 
    episode_length: int

    # MOO parameters
    num_objective_functions: int
    pareto_front_max_length: int

    # Initialisation parameters
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
    total_batch_size: int
    bandit_scaling_param: float
    dynamic_window_size: int
    dynamic_emitter: bool

    # Logging parameters
    metrics_log_period: int
    plot_repertoire_period: int
    checkpoint_period: int
    save_checkpoint_visualisations: bool
    save_final_visualisations: bool
    num_save_visualisations: int

    # TD3 params
    replay_buffer_size: int
    critic_hidden_layer_size: Tuple[int,...]
    critic_learning_rate: float
    greedy_learning_rate: float
    policy_learning_rate: float
    noise_clip: float 
    policy_noise: float 
    discount: float 
    reward_scaling: Tuple[float, ...]
    transitions_batch_size: int 
    soft_tau_update: float 
    policy_delay: int
    num_critic_training_steps: int 
    num_pg_training_steps: int 

    # Ablation parameters
    only_forward_emitter: bool
    only_energy_emitter: bool


@hydra.main(config_path="configs/brax/", config_name="brax_bandit_mopga")
def main(config: ExperimentConfig) -> None:

    # Init environment
    env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=config.fixed_init_state)

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
    keys = jax.random.split(subkey, num=config.total_batch_size)
    fake_batch = jnp.zeros(shape=(config.total_batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states (same initial state for each individual in env_batch)
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.total_batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # TO DO: save init_state  
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


    # Get the GA emitter
    variation_function = partial(
        isoline_variation, 
        iso_sigma=config.iso_sigma, 
        line_sigma=config.line_sigma
    )

    # Define the PG-emitter config
    




    if config.dynamic_emitter: 

        print("USING DYNAMIC BANDIT MULTI EMITTER")

        mopga_emitter_config = DynamicBanditMOPGAConfig(
            num_objective_functions=config.num_objective_functions,
            total_batch_size=config.total_batch_size,
            bandit_scaling_param=config.bandit_scaling_param,
            dynamic_window_size=config.dynamic_window_size,
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
            policy_delay=config.policy_delay,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps
        )

        bandit_mopga_emitter = DynamicBanditMOPGAEmitter(
            config=mopga_emitter_config,
            policy_network=policy_network,
            env=env,
            variation_fn=variation_function,
        )

    
    else:
        print("USING BANDIT MULTI EMITTER")

        mopga_emitter_config = BanditMOPGAConfig(
            num_objective_functions=config.num_objective_functions,
            total_batch_size=config.total_batch_size,
            bandit_scaling_param=config.bandit_scaling_param,
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
            policy_delay=config.policy_delay,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps
        )

        bandit_mopga_emitter = BanditMOPGAEmitter(
            config=mopga_emitter_config,
            policy_network=policy_network,
            env=env,
            variation_fn=variation_function,
        )

    bandit_mome = RunBanditMOPGA(
        pareto_front_max_length=config.pareto_front_max_length,
        num_descriptor_dimensions=env.behavior_descriptor_length,
        minval=config.minval,
        maxval=config.maxval,
        num_evaluations=config.num_evaluations, 
        num_centroids=config.num_centroids,
        num_init_cvt_samples=config.num_init_cvt_samples,
        batch_size=config.total_batch_size, 
        episode_length=config.episode_length,
        scoring_fn=brax_scoring_function,
        emitter=bandit_mopga_emitter,
        metrics_fn=metrics_function,
        num_objective_functions=config.num_objective_functions,
        metrics_log_period=config.metrics_log_period,
        plot_repertoire_period=config.plot_repertoire_period,
        checkpoint_period=config.checkpoint_period,
        save_checkpoint_visualisations=config.save_checkpoint_visualisations,
        save_final_visualisations=config.save_final_visualisations,
        num_save_visualisations=config.num_save_visualisations,
    )

    repertoire = bandit_mome.run(
        random_key, 
        init_genotypes,
        env,
        policy_network
    )

                                                                
if __name__ == '__main__':
    main()