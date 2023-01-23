from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitter
from qdax.core.emitters.pga_me_emitter import PGAMEConfig
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey


class AblationPGAMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        objective_index: int,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.mutation_ga_batch_size
        qpg_batch_size = config.mutation_qpg_batch_size * 2

        emitters = []

        qpg_config = QualityPGConfig(
            num_objective_functions=config.num_objective_functions,
            objective_function_index=objective_index,
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )

        emitters.append(q_emitter)

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        emitters.append(ga_emitter)

        super().__init__(emitters=tuple(emitters))