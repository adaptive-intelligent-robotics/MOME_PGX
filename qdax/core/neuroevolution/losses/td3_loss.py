""" Implements a function to create critic and actor losses for the TD3 algorithm."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey


def make_td3_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    reward_scaling: Tuple[float, ...],
    discount: float,
    noise_clip: float,
    policy_noise: float,
    num_objective_functions: int,
) -> Tuple[
    Callable[[Params, Params, Transition], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss functions for TD3.

    Args:
        policy_fn: forward pass through the neural network defining the policy.
        critic_fn: forward pass through the neural network defining the critic.
        reward_scaling: value to multiply the reward given by the environment.
        discount: discount factor.
        noise_clip: value that clips the noise to avoid extreme values.
        policy_noise: noise applied to smooth the bootstrapping.

    Returns:
        Return the loss functions used to train the policy and the critic in TD3.
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        critic_params: Params,
        transitions: Transition,
        objective_index: int,
    ) -> jnp.ndarray:
        """Policy loss function for TD3 agent"""
        print("--------POLICY LOSS FUNCTION--------")
        # Select action based from policy network
        action = policy_fn(policy_params, transitions.obs)
        print("ACTION:", action)
        # Estimate q values of this action
        q_value = critic_fn(
            critic_params, obs=transitions.obs, actions=action  # type: ignore
        )
        print("Q VALUE:", q_value)
        # Only use Q value from first critic estimate
        q1_action = q_value[0, objective_index] # take q values from first critic networks 
        print(" Q1 ACTION:", q1_action)
        # MC estimate of gradient
        policy_loss = -jnp.mean(q1_action, axis=0) #take mean across batch
        print("POLICY LOSS:", jnp.mean(q1_action, axis=0))
        return policy_loss

    @partial(
        jax.jit,
        static_argnames=("reward_scaling",)
    )
    def _critic_loss_fn(
        critic_params: Params,
        target_policy_params: Params,
        target_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent"""
        #print("--------CRITIC LOSS FUNCTION--------")

        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        #print("NOISE:", noise)
        next_action = (
            policy_fn(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        
        #print("NEXT ACTION:", next_action)

        next_q = critic_fn(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=next_action
        )

        #print("NEXT Q:", next_q)
        dones = jnp.repeat(jnp.expand_dims(transitions.dones, axis=-1), repeats=num_objective_functions, axis=-1)

        next_v = jnp.min(next_q, axis=0)

        #print("REWARD SCALING:", reward_scaling)
        #print("TRANSITION REWARDS:", transitions.rewards)
        #print("TRANSITION DONES:", dones)
        #print("DISCOUNT:", discount)
        #print("NEXT_V:", next_v)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * jnp.array(reward_scaling)
            + (1.0 - dones) * discount * next_v
        )
        #print("TARGET_Q:", target_q)

        q_old_action = critic_fn(  # type: ignore
            critic_params,
            obs=transitions.obs,
            actions=transitions.actions,
        )
        #print("Q OLD ACTION:", q_old_action)
        #print("TARGET_Q_EXPANDED:", jnp.expand_dims(target_q, 0))

        q_error = q_old_action - jnp.expand_dims(target_q, 0)

        #print("TRUNCATIONS:", jnp.expand_dims(1 - transitions.truncations, -1))
        # Better bootstrapping for truncated episodes.
        q_error *= jnp.expand_dims(1 - transitions.truncations, -1)

        #print("Q_ERROR:", q_error)
    
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        #print("Q_LOSS:", q_loss)
        #print("--------CRITIC LOSS FUNCTION--------")
        return q_loss

    return _policy_loss_fn, _critic_loss_fn
