from functools import partial
from typing import Optional, Tuple, List, Any

import jax
import numpy as np
from chex import ArrayTree
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.bandit_multi_emitter import BanditMultiEmitter
from qdax.core.neuroevolution.buffers.scores_buffer import ScoresBuffer
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey, Mask


class DynamicBanditMultiEmitterState(EmitterState):
    """contains state for bandit"""
    emitter_added_counts_buffer: ScoresBuffer
    emitter_batch_sizes_buffer: ScoresBuffer
    emitter_states: Tuple[EmitterState, ...]
    emitter_batch_sizes: jnp.ndarray
    emitter_masks: Tuple[Mask, ...]


class DynamicBanditMultiEmitter(BanditMultiEmitter):
    """Emitter that mixes several emitters in parallel.
    WARNING: this is not the emitter of Multi-Emitter MAP-Elites.
    """

    def __init__(
        self,
        emitters: Tuple[Emitter, ...],
        bandit_scaling_param: float,
        total_batch_size: int,
        num_emitters: int,
        dynamic_window_size: int,
    ):
        self.emitters = emitters
        self.bandit_scaling_param = bandit_scaling_param
        self.total_batch_size = total_batch_size
        self.num_emitters = num_emitters
        self.dynamic_window_size = dynamic_window_size



    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[EmitterState], RNGKey]:
        """
        Initialize the state of the emitter.
        Args:
            init_genotypes: The genotypes of the initial population.
            random_key: a random key to handle stochastic operations.
        Returns:
            The initial emitter state and a random key.
        """

        # prepare keys for each emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, self.num_emitters)

        # init all emitter states - gather them
        emitter_states = []
        for emitter, subkey_emitter in zip(self.emitters, subkeys):
            emitter_state, _ = emitter.init(init_genotypes, subkey_emitter)
            emitter_states.append(emitter_state)
        
        # Init emitter bandit scores
        emitter_added_counts_buffer = ScoresBuffer.init(
            buffer_size=self.dynamic_window_size,
            num_emitters=self.num_emitters
        )
        emitter_batch_sizes_buffer = ScoresBuffer.init(
            buffer_size=self.dynamic_window_size,
            num_emitters=self.num_emitters
        )

        # Start with each emitter having the same batch size
        emitter_batch_sizes =  jnp.array(jnp.ones(shape=self.num_emitters)*self.total_batch_size/self.num_emitters, dtype=int)

        # make sure batch sizes add up to constant value
        final_batch_size = self.total_batch_size - jnp.sum(emitter_batch_sizes[:-1])
        emitter_batch_sizes = emitter_batch_sizes.at[-1].set(final_batch_size)

        emitter_masks = self.get_emitter_masks(emitter_batch_sizes)

        bandit_state = DynamicBanditMultiEmitterState(
            emitter_added_counts_buffer = emitter_added_counts_buffer,
            emitter_batch_sizes_buffer= emitter_batch_sizes_buffer,
            emitter_states = emitter_states,
            emitter_batch_sizes = emitter_batch_sizes,
            emitter_masks = emitter_masks,
        )
        
        return bandit_state, random_key
    

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        emitter_state: Optional[DynamicBanditMultiEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
        emitters_added_list: Metrics = None,
        selection_timestep: int = 1,
    ) -> Optional[DynamicBanditMultiEmitterState]:


        # First update each sub-emitter state
        new_emitter_states = self.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update emitter total offspring
        new_batch_sizes_buffer = emitter_state.emitter_batch_sizes_buffer.insert(emitter_state.emitter_batch_sizes) 

        # separate added counts into counts per emitter
        all_emitter_added_counts = emitter_state.emitter_masks * emitters_added_list

        # Find total added counts of each emitter
        emitter_added_counts = jnp.array(jnp.sum(all_emitter_added_counts, axis=-1), dtype=int)

        # update added counts buffer
        new_emitter_added_counts_buffer = emitter_state.emitter_added_counts_buffer.insert(emitter_added_counts)

        # calculate emitter_bandit_scores
        emitter_bandit_scores = self.calculate_emitter_scores(new_batch_sizes_buffer,
            new_emitter_added_counts_buffer
        )

        # Get new batch sizes for next iteration
        new_emitter_batch_sizes = self.update_batch_sizes_from_scores(emitter_bandit_scores)

        # Get masks that correspond to new batch sizes
        new_emitter_masks = self.get_emitter_masks(new_emitter_batch_sizes)
    
        bandit_state = DynamicBanditMultiEmitterState(
            emitter_added_counts_buffer = new_emitter_added_counts_buffer,
            emitter_batch_sizes_buffer= new_batch_sizes_buffer,
            emitter_states = new_emitter_states,
            emitter_batch_sizes = new_emitter_batch_sizes,
            emitter_masks = new_emitter_masks,
        )

        return bandit_state


    @partial(jax.jit, static_argnames=("self",))     
    def calculate_emitter_scores(
        self,
        offspring_buffer: ScoresBuffer,
        added_counts_buffer: ScoresBuffer,
    ) -> jnp.array:

        # Find total offspring of all emitters
        new_emitter_total_offspring  = offspring_buffer.find_total_score()
        
        # Update average reward of emitter
        emitter_rewards = added_counts_buffer.data / offspring_buffer.data

        # calculate new emitter average rewards 
        new_emitter_average_rewards = jnp.nanmean(emitter_rewards, axis=0)

        # calculate uncertainty term of bandit score
        uncertainty_terms =  self.bandit_scaling_param * jnp.sqrt(jnp.log(jnp.sum(new_emitter_total_offspring))/new_emitter_total_offspring)

        # Calculate emitter success scores
        emitter_bandit_scores = new_emitter_average_rewards + uncertainty_terms

        return  emitter_bandit_scores


