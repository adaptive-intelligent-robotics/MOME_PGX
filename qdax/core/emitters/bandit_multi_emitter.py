from functools import partial
from typing import Optional, Tuple, List, Any

import jax
import numpy as np
from chex import ArrayTree
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.neuroevolution.buffers.scores_buffer import ScoresBuffer
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey, Mask


class BanditMultiEmitterState(EmitterState):
    """contains state for bandit"""
    emitter_added_counts_buffer: ScoresBuffer
    emitter_batch_sizes_buffer: ScoresBuffer
    emitter_states: Tuple[EmitterState, ...]
    emitter_batch_sizes: jnp.ndarray
    emitter_masks: Tuple[Mask, ...]


class BanditMultiEmitter(MultiEmitter):
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

        bandit_state = BanditMultiEmitterState(
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
        emitter_state: Optional[BanditMultiEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
        emitters_added_list: Metrics = None,
        selection_timestep: int = 1,
    ) -> Optional[BanditMultiEmitterState]:


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
    
        bandit_state = BanditMultiEmitterState(
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

    
    @partial(jax.jit, static_argnames=("self",))
    def get_emitter_masks(
        self,
        emitter_batch_sizes
    )-> Tuple[Mask]:

        # Create masks for each emitter that correspond to batch size of each

        batch_sizes = jnp.repeat(jnp.expand_dims(jnp.arange(1, self.total_batch_size+1), axis=0), self.num_emitters, axis=0)
        cumulative_batches = jnp.cumsum(emitter_batch_sizes)
        
        cumulative_mask = batch_sizes <= cumulative_batches.reshape(self.num_emitters, 1)
        shifted_mask = jnp.concatenate(
                            (jnp.expand_dims(jnp.zeros(self.total_batch_size), axis=0),
                            cumulative_mask[:-1]), 
                            axis=0
        )
        emitter_masks = jnp.logical_and(cumulative_mask, jnp.logical_not(shifted_mask))

        return jnp.array(emitter_masks, dtype=int)

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[Repertoire],
        emitter_state: Optional[BanditMultiEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.
        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            random_key: key for random operations.
        Returns:
            Offsprings and a new random key.
        """
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == self.num_emitters

        # prepare subkeys for each sub emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, self.num_emitters)

        # emit from all emitters and gather offsprings
        all_offsprings = []

        for emitter, sub_emitter_state, subkey_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            subkeys,
        ):
            genotype, _ = emitter.emit(repertoire, sub_emitter_state, subkey_emitter)
            batch_size = jax.tree_util.tree_leaves(genotype)[0].shape[0]
            assert batch_size == self.total_batch_size # All emitters should emit a total batch size
            
            # Add too many solutions for each emitter (and select desired ones later)
            all_offsprings.append(genotype)
        
        # concatenate all emitted offsprings together
        concatenated_offspring = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offsprings
        )

        # put all emitter masks into one mask     
        all_offspring_mask = jnp.hstack(emitter_state.emitter_masks)

        # Find indices of all offspring that correspond to batch for each emitter
        offspring_indices = jnp.where(
            jnp.array(all_offspring_mask, dtype=bool), 
            size=self.total_batch_size
        )

        # Select desired offspring from all offspring
        offspring = jax.tree_util.tree_map(
            lambda x: x.at[offspring_indices].get(),
            concatenated_offspring,
            )

        return offspring, random_key
    

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: Optional[BanditMultiEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[EmitterState, ...]:
        """Update emitter state by updating all sub emitter states.
        Args:
            emitter_state: current emitter state.
            repertoire: current repertoire of genotypes. Defaults to None.
            genotypes: proposed genotypes. Defaults to None.
            fitnesses: associated fitnesses. Defaults to None.
            descriptors: associated descriptors. Defaults to None.
            extra_scores: associated extra_scores. Defaults to None.
        Returns:
            The updated global emitter state.
        """
        if emitter_state is None:
            return None

        # update all the sub emitter states
        emitter_states = []

        def _get_sub_pytree(pytree: ArrayTree, start: int, end: int) -> ArrayTree:
            return jax.tree_util.tree_map(lambda x: x[start:end], pytree)

        for emitter, sub_emitter_state in zip(
            self.emitters,
            emitter_state.emitter_states,
        ):
        # update with all genotypes, fitnesses, etc...
            new_sub_emitter_state = emitter.state_update(
                sub_emitter_state,
                repertoire,
                genotypes,
                fitnesses,
                descriptors,
                extra_scores,
            )
            emitter_states.append(new_sub_emitter_state)
            # update only with the data of the emitted genotypes

        # return the update global emitter state
        return tuple(emitter_states)

    @partial(jax.jit, static_argnames=("self",))   
    def update_batch_sizes_from_scores(self, 
        emitter_bandit_scores: jnp.ndarray,
    )-> jnp.ndarray:

        # normalise bandit scores so that they add to 1
        normalised_bandit_scores = emitter_bandit_scores/jnp.sum(emitter_bandit_scores)

        # calculate batch size based on bandit scores
        new_batch_sizes =  jnp.array(normalised_bandit_scores * self.total_batch_size, dtype=int)
        
        # make sure batch sizes add up
        final_batch_size = self.total_batch_size - jnp.sum(new_batch_sizes[:-1])
        new_batch_sizes = new_batch_sizes.at[-1].set(final_batch_size)

        return jnp.array(new_batch_sizes, dtype=int)
        

    @partial(jax.jit, static_argnames=("self",))   
    def update_added_counts(
        self,
        container_addition_metrics: List,
        emitter_batch_sizes: List,
        emitter_masks: Mask,
        metrics: Metrics,
    ):

        added_list = container_addition_metrics[0]
        removed_list = container_addition_metrics[1]

        metrics["removed_count"] = jnp.sum(removed_list)

        all_emitter_added_counts = emitter_masks * added_list

        # Find rewards of each emitter
        emitter_counts = jnp.sum(all_emitter_added_counts, axis=-1)
    
        for emitter_index in range(self.num_emitters):
            # store total tried emissions for emitter
            metrics[f'emitter_{emitter_index+1}_emitted_count:'] = emitter_batch_sizes[emitter_index]
            # store total successful emissions for emitter
            metrics[f'emitter_{emitter_index+1}_added_count:'] = emitter_counts[emitter_index]

        return metrics