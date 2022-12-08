from functools import partial
from typing import Callable, Optional, Tuple, List

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Genotype, RNGKey, Metrics


class MixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.
        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.
        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key
        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

    @partial(jax.jit, static_argnames=("self",))   
    def update_added_counts(
        self,
        container_addition_metrics: List,
        metrics: Metrics,
    ):

        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        added_list = container_addition_metrics[0]
        removed_list = container_addition_metrics[1]

        metrics["removed_count"] = jnp.sum(removed_list)

        variation_added_list = added_list[:n_variation]
        mutation_added_list = added_list[n_variation+1:]

        metrics[f'emitter_variation_count:'] = jnp.sum(variation_added_list)
        metrics[f'emitter_mutation_count:'] = jnp.sum(mutation_added_list)

        return metrics