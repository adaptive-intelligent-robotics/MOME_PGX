from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.mome import MOME
from qdax.core.containers.biased_sampling_mome_repertoire import BiasedSamplingMOMERepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Centroid, RNGKey


class BanditMultiEmitterMOME(MOME):
    """Implements Multi-Objectives MAP Elites.

    Note: most functions are inherited from MAPElites. The only function
    that had to be overwritten is the init function as it has to take
    into account the specificities of the the Multi Objective repertoire.
    """
    

    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length"))
    def init(
        self,
        init_genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], RNGKey]:
        """Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            init_genotypes: genotypes of the initial population.
            centroids: centroids of the repertoire.
            pareto_front_max_length: maximum size of the pareto front. This is
                necessary to respect jax.jit fixed shape size constraint.
            random_key: a random key to handle stochasticity.

        Returns:
            The initial repertoire and emitter state, and a new random key.
        """

        # first score
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        if self._bias_sampling:
            repertoire, container_addition_metrics = BiasedSamplingMOMERepertoire.init(
                genotypes=init_genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                pareto_front_max_length=pareto_front_max_length,
            )

        else:
            repertoire, container_addition_metrics = MOMERepertoire.init(
                genotypes=init_genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                pareto_front_max_length=pareto_front_max_length,
            )
            
        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update the metrics
        # HAS TO BE BEFORE EMITTER STATE UPDATE AS BATCH INDEXES WILL CHANGE
        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(
            container_addition_metrics,
            emitter_state.emitter_batch_sizes,
            emitter_state.emitter_masks,
            metrics)

        # update emitter state
        emitter_state = self._emitter.update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            emitters_added_list=container_addition_metrics[0],
        )
        

        return repertoire, metrics, emitter_state, random_key


    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MOMERepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire, container_addition_metrics = repertoire.add(genotypes, descriptors, fitnesses)
        
        # update the metrics
        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(
            container_addition_metrics,
            emitter_state.emitter_batch_sizes,
            emitter_state.emitter_masks,
            metrics)

        # update emitter state after scoring is made
        emitter_state = self._emitter.update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            emitters_added_list=container_addition_metrics[0],
        )


        return repertoire, emitter_state, metrics, random_key
