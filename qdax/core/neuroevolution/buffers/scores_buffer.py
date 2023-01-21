from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from qdax.types import Action, Done, Observation, Reward, RNGKey, StateDescriptor




class ScoresBuffer(flax.struct.PyTreeNode):

    data: jnp.ndarray
    buffer_size: int = flax.struct.field(pytree_node=False)
    current_position: jnp.ndarray = flax.struct.field()
    current_size: jnp.ndarray = flax.struct.field()

    @classmethod
    def init(
        cls,
        buffer_size: int,
        num_emitters: int,
    ) -> ScoresBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.

        Args:
            buffer_size: the size of the replay buffer, e.g. 1e6
            transition: a transition object (might be a dummy one) to get
                the dimensions right
        """

        data = jnp.ones((buffer_size, num_emitters)) * jnp.nan
        current_size = jnp.array(0, dtype=int)
        current_position = jnp.array(0, dtype=int)
        return cls(
            data=data,
            current_size=current_size,
            current_position=current_position,
            buffer_size=buffer_size,
        )

    @jax.jit
    def insert(self, scores: jnp.array) -> ScoresBuffer:
        """
        Insert a new score per emitter
        """
        max_replay_size = self.buffer_size

        new_current_position = self.current_position + 1

        new_current_size = jnp.minimum(
            self.current_size + 1, max_replay_size
        )

        new_data = jax.lax.dynamic_update_slice_in_dim(
            self.data,
            jnp.array(jnp.expand_dims(scores, axis=0), dtype=float),
            start_index=self.current_position % max_replay_size,
            axis=0,
        )

        scores_buffer = self.replace(
            current_position=new_current_position,
            current_size=new_current_size,
            data=new_data,
        )

        return scores_buffer  # type: ignore
    

    def find_average_score(self,
        )-> jnp.array:

        average = jnp.nanmean(self.data, axis=0)

        return average