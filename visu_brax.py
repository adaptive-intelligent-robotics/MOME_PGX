import jax.numpy as jnp
import jax
import os

from brax.io import html
from brax.io.file import File
from brax.io.json import dumps
from IPython.display import HTML
from qdax.types import RNGKey
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire


def save_samples(
    env,
    policy_network,
    random_key: RNGKey,
    repertoire: MapElitesRepertoire,
    num_save_visualisations: int,
    iteration: str="final",
    save_dir: str="./",
):
    """ Select best individual and some random individuals from repertoire and visualise behaviour"""
    number_individuals = len(repertoire.fitnesses)

    # Visualise the best individual
    best_idx = jnp.argmax(repertoire.fitnesses)

    params = jax.tree_util.tree_map(
        lambda x: x[best_idx],
        repertoire.genotypes
        )
    
    visualise_individual(
        env,
        policy_network,
        params,
        f"best_iteration_{iteration}_individual_{best_idx}.html",
        save_dir
    )

    # Visualise somne random individuals
    random_indices = jax.random.randint(
            random_key,
            shape=(num_save_visualisations, ),
            minval=0,
            maxval=number_individuals
    )

    for index in random_indices:
        params = jax.tree_util.tree_map(
            lambda x: x[index],
            repertoire.genotypes
        )

        visualise_individual(
            env,
            policy_network,
            params,
            f"iteration_{iteration}_individual_{index}.html",
            save_dir
        )
        


def visualise_individual(
    env,
    policy_network,
    params,
    name,
    save_dir,
):
    """ Roll out individual policy and save visualisation"""
    path = os.path.join(save_dir, name)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    while not state.done:
        rollout.append(state)
        action = jit_inference_fn(params, state.obs)
        state = jit_env_step(state, action)

    with File(path, 'w') as fout:
        fout.write(html.render(env.sys, [s.qp for s in rollout], height=480))
