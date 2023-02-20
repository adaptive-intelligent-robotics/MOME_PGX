import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd
from functools import partial

from typing import List, Dict, Tuple
from qdax.utils.pareto_front import compute_pareto_dominance, compute_masked_pareto_front




def print_pairwise_coverage(parent_dirname: str,
    env_names: List[str],
    experiment_names: List[str],
) -> None:
    print("\n")
    print("---------------------------------------------------------------------------------")
    print("Calculating pairwise coverage of PF points for each env, for each experiment")
    print("Table scores: proportion of points in row global pf dominated by col global pf")
    print("---------------------------------------------------------------------------------")

    # Calculate coverage scores for each environment
    for env in env_names:

        print("\n")
        print(f"     ENV: {env}             ")

        env_dirname = os.path.join(parent_dirname, f"{env}/")
        _analysis_dir = os.path.join(env_dirname, "analysis/")

        # Calculate coverage scores for each experiment
        env_pareto_fronts = []

        for experiment in experiment_names:
            #print(f"------- Finding global PF for: {experiment} --------")
            global_pareto_fronts = get_pfs(env_dirname, experiment)
            env_pareto_fronts.append(global_pareto_fronts)
        
        median_coverages = calculate_pairwise_coverage(env_pareto_fronts, experiment_names)
        median_coverages = median_coverages.replace(0, np.NaN)
        median_coverages['mean'] = median_coverages.mean(axis = 1, skipna = True)
        median_coverages.loc['mean'] = median_coverages.mean(axis = 0, skipna = True)

        median_coverages.to_csv(f"{_analysis_dir}/median_pairwise_coverages.csv")
        
        print(median_coverages)


def get_pfs(
    dirname: str, 
    experiment_name: str
)-> pd.DataFrame:
    """
    Load in fitnesses of experiments and find global Pareto fronts

    """

    global_pareto_fronts = []

    for experiment_replication in os.scandir(os.path.join(dirname, experiment_name)):
        fitnesses = jnp.load(os.path.join(experiment_replication, "final/repertoire/fitnesses.npy"))
        global_pareto_front = get_global_pareto_front(fitnesses)
        global_pareto_fronts.append(global_pareto_front)
    
    return global_pareto_fronts


def calculate_pairwise_coverage(
    global_pareto_fronts: List[jnp.ndarray],
    experiment_names: List[str]
):
    """
    Find median pairwise coverage across replications
    """
    num_replications = len(global_pareto_fronts[0])

    metrics = []
    for replication in range(num_replications):
        rep_global_pfs = []
        for exp_num in range(len(experiment_names)):
            exp_rep_global_pf = global_pareto_fronts[exp_num][replication]
            rep_global_pfs.append(exp_rep_global_pf)
        rep_df = calculate_rep_coverage(rep_global_pfs, experiment_names)
        metrics.append(rep_df)

    metrics_concat = pd.concat(metrics)
    metrics = metrics_concat.groupby(metrics_concat.index)
    median_metrics = metrics.median()
    return median_metrics


def calculate_rep_coverage(
    rep_global_pfs: List[jnp.array],
    experiment_names: List[str],
)-> pd.DataFrame:
    """
    Calculate pairwise coverage for given replication of experiments
    """
    ## Table scores: proportion of points in row global pf dominated by col global pf
    rep_df = pd.DataFrame(columns=experiment_names, index=experiment_names)
    for exp1_num, exp1 in enumerate(experiment_names):
        exp1_dominated_list = []
        for exp2_num, exp2 in enumerate(experiment_names):
            partial_dominance_fn = partial(compute_pareto_dominance, batch_of_criteria=rep_global_pfs[exp2_num])
            exp1_dominated = jax.vmap(partial_dominance_fn)(rep_global_pfs[exp1_num])
            exp1_dominated_proportion = sum(exp1_dominated)/len(exp1_dominated)
            exp1_dominated_list.append(exp1_dominated_proportion)
        rep_df.loc[exp1] = exp1_dominated_list
    return rep_df
    

def get_global_pareto_front(
    fitnesses: jnp.array
):
    """
    Find global pareto front from final fitnesses

    """

    fitnesses= jnp.concatenate(fitnesses, axis=0)
    mask = jnp.any(fitnesses == -jnp.inf, axis=-1)
    pareto_mask = compute_masked_pareto_front(fitnesses, mask)
    pareto_indices = jnp.argwhere(pareto_mask).squeeze()
    pareto_front = jnp.take(fitnesses, pareto_indices, axis=0)
    
    return pareto_front
