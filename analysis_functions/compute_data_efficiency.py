import numpy as np
from typing import List, Any, Dict

def compute_data_efficiency(
    median_metrics: List,
    env_names: List[str],
    experiment_names: List[str],
    data_efficiency_params: Dict,
):

    """
    Calculate data-efficiency of test_algorithms vs baseline_algorithms on certain metrics
    """
    print("\n")
    print("--------------------------------------------------")
    print("    Calculating data-efficiency of algorithms    ")
    print("--------------------------------------------------")

    for test_algo in data_efficiency_params["test_algs"]:
        for baseline_algo in data_efficiency_params["baseline_algs"]:
            baseline_algo_num = int(np.argwhere(np.array(experiment_names)==baseline_algo))
            test_algo_num = int(np.argwhere(np.array(experiment_names)==test_algo))

            for env_num, env_name in enumerate(env_names):

                for comparison_metric in data_efficiency_params["metrics"]:

                    baseline_medians = median_metrics[env_num][baseline_algo_num][comparison_metric]
                    baseline_final_score = np.array(baseline_medians)[-1]
                    test_algo_medians = median_metrics[env_num][test_algo_num][comparison_metric]

                    num_iterations = len(test_algo_medians)
                    overtake_iteration = 0

                    for score in test_algo_medians:
                        if score < baseline_final_score:
                            overtake_iteration += 1

                    print("\n")
                    print(f" ENV:  {env_name}   METRIC: {comparison_metric}")
                    print(f"{test_algo} overtakes {baseline_algo} on iteration {overtake_iteration} out of {num_iterations}")

                    