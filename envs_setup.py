import numpy as np

def get_env_metrics(env_name,
    episode_length: int=1000):

    # Reward 1 = forward reward 
    # Reward 2 = energy cost 

    env_metrics = {}

    if env_name == "walker2d_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        """
        MIN SCORES ACROSS ALL EXPERIMENTS: [-205.34937, -5.514009]
        MAX SCORES ACROSS ALL EXPERIMENTS: [2186.4878, -1.3155421e-05]
        """
        env_metrics["min_rewards"] = [-210, -15]
        env_metrics["max_rewards"] = [2500, 0]

    elif env_name == "ant_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        """
        MIN SCORES ACROSS ALL EXPERIMENTS: [-307.27118, -3997.8]
        MAX SCORES ACROSS ALL EXPERIMENTS: [3339.4678, -0.2654624]
        """
        env_metrics["min_rewards"] = [-350, -4500]
        env_metrics["max_rewards"] = [4000, 0]
    
    
    elif env_name == "hopper_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        """
        MIN SCORES ACROSS ALL EXPERIMENTS: [-35.17073, -1.3313142]
        MAX SCORES ACROSS ALL EXPERIMENTS: [537.24084, -2.264291e-06]
        """
        env_metrics["min_rewards"] = [-50, -2]
        env_metrics["max_rewards"] = [800, 0]
    
    elif env_name == "halfcheetah_multi":
        # Empirically inferred min and max rewards for episode_length = 1000
        """
        MIN SCORES ACROSS ALL EXPERIMENTS: [-1591.811, -599.96326]
        MAX SCORES ACROSS ALL EXPERIMENTS: [6811.9365, -0.017732896]
        """
        env_metrics["min_rewards"] = [-2000, -800]
        env_metrics["max_rewards"] = [7000, 0]

    elif env_name == "humanoid_multi":
        # Empirically inferred min and max rewards for episode_length = 1000
        """
        MIN SCORES ACROSS ALL EXPERIMENTS: [-52.85332, -327.06708]
        MAX SCORES ACROSS ALL EXPERIMENTS: [365.48199, -0.023004716]
        """
        env_metrics["min_rewards"] = [-100, -500]
        env_metrics["max_rewards"] = [500, 0]
    
    # Multiply min and max rewards by number of timesteps
    env_metrics["min_rewards"] = np.array(env_metrics["min_rewards"]) * episode_length/1000
    env_metrics["max_rewards"] = np.array(env_metrics["max_rewards"]) * episode_length/1000

    return env_metrics