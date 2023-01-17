import numpy as np

def get_env_metrics(env_name,
    episode_length: int=1000):

    # Reward 1 = forward reward 
    # Reward 2 = energy cost 

    env_metrics = {}

    if env_name == "walker2d_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        env_metrics["min_rewards"] = [-200, -15]
        env_metrics["max_rewards"] = [2000, 0]

    elif env_name == "ant_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        env_metrics["min_rewards"] = [0, -15]
        env_metrics["max_rewards"] = [2000, 0]
    
    
    elif env_name == "hopper_multi":

        # Empirically inferred min and max rewards for episode_length = 1000
        env_metrics["min_rewards"] = [0, -15]
        env_metrics["max_rewards"] = [2000, 0]
    
    elif env_name == "halfcheetah_multi":
        # Empirically inferred min and max rewards for episode_length = 1000
        env_metrics["min_rewards"] = [-1500, -1000]
        env_metrics["max_rewards"] = [2000, 0]

    elif env_name == "humanoid_multi":
        # Empirically inferred min and max rewards for episode_length = 1000
        env_metrics["min_rewards"] = [0, 200]
        env_metrics["max_rewards"] = [2000, 0]
    
    # Multiply min and max rewards by number of timesteps
    env_metrics["min_rewards"] = np.array(env_metrics["min_rewards"]) * episode_length/1000
    env_metrics["max_rewards"] = np.array(env_metrics["max_rewards"]) * episode_length/1000

    return env_metrics