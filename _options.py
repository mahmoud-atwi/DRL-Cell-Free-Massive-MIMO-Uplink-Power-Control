import os
import yaml
import warnings

from torch.optim.adam import Adam
from torch.optim import SGD

from typing import Any, Dict, Optional, Tuple

# MobilityCFmMIMOEnv-delta_cf_se-v0
# MobilityCFmMIMOEnv-exp_delta_cf_se_clip-v0
# MobilityCFmMIMOEnv-log_delta_cf_se-v0
# MobilityCFmMIMOEnv-relative_cf_se-v0
# MobilityCFmMIMOEnv-exp_relative_cf_se_clip-v0
# MobilityCFmMIMOEnv-log_relative_cf_se-v0
# MobilityCFmMIMOEnv-delta_sinr-v0
# MobilityCFmMIMOEnv-exp_delta_sinr_clip-v0
# MobilityCFmMIMOEnv-log_delta_sinr-v0
# MobilityCFmMIMOEnv-log_delta_sinr-v1
# MobilityCFmMIMOEnv-relative_sinr-v0
# MobilityCFmMIMOEnv-exp_relative_sinr_clip-v0
# MobilityCFmMIMOEnv-log_relative_sinr-v0


def select_reward_option() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    # List of available reward options
    reward_options = {
        "1": "Channel Capacity",
        "2": "Geometric Mean SE",
        "3": "Mean SE",
        "4": "Min SE",
        "5": "Sum SE",
        "6": "Delta CF SE",
        "7": "Exp Delta CF SE Clip",
        "8": "Log Delta CF SE",
        "9": "Relative CF SE",
        "10": "Exp Relative CF SE Clip",
        "11": "Log Relative CF SE",
        "12": "Delta SINR",
        "13": "Exp Delta SINR Clip",
        "14": "Log Delta SINR",
        "15": "Log Delta SINR (int tuned weight)",
        "16": "Log Delta SINR (float tuned weight)",
        "17": "Log Delta SINR (re-tuned with int weight)",
        "18": "Log Delta SINR (re-tuned with float weight)",
        "19": "Relative SINR",
        "20": "Exp Relative SINR Clip",
        "21": "Log Relative SINR",
    }

    # Map of user selection to environment name
    env_map = {
        "1": ("MobilityCFmMIMOEnv-ch_capacity-v0", "channel_capacity", None, None),
        "2": ("MobilityCFmMIMOEnv-geo_mean_se-v0", "geo_mean_se", None, None),
        "3": ("MobilityCFmMIMOEnv-mean_se-v0", "mean_se", None, None),
        "4": ("MobilityCFmMIMOEnv-min_se-v0", "min_se", None, None),
        "5": ("MobilityCFmMIMOEnv-sum_se-v0", "sum_se", None, None),
        "6": ("MobilityCFmMIMOEnv-delta_cf_se-v0", None, "delta", "cf_se"),
        "7": ("MobilityCFmMIMOEnv-exp_delta_cf_se_clip-v0", None, "exp_delta_clip", "cf_se"),
        "8": ("MobilityCFmMIMOEnv-log_delta_cf_se-v0", None, "log_delta", "cf_se"),
        "9": ("MobilityCFmMIMOEnv-relative_cf_se-v0", None, "relative", "cf_se"),
        "10": ("MobilityCFmMIMOEnv-exp_relative_cf_se_clip-v0", None, "exp_relative_clip", "cf_se"),
        "11": ("MobilityCFmMIMOEnv-log_relative_cf_se-v0", None, "log_relative", "cf_se"),
        "12": ("MobilityCFmMIMOEnv-delta_sinr-v0", None, "delta", "sinr"),
        "13": ("MobilityCFmMIMOEnv-exp_delta_sinr_clip-v0", None, "exp_delta_clip", "sinr"),
        "14": ("MobilityCFmMIMOEnv-log_delta_sinr-v0", None, "log_delta", "sinr"),
        "15": ("MobilityCFmMIMOEnv-log_delta_sinr-v1", None, "log_delta", "sinr"),
        "16": ("MobilityCFmMIMOEnv-log_delta_sinr-v2", None, "log_delta", "sinr"),
        "17": ("MobilityCFmMIMOEnv-log_delta_sinr-v3", None, "log_delta", "sinr"),
        "18": ("MobilityCFmMIMOEnv-log_delta_sinr-v4", None, "log_delta", "sinr"),
        "19": ("MobilityCFmMIMOEnv-relative_sinr-v0", None, "relative", "sinr"),
        "20": ("MobilityCFmMIMOEnv-exp_relative_sinr_clip-v0", None, "exp_relative_clip", "sinr"),
        "21": ("MobilityCFmMIMOEnv-log_relative_sinr-v0", None, "log_relative", "sinr"),
    }

    while True:
        # Display options and get user selection
        for key, value in reward_options.items():
            print(f"{key}: {value}")

        selection = input("Select a reward option by entering the corresponding number: ")

        # Check for valid selection
        if selection in env_map:
            return env_map[selection]
        else:
            print("Invalid selection. Try again.")


# Function to load hyperparameters from YAML
def load_hyperparameters(yaml_file: str, env_reward: str) -> Dict[str, Any]:
    """
    Loads hyperparameters from a YAML file for a specified environment reward setting.

    :param yaml_file: The path to the YAML file containing hyperparameters configurations.
    :param env_reward: The key identifying the specific environment reward setting in the YAML file.
    :return: A dictionary of hyperparameters associated with the specified environment reward setting.

    The function uses `yaml.safe_load` to read the YAML file, ensuring safe loading without executing arbitrary code.
    """
    _hyperparams: Dict[str, Any] = dict()
    with open(yaml_file) as f:
        _hyperparams.update(yaml.safe_load(f).get(env_reward, {}))
    return _hyperparams


def get_config() -> Dict[str, Any]:
    # Initialize an empty config dictionary
    config: Dict[str, Any] = dict()

    # Get user input for algorithm
    algo = input("Enter the algorithm [currently only SAC and DDPG are supported](e.g., 'SAC'): ")
    if algo.upper() not in ['SAC', 'DDPG']:
        raise ValueError("Invalid algorithm. Currently only SAC and DDPG are supported.")
    config["algo"] = algo.upper()
    # config["algo"] = 'SAC'
    if config["algo"] not in ['SAC', 'DDPG']:
        raise ValueError("Invalid algorithm. Currently only SAC and DDPG are supported.")
    # Get user input for optimizer class
    optimizer_input = input("Enter the optimizer class ('Adam' or 'SGD'): ")
    if optimizer_input.lower() == 'adam':
        config["optimizer_class"] = Adam
    else:
        config["optimizer_class"] = SGD  # Default to SGD if input is not Adam

    # Get user input for total timesteps
    try:
        config["total_timesteps"] = int(input("Enter the total timesteps (e.g., 10000): "))
    except ValueError:
        print("Invalid input for timesteps. Setting default to 10000.")
        config["total_timesteps"] = 10000  # Default value

    return config


def get_hyperparameters(config: Dict[str, Any]) -> str:
    if config["optimizer_class"] is Adam and config["algo"] == 'SAC':
        yaml_file = os.path.join('hyperparameters', 'SAC_Adam.yaml')
    elif config["optimizer_class"] is SGD and config["algo"] == 'SAC':
        yaml_file = os.path.join('hyperparameters', 'SAC_SGD.yaml')
    elif config["optimizer_class"] is Adam and config["algo"] == 'DDPG':
        yaml_file = os.path.join('hyperparameters', 'DDPG_Adam.yaml')
    elif config["optimizer_class"] is SGD and config["algo"] == 'DDPG':
        yaml_file = os.path.join('hyperparameters', 'DDPG_SGD.yaml')
    else:
        warnings.warn("Pre-tuned hyperparameters doesn't for the selected optimizer. Model will use SGD optimizer.")
        config["optimizer_class"] = SGD
        yaml_file = os.path.join('hyperparameters', 'SAC_SGD.yaml')
    return yaml_file
