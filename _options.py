import os
import yaml
import warnings

from torch.optim.adam import Adam
from torch.optim import SGD


# MobilityCFmMIMOEnv-delta_cf_se-v0
# MobilityCFmMIMOEnv-exp_delta_cf_se_clip-v0
# MobilityCFmMIMOEnv-exp_delta_cf_se_clip-v1
# MobilityCFmMIMOEnv-log_delta_cf_se-v0
# MobilityCFmMIMOEnv-relative_cf_se-v0
# MobilityCFmMIMOEnv-exp_relative_cf_se_clip-v0
# MobilityCFmMIMOEnv-log_relative_cf_se-v0
# MobilityCFmMIMOEnv-delta_sinr-v0
# MobilityCFmMIMOEnv-exp_delta_sinr_clip-v0
# MobilityCFmMIMOEnv-log_delta_sinr-v0
# MobilityCFmMIMOEnv-relative_sinr-v0
# MobilityCFmMIMOEnv-exp_relative_sinr_clip-v0
# MobilityCFmMIMOEnv-log_relative_sinr-v0


def select_reward_option():
    # List of available reward options
    reward_options = {
        "1": "Channel Capacity",
        "2": "Geometric Mean SE",
        "3": "Mean SE",
        "4": "Min SE",
        "5": "Sum SE",
        "6": "Delta CF SE",
        "7": "Exp Delta CF SE Clip",
        "8": "Exp Delta CF SE Clip V1",
        "9": "Log Delta CF SE",
        "10": "Relative CF SE",
        "11": "Exp Relative CF SE Clip",
        "12": "Log Relative CF SE",
        "13": "Delta SINR",
        "14": "Exp Delta SINR Clip",
        "15": "Log Delta SINR",
        "16": "Relative SINR",
        "17": "Exp Relative SINR Clip",
        "18": "Log Relative SINR",
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
        "8": ("MobilityCFmMIMOEnv-exp_delta_cf_se_clip-v1", None, "exp_delta_clip", "cf_se"),
        "9": ("MobilityCFmMIMOEnv-log_delta_cf_se-v0", None, "log_delta", "cf_se"),
        "10": ("MobilityCFmMIMOEnv-relative_cf_se-v0", None, "relative", "cf_se"),
        "11": ("MobilityCFmMIMOEnv-exp_relative_cf_se_clip-v0", None, "exp_relative_clip", "cf_se"),
        "12": ("MobilityCFmMIMOEnv-log_relative_cf_se-v0", None, "log_relative", "cf_se"),
        "13": ("MobilityCFmMIMOEnv-delta_sinr-v0", None, "delta", "sinr"),
        "14": ("MobilityCFmMIMOEnv-exp_delta_sinr_clip-v0", None, "exp_delta_clip", "sinr"),
        "15": ("MobilityCFmMIMOEnv-log_delta_sinr-v0", None, "log_delta", "sinr"),
        "16": ("MobilityCFmMIMOEnv-relative_sinr-v0", None, "relative", "sinr"),
        "17": ("MobilityCFmMIMOEnv-exp_relative_sinr_clip-v0", None, "exp_relative_clip", "sinr"),
        "18": ("MobilityCFmMIMOEnv-log_relative_sinr-v0", None, "log_relative", "sinr"),
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
def load_hyperparameters(yaml_file, env_reward):
    _hyperparams = dict()
    with open(yaml_file) as f:
        _hyperparams.update(yaml.safe_load(f).get(env_reward, {}))
    return _hyperparams


def get_config():
    # Initialize an empty config dictionary
    config = dict()

    # Get user input for algorithm
    config["algo"] = input("Enter the algorithm [currently only SAC is supported](e.g., 'SAC'): ")
    config["algo"] = 'SAC'
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


def get_hyperparameters(config):
    if config["optimizer_class"] is Adam:
        yaml_file = os.path.join('hyperparameters', 'SAC_Adam.yaml')
    elif config["optimizer_class"] is SGD:
        yaml_file = os.path.join('hyperparameters', 'SAC_SGD.yaml')
    else:
        warnings.warn("Pre-tuned hyperparameters doesn't for the selected optimizer. Model will use SGD optimizer.")
        config["optimizer_class"] = SGD
        yaml_file = os.path.join('hyperparameters', 'SAC_SGD.yaml')
    return yaml_file
