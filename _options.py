import os
import yaml
import warnings

from torch.optim.adam import Adam
from torch.optim import SGD


def select_reward_option():
    # List of available reward options
    reward_options = {
        "1": "Channel Capacity",
        "2": "Geometric Mean SE",
        "3": "Mean SE",
        "4": "Min SE",
        "5": "Sum SE"
    }

    # Map of user selection to environment name
    env_map = {
        "1": "MobilityCFmMIMOEnv-ch_capacity-v0",
        "2": "MobilityCFmMIMOEnv-geo_mean_se-v0",
        "3": "MobilityCFmMIMOEnv-mean_se-v0",
        "4": "MobilityCFmMIMOEnv-min_se-v0",
        "5": "MobilityCFmMIMOEnv-sum_se-v0"
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
