import sys

import optuna
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm
import threading

from torch import cuda
from torch import optim
from torch.backends import mps

from mobility_env import MobilityCFmMIMOEnv
from simulation_para import square_length

if sys.platform == "darwin":  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:  # For other operating systems
    device = "cuda" if cuda.is_available() else "cpu"

if device == "mps":
    _device = "cpu"
else:
    _device = device

L = 64
K = 32

AP_locations = torch.rand(L, dtype=torch.complex64, device=_device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=_device) * square_length

config = {
    "algo": "SAC",
    "policy_type": "MlpPolicy",
    "total_timesteps": 10000,
    "env_name": 'MobilityCFmMIMOEnv',
    "learning_rate": 5e-4,
    "batch_size": 128,
    "optimizer_class": optim.SGD,
    "net_arch": [128, 256, 128],
}


def optimize_sac(trial):
    """ Objective function for Optuna study. """
    # Define hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 0.01, 0.1, log=True)

    # Create environment
    env = MobilityCFmMIMOEnv(APs_positions=AP_locations, UEs_positions=UE_initial_locations)

    # Initialize SAC model with sampled hyperparameters
    model = SAC(policy=config["policy_type"],
                env=env, learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef,
                verbose=2
                )

    evaluation_interval = 10

    model.learn(evaluation_interval)

    return 1


# Callback to track the progress

def trial_callback(study, trial):
    print(f"Finished trial {trial.number} with value: {trial.value} and parameters: {trial.params}")


if __name__ == '__main__':

    s = optuna.create_study(
        storage='sqlite:///sac.db',
        direction='maximize'
    )

    s.optimize(optimize_sac, n_trials=100, callbacks=[trial_callback], show_progress_bar=True)

    # Print the optimal hyperparameters
    print('Best trial:', s.best_trial.params)
