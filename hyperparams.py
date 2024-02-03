import optuna
import numpy as np
from typing import Tuple, Any, Dict

from torch.optim.adam import Adam
from torch.optim import SGD
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def sample_sac_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 10, 100, 1000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big", 'mixed', 'large'])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "mixed": [128, 256, 128],
        "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch_type]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_float('target_entropy', -10, 10)

    # optimizer_class = trial.suggest_categorical("optimizer_class", ["Adam", "SGD"])
    #
    # OPTIMIZER = {
    #     "Adam": Adam,
    #     "SGD": SGD,
    # }[optimizer_class]
    #
    OPTIMIZER = SGD

    # q_window = trial.suggest_categorical("q_window", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q_window = 10
    # r_alpha = trial.suggest_float("r_alpha", 1, 10)
    # r_beta = trial.suggest_float("r_beta", 1, 10)
    r_alpha = 1
    r_beta = 1

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(optimizer_class=OPTIMIZER, net_arch=net_arch),
    }

    extra_hyperparams = {
        "temporal_window_size": q_window,
        "r_alpha": r_alpha,
        "r_beta": r_beta,
    }

    return hyperparams, extra_hyperparams


def sample_ddpg_params(trial: optuna.Trial, n_actions: int = 32) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Sampler for DDPG hyperparams.

    :param trial:
    :param n_actions:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_float("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big", 'mixed', 'large'])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "mixed": [128, 256, 128],
        "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch_type]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    # q_window = trial.suggest_categorical("q_window", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q_window = 10

    extra_hyperparams = {
        "temporal_window_size": q_window,
        # "r_alpha": r_alpha,
        # "r_beta": r_beta,
    }

    return hyperparams, extra_hyperparams


HYPERPARAMS_SAMPLER = {
    "SAC": sample_sac_params,
    "DDPG": sample_ddpg_params,
}
