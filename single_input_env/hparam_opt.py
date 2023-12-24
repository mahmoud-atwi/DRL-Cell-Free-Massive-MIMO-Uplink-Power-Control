import os
import pickle as pkl
import sys
import threading
import time
from pathlib import Path
from pprint import pprint

import gymnasium as gym
import optuna
import torch
from gymnasium.envs.registration import register
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import cuda
from torch import optim
from torch.backends import mps

from algos import ALGOS
from hyperparams import HYPERPARAMS_SAMPLER
from simulation_para import square_length, L, K
from utils import TrialEvalCallback, check_and_launch_dashboard

if sys.platform == "darwin":  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:  # For other operating systems
    device = "cuda" if cuda.is_available() else "cpu"

if device == "mps":
    _device = "cpu"
else:
    _device = device

log_directory = Path("logs")
log_directory.mkdir(parents=True, exist_ok=True)

register(id=f'Mobility/CFmMIMOEnv-v0', entry_point="mobility_env:MobilityCFmMIMOEnv", )

AP_locations = torch.rand(L, dtype=torch.complex64, device=_device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=_device) * square_length

config = {
    "algo": "SAC",
    "policy_type": "MlpPolicy",
    "total_timesteps": 10000,
    "env_id": "CFmMIMOEnv/Mobility-v0",
    "env_name": 'MobilityCFmMIMOEnv',
    "optimizer_class": optim.SGD,
}

optuna_study_name = "Mobility_CF-mMIMO"
storage_url = f"sqlite:///{optuna_study_name}.sqlite3"

register(id=config["env_id"], entry_point="mobility_env:MobilityCFmMIMOEnv", )

environment_kwargs = {"APs_positions": AP_locations, "UEs_positions": UE_initial_locations, "UEs_mobility": True}

_hyperparams = dict()


def create_sampler(sampler_method: str, seed: int) -> BaseSampler:
    if sampler_method == "random":
        sampler: BaseSampler = RandomSampler(seed=seed)
    elif sampler_method == "tpe":
        sampler = TPESampler(n_startup_trials=10, seed=seed, multivariate=True)
    elif sampler_method == "skopt":
        from optuna.integration.skopt import SkoptSampler
        sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"Unknown sampler: {sampler_method}")
    return sampler


def create_pruner(pruner_method: str, n_startup_trials: int, n_evaluations: int) -> BasePruner:
    if pruner_method == "halving":
        pruner: BasePruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == "median":
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == "none":
        pruner = NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_method}")
    return pruner


def create_envs(env_id: str, n_envs: int, env_kwargs: dict, eval_env_kwargs: dict, vec_env_class, seed: int,
                save_path: str, eval_env=False, no_log=False):
    log_dir = None if eval_env or no_log else save_path
    spec = gym.spec(env_id)

    def make_env(**kwargs) -> gym.Env:
        return spec.make(**kwargs)

    actual_env_kwargs = eval_env_kwargs if eval_env else env_kwargs

    env = make_vec_env(make_env, n_envs=n_envs, seed=seed, env_kwargs=actual_env_kwargs, monitor_dir=log_dir,
                       vec_env_cls=vec_env_class, )

    return env


def objective(trial, algo, env_id, n_envs, env_kwargs, eval_env_kwargs, n_eval_episodes, n_timesteps, n_eval_envs,
              deterministic_eval, optimization_log_path, verbose, seed, save_path, no_log):
    kwargs = dict()
    sampled_hyperparams = HYPERPARAMS_SAMPLER[algo](trial)
    kwargs.update(sampled_hyperparams)
    env = create_envs(env_id, n_envs, env_kwargs, eval_env_kwargs, DummyVecEnv, seed, save_path, no_log)
    trial_verbosity = 0
    if verbose >= 2:
        trial_verbosity = verbose

    model = ALGOS[algo](policy=config["policy_type"], env=env, seed=None,  # Not seeding the trial
                        verbose=trial_verbosity, device=device, **kwargs, )

    eval_env = create_envs(env_id, n_envs=n_eval_envs, env_kwargs=env_kwargs, eval_env_kwargs=eval_env_kwargs,
                           vec_env_class=DummyVecEnv, seed=seed, save_path=save_path, eval_env=True)
    optuna_eval_freq = max(int(n_timesteps / n_eval_episodes) // n_envs, 1)

    path = os.path.join(optimization_log_path, f"trial_{trial.number}") if optimization_log_path else None
    callbacks = [
        TrialEvalCallback(eval_env, trial, best_model_save_path=path, log_path=path, n_eval_episodes=n_eval_episodes,
                          eval_freq=optuna_eval_freq, deterministic=deterministic_eval)]

    learn_kwargs = {}

    try:
        model.learn(n_timesteps, callback=callbacks, **learn_kwargs)
        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
        model.env.close()
        eval_env.close()
        print(e)
        print("Sampled hyperparams:")
        pprint(sampled_hyperparams)
        raise optuna.exceptions.TrialPruned() from e

    is_pruned = callbacks[0].is_pruned
    reward = callbacks[0].last_mean_reward

    del model.env, eval_env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


def hyperparameters_optimization(study_name, sampler_method, pruner_method, max_total_trials, algo, env_id, env_kwargs,
                                 eval_env_kwargs, n_timesteps, n_trials, n_envs, n_eval_envs, deterministic_eval,
                                 optimization_log_path, verbose, n_eval_episodes, save_path):
    seed = 0  # Define the seed if required
    sampler = create_sampler(sampler_method, seed)
    pruner = create_pruner(pruner_method, n_startup_trials=10, n_evaluations=1)

    study = optuna.create_study(
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )

    # Start the check-and-launch thread

    dashboard_thread = threading.Thread(target=check_and_launch_dashboard,
                                        args=(os.path.dirname(os.path.realpath(__file__)), storage_url, 9000))
    dashboard_thread.start()

    try:
        if max_total_trials is not None:
            counted_states = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.PRUNED]
            completed_trials = len(study.get_trials(states=counted_states))
            print(f"Running optimization trial {completed_trials + 1}/{max_total_trials}")
            if completed_trials < max_total_trials:
                study.optimize(
                    lambda _trial: objective(_trial, algo, env_id, n_envs, env_kwargs, eval_env_kwargs, n_eval_episodes,
                                             n_timesteps, n_eval_envs, deterministic_eval, optimization_log_path,
                                             verbose, seed=seed, save_path=save_path, no_log=True), n_jobs=1,
                    callbacks=[MaxTrialsCallback(max_total_trials, states=counted_states)], )
        else:
            print(f"Running optimization with a maximum of {n_trials} trials.")
            study.optimize(
                lambda _trial: objective(_trial, algo, env_id, n_envs, env_kwargs, eval_env_kwargs, n_eval_episodes,
                                         n_timesteps, n_eval_envs, deterministic_eval, optimization_log_path, verbose,
                                         seed=seed, save_path=save_path, no_log=True), n_jobs=1, n_trials=n_trials, )

    except KeyboardInterrupt:
        pass
    finally:
        dashboard_thread.join()

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    report_name = \
        f"report_{study_name}_{n_trials}-trials-{n_timesteps}-{sampler_method}-{pruner_method}_{int(time.time())}"
    log_path = os.path.join(save_path, report_name)

    if verbose:
        print(f"Writing report to {log_path}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    study.trials_dataframe().to_csv(f"{log_path}.csv")

    with open(f"{log_path}.pkl", "wb+") as f:
        pkl.dump(study, f)


if __name__ == "__main__":
    # Hyperparameter Tuning Configuration
    _study_name = "Mobility_CF-mMIMO"
    _sampler_method = "tpe"  # Options: "random", "tpe", "skopt"
    _pruner_method = "median"  # Options: "halving", "median", "none"
    _max_total_trials = 100  # Maximum number of trials
    _n_trials = 50  # Number of trials if max_total_trials is None
    _n_envs = 1  # Number of environments
    _n_eval_envs = 1  # Number of evaluation environments
    _n_timesteps = config["total_timesteps"]  # Total timesteps for each trial
    _n_eval_episodes = 20  # Number of evaluation episodes
    _optimization_log_path = "optuna_logs"  # Directory for optimization logs
    _verbose = 3  # Verbosity level
    _save_path = str(log_directory)  # Directory for saving logs

    # Start Hyperparameter Tuning
    hyperparameters_optimization(
        study_name=_study_name,
        sampler_method=_sampler_method,
        pruner_method=_pruner_method,
        max_total_trials=_max_total_trials,
        algo=config["algo"],
        env_id=config["env_id"],
        env_kwargs=environment_kwargs,
        eval_env_kwargs=environment_kwargs,
        n_timesteps=_n_timesteps,
        n_trials=_n_trials,
        n_envs=_n_envs,
        n_eval_envs=_n_eval_envs,
        deterministic_eval=True,
        optimization_log_path=_optimization_log_path,
        verbose=_verbose,
        n_eval_episodes=_n_eval_episodes,
        save_path=_save_path
    )
