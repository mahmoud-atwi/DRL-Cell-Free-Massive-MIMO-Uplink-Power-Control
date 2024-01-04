import os
import pickle as pkl
import sys
import threading
import time
from pathlib import Path
from pprint import pprint
from typing import Optional, Type, Union

import gymnasium as gym
import optuna
from gymnasium.envs.registration import register
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import cuda
from torch.backends import mps

from _utils import TrialEvalCallback, check_and_launch_dashboard, generate_ap_locations, generate_ue_locations
from algos import ALGOS
from hyperparams import HYPERPARAMS_SAMPLER
from simulation_para import square_length, L, K


class HyperparameterOptimizer:
    """
    A class to optimize hyperparameters for reinforcement learning algorithms
    using Optuna and Gymnasium environments.

    Attributes:
        device (str): The computing device to be used ('cpu', 'cuda', 'mps').
        env_id (str): Identifier of the Gymnasium environment.
        env_name (str): Name of the environment for logging purposes.
        entry_point (str): Entry point to the environment.
        env_kwargs (dict): Keyword arguments for the environment creation.
        eval_env_kwargs (dict): Keyword arguments for the evaluation environment creation.
        max_episode_steps (int): Maximum number of steps per episode.
        sampler_method (str): Method to sample hyperparameters.
        pruner_method (str): Method to prune trials.
        optimize_direction (str): Direction of optimization ('maximize' or 'minimize').
        deterministic_eval (bool): Whether the evaluation should be deterministic.
        load_if_exists (bool): Whether to load an existing study or create a new one.
        n_startup_trials (int): Number of startup trials for the sampler.
        n_warmup_steps (int): Number of warmup steps for the pruner.
        max_total_trials (Optional[int]): Maximum number of total trials. None for unlimited.
        n_trials (int): Number of trials to run.
        n_envs (int): Number of parallel environments.
        n_eval_envs (int): Number of parallel evaluation environments.
        n_timesteps (int): Number of timesteps for each trial.
        n_evaluations (int): Number of evaluations for each trial.
        n_eval_episodes (int): Number of episodes for evaluation.
        algo (str): Algorithm to be optimized.
        policy_type (str): Type of the policy (e.g., 'MlpPolicy').
        reward_method (Optional[str]): Method of calculating the reward.
        temporal_reward_method (Optional[str]): Method for temporal reward calculation.
        temporal_reward_operation (str): Operation for temporal reward calculation.
        temporal_reward_max (float): Maximum value for temporal reward.
        temporal_data (str): Type of data used for temporal reward calculation.
        temporal_window_size (int): Window size for temporal data calculation.
        seed (int): Random seed.
        study_name (str): Name of the Optuna study.
        storage_url (str): URL for the Optuna study storage.
        open_dashboard (bool): Whether to open the Optuna dashboard.
        dashboard_port (int): Port for the Optuna dashboard.
        verbose (int): Verbosity level.
        save_path (str): Path for saving logs.
        optimization_log_path (str): Path for saving optimization logs.
        no_log (bool): Whether to disable logging.
    """

    def __init__(self, **kwargs):
        self.device = self.set_device()
        self.env_id: str = kwargs.get('env_id')
        self.env_name: str = kwargs.get('env_name')
        self.entry_point: str = kwargs.get('entry_point')
        self.env_kwargs: dict = kwargs.get('env_kwargs')
        self.eval_env_kwargs: dict = kwargs.get('eval_env_kwargs', self.env_kwargs)
        self.max_episode_steps: int = kwargs.get('max_episode_steps', 10)
        self.sampler_method: str = kwargs.get('sampler_method', 'tpe')
        self.pruner_method: str = kwargs.get('pruner_method', 'median')
        self.optimize_direction: str = kwargs.get('optimize_direction', 'maximize')
        self.deterministic_eval: bool = kwargs.get('deterministic_eval', True)
        self.load_if_exists: bool = kwargs.get('load_if_exists', True)
        self.n_startup_trials: int = kwargs.get('n_startup_trials', 10)
        self.n_warmup_steps = kwargs.get('n_warmup_steps', 10)
        self.max_total_trials: Optional[int] = kwargs.get('max_total_trials', None)
        self.n_trials: int = kwargs.get('n_trials', 100)
        self.n_envs: int = kwargs.get('n_envs', 1)
        self.n_eval_envs: int = kwargs.get('n_eval_envs', 1)
        self.n_timesteps: int = kwargs.get('n_timesteps', 1000)
        self.n_evaluations: int = kwargs.get('n_evaluations', self.n_timesteps // 100)
        self.n_eval_episodes: int = kwargs.get('n_eval_episodes', 5)
        self.algo: str = kwargs.get('algo', 'SAC')
        self.policy_type: str = 'MlpPolicy'  # currently only MlpPolicy is supported
        self.reward_method: Optional = kwargs.get('reward_method', None)
        self.temporal_reward_method: Optional = kwargs.get('temporal_reward_method', 'exp_relative_clip')
        self.temporal_reward_operation: str = kwargs.get('temporal_reward_operation', 'mean')
        self.temporal_reward_max: float = kwargs.get('temporal_reward_max', 1.0)
        self.temporal_data: str = kwargs.get('temporal_data', 'cf_se')
        self.temporal_window_size: int = kwargs.get('temporal_window_size', 10)
        self.seed: int = kwargs.get('seed', 0)

        self.study_name: str = kwargs.get('study_name')
        self.storage_url: str = kwargs.get('storage_url', f"sqlite:///{self.env_name}.sqlite3")
        self.open_dashboard: bool = kwargs.get('open_dashboard', False)
        self.dashboard_port: int = kwargs.get('dashboard_port', 9000)

        self.verbose: int = kwargs.get('verbose', 0)

        log_directory = Path(f"logs_{self.study_name}")
        log_directory.mkdir(parents=True, exist_ok=True)
        self.save_path = str(log_directory)

        opt_log_directory = Path(f"opt_logs_{self.study_name}")
        opt_log_directory.mkdir(parents=True, exist_ok=True)
        self.optimization_log_path = opt_log_directory
        self.no_log: bool = kwargs.get('no_log', True)

    @staticmethod
    def set_device() -> str:
        """
        Determine the computing device based on system configuration.

        Returns:
            str: The name of the device ('cpu', 'cuda', or 'mps').
        """
        if sys.platform == "darwin":  # Check if macOS
            device = "mps" if mps.is_available() else "cpu"
        else:  # For other operating systems
            device = "cuda" if cuda.is_available() else "cpu"
        # if device == "mps":
        #     return "cpu"
        # else:
        return device

    def register_env(self) -> None:
        """
        Register the custom Gymnasium environment.
        """
        register(id=self.env_id, entry_point=self.entry_point, max_episode_steps=self.max_episode_steps, )

    def create_sampler(self) -> BaseSampler:
        """
         Create a sampler based on the specified method.

         Returns:
             BaseSampler: An instance of the chosen sampler.
         """
        if self.sampler_method == "random":
            sampler: BaseSampler = RandomSampler(seed=self.seed)
        elif self.sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=10, seed=self.seed, multivariate=True)
        elif self.sampler_method == "skopt":
            from optuna.integration.skopt import SkoptSampler
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_method}")
        return sampler

    def create_pruner(self) -> BasePruner:
        """
        Create a pruner based on the specified method.

        Returns:
            BasePruner: An instance of the chosen pruner.
        """
        if self.pruner_method == "halving":
            pruner: BasePruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif self.pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_warmup_steps // 3)
        elif self.pruner_method == "none":
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.pruner_method}")
        return pruner

    def create_envs(self, vec_env_class: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]], eval_env: bool = False,
                    no_log: bool = True) -> make_vec_env:
        """
        Create environments for training or evaluation.

        Args:
            vec_env_class (type): The class of vectorized environment to be used.
            eval_env (bool): Whether the environment is for evaluation.
            no_log (bool): Whether to disable logging.

        Returns:
            make_vec_env: An instance of the created vectorized environment.
        """
        log_dir = None if eval_env or no_log else self.save_path
        spec = gym.spec(self.env_id)

        def make_env(**kwargs) -> gym.Env:
            return spec.make(**kwargs)

        actual_env_kwargs = self.eval_env_kwargs if eval_env else self.env_kwargs

        env = make_vec_env(make_env, n_envs=self.n_envs, seed=self.seed, env_kwargs=actual_env_kwargs,
                           monitor_dir=log_dir, vec_env_cls=vec_env_class, )
        return env

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for the Optuna optimization.

        Args:
            trial (optuna.Trial): A trial instance from Optuna.

        Returns:
            float: The objective value (e.g., mean reward) for the trial.
        """
        kwargs = dict()
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)
        env = self.create_envs(vec_env_class=DummyVecEnv, eval_env=False, no_log=self.no_log)
        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](policy=self.policy_type, env=env, seed=None,  # Not seeding the trial
                                 verbose=trial_verbosity, device=self.device, **kwargs, )

        eval_env = self.create_envs(vec_env_class=DummyVecEnv, eval_env=True, no_log=self.no_log)

        optuna_eval_freq = max(int(self.n_timesteps / self.n_evaluations) // self.n_envs, 1)

        path = os.path.join(self.optimization_log_path, f"trial_{trial.number}") if self.optimization_log_path else None
        callbacks = [
            TrialEvalCallback(eval_env, trial, best_model_save_path=path, log_path=path,
                              n_eval_episodes=self.n_eval_episodes,
                              eval_freq=optuna_eval_freq, deterministic=self.deterministic_eval)]

        try:
            model.learn(self.n_timesteps, callback=callbacks)
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            model.env.close()
            eval_env.close()
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e

        is_pruned = callbacks[0].is_pruned
        reward = callbacks[0].last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def run(self) -> None:
        """
        Run the hyperparameter optimization.
        """
        sampler = self.create_sampler()
        pruner = self.create_pruner()

        study = optuna.create_study(
            storage=self.storage_url,
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            direction=self.optimize_direction,
            load_if_exists=self.load_if_exists,
        )

        # Start the check-and-launch thread
        dashboard_thread = None
        if self.open_dashboard:
            dashboard_thread = threading.Thread(
                target=check_and_launch_dashboard,
                args=(os.path.dirname(os.path.realpath(__file__)), self.storage_url, self.dashboard_port)
            )
            dashboard_thread.start()

        try:
            if self.max_total_trials is not None:
                counted_states = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.PRUNED]
                completed_trials = len(study.get_trials(states=counted_states))
                print(f"Running optimization trial {completed_trials + 1}/{self.max_total_trials}")
                if completed_trials < self.max_total_trials:
                    study.optimize(
                        lambda _trial: self.objective(_trial), n_jobs=1,
                        callbacks=[MaxTrialsCallback(self.max_total_trials, states=counted_states)], )
            else:
                print(f"Running optimization with a maximum of {self.n_trials} trials.")
                study.optimize(
                    lambda _trial: self.objective(_trial), n_jobs=1, n_trials=self.n_trials, )

        except KeyboardInterrupt:
            pass
        finally:
            if dashboard_thread:
                dashboard_thread.join()

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (f"report_{self.study_name}_{self.n_trials}-trials-{self.n_timesteps}-{self.sampler_method}-"
                       f"{self.pruner_method}_{int(time.time())}")

        log_path = os.path.join(self.save_path, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        with open(f"{log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)


if __name__ == "__main__":
    # reward_method
    # options:
    # 1) channel_capacity
    # 2) min_se
    # 3) mean_se
    # 4) sum_se
    # 5) geo_mean_se
    # 6) cf_min_se
    # 7) cf_mean_se
    # 8) cf_sum_se
    #
    # temporal_reward_method
    # options:
    # 1) delta
    # 2) relative
    # 3) exp_delta_clip
    # 4) exp_relative_clip
    # 5) log_delta
    # 6) log_relative
    area_bounds = (0, square_length, 0, square_length)
    AP_locations = generate_ap_locations(L, 100, area_bounds)
    UE_initial_locations = generate_ue_locations(K, area_bounds)

    env_id = "env/MobilityCFmMIMOEnv-v0"
    env_name = "MobilityCFmMIMOEnv"

    temporal_reward_method = "exp_delta_clip"
    temporal_data = "cf_se"  # cf_se | sinr
    temporal_reward_operation = "mean"

    environment_kwargs = {"APs_positions": AP_locations, "UEs_positions": UE_initial_locations, "UEs_mobility": True,
                          "temporal_reward_method": temporal_reward_method,
                          "temporal_reward_operation": temporal_reward_operation,
                          "temporal_data": temporal_data}

    study_name = f"{env_name}-SAC-{temporal_reward_method}-{temporal_data}-{temporal_reward_operation}"
    opt = HyperparameterOptimizer(env_id=env_id, env_name=env_name, entry_point="env:MobilityCFmMIMOEnv",
                                  env_kwargs=environment_kwargs, study_name=study_name, n_timesteps=2000, n_trials=200,
                                  open_dashboard=True, dashboard_port=8080, verbose=0)
    opt.register_env()
    opt.run()

# gmean with:
#   - relative
#   - exp_delta_clip
#   - exp_relative_clip
