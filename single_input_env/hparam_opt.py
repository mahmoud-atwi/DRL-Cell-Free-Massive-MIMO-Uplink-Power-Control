import os
import pickle as pkl
import sys
import time
from pprint import pprint
from typing import Dict, Any, Optional
from pathlib import Path

import gymnasium as gym
import optuna
import torch

from gymnasium.envs.registration import register
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from torch import cuda
from torch import optim
from torch.backends import mps

from hyperparams import HYPERPARAMS_SAMPLER
from algos import ALGOS
from utils import get_callback_list

from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

if sys.platform == "darwin":  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:  # For other operating systems
    device = "cuda" if cuda.is_available() else "cpu"

db_dir = Path("optuna_db")
db_dir.mkdir(parents=True, exist_ok=True)

model_config = {
    "env_id": "CFmMIMOEnv/MobileWorld-v1",
    "env_name": "MobilityCFmMIMOEnv",
    "policy_type": "MlpPolicy",
    "total_timesteps": 100,
    "optimizer_class": optim.SGD,
}

register(
    id=f'{model_config["env_id"]}',
    entry_point="mobility_env:MobilityCFmMIMOEnv",
)


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
            self,
            eval_env: VecEnv,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 2,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elapsed time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class ExperimentManager:
    def __init__(self, study_name, algo, env_id, env_kwargs, eval_env_kwargs, n_timesteps, vec_env_class, n_trials,
                 n_jobs, sampler, pruner,
                 max_total_trials, n_envs, verbose, n_eval_envs, optimization_log_path, n_eval_episodes,
                 specified_callbacks, deterministic_eval,
                 save_path):
        self.study_name = study_name
        self.algo = algo
        self.env_id = env_id
        self.env_name = None
        self.log_folder = None
        self.n_timesteps = n_timesteps
        self.env_kwargs = env_kwargs
        self.eval_env_kwargs = eval_env_kwargs
        self.no_optim_plots = False
        self.vec_env_class = vec_env_class
        self.save_path = save_path
        self.deterministic_eval = deterministic_eval
        self.specified_callbacks = specified_callbacks
        self.n_eval_episodes = n_eval_episodes
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.max_total_trials = max_total_trials
        self.seed = 0
        self.n_startup_trials = 10
        self.n_evaluations = 1
        self._hyperparams: Dict[str, Any] = {}
        self.n_envs = n_envs
        self.verbose = verbose
        self.device = device
        self.n_eval_envs = n_eval_envs
        self.optimization_log_path = optimization_log_path

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of steps.
        if sampler_method == "random":
            sampler: BaseSampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        elif sampler_method == "skopt":
            from optuna.integration.skopt import SkoptSampler
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner: BasePruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def create_envs(self, n_envs, eval_env=False, no_log=False):
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        spec = gym.spec(self.env_id)

        def make_env(**kwargs) -> gym.Env:
            return spec.make(**kwargs)

        env_kwargs = self.eval_env_kwargs if eval_env else self.env_kwargs

        # On most env, SubprocVecEnv does not help and is quite memory hungry,
        # therefore, we use DummyVecEnv by default
        env = make_vec_env(
            make_env,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=env_kwargs,
            monitor_dir=log_dir,
            vec_env_cls=self.vec_env_class,
        )

        return env

    def objective(self, trial):
        kwargs = self._hyperparams.copy()
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)
        env = self.create_envs(self.n_envs, no_log=True)
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            policy=model_config["policy_type"],
            env=env,
            # We do not seed the trial
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{trial.number!s}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}

        try:
            model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)  # type: ignore[arg-type]
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self):
        storage_url = f"sqlite:///{self.study_name}.sqlite3"
        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)
        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            direction="maximize",
            load_if_exists=True,
        )

        try:
            if self.max_total_trials is not None:
                # Note: we count already running trials here otherwise we get
                #  (max_total_trials + number of workers) trials in total.
                counted_states = [
                    TrialState.COMPLETE,
                    TrialState.RUNNING,
                    TrialState.PRUNED,
                ]
                completed_trials = len(study.get_trials(states=counted_states))
                print(f"Running optimization trial {completed_trials + 1}/{self.max_total_trials}")
                if completed_trials < self.max_total_trials:
                    study.optimize(
                        self.objective,
                        n_jobs=self.n_jobs,
                        callbacks=[
                            MaxTrialsCallback(
                                self.max_total_trials,
                                states=counted_states,
                            )
                        ],
                    )
            else:
                print(f"Running forever with a maximum of {self.n_trials} trials.")
                study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)
                print(f"Best value: {study.best_value} (params: {study.best_params})")
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_name}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )

        log_path = os.path.join(self.log_folder, self.algo, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        # Skip plots
        if self.no_optim_plots:
            return

        # Plot optimization result
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass


if device == "mps":
    _device = "cpu"
else:
    _device = device

AP_locations = torch.rand(L, dtype=torch.complex64, device=_device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=_device) * square_length

env_keywords = {"APs_positions": AP_locations, "UEs_positions": UE_initial_locations}

experiment_manager = ExperimentManager(
    study_name="CF_mMIMO_SAC",
    algo="SAC",
    env_id=model_config["env_id"],
    env_kwargs=env_keywords,
    eval_env_kwargs=env_keywords,
    n_timesteps=1,
    vec_env_class=DummyVecEnv,
    n_trials=100,
    n_jobs=1,
    sampler="tpe",
    pruner="median",
    max_total_trials=10,
    n_envs=1,
    verbose=2,
    n_eval_envs=1,
    optimization_log_path=None,
    n_eval_episodes=5,
    specified_callbacks=None,
    deterministic_eval=True,
    save_path="./logs/",
)

experiment_manager.hyperparameters_optimization()
