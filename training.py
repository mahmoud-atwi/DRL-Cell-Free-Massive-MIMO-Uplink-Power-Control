import os
import sys
import warnings
from copy import copy
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.env_checker import check_env
from torch import cuda
from torch.backends import mps
from torch.optim import SGD, Adam

from _options import select_reward_option, load_hyperparameters, get_config, get_hyperparameters
from _utils import generate_ap_locations, generate_ue_locations
from algos import ALGOS
from env import MobilityCFmMIMOEnv as CFmMIMOEnv
from simulation_para import L, K, square_length


class RLModelTrainer:
    def __init__(self, seed=0, verbose=0):
        self.seed: int = seed
        self.verbose: int = verbose
        self.optim_cls: str = ''
        self.env_reward: str = ''
        self.device: str = ''
        self.config: dict = dict()
        self.initialize()

    @staticmethod
    def _set_device():
        if sys.platform == 'darwin':
            return "mps" if mps.is_available() else "cpu"
        else:
            return "cuda" if cuda.is_available() else "cpu"

    def initialize(self):
        self.device = self._set_device()
        self._setup_directories()
        self.config = get_config()
        self._load_hyperparams()
        self._setup_environment()

    def _setup_directories(self):
        self.log_dir = Path('logs')
        self.models_dir = 'models'

    def _load_hyperparams(self):
        self.yaml_file = get_hyperparameters(self.config)
        env_name, self.reward_method, self.temporal_reward_method, self.temporal_data = select_reward_option()
        self.env_reward = self.temporal_reward_method if self.temporal_reward_method is not None else self.reward_method
        self.hyperparams = load_hyperparameters(self.yaml_file, env_name)
        self._set_optimizer_class()

    def _set_optimizer_class(self):
        try:
            self.optim_cls = copy(self.hyperparams["policy_kwargs"]['optimizer_class'])
        except KeyError:
            warnings.warn("No optimizer_class specified in policy_kwargs, defaulting to SGD.")
            self.optim_cls = "SGD"

        if self.optim_cls is None or self.optim_cls.upper() == "SGD":
            self.hyperparams["policy_kwargs"]['optimizer_class'] = SGD
        elif self.optim_cls.upper() == "ADAM":
            self.hyperparams["policy_kwargs"]['optimizer_class'] = Adam

    def _setup_environment(self):
        area_bounds = (0, square_length, 0, square_length)
        self.APs_positions = generate_ap_locations(L, 100, area_bounds)
        self.UEs_positions = generate_ue_locations(K, area_bounds)

        self.env = CFmMIMOEnv(
            APs_positions=self.APs_positions,
            UEs_positions=self.UEs_positions,
            UEs_mobility=True,
            reward_method=self.reward_method,
            temporal_reward_method=self.temporal_reward_method,
            temporal_data=self.temporal_data,
        )
        check_env(self.env, warn=True)

    def train(self):
        model = ALGOS[self.config["algo"]](
            policy=self.hyperparams["policy"],
            env=self.env,
            learning_rate=self.hyperparams["learning_rate"],
            buffer_size=self.hyperparams["buffer_size"],
            learning_starts=self.hyperparams["learning_starts"],
            batch_size=self.hyperparams["batch_size"],
            tau=self.hyperparams["tau"],
            gamma=self.hyperparams["gamma"],
            train_freq=self.hyperparams["train_freq"],
            gradient_steps=self.hyperparams["gradient_steps"],
            ent_coef=self.hyperparams["ent_coef"],
            policy_kwargs=self.hyperparams["policy_kwargs"],
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
        )
        log_path = self._get_log_path()
        model.learn(total_timesteps=self.config["total_timesteps"], log_interval=1000, tb_log_name=log_path,
                    progress_bar=True)
        self._save_model(model)

    def _get_log_path(self):
        return os.path.join(self.log_dir, f'{self.config["algo"]}', f'{self.optim_cls.upper()}',
                            f'{self.env_reward.upper()}', f'{self.temporal_data.upper()}')

    def _save_model(self, model):
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        model_name = (f'MODEL_{self.config["algo"]}_{self.optim_cls.upper()}_{self.env_reward.upper()}_'
                      f'{self.temporal_data.upper()}_{current_time}')
        save_path = os.path.join(self.models_dir, model_name)
        model.save(save_path)
        del model  # delete trained model from memory


if __name__ == '__main__':
    trainer = RLModelTrainer()
    trainer.train()


# 14 Exp Delta SINR Clip: failed
