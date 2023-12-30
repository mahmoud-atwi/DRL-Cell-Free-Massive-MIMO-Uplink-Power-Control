import os
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.env_checker import check_env
from torch import cuda
from torch.backends import mps

from _options import select_reward_option, load_hyperparameters, get_config, get_hyperparameters
from _utils import generate_ap_locations, generate_ue_locations
from algos import ALGOS
from env import MobilityCFmMIMOEnv as CFmMIMOEnv
from simulation_para import L, K, square_length

if sys.platform == 'darwin':  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:
    device = "cuda" if cuda.is_available() else "cpu"

verbose = 0
seed = 0

config = get_config()

print(f'Algo: {config["algo"]}')
print(f'Optimizer: {config["optimizer_class"].__name__}')
print(f'total_timesteps: {config["total_timesteps"]}')

yaml_file = get_hyperparameters(config)

# Reward (1): Channel Capacity  --> MobilityCFmMIMOEnv-ch_capacity-v0
# Reward (2): Geometric Mean SE --> MobilityCFmMIMOEnv-geo_mean_se-v0
# Reward (3): Mean SE           --> MobilityCFmMIMOEnv-mean_se-v0
# Reward (4): Min SE            --> MobilityCFmMIMOEnv-min_se-v0
# Reward (5): Sum SE            --> MobilityCFmMIMOEnv-sum_se-v0

# load pre-tuned hyperparams
_env_reward = select_reward_option()
_hyperparams = load_hyperparameters(yaml_file, _env_reward)
_hyperparams["policy_kwargs"]['optimizer_class'] = config["optimizer_class"]

# logs directory
log_dir = Path('logs')
log_path = os.path.join(log_dir, f'{_env_reward}-{config["algo"]}-{str(config["optimizer_class"].__name__)}')
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# distribute APs and UEs
area_bounds = (0, square_length, 0, square_length)
APs_positions = generate_ap_locations(L, 100, area_bounds)
UEs_positions = generate_ue_locations(K, area_bounds)

if _env_reward == "MobilityCFmMIMOEnv-ch_capacity-v0":
    reward_method = 'channel_capacity'
elif _env_reward == "MobilityCFmMIMOEnv-geo_mean_se-v0":
    reward_method = 'geo_mean_se'
elif _env_reward == "MobilityCFmMIMOEnv-mean_se-v0":
    reward_method = 'mean_se'
elif _env_reward == "MobilityCFmMIMOEnv-min_se-v0":
    reward_method = 'min_se'
elif _env_reward == "MobilityCFmMIMOEnv-sum_se-v0":
    reward_method = 'sum_se'
else:
    raise ValueError("Invalid reward option selected.")

# directory to save models
models_dir = 'models'
model_path = os.path.join(models_dir, f'{config["algo"]}-{str(config["optimizer_class"].__name__)}-{reward_method}')
os.makedirs(os.path.dirname(model_path), exist_ok=True)


env = CFmMIMOEnv(
    APs_positions=APs_positions,
    UEs_positions=UEs_positions,
    UEs_mobility=True,
    reward_method=reward_method,
)

# check the custom environment
check_env(env, warn=True)

# model
print(_hyperparams["policy_kwargs"])
model = ALGOS[config["algo"]](
    policy=_hyperparams["policy"],
    env=env,
    learning_rate=_hyperparams["learning_rate"],
    buffer_size=_hyperparams["buffer_size"],
    learning_starts=_hyperparams["learning_starts"],
    batch_size=_hyperparams["batch_size"],
    tau=_hyperparams["tau"],
    gamma=_hyperparams["gamma"],
    train_freq=_hyperparams["train_freq"],
    gradient_steps=_hyperparams["gradient_steps"],
    ent_coef=_hyperparams["ent_coef"],
    policy_kwargs=_hyperparams["policy_kwargs"],
    seed=seed,
    verbose=verbose,
    device=device,
)

print(f'model:{model.policy}')
print(f'model optimizer:{model.policy.optimizer_class}')

# train
model.learn(total_timesteps=config["total_timesteps"], log_interval=1000, tb_log_name=log_path, progress_bar=True)

# save
current_time = datetime.now().strftime("%Y%m%d-%H%M")
model_name = \
    f'{env.__class__.__name__}_{config["algo"]}_{config["optimizer_class"].__name__}_{reward_method}_{current_time}'
save_path = os.path.join(model_path, model_name)
model.save(save_path)

del model  # delete trained model from memory
