import torch
import torch.optim as optim

from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from env import MobilityCFmMIMOEnv as CFmMIMOEnv
from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = Path("models/SAC")
models_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 20000,
    "env_name": "CFmMIMOEnv",
    "gamma": 0.9,
    "learning_rate": 0.00065,
    "batch_size": 64,
    "tau": 0.02,
    "log_std_init": 0.7187,
    "buffer_size": 10000,
    "optimizer_class": optim.SGD,
    "net_arch": [400, 300],
}

area_bounds = (0, square_length, 0, square_length)
AP_locations = generate_ap_locations(L, 100, area_bounds)
UE_initial_locations = generate_ue_locations(K, area_bounds)

env = CFmMIMOEnv(APs_positions=AP_locations, UEs_positions=UE_initial_locations, UEs_mobility=True,)

# It will check your custom environment and output additional warnings if needed
check_env(env, warn=True)

model = SAC(
    config["policy_type"],
    env,
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    gamma=config["gamma"],
    tau=config["tau"],
    buffer_size=config["buffer_size"],
    policy_kwargs=dict(net_arch=config["net_arch"], log_std_init=config["log_std_init"],),
    verbose=1,
    device="mps",
)

print(model.policy)

model.learn(config["total_timesteps"], progress_bar=True, )

model.save("SAC_CFmMIMO_SGD[MPS]")

del model