import sys
from pathlib import Path

import torch
import torch.optim as optim

from stable_baselines3.common.env_checker import check_env
from torch import cuda
from torch.backends import mps

from algos import ALGOS

from mobility_env import MobilityCFmMIMOEnv
from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

if sys.platform == "darwin":  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:  # For other operating systems
    device = "cuda" if cuda.is_available() else "cpu"

models_dir = Path("models/SAC")
models_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

if device == "mps":
    _device = "cpu"
else:
    _device = device

AP_locations = torch.rand(L, dtype=torch.complex64, device=_device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=_device) * square_length

env = MobilityCFmMIMOEnv(L=L, K=K, tau_p=tau_p, initial_power=initial_power, min_power=min_power, max_power=max_power,
                         APs_positions=AP_locations, UEs_positions=UE_initial_locations, square_length=square_length,
                         decorr=decorr, sigma_sf=sigma_sf, noise_variance_dbm=noise_variance_dbm, delta=delta)

config = {
    "algo": "SAC",
    "policy_type": "MlpPolicy",
    "total_timesteps": 10000,
    "env_name": env.__class__.__name__,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "optimizer_class": optim.SGD,
    "net_arch": [128, 256, 128],
}


# It will check your custom environment and output additional warnings if needed
check_env(env, warn=True)

model = ALGOS[config["algo"]](
    config["policy_type"],
    env,
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    policy_kwargs=dict(net_arch=config["net_arch"]),
    verbose=1,
    device=device,
)


print(model.policy)

model.learn(config["total_timesteps"], progress_bar=True, )

model.save("SAC_CFmMIMO_SGD[MPS][M]")

del model
