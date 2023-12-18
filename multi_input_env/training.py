import torch
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from env import CFmMIMOEnv
from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 20000,
    "env_name": "CFmMIMOEnv",
    "learning_rate": 5e-4,
    "batch_size": 128,
    "optimizer_class": optim.SGD,
    "net_arch": [256, 256],
}

AP_locations = torch.rand(L, dtype=torch.complex64, device=device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=device) * square_length

env = CFmMIMOEnv(L=L, K=K, tau_p=tau_p, initial_power=initial_power, min_power=min_power, max_power=max_power,
                 APs_positions=AP_locations, UEs_positions=UE_initial_locations, square_length=square_length,
                 decorr=decorr, sigma_sf=sigma_sf, noise_variance_dbm=noise_variance_dbm, delta=delta)

check_env(env, warn=True)

model = SAC(
    config["policy_type"],
    env,
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    verbose=1,
)

model.learn(config["total_timesteps"], progress_bar=True, )

model.save("SAC_CFmMIMO_SGD[256, 256]")

del model
