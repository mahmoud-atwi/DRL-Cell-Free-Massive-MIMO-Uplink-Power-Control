import torch
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter

from mob_env import MobCFmMIMOEnv
from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 10000,
    "env_name": "CFmMIMOEnv",
    "learning_rate": 5e-4,
    "batch_size": 128,
    "optimizer_class": optim.SGD,
    "net_arch": [128, 256, 128],
}

AP_locations = torch.rand(L, dtype=torch.complex64, device=device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=device) * square_length

env = MobCFmMIMOEnv(L=L, K=K, tau_p=tau_p, initial_power=initial_power, min_power=min_power, max_power=max_power,
                    APs_positions=AP_locations, UEs_positions=UE_initial_locations, square_length=square_length,
                    decorr=decorr, sigma_sf=sigma_sf, noise_variance_dbm=noise_variance_dbm, delta=delta,
                    with_mobility=True)

check_env(env, warn=True)

mob_env = DummyVecEnv([lambda: env])
mob_env = VecCheckNan(mob_env, raise_exception=True)

model = SAC(
    config["policy_type"],
    mob_env,
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    policy_kwargs=dict(optimizer_class=config["optimizer_class"], net_arch=config["net_arch"]),
    device="mps",
    verbose=1,
    tensorboard_log="./logs/",
)

model.learn(config["total_timesteps"], progress_bar=True, )

model.save("SAC_MobCFmMIMO_SGD[128 256 128][10K][M]")

del model
