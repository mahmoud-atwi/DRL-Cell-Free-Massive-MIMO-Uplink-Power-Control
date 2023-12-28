import sys
import yaml
from pathlib import Path

from stable_baselines3.common.env_checker import check_env
from torch import cuda, optim
from torch.backends import mps

from algos import ALGOS
from _utils import generate_ap_locations, generate_ue_locations

from env import MobilityCFmMIMOEnv
from simulation_para import L, K, square_length

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

yaml_path = "sac.yaml"

with open(yaml_path) as f:
    hyperparams_dict = yaml.safe_load(f)

area_bounds = (0, square_length, 0, square_length)
AP_locations = generate_ap_locations(L, 100, area_bounds)
UE_initial_locations = generate_ue_locations(K, area_bounds)

verbose = 1
seed = 0

config = {
    "algo": "SAC",
    "optimizer_class": optim.SGD,
}


env = MobilityCFmMIMOEnv(APs_positions=AP_locations, UEs_positions=UE_initial_locations, UEs_mobility=True,)

# Check the environment before training
check_env(env, warn=True)

meanSE_20 = "MobilityCFmMIMOEnv-v0"
meanSE = "MobilityCFmMIMOEnv-v1"
minSE = "MobilityCFmMIMOEnv-v2"
sumSE = "MobilityCFmMIMOEnv-v3"
geo_meanSE = "MobilityCFmMIMOEnv-v4"


kwargs = dict()
kwargs.update(hyperparams_dict[meanSE_20])

model = ALGOS[config['algo']](env=env, seed=seed, verbose=verbose, device=device, **kwargs,)

print(model.policy)

# model.learn(config["total_timesteps"], progress_bar=True, )
#
# model.save("SAC_CFmMIMO_SGD[MPS][M]")
#
# del model
