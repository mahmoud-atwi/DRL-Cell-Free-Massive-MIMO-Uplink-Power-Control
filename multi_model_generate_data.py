import os
import sys

from stable_baselines3 import SAC
from torch import cuda
from torch.backends import mps

from _utils import generate_ap_locations, generate_ue_locations
from benchmark import Benchmark, MultiModelBenchmark
from env import MobilityCFmMIMOEnv
from simulation_para import L, K, square_length

if sys.platform == 'darwin':  # Check if macOS
    device = "mps" if mps.is_available() else "cpu"
else:
    device = "cuda" if cuda.is_available() else "cpu"

area_bounds = (0, square_length, 0, square_length)
APs_positions = generate_ap_locations(L, 100, area_bounds)
UEs_positions = generate_ue_locations(K, area_bounds)

algo_name = "SAC"
optim_name = "SGD"

reward_method = "channel_capacity"
models_dir = 'models'
models_folder = f'{algo_name}-{optim_name}-{reward_method}'
model1_name = 'MobilityCFmMIMOEnv_SAC_SGD_channel_capacity_20231230-2008'
model_path = os.path.join(models_dir, models_folder, model1_name)
model_ch_cap = SAC.load(model_path)

reward_method = "geo_mean_se"
models_folder = f'{algo_name}-{optim_name}-{reward_method}'
model2_name = 'MobilityCFmMIMOEnv_SAC_SGD_geo_mean_se_20231230-2035'
model_path = os.path.join(models_dir, models_folder, model2_name)
model_geo_mean = SAC.load(model_path)

reward_method = "mean_se"
models_folder = f'{algo_name}-{optim_name}-{reward_method}'
model2_name = 'MobilityCFmMIMOEnv_SAC_SGD_mean_se_20231230-1528'
model_path = os.path.join(models_dir, models_folder, model2_name)
model_mean = SAC.load(model_path)

reward_method = "min_se"
models_folder = f'{algo_name}-{optim_name}-{reward_method}'
model4_name = 'MobilityCFmMIMOEnv_SAC_SGD_min_se_20231230-2044'
model_path = os.path.join(models_dir, models_folder, model4_name)
model_min = SAC.load(model_path)

reward_method = "sum_se"
models_folder = f'{algo_name}-{optim_name}-{reward_method}'
model5_name = 'MobilityCFmMIMOEnv_SAC_SGD_sum_se_20231230-2054'
model_path = os.path.join(models_dir, models_folder, model5_name)
model_sum = SAC.load(model_path)

env = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True)

models_dict = {
    'Model_ch_cap': model_ch_cap,
    'Model_geo_mean': model_geo_mean,
    'Model_mean': model_mean,
    'Model_min': model_min,
    'Model_sum': model_sum
}

bm = MultiModelBenchmark(models=models_dict, env=env, num_of_iterations=1000, mobility=True)


results = bm.run(show_progress=True)

for key, value in results.items():
    algo_name = "SAC"
    optim_name = "SGD_with_cf_se"
    results_dir = 'results'
    results_folder = f'{algo_name}-{optim_name}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    results[key].to_csv(file)
