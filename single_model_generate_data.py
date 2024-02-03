import os
import sys

from stable_baselines3 import SAC
from torch import cuda
from torch.backends import mps

from _utils import generate_ap_locations, generate_ue_locations
from benchmark import Benchmark
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

env_ch_cap = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True,
                                reward_method='channel_capacity')

env_geo_mean = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True,
                                  reward_method='geo_mean_se')

env_mean = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True,
                              reward_method='mean_se')

env_min = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True,
                             reward_method='min_se')

env_sum = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True,
                             reward_method='sum_se')

ch_cap_benchmark = Benchmark(model=model_ch_cap, env=env_ch_cap, num_of_iterations=1000)

geo_mean_benchmark = Benchmark(model=model_geo_mean, env=env_geo_mean, num_of_iterations=1000)

mean_benchmark = Benchmark(model=model_mean, env=env_mean, num_of_iterations=1000)

min_benchmark = Benchmark(model=model_min, env=env_min, num_of_iterations=1000)

sum_benchmark = Benchmark(model=model_sum, env=env_sum, num_of_iterations=1000)

ch_cap_results = ch_cap_benchmark.run(show_progress=True)
geo_mean_results = geo_mean_benchmark.run(show_progress=True)
mean_results = mean_benchmark.run(show_progress=True)
min_results = min_benchmark.run(show_progress=True)
sum_results = sum_benchmark.run(show_progress=True)

for key, value in ch_cap_results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    reward_method = "channel_capacity"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}-{reward_method}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{reward_method}-{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    ch_cap_results[key].to_csv(file)

for key, value in geo_mean_results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    reward_method = "geo_mean_se"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}-{reward_method}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{reward_method}-{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    geo_mean_results[key].to_csv(file)

for key, value in mean_results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    reward_method = "mean_se"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}-{reward_method}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{reward_method}-{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    mean_results[key].to_csv(file)

for key, value in min_results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    reward_method = "min_se"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}-{reward_method}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{reward_method}-{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    min_results[key].to_csv(file)

for key, value in sum_results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    reward_method = "sum_se"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}-{reward_method}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{reward_method}-{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    sum_results[key].to_csv(file)
