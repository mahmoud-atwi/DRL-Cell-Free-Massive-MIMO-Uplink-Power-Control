import os

import pandas as pd
from stable_baselines3 import SAC, DDPG

from _utils import generate_ap_locations, generate_ue_locations
from benchmark import MultiModelBenchmark
from env import MobilityCFmMIMOEnv

from simulation_para import L, K, square_length

area_bounds = (0, square_length, 0, square_length)
APs_positions = generate_ap_locations(L, 100, area_bounds)
UEs_positions = generate_ue_locations(K, area_bounds)

models_dir = 'models'
models_folder = 'Models'
algo_name = "SAC"
optim_name = "SGD"

models_dict = {
    'SAC_DELTA_SINR': 'MODEL_SAC_SGD_DELTA_SINR_202401050208',
    'SAC_EXP_DELTA_CLIP_SINR': 'MODEL_SAC_SGD_EXP_DELTA_CLIP_SINR_202401050217',
    'SAC_LOG_DELTA_SINR': 'MODEL_SAC_SGD_LOG_DELTA_SINR_202401121323',
    'SAC_RELATIVE_SINR': 'MODEL_SAC_SGD_RELATIVE_SINR_202401050237',
    'SAC_EXP_RELATIVE_CLIP_SINR': 'MODEL_SAC_SGD_EXP_RELATIVE_CLIP_SINR_202401050309',
    'SAC_LOG_RELATIVE_SINR': 'MODEL_SAC_SGD_LOG_RELATIVE_SINR_202401050300',
    # 'DDPG_DELTA_SINR': 'MODEL_DDPG_SGD_DELTA_SINR_202401211437',
    # 'DDPG_EXP_DELTA_SINR': 'MODEL_DDPG_SGD_EXP_DELTA_CLIP_SINR_202401211444',
    # 'DDPG_LOG_DELTA_SINR': 'MODEL_DDPG_SGD_LOG_DELTA_SINR_202401211517',
    # 'DDPG_RELATIVE_SINR': 'MODEL_DDPG_SGD_RELATIVE_SINR_202401211523',
    # 'DDPG_LOG_RELATIVE_SINR': 'MODEL_DDPG_SGD_LOG_RELATIVE_SINR_202401211531',
}

for key, value in models_dict.items():
    model_name = value
    model_path = os.path.join(models_dir, models_folder, model_name)
    if 'SAC' in model_name:
        print(model_name)
        models_dict[key] = SAC.load(model_path)
    if 'DDPG' in model_name:
        print(model_name)
        models_dict[key] = DDPG.load(model_path)

env = MobilityCFmMIMOEnv(L=L, K=K, APs_positions=APs_positions, UEs_positions=UEs_positions,
                         square_length=square_length, UEs_mobility=True, eval=True)

bm = MultiModelBenchmark(models=models_dict, env=env, num_of_iterations=10000, mobility=True, include_maxmin=True,
                         include_maxprod=True, include_sumrate=True)

results = bm.run(show_progress=True)

AP_LOCATION_SAVED = False

for key, value in results.items():
    results_dir = 'RESULTS'
    results_folder = 'SAC-SGD'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    results[key].to_csv(file)
    # save APs locations
    while not AP_LOCATION_SAVED:
        pd.DataFrame(APs_positions).to_csv(os.path.join(results_dir, results_folder, 'APs_locations.csv'))
        AP_LOCATION_SAVED = True
