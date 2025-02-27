import os

from stable_baselines3 import SAC, DDPG

from env import MobilityCFmMIMOEnv
from benchmark import ComputationalComplexityBenchmark
from simulation_para import L, K, square_length
from _utils import generate_ap_locations, generate_ue_locations

area_bounds = (0, square_length, 0, square_length)
APs_positions = generate_ap_locations(L, 100, area_bounds)
UEs_positions = generate_ue_locations(K, area_bounds)

models_dir = 'models'
models_folder = 'Models'
algo_name = "SAC"
optim_name = "SGD"

models_dict = {
    'MODEL_DELTA_SINR': 'MODEL_SAC_SGD_DELTA_SINR_202401050208',
    'MODEL_EXP_DELTA_CLIP_SINR': 'MODEL_SAC_SGD_EXP_DELTA_CLIP_SINR_202401050217',
    'MODEL_LOG_DELTA_SINR': 'MODEL_SAC_SGD_LOG_DELTA_SINR_202401121323',
    'MODEL_RELATIVE_SINR': 'MODEL_SAC_SGD_RELATIVE_SINR_202401050237',
    'MODEL_EXP_RELATIVE_CLIP_SINR': 'MODEL_SAC_SGD_EXP_RELATIVE_CLIP_SINR_202401050309',
    'MODEL_LOG_RELATIVE_SINR': 'MODEL_SAC_SGD_LOG_RELATIVE_SINR_202401050300'
}

# models_dict = {
#     # 'MODEL_DELTA_SE': 'MODEL_SAC_SGD_DELTA_CF_SE_202401050057',
#     # 'MODEL_EXP_DELTA_CLIP_SEv0': 'MODEL_SAC_SGD_EXP_DELTA_CLIP_CF_SE_202401050105V0',
#     # 'MODEL_LOG_DELTA_SE': 'MODEL_SAC_SGD_LOG_DELTA_CF_SE_202401121359',
#     # 'MODEL_RELATIVE_SE': 'MODEL_SAC_SGD_RELATIVE_CF_SE_202401050139',
#     # 'MODEL_EXP_RELATIVE_CLIP_SE': 'MODEL_SAC_SGD_EXP_RELATIVE_CLIP_CF_SE_202401050152',
#     # 'MODEL_LOG_RELATIVE_SE': 'MODEL_SAC_SGD_LOG_RELATIVE_CF_SE_202401050200',
#     'SAC_DELTA_SINR': 'MODEL_SAC_SGD_DELTA_SINR_202401050208',
#     # 'SAC_EXP_DELTA_CLIP_SINR': 'MODEL_SAC_SGD_EXP_DELTA_CLIP_SINR_202401050217',
#     # 'SAC_LOG_DELTA_SINR': 'MODEL_SAC_SGD_LOG_DELTA_SINR_202401121323',
#     # 'SAC_RELATIVE_SINR': 'MODEL_SAC_SGD_RELATIVE_SINR_202401050237',
#     # 'SAC_EXP_RELATIVE_CLIP_SINR': 'MODEL_SAC_SGD_EXP_RELATIVE_CLIP_SINR_202401050309',
#     'SAC_LOG_RELATIVE_SINR': 'MODEL_SAC_SGD_LOG_RELATIVE_SINR_202401050300',
#     'DDPG_DELTA_SINR': 'MODEL_DDPG_SGD_DELTA_SINR_202401211437',
#     # 'DDPG_EXP_DELTA_SINR': 'MODEL_DDPG_SGD_EXP_DELTA_CLIP_SINR_202401211444',
#     # 'DDPG_LOG_DELTA_SINR': 'MODEL_DDPG_SGD_LOG_DELTA_SINR_202401211517',
#     # 'DDPG_RELATIVE_SINR': 'MODEL_DDPG_SGD_RELATIVE_SINR_202401211523',
#     'DDPG_LOG_RELATIVE_SINR': 'MODEL_DDPG_SGD_LOG_RELATIVE_SINR_202401211531',
# }

for key, value in models_dict.items():
    model_name = value
    model_path = os.path.join(models_dir, models_folder, model_name)
    if 'SAC' in model_name:
        print(model_name)
        models_dict[key] = SAC.load(model_path)
    if 'DDPG' in model_name:
        print(model_name)
        models_dict[key] = DDPG.load(model_path)


env = MobilityCFmMIMOEnv(APs_positions=APs_positions, UEs_positions=UEs_positions, UEs_mobility=True, eval=True)

bm = ComputationalComplexityBenchmark(models=models_dict, env=env, num_of_iterations=100, mobility=True)

results = bm.run(show_progress=True)

for key, value in results.items():
    algo_name = "SAC"
    optim_name = "SGD"
    results_dir = 'RESULTS'
    results_folder = f'{algo_name}-{optim_name}'
    os.makedirs(os.path.join(results_dir, results_folder), exist_ok=True)
    file_name = f'{key}.csv'
    file = os.path.join(results_dir, results_folder, file_name)
    results[key].to_csv(file)
