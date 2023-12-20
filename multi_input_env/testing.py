import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from fix_env import CFmMIMOEnv
from power_optimization import compute_se_np
from simulation_para import L, K, tau_p, min_power, max_power, initial_power, square_length, decorr, sigma_sf, \
    noise_variance_dbm, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AP_locations = torch.rand(L, dtype=torch.complex64, device=device) * square_length
UE_initial_locations = torch.rand(K, dtype=torch.complex64, device=device) * square_length

env = CFmMIMOEnv(L=L, K=K, tau_p=tau_p, initial_power=initial_power, min_power=min_power, max_power=max_power,
                 APs_positions=AP_locations, UEs_positions=UE_initial_locations, square_length=square_length,
                 decorr=decorr, sigma_sf=sigma_sf, noise_variance_dbm=noise_variance_dbm, delta=delta)

check_env(env, warn=True)

model = SAC.load("SAC_CFmMIMO_SGD(2x256)")

obs, info = env.reset()

num_of_setups = 1000

cf_signal = np.zeros((K, num_of_setups))
cf_interference = np.zeros((K, K, num_of_setups))
cf_pred_power = np.zeros((K, num_of_setups))

_, _ = env.reset()
over_all_time = time.time()
for n in range(num_of_setups):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    cf_signal[:, n] = info['signal']
    cf_interference[:, :, n] = info['interference']
    cf_pred_power[:, n] = info['predicted_power']
print('Total inference: ', time.time() - over_all_time)

prelog_factor = 1
cf_SE = compute_se_np(cf_signal, cf_interference, cf_pred_power, prelog_factor)

# Flatten the arrays
cf_SE_flatter = cf_SE.flatten()

# Sort the arrays
sorted_cf_SE = np.sort(cf_SE_flatter)

# Compute the CDF values
cdf_SE_pred = np.linspace(0, 1, len(sorted_cf_SE))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sorted_cf_SE, cdf_SE_pred, label='SE DRL(SAC) pred', linewidth=2)

plt.xlabel('Spectral Efficiency (SE)')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.title('CDF of Spectral Efficiencies')
plt.legend()
plt.grid(True)
plt.show()
