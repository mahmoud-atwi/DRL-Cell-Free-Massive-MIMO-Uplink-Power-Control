from math import sqrt

import numpy as np
import torch
from scipy.linalg import sqrtm

from compute_spectral_efficiency import compute_se_cf_ul
from helper_functions import db2pow
from pathloss_threeslope import path_loss_three_slope

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def CF_mMIMO_Env(L, K, tau_p, max_power, UEs_power, APs_positions, UEs_positions, square_length, decorr, sigma_sf,
                 noise_variance_dbm, delta, pilot_index_loopback=None, beta_val_loopback=None):

    if beta_val_loopback is None:
        gain_over_noise_db = torch.zeros((L, K), device=device)
        distances_to_UE = torch.zeros((L, K), device=device)

        # Compute alternative AP and UE locations by using wrap around
        wrap_horizontal = torch.tile(torch.tensor([-square_length, 0, square_length]), (3, 1)).to(device)
        wrap_vertical = wrap_horizontal.T
        wrap_locations = (wrap_horizontal + 1j * wrap_vertical).T.flatten()

        wrapped_AP_positions = torch.tile(APs_positions.flatten(), (len(wrap_locations), 1)).T + torch.tile(
            wrap_locations,
            (L, 1))
        wrapped_UE_positions = torch.tile(UEs_positions.flatten(), (len(wrap_locations), 1)).T + torch.tile(
            wrap_locations,
            (K, 1))

        shadow_corr_matrix_APs = torch.zeros((L, L), device=device)
        shadow_corr_matrix_UEs = torch.zeros((K, K), device=device)

        for l in range(L):
            distance_to_AP = torch.min(torch.abs(wrapped_AP_positions - APs_positions[l]), dim=1).values
            shadow_corr_matrix_APs[:, l] = 2 ** (-1 * distance_to_AP / decorr)

        for k in range(K):
            distance_to_UE = torch.min(torch.abs(wrapped_UE_positions - UEs_positions[k]), dim=1).values
            shadow_corr_matrix_UEs[:, k] = 2 ** (-1 * distance_to_UE / decorr)

        # Generate shadow fading realizations
        a = sigma_sf * sqrtm(shadow_corr_matrix_APs.cpu()) @ np.random.randn(L, )
        b = sigma_sf * sqrtm(shadow_corr_matrix_UEs.cpu()) @ np.random.randn(K, )

        a = torch.from_numpy(a.real).to(device)
        b = torch.from_numpy(b.real).to(device)

        # Compute distances and channel gain
        for k in range(K):
            distance_to_UE, position_indices = torch.min(
                torch.abs(wrapped_AP_positions - torch.tile(UEs_positions[k], wrapped_AP_positions.shape)), dim=1)
            distances_to_UE[:, k] = distance_to_UE

            gain_over_noise_db[:, k] = path_loss_three_slope(distance_to_UE) - noise_variance_dbm

            # Add shadow fading over 50 meters
            mask = distance_to_UE > 50
            gain_over_noise_db[mask, k] += sqrt(delta) * a[mask] + sqrt(1 - delta) * b[k]

        beta_val = db2pow(gain_over_noise_db)
    else:
        beta_val = torch.from_numpy(beta_val_loopback).to(device)

    if pilot_index_loopback is None:
        pilot_index_CF = (torch.randperm(K) % tau_p)
    else:
        pilot_index_CF = torch.from_numpy(pilot_index_loopback).to(device)

    # Run greedy algorithm to find optimal pilot allocation in cell-free network
    tau_c = tau_p + 1
    for k in range(K):

        SE_CF = compute_se_cf_ul(max_power, UEs_power, L, K, tau_p, tau_c, pilot_index_CF, beta_val)

        # Find UE with the lowest SE
        min_SE_UE = torch.argmin(SE_CF)

        # Remove UE with the lowest SE from pilot allocation
        pilot_index_CF[min_SE_UE] = -1

        # Compute the interference level for each of the pilots
        pilot_interference_CF = torch.zeros(tau_p, device=device)

        for i in range(tau_p):
            mask = pilot_index_CF == i
            pilot_interference_CF[i] = torch.sum(beta_val[:, mask], dim=(0, 1))

        # Find the pilot with the lowest interference
        min_interference_pilot = torch.argmin(pilot_interference_CF)

        # Assign the UE with the lowest SE to the pilot with the lowest interference
        pilot_index_CF[min_SE_UE] = min_interference_pilot

    max_power_tau_p = max_power * tau_p
    beta_val_squared = beta_val ** 2
    gamma_val = torch.zeros((L, K), device=device)
    interference_CF = torch.zeros((K, K), device=device)

    for k in range(K):
        # Cell-Free mMIMO Computations
        mask_CF = pilot_index_CF[k] == pilot_index_CF
        denominator = max_power_tau_p * torch.sum(beta_val[:, mask_CF], dim=1) + 1
        gamma_val[:, k] = (max_power_tau_p * beta_val_squared[:, k]) / denominator
        noise_term = torch.sum(gamma_val[:, k])
        interference_CF[:, k] += torch.sum(torch.tile(gamma_val[:, k], [K, 1]).T * beta_val, dim=0) / noise_term

        for ind in torch.where(mask_CF)[0]:
            if ind != k:
                interference_CF[ind, k] += torch.sum(
                    gamma_val[:, k] * beta_val[:, ind] / beta_val[:, k]) ** 2 / noise_term

    signal_CF = torch.sum(gamma_val, dim=0)

    B_k = torch.sum(beta_val, dim=0)

    return B_k, signal_CF, interference_CF, SE_CF, pilot_index_CF, beta_val, APs_positions, UEs_positions
