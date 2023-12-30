from math import sqrt
from typing import Tuple, Optional, Union

import numpy as np
from scipy.linalg import sqrtm

from _utils import db2pow
from compute_spectral_efficiency import compute_se_cf_ul
from pathloss_threeslope import path_loss_three_slope


def cf_mimo_simulation(L: int, K: int, tau_p: int, max_power: float, UEs_power: Union[float, np.ndarray],
                       APs_positions: np.ndarray, UEs_positions: np.ndarray, square_length: float, decorr: float,
                       sigma_sf: float, noise_variance_dbm: float, delta: float,
                       pilot_index_loopback: Optional[np.ndarray] = None,
                       beta_val_loopback: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    """
    Simulates a cell-free massive MIMO system to calculate various system parameters.

    This function simulates a cell-free (CF) Massive MIMO environment given the system parameters. It calculates the
    large-scale fading coefficients, pilot assignments, and other relevant metrics for the system.

    :param L: Number of antennas at the base station.
    :param K: Number of users in the system.
    :param tau_p: Length of the pilot sequence.
    :param max_power: Maximum transmit power.
    :param UEs_power: Uplink power levels for each user.
    :param APs_positions: Positions of the access points.
    :param UEs_positions: Positions of the user equipments.
    :param square_length: Length of the area square in meters.
    :param decorr: Decorrelation distance for shadow fading.
    :param sigma_sf: Shadow fading standard deviation in dB.
    :param noise_variance_dbm: Noise variance in dBm.
    :param delta: Shadow fading decorrelation parameter.
    :param pilot_index_loopback: Optional array for pilot index reuse.
    :param beta_val_loopback: Optional array for beta values reuse.
    :return: A tuple containing Beta_K, cf_signal, cf_interference, cf_spectral_efficiency, pilot_index_CF, beta_val,
             APs_positions, and UEs_positions.
    """

    if beta_val_loopback is None:
        gain_over_noise_db = np.zeros((L, K))
        distances_to_UE = np.zeros((L, K))

        wrap_horizontal = np.tile(np.array([-square_length, 0, square_length]), (3, 1))
        wrap_vertical = wrap_horizontal.T
        wrap_locations = (wrap_horizontal + 1j * wrap_vertical).T.flatten()

        wrapped_AP_positions = np.tile(APs_positions.flatten(), (len(wrap_locations), 1)).T + np.tile(
            wrap_locations,
            (L, 1))
        wrapped_UE_positions = np.tile(UEs_positions.flatten(), (len(wrap_locations), 1)).T + np.tile(
            wrap_locations,
            (K, 1))

        shadow_corr_matrix_APs = np.zeros((L, L))
        shadow_corr_matrix_UEs = np.zeros((K, K))

        for l in range(L):
            distance_to_AP = np.min(np.abs(wrapped_AP_positions - APs_positions[l]), axis=1)
            shadow_corr_matrix_APs[:, l] = 2 ** (-1 * distance_to_AP / decorr)

        for k in range(K):
            distance_to_UE = np.min(np.abs(wrapped_UE_positions - UEs_positions[k]), axis=1)
            shadow_corr_matrix_UEs[:, k] = 2 ** (-1 * distance_to_UE / decorr)

        a = sigma_sf * sqrtm(shadow_corr_matrix_APs) @ np.random.randn(L, )
        b = sigma_sf * sqrtm(shadow_corr_matrix_UEs) @ np.random.randn(K, )

        a = a.real
        b = b.real

        for k in range(K):
            distance_to_UE = np.min(
                np.abs(wrapped_AP_positions - np.tile(UEs_positions[k], wrapped_AP_positions.shape)), axis=1)

            distances_to_UE[:, k] = distance_to_UE

            gain_over_noise_db[:, k] = path_loss_three_slope(distance_to_UE) - noise_variance_dbm

            mask = distance_to_UE > 50
            gain_over_noise_db[mask, k] += sqrt(delta) * a[mask] + sqrt(1 - delta) * b[k]

        beta_val = db2pow(gain_over_noise_db)
    else:
        beta_val = beta_val_loopback

    if pilot_index_loopback is None:
        pilot_index_CF = np.random.permutation(K) % tau_p
    else:
        pilot_index_CF = pilot_index_loopback

    tau_c = tau_p + 1
    cf_spectral_efficiency = np.zeros(K)
    for k in range(K):
        cf_spectral_efficiency = compute_se_cf_ul(max_power, UEs_power, K, tau_p, tau_c, pilot_index_CF, beta_val)
        min_SE_UE = np.argmin(cf_spectral_efficiency)
        pilot_index_CF[min_SE_UE] = -1

        pilot_interference_CF = np.zeros(tau_p)
        for i in range(tau_p):
            mask = pilot_index_CF == i
            pilot_interference_CF[i] = np.sum(beta_val[:, mask], axis=(0, 1))

        min_interference_pilot = np.argmin(pilot_interference_CF)
        pilot_index_CF[min_SE_UE] = min_interference_pilot

    max_power_tau_p = max_power * tau_p
    beta_val_squared = beta_val ** 2
    gamma_val = np.zeros((L, K))
    cf_interference = np.zeros((K, K))

    for k in range(K):
        mask_CF = pilot_index_CF[k] == pilot_index_CF
        denominator = max_power_tau_p * np.sum(beta_val[:, mask_CF], axis=1) + 1
        gamma_val[:, k] = (max_power_tau_p * beta_val_squared[:, k]) / denominator
        noise_term = np.sum(gamma_val[:, k])
        cf_interference[:, k] += np.sum(np.tile(gamma_val[:, k], [K, 1]).T * beta_val, axis=0) / noise_term

        for ind in np.where(mask_CF)[0]:
            if ind != k:
                cf_interference[ind, k] += np.sum(
                    gamma_val[:, k] * beta_val[:, ind] / beta_val[:, k]) ** 2 / noise_term

    cf_signal = np.sum(gamma_val, axis=0)
    Beta_K = np.sum(beta_val, axis=0).astype(np.float32)

    return (Beta_K, cf_signal, cf_interference, cf_spectral_efficiency, pilot_index_CF, beta_val, APs_positions,
            UEs_positions)
