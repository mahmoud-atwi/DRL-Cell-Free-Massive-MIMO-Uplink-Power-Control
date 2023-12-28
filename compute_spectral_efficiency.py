import numpy as np

from typing import Union, Iterable


def compute_se_cf_ul(max_power: float,
                     p: Union[float, np.ndarray],
                     K: int,
                     tau_p: int,
                     tau_c: int,
                     pilot_index: np.ndarray,
                     beta_val: np.ndarray) -> np.ndarray:
    """
    Computes the uplink spectral efficiency in a cell-free (CF) system using a user-centric approach.

    This function calculates the spectral efficiency for each user in a cell-free massive MIMO system
    with a given set of system parameters. It accounts for pilot contamination and other interference
    factors to provide a realistic measure of the spectral efficiency.

    :param max_power: The maximum transmit power.
    :param p: A scalar or a numpy array representing the power control coefficients for each user.
              It can be a single value applied to all users or an array of values, one for each user.
    :param K: The number of users in the system.
    :param tau_p: The length of the pilot sequence.
    :param tau_c: The coherence interval.
    :param pilot_index: A numpy array indicating the pilot index for each user.
    :param beta_val: A numpy array representing the large-scale fading coefficients.
    :return: A numpy array containing the spectral efficiency for each user.
    """
    # Ensure p is an array
    if isinstance(p, (int, float)):
        p_array = np.full((K,), p, dtype=np.float32)
    elif isinstance(p, np.ndarray):
        if p.shape == ():
            p_scalar = p.item()  # Convert NumPy scalar to Python scalar
            p_array = np.full((K,), p_scalar, dtype=np.float32)
        elif p.shape == (K,):
            p_array = p
        else:
            raise ValueError('p should be either a single value or an array with size K')
    else:
        raise ValueError('p should be either a single value or a NumPy array with size K')

    # Ensure pilot_index is an array
    if not isinstance(pilot_index, np.ndarray):
        raise ValueError('pilot_index should be a NumPy array')

    # Vectorized gamma calculation
    mask = np.equal.outer(pilot_index, pilot_index)  # Shape: [K, K]
    interference_term = max_power * tau_p * np.sum(beta_val[:, np.newaxis] * mask, axis=2)
    gamma_val = max_power * tau_p * np.power(beta_val, 2) / (interference_term + 1)  # Broadcasting over L

    # Compute signal term
    signal = p_array * np.power(np.sum(gamma_val, axis=0), 2)

    interference = np.zeros((K,), dtype=np.float32)

    for k in range(K):
        interference[k] += np.sum(gamma_val[:, k] * np.sum(p_array * beta_val, axis=1)) + np.sum(gamma_val[:, k])

        # Compute interference due to pilot contamination
        co_pilot = pilot_index[k] == pilot_index
        co_pilot[k] = False
        same_pilot_indices = np.where(co_pilot)[0]
        for index in same_pilot_indices:
            interference[k] += (p_array[index] *
                                np.power(np.sum(gamma_val[:, k] * beta_val[:, index] / beta_val[:, k]), 2))

    # Compute UL CF Spectral Efficiency
    cf_spectral_efficiency = (1 - tau_p / tau_c) * np.log2(1 + signal / interference)
    return cf_spectral_efficiency


def compute_se(signal: Union[np.ndarray, Iterable[float]],
               interference: Union[np.ndarray, Iterable[float]],
               rho: Union[np.ndarray,
               Iterable[float]],
               prelog_factor: float) -> np.ndarray:
    """
    Computes the spectral efficiency using the Signal-to-Interference-plus-Noise Ratio (SINR).

    This function calculates the spectral efficiency for a given set of signal, interference,
    and power allocation coefficients (rho), along with a pre-logarithmic factor. It first computes
    the SINR and then uses it to calculate the spectral efficiency.

    :param signal: An array or iterable of signal power values.
    :param interference: A matrix of interference values.
    :param rho: An array or iterable of power allocation coefficients.
    :param prelog_factor: A scalar pre-logarithmic factor used in spectral efficiency calculation.
    :return: An array representing the spectral efficiency for each element in the signal array.
    """
    sinr = signal * rho / (interference.T @ rho + 1)
    spectral_efficiency = prelog_factor * np.log2(1 + sinr)
    return spectral_efficiency
