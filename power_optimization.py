from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from _utils import feasibility_problem_cvx, compute_prod_sinr
from compute_spectral_efficiency import compute_se


def power_opt_maxmin(signal: np.ndarray,
                     interference: np.ndarray,
                     max_power: float,
                     prelog_factor: float,
                     return_spectral_efficiency: bool = False) \
        -> Union[Tuple[Optional[float], np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Optimizes power allocation to maximize the minimum rate among users under a max power constraint.

    :param signal: An array containing the signal levels for each user.
    :param interference: An array containing the interference levels for each user.
    :param max_power: The maximum power that can be allocated to a user.
    :param prelog_factor: The prelog factor used in the spectral efficiency computation.
    :param return_spectral_efficiency: A flag to indicate whether to return the spectral efficiency along
                                       with the optimal power allocation. Defaults to False.
    :return: If return_spectral_efficiency is True, returns a tuple containing the spectral efficiency and the optimal
             power allocation for each user. Otherwise, returns None for spectral efficiency and the optimal power
             allocation.
    """

    K = signal.shape[0]

    lower_rate = 0
    upper_rate = np.log2(1 + max_power * np.min(signal))

    delta = 0.01

    p_best = np.zeros(K)

    while np.linalg.norm(upper_rate - lower_rate) > delta:
        candidate_rate = (upper_rate + lower_rate) / 2
        candidate_sinr = (2 ** candidate_rate) - 1

        feasible, p_solution = feasibility_problem_cvx(signal, interference, max_power, K, candidate_sinr)

        if feasible:
            lower_rate = candidate_rate
            p_best = p_solution
        else:
            upper_rate = candidate_rate

    if return_spectral_efficiency:
        SE = compute_se(signal, interference, p_best, prelog_factor)
        return SE, p_best
    else:
        return None, p_best


def power_opt_prod_sinr(signal: np.ndarray,
                        interference: np.ndarray,
                        max_power: float,
                        prelog_factor: float,
                        return_spectral_efficiency: bool = False
                        ) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    """
    Optimizes power allocation to maximize the product of SINR among users under a maximum power constraint.

    :param signal: An ndarray containing the signal power levels for each user.
    :param interference: An ndarray containing the interference power levels for each user.
    :param max_power: The maximum total power that can be allocated across all users.
    :param prelog_factor: A factor used in the spectral efficiency computation, representing bandwidth and noise.
    :param return_spectral_efficiency: A flag indicating whether to return the spectral efficiency along with the
                                       optimal power allocation. Defaults to False.
    :return: A tuple where the first element is the spectral efficiency (if requested) or None, and the second element
             is an ndarray of the optimal power allocation for each user.

    The function uses a gradient-based method (L-BFGS-B) to find power allocations that maximize the product of users'
    SINR, considering the given power constraints.
    """
    # if not isinstance(signal, np.ndarray):
    #     signal = signal.cpu().numpy()
    # if not isinstance(interference, np.ndarray):
    #     interference = interference.cpu().numpy()

    K = signal.shape[0]

    rhoSolution = np.full(K, max_power / K)

    bounds = [(1e-6, max_power)] * K

    def objective_func(rho):
        return -np.sum(compute_prod_sinr(signal, interference, rho, prelog_factor))

    result = minimize(objective_func, rhoSolution, bounds=bounds, method='L-BFGS-B')
    rhoBest = result.x

    # Compute SE with the initial solution
    if return_spectral_efficiency:
        SE = compute_se(signal, interference, rhoSolution, prelog_factor)
        return SE, rhoBest
    else:
        return None, rhoBest


def power_opt_sum_rate(signal: np.ndarray,
                       interference: np.ndarray,
                       max_power: float,
                       prelog_factor: float,
                       return_spectral_efficiency: bool = False
                       ) -> Tuple[Optional[np.ndarray, None], np.ndarray]:
    """
    Optimizes power allocation to maximize the sum rate among users under a maximum power constraint.

    :param signal: An ndarray containing the signal power levels for each user.
    :param interference: An ndarray containing the interference power levels for each user.
    :param max_power: The maximum total power that can be allocated across all users.
    :param prelog_factor: A factor used in the spectral efficiency computation, representing bandwidth and noise.
    :param return_spectral_efficiency: A flag indicating whether to return the spectral efficiency along with the
                                       optimal power allocation. Defaults to False.
    :return: A tuple where the first element is the spectral efficiency (if requested) or None, and the second element
             is an ndarray of the optimal power allocation for each user.

    The function employs a gradient-based optimization method (L-BFGS-B) to solve the power allocation problem,
    targeting the maximization of the total spectral efficiency subject to individual power constraints for each user.
    """
    # if not isinstance(signal, np.ndarray):
    #     signal = signal.cpu().numpy()
    # if not isinstance(interference, np.ndarray):
    #     interference = interference.cpu().numpy()

    K = signal.shape[0]

    # Nonlinear optimization

    rhoSolution = np.full(K, max_power / K)

    bounds = [(0, max_power)] * K

    def objective_func(rho):
        return -np.sum(compute_se(signal, interference, rho, prelog_factor))

    result = minimize(objective_func, rhoSolution, bounds=bounds, method='L-BFGS-B')

    rhoBest = result.x

    # Compute the SEs
    if return_spectral_efficiency:
        SE = compute_se(signal, interference, rhoBest, prelog_factor)
        return SE, rhoBest
    else:
        return None, rhoBest
