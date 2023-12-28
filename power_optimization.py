import numpy as np

from scipy.optimize import minimize

from _utils import feasibility_problem_cvx, compute_prod_sinr
from compute_spectral_efficiency import compute_se


def power_opt_maxmin(signal, interference, max_power, prelog_factor, return_spectral_efficiency=False):

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


def power_opt_prod_sinr(signal, interference, max_power, prelog_factor, return_spectral_efficiency=False):

    if not isinstance(signal, np.ndarray):
        signal = signal.cpu().numpy()
    if not isinstance(interference, np.ndarray):
        interference = interference.cpu().numpy()

    K = signal.shape[0]

    rhoSolution = np.full(K, max_power/K)

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


def power_opt_sum_rate(signal, interference, max_power, prelog_factor, return_spectral_efficiency=False):

    if not isinstance(signal, np.ndarray):
        signal = signal.cpu().numpy()
    if not isinstance(interference, np.ndarray):
        interference = interference.cpu().numpy()

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
    