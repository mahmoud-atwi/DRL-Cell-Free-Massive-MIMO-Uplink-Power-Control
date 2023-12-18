import time
import torch
import numpy as np

from scipy.optimize import minimize

from helper_functions import feasibility_problem_cvx_np, compute_prod_sinr
from compute_spectral_efficiency import compute_se, compute_se_np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def power_opt_maxmin(signal, interference, max_power, prelog_factor, return_SE=False):

    if not isinstance(signal, np.ndarray):
        signal = signal.cpu().numpy()
    if not isinstance(interference, np.ndarray):
        interference = interference.cpu().numpy()

    K = signal.shape[0]

    lower_rate = 0
    upper_rate = np.log2(1 + max_power * np.min(signal))

    delta = 0.01

    p_best = np.zeros(K)

    while np.linalg.norm(upper_rate - lower_rate) > delta:
        candidate_rate = (upper_rate + lower_rate) / 2
        candidate_SINR = (2 ** candidate_rate) - 1

        feasible, p_solution = feasibility_problem_cvx_np(signal, interference, max_power, K, candidate_SINR)

        if feasible:
            lower_rate = candidate_rate
            p_best = p_solution
        else:
            upper_rate = candidate_rate

    if return_SE:
        SE = compute_se_np(signal, interference, p_best, prelog_factor)
        return SE, p_best
    else:
        return None, p_best


def power_opt_prod_sinr(signal, interference, max_power, prelog_factor, return_SE=False):

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
    if return_SE:
        SE = compute_se_np(signal, interference, rhoSolution, prelog_factor)
        return SE, rhoBest
    else:
        return None, rhoBest


def power_opt_sum_rate(signal, interference, max_power, prelog_factor, return_SE=False):

    if not isinstance(signal, np.ndarray):
        signal = signal.cpu().numpy()
    if not isinstance(interference, np.ndarray):
        interference = interference.cpu().numpy()

    K = signal.shape[0]

    # Nonlinear optimization

    rhoSolution = np.full(K, max_power / K)

    bounds = [(0, max_power)] * K

    def objective_func(rho):
        return -np.sum(compute_se_np(signal, interference, rho, prelog_factor))

    result = minimize(objective_func, rhoSolution, bounds=bounds, method='L-BFGS-B')

    rhoBest = result.x

    # Compute the SEs
    if return_SE:
        SE = compute_se_np(signal, interference, rhoBest, prelog_factor)
        return SE, rhoBest
    else:
        return None, rhoBest


if __name__ == '__main__':
    _K = 40

    _signal = torch.randn(_K, dtype=torch.float32, device=device)
    _interference = torch.randn(_K, _K, dtype=torch.float32, device=device)
    _max_power = 100
    _prelog_factor = 1

    print('power_opt_maxmin started')
    start = time.time()
    _SE, _p_best = power_opt_maxmin(_signal, _interference, _max_power, _prelog_factor, return_SE=True)
    print(f'power_opt_maxmin Done | Time elapsed: {time.time() - start}')
    # print SE value and shape
    print(f'SE: {_SE}')
    print(f'SE: {_SE.shape}')
    # print p_best value and shape
    print(f'p_best: {_p_best}')
    print(f'p_best: {_p_best.shape}')

    _SE, _rhoBest = power_opt_prod_sinr(_signal, _interference, _max_power, _prelog_factor, return_SE=True)
    # print SE value and shape
    print(f'_SE: {_SE}')
    print(f'_SE: {_SE.shape}')
    # print rhoBest value and shape
    print(f'_rhoBest: {_rhoBest}')
    print(f'_rhoBest: {_rhoBest.shape}')

    print('power_opt_sum_rate started')
    start = time.time()
    _SE, _rhoBest = power_opt_sum_rate(_signal, _interference, _max_power, _prelog_factor, return_SE=True)
    print(f'power_opt_sum_rate Done | Time elapsed: {time.time() - start}')
    # print SE value and shape
    print(f'_SE: {_SE}')
    print(f'_SE: {_SE.shape}')
    # print rhoBest value and shape
    print(f'_rhoBest: {_rhoBest}')
    print(f'_rhoBest: {_rhoBest.shape}')
