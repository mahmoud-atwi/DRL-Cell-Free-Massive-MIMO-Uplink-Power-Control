import os
import socket
import subprocess
import importlib
import time
import torch
import cvxpy as cp
import numpy as np
from typing import Tuple

from scipy.linalg import sqrtm

from typing import Any, Dict, List, Type, Optional

import optuna

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

import warnings
warnings.simplefilter('error', RuntimeWarning)


def get_class_by_name(name: str) -> Type:
    """
    Imports and returns a class given the name, e.g. passing
    'stable_baselines3.common.callbacks.CheckpointCallback' returns the
    CheckpointCallback class.

    :param name:
    :return:
    """

    def get_module_name(cls_name: str) -> str:
        return ".".join(cls_name.split(".")[:-1])

    def get_class_name(cls_name: str) -> str:
        return cls_name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyperparameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - rl_zoo3.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    callbacks: List[BaseCallback] = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = next(iter(callback_dict.keys()))
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}

            callback_class = get_class_by_name(callback_name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
            self,
            eval_env: VecEnv,
            trial: optuna.Trial,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 2,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def get_machine_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return "localhost"


def launch_optuna_dashboard(script_dir, storage_url, port=80):
    host = get_machine_ip()
    os.chdir(script_dir)
    bash_cmd = f'optuna-dashboard {storage_url} --port {port} --host {host}'
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())


def check_and_launch_dashboard(script_dir, storage_url, port, check_interval=5):
    """
    Checks for the existence of the SQLite database file and launches the dashboard.
    """
    while True:
        if os.path.exists(storage_url.split("///")[1]):
            launch_optuna_dashboard(script_dir, storage_url, port)
            break
        time.sleep(check_interval)


def db2pow(db):
    """Convert dB to linear scale power."""
    return np.power(10, db / 10)


def dbm2mW(dbm):
    """Convert dBm to mW."""
    return 10 ** (dbm / 10)


def toeplitz_pytorch(c, r=None):
    """
    Create a Toeplitz matrix using PyTorch.

    Args:
    c (torch.Tensor): 1D tensor containing the first column of the matrix.
    r (torch.Tensor): 1D tensor containing the first row of the matrix.
                      If None, r is taken as same as c.

    Returns:
    torch.Tensor: Toeplitz matrix
    """
    if r is None:
        r = c
    else:
        if r[0] != c[0]:
            raise ValueError("First element of input column must be equal to first element of input row.")

    c_size = c.size(0)
    r_size = r.size(0)
    a, b = torch.meshgrid(torch.arange(c_size), torch.arange(r_size), indexing='ij')
    idx_matrix = a - b + (r_size - 1)

    # Use flattened c and r to fill in the Toeplitz matrix
    flattened = torch.cat((r.flip(0)[1:], c))
    return flattened[idx_matrix]


def get_sqrtm(matrix, method='newton_schulz', iters=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if method == 'cholesky':
        try:
            # Use Cholesky decomposition in PyTorch for positive definite matrices
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            # Fallback to Newton-Schulz if Cholesky fails
            return sqrtm_newton_schulz(matrix, iters)[0]
    elif method == 'newton_schulz':
        # Use Newton-Schulz Iterative method
        return sqrtm_newton_schulz(matrix, iters)[0]
    else:
        # If not positive definite, use SciPy sqrtm on CPU and transfer to PyTorch
        return torch.tensor(sqrtm(matrix.cpu().numpy()), dtype=torch.complex64).to(device)


def sqrtm_newton_schulz(matrix, num_iters):
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

    return Y * torch.sqrt(norm_of_matrix), None


def feasibility_problem_cvx(signal, interference, max_power, K, sinr_constraint):
    rho = cp.Variable(K)
    scaling = cp.Variable()

    objective = cp.Minimize(scaling)

    constraints = []

    for k in range(K):
        constraints.append(
            sinr_constraint * (cp.sum(cp.multiply(rho, interference[:, k])) + 1) - (rho[k] * signal[k]) <= 0)
        constraints.append(rho[k] >= 0)
        constraints.append(rho[k] <= scaling * max_power)

    constraints.append(scaling >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    # Analyze the output and prepare the output variables
    if problem.status not in ["infeasible", "unbounded"]:
        if scaling.value > 1:  # Only the power minimization problem is feasible
            feasible = False
            rhoSolution = rho.value
        else:  # Both the power minimization problem and feasibility problem are feasible
            feasible = True
            rhoSolution = rho.value
    else:  # Both problems are infeasible
        feasible = False
        rhoSolution = None

    return feasible, rhoSolution


def compute_prod_sinr(signal, interference, rho, prelog_factor):
    SINR = signal * rho / (np.dot(interference.T, rho) + 1)
    prod_SINR = prelog_factor * np.log2(SINR)
    return prod_SINR


def calc_SINR(power, signal, interference):
    # Calculate the denominator and numerator using NumPy operations
    denominator = np.dot(interference, power) + 1
    numerator = np.multiply(signal, power)

    # Calculate SINR
    SINR = numerator / denominator

    return SINR


def calculate_max_aps(area_bounds: Tuple[float, float, float, float], min_inter_distance: float) -> float:
    """
    Calculate the maximum number of Access Points (APs) that can be placed within a given area
    while maintaining a minimum inter-AP distance.

    :param area_bounds: Tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :param min_inter_distance: Minimum distance required between any two APs.
    :return: Maximum number of APs that can be placed within the given area.
    """
    x_min, x_max, y_min, y_max = area_bounds
    area_width = x_max - x_min
    area_height = y_max - y_min

    # Estimate the area required per AP, approximating each AP's influence area as a circle
    ap_area = np.pi * (min_inter_distance / 2) ** 2

    # Calculate the total available area
    total_area = area_width * area_height

    # Estimate the maximum number of APs
    max_aps = total_area / ap_area
    return max_aps


def generate_ap_locations(num_aps: int, min_inter_distance: float, area_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Generate random AP locations as complex numbers with a minimum inter-AP distance.
    Raises an error if it is not possible to place all APs with the given constraints.

    :param num_aps: Number of APs to generate.
    :param min_inter_distance: Minimum distance between any two APs.
    :param area_bounds: Tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :return: NumPy array of AP locations in complex number format.
    """
    if num_aps > calculate_max_aps(area_bounds, min_inter_distance):
        raise ValueError("Too many APs for the given area and minimum distance constraints.")

    # Initialize an array to store AP locations
    ap_locations = np.zeros(num_aps, dtype=np.complex64)
    x_min, x_max, y_min, y_max = area_bounds

    for i in range(num_aps):
        while True:
            location = (np.random.uniform(x_min, x_max) +
                        1j * np.random.uniform(y_min, y_max))

            if i == 0 or np.all(np.abs(ap_locations[:i] - location) >= min_inter_distance):
                ap_locations[i] = location
                break

    return ap_locations


def generate_ue_locations(num_ues: int, area_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Generate random UE locations within a bounded area.

    :param num_ues: Number of UEs to generate.
    :param area_bounds: Tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :return: NumPy array of UE locations in complex number format.
    """
    x_min, x_max, y_min, y_max = area_bounds

    # Initialize an array to store UE locations
    ue_locations = np.zeros(num_ues, dtype=np.complex128)

    # Generate random locations for each UE
    ue_locations.real = np.random.uniform(x_min, x_max, num_ues)
    ue_locations.imag = np.random.uniform(y_min, y_max, num_ues)

    return ue_locations
