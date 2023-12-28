import os
import time
import torch
import socket
import optuna
import warnings
import importlib
import subprocess
import cvxpy as cp
import numpy as np

from scipy.linalg import sqrtm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from typing import Any, Dict, List, Type, Optional, Union, Tuple

warnings.simplefilter('error', RuntimeWarning)


def get_class_by_name(name: str) -> Type:
    """
    Given the fully qualified name of a class, imports and returns the class.

    This function dynamically imports a class from its fully qualified name, which includes
    the module and class name. For example, providing 'module.submodule.ClassName' would
    import and return 'ClassName' from 'module.submodule'.

    :param name: The fully qualified name of the class to import.
    :return: The class object corresponding to the given name.
    """

    def get_module_name(cls_name: str) -> str:
        return ".".join(cls_name.split(".")[:-1])

    def get_class_name(cls_name: str) -> str:
        return cls_name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Constructs a list of callback instances from callback class names specified in the hyperparameters.

    This function supports both single and multiple callback class names. The class names must be fully qualified.
    It also handles keyword arguments for class instantiation if provided.

    :param hyperparams: A dictionary containing the hyperparameters, which may include 'callback' keys.
    :return: A list of instantiated callback objects.
    """
    callbacks: List[BaseCallback] = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        callback_names = [callback_name] if not isinstance(callback_name, list) else callback_name

        for callback_name in callback_names:
            kwargs = {}
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, f"Error in YAML formatting near {callback_name}. Check indentation."
                callback_dict = callback_name
                callback_name = next(iter(callback_dict.keys()))
                kwargs = callback_dict[callback_name]

            callback_class = get_class_by_name(callback_name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


class TrialEvalCallback(EvalCallback):
    """
    A custom callback for Optuna that evaluates and reports the performance of a trial.

    This callback extends the standard EvalCallback of Stable Baselines3, adding functionality
    to report the evaluation results to an Optuna trial and check for pruning conditions.

    :param eval_env: The environment used for evaluation.
    :param trial: The Optuna trial object.
    :param eval_freq: Frequency of evaluations.
    :param deterministic: Whether to use deterministic actions during evaluation.
    :param verbose: Verbosity level.
    :param best_model_save_path: Path to save the best model.
    :param log_path: Path for logging.
    """

    def __init__(
            self,
            eval_env: VecEnv,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 2,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
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
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def get_machine_ip() -> str:
    """
    Retrieves the machine's public IP address.

    Tries to establish a UDP connection to a public DNS server (8.8.8.8) to determine the public IP address
    of the machine. If unsuccessful, returns 'localhost'.

    :return: The public IP address of the machine, or 'localhost' if an error occurs.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return "localhost"


def launch_optuna_dashboard(script_dir: str, storage_url: str, port: Optional[int] = 80) -> None:
    """
    Launches an Optuna dashboard.

    Changes the current working directory to the specified script directory, then runs a bash command to
    launch the Optuna dashboard using the given storage URL and port.

    :param script_dir: Directory where the Optuna dashboard script is located.
    :param storage_url: URL of the SQLite database used by Optuna.
    :param port: Port number to use for the dashboard. Default is 80.
    """
    host = get_machine_ip()
    os.chdir(script_dir)
    bash_cmd = f'optuna-dashboard {storage_url} --port {port} --host {host}'
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())


def check_and_launch_dashboard(script_dir: str, storage_url: str, port: Optional[int] = 80,
                               check_interval: Optional[int] = 5) -> None:
    """
    Periodically checks for the existence of a SQLite database file and launches the Optuna dashboard.

    Continuously checks at specified intervals for the presence of a SQLite database file at the given storage URL.
    Once the file is found, it launches the Optuna dashboard using the specified script directory and port.

    :param script_dir: Directory where the Optuna dashboard script is located.
    :param storage_url: URL of the SQLite database used by Optuna.
    :param port: Port number to use for the dashboard. Default is 80.
    :param check_interval: Time interval (in seconds) between checks for the database file. Default is 5 seconds.
    """
    while True:
        if os.path.exists(storage_url.split("///")[1]):
            launch_optuna_dashboard(script_dir, storage_url, port)
            break
        time.sleep(check_interval)


def db2pow(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts a power ratio from decibels (dB) to a linear scale.

    :param db: Power ratio in decibels (dB).
    :return: Equivalent power ratio on a linear scale.
    """
    return np.power(10, db / 10)


def dbm2mW(dbm: float) -> float:
    """
    Converts a power value from decibel-milliwatts (dBm) to milliwatts (mW).

    :param dbm: Power value in decibel-milliwatts (dBm).
    :return: Equivalent power in milliwatts (mW).
    """
    return 10 ** (dbm / 10)


def toeplitz_pytorch(c: torch.Tensor, r: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Creates a Toeplitz matrix using PyTorch.

    :param c: A 1D tensor containing the first column of the matrix.
    :param r: An optional 1D tensor containing the first row of the matrix. If None, r is assumed to be the same as c.
              The first element of c must be equal to the first element of r.
    :return: A Toeplitz matrix constructed from the input column and row tensors.

    Raises:
        ValueError: If the first element of the input column is not equal to the first element of the input row.
    """
    if r is None:
        r = c
    elif r[0] != c[0]:
        raise ValueError("First element of input column must be equal to first element of input row.")

    c_size = c.size(0)
    r_size = r.size(0)
    a, b = torch.meshgrid(torch.arange(c_size), torch.arange(r_size), indexing='ij')
    idx_matrix = a - b + (r_size - 1)

    # Use flattened c and r to fill in the Toeplitz matrix
    flattened = torch.cat((r.flip(0)[1:], c))
    return flattened[idx_matrix]


def get_sqrtm(matrix: torch.Tensor, method: str = 'newton_schulz', iters: int = 10) -> torch.Tensor:
    """
    Computes the matrix square root using the specified method.

    :param matrix: A PyTorch tensor representing the matrix for which the square root is to be computed.
    :param method: The method to use for computing the square root ('newton_schulz', 'cholesky', or other).
                   Uses the Newton-Schulz iterative method by default, and Cholesky decomposition for positive
                   definite matrices if specified. For non-positive definite matrices, SciPy's sqrtm is used.
    :param iters: The number of iterations for the Newton-Schulz algorithm. Default is 10.
    :return: A PyTorch tensor representing the square root of the input matrix.
    """
    if method == 'cholesky':
        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            return sqrtm_newton_schulz(matrix, iters)
    elif method == 'newton_schulz':
        return sqrtm_newton_schulz(matrix, iters)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(sqrtm(matrix.cpu().numpy()), dtype=torch.complex64).to(device)


def sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int) -> torch.Tensor:
    """
    Computes the matrix square root using the Newton-Schulz iterative method.

    :param matrix: A square PyTorch tensor representing the matrix for which the square root is to be computed.
    :param num_iters: The number of iterations to perform in the Newton-Schulz algorithm.
    :return: A PyTorch tensor representing the square root of the input matrix.
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

    return Y * torch.sqrt(norm_of_matrix)


def feasibility_problem_cvx(signal: np.ndarray, interference: np.ndarray, max_power: float, K: int,
                            sinr_constraint: float) -> Tuple[bool, Union[np.ndarray, None]]:
    """
    Employs convex optimization to ascertain the feasibility of a solution under a SINR constraint and delivers the
    optimal power allocation if the solution is viable.

    :param signal: An array of signal power values.
    :param interference: A matrix of interference values.
    :param max_power: The maximum power that can be allocated.
    :param K: The number of users or elements in the system.
    :param sinr_constraint: The SINR constraint that needs to be satisfied.
    :return: A tuple (feasible, rhoSolution) where 'feasible' is a boolean indicating if the problem is feasible,
             and 'rhoSolution' is an array representing the power allocation if feasible, or None if infeasible.
    """
    # Initialize CVXPY variables
    rho = cp.Variable(K)
    scaling = cp.Variable()

    # Define the objective function
    objective = cp.Minimize(scaling)

    # Initialize constraints
    constraints = []

    # Add constraints for each user
    for k in range(K):
        constraints.append(
            sinr_constraint * (cp.sum(cp.multiply(rho, interference[:, k])) + 1) - (rho[k] * signal[k]) <= 0)
        constraints.append(rho[k] >= 0)
        constraints.append(rho[k] <= scaling * max_power)

    # Add constraint for scaling
    constraints.append(scaling >= 0)

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    # Analyze the output and prepare the output variables
    if problem.status not in ["infeasible", "unbounded"]:
        feasible = scaling.value <= 1
        rhoSolution = rho.value
    else:
        feasible = False
        rhoSolution = None

    return feasible, rhoSolution


def compute_prod_sinr(signal: np.ndarray, interference: np.ndarray, rho: np.ndarray,
                      prelog_factor: float) -> np.ndarray:
    """
    Calculate the product of the Signal-to-Interference-plus-Noise Ratio (SINR) and a pre-logarithmic factor.

    :param signal: An array of signal power values.
    :param interference: A matrix of interference values.
    :param rho: An array representing power allocation coefficients.
    :param prelog_factor: A scalar representing the pre-logarithmic factor.
    :return: An array representing the product of SINR values and the pre-logarithmic factor.
    """
    sinr = signal * rho / (np.dot(interference.T, rho) + 1)
    prod_sinr = prelog_factor * np.log2(sinr)
    return prod_sinr


def calc_sinr(power: np.ndarray, signal: np.ndarray, interference: np.ndarray) -> np.ndarray:
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

    :param power: An array of power values.
    :param signal: An array of signal values.
    :param interference: An array of interference values.
    :return: An array representing the SINR values.
    """
    # Calculate the denominator and numerator
    denominator = np.dot(interference, power) + 1
    numerator = np.multiply(signal, power)

    # Calculate SINR
    sinr = numerator / denominator

    return sinr


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


def generate_ap_locations(num_aps: int, min_inter_distance: float,
                          area_bounds: Tuple[float, float, float, float]) -> np.ndarray:
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
