import importlib
import os
import socket
import subprocess
import time
import warnings
from typing import Any, Dict, List, Type, Optional, Union, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from scipy.linalg import sqrtm
from scipy.stats import stats
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

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


def plot_cdf_pdf(data: Dict[str, Dict[str, Union[str, np.ndarray, pd.DataFrame, pd.Series]]],
                 title: str, xlabel: Optional[str], operation: Optional[str], cumulative: bool) -> None:
    """
    Plot CDF or PDF for multiple models using matplotlib, with an optional operation applied across iterations.
    """
    operations = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    if operation is not None and operation not in operations:
        raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")

    plt.figure(figsize=(12, 6))

    for key, info in data.items():
        value = info['data']

        if isinstance(value, (pd.DataFrame, pd.Series)):
            value = value.to_numpy()

        if operation is not None:
            value = operations[operation](value, axis=0)  # Apply operation across iterations

        value = value.flatten()

        if cumulative:
            sorted_data = np.sort(value)
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, yvals,
                     label=info['label'],
                     color=info.get('color'),
                     linestyle=info.get('linestyle'),
                     marker=info.get('marker', None),
                     linewidth=info.get('linewidth'))
        else:
            if len(np.unique(value)) > 1:
                bins = np.linspace(np.min(value), np.max(value), 30)
                counts, bin_edges = np.histogram(value, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, counts,
                         label=info['label'],
                         color=info.get('color'),
                         linestyle=info.get('linestyle'),
                         marker=info.get('marker', None),
                         linewidth=info.get('linewidth'))
            else:
                print(f"Data for {key} lacks variability, skipping PDF plot.")

    plt.xlabel('Spectral Efficiency (SE)' if xlabel is None else xlabel)
    plt.ylabel('Cumulative Distribution' if cumulative else 'Probability Density')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center right')
    plt.grid(True)
    plt.show()


def compare_models(models_data: Dict[str, Dict[str, Union[str, np.ndarray, pd.DataFrame, pd.Series]]], data_label: str,
                   operation: Optional[str] = None):
    """
    Compare multiple models based on their spectral efficiency.

    :param operation:
    :param models_data: A dictionary where keys are model names and the values are dictionaries with 'label' and 'data'
                        keys.
    :return: A DataFrame with comparison metrics for each model.
    """
    comparison_metrics = {}

    operations = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    if operation is not None and operation not in operations:
        raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")

    for model_name, model_info in models_data.items():
        se_values = model_info['data'].to_numpy()  # Assuming the data is a DataFrame
        if operation is not None:
            se_values = operations[operation](se_values, axis=0)  # Apply operation across iterations
        se_values = se_values.flatten()
        comparison_metrics[model_name] = {
            'Label': model_info['label'],  # Include the label in the comparison
            f'Average {data_label}': np.mean(se_values),
            f'Max {data_label}': np.max(se_values),
            'Standard Deviation': np.std(se_values),
            '25th Percentile': np.percentile(se_values, 25),
            '50th Percentile': np.percentile(se_values, 50),
            '75th Percentile': np.percentile(se_values, 75),
        }

    return pd.DataFrame(comparison_metrics).T


def compare_cdfs_ks(data_dict: Dict[str, Dict[str, Union[str, np.ndarray, pd.DataFrame, pd.Series]]],
                    operation: Optional[str] = None):
    comparison_data = []
    keys = list(data_dict.keys())

    operations = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    if operation is not None and operation not in operations:
        raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")

    # Pairwise KS Test and Area between CDFs
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # Extract and process data
            data_i = data_dict[keys[i]]['data'].to_numpy().flatten()
            data_j = data_dict[keys[j]]['data'].to_numpy().flatten()

            # Apply operation if specified
            if operation is not None:
                data_i = operations[operation](data_i, axis=0)
                data_j = operations[operation](data_j, axis=0)

            # KS Test
            ks_stat, ks_pvalue = stats.ks_2samp(data_i, data_j)

            # Area between CDFs
            sorted_i = np.sort(data_i)
            sorted_j = np.sort(data_j)
            cdf_i = np.arange(1, len(sorted_i) + 1) / len(sorted_i)
            cdf_j = np.arange(1, len(sorted_j) + 1) / len(sorted_j)
            area = np.trapz(np.abs(np.interp(sorted_j, sorted_i, cdf_i) - cdf_j), sorted_j)

            # Append results to comparison_data
            comparison_data.append({
                'Model1': data_dict[keys[i]]["label"],
                'Model2': data_dict[keys[j]]["label"],
                'KS Statistic': ks_stat,
                'P-Value': ks_pvalue,
                'Area Between CDFs': area
            })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Determine the best model
    best_model = determine_best_model_ks(comparison_df)

    results = {
        'Best Model': best_model,
        'Details': comparison_df
    }
    return results


def determine_best_model_ks(comparison_df):
    # Initialize dictionaries to store aggregated values
    ks_stats = {}
    p_values = {}
    areas = {}

    # Aggregate KS statistic, P-value, and area for each model
    for _, row in comparison_df.iterrows():
        models = [row['Model1'], row['Model2']]
        for model in models:
            ks_stats.setdefault(model, []).append(row['KS Statistic'])
            p_values.setdefault(model, []).append(row['P-Value'])
            areas.setdefault(model, []).append(row['Area Between CDFs'])

    # Compute average values for each model
    avg_ks = {k: np.mean(v) for k, v in ks_stats.items()}
    avg_p = {k: np.mean(v) for k, v in p_values.items()}
    avg_area = {k: np.mean(v) for k, v in areas.items()}

    # Determine best model based on criteria
    best_model_ks = min(avg_ks, key=avg_ks.get)
    best_model_p = max(avg_p, key=avg_p.get)
    best_model_area = min(avg_area, key=avg_area.get)

    # Create a DataFrame for best model results
    best_model_df = pd.DataFrame({
        'Criteria': ['KS Statistic', 'P-Value', 'Area Between CDFs'],
        'Best Model': [best_model_ks, best_model_p, best_model_area]
    })

    return best_model_df


def compare_cdfs_emd(data_dict: Dict[str, Dict[str, Union[str, np.ndarray, pd.DataFrame, pd.Series]]],
                     operation: Optional[str] = None):
    """
    Compare multiple models based on Earth Mover's Distance (EMD).
    """
    comparison_data = []
    keys = list(data_dict.keys())

    operations = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    if operation is not None and operation not in operations:
        raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")

    # Pairwise EMD comparison
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # Extract and process data
            data_i = data_dict[keys[i]]['data'].to_numpy().flatten()
            data_j = data_dict[keys[j]]['data'].to_numpy().flatten()

            # Apply operation if specified
            if operation is not None:
                data_i = operations[operation](data_i, axis=0)
                data_j = operations[operation](data_j, axis=0)

            # Calculate EMD
            emd_value = stats.wasserstein_distance(data_i, data_j)
            comparison_data.append({
                'Model1': data_dict[keys[i]]["label"],
                'Model2': data_dict[keys[j]]["label"],
                'EMD Value': emd_value
            })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Rank models
    models_ranking_df = rank_models_emd(comparison_df)

    results = {
        'Ranked Models': models_ranking_df,
        'Details': comparison_df
    }

    return results


def rank_models_emd(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models from best to worst based on Earth Mover's Distance (EMD).
    """
    emd_sums = {}

    # Aggregate EMD values for each model
    for _, row in comparison_df.iterrows():
        models = [row['Model1'], row['Model2']]
        for model in models:
            emd_sums[model] = emd_sums.get(model, 0) + row['EMD Value']

    # Calculate average EMD for each model
    num_models = len(emd_sums)
    avg_emd = {model: total_emd / (num_models - 1) for model, total_emd in emd_sums.items()}

    # Sort models based on average EMD (lower is better)
    sorted_models = sorted(avg_emd.items(), key=lambda x: x[1])

    # Create a DataFrame for the ranked models
    models_ranking_df = pd.DataFrame(sorted_models, columns=['Model', 'Average EMD'])

    return models_ranking_df


def compare_cdfs_moments(data_dict: Dict[str, Dict[str, Union[str, np.ndarray, pd.DataFrame, pd.Series]]],
                         operation: Optional[str] = None, criteria: Optional[str] = 'mean'):
    """
    Compare multiple models based on statistical moments of their distributions.

    :param data_dict: A dictionary where keys are model names and the values are dictionaries with 'label' and 'data'
    :param operation:
    :param criteria: The criteria to determine the ranking ('mean', 'variance', 'skewness', 'kurtosis').
                      keys.
    :return: A dictionary containing the statistical moments for each model.
    """
    moments_data = []

    operations = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    if operation is not None and operation not in operations:
        raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")

    for model_name, model_info in data_dict.items():
        data = model_info['data'].to_numpy().flatten()
        if operation is not None:
            data = operations[operation](data, axis=0)

        moments_data.append({
            'Model': model_name,
            'Mean': np.mean(data),
            'Variance': np.var(data),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        })

    # Convert to DataFrame
    moments_df = pd.DataFrame(moments_data)

    # Rank models
    best_model_df = rank_models_moments(moments_df, criteria)

    results = {
        'Ranked Models': best_model_df,
        'Details': moments_df
    }

    return results


def rank_models_moments(moments_df: pd.DataFrame, criteria: str = 'mean') -> pd.DataFrame:
    """
    Rank models from best to worst based on a specified statistical moment.
    """
    if criteria not in ['mean', 'variance', 'skewness', 'kurtosis']:
        raise ValueError("Invalid criteria. Choose from 'mean', 'variance', 'skewness', 'kurtosis'.")

    reverse_order = (criteria == 'mean')  # For mean, higher is better; for others, lower is better
    sorted_df = moments_df.sort_values(by=[criteria.capitalize()], ascending=not reverse_order)

    # Add a rank column
    sorted_df['Rank'] = range(1, len(sorted_df) + 1)

    return sorted_df[['Model', 'Rank']]


def calculate_and_rank_percentiles(datasets, percentile_ranks=None):
    """
    Calculate given percentiles for multiple named datasets and rank them from best to worst
    based on higher throughput values.
    """
    if percentile_ranks is None:
        percentile_ranks = [50, 90]

    # Prepare data for DataFrame
    percentile_data = []

    for name, data_info in datasets.items():
        data = data_info['data']
        percentiles = {f'{p}th Percentile': np.percentile(data, p) for p in percentile_ranks}
        percentiles['Model'] = data_info['label']
        percentile_data.append(percentiles)

    # Create DataFrame
    percentile_df = pd.DataFrame(percentile_data)

    # Rank models based on the highest percentile value and convert rank to integer
    percentile_df['Rank'] = percentile_df[f'{percentile_ranks[-1]}th Percentile'] \
        .rank(ascending=False, method='min').astype(int)

    # Arrange columns
    column_order = ['Model'] + [f'{p}th Percentile' for p in percentile_ranks] + ['Rank']
    percentile_df = percentile_df[column_order]

    # Sort DataFrame by rank
    percentile_df.sort_values(by='Rank', inplace=True)

    return percentile_df


def calculate_area_throughput(df, bandwidth_hz, square_side_m, return_type='numpy'):
    # Convert square side from meters to kilometers
    square_side_km = square_side_m / 1000

    # Calculate the area in square kilometers
    area_km2 = square_side_km ** 2

    # Calculate cell density (cells per km²)
    cell_density_per_km2 = len(df) / area_km2

    # Initialize a list to store area throughput for each iteration
    area_throughput_per_iteration = []

    # Calculate area throughput for each iteration
    for iteration in range(len(df.columns)):
        # Sum of spectral efficiency across all cells for the iteration
        total_spectral_efficiency = df.iloc[:, iteration].sum()

        # Calculate area throughput for this iteration
        iteration_throughput = bandwidth_hz * cell_density_per_km2 * total_spectral_efficiency

        # Append the result to the list
        area_throughput_per_iteration.append(iteration_throughput)

    # Return the result based on the specified return type
    if return_type == 'pandas':
        return pd.DataFrame(area_throughput_per_iteration, columns=['Area Throughput'])
    else:  # default to numpy if anything other than 'pandas' is specified
        return np.array(area_throughput_per_iteration)


def plot_sinr_heatmap(sinr_df, location_df, ap_location_df, grid_size, rounding_precision=0, colorbar_ticks=None):
    """
    Plot a heatmap of SINR values in dB based on UE locations with reduced granularity, AP locations,
    and display mean SINR in each grid cell, leaving empty areas blank. Allows customization of colorbar ticks.

    :param sinr_df: A DataFrame of SINR values in mW.
    :param location_df: A DataFrame of UE locations in 'x+yj' string format.
    :param ap_location_df: A DataFrame of AP locations in 'x+yj' string format.
    :param grid_size: Tuple specifying the grid size (num_rows, num_cols).
    :param rounding_precision: Number of decimal places to round coordinates.
    :param colorbar_ticks: List of tuples for colorbar ticks and labels [(tick_value, 'label'), ...].
    """
    if sinr_df.shape != location_df.shape:
        raise ValueError("SINR DataFrame and location DataFrame must have the same shape.")

    if colorbar_ticks == 'custom':
        # Customized colorbar ticks
        colorbar_ticks = [
            (-30, '-30 dB'),
            (-25, '-25 dB'),
            (-20, '-20 dB'),
            (-15, '-15 dB'),
            (-10, '-10 dB'),
            (-5, '-5 dB'),
            (0, '0 dB'),
            (5, '5 dB'),
            (10, '10 dB'),
            (15, '15 dB'),
            (20, '20 dB'),
            (25, '25 dB'),
            (30, '30 dB')
        ]

    # Replace non-positive values with NaN
    sinr_df[sinr_df <= 0] = np.nan

    # Convert SINR values from mW to dB
    sinr_df = 10 * np.log10(sinr_df)

    data_list = []

    for i in range(sinr_df.shape[0]):
        for j in range(sinr_df.shape[1]):
            location_str = location_df.iloc[i, j]
            location = complex(location_str)
            x, y = round(location.real, rounding_precision), round(location.imag, rounding_precision)
            sinr_value = sinr_df.iloc[i, j]
            data_list.append({'x': x, 'y': y, 'SINR': sinr_value})

    df = pd.DataFrame(data_list)

    # Normalize and group data into grid cells
    df['x_norm'] = pd.cut(df['x'], bins=grid_size[1], labels=range(grid_size[1]))
    df['y_norm'] = pd.cut(df['y'], bins=grid_size[0], labels=range(grid_size[0]))

    # Calculate mean SINR for each grid cell
    grid_data = df.groupby(['y_norm', 'x_norm'], observed=True).SINR.mean().unstack()  # NaN for empty cells

    plt.figure(figsize=(10, 8))
    _colors = sns.color_palette("hls", 60)
    heatmap = sns.heatmap(grid_data, vmin=-30, vmax=30, annot=False, cmap=_colors, mask=grid_data.isnull())
    plt.title("SINR Heatmap with Grid (dB)")

    # Normalize and plot AP locations
    for location_str in ap_location_df.iloc[:, 0]:
        ap_location = complex(location_str)
        ap_x_norm = np.digitize(ap_location.real, np.linspace(df['x'].min(), df['x'].max(), grid_size[1])) - 1
        ap_y_norm = np.digitize(ap_location.imag, np.linspace(df['y'].min(), df['y'].max(), grid_size[0])) - 1
        heatmap.scatter(ap_x_norm, ap_y_norm, color='black', s=100, marker='*')

    # Customize colorbar ticks if specified
    if colorbar_ticks:
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_ticks([tick for tick, _ in colorbar_ticks])
        colorbar.set_ticklabels([label for _, label in colorbar_ticks])

    # Hide coordinates on the axes
    heatmap.set_xticklabels([])
    heatmap.set_yticklabels([])
    heatmap.set_xlabel('')
    heatmap.set_ylabel('')

    plt.show()


def duration_benchmarking(duration_data):
    _results = []

    for model, data in duration_data.items():
        df = data['data']
        _stats = {
            'Model': data['label'],
            'Mean Duration': df.mean(axis=1).iloc[0],
            'Median Duration': df.median(axis=1).iloc[0],
            'Standard Deviation': df.std(axis=1).iloc[0],
            'Max Duration': df.max(axis=1).iloc[0],
            'Min Duration': df.min(axis=1).iloc[0]
        }
        _results.append(_stats)

    return pd.DataFrame(_results)
