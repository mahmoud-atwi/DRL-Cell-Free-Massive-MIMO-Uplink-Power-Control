import warnings
import numpy as np
import gymnasium as gym

from time import time
from gymnasium import spaces
from collections import deque
from scipy.stats import gmean
from typing import Tuple, Optional

import simulation_para as sim_para
from random_waypoint import random_waypoint
from simulation_setup import cf_mimo_simulation
from compute_spectral_efficiency import compute_se
from _utils import generate_ap_locations, generate_ue_locations, calc_sinr
from power_optimization import power_opt_maxmin, power_opt_prod_sinr, power_opt_sum_rate


class MobilityCFmMIMOEnv(gym.Env):
    """
    A custom Gymnasium (Gym) environment for a Cell-Free Massive MIMO system.

    This environment simulates a cell-free massive MIMO system, incorporating user mobility and dynamic
    power control. The observation space consists of aggregated large-scale fading coefficients for each
    user, and the action space involves adjusting uplink power levels for each user.

    Attributes:
        L (int): Number of APs.
        K (int): Number of UEs.
        tau_p (int): Number of orthogonal pilots.
        max_power (float): Maximum transmit power.
        min_power (float): Minimum transmit power.
        initial_power (float): Initial power setting for simulations.
        UEs_power (Optional[np.ndarray]): Uplink power levels for each user.
        APs_positions (np.ndarray): Positions of the APs.
        UEs_positions (np.ndarray): Positions of the UEs.
        square_length (float): Length of the area square in meters.
        decorr (float): Decorrelation distance for shadow fading.
        sigma_sf (float): Shadow fading standard deviation in dB.
        noise_variance_dbm (float): Noise variance in dBm.
        delta (float): Shadow fading decorrelation parameter.
        UEs_mobility (bool): Flag to enable mobility for UEs.
        relocate_APs_on_reset (bool): Flag to relocate APs on environment reset.
        relocate_UEs_on_reset (bool): Flag to relocate UEs on environment reset.
        area_bounds (Tuple[float, float, float, float]): Bounds of the simulation area.
        reward_method (str): Method used to calculate the reward. It can be one of the following:
            - 'channel_capacity': Calculates reward based on channel capacity, using the sum of logarithmic
              Signal-to-Interference-plus-Noise Ratios (SINRs).
            - 'min_se': Uses the minimum spectral efficiency among all users, emphasizing the worst-case performance.
            - 'mean_se': Employs the average spectral efficiency across all users, providing a balance between
              fairness and efficiency.
            - 'sum_se': Considers the sum of spectral efficiencies of all users, prioritizing total system throughput.
            - 'geo_mean_se': Uses the geometric mean of spectral efficiencies, offering a compromise between
              fairness and total throughput.
        action_space (spaces.Box): The action space representing UL power levels.
        observation_space (spaces.Box): The observation space representing B_k values.
        state (np.ndarray): Current state of the environment.
    """

    def __init__(self, **kwargs):
        super(MobilityCFmMIMOEnv, self).__init__()

        self.L = kwargs.get('L', 64)
        self.K = kwargs.get('K', 32)
        self.tau_p = kwargs.get('tau_p', 20)
        self.max_power = kwargs.get('max_power', 100)
        self.min_power = kwargs.get('min_power', 0)
        self.initial_power = kwargs.get('initial_power', 100)
        self.UEs_power = kwargs.get('UEs_power', None)
        self.APs_positions = kwargs.get('APs_positions')
        self.UEs_positions = kwargs.get('UEs_positions')
        self.square_length = kwargs.get('square_length', 1000)
        self.decorr = kwargs.get('decorr', 100)
        self.sigma_sf = kwargs.get('sigma_sf', 8)
        self.noise_variance_dbm = kwargs.get('noise_variance_dbm', -92)
        self.delta = kwargs.get('delta', 0.5)
        self.UEs_mobility = kwargs.get('UEs_mobility', False)
        self.relocate_APs_on_reset = kwargs.get('relocate_AP_on_reset', False)
        self.relocate_UEs_on_reset = kwargs.get('relocate_UEs_on_reset', False)
        self.area_bounds = (0, self.square_length, 0, self.square_length)
        self.reward_method = kwargs.get('reward_method', None)
        self.temporal_reward_method = kwargs.get('temporal_reward_method', None)
        self.eval = kwargs.get('eval', False)

        # raise error if both reward_method and temporal_reward_method are provided
        if self.reward_method and self.temporal_reward_method:
            raise ValueError(
                """
                Only one reward method can be used at a time. 
                Please choose either reward_method or temporal_reward_method.
                
                reward_method options: 
                    1) channel_capacity
                    2) min_se
                    3) mean_se
                    4) sum_se
                    5) geo_mean_se
                    6) cf_min_se
                    7) cf_mean_se
                    8) cf_sum_se
                temporal_reward_method options:
                    1) delta
                    2) relative
                    3) exp_delta_clip
                    4) exp_relative_clip
                    5) log_delta
                    6) log_relative'
                """
            )

        methods = ["delta", "relative", "exp_delta_clip", "exp_relative_clip", "log_delta", "log_relative"]
        if not self.eval:
            if self.temporal_reward_method not in methods:
                raise ValueError(f"Invalid temporal reward method. select one of: {methods}")

        # Temporal parameters
        self.temporal_reward_operation = kwargs.get('temporal_reward_operation', 'mean')
        self.temporal_reward_max = kwargs.get('temporal_reward_max', 10)
        self.temporal_data = kwargs.get('temporal_data', "cf_se")
        self.temporal_window_size = kwargs.get('temporal_window_size', 10)
        self.temporal_history = deque(maxlen=self.temporal_window_size)

        # Define action space (continuous UL power levels for each UE)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.K,), dtype=np.float32)

        # Define observation space (B_k values)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(self.K,),
                                            dtype=np.float32)

        # Initial state (B_K values)
        self.state, _init_signal, _init_interference, _init_cf_se, *_ = self.initialize_state()

        if self.temporal_data == "cf_se":
            self.temporal_history.append(_init_cf_se)
        elif self.temporal_data == "sinr":
            _init_ues_power = np.full(self.K, self.initial_power)
            _init_sinr = calc_sinr(_init_ues_power, _init_signal, _init_interference)
            self.temporal_history.append(_init_sinr)
        else:
            warnings.warn(f"Temporal data {self.temporal_data} not supported. Falling back to spectral efficiency.")
            self.temporal_history.append(_init_cf_se)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # Initialize/reset the state (B_K values)
        _init_Beta_K, _init_signal, _init_interference, _init_cf_se, _init_pilot_index, _init_beta_val = (
            self.initialize_state())

        self.state = _init_Beta_K

        if self.temporal_data == "cf_se":
            self.temporal_history.append(_init_cf_se)
        elif self.temporal_data == "sinr":
            _init_ues_power = np.full(self.K, self.initial_power)
            _init_sinr = calc_sinr(_init_ues_power, _init_signal, _init_interference)
            self.temporal_history.append(_init_sinr)
        else:
            warnings.warn(f"Temporal data {self.temporal_data} not supported. Falling back to spectral efficiency.")
            self.temporal_data = "cf_se"
            self.temporal_history.append(_init_cf_se)

        _info = dict()

        _info['init_signal'] = self.state
        _info['init_interference'] = _init_interference
        _info['init_pilot_index'] = _init_pilot_index
        _info['init_beta_val'] = _init_beta_val

        return self.state, _info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Step method implementation

        # Adjust UL power based on action
        _rescaled_action = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power

        self.UEs_power = _rescaled_action  # Action is the new UL power levels

        if self.UEs_mobility:
            self.UEs_positions = self.update_ue_positions()

        # Recalculate B_k based on the new UL power
        _updated_Beta_K, _updated_signal, _updated_interference, _updated_cf_se = self.update_state()

        # Calculate reward
        _reward = self.calculate_reward(_updated_signal, _updated_interference, _updated_cf_se)

        # add to temporal history
        if self.temporal_data == "cf_se":
            self.temporal_history.append(_updated_cf_se)
        elif self.temporal_data == "sinr":
            _updated_sinr = calc_sinr(self.UEs_power, _updated_signal, _updated_interference)
            self.temporal_history.append(_updated_sinr)

        done = False
        truncated = False

        # Additional info
        _info = dict()

        _info['signal'] = _updated_signal
        _info['interference'] = _updated_interference
        _info['predicted_power'] = _rescaled_action
        _info['ues_positions'] = self.UEs_positions

        # Update the state
        self.state = _updated_Beta_K

        return self.state, float(_reward), done, truncated, _info

    def initialize_state(self) -> Tuple[np.ndarray, ...]:
        # Method to initialize the state
        if self.relocate_APs_on_reset:
            self.APs_positions = generate_ap_locations(self.L, 100, self.area_bounds)
        if self.relocate_UEs_on_reset:
            self.UEs_positions = generate_ue_locations(self.K, self.area_bounds)

        (
            _init_Beta_K, _init_signal, _init_interference, _init_cf_spectral_efficiency, _init_pilot_index,
            _init_beta_val, *_
        ) = cf_mimo_simulation(
            self.L, self.K, self.tau_p, self.max_power, self.initial_power, self.APs_positions, self.UEs_positions,
            self.square_length, self.decorr, self.sigma_sf, self.noise_variance_dbm, self.delta)

        return (_init_Beta_K, _init_signal, _init_interference, _init_cf_spectral_efficiency, _init_pilot_index,
                _init_beta_val)

    def update_state(self) -> Tuple[np.ndarray, ...]:
        # Method to update the state
        _updated_Beta_K, _updated_signal, _updated_interference, _updated_cf_spectral_efficiency, *_ = (
            cf_mimo_simulation(
                self.L, self.K, self.tau_p,
                self.max_power,
                self.UEs_power,
                self.APs_positions,
                self.UEs_positions,
                self.square_length,
                self.decorr, self.sigma_sf,
                self.noise_variance_dbm,
                self.delta)
        )

        return _updated_Beta_K, _updated_signal, _updated_interference, _updated_cf_spectral_efficiency

    def calculate_reward(self, signal: np.ndarray, interference: np.ndarray,
                         cf_spectral_efficiency: np.ndarray) -> float:
        # Method to calculate the reward
        if self.reward_method is None:
            if self.temporal_reward_method is None:
                warnings.warn(
                    "No reward method provided. temporal reward will be used with 'exp_relative_clip' method.")
                self.temporal_reward_method = "exp_relative_clip"

            if self.temporal_data == "cf_se":
                return self._calculate_temporal_reward(cf_spectral_efficiency)
            elif self.temporal_data == "sinr":
                _sinr = calc_sinr(self.UEs_power, signal, interference)
                return self._calculate_temporal_reward(_sinr)
            else:
                warnings.warn(
                    f"Temporal data {self.temporal_data} not supported. Falling back to spectral efficiency.")
                self.temporal_data = "cf_se"
                return self._calculate_temporal_reward(cf_spectral_efficiency)
        else:
            if self.reward_method == "channel_capacity":
                _SINR = calc_sinr(self.UEs_power, signal, interference)
                return np.sum(np.log2(1 + _SINR)).astype(np.float32)
            elif self.reward_method == "min_se":
                SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
                return np.min(SE).astype(np.float32)
            elif self.reward_method == "mean_se":
                SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
                return np.mean(SE).astype(np.float32)
            elif self.reward_method == "sum_se":
                SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
                return np.sum(SE).astype(np.float32)
            elif self.reward_method == "geo_mean_se":
                SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
                # Calculate geometric mean
                geometric_mean = np.prod(SE) ** (1.0 / len(SE))
                return geometric_mean.astype(np.float32)
            elif self.reward_method == "cf_min_se":
                return np.min(cf_spectral_efficiency).astype(np.float32)
            elif self.reward_method == "cf_mean_se":
                return np.mean(cf_spectral_efficiency).astype(np.float32)
            elif self.reward_method == "cf_geo_mean_se":
                # Calculate geometric mean
                geometric_mean = np.prod(cf_spectral_efficiency) ** (1.0 / len(cf_spectral_efficiency))
                return geometric_mean.astype(np.float32)
            elif self.reward_method == "cf_sum_se":
                return np.sum(cf_spectral_efficiency).astype(np.float32)
            else:
                # fallback to min_se in case the reward_method provided is not supported and raise warning
                print("Warning: reward method not supported. Falling back to min_se.")
                SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
                return np.min(SE)

    def _calculate_temporal_reward(self, current: np.ndarray) -> float:
        # Method to calculate temporal reward
        methods = ["delta", "relative", "exp_delta_clip", "exp_relative_clip", "log_delta", "log_relative"]
        if self.temporal_reward_method not in methods:
            raise ValueError(f"Invalid temporal reward method. select one of: {methods}")
        _method = self.temporal_reward_method

        operations = {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'sum': np.sum,
            'gmean': gmean
        }

        if self.temporal_reward_operation is None or self.temporal_reward_operation not in operations:
            raise ValueError(f"Invalid operation. Choose from {list(operations.keys())}")
        _operation = self.temporal_reward_operation

        # if history is empty, return 0.0
        if not self.temporal_history:
            return 0.0

        avg_history = np.mean(self.temporal_history, axis=0)
        avg_history = np.where(avg_history == 0, 1e-6, avg_history)

        if _method == "delta":
            return operations[_operation](current - avg_history).astype(np.float32)

        elif _method == "relative":
            return operations[_operation](current / avg_history).astype(np.float32)

        elif _method == "exp_delta_clip":
            delta = current - avg_history
            scaling_factor = 1.0 / (self.temporal_reward_max - 1.0)
            exp_delta = np.exp(delta * scaling_factor)
            # Clipping the rewards
            temporal_reward = operations[_operation](np.clip(exp_delta, None, self.temporal_reward_max))
            return temporal_reward.astype(np.float32)

        elif _method == "exp_relative_clip":
            ratio = current / avg_history
            exp_ratio = np.exp(ratio - 1)
            # Clipping the rewards
            max_reward = self.temporal_reward_max
            temporal_reward = operations[_operation](np.clip(exp_ratio, None, max_reward))
            return temporal_reward.astype(np.float32)

        elif _method == "log_delta":
            delta = current - avg_history
            constant_offset = 1e-6
            # log_delta = np.log(np.abs(delta) + constant_offset)
            # temporal_reward = operations[_operation](log_delta)

            # Separate positive and negative deltas
            positive_deltas = np.maximum(delta, 0) + constant_offset
            negative_deltas = -np.minimum(delta, 0) + constant_offset

            # Calculate log for positive and negative deltas
            log_positive_deltas = np.log(positive_deltas)
            log_negative_deltas = np.log(negative_deltas)

            # Calculate rewards for positive and negative deltas
            r_plus = np.sum(log_positive_deltas) / np.count_nonzero(delta > 0) if np.count_nonzero(delta > 0) > 0 else 0
            r_minus = np.sum(log_negative_deltas) / np.count_nonzero(delta < 0) if np.count_nonzero(
                delta < 0) > 0 else 0

            # Final temporal reward
            temporal_reward = r_plus - r_minus
            return temporal_reward.astype(np.float32)

        elif _method == "log_relative":
            ratio = current / avg_history
            log_ratio = np.log(ratio + 1e-6)
            temporal_reward = operations[_operation](log_ratio)
            return temporal_reward.astype(np.float32)

    def calculate(self, signal: np.ndarray, interference: np.ndarray, action: np.ndarray,
                  lagging_spectral_efficiency: bool, pilot_index: np.ndarray, beta_val: np.ndarray) -> dict:
        # Method to apply action and calculate next state without updating the state

        _info = dict()

        # Adjust UL power based on action
        _start_time = time()
        _allocated_UEs_power = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power
        _end_time = time()

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _allocated_UEs_power
            _info['duration'] = _end_time - _start_time

        if not lagging_spectral_efficiency:
            # Recalculate new B_k, signal and interference based on the new UL power
            _new_Beta_K, _new_signal, _new_interference, *_ = cf_mimo_simulation(self.L, self.K, self.tau_p,
                                                                                 self.max_power,
                                                                                 _allocated_UEs_power,
                                                                                 self.APs_positions,
                                                                                 self.UEs_positions, self.square_length,
                                                                                 self.decorr, self.sigma_sf,
                                                                                 self.noise_variance_dbm, self.delta,
                                                                                 pilot_index, beta_val)
            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['predicted_power'] = _allocated_UEs_power
            _info['duration'] = _end_time - _start_time

        return _info

    def maxmin_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                    prelog_factor: float, lagging_spectral_efficiency: bool, pilot_index: Optional[np.ndarray] = None,
                    beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-min power control algorithm
        _info = dict()
        _current_Beta_K = self.state
        _start_time = time()
        _, _opt_power = power_opt_maxmin(signal, interference, max_power, prelog_factor,
                                         return_spectral_efficiency=False)
        _end_time = time()

        if lagging_spectral_efficiency:
            _info['Beta_K'] = _current_Beta_K
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power
            _info['sinr'] = calc_sinr(_opt_power, signal, interference)
            _info['duration'] = _end_time - _start_time

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, _cf_spectral_efficiency, *_ = (
                cf_mimo_simulation(self.L, self.K, self.tau_p,
                                   self.max_power,
                                   _opt_power, self.APs_positions,
                                   self.UEs_positions, self.square_length,
                                   self.decorr, self.sigma_sf,
                                   self.noise_variance_dbm, self.delta,
                                   pilot_index, beta_val))

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power
            _info['cf_spectral_efficiency'] = _cf_spectral_efficiency
            _info['sinr'] = calc_sinr(_opt_power, _new_signal, _new_interference)
            _info['duration'] = _end_time - _start_time

        return _info

    def maxprod_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                     prelog_factor: float, lagging_spectral_efficiency: bool, pilot_index: Optional[np.ndarray] = None,
                     beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-product SINR power control algorithm
        _info = dict()
        _start_time = time()
        _, _opt_power = power_opt_prod_sinr(signal, interference, max_power, prelog_factor,
                                            return_spectral_efficiency=False)
        _end_time = time()

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power
            _info['sinr'] = calc_sinr(_opt_power, signal, interference)
            _info['duration'] = _end_time - _start_time

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, _cf_spectral_efficiency, *_ = (
                cf_mimo_simulation(self.L, self.K, self.tau_p,
                                   self.max_power,
                                   _opt_power, self.APs_positions,
                                   self.UEs_positions, self.square_length,
                                   self.decorr, self.sigma_sf,
                                   self.noise_variance_dbm, self.delta,
                                   pilot_index, beta_val))

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power
            _info['cf_spectral_efficiency'] = _cf_spectral_efficiency
            _info['sinr'] = calc_sinr(_opt_power, _new_signal, _new_interference)
            _info['duration'] = _end_time - _start_time

        return _info

    def maxsumrate_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                        prelog_factor: float, lagging_spectral_efficiency: bool,
                        pilot_index: Optional[np.ndarray] = None, beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-sum-rate power control algorithm
        _info = dict()
        _start_time = time()
        _, _opt_power = power_opt_sum_rate(signal, interference, max_power, prelog_factor,
                                           return_spectral_efficiency=False)
        _end_time = time()

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power
            _info['sinr'] = calc_sinr(_opt_power, signal, interference)
            _info['duration'] = _end_time - _start_time

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, _cf_spectral_efficiency, *_ = (
                cf_mimo_simulation(self.L, self.K, self.tau_p,
                                   self.max_power,
                                   _opt_power, self.APs_positions,
                                   self.UEs_positions, self.square_length,
                                   self.decorr, self.sigma_sf,
                                   self.noise_variance_dbm, self.delta,
                                   pilot_index, beta_val))

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power
            _info['cf_spectral_efficiency'] = _cf_spectral_efficiency
            _info['sinr'] = calc_sinr(_opt_power, _new_signal, _new_interference)
            _info['duration'] = _end_time - _start_time

        return _info

    def update_ue_positions(self, locations: Optional[np.array] = None) -> np.ndarray:
        # Method to update user equipment positions based on mobility
        if locations is None:
            _ues_locations = self.UEs_positions
        else:
            _ues_locations = locations
        return random_waypoint(_ues_locations, self.area_bounds, sim_para.speed_range, sim_para.max_pause_time,
                               sim_para.time_step, sim_para.pause_prob)

    def simulate(self, action, ues_positions) -> Tuple[np.ndarray, dict]:
        # Step method implementation

        # Adjust UL power based on action
        _rescaled_action = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power

        self.UEs_power = _rescaled_action  # Action is the new UL power levels

        if ues_positions is not None:
            self.UEs_positions = ues_positions
        else:
            self.UEs_positions = self.update_ue_positions()

        # Recalculate B_k based on the new UL power
        _updated_Beta_K, _updated_signal, _updated_interference, _cf_spectral_efficiency = self.update_state()

        # Additional info
        _info = dict()

        _info['signal'] = _updated_signal
        _info['interference'] = _updated_interference
        _info['predicted_power'] = _rescaled_action
        _info['ues_positions'] = self.UEs_positions
        _info['cf_spectral_efficiency'] = _cf_spectral_efficiency
        _info['sinr'] = calc_sinr(self.UEs_power, _updated_signal, _updated_interference)

        # Update the state
        self.state = _updated_Beta_K

        return self.state, _info

    def close(self):
        pass
