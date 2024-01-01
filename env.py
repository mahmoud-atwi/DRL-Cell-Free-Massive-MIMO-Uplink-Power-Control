from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import simulation_para as sim_para
from _utils import generate_ap_locations, generate_ue_locations, calc_sinr
from compute_spectral_efficiency import compute_se
from power_optimization import power_opt_maxmin, power_opt_prod_sinr, power_opt_sum_rate
from random_waypoint import random_waypoint
from simulation_setup import cf_mimo_simulation


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
        self.reward_method = kwargs.get('reward_method', 'min_se')

        # Define action space (continuous UL power levels for each UE)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.K,), dtype=np.float32)

        # Define observation space (B_k values)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(self.K,),
                                            dtype=np.float32)

        # Initial state (B_K values)
        self.state, *_ = self.initialize_state()

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # Initialize/reset the state (B_K values)
        _init_Beta_K, _init_signal, _init_interference, _init_SE, _init_pilot_index, _init_beta_val = (
            self.initialize_state())

        self.state = _init_Beta_K

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
        _updated_Beta_K, _updated_signal, _updated_interference, _cf_spectral_efficiency = self.update_state()

        # Calculate reward
        _reward = self.calculate_reward(_updated_signal, _updated_interference, _cf_spectral_efficiency)

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
        else:
            # fallback to min_se in case the reward_method provided is not supported and raise warning
            print("Warning: reward method not supported. Falling back to min_se.")
            SE = compute_se(signal, interference, self.UEs_power, sim_para.prelog_factor)
            return np.min(SE)

    def calculate(self, signal: np.ndarray, interference: np.ndarray, action: np.ndarray,
                  lagging_spectral_efficiency: bool, pilot_index: np.ndarray, beta_val: np.ndarray) -> dict:
        # Method to apply action and calculate next state without updating the state

        _info = dict()

        # Adjust UL power based on action
        _allocated_UEs_power = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _allocated_UEs_power

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

        return _info

    def maxmin_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                    prelog_factor: float, lagging_spectral_efficiency: bool, pilot_index: Optional[np.ndarray] = None,
                    beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-min power control algorithm
        _info = dict()
        _current_Beta_K = self.state

        _, _opt_power = power_opt_maxmin(signal, interference, max_power, prelog_factor,
                                         return_spectral_efficiency=False)

        if lagging_spectral_efficiency:
            _info['Beta_K'] = _current_Beta_K
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, *_ = cf_mimo_simulation(self.L, self.K, self.tau_p,
                                                                                 self.max_power,
                                                                                 _opt_power, self.APs_positions,
                                                                                 self.UEs_positions, self.square_length,
                                                                                 self.decorr, self.sigma_sf,
                                                                                 self.noise_variance_dbm, self.delta,
                                                                                 pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power

        return _info

    def maxprod_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                     prelog_factor: float, lagging_spectral_efficiency: bool, pilot_index: Optional[np.ndarray] = None,
                     beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-product SINR power control algorithm
        _info = dict()
        _, _opt_power = power_opt_prod_sinr(signal, interference, max_power, prelog_factor,
                                            return_spectral_efficiency=False)

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, *_ = cf_mimo_simulation(self.L, self.K, self.tau_p,
                                                                                 self.max_power,
                                                                                 _opt_power, self.APs_positions,
                                                                                 self.UEs_positions, self.square_length,
                                                                                 self.decorr, self.sigma_sf,
                                                                                 self.noise_variance_dbm, self.delta,
                                                                                 pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power

        return _info

    def maxsumrate_algo(self, signal: np.ndarray, interference: np.ndarray, max_power: float, ues_positions: np.ndarray,
                        prelog_factor: float, lagging_spectral_efficiency: bool,
                        pilot_index: Optional[np.ndarray] = None, beta_val: Optional[np.ndarray] = None) -> dict:
        # Method implementing the max-sum-rate power control algorithm
        _info = dict()
        _, _opt_power = power_opt_sum_rate(signal, interference, max_power, prelog_factor,
                                           return_spectral_efficiency=False)

        if lagging_spectral_efficiency:
            _info['Beta_K'] = self.state
            _info['signal'] = signal
            _info['interference'] = interference
            _info['optimized_power'] = _opt_power

        if not lagging_spectral_efficiency:
            if ues_positions is not None:
                self.UEs_positions = ues_positions
            _new_Beta_K, _new_signal, _new_interference, *_ = cf_mimo_simulation(self.L, self.K, self.tau_p,
                                                                                 self.max_power,
                                                                                 _opt_power, self.APs_positions,
                                                                                 self.UEs_positions, self.square_length,
                                                                                 self.decorr, self.sigma_sf,
                                                                                 self.noise_variance_dbm, self.delta,
                                                                                 pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_K
            _info['signal'] = _new_signal
            _info['interference'] = _new_interference
            _info['optimized_power'] = _opt_power

        return _info

    def update_ue_positions(self) -> np.ndarray:
        # Method to update user equipment positions based on mobility
        return random_waypoint(self.UEs_positions, self.area_bounds, sim_para.speed_range, sim_para.max_pause_time,
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

        # Update the state
        self.state = _updated_Beta_K

        return self.state, _info

    def close(self):
        pass
