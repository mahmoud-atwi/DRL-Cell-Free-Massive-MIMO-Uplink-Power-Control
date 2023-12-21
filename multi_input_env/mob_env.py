import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from helper_functions import calc_SINR
from power_optimization import power_opt_maxmin, power_opt_prod_sinr, power_opt_sum_rate
from simulation_setup import CF_mMIMO_Env
from random_waypoint import random_waypoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MobCFmMIMOEnv(gym.Env):
    """
    Custom Environment for Cell-Free Massive MIMO system that follows the gym interface.
    Observation state is B_K, an aggregated LSF coefficient for each user k.
    """

    def __init__(self, **kwargs):
        super(MobCFmMIMOEnv, self).__init__()

        self.L = kwargs.get('L')
        self.K = kwargs.get('K')
        self.tau_p = kwargs.get('tau_p')
        self.max_power = kwargs.get('max_power')
        self.min_power = kwargs.get('min_power')
        self.initial_power = kwargs.get('initial_power')
        self.UEs_power = None
        self.APs_positions = kwargs.get('APs_positions')
        self.UEs_positions = kwargs.get('UEs_positions')
        self.square_length = kwargs.get('square_length')
        self.decorr = kwargs.get('decorr')
        self.sigma_sf = kwargs.get('sigma_sf')
        self.noise_variance_dbm = kwargs.get('noise_variance_dbm')
        self.delta = kwargs.get('delta')
        self.UEs_mobility = kwargs.get('with_mobility')

        # Define action space (continuous UL power levels for each UE) (scaled to -1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.K,), dtype=np.float32)
        # Store the original power range for rescaling
        self.power_range = (self.min_power, self.max_power)

        # Define observation space (B_k values and allocated power)
        self.observation_space = spaces.Dict({
            'B_k': spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(self.K,), dtype=np.float32),
            'power_allocation': spaces.Box(low=np.float32(-1), high=np.float32(1), shape=(self.K,), dtype=np.float32),
        })

        # Initial state
        init_B_k, init_signal, init_interference, init_pilot_index, init_beta_val, init_power = self.initialize_state()
        self.state = {'B_k': init_B_k, 'power_allocation': init_power}

    def reset(self, **kwargs):
        """
        Reset the state of the environment to an initial state
        """
        # Initialize/reset the state
        init_B_k, init_signal, init_interference, init_pilot_index, init_beta_val, init_power = self.initialize_state()
        self.state = {'B_k': init_B_k, 'power_allocation': init_power}

        init_info = dict()

        init_info['init_signal'] = init_signal
        init_info['init_interference'] = init_interference
        init_info['init_pilot_index'] = init_pilot_index
        init_info['init_beta_val'] = init_beta_val
        return self.state, init_info

    def step(self, action):
        """
        Apply action (adjust UL power) and calculate next state (B_k values)
        """
        # Adjust UL power based on action
        min_power, max_power = self.power_range
        rescaled_action = ((action + 1) / 2) * (max_power - min_power) + min_power

        self.UEs_power = rescaled_action  # rescaled action is the new UL power levels

        # update UEs positions
        self.UEs_positions = self.update_ue_positions()

        # Recalculate B_k and SINR terms based on the new UL power
        updated_B_k, updated_signal, updated_interference = self.update_state()

        # Calculate reward
        reward = self.calculate_reward(updated_signal, updated_interference)

        done = False
        truncated = False

        # Additional info
        step_info = dict()

        step_info['signal'] = updated_signal
        step_info['interference'] = updated_interference
        step_info['predicted_power'] = rescaled_action

        # Update the state
        self.state = {'B_k': updated_B_k, 'power_allocation': action}

        return self.state, reward, done, truncated, step_info

    def initialize_state(self):
        # distribute UEs randomly on the grid
        UE_init_locations = torch.rand(self.K, dtype=torch.complex64, device=device) * self.square_length

        _init_B_k, _init_signal, _init_interference, _init_pilot_index, _init_beta_val, *_ \
            = CF_mMIMO_Env(self.L,
                           self.K,
                           self.tau_p,
                           self.max_power,
                           self.initial_power,
                           self.APs_positions,
                           UE_init_locations,
                           self.square_length,
                           self.decorr,
                           self.sigma_sf,
                           self.noise_variance_dbm,
                           self.delta)

        _init_B_k_np = _init_B_k.detach().cpu().numpy()
        _signal_np = _init_signal.detach().cpu().numpy()
        _interference_np = _init_interference.detach().cpu().numpy()
        _pilot_index_np = _init_pilot_index.detach().cpu().numpy()
        _beta_val_np = _init_beta_val.detach().cpu().numpy()

        # check if self.initial_power is a scalar or a vector
        if isinstance(self.initial_power, (int, float)):
            _initial_power = np.ones(self.K) * self.initial_power
            # scale the initial power to [-1, 1]
            _scaled_power = ((2 * (_initial_power - self.min_power)) / (self.max_power - self.min_power)) - 1
            _scaled_power = _scaled_power.astype(np.float32)
        else:
            # check if self.initial_power is within the power range [min_power, max_power] and scale it to [-1, 1]
            if np.any(self.initial_power < self.min_power) or np.any(self.initial_power > self.max_power):
                raise ValueError('Initial power values must be within the power range [min_power, max_power]')
            else:
                _scaled_power = ((2 * (self.initial_power - self.min_power)) / (self.max_power - self.min_power)) - 1
                _scaled_power = _scaled_power.astype(np.float32)

        return _init_B_k_np, _signal_np, _interference_np, _pilot_index_np, _beta_val_np, _scaled_power

    def update_state(self):
        """
        Update the B_K values based on the given action.
        """
        _updated_B_k, _update_signal, _update_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p,
                                                                              self.max_power, self.UEs_power,
                                                                              self.APs_positions, self.UEs_positions,
                                                                              self.square_length, self.decorr,
                                                                              self.sigma_sf, self.noise_variance_dbm,
                                                                              self.delta)

        _updated_B_k_np = _updated_B_k.detach().cpu().numpy()
        _update_signal_np = _update_signal.detach().cpu().numpy()
        _update_interference_np = _update_interference.detach().cpu().numpy()

        return _updated_B_k_np, _update_signal_np, _update_interference_np

    def calculate_reward(self, signal, interference):
        """
        Calculate the reward for the given action and state.
        """
        SINR = calc_SINR(self.UEs_power, signal, interference)
        r = np.sum(np.log2(1 + SINR))

        return float(r)

    def calculate(self, signal, interference, action, lagging_SE, pilot_index, beta_val):
        """
        Apply action (adjust UL power) and calculate next state (B_k values) without updating the state
        """
        _info = dict()
        _new_B_k = self.state
        # Adjust UL power based on action
        min_power, max_power = self.power_range
        allocated_power = ((action + 1) / 2) * (max_power - min_power) + min_power

        if lagging_SE:
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = allocated_power

        if not lagging_SE:
            # Recalculate new B_k, signal and interference based on the new UL power
            _new_B_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                        allocated_power, self.APs_positions,
                                                                        self.UEs_positions, self.square_length,
                                                                        self.decorr, self.sigma_sf,
                                                                        self.noise_variance_dbm, self.delta,
                                                                        pilot_index, beta_val)

            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = allocated_power

        return _new_B_k, _info

    def maxmin_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _new_B_k = self.state

        _, _optimized_power = power_opt_maxmin(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _optimized_power

        if not lagging_SE:
            _new_B_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                        _optimized_power, self.APs_positions,
                                                                        self.UEs_positions, self.square_length,
                                                                        self.decorr,
                                                                        self.sigma_sf, self.noise_variance_dbm,
                                                                        self.delta,
                                                                        pilot_index, beta_val)

            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _optimized_power

        return _new_B_k, _info

    def maxprod_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _new_B_k = self.state

        _, _optimized_power = power_opt_prod_sinr(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _optimized_power

        if not lagging_SE:
            _new_B_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                        _optimized_power, self.APs_positions,
                                                                        self.UEs_positions, self.square_length,
                                                                        self.decorr, self.sigma_sf,
                                                                        self.noise_variance_dbm, self.delta,
                                                                        pilot_index, beta_val)

            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _optimized_power

        return _new_B_k, _info

    def maxsumrate_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _new_B_k = self.state

        _, _optimized_power = power_opt_sum_rate(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _optimized_power

        if not lagging_SE:
            _new_B_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                        _optimized_power, self.APs_positions,
                                                                        self.UEs_positions, self.square_length,
                                                                        self.decorr, self.sigma_sf,
                                                                        self.noise_variance_dbm, self.delta,
                                                                        pilot_index, beta_val)

            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _optimized_power

        return _new_B_k, _info

    def update_ue_positions(self):
        area_bounds = [0, self.square_length, 0, self.square_length]
        speed_range = [0.5, 5]
        max_pause_time = 5
        time_step = 1
        pause_prob = 0.3

        if self.UEs_mobility:
            return random_waypoint(self.UEs_positions, area_bounds, speed_range, max_pause_time, time_step, pause_prob)
        else:
            return self.UEs_positions

    def close(self):
        pass
