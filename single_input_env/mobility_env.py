import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import simulation_para as sim_para
from helper_functions import calc_SINR
from power_optimization import power_opt_maxmin, power_opt_prod_sinr, power_opt_sum_rate
from random_waypoint import random_waypoint
from simulation_setup import CF_mMIMO_Env

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MobilityCFmMIMOEnv(gym.Env):
    """
    Custom Environment for Cell-Free Massive MIMO system that follows the gym interface.
    Observation state is B_K, an aggregated LSF coefficient for each user k.
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

        # Define action space (continuous UL power levels for each UE)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.K,), dtype=np.float32)

        # Define observation space (B_k values)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(self.K,),
                                            dtype=np.float32)

        # Initial state (B_K values)
        self.state, *_ = self.initialize_state()

    def reset(self, **kwargs):
        """
        Reset the state of the environment to an initial state
        """
        # Initialize/reset the state (B_K values)
        self.state, _init_signal, _init_interference, _init_pilot_index, _init_beta_val = self.initialize_state()

        info = dict()

        info['init_signal'] = _init_signal
        info['init_interference'] = _init_interference
        info['init_pilot_index'] = _init_pilot_index
        info['init_beta_val'] = _init_beta_val

        return self.state, info

    def step(self, action):
        """
        Apply action (adjust UL power) and calculate next state (B_k values)
        """
        # Adjust UL power based on action
        _rescaled_action = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power

        self.UEs_power = _rescaled_action  # Action is the new UL power levels

        if self.UEs_mobility:
            self.UEs_positions = self.update_ue_positions()

        # Recalculate B_k based on the new UL power
        _updated_Beta_k, _updated_signal, _updated_interference = self.update_state()

        # Calculate reward
        _reward = self.calculate_reward(_updated_signal, _updated_interference)

        done = False
        truncated = False

        # Additional info
        _info = dict()

        _info['signal'] = _updated_signal
        _info['interference'] = _updated_interference
        _info['predicted_power'] = _rescaled_action

        # Update the state
        self.state = _updated_Beta_k

        return _updated_Beta_k, _reward, done, truncated, _info

    def initialize_state(self):

        _UE_init_locations = torch.rand(self.K, dtype=torch.complex64, device=device) * self.square_length

        _init_B_k, _init_signal, _init_interference, _init_pilot_index, _init_beta_val, *_ = (
            CF_mMIMO_Env(self.L,
                         self.K,
                         self.tau_p,
                         self.max_power,
                         self.initial_power,
                         self.APs_positions,
                         _UE_init_locations,
                         self.square_length,
                         self.decorr,
                         self.sigma_sf,
                         self.noise_variance_dbm,
                         self.delta))

        _init_Beta_k_np = _init_B_k.detach().cpu().numpy()
        _init_signal_np = _init_signal.detach().cpu().numpy()
        _init_interference_np = _init_interference.detach().cpu().numpy()
        _init_pilot_index_np = _init_pilot_index.detach().cpu().numpy()
        _init_beta_val_np = _init_beta_val.detach().cpu().numpy()

        return _init_Beta_k_np, _init_signal_np, _init_interference_np, _init_pilot_index_np, _init_beta_val_np

    def update_state(self):
        """
        Update the Beta_K values based on the given action.
        """
        _updated_Beta_k, _updated_signal, _updated_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p,
                                                                                   self.max_power,
                                                                                   self.UEs_power,
                                                                                   self.APs_positions,
                                                                                   self.UEs_positions,
                                                                                   self.square_length, self.decorr,
                                                                                   self.sigma_sf,
                                                                                   self.noise_variance_dbm,
                                                                                   self.delta)

        _updated_Beta_k_np = _updated_Beta_k.detach().cpu().numpy()
        _update_signal_np = _updated_signal.detach().cpu().numpy()
        _updated_interference_np = _updated_interference.detach().cpu().numpy()

        return _updated_Beta_k_np, _update_signal_np, _updated_interference_np

    def calculate_reward(self, signal, interference):
        """
        Calculate the reward for the given action and state.
        """
        _SINR = calc_SINR(self.UEs_power, signal, interference)
        _r = np.sum(np.log2(1 + _SINR))
        return float(_r)

    def calculate(self, signal, interference, action, lagging_SE, pilot_index, beta_val):
        """
        Apply action (adjust UL power) and calculate next state (Beta_k values) without updating the state
        """
        _info = dict()
        _current_Beta_k = self.state
        # Adjust UL power based on action
        _rescaled_action = ((action + 1) / 2) * (self.max_power - self.min_power) + self.min_power

        _allocated_UEs_power = _rescaled_action  # Action is the new UL power levels

        if lagging_SE:
            _info['Beta_K'] = _current_Beta_k
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _rescaled_action

        if not lagging_SE:
            # Recalculate new B_k, signal and interference based on the new UL power
            _new_Beta_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                           _allocated_UEs_power, self.APs_positions,
                                                                           self.UEs_positions, self.square_length,
                                                                           self.decorr, self.sigma_sf,
                                                                           self.noise_variance_dbm, self.delta,
                                                                           pilot_index, beta_val)
            _info['Beta_K'] = _new_Beta_k.detach().cpu().numpy()
            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _rescaled_action

        return _info

    def maxmin_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _current_Beta_k = self.state

        _, _opt_power = power_opt_maxmin(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['Beta_K'] = _current_Beta_k
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _opt_power

        if not lagging_SE:
            _new_Beta_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                           _opt_power, self.APs_positions,
                                                                           self.UEs_positions, self.square_length,
                                                                           self.decorr, self.sigma_sf,
                                                                           self.noise_variance_dbm, self.delta,
                                                                           pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_k.detach().cpu().numpy()
            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _opt_power

        return _info

    def maxprod_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _current_Beta_k = self.state

        _, _opt_power = power_opt_prod_sinr(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['Beta_K'] = _current_Beta_k
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _opt_power

        if not lagging_SE:
            _new_Beta_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                           _opt_power, self.APs_positions,
                                                                           self.UEs_positions, self.square_length,
                                                                           self.decorr, self.sigma_sf,
                                                                           self.noise_variance_dbm, self.delta,
                                                                           pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_k.detach().cpu().numpy()
            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _opt_power

        return _info

    def maxsumrate_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        _info = dict()
        _current_Beta_k = self.state

        _, _opt_power = power_opt_sum_rate(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            _info['Beta_K'] = _current_Beta_k
            _info['signal'] = signal
            _info['interference'] = interference
            _info['predicted_power'] = _opt_power

        if not lagging_SE:
            _new_Beta_k, _new_signal, _new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                           _opt_power, self.APs_positions,
                                                                           self.UEs_positions, self.square_length,
                                                                           self.decorr, self.sigma_sf,
                                                                           self.noise_variance_dbm, self.delta,
                                                                           pilot_index, beta_val)

            _info['Beta_K'] = _new_Beta_k.detach().cpu().numpy()
            _info['signal'] = _new_signal.detach().cpu().numpy()
            _info['interference'] = _new_interference.detach().cpu().numpy()
            _info['predicted_power'] = _opt_power

        return _info

    def update_ue_positions(self):
        area_bounds = [0, self.square_length, 0, self.square_length]
        return random_waypoint(self.UEs_positions, area_bounds, sim_para.speed_range, sim_para.max_pause_time,
                               sim_para.time_step, sim_para.pause_prob)

    def close(self):
        pass


if __name__ == '__main__':
    L = 64
    K = 32
    square_length = 1000
    AP_locations = torch.rand(L, dtype=torch.complex64) * square_length
    UE_initial_locations = torch.rand(K, dtype=torch.complex64) * square_length
    env = MobilityCFmMIMOEnv(APs_positions=AP_locations, UEs_positions=UE_initial_locations)
    observation, _ = env.reset()  # Reset the environment
    done = False
    while not done:
        action = env.action_space.sample()  # Sample a random action
        observation, reward, done, truncated, info = env.step(action)  # Take a step
        print(f"Obs: {observation}, Reward: {reward}, Done: {done}, Info: {info}")