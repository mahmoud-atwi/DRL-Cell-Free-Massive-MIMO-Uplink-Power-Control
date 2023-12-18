import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from helper_functions import calc_SINR
from power_optimization import power_opt_maxmin, power_opt_prod_sinr, power_opt_sum_rate
from simulation_setup import CF_mMIMO_Env

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CFmMIMOEnv(gym.Env):
    """
    Custom Environment for Cell-Free Massive MIMO system that follows the gym interface.
    Observation state is B_K, an aggregated LSF coefficient for each user k.
    """

    def __init__(self, L, K, tau_p, initial_power, min_power, max_power, APs_positions, UEs_positions, square_length,
                 decorr, sigma_sf, noise_variance_dbm, delta):
        super(CFmMIMOEnv, self).__init__()

        self.L = L
        self.K = K
        self.tau_p = tau_p
        self.max_power = max_power
        self.min_power = min_power
        self.initial_power = initial_power
        self.UEs_power = None
        self.APs_positions = APs_positions
        self.UEs_positions = UEs_positions
        self.square_length = square_length
        self.decorr = decorr
        self.sigma_sf = sigma_sf
        self.noise_variance_dbm = noise_variance_dbm
        self.delta = delta

        # Define action space (continuous UL power levels for each UE)
        self.action_space = spaces.Box(low=-1, high=1, shape=(K,), dtype=np.float32)
        # Store the original power range for rescaling
        self.power_range = (min_power, max_power)

        # Define observation space (B_k values)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(K,), dtype=np.float32)

        # Initial state (B_K values)
        self.state, *_ = self.initialize_B_K()

    def reset(self, **kwargs):
        """
        Reset the state of the environment to an initial state
        """
        # Initialize/reset the state (B_K values)
        self.state, init_signal_CF, init_interference_CF, pilot_index_CF, beta_val_CF = self.initialize_B_K()

        info = dict()

        info['init_signal'] = init_signal_CF
        info['init_interference'] = init_interference_CF
        info['init_pilot_index'] = pilot_index_CF
        info['init_beta_val'] = beta_val_CF

        return self.state, info

    def step(self, action):
        """
        Apply action (adjust UL power) and calculate next state (B_k values)
        """
        # Adjust UL power based on action
        min_power, max_power = self.power_range
        rescaled_action = ((action + 1) / 2) * (max_power - min_power) + min_power

        self.UEs_power = rescaled_action  # Action is the new UL power levels

        # Recalculate B_k based on the new UL power
        updated_B_k, signal_CF, interference_CF = self.update_B_K()

        # Calculate reward
        reward = self.calculate_reward(signal_CF, interference_CF)

        done = False
        truncated = False

        # Additional info
        info = dict()

        info['signal'] = signal_CF
        info['interference'] = interference_CF
        info['predicted_power'] = rescaled_action

        # Update the state
        self.state = updated_B_k

        return updated_B_k, reward, done, truncated, info

    def initialize_B_K(self):

        UE_init_locations = torch.rand(self.K, dtype=torch.complex64, device=device) * self.square_length

        init_B_k, signal_CF, interference_CF, pilot_index, beta_val, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p,
                                                                                       self.max_power,
                                                                                       self.initial_power,
                                                                                       self.APs_positions,
                                                                                       UE_init_locations,
                                                                                       self.square_length, self.decorr,
                                                                                       self.sigma_sf,
                                                                                       self.noise_variance_dbm,
                                                                                       self.delta)

        init_B_k_np = init_B_k.detach().cpu().numpy()
        signal_CF_np = signal_CF.detach().cpu().numpy()
        interference_CF_np = interference_CF.detach().cpu().numpy()
        pilot_index_np = pilot_index.detach().cpu().numpy()
        beta_val_np = beta_val.detach().cpu().numpy()

        return init_B_k_np, signal_CF_np, interference_CF_np, pilot_index_np, beta_val_np

    def update_B_K(self):
        """
        Update the B_K values based on the given action.
        """
        updated_B_k, signal_CF, interference_CF, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                   self.UEs_power, self.APs_positions,
                                                                   self.UEs_positions, self.square_length, self.decorr,
                                                                   self.sigma_sf, self.noise_variance_dbm, self.delta)

        updated_B_k_np = updated_B_k.detach().cpu().numpy()
        signal_CF_np = signal_CF.detach().cpu().numpy()
        interference_CF_np = interference_CF.detach().cpu().numpy()

        return updated_B_k_np, signal_CF_np, interference_CF_np

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
        info = dict()
        new_B_k = self.state
        # Adjust UL power based on action
        min_power, max_power = self.power_range
        rescaled_action = ((action + 1) / 2) * (max_power - min_power) + min_power

        self.UEs_power = rescaled_action  # Action is the new UL power levels

        if lagging_SE:
            info['signal'] = signal
            info['interference'] = interference
            info['predicted_power'] = rescaled_action

        if not lagging_SE:
            # Recalculate new B_k, signal and interference based on the new UL power
            new_B_k, new_signal, new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                     self.UEs_power, self.APs_positions,
                                                                     self.UEs_positions, self.square_length,
                                                                     self.decorr,
                                                                     self.sigma_sf, self.noise_variance_dbm, self.delta,
                                                                     pilot_index, beta_val)

            info['signal'] = new_signal.detach().cpu().numpy()
            info['interference'] = new_interference.detach().cpu().numpy()
            info['predicted_power'] = rescaled_action

        return new_B_k, info

    def maxmin_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        info = dict()
        new_B_k = self.state

        _, opt_power = power_opt_maxmin(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            info['signal'] = signal
            info['interference'] = interference
            info['predicted_power'] = opt_power

        if not lagging_SE:
            new_B_k, new_signal, new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                     opt_power, self.APs_positions,
                                                                     self.UEs_positions, self.square_length,
                                                                     self.decorr,
                                                                     self.sigma_sf, self.noise_variance_dbm, self.delta,
                                                                     pilot_index, beta_val)

            info['signal'] = new_signal.detach().cpu().numpy()
            info['interference'] = new_interference.detach().cpu().numpy()
            info['predicted_power'] = opt_power

        return new_B_k, info

    def maxprod_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        info = dict()
        new_B_k = self.state

        _, opt_power = power_opt_prod_sinr(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            info['signal'] = signal
            info['interference'] = interference
            info['predicted_power'] = opt_power

        if not lagging_SE:
            new_B_k, new_signal, new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                     opt_power, self.APs_positions,
                                                                     self.UEs_positions, self.square_length,
                                                                     self.decorr,
                                                                     self.sigma_sf, self.noise_variance_dbm, self.delta,
                                                                     pilot_index, beta_val)

            info['signal'] = new_signal.detach().cpu().numpy()
            info['interference'] = new_interference.detach().cpu().numpy()
            info['predicted_power'] = opt_power

        return new_B_k, info

    def maxsumrate_algo(self, signal, interference, max_power, prelog_factor, lagging_SE, pilot_index, beta_val):

        info = dict()
        new_B_k = self.state

        _, opt_power = power_opt_sum_rate(signal, interference, max_power, prelog_factor, return_SE=False)

        if lagging_SE:
            info['signal'] = signal
            info['interference'] = interference
            info['predicted_power'] = opt_power

        if not lagging_SE:
            new_B_k, new_signal, new_interference, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                     opt_power, self.APs_positions,
                                                                     self.UEs_positions, self.square_length,
                                                                     self.decorr,
                                                                     self.sigma_sf, self.noise_variance_dbm, self.delta,
                                                                     pilot_index, beta_val)

            info['signal'] = new_signal.detach().cpu().numpy()
            info['interference'] = new_interference.detach().cpu().numpy()
            info['predicted_power'] = opt_power

        return new_B_k, info

    def close(self):
        pass
