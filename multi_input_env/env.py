import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from helper_functions import calc_SINR
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
        self.UEs_power = initial_power
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
        self.observation_space = spaces.Dict({
            'B_k': spaces.Box(low=np.float32(0), high=np.float32(np.inf), shape=(K,), dtype=np.float32),
            'power_allocation': spaces.Box(low=np.float32(-1), high=np.float32(1), shape=(K,), dtype=np.float32),
        })

        # Initial state (B_K values)
        init_B_k, init_signal_Cf, init_interference_CF, init_power = self.initialize_state()
        self.state = {'B_k': init_B_k, 'power_allocation': init_power}

    def reset(self, **kwargs):
        """
        Reset the state of the environment to an initial state
        """
        # Initialize/reset the state (B_K values)
        init_B_k, init_signal_Cf, init_interference_CF, init_power = self.initialize_state()
        self.state = {'B_k': init_B_k, 'power_allocation': init_power}

        info = {}
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
        updated_B_k, signal_CF, interference_CF = self.update_state()

        # Calculate reward
        reward = self.calculate_reward(signal_CF, interference_CF)

        done = False
        truncated = False

        # Additional info (optional)
        info = dict()

        info['signal'] = signal_CF
        info['interference'] = interference_CF
        info['predicted_power'] = rescaled_action

        # Update the state
        self.state = {'B_k': updated_B_k, 'power_allocation': action}

        return self.state, reward, done, truncated, info

    def initialize_state(self):
        """
        Initialize the B_K values for each user.
        """
        init_B_k, signal_CF, interference_CF, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                self.UEs_power, self.APs_positions,
                                                                self.UEs_positions, self.square_length, self.decorr,
                                                                self.sigma_sf,
                                                                self.noise_variance_dbm, self.delta)

        # Scale UL Power to [-1, 1] for all UEs
        scaled_ul_power = 2 * (self.UEs_power - self.min_power) / (self.max_power - self.min_power) - 1
        scaled_ul_power_array = np.full((self.K,), scaled_ul_power, dtype=np.float32)

        init_B_k_np = init_B_k.cpu().numpy()
        signal_CF_np = signal_CF.cpu().numpy()
        interference_CF_np = interference_CF.cpu().numpy()

        return init_B_k_np, signal_CF_np, interference_CF_np, scaled_ul_power_array

    def update_state(self):
        """
        Update the B_K values based on the given action.
        """
        updated_B_k, signal_CF, interference_CF, *_ = CF_mMIMO_Env(self.L, self.K, self.tau_p, self.max_power,
                                                                   self.UEs_power, self.APs_positions,
                                                                   self.UEs_positions, self.square_length, self.decorr,
                                                                   self.sigma_sf, self.noise_variance_dbm, self.delta)

        updated_B_k_np = updated_B_k.cpu().numpy()
        signal_CF_np = signal_CF.cpu().numpy()
        interference_CF_np = interference_CF.cpu().numpy()

        return updated_B_k_np, signal_CF_np, interference_CF_np

    def calculate_reward(self, signal, interference):
        """
        Calculate the reward for the given action and state.
        """
        SINR = calc_SINR(self.UEs_power, signal, interference)
        r = np.sum(np.log2(1 + SINR))
        return float(r)

    def close(self):
        pass
