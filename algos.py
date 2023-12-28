from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Dict, Type

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "A2C": A2C,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}
