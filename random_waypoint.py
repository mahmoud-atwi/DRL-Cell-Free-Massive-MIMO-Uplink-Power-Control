import numpy as np
from typing import Tuple


def random_waypoint(ue_locations: np.ndarray,
                    area_bounds: Tuple[float, float, float, float],
                    speed_range: Tuple[float, float],
                    max_pause_time: float,
                    time_step: float,
                    pause_prob: float) -> np.ndarray:
    """
    Apply the Random Waypoint mobility model to update UE locations using NumPy.

    :param ue_locations: NumPy array of complex numbers representing current UE locations.
    :param area_bounds: tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :param speed_range: tuple of (min_speed, max_speed).
    :param max_pause_time: maximum pause time at destination.
    :param time_step: time step for the simulation.
    :param pause_prob: probability of pausing at a destination.
    :return: Updated UE locations as a NumPy array of complex numbers.
    """
    x_min, x_max, y_min, y_max = area_bounds
    min_speed, max_speed = speed_range
    num_ues = ue_locations.shape[0]

    if "destinations" not in random_waypoint.__dict__:
        random_waypoint.destinations = (np.random.rand(num_ues) * (x_max - x_min) + x_min) + \
                                       1j * (np.random.rand(num_ues) * (y_max - y_min) + y_min)
        random_waypoint.states = np.ones(num_ues)

    if "pause_times" not in random_waypoint.__dict__:
        random_waypoint.pause_times = np.zeros(num_ues)

    new_locations = ue_locations.copy()

    for i in range(num_ues):
        if random_waypoint.states[i] == 1:  # Moving
            distance = np.abs(random_waypoint.destinations[i] - ue_locations[i])
            direction = np.angle(random_waypoint.destinations[i] - ue_locations[i])
            speed = np.random.rand() * (max_speed - min_speed) + min_speed
            step_size = min(speed * time_step, distance)

            new_locations[i] += step_size * np.exp(1j * direction)

            if step_size >= distance:
                random_waypoint.states[i] = 0  # Pause
                random_waypoint.pause_times[i] = np.random.rand() * max_pause_time
                if np.random.rand() > pause_prob:
                    random_waypoint.destinations[i] = (np.random.rand() * (x_max - x_min) + x_min) + \
                                                      1j * (np.random.rand() * (y_max - y_min) + y_min)
                    random_waypoint.states[i] = 1
        else:  # Paused
            random_waypoint.pause_times[i] -= time_step
            if random_waypoint.pause_times[i] <= 0:
                random_waypoint.destinations[i] = (np.random.rand() * (x_max - x_min) + x_min) + \
                                                  1j * (np.random.rand() * (y_max - y_min) + y_min)
                random_waypoint.states[i] = 1

    return new_locations
