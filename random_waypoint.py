import time

import numpy as np


def random_waypoint(ue_locations, area_bounds, speed_range, max_pause_time, time_step, pause_prob):
    """
    Apply the Random Waypoint mobility model to update UE locations.

    :param ue_locations: numpy array of complex numbers representing current UE locations.
    :param area_bounds: tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :param speed_range: tuple of (min_speed, max_speed).
    :param max_pause_time: maximum pause time at destination.
    :param time_step: time step for the simulation.
    :param pause_prob: probability of pausing at a destination.
    :return: Updated UE locations as a numpy array of complex numbers.
    """
    x_min, x_max, y_min, y_max = area_bounds
    min_speed, max_speed = speed_range
    num_ues = ue_locations.shape[0]

    # Destination and state for each UE: Moving (1) or Paused (0)
    if 'destinations' not in random_waypoint.__dict__:
        random_waypoint.destinations = np.random.uniform(x_min, x_max, num_ues) + 1j * np.random.uniform(y_min, y_max,
                                                                                                         num_ues)
        random_waypoint.states = np.ones(num_ues)

    # Pause times for each UE
    if 'pause_times' not in random_waypoint.__dict__:
        random_waypoint.pause_times = np.zeros(num_ues)

    # Move UEs or update pause times
    new_locations = np.copy(ue_locations)
    for i in range(num_ues):
        if random_waypoint.states[i] == 1:  # Moving
            distance = np.abs(random_waypoint.destinations[i] - ue_locations[i])
            direction = np.angle(random_waypoint.destinations[i] - ue_locations[i])
            speed = np.random.uniform(min_speed, max_speed)
            step_size = min(speed * time_step, distance)
            new_locations[i] += step_size * np.exp(1j * direction)

            # Check if UE has reached its destination
            if step_size >= distance:
                random_waypoint.states[i] = 0  # Pause
                random_waypoint.pause_times[i] = np.random.uniform(0, max_pause_time)
                if np.random.rand() > pause_prob:
                    # Select a new destination immediately without pausing
                    random_waypoint.destinations[i] = np.random.uniform(x_min, x_max) + 1j * np.random.uniform(y_min,
                                                                                                               y_max)
                    random_waypoint.states[i] = 1
        else:  # Paused
            random_waypoint.pause_times[i] -= time_step
            if random_waypoint.pause_times[i] <= 0:
                # Pause time over, select a new destination
                random_waypoint.destinations[i] = np.random.uniform(x_min, x_max) + 1j * np.random.uniform(y_min, y_max)
                random_waypoint.states[i] = 1

    return new_locations


if __name__ == '__main__':
    # Example usage
    _ue_locations = np.array([1 + 1j, 2 + 2j, 3 + 3j])  # Initial UE locations
    _area_bounds = (0, 10, 0, 10)  # Area boundaries
    _speed_range = (0.5, 1.5)  # Speed range (units per time step)
    _max_pause_time = 5  # Maximum pause time at each destination
    _time_step = 1  # Simulation time step
    _pause_prob = 0.3  # Probability of pausing at a destination

    # Update UE locations
    start = time.time()
    _new_ue_locations = random_waypoint(_ue_locations, _area_bounds, _speed_range, _max_pause_time, _time_step,
                                        _pause_prob)
    print("Time taken:", time.time() - start)
    print("New UE Locations:", _new_ue_locations)
