import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_waypoint(ue_locations, area_bounds, speed_range, max_pause_time, time_step, pause_prob):
    """
    Apply the Random Waypoint mobility model to update UE locations using PyTorch.

    :param ue_locations: torch tensor of complex numbers representing current UE locations.
    :param area_bounds: tuple of (x_min, x_max, y_min, y_max) representing the area boundaries.
    :param speed_range: tuple of (min_speed, max_speed).
    :param max_pause_time: maximum pause time at destination.
    :param time_step: time step for the simulation.
    :param pause_prob: probability of pausing at a destination.
    :return: Updated UE locations as a torch tensor of complex numbers.
    """
    x_min, x_max, y_min, y_max = area_bounds
    min_speed, max_speed = speed_range
    num_ues = ue_locations.size(0)

    # Destination and state for each UE: Moving (1) or Paused (0)
    if "destinations" not in random_waypoint.__dict__:
        random_waypoint.destinations = (torch.rand(num_ues) * (x_max - x_min) + x_min) + 1j * (
                torch.rand(num_ues) * (y_max - y_min) + y_min)
        random_waypoint.states = torch.ones(num_ues)

    # Pause times for each UE
    if "pause_times" not in random_waypoint.__dict__:
        random_waypoint.pause_times = torch.zeros(num_ues)

    # Move UEs or update pause times
    new_locations = ue_locations.clone()

    for i in range(num_ues):
        if random_waypoint.states[i] == 1:  # Moving
            distance = torch.abs(random_waypoint.destinations[i] - ue_locations[i])
            direction = torch.angle(random_waypoint.destinations[i] - ue_locations[i])
            speed = (torch.rand(1) * (max_speed - min_speed) + min_speed).to(device)
            step_size = torch.min(speed * time_step, distance)
            step_size_scalar = step_size.item()  # Convert to scalar

            new_locations[i] += step_size_scalar * torch.exp(1j * direction)

            # Check if UE has reached its destination
            if step_size >= distance:
                random_waypoint.states[i] = 0  # Pause
                random_waypoint.pause_times[i] = torch.rand(1) * max_pause_time
                if torch.rand(1) > pause_prob:
                    # Select a new destination immediately without pausing
                    random_waypoint.destinations[i] = (torch.rand(1) * (x_max - x_min) + x_min) + 1j * (
                            torch.rand(1) * (y_max - y_min) + y_min)
                    random_waypoint.states[i] = 1
        else:  # Paused
            random_waypoint.pause_times[i] -= time_step
            if random_waypoint.pause_times[i] <= 0:
                # Pause time over, select a new destination
                random_waypoint.destinations[i] = (torch.rand(1) * (x_max - x_min) + x_min) + 1j * (
                        torch.rand(1) * (y_max - y_min) + y_min)
                random_waypoint.states[i] = 1

    return new_locations


def animate(frame):
    global ue_locations
    ue_locations = random_waypoint(ue_locations, area_bounds, speed_range, max_pause_time, time_step, pause_prob)
    scatter.set_offsets(torch.stack((ue_locations.real, ue_locations.imag), dim=1).numpy())
    return scatter,


if __name__ == '__main__':

    ue_locations = (torch.rand(40) + 1j * torch.rand(40)) * 1000  # Initial UE locations as torch tensor
    area_bounds = (0, 1000, 0, 1000)  # Area boundaries
    speed_range = (0.5, 5)  # Speed range
    max_pause_time = 5  # Maximum pause time
    time_step = 1  # Simulation time step
    pause_prob = 0.3  # Pause probability

    fig, ax = plt.subplots()
    _x_min, _x_max, _y_min, _y_max = area_bounds
    ax.set_xlim(_x_min, _x_max)
    ax.set_ylim(_y_min, _y_max)
    scatter = ax.scatter(ue_locations.real, ue_locations.imag)

    # Animation
    anim = FuncAnimation(fig, animate, frames=torch.arange(100), interval=1000*time_step, blit=True)

    plt.show()
