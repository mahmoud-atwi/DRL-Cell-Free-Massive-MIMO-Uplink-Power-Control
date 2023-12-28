from math import log10

# Simulation Parameters
L = 64  # Number of APs
K = 32  # Number of UEs
tau_p = 20  # Number of orthogonal pilots
min_power = 0  # Minimum transmit power
max_power = 100  # Maximum transmit power
initial_power = 100  # Initial power setting for simulations
prelog_factor = 1  # Pre-logarithmic factor in capacity calculations

# Three-Slope Model Simulation Parameters for large-scale fading
square_length = 1000  # Square length of the area in meters
bandwidth = 20e6  # System bandwidth in Hertz
noise_figure = 9  # Noise figure in decibels (dB)
noise_variance_dbm = -174 + 10 * log10(bandwidth) + noise_figure  # Total noise power in dBm
sigma_sf = 8.0  # Shadow fading standard deviation in dB
delta = 0.5  # Shadow fading decorrelation parameter
decorr = 100  # Decorrelation distance for shadow fading in meters
antenna_spacing = 0.5  # Antenna spacing in wavelengths
angular_sd_deg = 15  # Standard deviation of the angle of arrival/departure in degrees

# Mobility Parameters
speed_range = [0.5, 5]  # Range of user speeds
max_pause_time = 5  # Maximum pause time for a user
time_step = 1  # Time step for the mobility model
pause_prob = 0.3  # Probability of a user pausing at each time step
