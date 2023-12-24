from math import log10

# Simulation Parameters
L = 64
K = 32
tau_p = 20
min_power = 0
max_power = 100
initial_power = 100

# Three-Slope Simulation Parameters
square_length = 1000  # meters
bandwidth = 20e6  # bandwidth in Hz
noise_figure = 9  # noise figure in dB
noise_variance_dbm = -174 + 10 * log10(bandwidth) + noise_figure  # noise power in dBm
sigma_sf = 8  # shadow fading standard deviation in dB
delta = 0.5  # shadow fading decorrelation parameter
decorr = 100  # decorrelation distance in meters
antenna_spacing = 0.5  # antenna spacing in wavelengths
angular_sd_deg = 15  # angular standard deviation in degrees

# for mobility
speed_range = [0.5, 5]
max_pause_time = 5
time_step = 1
pause_prob = 0.3
