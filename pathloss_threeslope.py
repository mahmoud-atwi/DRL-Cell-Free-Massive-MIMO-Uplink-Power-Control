import numpy as np
from typing import Union, Iterable


def path_loss_three_slope(distances_vector: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
    """
    Calculates the path loss using a three-slope model over a given set of distances.

    This function computes the path loss for each distance in the input vector according to a three-slope
    path loss model. The model divides the range of distances into three segments, each with a different
    path loss exponent:
    - First slope for distances less than or equal to 10 meters.
    - Second slope for distances greater than 10 meters and up to 50 meters.
    - Third slope for distances greater than 50 meters.

    :param distances_vector: An array or iterable of distances in meters for which path losses are to be calculated.
    :return: An array of path loss values in dB for each input distance.
    """
    d0 = 10  # Distance threshold for the first slope in meters
    d1 = 50  # Distance threshold for the second slope in meters

    L = 140.7151

    path_losses = np.zeros_like(distances_vector)

    # First slope
    mask = distances_vector <= d0
    path_losses[mask] = -L - 15 * np.log10(d1 / 1000) - 20 * np.log10(d0 / 1000)

    # Second slope
    mask = (distances_vector > d0) & (distances_vector <= d1)
    path_losses[mask] = -L - 15 * np.log10(d1 / 1000) - 20 * np.log10(distances_vector[mask] / 1000)

    # Third slope
    mask = distances_vector > d1
    path_losses[mask] = -L - 35 * np.log10(distances_vector[mask] / 1000)

    return path_losses
