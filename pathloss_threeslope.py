import torch
from math import log10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def path_loss_three_slope(distances_vector):

    d0 = 10
    d1 = 50

    L = 140.7151

    path_losses = torch.zeros_like(distances_vector, device=device)

    # First slope
    mask = distances_vector <= d0
    path_losses[mask] = -L - 15 * log10(d1 / 1000) - 20 * log10(d0 / 1000)

    # Second slope
    mask = (distances_vector > d0) & (distances_vector <= d1)
    path_losses[mask] = -L - 15 * log10(d1 / 1000) - 20 * torch.log10(distances_vector[mask] / 1000)

    # Third slope
    mask = distances_vector > d1
    path_losses[mask] = -L - 35 * torch.log10(distances_vector[mask] / 1000)

    return path_losses
