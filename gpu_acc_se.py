import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_se_cf_ul(max_power, p, L, K, tau_p, tau_c, pilot_index, beta_val):

    if isinstance(p, (int, float)):
        p_tensor = torch.full((K,), p, dtype=torch.float32).to(device)
    elif isinstance(p, np.ndarray):
        if p.shape == ():
            p_scalar = p.item()  # Convert NumPy scalar to Python scalar
            p_tensor = torch.full((K,), p_scalar, dtype=torch.float32).to(device)
        elif p.shape == (K,):
            p_tensor = torch.from_numpy(p).to(device)
        else:
            raise ValueError('p should be either a single value or an array with size K')
    elif isinstance(p, torch.Tensor) and p.shape == (K,):
        p_tensor = p
    else:
        raise ValueError('p should be either a single value or an array with size K')

    pilot_index = pilot_index.to(device)
    # Vectorized gamma calculation
    mask = pilot_index.unsqueeze(1) == pilot_index.unsqueeze(0)  # Shape: [K, K]
    interference_term = max_power * tau_p * torch.sum(beta_val.unsqueeze(1) * mask, dim=2)
    gamma_val = max_power * tau_p * beta_val.pow(2) / (interference_term + 1)  # Broadcasting over L

    # Compute signal term
    signal = p_tensor * torch.sum(gamma_val, dim=0).pow(2)

    interference = torch.zeros((K,), device=device)

    prep = torch.tile(p_tensor, (L, 1))

    for k in range(K):
        interference[k] += torch.sum(gamma_val[:, k] * torch.sum(prep * beta_val, dim=1)) + torch.sum(gamma_val[:, k])

        # Compute interference due to pilot contamination
        co_pilot = pilot_index[k] == pilot_index
        co_pilot[k] = False
        same_pilot_indices = torch.where(co_pilot)[0]
        for index in same_pilot_indices:
            interference[k] += (p_tensor[index] *
                                (torch.sum(gamma_val[:, k] * beta_val[:, index] / beta_val[:, k])).pow(2))

    # Compute the SE
    SE = (1 - tau_p / tau_c) * torch.log2(1 + signal / interference)
    return SE


def compute_se(signal, interference, rho, prelog_factor):
    SINR = signal * rho / (torch.matmul(interference.T, rho) + 1)
    SE = prelog_factor * torch.log2(1 + SINR)
    return SE


def compute_se_np(signal, interference, rho, prelog_factor):
    SINR = signal * rho / (interference.T @ rho + 1)
    SE = prelog_factor * np.log2(1 + SINR)
    return SE
