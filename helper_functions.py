import torch
import cvxpy as cp
import numpy as np

from scipy.linalg import sqrtm


def db2pow(db):
    """Convert dB to linear scale power."""
    return torch.pow(10, db / 10)


def dbm2mW(dbm):
    """Convert dBm to mW."""
    return 10 ** (dbm / 10)


def toeplitz_pytorch(c, r=None):
    """
    Create a Toeplitz matrix using PyTorch.

    Args:
    c (torch.Tensor): 1D tensor containing the first column of the matrix.
    r (torch.Tensor): 1D tensor containing the first row of the matrix.
                      If None, r is taken as same as c.

    Returns:
    torch.Tensor: Toeplitz matrix
    """
    if r is None:
        r = c
    else:
        if r[0] != c[0]:
            raise ValueError("First element of input column must be equal to first element of input row.")

    c_size = c.size(0)
    r_size = r.size(0)
    a, b = torch.meshgrid(torch.arange(c_size), torch.arange(r_size), indexing='ij')
    idx_matrix = a - b + (r_size - 1)

    # Use flattened c and r to fill in the Toeplitz matrix
    flattened = torch.cat((r.flip(0)[1:], c))
    return flattened[idx_matrix]


def get_sqrtm(matrix, method='newton_schulz', iters=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if method == 'cholesky':
        try:
            # Use Cholesky decomposition in PyTorch for positive definite matrices
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            # Fallback to Newton-Schulz if Cholesky fails
            return sqrtm_newton_schulz(matrix, iters)[0]
    elif method == 'newton_schulz':
        # Use Newton-Schulz Iterative method
        return sqrtm_newton_schulz(matrix, iters)[0]
    else:
        # If not positive definite, use SciPy sqrtm on CPU and transfer to PyTorch
        return torch.tensor(sqrtm(matrix.cpu().numpy()), dtype=torch.complex64).to(device)


def sqrtm_newton_schulz(matrix, num_iters):
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

    return Y * torch.sqrt(norm_of_matrix), None


def feasibility_problem_cvx_np(signal, interference, max_power, K, sinr_constraint):

    rho = cp.Variable(K)
    scaling = cp.Variable()

    objective = cp.Minimize(scaling)

    constraints = []

    for k in range(K):
        constraints.append(sinr_constraint * (cp.sum(cp.multiply(rho, interference[:, k])) + 1) - (rho[k] * signal[k]) <= 0)
        constraints.append(rho[k] >= 0)
        constraints.append(rho[k] <= scaling * max_power)

    constraints.append(scaling >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    # Analyze the output and prepare the output variables
    if problem.status not in ["infeasible", "unbounded"]:
        if scaling.value > 1:  # Only the power minimization problem is feasible
            feasible = False
            rhoSolution = rho.value
        else:  # Both the power minimization problem and feasibility problem are feasible
            feasible = True
            rhoSolution = rho.value
    else:  # Both problems are infeasible
        feasible = False
        rhoSolution = None

    return feasible, rhoSolution


def compute_prod_sinr(signal, interference, rho, prelog_factor):
    SINR = signal * rho / (np.dot(interference.T, rho) + 1)
    prod_SINR = prelog_factor * np.log2(SINR)
    return prod_SINR


def calc_SINR(power, signal, interference):
    # Calculate the denominator and numerator using NumPy operations
    denominator = np.dot(interference, power) + 1
    numerator = np.multiply(signal, power)

    # Calculate SINR
    SINR = numerator / denominator

    return SINR
