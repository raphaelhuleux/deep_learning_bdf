import numpy as np
from quantecon.optimize import brent_max
import quantecon as qe
import numba as nb 
import matplotlib.pyplot as plt

# Preferences
beta = 0.9      # Discount factor
gamma = 2.0     # Risk aversion (CRRA)
r = (1/beta-1) * 0.9

# Income process (log-AR(1))
rho_Z = 0.9       # Persistence
sigma_Z = 0.2     # Shock standard deviation

# =============================================================================
# Create Grids using QuantEcon (MUST be before njit functions that use them)
# =============================================================================

nA = 1000
Amin = 0.0
Amax = 10.0
A_grid = np.linspace(Amin, Amax, nA)

# TFP grid using QuantEcon's Tauchen method
nZ = 9
tauchen_result = qe.tauchen(nZ, rho_Z, sigma_Z, mu=0, n_std=3)
log_Z_grid = tauchen_result.state_values  # log(Z) grid
Pi_Z = tauchen_result.P  # Transition matrix
Z_grid = np.exp(log_Z_grid)  # Z grid in levels

@nb.njit
def linear_interp(x_grid, y_grid, x):
    """Simple linear interpolation - numba compatible"""
    n = len(x_grid)

    # Handle boundary cases
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[n-1]:
        return y_grid[n-1]

    # Find interval via binary search
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid

    # Linear interpolation
    t = (x - x_grid[lo]) / (x_grid[hi] - x_grid[lo])
    return y_grid[lo] + t * (y_grid[hi] - y_grid[lo])


@nb.njit
def compute_EV(iz, K_next, V):
    """Compute expected continuation value E[V(Z', K') | Z]"""
    EV = 0.0
    for iz_next in range(nZ):
        EV += Pi_Z[iz, iz_next] * linear_interp(A_grid, V[iz_next, :], K_next)
    return EV


@nb.njit
def bellman_objective(A_next, A, Z, iz, V):
    """Bellman objective for optimization"""
    coh = A * (1+r) + Z
    C = coh - A_next

    u = (C**(1 - gamma)) / (1 - gamma)
    return u + beta * compute_EV(iz, A_next, V)

    
def vfi_step(V):
    """Single VFI iteration (not jitted - brent_max not numba compatible)""" 

    V_new = np.zeros_like(V)
    policy_A = np.zeros_like(V) 

    for iz in range(nZ):
        Z = Z_grid[iz]

        for ia in range(nA):
            A = A_grid[ia    ]
            coh = A * (1+r) + Z

            # Bounds for A' (must have positive consumption)
            A_next_max = min(coh - 1e-8, Amax)

            A_next_opt, V_opt, _ = brent_max(bellman_objective, 0.0, A_next_max, args=(A, Z, iz, V))

            V_new[iz, ia] = V_opt
            policy_A[iz, ia] = A_next_opt
    
    return V_new, policy_A


def vfi(tol = 1e-6, max_iter = 1000, verbose = True):

    # Initialize value function (use steady-state utility as starting point)
    C_guess = A_grid[np.newaxis:] * r + Z_grid[:,np.newaxis]
    V = C_guess**(1 - gamma) / (1 - gamma) / (1 - beta)

    # VFI loop
    for iteration in range(max_iter):
        V_new, policy_A = vfi_step(V)
        diff = np.max(np.abs(V_new - V))
        if verbose:
            print(f"Iteration {iteration + 1}: max value function diff = {diff:.8f}")

        if diff < tol:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations!")
            break

        V = V_new.copy()

    return V, policy_A

