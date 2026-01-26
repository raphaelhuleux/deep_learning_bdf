import numpy as np
from quantecon.optimize import brent_max
import numba as nb
import matplotlib.pyplot as plt

# Preferences
beta = 0.9      # Discount factor
gamma = 2.0     # Risk aversion (CRRA)

# Firms
alpha = 1/3     # Capital share
delta = 0.05    # Depreciation rate

# Steady-state interest rate (from Euler equation with Z=1)
r_ss = 1/beta - 1

# Steady-state capital (from FOC of firm: r = alpha * Z * K^(alpha-1) - delta)
K_ss = ((r_ss + delta) / (alpha))**(1/(alpha - 1))

# Capital grid bounds
K_min = K_ss * 0.5
K_max = K_ss * 2.0


# =============================================================================
# Helper Functions
# =============================================================================
@nb.njit
def production(K: np.ndarray) -> np.ndarray:
    """Cobb-Douglas production function: Y = Z * K^alpha"""
    return K**alpha

@nb.njit
def cash_on_hand(K: np.ndarray) -> np.ndarray:
    """Cash-on-hand: output plus undepreciated capital"""
    return production(K) + (1 - delta) * K

@nb.njit
def utility(C: np.ndarray) -> np.ndarray:
    """CRRA utility function: u(c) = (c^(1-gamma) - 1) / (1-gamma)"""
    if gamma == 1:
        return np.log(np.maximum(C, 1e-10))
    else:
        C_safe = np.maximum(C, 1e-10)
        return (C_safe**(1 - gamma) - 1) / (1 - gamma)

# =============================================================================
# Create Grids using QuantEcon (MUST be before njit functions that use them)
# =============================================================================

n_K = 500
K_grid = np.linspace(K_min, K_max, n_K)


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
def bellman_objective(K_next, K, V):
    """Bellman objective for optimization"""
    coh = cash_on_hand(K)
    C = coh - K_next

    if C <= 0:
        return -np.inf

    return utility(C) + beta * linear_interp(K_grid, V, K_next)

    
def vfi_step(V):
    """Single VFI iteration (not jitted - brent_max not numba compatible)""" 

    V_new = np.zeros_like(V)
    policy_K = np.zeros_like(V) 

    for ik in range(n_K):
        K = K_grid[ik]
        coh = cash_on_hand(K)

        # Bounds for K' (must have positive consumption)
        K_next_min = K_min
        K_next_max = min(coh - 1e-8, K_max)
        K_next_opt, V_opt, _ = brent_max(bellman_objective, K_next_min, K_next_max, args=(K, V))

        V_new[ik] = V_opt
        policy_K[ik] = K_next_opt
    
    return V_new, policy_K


def vfi(tol = 1e-8, max_iter = 1000, verbose = True):

    # Initialize value function (use steady-state utility as starting point)
    C_ss = cash_on_hand(K_ss) - K_ss
    V = np.ones((n_K)) * utility(C_ss) / (1 - beta)


    # Policy function
    policy_K = np.zeros((n_K))

    # VFI loop
    for iteration in range(max_iter):
        V_new, policy_K = vfi_step(V)
        diff = np.max(np.abs(V_new - V))
        if verbose:
            print(f"Iteration {iteration + 1}: max value function diff = {diff:.8f}")

        if diff < tol:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations!")
            break

        V = V_new.copy()

    return V, policy_K


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_policy_functions(policy_K: np.ndarray, policy_C: np.ndarray,
                         policy_s: np.ndarray):
    """Plot capital, consumption, and savings rate policies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    # Capital policy
    ax = axes[0]
    ax.plot(K_grid, K_grid, 'k--', alpha=0.5, label='45° line')
    
    ax.plot(K_grid, policy_K, label="K' policy")
    ax.axvline(x=K_ss, linestyle=':', color='gray', alpha=0.5)
    ax.set_xlabel('Capital K')
    ax.set_ylabel("K'(Z, K)")
    ax.set_title('Capital Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Consumption policy
    ax = axes[1]
    
    coh = cash_on_hand(K_grid)
    ax.plot(K_grid, coh, '--', alpha=0.5, label='Cash-on-hand')
    ax.plot(K_grid, policy_C, label='Consumption policy')
    ax.axvline(x=K_ss, linestyle=':', color='gray', alpha=0.5)
    ax.set_xlabel('Capital K')
    ax.set_ylabel('C(Z, K)')
    ax.set_title('Consumption Policy (dashed = cash-on-hand)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Savings rate policy
    ax = axes[2]
    
    ax.plot(K_grid, policy_s, label='Savings rate')
    ax.axhline(y=0.5, linestyle='--', color='red', alpha=0.5)
    ax.axvline(x=K_ss, linestyle=':', color='gray', alpha=0.5)
    ax.set_xlabel('Capital K')
    ax.set_ylabel('s(Z, K) = K\'/CoH')
    ax.set_title('Savings Rate Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def extract_policies(policy_K):
    """Extract consumption and savings rate policies from capital policy"""

    coh = cash_on_hand(K_grid)
    policy_C = coh - policy_K
    policy_s = policy_K / coh

    return policy_C, policy_s


if __name__ == "__main__":
    print("=" * 60)
    print("Value Function Iteration for Ramsey Model")
    print("(Using QuantEcon: Brent's method)")
    print("=" * 60)
    print(f"\nSteady-state capital: K_ss = {K_ss:.4f}")
    print(f"Capital grid: [{K_min:.4f}, {K_max:.4f}], {n_K} points")


    # Solve the model
    print("\nSolving via VFI...")
    V, policy_K = vfi(tol=1e-8, max_iter=10_000)

    # Extract policy functions
    policy_C, policy_s = extract_policies(policy_K)

    # Report steady-state values
    ik_ss = np.argmin(np.abs(K_grid - K_ss))
    print(f"\nSteady-state check (Z ≈ 1, K ≈ K_ss):")
    print(f"  K_ss = {K_ss:.4f}")
    print(f"  K'(Z_ss, K_ss) = {policy_K[ik_ss]:.4f}")
    print(f"  Savings rate at SS = {policy_s[ik_ss]:.4f}")

    # Plot results
    print("\nGenerating plots...")
    plot_policy_functions(policy_K, policy_C, policy_s)

    print("\nDone!")
