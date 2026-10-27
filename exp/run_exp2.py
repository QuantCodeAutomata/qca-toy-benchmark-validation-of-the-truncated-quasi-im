"""
Experiment 2: Quintic OU model option pricing.

Computes implied-volatility smiles via the Lewis formula with Heston control
variate and validates against a simplified Monte Carlo benchmark.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.makedirs(Path(__file__).parent.parent / 'results', exist_ok=True)

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre

from src.quintic_ou import (
    ou_params_from_alpha_epsilon, char_func_riccati, char_func_batch,
    heston_char_func, fit_heston_control_variate,
    bs_implied_vol, bs_call_price, g0_func_builder, build_pp_conv,
)
from src.vol_swap import mc_vol_swap_rate

# ---------------------------------------------------------------------------
# Model parameters (paper defaults)
# ---------------------------------------------------------------------------
RHO = -0.65
ALPHA = -0.6
EPSILON = 1.0 / 52
P_COEFFS = [0.01, 1.0, 0.0, 0.214, 0.0, 0.227]  # p0,p1,0,p3,0,p5
XI0 = 0.025
S0 = 1.0

# Experiment settings
MATURITIES = [1.0 / 52, 0.5, 1.0, 2.0]
LM_GRID = np.linspace(-0.25, 0.25, 25)
M = 20
N_QUAD = 32
N_MC_PATHS = 10_000
N_MC_STEPS = 100

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def xi0_func(t):
    return XI0


def price_smile_vectorised(T, M, a, b, c, rho, p_coeffs, xi0_func,
                            heston_p, lm_grid, S0, n_quad=32):
    """Compute call prices and IVs for all strikes sharing CF evaluations.

    Uses char_func_batch to evaluate phi(u-0.5j) for all n_quad nodes in a
    single vectorised solve, then loops over strikes.
    """
    gl_nodes, gl_weights = roots_laguerre(n_quad)

    # Batch-compute model CF at all shifted nodes u_j - 0.5j  (fast)
    phi_m = char_func_batch(gl_nodes, T, M, a, b, c, rho,
                             p_coeffs, xi0_func)          # shape (n_quad,)

    # Heston CF at each shifted node (analytic, negligible cost)
    phi_h = np.array([
        heston_char_func(u - 0.5j, T, heston_p['v0'], heston_p['kappa'],
                         heston_p['theta'], heston_p['sigma_h'],
                         heston_p['rho_h'])
        for u in gl_nodes
    ], dtype=complex)

    # Lewis weights (shared across strikes)
    w_eff = gl_weights * np.exp(gl_nodes)  # Gauss-Laguerre exp-weight
    denom = gl_nodes ** 2 + 0.25

    prices = []
    ivs = []
    for lm in lm_grid:
        K = S0 * math.exp(lm)
        k = lm  # log(K/S0)

        # Phase: exp((i*u + 0.5)*k)
        phase = np.exp((1j * gl_nodes + 0.5) * k)

        # Heston Lewis integral
        integ_h = w_eff * np.real(phase * phi_h) / denom
        C_Heston = S0 - K / math.pi * float(np.sum(integ_h))
        C_Heston = max(C_Heston, max(S0 - K, 0.0))

        # CV correction
        integ_cv = w_eff * np.real(phase * (phi_m - phi_h)) / denom
        C_model = C_Heston - K / math.pi * float(np.sum(integ_cv))
        C_model = max(C_model, max(S0 - K, 0.0))

        iv = bs_implied_vol(C_model, S0, K, T)
        prices.append(C_model)
        ivs.append(iv if np.isfinite(iv) else np.nan)

    return np.array(prices), np.array(ivs)


def main():
    print("=" * 60)
    print("Experiment 2: Quintic OU Option Pricing")
    print("=" * 60)

    a, b, c = ou_params_from_alpha_epsilon(ALPHA, EPSILON)
    print(f"OU params: a={a}, b={b:.2f}, c={c:.4f}")
    print(f"M={M}, n_quad={N_QUAD}, MC: {N_MC_PATHS} paths x {N_MC_STEPS} steps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    results_md = []
    results_md.append("## Experiment 2: Quintic OU Option Pricing\n")
    results_md.append(f"Parameters: rho={RHO}, alpha={ALPHA}, epsilon={EPSILON:.4f}, "
                      f"xi0={XI0}, M={M}\n")
    results_md.append(f"p_coeffs: {P_COEFFS}\n\n")

    for ax_i, T in enumerate(MATURITIES):
        print(f"\n--- Maturity T = {T:.4f} ---")
        t0 = time.time()
        ax = axes[ax_i]

        # Fit Heston control variate
        u_fit = np.linspace(0.5, 5.0, 8)
        try:
            heston_p = fit_heston_control_variate(
                u_fit, T, M, a, b, c, RHO, P_COEFFS, xi0_func, XI0
            )
        except Exception as e:
            print(f"  Heston CV fit failed: {e}")
            heston_p = {'v0': XI0, 'kappa': 2.0, 'theta': XI0,
                        'sigma_h': 0.3, 'rho_h': RHO * 0.8}

        # Compute model smiles (vectorised)
        try:
            prices, ivs = price_smile_vectorised(
                T, M, a, b, c, RHO, P_COEFFS, xi0_func,
                heston_p, LM_GRID, S0, n_quad=N_QUAD
            )
        except Exception as e:
            print(f"  Pricing failed: {e}")
            prices = np.full(len(LM_GRID), np.nan)
            ivs = np.full(len(LM_GRID), np.nan)

        t_riccati = time.time() - t0
        print(f"  Riccati pricing: {t_riccati:.1f}s")

        # Monte Carlo benchmark (simplified)
        t_mc = time.time()
        try:
            mc_mean, mc_lo, mc_hi = mc_vol_swap_rate(
                T, N_MC_PATHS, N_MC_STEPS, b, c, xi0_func,
                vol_model='quintic_ou', p_coeffs=P_COEFFS
            )
        except Exception as e:
            print(f"  MC failed: {e}")
            mc_mean, mc_lo, mc_hi = np.nan, np.nan, np.nan
        t_mc_elapsed = time.time() - t_mc
        print(f"  MC vol swap rate: {mc_mean:.4f} [{mc_lo:.4f}, {mc_hi:.4f}] ({t_mc_elapsed:.1f}s)")

        # Plot
        mask = np.isfinite(ivs)
        ax.plot(LM_GRID[mask], ivs[mask], 'b-', lw=2, label='Riccati (M=20)')
        if np.isfinite(mc_mean):
            ax.axhline(mc_mean, color='r', ls='--', alpha=0.7, label=f'MC vol swap')
            ax.axhspan(mc_lo, mc_hi, alpha=0.15, color='red', label='MC 95% CI')
        ax.set_title(f'T = {T:.3f} yr  (ATM IV ≈ {np.nanmedian(ivs):.3f})')
        ax.set_xlabel('Log-moneyness')
        ax.set_ylabel('Implied vol')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(LM_GRID[0], LM_GRID[-1])

        # Record results
        atm_iv = float(np.nanmedian(ivs))
        results_md.append(f"### T = {T:.4f}\n")
        results_md.append(f"- ATM IV: {atm_iv:.4f}\n")
        results_md.append(f"- MC vol swap: {mc_mean:.4f} ({mc_lo:.4f}, {mc_hi:.4f})\n")
        results_md.append(f"- Riccati time: {t_riccati:.1f}s, MC time: {t_mc_elapsed:.1f}s\n\n")

    fig.suptitle('Quintic OU Model: Implied Volatility Smiles', fontsize=13)
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'exp2_quintic_ou_smiles.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # Convergence in M
    print("\n--- Convergence in M ---")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    T_conv = 1.0
    colors = ['blue', 'green', 'red', 'purple']
    for M_test, col in zip([10, 20, 40], colors):
        try:
            heston_p = fit_heston_control_variate(
                np.linspace(0.5, 5.0, 8), T_conv, M_test, a, b, c,
                RHO, P_COEFFS, xi0_func, XI0
            )
            _, ivs_m = price_smile_vectorised(
                T_conv, M_test, a, b, c, RHO, P_COEFFS, xi0_func,
                heston_p, LM_GRID, S0, n_quad=N_QUAD
            )
            mask = np.isfinite(ivs_m)
            ax2.plot(LM_GRID[mask], ivs_m[mask], color=col,
                     lw=2, label=f'M={M_test}')
            print(f"  M={M_test}: ATM IV = {np.nanmedian(ivs_m):.4f}")
        except Exception as e:
            print(f"  M={M_test} failed: {e}")

    ax2.set_title(f'Quintic OU Smile Convergence in M (T={T_conv})')
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Implied vol')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2_path = RESULTS_DIR / 'exp2_convergence_in_M.png'
    plt.savefig(fig2_path, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # Append to RESULTS.md
    md_path = RESULTS_DIR / 'RESULTS.md'
    with open(md_path, 'a') as f:
        f.write("\n".join(results_md))
        f.write(f"\nFigures: exp2_quintic_ou_smiles.png, exp2_convergence_in_M.png\n\n")
    print(f"Appended results to {md_path}")

    print("\nExperiment 2 complete.")


if __name__ == '__main__':
    main()
