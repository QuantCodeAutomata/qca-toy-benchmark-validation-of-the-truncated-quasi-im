"""
Experiment 3: One-factor Bergomi model option pricing.

Computes implied-volatility smiles for the polynomial approximation of the
Bergomi model and validates against a simplified Monte Carlo benchmark.
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
    ou_params_from_alpha_epsilon,
    heston_char_func, fit_heston_control_variate,
    bs_implied_vol, bs_call_price,
)
from src.bergomi import (
    bergomi_poly_coeffs, char_func_bergomi, char_func_bergomi_batch,
)
from src.vol_swap import mc_vol_swap_rate

# ---------------------------------------------------------------------------
# Model parameters (paper defaults)
# ---------------------------------------------------------------------------
RHO = -0.7
ALPHA = -0.7
EPSILON = 1.0 / 52
ETA = 1.2
L = 8
XI0 = 0.025
S0 = 1.0

MATURITIES = [1.0 / 52, 0.5, 1.0, 2.0]
LM_GRID = np.linspace(-0.25, 0.25, 25)
M = 20
N_QUAD = 32
N_MC_PATHS = 10_000
N_MC_STEPS = 100

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def xi0_func(t):
    return XI0


def price_smile_bergomi(T, M, a, b, c, rho, eta, L, xi0_func,
                        heston_p, lm_grid, S0, n_quad=32):
    """Price Bergomi smiles using batch CF evaluation (fast)."""
    gl_nodes, gl_weights = roots_laguerre(n_quad)

    # batch: phi_m[j] = phi(u_j - 0.5j) solved once for all nodes
    phi_m = char_func_bergomi_batch(gl_nodes, T, M, a, b, c, rho, eta, L, xi0_func)
    phi_h = np.array([
        heston_char_func(u - 0.5j, T, heston_p['v0'], heston_p['kappa'],
                         heston_p['theta'], heston_p['sigma_h'], heston_p['rho_h'])
        for u in gl_nodes
    ], dtype=complex)

    w_eff = gl_weights * np.exp(gl_nodes)
    denom = gl_nodes ** 2 + 0.25

    prices, ivs = [], []
    for lm in lm_grid:
        K = S0 * math.exp(lm)
        phase = np.exp((1j * gl_nodes + 0.5) * lm)

        integ_h = w_eff * np.real(phase * phi_h) / denom
        C_Heston = max(S0 - K / math.pi * float(np.sum(integ_h)), max(S0 - K, 0.0))

        integ_cv = w_eff * np.real(phase * (phi_m - phi_h)) / denom
        C_model = max(C_Heston - K / math.pi * float(np.sum(integ_cv)), max(S0 - K, 0.0))

        iv = bs_implied_vol(C_model, S0, K, T)
        prices.append(C_model)
        ivs.append(iv if np.isfinite(iv) else np.nan)

    return np.array(prices), np.array(ivs)


def main():
    print("=" * 60)
    print("Experiment 3: One-Factor Bergomi Model Option Pricing")
    print("=" * 60)

    a, b, c = ou_params_from_alpha_epsilon(ALPHA, EPSILON)
    p_coeffs = bergomi_poly_coeffs(ETA, L)
    print(f"OU params: a={a}, b={b:.2f}, c={c:.4f}")
    print(f"eta={ETA}, L={L}, M={M}, n_quad={N_QUAD}")
    print(f"MC: {N_MC_PATHS} paths x {N_MC_STEPS} steps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    results_md = ["## Experiment 3: One-Factor Bergomi Model Option Pricing\n",
                  f"Parameters: rho={RHO}, alpha={ALPHA}, epsilon={EPSILON:.4f}, "
                  f"eta={ETA}, L={L}, xi0={XI0}, M={M}\n\n"]

    for ax_i, T in enumerate(MATURITIES):
        print(f"\n--- Maturity T = {T:.4f} ---")
        t0 = time.time()
        ax = axes[ax_i]

        # Heston CV fit
        u_fit = np.linspace(0.5, 5.0, 8)
        try:
            heston_p = fit_heston_control_variate(
                u_fit, T, M, a, b, c, RHO, p_coeffs, xi0_func, XI0
            )
        except Exception as e:
            print(f"  Heston CV fit failed: {e}")
            heston_p = {'v0': XI0, 'kappa': 2.0, 'theta': XI0,
                        'sigma_h': 0.3, 'rho_h': RHO * 0.8}

        # Bergomi smiles
        try:
            prices, ivs = price_smile_bergomi(
                T, M, a, b, c, RHO, ETA, L, xi0_func,
                heston_p, LM_GRID, S0, n_quad=N_QUAD
            )
        except Exception as e:
            print(f"  Pricing failed: {e}")
            prices = np.full(len(LM_GRID), np.nan)
            ivs = np.full(len(LM_GRID), np.nan)

        t_riccati = time.time() - t0

        # MC benchmark (Bergomi exact dynamics)
        t_mc = time.time()
        try:
            mc_mean, mc_lo, mc_hi = mc_vol_swap_rate(
                T, N_MC_PATHS, N_MC_STEPS, b, c, xi0_func,
                vol_model='bergomi', eta=ETA
            )
        except Exception as e:
            print(f"  MC failed: {e}")
            mc_mean, mc_lo, mc_hi = np.nan, np.nan, np.nan
        t_mc_elapsed = time.time() - t_mc

        print(f"  Riccati: {t_riccati:.1f}s, MC vol swap: {mc_mean:.4f} "
              f"[{mc_lo:.4f}, {mc_hi:.4f}] ({t_mc_elapsed:.1f}s)")

        mask = np.isfinite(ivs)
        ax.plot(LM_GRID[mask], ivs[mask], 'b-', lw=2, label='Riccati (M=20)')
        if np.isfinite(mc_mean):
            ax.axhline(mc_mean, color='r', ls='--', alpha=0.7, label='MC vol swap')
            ax.axhspan(mc_lo, mc_hi, alpha=0.15, color='red')
        ax.set_title(f'T = {T:.3f} yr')
        ax.set_xlabel('Log-moneyness')
        ax.set_ylabel('Implied vol')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(LM_GRID[0], LM_GRID[-1])

        results_md.append(f"### T = {T:.4f}\n")
        results_md.append(f"- ATM IV: {np.nanmedian(ivs):.4f}\n")
        results_md.append(f"- MC vol swap: {mc_mean:.4f}\n\n")

    fig.suptitle('One-Factor Bergomi: Implied Volatility Smiles', fontsize=13)
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'exp3_bergomi_smiles.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # Convergence in M
    print("\n--- Convergence in M ---")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    T_conv = 1.0
    for M_test, col in zip([10, 20, 40], ['blue', 'green', 'red']):
        try:
            heston_p = fit_heston_control_variate(
                np.linspace(0.5, 5.0, 8), T_conv, M_test, a, b, c,
                RHO, p_coeffs, xi0_func, XI0
            )
            _, ivs_m = price_smile_bergomi(
                T_conv, M_test, a, b, c, RHO, ETA, L, xi0_func,
                heston_p, LM_GRID, S0, n_quad=N_QUAD
            )
            mask = np.isfinite(ivs_m)
            ax2.plot(LM_GRID[mask], ivs_m[mask], color=col, lw=2, label=f'M={M_test}')
            print(f"  M={M_test}: ATM IV = {np.nanmedian(ivs_m):.4f}")
        except Exception as e:
            print(f"  M={M_test} failed: {e}")

    ax2.set_title(f'Bergomi Smile Convergence in M (T={T_conv})')
    ax2.set_xlabel('Log-moneyness')
    ax2.set_ylabel('Implied vol')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2_path = RESULTS_DIR / 'exp3_convergence_in_M.png'
    plt.savefig(fig2_path, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    md_path = RESULTS_DIR / 'RESULTS.md'
    with open(md_path, 'a') as f:
        f.write("\n".join(results_md))
        f.write(f"\nFigures: exp3_bergomi_smiles.png, exp3_convergence_in_M.png\n\n")
    print(f"Appended results to {md_path}")

    print("\nExperiment 3 complete.")


if __name__ == '__main__':
    main()
