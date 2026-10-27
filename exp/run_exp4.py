"""
Experiment 4: Volatility swap pricing via Laplace inversion.

Computes the vol-swap term structure for both the Quintic OU and Bergomi
models and validates against Monte Carlo benchmarks.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.makedirs(Path(__file__).parent.parent / 'results', exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.quintic_ou import ou_params_from_alpha_epsilon
from src.bergomi import bergomi_poly_coeffs
from src.vol_swap import (
    vol_swap_term_structure,
    mc_vol_swap_rate,
)

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
V0 = 0.025
K_RATE = 5.0
VINF = 0.06
T_VALUES = np.linspace(0.1, 2.0, 20)
M = 32
N_QUAD = 64
N_MC_PATHS = 10_000
N_MC_STEPS = 252
MC_T_SUBSAMPLE = [0.25, 0.5, 1.0, 2.0]   # subset for MC validation

# Quintic OU
QOU_RHO = -0.65
QOU_ALPHA = -0.6
QOU_EPSILON = 1.0 / 52
QOU_P = [0.01, 1.0, 0.0, 0.214, 0.0, 0.227]

# Bergomi
BGM_RHO = -0.7
BGM_ALPHA = -0.7
BGM_EPSILON = 1.0 / 52
BGM_ETA = 1.2
BGM_L = 8

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def xi0_func(t):
    """Exponentially mean-reverting forward variance curve."""
    return V0 * np.exp(-K_RATE * t) + VINF * (1.0 - np.exp(-K_RATE * t))


def main():
    print("=" * 60)
    print("Experiment 4: Volatility Swap Term Structure")
    print("=" * 60)
    print(f"xi0(t) = {V0}*exp(-{K_RATE}*t) + {VINF}*(1-exp(-{K_RATE}*t))")
    print(f"Maturities: {T_VALUES[0]:.2f} to {T_VALUES[-1]:.2f} ({len(T_VALUES)} points)")
    print(f"M={M}, n_quad={N_QUAD}")

    results_md = ["## Experiment 4: Volatility Swap Term Structure\n",
                  f"xi0: mean-reverting from {V0} to {VINF}, k={K_RATE}\n\n"]

    # -----------------------------------------------------------------------
    # Quintic OU
    # -----------------------------------------------------------------------
    print("\n--- Quintic OU ---")
    a_q, b_q, c_q = ou_params_from_alpha_epsilon(QOU_ALPHA, QOU_EPSILON)

    t0 = time.time()
    try:
        vs_qou = vol_swap_term_structure(
            T_VALUES, M, a_q, b_q, c_q, QOU_P, xi0_func, n_quad=N_QUAD
        )
        print(f"  Transform method: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Failed: {e}")
        vs_qou = np.full(len(T_VALUES), np.nan)

    # MC benchmarks (subset of maturities)
    mc_means_q, mc_los_q, mc_his_q, mc_T_q = [], [], [], []
    for T in MC_T_SUBSAMPLE:
        try:
            mean, lo, hi = mc_vol_swap_rate(
                T, N_MC_PATHS, N_MC_STEPS, b_q, c_q, xi0_func,
                vol_model='quintic_ou', p_coeffs=QOU_P
            )
            mc_means_q.append(mean)
            mc_los_q.append(lo)
            mc_his_q.append(hi)
            mc_T_q.append(T)
            print(f"  MC T={T}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  MC T={T} failed: {e}")

    # -----------------------------------------------------------------------
    # Bergomi
    # -----------------------------------------------------------------------
    print("\n--- Bergomi ---")
    a_b, b_b, c_b = ou_params_from_alpha_epsilon(BGM_ALPHA, BGM_EPSILON)
    p_bgm = bergomi_poly_coeffs(BGM_ETA, BGM_L)

    t0 = time.time()
    try:
        vs_bgm = vol_swap_term_structure(
            T_VALUES, M, a_b, b_b, c_b, p_bgm, xi0_func, n_quad=N_QUAD
        )
        print(f"  Transform method: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Failed: {e}")
        vs_bgm = np.full(len(T_VALUES), np.nan)

    mc_means_b, mc_los_b, mc_his_b, mc_T_b = [], [], [], []
    for T in MC_T_SUBSAMPLE:
        try:
            mean, lo, hi = mc_vol_swap_rate(
                T, N_MC_PATHS, N_MC_STEPS, b_b, c_b, xi0_func,
                vol_model='bergomi', eta=BGM_ETA
            )
            mc_means_b.append(mean)
            mc_los_b.append(lo)
            mc_his_b.append(hi)
            mc_T_b.append(T)
            print(f"  MC T={T}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  MC T={T} failed: {e}")

    # -----------------------------------------------------------------------
    # Flat-variance reference
    # -----------------------------------------------------------------------
    vs_flat = np.sqrt(np.array([xi0_func(T) for T in T_VALUES]))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Quintic OU
    ax1.plot(T_VALUES, vs_qou, 'b-', lw=2, label='Quintic OU (Laplace)')
    ax1.plot(T_VALUES, vs_flat, 'k--', lw=1, alpha=0.6, label=r'$\sqrt{\xi_0(T)}$')
    if mc_T_q:
        ax1.errorbar(mc_T_q, mc_means_q,
                     yerr=[np.array(mc_means_q) - np.array(mc_los_q),
                           np.array(mc_his_q) - np.array(mc_means_q)],
                     fmt='rs', ms=5, capsize=4, label='MC (95% CI)')
    ax1.set_title('Quintic OU Vol Swap Term Structure')
    ax1.set_xlabel('Maturity T')
    ax1.set_ylabel('Vol-swap rate $K_{1/2}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bergomi
    ax2.plot(T_VALUES, vs_bgm, 'b-', lw=2, label='Bergomi (Laplace)')
    ax2.plot(T_VALUES, vs_flat, 'k--', lw=1, alpha=0.6, label=r'$\sqrt{\xi_0(T)}$')
    if mc_T_b:
        ax2.errorbar(mc_T_b, mc_means_b,
                     yerr=[np.array(mc_means_b) - np.array(mc_los_b),
                           np.array(mc_his_b) - np.array(mc_means_b)],
                     fmt='rs', ms=5, capsize=4, label='MC (95% CI)')
    ax2.set_title('Bergomi Vol Swap Term Structure')
    ax2.set_xlabel('Maturity T')
    ax2.set_ylabel('Vol-swap rate $K_{1/2}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'exp4_vol_swap_term_structure.png'
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # Comparison plot
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_VALUES, vs_qou, 'b-', lw=2, label='Quintic OU')
    ax.plot(T_VALUES, vs_bgm, 'r-', lw=2, label='Bergomi')
    ax.plot(T_VALUES, vs_flat, 'k--', lw=1.5, alpha=0.7, label=r'$\sqrt{\xi_0(T)}$')
    ax.set_title('Vol Swap Term Structure: Model Comparison')
    ax.set_xlabel('Maturity T')
    ax.set_ylabel('Vol-swap rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2_path = RESULTS_DIR / 'exp4_vol_swap_comparison.png'
    plt.savefig(fig2_path, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # Record results
    results_md.append("### Quintic OU\n")
    for T, vs in zip(T_VALUES, vs_qou):
        results_md.append(f"- T={T:.2f}: K_{{1/2}} = {vs:.4f}\n")
    results_md.append("\n### Bergomi\n")
    for T, vs in zip(T_VALUES, vs_bgm):
        results_md.append(f"- T={T:.2f}: K_{{1/2}} = {vs:.4f}\n")
    results_md.append("\n")

    md_path = RESULTS_DIR / 'RESULTS.md'
    with open(md_path, 'a') as f:
        f.write("\n".join(results_md))
        f.write(f"\nFigures: exp4_vol_swap_term_structure.png, exp4_vol_swap_comparison.png\n\n")
    print(f"Appended results to {md_path}")

    print("\nExperiment 4 complete.")


if __name__ == '__main__':
    main()
