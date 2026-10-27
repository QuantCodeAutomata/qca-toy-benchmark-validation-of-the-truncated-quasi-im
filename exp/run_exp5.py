"""
Experiment 5: Market calibration using synthetic SPX option data.

Since real CBOE data (23 Oct 2017 / 9 Nov 2021) is unavailable, we generate
synthetic data from the Quintic OU model with known parameters, add a
realistic bid-ask spread, calibrate back, and report parameter recovery.
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

from src.calibration import (
    generate_synthetic_market,
    calibrate_model,
    plot_calibration_results,
)
from src.quintic_ou import ou_params_from_alpha_epsilon

# ---------------------------------------------------------------------------
# True parameters (paper defaults)
# ---------------------------------------------------------------------------
TRUE_PARAMS = {
    'rho': -0.65,
    'alpha': -0.6,
    'epsilon': 1.0 / 52,
    'p_coeffs': [0.01, 1.0, 0.0, 0.214, 0.0, 0.227],
    'xi0': 0.025,
}
XI0 = TRUE_PARAMS['xi0']
S0 = 1.0

MATURITIES = [0.5, 1.0, 2.0]
LM_GRID = np.linspace(-0.20, 0.20, 15)
M = 20
BID_ASK_SPREAD = 0.002  # ±0.2 vol points

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def xi0_func(t):
    return XI0


def main():
    print("=" * 60)
    print("Experiment 5: Synthetic Market Calibration")
    print("=" * 60)
    print("Using synthetic data from Quintic OU model (known parameters)")
    print(f"True params: {TRUE_PARAMS}\n")

    results_md = [
        "## Experiment 5: Synthetic Market Calibration\n",
        "Real CBOE data not available; using synthetic market.\n",
        f"True parameters: {TRUE_PARAMS}\n\n",
    ]

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic market data
    # -----------------------------------------------------------------------
    print("Step 1: Generating synthetic market data...")
    t0 = time.time()
    try:
        market_data = generate_synthetic_market(
            model='quintic_ou',
            params=TRUE_PARAMS,
            maturities=MATURITIES,
            strikes_lm=LM_GRID,
            S0=S0,
        )
        market_data['bid'] = market_data['iv'] - BID_ASK_SPREAD
        market_data['ask'] = market_data['iv'] + BID_ASK_SPREAD
        print(f"  Generated {len(market_data)} data points in {time.time()-t0:.1f}s")
        print(f"  IV range: [{market_data['iv'].min():.4f}, {market_data['iv'].max():.4f}]")
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # -----------------------------------------------------------------------
    # Step 2: Initial parameters (perturbed from true)
    # -----------------------------------------------------------------------
    # calibrate_model expects 'p3' and 'p5' as keys (not 'p_coeffs')
    initial_params = {
        'rho': -0.50,
        'alpha': -0.50,
        'epsilon': 1.0 / 52,
        'p3': 0.15,
        'p5': 0.15,
        'xi0': 0.030,
    }
    print(f"\nInitial (perturbed) params: {initial_params}")

    # -----------------------------------------------------------------------
    # Step 3: Calibrate
    # -----------------------------------------------------------------------
    print("\nStep 3: Running Nelder-Mead calibration...")
    t0 = time.time()
    try:
        result = calibrate_model(
            model='quintic_ou',
            market_data=market_data,
            S0=S0,
            xi0_func=xi0_func,
            initial_params=initial_params,
            M=M,
            method='Nelder-Mead',
        )
        print(f"  Calibration complete in {result['runtime']:.1f}s")
        print(f"  Iterations: {result.get('n_iterations', 'N/A')}")
        print(f"  Objective (MSE): {result['objective']:.6f}")
        cal_params = result['params']
        print(f"  Calibrated params: {cal_params}")
    except Exception as e:
        print(f"  Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        cal_params = initial_params
        result = {'objective': np.nan, 'runtime': 0, 'n_iterations': 0, 'params': cal_params}

    # -----------------------------------------------------------------------
    # Step 4: Report recovery
    # -----------------------------------------------------------------------
    print("\n--- Parameter Recovery ---")
    results_md.append("### Parameter Recovery\n")
    results_md.append("| Parameter | True | Initial | Calibrated |\n")
    results_md.append("|-----------|------|---------|------------|\n")

    # True p_coeffs[3] = p3, p_coeffs[5] = p5
    true_p3 = TRUE_PARAMS['p_coeffs'][3]
    true_p5 = TRUE_PARAMS['p_coeffs'][5]
    recovery_map = {
        'rho': (TRUE_PARAMS['rho'], initial_params['rho']),
        'alpha': (TRUE_PARAMS['alpha'], initial_params['alpha']),
        'p3': (true_p3, initial_params['p3']),
        'p5': (true_p5, initial_params['p5']),
    }
    for key, (true_v, init_v) in recovery_map.items():
        cal_v = cal_params.get(key, np.nan)
        print(f"  {key}: true={true_v:.4f}, init={init_v:.4f}, cal={cal_v:.4f}")
        results_md.append(f"| {key} | {true_v:.4f} | {init_v:.4f} | {cal_v:.4f} |\n")

    results_md.append(f"\nObjective MSE: {result['objective']:.6f}\n")
    results_md.append(f"Runtime: {result['runtime']:.1f}s\n\n")

    # -----------------------------------------------------------------------
    # Step 5: Plot calibrated smiles vs synthetic market
    # -----------------------------------------------------------------------
    print("\nStep 5: Plotting results...")
    try:
        save_path = str(RESULTS_DIR / 'exp5_calibration_smiles.png')
        plot_calibration_results(
            model='quintic_ou',
            calibrated_params=cal_params,
            market_data=market_data,
            xi0_func=xi0_func,
            S0=S0,
            save_path=save_path,
        )
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")
        # Fallback: simple matplotlib plot
        fig, axes = plt.subplots(1, len(MATURITIES), figsize=(14, 5))
        if len(MATURITIES) == 1:
            axes = [axes]
        for ax, T in zip(axes, MATURITIES):
            sub = market_data[market_data['maturity'] == T]
            if len(sub) > 0:
                ax.fill_between(sub['log_moneyness'], sub['bid'], sub['ask'],
                                alpha=0.3, color='gray', label='Bid-Ask')
                ax.plot(sub['log_moneyness'], sub['iv'], 'ko', ms=3, label='Market mid')
            ax.set_title(f'T={T}')
            ax.set_xlabel('Log-moneyness')
            ax.set_ylabel('IV')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle('Calibration Results (Quintic OU)')
        plt.tight_layout()
        fig_path = RESULTS_DIR / 'exp5_calibration_smiles.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Fallback plot saved: {fig_path}")

    # -----------------------------------------------------------------------
    # Step 6: Save results
    # -----------------------------------------------------------------------
    md_path = RESULTS_DIR / 'RESULTS.md'
    with open(md_path, 'a') as f:
        f.write("\n".join(results_md))
        f.write("\nFigures: exp5_calibration_smiles.png\n\n")
    print(f"Appended results to {md_path}")

    print("\nExperiment 5 complete.")


if __name__ == '__main__':
    main()
