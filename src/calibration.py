"""
Market calibration framework (exp_5).

Since real CBOE DataShop data (23 Oct 2017 and 9 Nov 2021) is not available,
this module implements a synthetic calibration that demonstrates the methodology
using data generated from known model parameters.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.quintic_ou import (
    ou_params_from_alpha_epsilon, char_func_riccati,
    lewis_price_with_cv, lewis_price_batch_strikes,
    bs_implied_vol, bs_call_price,
    fit_heston_control_variate, g0_func_builder, build_pp_conv,
)
from src.bergomi import char_func_bergomi, bergomi_poly_coeffs


# ---------------------------------------------------------------------------
# Default paper parameters
# ---------------------------------------------------------------------------

_QUINTIC_DEFAULTS = {
    'rho': -0.65, 'alpha': -0.6, 'epsilon': 1.0 / 52,
    'p_coeffs': [0.01, 1.0, 0.0, 0.214, 0.0, 0.227],
    'xi0': 0.025,
}

_BERGOMI_DEFAULTS = {
    'rho': -0.7, 'alpha': -0.7, 'epsilon': 1.0 / 52,
    'eta': 1.2, 'xi0': 0.025, 'L': 8,
}


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------

def generate_synthetic_market(model='quintic_ou', params=None,
                               maturities=None, strikes_lm=None,
                               S0=1.0):
    """Generate synthetic option market data from the polynomial vol model.

    Parameters
    ----------
    model       : 'quintic_ou' or 'bergomi'
    params      : dict of model parameters (uses paper defaults if None)
    maturities  : list of float (calendar years)
    strikes_lm  : list of float (log-moneyness k = log(K/S0) grid)
    S0          : float – spot price

    Returns
    -------
    pd.DataFrame with columns: maturity, log_moneyness, price, iv, bid, ask
    """
    if maturities is None:
        maturities = [1.0 / 52, 0.5, 1.0, 2.0]
    if strikes_lm is None:
        strikes_lm = list(np.linspace(-0.2, 0.2, 15))

    if model == 'quintic_ou':
        p_def = _QUINTIC_DEFAULTS.copy()
        if params:
            p_def.update(params)
        rho = p_def['rho']
        alpha = p_def['alpha']
        epsilon = p_def['epsilon']
        p_coeffs = p_def['p_coeffs']
        xi0 = p_def['xi0']
        a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
        xi0_func = lambda t, xi=xi0: xi
        M = 20

    elif model == 'bergomi':
        p_def = _BERGOMI_DEFAULTS.copy()
        if params:
            p_def.update(params)
        rho = p_def['rho']
        alpha = p_def['alpha']
        epsilon = p_def['epsilon']
        eta = p_def['eta']
        xi0 = p_def['xi0']
        L = p_def['L']
        a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
        p_coeffs = list(bergomi_poly_coeffs(eta, L))
        xi0_func = lambda t, xi=xi0: xi
        M = 20

    else:
        raise ValueError(f"Unknown model: {model!r}")

    lm_arr = np.asarray(strikes_lm, dtype=float)
    K_arr  = S0 * np.exp(lm_arr)
    spread = 0.001  # half bid-ask spread in IV space

    rows = []
    for T in maturities:
        # Build Heston CV once per maturity
        u_fit = np.linspace(0.5, 8.0, 10)
        try:
            heston_p = fit_heston_control_variate(
                u_fit, T, M, a, b, c, rho, p_coeffs, xi0_func, xi0
            )
        except Exception:
            heston_p = {'v0': xi0, 'kappa': 2.0, 'theta': xi0,
                        'sigma_h': 0.3, 'rho_h': rho * 0.8}

        try:
            prices = lewis_price_batch_strikes(
                S0, K_arr, T, M, a, b, c, rho, p_coeffs,
                xi0_func, heston_p, n_quad=32
            )
        except Exception:
            prices = np.maximum(S0 - K_arr, 0.0)

        for lm, K, price in zip(lm_arr, K_arr, prices):
            try:
                iv = bs_implied_vol(float(price), S0, float(K), T)
            except Exception:
                iv = np.nan
            rows.append({
                'maturity': T,
                'log_moneyness': float(lm),
                'price': float(price),
                'iv': iv if np.isfinite(iv) else np.nan,
                'bid': max(iv - spread, 0.001) if np.isfinite(iv) else np.nan,
                'ask': iv + spread if np.isfinite(iv) else np.nan,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration objective
# ---------------------------------------------------------------------------

def calibration_objective(theta, model, market_data, S0, xi0_func, M=32):
    """MSE between model IVs and market IVs.

    Parameters
    ----------
    theta       : np.ndarray – parameter vector
    model       : 'quintic_ou' or 'bergomi'
    market_data : pd.DataFrame (columns: maturity, log_moneyness, iv)
    S0          : float
    xi0_func    : callable
    M           : int – truncation order

    Returns
    -------
    float – mean squared error in IV space
    """
    try:
        if model == 'quintic_ou':
            rho, alpha, epsilon = theta[0], theta[1], max(theta[2], 1e-5)
            p3, p5 = theta[3], theta[4]
            p_coeffs = [0.01, 1.0, 0.0, p3, 0.0, p5]
            a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)

        elif model == 'bergomi':
            rho, alpha, epsilon, eta = (theta[0], theta[1],
                                        max(theta[2], 1e-5), max(theta[3], 0.01))
            a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
            p_coeffs = list(bergomi_poly_coeffs(eta, L=8))
        else:
            return 1e10

        rho = float(np.clip(rho, -0.999, 0.999))

        total_err = 0.0
        n = 0
        xi0_val = xi0_func(0.0)

        for T in market_data['maturity'].unique():
            sub = market_data[market_data['maturity'] == T].copy()
            valid_mask = sub['iv'].apply(np.isfinite)
            sub = sub[valid_mask]
            if sub.empty:
                continue

            K_arr_T  = (S0 * np.exp(sub['log_moneyness'].values)).astype(float)
            iv_mkt   = sub['iv'].values

            u_fit = np.linspace(0.5, 6.0, 8)
            try:
                heston_p = fit_heston_control_variate(
                    u_fit, T, M, a, b, c, rho, p_coeffs, xi0_func, xi0_val
                )
            except Exception:
                heston_p = {'v0': xi0_val, 'kappa': 2.0, 'theta': xi0_val,
                            'sigma_h': 0.3, 'rho_h': rho}

            try:
                prices_T = lewis_price_batch_strikes(
                    S0, K_arr_T, T, M, a, b, c, rho, p_coeffs,
                    xi0_func, heston_p, n_quad=24
                )
            except Exception:
                total_err += len(K_arr_T)
                n         += len(K_arr_T)
                continue

            for K, price, iv_mkt_k in zip(K_arr_T, prices_T, iv_mkt):
                try:
                    iv_model = bs_implied_vol(float(price), S0, float(K), T)
                except Exception:
                    iv_model = np.nan

                if np.isfinite(iv_model):
                    total_err += (iv_model - iv_mkt_k) ** 2
                else:
                    total_err += 1.0
                n += 1

        return total_err / max(n, 1)

    except Exception:
        return 1e10


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

def calibrate_model(model, market_data, S0, xi0_func, initial_params, M=32,
                    method='Nelder-Mead'):
    """Run Nelder-Mead calibration of the polynomial vol model.

    Parameters
    ----------
    model          : 'quintic_ou' or 'bergomi'
    market_data    : pd.DataFrame
    S0             : float
    xi0_func       : callable
    initial_params : dict with model parameters as starting values
    M              : int
    method         : str (default 'Nelder-Mead')

    Returns
    -------
    dict with keys: params, objective, runtime, n_iterations
    """
    if model == 'quintic_ou':
        x0 = np.array([
            initial_params.get('rho', -0.65),
            initial_params.get('alpha', -0.6),
            initial_params.get('epsilon', 1.0 / 52),
            initial_params.get('p3', 0.214),
            initial_params.get('p5', 0.227),
        ])
    elif model == 'bergomi':
        x0 = np.array([
            initial_params.get('rho', -0.7),
            initial_params.get('alpha', -0.7),
            initial_params.get('epsilon', 1.0 / 52),
            initial_params.get('eta', 1.2),
        ])
    else:
        raise ValueError(f"Unknown model: {model!r}")

    t0 = time.time()
    result = minimize(
        calibration_objective,
        x0,
        args=(model, market_data, S0, xi0_func, M),
        method=method,
        options={'maxiter': 300, 'xatol': 5e-4, 'fatol': 1e-5},
    )
    elapsed = time.time() - t0

    if model == 'quintic_ou':
        calibrated = {
            'rho': float(result.x[0]),
            'alpha': float(result.x[1]),
            'epsilon': float(result.x[2]),
            'p3': float(result.x[3]),
            'p5': float(result.x[4]),
        }
    else:
        calibrated = {
            'rho': float(result.x[0]),
            'alpha': float(result.x[1]),
            'epsilon': float(result.x[2]),
            'eta': float(result.x[3]),
        }

    return {
        'params': calibrated,
        'objective': float(result.fun),
        'runtime': elapsed,
        'n_iterations': result.nit,
        'success': result.success,
    }


# ---------------------------------------------------------------------------
# Plot calibration results
# ---------------------------------------------------------------------------

def plot_calibration_results(model, calibrated_params, market_data, xi0_func,
                             S0=1.0, save_path=None):
    """Plot model smiles vs market bid/ask bands.

    Parameters
    ----------
    model             : 'quintic_ou' or 'bergomi'
    calibrated_params : dict of calibrated parameters
    market_data       : pd.DataFrame
    xi0_func          : callable
    S0                : float
    save_path         : str or None
    """
    maturities = sorted(market_data['maturity'].unique())
    n_rows = (len(maturities) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = np.array(axes).ravel()

    if model == 'quintic_ou':
        rho = calibrated_params['rho']
        alpha = calibrated_params['alpha']
        epsilon = calibrated_params['epsilon']
        p3 = calibrated_params.get('p3', 0.214)
        p5 = calibrated_params.get('p5', 0.227)
        p_coeffs = [0.01, 1.0, 0.0, p3, 0.0, p5]
        a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
        M = 20
    elif model == 'bergomi':
        rho = calibrated_params['rho']
        alpha = calibrated_params['alpha']
        epsilon = calibrated_params['epsilon']
        eta = calibrated_params['eta']
        a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
        p_coeffs = list(bergomi_poly_coeffs(eta, L=8))
        M = 20

    for ax_i, T in enumerate(maturities):
        ax = axes[ax_i]
        sub = market_data[market_data['maturity'] == T].copy()
        lm_grid = sub['log_moneyness'].values
        bids = sub['bid'].values
        asks = sub['ask'].values

        # Market bands
        ax.fill_between(lm_grid, bids, asks, alpha=0.3, color='blue',
                        label='Market bid/ask')

        # Model smile
        xi0_val = xi0_func(0.0)
        u_fit = np.linspace(0.5, 6.0, 8)
        try:
            heston_p = fit_heston_control_variate(
                u_fit, T, M, a, b, c, rho, p_coeffs, xi0_func, xi0_val
            )
        except Exception:
            heston_p = {'v0': xi0_val, 'kappa': 2.0, 'theta': xi0_val,
                        'sigma_h': 0.3, 'rho_h': rho}

        K_plot = S0 * np.exp(lm_grid)
        try:
            prices_plot = lewis_price_batch_strikes(
                S0, K_plot, T, M, a, b, c, rho, p_coeffs,
                xi0_func, heston_p, n_quad=24
            )
        except Exception:
            prices_plot = np.full(len(lm_grid), np.nan)

        model_ivs = []
        for K, price in zip(K_plot, prices_plot):
            try:
                iv = bs_implied_vol(float(price), S0, float(K), T)
            except Exception:
                iv = np.nan
            model_ivs.append(iv)

        model_ivs = np.array(model_ivs, dtype=float)
        mask = np.isfinite(model_ivs)
        ax.plot(lm_grid[mask], model_ivs[mask], 'r-', lw=2, label=f'{model}')
        ax.set_title(f'T = {T:.3f} yr')
        ax.set_xlabel('Log-moneyness')
        ax.set_ylabel('Implied vol')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax_i in range(len(maturities), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(f'{model} calibration vs synthetic market', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
