"""
One-factor Bergomi model for option pricing (exp_3).

The original model:
    sigma_t^orig = sqrt(xi0(t)) * exp(0.5*eta*X_t - 0.25*eta^2 * E[X_t^2])

Polynomial approximation:
    sigma_t^approx = g0(t) * p(X_t),  p_k = eta^k / (2^k * k!),  k = 0,...,L

where g0(t) = sqrt(xi0(t) / E[p(X_t)^2]).

Paper defaults: rho=-0.7, alpha=-0.7, epsilon=1/52, eta=1.2, xi0=0.025, L=8.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.quintic_ou import (
    ou_params_from_alpha_epsilon, ou_variance,
    poly_second_moment, g0_func_builder,
    build_pp_conv, char_func_riccati, char_func_batch,
)
from src.riccati_solver import build_Q_matrix, build_N_matrix, solve_riccati_adaptive


# ---------------------------------------------------------------------------
# Polynomial approximation to exp(0.5*eta*x)
# ---------------------------------------------------------------------------

def bergomi_poly_coeffs(eta, L=8):
    """Taylor coefficients of exp(0.5*eta*x) truncated at degree L.

    p_k = eta^k / (2^k * k!)  for k = 0, ..., L.

    Parameters
    ----------
    eta : float – vol-of-vol
    L   : int   – polynomial degree (default 8)

    Returns
    -------
    np.ndarray of shape (L+1,)
    """
    coeffs = np.empty(L + 1)
    for k in range(L + 1):
        coeffs[k] = (eta / 2.0) ** k / math.factorial(k)
    return coeffs


# ---------------------------------------------------------------------------
# Original Bergomi instantaneous variance
# ---------------------------------------------------------------------------

def bergomi_orig_variance(X_t, xi0_t, eta, EX2):
    """Instantaneous variance in the original Bergomi model.

    sigma_t^2 = xi0(t) * exp(eta * X_t - 0.5 * eta^2 * E[X_t^2])

    Parameters
    ----------
    X_t   : float or np.ndarray – OU factor realisation
    xi0_t : float or np.ndarray – forward variance at t
    eta   : float – vol-of-vol
    EX2   : float – E[X_t^2] = Var[X_t]

    Returns
    -------
    float or np.ndarray
    """
    return xi0_t * np.exp(eta * X_t - 0.5 * eta ** 2 * EX2)


# ---------------------------------------------------------------------------
# Build polynomial system
# ---------------------------------------------------------------------------

def build_bergomi_system(eta, alpha, epsilon, L=8):
    """Build the Bergomi polynomial approximation system.

    Parameters
    ----------
    eta     : float – vol-of-vol
    alpha   : float – Hurst-like exponent
    epsilon : float – time scale
    L       : int   – polynomial truncation degree

    Returns
    -------
    dict with keys: p_coeffs, a, b, c
    """
    a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)
    p_coeffs = bergomi_poly_coeffs(eta, L)
    return {'p_coeffs': p_coeffs, 'a': a, 'b': b, 'c': c, 'eta': eta, 'L': L}


# ---------------------------------------------------------------------------
# Bergomi characteristic function
# ---------------------------------------------------------------------------

def char_func_bergomi(u, T, M, a, b, c, rho, eta, L, xi0_func,
                      n_steps: int = None):
    """Characteristic function for the Bergomi polynomial approximation (single u).

    Parameters
    ----------
    u       : float or complex
    T       : float – maturity
    M       : int   – Riccati truncation order
    a, b, c : float – OU parameters
    rho     : float – correlation
    eta     : float – vol-of-vol
    L       : int   – Bergomi polynomial degree
    xi0_func: callable t -> float
    n_steps : int or None – fixed step count (None → auto)

    Returns
    -------
    complex
    """
    p_coeffs = bergomi_poly_coeffs(eta, L)
    return char_func_riccati(u, T, M, a, b, c, rho, p_coeffs, xi0_func, n_steps)


def char_func_bergomi_batch(u_arr: np.ndarray, T: float, M: int,
                             a: float, b: float, c: float, rho: float,
                             eta: float, L: int, xi0_func,
                             n_steps: int = None) -> np.ndarray:
    """Batch CF evaluation for Bergomi polynomial approximation.

    Parameters
    ----------
    u_arr   : 1-D real np.ndarray – quadrature nodes
    T       : float – maturity
    M       : int   – Riccati truncation
    a, b, c : float – OU parameters
    rho     : float – correlation
    eta     : float – vol-of-vol
    L       : int   – polynomial degree
    xi0_func: callable t -> float
    n_steps : int or None

    Returns
    -------
    np.ndarray of shape (len(u_arr),) – phi(u_j - 0.5j)
    """
    p_coeffs = bergomi_poly_coeffs(eta, L)
    return char_func_batch(u_arr, T, M, a, b, c, rho, p_coeffs, xi0_func, n_steps)
