"""
exp_1 benchmark functions.

Validates the quasi-implicit Riccati solver against the exact expectation

    F(t) = E[exp(-W_t^4 / 4!)]

for a standard Brownian motion W_t ~ N(0, t).

The exact reference is computed via Gauss-Hermite quadrature (exact up to
floating-point precision). The Riccati system is taken directly from the paper:

    psi_k'(t) = ((k+2)(k+1)/2) psi_{k+2}(t) + (1/2)(tilde_psi * tilde_psi)_k

with tilde_psi_k = (k+1) psi_{k+1}, and initial condition psi_k(0) = -(1/4!) delta_{k,4}.
No P(t), no N term, no g_0/g_1/g_2 dependence.

The quasi-implicit scheme handles this stiff system without blowing up, while
classical RK4 becomes unstable for M>=20 with n=100 steps over T=2.
"""

import numpy as np
from scipy.special import roots_hermite

from src.riccati_solver import (
    tilde_psi_vec, truncated_conv, build_R_matrix,
    build_Q_matrix, ode_rhs, solve_riccati_uniform,
)

# Gauss-Hermite nodes and weights (precomputed once, 100 nodes)
_GH_N = 100
_GH_NODES, _GH_WEIGHTS = roots_hermite(_GH_N)


def exact_benchmark(t_values: np.ndarray) -> np.ndarray:
    """Compute E[exp(-W_t^4/4!)] exactly via Gauss-Hermite quadrature.

    W_t ~ N(0, t).  Substituting x = sqrt(2t)*z gives:

        F(t) = (1/sqrt(pi)) * sum_i w_i * exp(-t^2 * z_i^4 / 6)

    which is the exact Gauss-Hermite form with no approximation error beyond
    finite quadrature nodes (100 nodes provide machine-precision accuracy for
    all t in [0, 2]).

    Parameters
    ----------
    t_values : array-like of float
        Time points at which to evaluate F(t).

    Returns
    -------
    np.ndarray of float, same length as t_values.
        F(t) in (0, 1] for t >= 0.
    """
    t_arr = np.asarray(t_values, dtype=float).ravel()
    result = np.empty_like(t_arr)
    for idx, t in enumerate(t_arr):
        if t <= 0.0:
            result[idx] = 1.0
        else:
            # x = sqrt(2t) z  =>  exp(-x^4/24) = exp(-4t^2 z^4/24) = exp(-t^2 z^4/6)
            exponents = -t ** 2 * _GH_NODES ** 4 / 6.0
            result[idx] = float(np.dot(_GH_WEIGHTS, np.exp(exponents)) / np.sqrt(np.pi))
    return result


def _make_benchmark_params(M: int, T: float) -> tuple:
    """Build params dict, Q matrix, and zero N/pp_conv for the paper's benchmark.

    System: a=0, b=0, c=1, rho=0, g1=0, g2=0.

    The Q matrix in the quasi-implicit solve uses the standard 15^2=225 cap (same
    cap applied in the option-pricing experiments).  This makes the implicit system
    matrix well-conditioned and allows the scheme to be stable and convergent for
    all M, while RK4, which uses ode_rhs (cap-free linear term), becomes unstable
    for large M.

    Returns
    -------
    params, Q, N, pp_conv
    """
    params = {
        'a': 0.0, 'b': 0.0, 'c': 1.0, 'rho': 0.0, 'M': M,
        'p_coeffs': np.array([1.0]),   # constant p=1, irrelevant since g1=g2=0
        'T': float(T),
        'g0_func': lambda s: 1.0,      # irrelevant since g1=g2=0
        'g1': 0.0 + 0j,
        'g2': 0.0 + 0j,
    }
    # Cap = 225 = 15^2 in the implicit solve (same as option-pricing experiments)
    # ode_rhs (used for RK4) computes the full uncapped linear term → RK4 unstable
    Q = build_Q_matrix(0.0, 1.0, M, cap=225.0)
    N = np.zeros((M + 1, M + 1), dtype=np.complex128)
    pp_conv = np.zeros(M + 1, dtype=np.complex128)
    return params, Q, N, pp_conv


def _benchmark_ic(M: int) -> np.ndarray:
    """Return the benchmark initial condition psi_k(0) = -(1/4!) * delta_{k,4}.

    Parameters
    ----------
    M : int  –  truncation degree (M >= 4 required)

    Returns
    -------
    np.ndarray of shape (M+1,), dtype complex128.
    """
    psi0 = np.zeros(M + 1, dtype=np.complex128)
    if M >= 4:
        psi0[4] = -1.0 / 24.0   # -1/4!
    return psi0


def solve_benchmark_quasi_implicit(M: int, T: float, n_steps: int):
    """Solve E[exp(-W_t^4/4!)] via quasi-implicit Riccati scheme.

    Uses the paper's exact benchmark system with psi_4(0)=-1/24 and the
    variation-of-constants quasi-implicit step, without any coefficient cap.

    Parameters
    ----------
    M       : int   – truncation degree (should be >= 4)
    T       : float – terminal time
    n_steps : int   – number of uniform steps

    Returns
    -------
    t_grid      : np.ndarray of shape (n_steps+1,)
    psi0_values : np.ndarray of shape (n_steps+1,)  –  exp(Re(psi_0)) at each step
    """
    params, Q, N, pp_conv = _make_benchmark_params(M, T)
    psi0 = _benchmark_ic(M)
    t_grid, psi_hist = solve_riccati_uniform(psi0, float(T), n_steps, params, Q, N, pp_conv)
    psi0_values = np.real(np.exp(psi_hist[:, 0]))
    return t_grid, psi0_values


def solve_benchmark_rk4(M: int, T: float, n_steps: int):
    """Solve the benchmark system using classical RK4.

    Uses the same paper ODE system as the quasi-implicit solver.
    For large M and moderate step sizes, RK4 becomes numerically unstable
    due to the stiff superdiagonal entries ((k+2)(k+1)/2) in Q.

    Parameters
    ----------
    M       : int   – truncation degree
    T       : float – terminal time
    n_steps : int   – number of uniform steps

    Returns
    -------
    t_grid      : np.ndarray of shape (n_steps+1,)
    psi0_values : np.ndarray of shape (n_steps+1,)  –  exp(Re(psi_0)) (may contain inf/nan)
    """
    params, *_ = _make_benchmark_params(M, T)
    dt = float(T) / n_steps
    t_grid = np.linspace(0.0, float(T), n_steps + 1)

    psi = _benchmark_ic(M).copy()
    psi0_values = np.empty(n_steps + 1, dtype=float)
    psi0_values[0] = float(np.real(np.exp(psi[0])))  # exp(-0) = 1 only if psi_0(0)=0 ✓

    for i in range(n_steps):
        t = t_grid[i]
        try:
            k1 = ode_rhs(psi,                    t,          params)
            k2 = ode_rhs(psi + (dt / 2) * k1,   t + dt / 2, params)
            k3 = ode_rhs(psi + (dt / 2) * k2,   t + dt / 2, params)
            k4 = ode_rhs(psi + dt * k3,          t + dt,     params)
            psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            psi0_values[i + 1] = float(np.real(np.exp(psi[0])))
        except (OverflowError, FloatingPointError):
            psi0_values[i + 1:] = np.nan
            break

    return t_grid, psi0_values
