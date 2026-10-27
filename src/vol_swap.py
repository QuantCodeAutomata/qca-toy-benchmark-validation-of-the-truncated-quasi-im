"""
Volatility swap pricing via Laplace inversion (exp_4).

Implements the formula:
    K_{1/2}(0,T) = E[ sqrt( (1/T) * integral_0^T sigma_s^2 ds ) ]
                 = sigma_BS + (1/(2*sqrt(pi)))
                   * integral_0^inf (F_tilde_BS(u) - F_tilde(u)) / u^{3/2} du

where F_tilde(u) = E[exp(-u/T * integral_0^T sigma_s^2 ds)] is computed
via the Riccati ODE system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np

from src.quintic_ou import (
    ou_params_from_alpha_epsilon, ou_variance,
    poly_second_moment, g0_func_builder, build_pp_conv,
    _n_steps_for_T,
)
from src.riccati_solver import (
    build_Q_matrix, build_N_matrix, build_R_matrix,
    compute_G_diagonal, solve_riccati_adaptive,
)


# ---------------------------------------------------------------------------
# Laplace transform via Riccati
# ---------------------------------------------------------------------------

def laplace_transform_riccati(u_val, T, M, a, b, c, p_coeffs, xi0_func):
    """Compute F_tilde(u) = E[exp(-u/T * integral_0^T sigma_s^2 ds)].

    Uses the Riccati system with g1=0, g2=-u/T.

    Parameters
    ----------
    u_val    : float – Laplace argument (non-negative)
    T        : float – maturity
    M        : int   – polynomial truncation order
    a, b, c  : float – OU parameters (dX = (a+b*X)dt + c*dW)
    p_coeffs : sequence – polynomial coefficients for sigma = g0*p(X)
    xi0_func : callable t -> float (forward variance at calendar time t)

    Returns
    -------
    complex  (imaginary part ≈ 0 for real u_val)
    """
    g0_func = g0_func_builder(xi0_func, p_coeffs, b, c)
    Q_mat = build_Q_matrix(a, c, M, cap=225.0)
    N_mat = build_N_matrix(p_coeffs, M)
    pp_conv = build_pp_conv(p_coeffs, M)

    params = {
        'a': a, 'b': b, 'c': c, 'rho': 0.0, 'M': M,
        'p_coeffs': np.asarray(p_coeffs),
        'T': T,
        'g0_func': g0_func,
        'g1': complex(0.0),
        'g2': complex(-u_val / T),
    }

    psi0 = np.zeros(M + 1, dtype=complex)
    _, psi_hist = solve_riccati_adaptive(
        psi0, T, params, Q_mat, N_mat, pp_conv,
        zeta=1e-6,
        delta_max=min(T / 10.0, 1.0 / 52),
        store_all=False,
    )
    return complex(np.exp(psi_hist[-1, 0]))


# ---------------------------------------------------------------------------
# Batch Laplace transform (shared time-loop, one solve per u per step)
# ---------------------------------------------------------------------------

def laplace_transform_batch(u_arr: np.ndarray, T: float, M: int,
                             a: float, b: float, c: float,
                             p_coeffs, xi0_func,
                             n_steps: int = None) -> np.ndarray:
    """Compute F_tilde(u_j) for all u_j simultaneously via a shared time-loop.

    g1 = 0, g2 = -u_j/T  →  ell = 0  for all u (no correlation term).
    J_i(u) = I − G_i (c²/2 R^(i)(u) + Q),  which still depends on psi(u)
    through R but not on N (since ell=0).

    The inner u-loop is vectorised: R^(i) matrices for all u are assembled
    simultaneously via 3-D index broadcasting, and the linear systems are
    solved in a single batched ``np.linalg.solve`` call.

    Step-count note: the quasi-implicit linearisation of the quadratic term
    requires delta to be small relative to 1/(|b|·c²) to avoid resonance-like
    blow-up at certain Laplace arguments u.  The default heuristic
    n_steps = max(500, ceil(|b|·c²·T/5)) ensures this for both the Quintic OU
    and Bergomi parameterisations used in the paper.

    Parameters
    ----------
    u_arr    : 1-D real np.ndarray – Laplace arguments (≥ 0)
    T        : float – maturity
    M        : int   – truncation order
    a,b,c    : float – OU parameters
    p_coeffs : sequence
    xi0_func : callable t -> float
    n_steps  : int or None – if None, chosen automatically from |b|·c² to
               guarantee stability across the full u-grid.

    Returns
    -------
    F_arr : real np.ndarray of shape (n_u,)  – F_tilde(u_j), clipped to [0,1]
    """
    u_arr   = np.asarray(u_arr, dtype=float)
    n_u     = len(u_arr)
    if n_steps is None:
        # Scale with stiffness |b|·c² so that delta ≪ 1/(|b|·c²).
        # Minimum of 500 covers the Bergomi branch (|b|·c² ≈ 9194, T ≈ 0.2)
        # and is fast enough for all maturities in exp_4.
        stiffness = abs(b) * c ** 2
        n_steps = max(500, int(np.ceil(stiffness * T / 5.0)))

    g0_func  = g0_func_builder(xi0_func, p_coeffs, b, c)
    Q_mat    = build_Q_matrix(a, c, M, cap=225.0).astype(np.complex128)
    pp_conv  = np.asarray(build_pp_conv(p_coeffs, M), dtype=np.complex128)

    delta    = T / n_steps
    k_arr    = np.arange(M + 1, dtype=float)
    exp_A    = np.exp(b * k_arr * delta)          # (M+1,) diagonal propagator
    G_diag   = compute_G_diagonal(b, M, delta).astype(np.complex128)  # (M+1,)

    # factor per u: g2 = -u_j / T  (Laplace setting)
    factor_u = (-u_arr / T).astype(np.complex128)  # (n_u,)

    # Pre-compute R-matrix sparsity pattern once (independent of psi)
    j_grid, k_grid = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    n_grid  = j_grid + 2 - k_grid            # index into psi for R_{j,k}
    valid   = (k_grid >= 1) & (n_grid >= 0) & (n_grid <= M)
    j_v     = np.where(valid)[0]             # row indices
    k_v     = np.where(valid)[1]             # column indices
    n_v     = n_grid[j_v, k_v]              # psi-index for each (j,k)
    coeff_jk = (k_v * n_v).astype(np.complex128)  # k*(j+2-k)

    eye_M   = np.eye(M + 1, dtype=np.complex128)
    Q_3d    = Q_mat[np.newaxis, :, :]        # (1, M+1, M+1) for broadcasting

    psi     = np.zeros((n_u, M + 1), dtype=np.complex128)

    for step in range(n_steps):
        t_i    = step * delta
        g0_val = float(g0_func(T - t_i))
        g0_sq  = g0_val ** 2

        # Source term P(t_i): shape (n_u, M+1)
        P_batch = np.outer(factor_u * g0_sq, pp_conv)

        # RHS = exp(A*delta)*psi + G*P   shape (n_u, M+1)
        rhs_batch = exp_A[np.newaxis, :] * psi + G_diag[np.newaxis, :] * P_batch

        # Vectorised R^(i) for all u: shape (n_u, M+1, M+1)
        # R_batch[u, j_v[l], k_v[l]] = coeff_jk[l] * psi[u, n_v[l]]
        R_batch = np.zeros((n_u, M + 1, M + 1), dtype=np.complex128)
        R_batch[:, j_v, k_v] = coeff_jk[np.newaxis, :] * psi[:, n_v]

        # J(u) = I - G_diag[:, None] * (c^2/2 * R(u) + Q)
        # G_mat is diagonal: G_mat @ X = G_diag[:, None] * X  (row scaling)
        gRQ   = c ** 2 / 2.0 * R_batch + Q_3d               # (n_u, M+1, M+1)
        J_batch = eye_M[np.newaxis] - G_diag[np.newaxis, :, np.newaxis] * gRQ

        # Batch solve: np.linalg.solve handles (n_u, M+1, M+1) input natively
        psi = np.linalg.solve(J_batch, rhs_batch)

    # By definition F_tilde(u) = E[exp(-u/T * int sigma^2)] is in [0,1] for u>=0.
    # Clip any floating-point overshoot from numerical error.
    return np.clip(np.real(np.exp(psi[:, 0])), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Volatility swap rate
# ---------------------------------------------------------------------------

def vol_swap_rate(T, M, a, b, c, p_coeffs, xi0_func, sigma_bs=None, n_quad=64):
    """Compute the volatility swap rate K_{1/2}(0,T).

    K_{1/2} = sigma_BS + (1/(2*sqrt(pi)))
              * integral_0^inf (F_tilde_BS(u) - F_tilde(u)) / u^{3/2} du

    The integral is regularised by the substitution v = sqrt(u):
        integral ... du = 2 * integral_0^inf (F_tilde_BS(v^2) - F_tilde(v^2)) / v^2 dv

    The v-grid avoids the singularity at v=0 (the integrand vanishes there
    when sigma_BS = sqrt(xi0_avg)).

    Parameters
    ----------
    T        : float – maturity
    M        : int   – Riccati truncation order
    a,b,c    : float – OU parameters
    p_coeffs : sequence
    xi0_func : callable t -> float
    sigma_bs : float or None – control variate vol; if None use sqrt(xi0_avg)
    n_quad   : int   – quadrature nodes

    Returns
    -------
    float – volatility swap rate
    """
    if sigma_bs is None:
        t_grid = np.linspace(0.0, T, 100)
        xi0_avg = float(np.mean([xi0_func(t) for t in t_grid]))
        sigma_bs = math.sqrt(max(xi0_avg, 1e-10))

    # Evaluate F_tilde on a uniform grid in v = sqrt(u)
    v_max = 15.0
    # Uniform grid excluding v=0 (the integrand is ~0 there)
    dv = v_max / n_quad
    v_grid = np.linspace(dv, v_max, n_quad)
    u_grid = v_grid ** 2

    F_BS = np.exp(-u_grid * sigma_bs ** 2)

    # Custom — batch solve: one shared time-loop for all u nodes
    F_model = laplace_transform_batch(u_grid, T, M, a, b, c, p_coeffs, xi0_func)

    # Integrand in v-space: 2*(F_BS(v^2) - F_model(v^2)) / v^2
    integrand = 2.0 * (F_BS - F_model) / v_grid ** 2

    # Prepend v=0 with integrand value 0
    v_full = np.concatenate([[0.0], v_grid])
    integ_full = np.concatenate([[0.0], integrand])

    integral = float(np.trapz(integ_full, v_full))
    rate = sigma_bs + integral / (2.0 * math.sqrt(math.pi))
    return float(rate)


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------

def vol_swap_term_structure(T_values, M, a, b, c, p_coeffs, xi0_func,
                            n_quad=64):
    """Compute vol swap rates for an array of maturities.

    Parameters
    ----------
    T_values : array-like of float
    M        : int
    a,b,c    : float
    p_coeffs : sequence
    xi0_func : callable
    n_quad   : int

    Returns
    -------
    np.ndarray of shape (len(T_values),) – vol swap rates
    """
    return np.array([
        vol_swap_rate(T, M, a, b, c, p_coeffs, xi0_func, n_quad=n_quad)
        for T in T_values
    ])


# ---------------------------------------------------------------------------
# Monte Carlo benchmark
# ---------------------------------------------------------------------------

def mc_vol_swap_rate(T, N_paths, N_steps, b, c, xi0_func,
                     vol_model='quintic_ou', p_coeffs=None, eta=None):
    """Monte Carlo estimate of the vol swap rate.

    Simulates X_t via exact OU transitions and integrates realized variance.

    Parameters
    ----------
    T        : float – maturity
    N_paths  : int   – number of paths
    N_steps  : int   – time steps
    b        : float – OU mean-reversion
    c        : float – OU diffusion
    xi0_func : callable t -> float
    vol_model: 'quintic_ou' or 'bergomi'
    p_coeffs : sequence (required for quintic_ou)
    eta      : float (required for bergomi)

    Returns
    -------
    (mean_vol_swap, lower_95ci, upper_95ci)
    """
    rng = np.random.default_rng(42)
    dt = T / N_steps
    t_grid = np.linspace(0.0, T, N_steps + 1)

    # OU exact transition parameters
    if b == 0.0:
        exp_b_dt = 1.0
        cond_std = c * math.sqrt(dt)
    else:
        exp_b_dt = math.exp(b * dt)
        cond_var = c ** 2 * (1.0 - math.exp(2.0 * b * dt)) / (-2.0 * b)
        cond_std = math.sqrt(max(cond_var, 0.0))

    # Pre-compute g0 and OU variance at each time step
    ou_var_t = np.array([float(ou_variance(t, b, c)) for t in t_grid])

    if vol_model == 'quintic_ou':
        if p_coeffs is None:
            raise ValueError("p_coeffs required for quintic_ou")
        p_arr = np.asarray(p_coeffs, dtype=float)
        ep2_t = np.array([poly_second_moment(p_arr, float(v)) for v in ou_var_t])
        xi0_t = np.array([float(xi0_func(t)) for t in t_grid])
        g0_t = np.sqrt(xi0_t / np.maximum(ep2_t, 1e-30))

        # Vectorised simulation
        X = np.zeros(N_paths)
        int_var = np.zeros(N_paths)
        for i in range(N_steps):
            Z = rng.standard_normal(N_paths)
            X = exp_b_dt * X + cond_std * Z
            sigma_sq = (g0_t[i + 1] * np.polyval(p_arr[::-1], X)) ** 2
            int_var += sigma_sq * dt

    elif vol_model == 'bergomi':
        if eta is None:
            raise ValueError("eta required for bergomi")
        from src.bergomi import bergomi_poly_coeffs
        p_arr = bergomi_poly_coeffs(eta, L=8)
        ep2_t = np.array([poly_second_moment(p_arr, float(v)) for v in ou_var_t])
        xi0_t = np.array([float(xi0_func(t)) for t in t_grid])
        X = np.zeros(N_paths)
        int_var = np.zeros(N_paths)
        for i in range(N_steps):
            Z = rng.standard_normal(N_paths)
            X = exp_b_dt * X + cond_std * Z
            EX2 = ou_var_t[i + 1]
            sigma_sq = xi0_t[i + 1] * np.exp(eta * X - 0.5 * eta ** 2 * EX2)
            int_var += sigma_sq * dt
    else:
        raise ValueError(f"Unknown vol_model: {vol_model!r}")

    realized_vol = np.sqrt(int_var / T)
    mean_vol = float(np.mean(realized_vol))
    se = float(np.std(realized_vol)) / math.sqrt(N_paths)
    return mean_vol, mean_vol - 1.96 * se, mean_vol + 1.96 * se
