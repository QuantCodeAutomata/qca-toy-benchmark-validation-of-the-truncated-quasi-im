"""
Quintic OU model for option pricing (exp_2).

Implements the polynomial stochastic-volatility model
    sigma_t = g0(t) * p(X_t),  p(x) = p0 + p1*x + p3*x^3 + p5*x^5
where X_t is an Ornstein-Uhlenbeck process.

Paper defaults: rho=-0.65, alpha=-0.6, epsilon=1/52,
p0=0.01, p1=1, p3=0.214, p5=0.227, xi0=0.025.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import roots_laguerre
from scipy.stats import norm

from src.riccati_solver import (
    build_Q_matrix, build_N_matrix, truncated_conv,
    solve_riccati_adaptive, riccati_quasi_implicit_step, compute_G_diagonal,
)

# ---------------------------------------------------------------------------
# OU parameter helpers
# ---------------------------------------------------------------------------

def ou_params_from_alpha_epsilon(alpha, epsilon):
    """Derive OU parameters from the rough-vol parameterisation.

    X_t = epsilon^alpha * integral_0^t exp(alpha/epsilon*(t-s)) dW_s

    Parameters
    ----------
    alpha   : float – Hurst-like exponent (negative for rough vol)
    epsilon : float – time scale

    Returns
    -------
    (a, b, c) : OU drift constant, mean-reversion rate, diffusion coefficient
    """
    a = 0.0
    b = alpha / epsilon
    c = epsilon ** alpha
    return a, b, c


def ou_variance(t, b, c):
    """Var[X_t] for the OU process dX = b*X dt + c*dW started at X_0=0.

    Parameters
    ----------
    t : float or np.ndarray
    b : float – mean-reversion (negative for stationary)
    c : float – diffusion coefficient

    Returns
    -------
    float or np.ndarray  (same shape as t)
    """
    t = np.asarray(t, dtype=float)
    if b == 0.0:
        return c ** 2 * t
    result = c ** 2 * (np.exp(2.0 * b * t) - 1.0) / (2.0 * b)
    return np.maximum(result, 0.0)


# ---------------------------------------------------------------------------
# Gaussian moments and polynomial expectation
# ---------------------------------------------------------------------------

def gaussian_moment(n, sigma2):
    """E[Z^n] for Z ~ N(0, sigma2).

    Returns 0 for odd n; (n-1)!! * sigma2^(n/2) for even n.
    """
    if n % 2 == 1:
        return 0.0
    k = n // 2
    # (n-1)!! = 1*3*5*...*(n-1)
    double_fact = 1
    for i in range(1, n, 2):
        double_fact *= i
    return double_fact * float(sigma2) ** k


def poly_second_moment(p_coeffs, sigma2):
    """E[p(X)^2] where X ~ N(0, sigma2) and p(x) = sum p_k x^k.

    Expands p(x)^2 and applies Gaussian moments.
    """
    p = list(p_coeffs)
    total = 0.0
    for j, pj in enumerate(p):
        for k, pk in enumerate(p):
            total += pj * pk * gaussian_moment(j + k, sigma2)
    return total


# ---------------------------------------------------------------------------
# Normalisation factor g0
# ---------------------------------------------------------------------------

def g0_func_builder(xi0_func, p_coeffs, ou_b, ou_c):
    """Build the normalisation function g0(t).

    g0(t) = sqrt( xi0(t) / E[p(X_t)^2] )

    where E[p(X_t)^2] uses ou_variance(t, ou_b, ou_c).

    Parameters
    ----------
    xi0_func : callable t -> float   (forward variance at calendar time t)
    p_coeffs : sequence of floats
    ou_b     : float – OU mean-reversion
    ou_c     : float – OU diffusion

    Returns
    -------
    callable t -> float
    """
    def g0(t):
        sigma2 = float(ou_variance(max(float(t), 0.0), ou_b, ou_c))
        ep2 = poly_second_moment(p_coeffs, sigma2)
        return math.sqrt(xi0_func(t) / max(ep2, 1e-30))
    return g0


# ---------------------------------------------------------------------------
# Polynomial convolution helper
# ---------------------------------------------------------------------------

def build_pp_conv(p_coeffs, M):
    """Compute (p*p)_k for k=0,...,M (truncated self-convolution of p).

    Parameters
    ----------
    p_coeffs : sequence of floats – polynomial coefficients
    M        : int – truncation degree

    Returns
    -------
    np.ndarray of shape (M+1,)
    """
    p_arr = np.zeros(M + 1)
    Lp = min(len(p_coeffs), M + 1)
    p_arr[:Lp] = np.asarray(p_coeffs[:Lp], dtype=float)
    return np.convolve(p_arr, p_arr)[: M + 1]


# ---------------------------------------------------------------------------
# Batch CF computation (vectorized over u-values, fast)
# ---------------------------------------------------------------------------

def _n_steps_for_T(T: float, base: int = 200) -> int:
    """Return a fixed step count that scales sensibly with T."""
    return max(base, int(np.ceil(T * 52 * 2)))   # ≥ base, ~2 steps/day


def char_func_batch(u_arr: np.ndarray, T: float, M: int,
                    a: float, b: float, c: float, rho: float,
                    p_coeffs, xi0_func, n_steps: int = None) -> np.ndarray:
    """Compute CF phi(u-0.5j) for all u in u_arr simultaneously (fast batch).

    Solves the Riccati system once per time step for ALL u-values jointly,
    sharing Q, N, G_i, exp_A_delta and g0 evaluations across the batch.
    Each u gets its own psi vector (shape n_u × (M+1)), but the time-loop is
    shared.  J_i(u) is assembled and LU-solved independently for each u.

    Parameters
    ----------
    u_arr    : 1-D real np.ndarray – Gauss-Laguerre quadrature nodes
    T        : float – maturity
    M        : int   – truncation degree
    a,b,c    : float – OU parameters
    rho      : float – correlation
    p_coeffs : sequence – polynomial coefficients
    xi0_func : callable t -> float
    n_steps  : int or None – fixed step count (None → auto)

    Returns
    -------
    phi_arr : complex np.ndarray of shape (n_u,)
        CF values phi(u_j - 0.5j) at the final time T.
    """
    from scipy.linalg import lu_factor, lu_solve

    u_arr = np.asarray(u_arr, dtype=float)
    n_u = len(u_arr)
    if n_steps is None:
        n_steps = _n_steps_for_T(T)

    Q_mat    = build_Q_matrix(a, c, M, cap=225.0)
    N_mat    = build_N_matrix(p_coeffs, M)
    pp_conv  = np.asarray(build_pp_conv(p_coeffs, M), dtype=np.complex128)
    g0_func  = g0_func_builder(xi0_func, p_coeffs, b, c)

    # Shifted argument u - 0.5j for the Lewis formula
    xi_arr   = u_arr - 0.5j   # complex, shape (n_u,)

    delta    = T / n_steps
    k_arr    = np.arange(M + 1, dtype=float)

    # Propagator and G-diagonal: identical for all u (depends only on b)
    exp_A    = np.exp(b * k_arr * delta)                         # (M+1,)
    G_diag   = compute_G_diagonal(b, M, delta)                   # (M+1,)
    G_mat    = np.diag(G_diag.astype(np.complex128))             # (M+1, M+1)

    # Precompute u-dependent scalars at t_i (updated inside loop)
    # ell_u = c * rho * g1_u * g0_val  where g1_u = i*xi_arr
    # factor_u = g2 + g1_u*(g1_u-1)/2 = i*xi*(i*xi-1)/2 = -(xi^2+i*xi)/2
    factor_u = (1j * xi_arr) * (1j * xi_arr - 1) / 2           # (n_u,) complex

    # Batch psi: shape (n_u, M+1)
    psi      = np.zeros((n_u, M + 1), dtype=np.complex128)

    c2h      = c ** 2 / 2.0

    # Pre-compute R-matrix sparsity pattern (independent of psi and u)
    j_grid, k_grid = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    n_grid  = j_grid + 2 - k_grid
    valid   = (k_grid >= 1) & (n_grid >= 0) & (n_grid <= M)
    j_v     = np.where(valid)[0]
    k_v     = np.where(valid)[1]
    n_v     = n_grid[j_v, k_v]
    coeff_jk = (k_v * n_v).astype(np.complex128)  # k*(j+2-k)

    eye_M   = np.eye(M + 1, dtype=np.complex128)
    Q_3d    = Q_mat.astype(np.complex128)[np.newaxis, :, :]
    N_3d    = N_mat.astype(np.complex128)[np.newaxis, :, :]

    G_diag_c = G_diag.astype(np.complex128)

    for step in range(n_steps):
        t_i   = step * delta
        g0_val = float(g0_func(T - t_i))

        # Precompute ell for each u: shape (n_u,)
        g1_arr  = 1j * xi_arr
        ell_arr = c * rho * g1_arr * g0_val    # (n_u,) complex

        # Source term P(t_i): shape (n_u, M+1)
        P_batch = np.outer(factor_u * g0_val**2, pp_conv)

        # RHS: shape (n_u, M+1)
        rhs_batch = exp_A[np.newaxis, :] * psi + G_diag_c[np.newaxis, :] * P_batch

        # Vectorised R_batch: shape (n_u, M+1, M+1)
        R_batch = np.zeros((n_u, M + 1, M + 1), dtype=np.complex128)
        R_batch[:, j_v, k_v] = coeff_jk[np.newaxis, :] * psi[:, n_v]

        # J(u) = I - G_diag[:, None] * (c2h*R(u) + ell(u)*N + Q)
        # Shape: (n_u, M+1, M+1)
        inner = c2h * R_batch + ell_arr[:, np.newaxis, np.newaxis] * N_3d + Q_3d
        J_batch = eye_M[np.newaxis] - G_diag_c[np.newaxis, :, np.newaxis] * inner

        psi = np.linalg.solve(J_batch, rhs_batch)

    return np.exp(psi[:, 0])


# ---------------------------------------------------------------------------
# Characteristic function via Riccati (single u, for compatibility)
# ---------------------------------------------------------------------------

def char_func_riccati(u, T, M, a, b, c, rho, p_coeffs, xi0_func,
                      n_steps: int = None):
    """Characteristic function phi(u) = E[exp(i*u*log(S_T/S_0))] via Riccati.

    Single-u wrapper around char_func_batch.  The argument u is taken as the
    shifted quantity (i.e. u - 0.5j has already been applied by the caller if
    needed; here u is passed directly as the CF argument g1 = i*u).

    Parameters
    ----------
    u       : float or complex – CF argument
    T       : float – maturity
    M       : int   – polynomial truncation order
    a, b, c : float – OU parameters
    rho     : float – correlation
    p_coeffs: sequence – polynomial coefficients
    xi0_func: callable t -> float (forward variance)
    n_steps : int or None – fixed step count (None → auto)

    Returns
    -------
    complex – characteristic function value phi(u)
    """
    from scipy.linalg import lu_factor, lu_solve

    Q_mat   = build_Q_matrix(a, c, M, cap=225.0)
    N_mat   = build_N_matrix(p_coeffs, M)
    pp_conv = np.asarray(build_pp_conv(p_coeffs, M), dtype=np.complex128)
    g0_func = g0_func_builder(xi0_func, p_coeffs, b, c)

    if n_steps is None:
        n_steps = _n_steps_for_T(T)

    delta   = T / n_steps
    k_arr   = np.arange(M + 1, dtype=float)
    exp_A   = np.exp(b * k_arr * delta)
    G_diag  = compute_G_diagonal(b, M, delta)
    G_mat   = np.diag(G_diag.astype(np.complex128))

    g1      = complex(u)   # may already be shifted, e.g. u_real - 0.5j
    g2      = 0.0 + 0j
    factor  = g2 + g1 * (g1 - 1) / 2
    c2h     = c ** 2 / 2.0

    psi = np.zeros(M + 1, dtype=np.complex128)

    for step in range(n_steps):
        t_i    = step * delta
        g0_val = float(g0_func(T - t_i))
        ell    = c * rho * g1 * g0_val
        P      = factor * g0_val**2 * pp_conv

        from src.riccati_solver import build_R_matrix
        R      = build_R_matrix(psi, M)
        linear_op = c2h * R + ell * N_mat + Q_mat
        J      = np.eye(M + 1, dtype=np.complex128) - G_mat @ linear_op

        rhs    = exp_A * psi + G_diag * P
        lu_piv = lu_factor(J)
        psi    = lu_solve(lu_piv, rhs)

    return complex(np.exp(psi[0]))


# ---------------------------------------------------------------------------
# Heston characteristic function (Gatheral form)
# ---------------------------------------------------------------------------

def heston_char_func(u, T, v0, kappa, theta, sigma_h, rho_h):
    """Heston characteristic function for log(S_T/S_0) (Gatheral form).

    Parameters
    ----------
    u       : float or complex
    T       : float – maturity
    v0      : float – initial variance
    kappa   : float – mean reversion speed
    theta   : float – long-term variance
    sigma_h : float – vol-of-vol
    rho_h   : float – correlation

    Returns
    -------
    complex
    """
    u = complex(u)
    xi = kappa - rho_h * sigma_h * 1j * u
    d = np.sqrt(xi ** 2 + sigma_h ** 2 * (u ** 2 + 1j * u))

    # Handle branch cut: choose d such that Re(d) >= 0
    if d.real < 0:
        d = -d

    denom_g = xi + d
    if abs(denom_g) < 1e-14:
        g = 0.0 + 0j
    else:
        g = (xi - d) / denom_g

    exp_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_dT
    one_minus_g = 1.0 - g

    if abs(one_minus_g) < 1e-14 or abs(one_minus_g_exp) < 1e-14:
        return 1.0 + 0j

    log_ratio = np.log(one_minus_g_exp / one_minus_g)
    C = (kappa * theta / sigma_h ** 2) * ((xi - d) * T - 2.0 * log_ratio)
    D = ((xi - d) / sigma_h ** 2) * (1.0 - exp_dT) / one_minus_g_exp

    return complex(np.exp(C + D * v0))


def heston_char_func_vec(u_arr: np.ndarray, T: float,
                         v0: float, kappa: float, theta: float,
                         sigma_h: float, rho_h: float) -> np.ndarray:
    """Vectorised Heston CF for an array of (complex) arguments.

    Uses full numpy broadcasting; avoids a Python loop over nodes.
    Branch-cut is handled element-wise: force Re(d) >= 0.
    """
    u_arr = np.asarray(u_arr, dtype=np.complex128)
    xi    = kappa - rho_h * sigma_h * 1j * u_arr
    d     = np.sqrt(xi ** 2 + sigma_h ** 2 * (u_arr ** 2 + 1j * u_arr))
    d     = np.where(d.real < 0, -d, d)

    denom_g = xi + d
    safe    = np.abs(denom_g) > 1e-14
    g       = np.where(safe, (xi - d) / np.where(safe, denom_g, 1.0), 0.0 + 0j)

    exp_dT        = np.exp(-d * T)
    one_minus_g   = 1.0 - g
    one_minus_gexp = 1.0 - g * exp_dT

    # Fallback to 1 where degenerate
    ok = (np.abs(one_minus_g) > 1e-14) & (np.abs(one_minus_gexp) > 1e-14)
    log_ratio = np.where(ok, np.log(np.where(ok, one_minus_gexp / np.where(ok, one_minus_g, 1.0), 1.0)), 0.0)

    C = (kappa * theta / sigma_h ** 2) * ((xi - d) * T - 2.0 * log_ratio)
    D = ((xi - d) / sigma_h ** 2) * (1.0 - exp_dT) / np.where(ok, one_minus_gexp, 1.0)

    result = np.exp(C + D * v0)
    return np.where(ok, result, 1.0 + 0j)


# ---------------------------------------------------------------------------
# Heston control variate fitting
# ---------------------------------------------------------------------------

def fit_heston_control_variate(u_nodes, T, M, a, b, c, rho, p_coeffs,
                               xi0_func, xi0_val):
    """Fit Heston parameters to the quintic OU characteristic function.

    Minimises sum |log(phi_model(u)) - log(phi_heston(u))|^2 over
    Heston parameters (kappa, theta, sigma_h, rho_h) with v0 = theta.

    Parameters
    ----------
    u_nodes  : array-like of real floats (integration nodes for fitting)
    T        : float
    M        : int
    a,b,c    : float – OU parameters
    rho      : float
    p_coeffs : sequence
    xi0_func : callable
    xi0_val  : float – constant forward variance (used for v0 = theta init)

    Returns
    -------
    dict with keys: v0, kappa, theta, sigma_h, rho_h
    """
    u_nodes = np.asarray(u_nodes, dtype=float)

    # Compute model CF at fitting nodes (shifted u - 0.5j, as used in Lewis)
    # char_func_batch internally applies the Lewis shift xi = u - 0.5j
    model_vals = char_func_batch(u_nodes, T, M, a, b, c, rho,
                                  p_coeffs, xi0_func)

    log_model = np.log(model_vals + 1e-30)

    u_complex = u_nodes - 0.5j  # Lewis shift, shape (n_u,)

    def objective(x):
        kappa, theta, sigma_h, rho_h = x
        kappa   = max(kappa,   0.01)
        theta   = max(theta,   1e-5)
        sigma_h = max(sigma_h, 0.01)
        rho_h   = float(np.clip(rho_h, -0.999, 0.999))
        try:
            log_h = np.log(
                heston_char_func_vec(u_complex, T, theta, kappa, theta, sigma_h, rho_h)
                + 1e-30
            )
            return float(np.sum(np.abs(log_model - log_h) ** 2))
        except Exception:
            return 1e8

    x0 = [2.0, xi0_val, 0.3, rho * 0.8]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-6})
    kappa, theta, sigma_h, rho_h = result.x
    kappa = max(kappa, 0.01)
    theta = max(theta, 1e-5)
    sigma_h = max(sigma_h, 0.01)
    rho_h = float(np.clip(rho_h, -0.999, 0.999))

    return {'v0': theta, 'kappa': kappa, 'theta': theta,
            'sigma_h': sigma_h, 'rho_h': rho_h}


# ---------------------------------------------------------------------------
# Lewis pricing with Heston control variate
# ---------------------------------------------------------------------------

def lewis_price_with_cv(S0, K, T, M, a, b, c, rho, p_coeffs, xi0_func,
                        heston_params, n_quad=64):
    """Price a European call via the Lewis formula with Heston control variate.

    C = C_Heston - (K/pi) * integral_0^inf
            Re(exp((iu+0.5)*k) * (phi(u-0.5j) - phi_H(u-0.5j))) / (u^2+0.25) du

    Uses ``char_func_batch`` for a single vectorised Riccati solve across all
    Gauss-Laguerre nodes (instead of one solve per node).

    Parameters
    ----------
    S0, K        : float – spot and strike
    T            : float – maturity
    M            : int   – Riccati truncation order
    a,b,c        : float – OU parameters
    rho          : float
    p_coeffs     : sequence
    xi0_func     : callable
    heston_params: dict with keys v0, kappa, theta, sigma_h, rho_h
    n_quad       : int – Gauss-Laguerre quadrature nodes

    Returns
    -------
    float – call price
    """
    prices = lewis_price_batch_strikes(
        S0, np.array([K]), T, M, a, b, c, rho, p_coeffs, xi0_func,
        heston_params, n_quad=n_quad,
    )
    return float(prices[0])


def lewis_price_batch_strikes(S0, K_arr, T, M, a, b, c, rho, p_coeffs, xi0_func,
                               heston_params, n_quad=64):
    """Price European calls for multiple strikes via a single batched Riccati solve.

    The characteristic function is computed once for all Gauss-Laguerre nodes
    using ``char_func_batch``; the Lewis integral is then evaluated for each
    strike via vectorised numpy operations.

    Parameters
    ----------
    S0           : float
    K_arr        : np.ndarray – array of strikes
    T, M         : float, int
    a,b,c,rho    : float – OU + correlation parameters
    p_coeffs     : sequence
    xi0_func     : callable
    heston_params: dict
    n_quad       : int

    Returns
    -------
    prices : np.ndarray – call prices, same length as K_arr
    """
    K_arr = np.asarray(K_arr, dtype=float)
    k_arr = np.log(K_arr / S0)                 # log-moneyness per strike

    v0      = heston_params['v0']
    kappa   = heston_params['kappa']
    theta   = heston_params['theta']
    sigma_h = heston_params['sigma_h']
    rho_h   = heston_params['rho_h']

    gl_nodes, gl_weights = roots_laguerre(n_quad)

    # GL weights adjusted for exp(u) (standard GL convention)
    w_adj = gl_weights * np.exp(gl_nodes)      # (n_quad,)

    # --- Single batch Riccati solve for all GL nodes ---
    phi_m = char_func_batch(gl_nodes, T, M, a, b, c, rho, p_coeffs, xi0_func)

    # --- Heston CF at same nodes (vectorised) ---
    xi_arr    = gl_nodes - 0.5j                # (n_quad,) complex
    phi_h_arr = heston_char_func_vec(xi_arr, T, v0, kappa, theta, sigma_h, rho_h)

    denom   = gl_nodes ** 2 + 0.25             # (n_quad,)
    diff_cf = phi_m - phi_h_arr                # (n_quad,) complex

    # Lewis integral for each strike: vectorised over strikes and nodes
    # phase[i, j] = exp((i*u_j + 0.5) * k_i)  shape (n_K, n_quad)
    phase = np.exp(np.outer(k_arr, 0.5 + 0j) +
                   np.outer(k_arr, 1j) * gl_nodes[np.newaxis, :])

    # CV integrand: Re(phase * diff_cf) / denom   shape (n_K, n_quad)
    cv_integrand = np.real(phase * diff_cf[np.newaxis, :]) / denom[np.newaxis, :]
    heston_integrand = np.real(
        phase * phi_h_arr[np.newaxis, :]
    ) / denom[np.newaxis, :]

    # Gauss-Laguerre sums
    C_Heston = S0 - K_arr / math.pi * (heston_integrand @ w_adj)
    C_model  = C_Heston - K_arr / math.pi * (cv_integrand @ w_adj)

    # Clamp to intrinsic
    intrinsic = np.maximum(S0 - K_arr, 0.0)
    return np.maximum(C_model, intrinsic)


# ---------------------------------------------------------------------------
# Black-Scholes implied volatility
# ---------------------------------------------------------------------------

def bs_call_price(S0, K, T, sigma, r=0.0):
    """Black-Scholes call price (zero dividend)."""
    if sigma <= 0 or T <= 0:
        return max(S0 * math.exp(-0.0) - K * math.exp(-r * T), 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(C, S0, K, T, r=0.0, option_type='c'):
    """Black-Scholes implied volatility via Brent's method.

    Parameters
    ----------
    C           : float – option price
    S0, K       : float – spot and strike
    T           : float – time to expiry
    r           : float – risk-free rate (default 0)
    option_type : str   – 'c' for call (put not yet implemented)

    Returns
    -------
    float – implied volatility, or np.nan if root not found
    """
    intrinsic = max(S0 * math.exp(-0.0) - K * math.exp(-r * T), 0.0)
    if C <= intrinsic + 1e-10:
        return np.nan

    def f(sigma):
        return bs_call_price(S0, K, T, sigma, r) - C

    try:
        return brentq(f, 1e-6, 10.0, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan
