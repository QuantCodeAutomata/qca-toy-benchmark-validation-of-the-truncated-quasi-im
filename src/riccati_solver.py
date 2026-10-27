"""
Truncated quasi-implicit Riccati ODE solver.

Implements the variation-of-constants quasi-implicit scheme for the system:

    psi_k'(t) = P_k(t) + b*k*psi_k + a*(k+1)*psi_{k+1}
                + (c^2/2)*(k+2)*(k+1)*psi_{k+2}
                + (c^2/2)*(tilde_psi * tilde_psi)_k
                + c*rho*g1*g0*(p*tilde_psi)_k

where tilde_psi_k = (k+1)*psi_{k+1}.

The system is truncated at degree M and solved forward in t from 0 to T.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def tilde_psi_vec(psi):
    """Compute tilde_psi_k = (k+1)*psi_{k+1} for k=0,...,M.

    The last element is zero because psi_{M+1} is outside the truncation.

    Parameters
    ----------
    psi : np.ndarray of shape (M+1,)
        Coefficient vector.

    Returns
    -------
    np.ndarray of shape (M+1,), same dtype as psi.
        tilde_psi vector.
    """
    M = len(psi) - 1
    result = np.zeros(M + 1, dtype=psi.dtype)
    k = np.arange(1, M + 1, dtype=float)
    result[:-1] = k * psi[1:]
    return result


def truncated_conv(a, b):
    """Truncated convolution: (a*b)_k = sum_{l=0}^k a_l * b_{k-l}, k=0,...,M.

    Parameters
    ----------
    a, b : np.ndarray of shape (M+1,)
        Input vectors.

    Returns
    -------
    np.ndarray of shape (M+1,)
        First M+1 elements of the full convolution.
    """
    return np.convolve(a, b)[: len(a)]


def build_R_matrix(psi_i, M):
    """Build the linearisation matrix R^(i) of shape (M+1, M+1).

    R^(i)_{j,k} = k * (j+2-k) * psi_i[j+2-k]

    for j+2-k in [0, M] and k >= 1; zero otherwise.

    This linearises (tilde_psi(t_i) * tilde_psi(t_{i+1}))_j as (R^(i) @ psi_{i+1})_j,
    enabling the quasi-implicit step.

    Parameters
    ----------
    psi_i : np.ndarray of shape (M+1,)
        Coefficient vector at the current time step.
    M : int
        Truncation degree.

    Returns
    -------
    np.ndarray of shape (M+1, M+1), dtype complex128.
    """
    R = np.zeros((M + 1, M + 1), dtype=np.complex128)
    # Use meshgrid to get full (M+1, M+1) index arrays (avoids shape-mismatch
    # when applying a 2-D boolean mask to a (1, M+1) or (M+1, 1) view).
    j_grid, k_grid = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    n = j_grid + 2 - k_grid  # index into psi_i
    valid = (k_grid >= 1) & (n >= 0) & (n <= M)
    j_v, k_v = np.nonzero(valid)
    n_v = n[j_v, k_v]
    R[j_v, k_v] = k_v * n_v * psi_i[n_v]
    return R


def build_N_matrix(p_coeffs, M):
    """Build the linearisation matrix N of shape (M+1, M+1).

    N_{j,k} = k * p_{j+1-k}

    where p_{j+1-k} is taken from p_coeffs (zero-padded for out-of-range indices).

    This linearises (p * tilde_psi(t_{i+1}))_j as (N @ psi_{i+1})_j.

    Parameters
    ----------
    p_coeffs : array-like
        Coefficients of p(x): p_coeffs[l] is the coefficient of x^l.
    M : int
        Truncation degree.

    Returns
    -------
    np.ndarray of shape (M+1, M+1), dtype complex128.
    """
    p = np.asarray(p_coeffs, dtype=np.complex128)
    L = len(p)
    N = np.zeros((M + 1, M + 1), dtype=np.complex128)
    j_grid, k_grid = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    n = j_grid + 1 - k_grid  # index into p_coeffs
    valid = (k_grid >= 1) & (n >= 0) & (n < L)
    j_v, k_v = np.nonzero(valid)
    n_v = n[j_v, k_v]
    N[j_v, k_v] = k_v * p[n_v]
    return N


def build_Q_matrix(a, c, M, cap=225.0):
    """Build the linear off-diagonal part Q of shape (M+1, M+1).

    Q_{k,k+1} = a*(k+1)              for k=0,...,M-1
    Q_{k,k+2} = (c^2/2)*min((k+1)*(k+2), cap)  for k=0,...,M-2

    The cap (default 15^2=225) prevents runaway large entries in the implicit
    system matrix for high truncation degrees.

    Parameters
    ----------
    a : float or complex
        Drift coefficient.
    c : float or complex
        Diffusion coefficient.
    M : int
        Truncation degree.
    cap : float, optional
        Maximum value for the super-super-diagonal entries.

    Returns
    -------
    np.ndarray of shape (M+1, M+1), dtype complex128.
    """
    Q = np.zeros((M + 1, M + 1), dtype=np.complex128)
    if M >= 1:
        k = np.arange(M)
        Q[k, k + 1] = a * (k + 1)
    if M >= 2:
        k = np.arange(M - 1)
        Q[k, k + 2] = (c ** 2 / 2) * np.minimum((k + 1) * (k + 2), cap)
    return Q


def compute_G_diagonal(b, M, delta):
    """Compute the diagonal of the variation-of-constants kernel G_i.

    G_{k} = integral_0^{delta} exp(b*k*s) ds

    For k=0 or b=0: G_k = delta.
    Otherwise:        G_k = (exp(b*k*delta) - 1) / (b*k).

    Uses np.expm1 for numerical stability when b*k*delta is small.

    Parameters
    ----------
    b : float or complex
        Diagonal drift coefficient.
    M : int
        Truncation degree.
    delta : float
        Step size.

    Returns
    -------
    np.ndarray of shape (M+1,), dtype complex128 if b is complex else float64.
    """
    dtype = np.complex128 if np.iscomplex(b) else np.float64
    G = np.empty(M + 1, dtype=dtype)
    k = np.arange(M + 1, dtype=np.float64)
    bk = b * k
    zero_mask = bk == 0
    G[zero_mask] = delta
    nz = ~zero_mask
    G[nz] = np.expm1(bk[nz] * delta) / bk[nz]
    return G


def ode_rhs(psi, t, params):
    """Compute the full ODE right-hand side at (psi, t).

    psi_k' = P_k(t) + b*k*psi_k + a*(k+1)*psi_{k+1}
             + (c^2/2)*(k+2)*(k+1)*psi_{k+2}
             + (c^2/2)*(tilde_psi * tilde_psi)_k
             + c*rho*g1*g0(T-t)*(p * tilde_psi)_k

    where P_k(t) = (g2 + g1*(g1-1)/2) * g0(T-t)^2 * (p*p)_k
    and   tilde_psi_k = (k+1)*psi_{k+1}.

    Parameters
    ----------
    psi : np.ndarray of shape (M+1,)
        Current coefficient vector.
    t : float
        Current time (integration runs from 0 to T).
    params : dict
        Keys: a, b, c, rho, M, p_coeffs, T, g0_func, g1, g2, xi0_func.

    Returns
    -------
    np.ndarray of shape (M+1,), dtype complex128.
    """
    a = params['a']
    b = params['b']
    c = params['c']
    rho = params['rho']
    M = params['M']
    p_coeffs = params['p_coeffs']
    T = params['T']
    g0_func = params['g0_func']
    g1 = params['g1']
    g2 = params['g2']

    psi = np.asarray(psi, dtype=np.complex128)
    g0_val = g0_func(T - t)
    c2h = c ** 2 / 2
    k_arr = np.arange(M + 1, dtype=np.float64)

    rhs = b * k_arr * psi
    if M >= 1:
        rhs[:-1] += a * (k_arr[:-1] + 1) * psi[1:]
    if M >= 2:
        rhs[:-2] += c2h * (k_arr[:-2] + 2) * (k_arr[:-2] + 1) * psi[2:]

    tp = tilde_psi_vec(psi)
    rhs += c2h * truncated_conv(tp, tp)

    p_arr = np.zeros(M + 1, dtype=np.complex128)
    p_arr[: min(len(p_coeffs), M + 1)] = p_coeffs[: min(len(p_coeffs), M + 1)]
    pp_conv = truncated_conv(p_arr, p_arr)

    ell_t = c * rho * g1 * g0_val
    if ell_t != 0:
        rhs += ell_t * truncated_conv(p_arr, tp)

    factor = g2 + g1 * (g1 - 1) / 2
    rhs += factor * g0_val ** 2 * pp_conv

    return rhs


def riccati_quasi_implicit_step(psi_i, t_i, delta, params, Q, N, pp_conv):
    """Perform one quasi-implicit variation-of-constants step.

    Solves for psi_{i+1} such that:

        J_i @ psi_{i+1} = exp(A*delta)*psi_i + G_i*P(t_i)

    where:
        A_k = b*k  (diagonal)
        G_i diagonal = integral_0^{delta} exp(A_k*s) ds
        J_i = I - G_i_mat * (c^2/2 * R^(i) + ell*N + Q)
        P_k(t_i) = (g2 + g1*(g1-1)/2) * g0(T-t_i)^2 * pp_conv_k

    Parameters
    ----------
    psi_i : np.ndarray of shape (M+1,)
        Coefficient vector at t_i.
    t_i : float
        Current time.
    delta : float
        Step size.
    params : dict
        Keys: a, b, c, rho, M, g0_func, T, g1, g2.
    Q : np.ndarray of shape (M+1, M+1)
        Precomputed linear off-diagonal matrix (from build_Q_matrix).
    N : np.ndarray of shape (M+1, M+1)
        Precomputed p*tilde_psi linearisation matrix (from build_N_matrix).
    pp_conv : np.ndarray of shape (M+1,)
        Precomputed (p*p) convolution coefficients.

    Returns
    -------
    np.ndarray of shape (M+1,), dtype complex128.
        psi at t_i + delta.
    """
    b = params['b']
    c = params['c']
    rho = params['rho']
    M = params['M']
    g0_func = params['g0_func']
    T = params['T']
    g1 = params['g1']
    g2 = params['g2']

    psi_i = np.asarray(psi_i, dtype=np.complex128)
    k_arr = np.arange(M + 1, dtype=np.float64)
    g0_val = g0_func(T - t_i)

    # Diagonal propagator exp(b*k*delta)
    exp_A_delta = np.exp(b * k_arr * delta)

    # Variation-of-constants kernel diagonal
    G_i = compute_G_diagonal(b, M, delta)

    # Source term P_k(t_i)
    factor = g2 + g1 * (g1 - 1) / 2
    P = np.asarray(factor * g0_val ** 2 * pp_conv, dtype=np.complex128)

    # Linearisation of quadratic term
    R = build_R_matrix(psi_i, M)
    ell = c * rho * g1 * g0_val

    # System matrix J_i = I - diag(G_i) @ (c^2/2 * R + ell*N + Q)
    G_mat = np.diag(G_i.astype(np.complex128))
    linear_op = (c ** 2 / 2) * R + ell * N + Q
    J = np.eye(M + 1, dtype=np.complex128) - G_mat @ linear_op

    # Right-hand side
    rhs = exp_A_delta * psi_i + G_i * P

    lu_piv = lu_factor(J)
    psi_next = lu_solve(lu_piv, rhs)
    return psi_next


def estimate_psi_double_prime(psi_curr, psi_prev, delta):
    """Estimate max |psi''| from two consecutive ODE right-hand side evaluations.

    Uses the finite-difference approximation:

        |psi''| ≈ max(|rhs_curr - rhs_prev|) / delta

    where psi_curr and psi_prev are successive ODE RHS vectors separated by
    the step size delta.

    Parameters
    ----------
    psi_curr : np.ndarray
        ODE right-hand side at the current time point.
    psi_prev : np.ndarray
        ODE right-hand side at the previous time point.
    delta : float
        Step size between the two evaluations.

    Returns
    -------
    float
        Maximum absolute component of the estimated second derivative.
    """
    diff = np.asarray(psi_curr) - np.asarray(psi_prev)
    return float(np.max(np.abs(diff))) / delta


def solve_riccati_adaptive(
    psi0, T, params, Q, N, pp_conv,
    zeta=1e-9, delta_min=5e-5, delta_max=None, store_all=False
):
    """Solve the Riccati system with adaptive step size control.

    Uses the quasi-implicit variation-of-constants scheme with a step size
    chosen to keep the local truncation error below the tolerance zeta.

    The adaptive rule is:
        delta_new = clip(sqrt(2*zeta / max(|psi''|, 1e-30)), delta_min, delta_max)

    where |psi''| is estimated from consecutive ODE RHS evaluations.

    Parameters
    ----------
    psi0 : np.ndarray of shape (M+1,)
        Initial condition at t=0.
    T : float
        Terminal time.
    params : dict
        Solver parameters (a, b, c, rho, M, p_coeffs, T, g0_func, g1, g2, xi0_func).
    Q : np.ndarray of shape (M+1, M+1)
        Precomputed linear off-diagonal matrix.
    N : np.ndarray of shape (M+1, M+1)
        Precomputed p*tilde_psi linearisation matrix.
    pp_conv : np.ndarray of shape (M+1,)
        Precomputed (p*p) convolution coefficients.
    zeta : float, optional
        Local truncation error tolerance. Default 1e-9.
    delta_min : float, optional
        Minimum step size. Default 5e-5.
    delta_max : float or None, optional
        Maximum step size. If None, set to max((1/252)*0.1, T*0.1).
    store_all : bool, optional
        If True, store psi at every step; otherwise store only the final value.

    Returns
    -------
    t_grid : np.ndarray of shape (n_steps+1,) or (2,)
        Time points at which psi was stored.
    psi_history : np.ndarray of shape (n_steps+1, M+1) or (2, M+1)
        Stored psi values.
    """
    if delta_max is None:
        delta_max = max((1 / 252) * 0.1, T * 0.1)

    psi_curr = np.asarray(psi0, dtype=np.complex128).copy()
    t = 0.0
    delta = delta_max

    t_list = [0.0]
    psi_list = [psi_curr.copy()]

    rhs_prev = None

    while t < T - 1e-14:
        rhs_curr = ode_rhs(psi_curr, t, params)

        if rhs_prev is not None:
            psi_dp = estimate_psi_double_prime(rhs_curr, rhs_prev, delta)
            delta = min(
                max(np.sqrt(2 * zeta / max(psi_dp, 1e-30)), delta_min),
                delta_max,
            )

        delta = min(delta, T - t)
        psi_next = riccati_quasi_implicit_step(psi_curr, t, delta, params, Q, N, pp_conv)

        t += delta
        rhs_prev = rhs_curr
        psi_curr = psi_next

        if store_all:
            t_list.append(t)
            psi_list.append(psi_curr.copy())

    if not store_all:
        t_list.append(t)
        psi_list.append(psi_curr.copy())

    return np.array(t_list), np.array(psi_list)


def solve_riccati_uniform(psi0, T, n_steps, params, Q, N, pp_conv):
    """Solve the Riccati system with a uniform step size.

    Uses n_steps quasi-implicit steps of equal size delta = T/n_steps,
    storing psi at every step.

    Parameters
    ----------
    psi0 : np.ndarray of shape (M+1,)
        Initial condition at t=0.
    T : float
        Terminal time.
    n_steps : int
        Number of uniform steps.
    params : dict
        Solver parameters (a, b, c, rho, M, p_coeffs, T, g0_func, g1, g2, xi0_func).
    Q : np.ndarray of shape (M+1, M+1)
        Precomputed linear off-diagonal matrix.
    N : np.ndarray of shape (M+1, M+1)
        Precomputed p*tilde_psi linearisation matrix.
    pp_conv : np.ndarray of shape (M+1,)
        Precomputed (p*p) convolution coefficients.

    Returns
    -------
    t_grid : np.ndarray of shape (n_steps+1,)
        Uniform time grid from 0 to T.
    psi_history : np.ndarray of shape (n_steps+1, M+1)
        psi at each grid point.
    """
    delta = T / n_steps
    psi_curr = np.asarray(psi0, dtype=np.complex128).copy()
    t_grid = np.linspace(0.0, T, n_steps + 1)
    M = params['M']
    psi_history = np.zeros((n_steps + 1, M + 1), dtype=np.complex128)
    psi_history[0] = psi_curr

    for i in range(n_steps):
        psi_curr = riccati_quasi_implicit_step(
            psi_curr, t_grid[i], delta, params, Q, N, pp_conv
        )
        psi_history[i + 1] = psi_curr

    return t_grid, psi_history
