import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.riccati_solver import (
    tilde_psi_vec, truncated_conv, build_R_matrix, build_N_matrix,
    build_Q_matrix, compute_G_diagonal
)


# ---------------------------------------------------------------------------
# tilde_psi_vec
# ---------------------------------------------------------------------------

def test_tilde_psi_basic():
    """tilde_psi[k] = (k+1)*psi[k+1] for k < M; last element is 0."""
    psi = np.array([1.0, 2.0, 3.0, 4.0])  # M = 3
    tpsi = tilde_psi_vec(psi)

    assert tpsi[0] == pytest.approx(1 * psi[1])   # 1*2 = 2
    assert tpsi[1] == pytest.approx(2 * psi[2])   # 2*3 = 6
    assert tpsi[2] == pytest.approx(3 * psi[3])   # 3*4 = 12
    assert tpsi[3] == pytest.approx(0.0)           # last element always 0


def test_tilde_psi_zeros():
    """tilde_psi of the zero vector is the zero vector."""
    psi = np.zeros(5)
    tpsi = tilde_psi_vec(psi)
    np.testing.assert_array_equal(tpsi, np.zeros(5))


# ---------------------------------------------------------------------------
# truncated_conv
# ---------------------------------------------------------------------------

def test_truncated_conv_basic():
    """Verify truncated convolution coefficients up to degree M=2."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 0.0, 1.0])
    c = truncated_conv(a, b)

    # (a*b)_0 = 1*1 = 1
    assert c[0] == pytest.approx(1.0)
    # (a*b)_1 = 1*0 + 2*1 = 2
    assert c[1] == pytest.approx(2.0)
    # (a*b)_2 = 1*1 + 2*0 + 3*1 = 4
    assert c[2] == pytest.approx(4.0)


def test_truncated_conv_identity():
    """Convolving with the identity [1,0,...,0] returns the original vector."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal(5)
    a = np.zeros(5)
    a[0] = 1.0
    result = truncated_conv(a, b)
    np.testing.assert_allclose(result, b, rtol=1e-12)


# ---------------------------------------------------------------------------
# build_R_matrix
# ---------------------------------------------------------------------------

def test_build_R_matrix_shape():
    """R must be a square (M+1) x (M+1) matrix."""
    M = 5
    rng = np.random.default_rng(1)
    psi = rng.standard_normal(M + 1)
    R = build_R_matrix(psi, M)
    assert R.shape == (M + 1, M + 1)


def test_build_R_matrix_approximation():
    """R(psi_i) @ psi_next reproduces tilde_psi(psi_i) ★ tilde_psi(psi_next)."""
    M = 3
    psi_i = np.array([1.0, 0.5, 0.25, 0.125])
    psi_next = np.array([1.0, 0.3, 0.1, 0.05])

    R = build_R_matrix(psi_i, M)
    result = R @ psi_next

    tpsi_i = tilde_psi_vec(psi_i)
    tpsi_next = tilde_psi_vec(psi_next)
    expected = truncated_conv(tpsi_i, tpsi_next)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# build_Q_matrix
# ---------------------------------------------------------------------------

def test_build_Q_matrix_shape():
    """Q must be a square (M+1) x (M+1) matrix."""
    Q = build_Q_matrix(0, 1, 10)
    assert Q.shape == (11, 11)


def test_build_Q_matrix_cap():
    """Q[k, k+2] = 0.5 * min((k+1)*(k+2), cap); verify below-cap and at-cap."""
    cap = 225.0
    Q = build_Q_matrix(0, 1, 20, cap=cap)

    # k=0: 0.5 * min(1*2, 225) = 1.0  (well below cap)
    assert Q[0, 2] == pytest.approx(0.5 * min(1 * 2, cap))

    # k=13: 0.5 * min(14*15, 225) = 0.5*210 = 105  (below cap)
    assert Q[13, 15] == pytest.approx(0.5 * min(14 * 15, cap))

    # k=14: 0.5 * min(15*16, 225) = 0.5*225 = 112.5  (capped, 15*16=240 > 225)
    assert Q[14, 16] == pytest.approx(0.5 * min(15 * 16, cap))


# ---------------------------------------------------------------------------
# compute_G_diagonal
# ---------------------------------------------------------------------------

def test_compute_G_diagonal_b_zero():
    """When b=0 every diagonal entry equals delta (step size)."""
    delta = 0.1
    M = 5
    G = compute_G_diagonal(0, M, delta)
    assert len(G) == M + 1
    np.testing.assert_allclose(G, delta, rtol=1e-10)


def test_compute_G_diagonal_b_nonzero():
    """Spot-check G entries for b=-1, delta=0.1."""
    b = -1.0
    delta = 0.1
    M = 3
    G = compute_G_diagonal(b, M, delta)

    # k=0: G[0] == delta
    assert G[0] == pytest.approx(delta, rel=1e-10)

    # k=1: G[1] = (exp(b*delta) - 1) / b
    expected_g1 = (np.exp(b * delta) - 1.0) / b
    assert G[1] == pytest.approx(expected_g1, rel=1e-10)


# ---------------------------------------------------------------------------
# build_N_matrix
# ---------------------------------------------------------------------------

def test_build_N_matrix_shape():
    """N must be a square (M+1) x (M+1) matrix."""
    p = np.array([1.0, 2.0, 3.0])
    M = 5
    N = build_N_matrix(p, M)
    assert N.shape == (M + 1, M + 1)


def test_build_N_matrix_values():
    """N[j, k] = k * p[j+1-k] for valid indices; spot-check two entries."""
    p = np.array([1.0, 2.0, 3.0])
    M = 5
    N = build_N_matrix(p, M)

    # j=1, k=1: N[1,1] = 1 * p[1+1-1] = 1 * p[1] = 2
    assert N[1, 1] == pytest.approx(1 * p[1])

    # j=2, k=2: N[2,2] = 2 * p[2+1-2] = 2 * p[1] = 4
    assert N[2, 2] == pytest.approx(2 * p[1])
