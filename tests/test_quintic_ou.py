import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.stats import norm
from src.quintic_ou import (
    ou_params_from_alpha_epsilon, ou_variance, gaussian_moment,
    poly_second_moment, build_pp_conv, bs_implied_vol
)


# ---------------------------------------------------------------------------
# ou_params_from_alpha_epsilon
# ---------------------------------------------------------------------------

def test_ou_params():
    """OU parameters derived from (alpha, epsilon): a=0, b=alpha/epsilon, c=epsilon**alpha."""
    alpha = -0.6
    epsilon = 1 / 52
    a, b, c = ou_params_from_alpha_epsilon(alpha, epsilon)

    assert a == 0
    assert abs(b - alpha / epsilon) < 1e-10
    assert abs(c - epsilon ** alpha) < 1e-10


# ---------------------------------------------------------------------------
# ou_variance
# ---------------------------------------------------------------------------

def test_ou_variance_positive():
    """Variance of the OU process must be strictly positive for t > 0."""
    b = -31.2
    c = 7.16
    v = ou_variance(0.5, b, c)
    assert v > 0


# ---------------------------------------------------------------------------
# gaussian_moment
# ---------------------------------------------------------------------------

def test_gaussian_moment_odd_zero():
    """Odd moments of a centred Gaussian are exactly zero."""
    sigma2 = 2.5
    for n in [1, 3, 5, 7]:
        assert gaussian_moment(n, sigma2) == pytest.approx(0.0), (
            f"gaussian_moment({n}, {sigma2}) should be 0"
        )


def test_gaussian_moment_even():
    """Even moments follow E[X^{2k}] = (2k-1)!! * sigma2^k."""
    sigma2 = 1.5

    assert gaussian_moment(0, sigma2) == pytest.approx(1.0)
    assert gaussian_moment(2, sigma2) == pytest.approx(sigma2)
    assert gaussian_moment(4, sigma2) == pytest.approx(3 * sigma2 ** 2)
    assert gaussian_moment(6, sigma2) == pytest.approx(15 * sigma2 ** 3)


# ---------------------------------------------------------------------------
# poly_second_moment
# ---------------------------------------------------------------------------

def test_poly_second_moment_constant():
    """For constant p(x)=2, E[p(X)^2] = E[4] = 4."""
    result = poly_second_moment([2.0], 1.0)
    assert result == pytest.approx(4.0)


def test_poly_second_moment_linear():
    """For p(x)=x, E[p(X)^2] = E[X^2] = sigma2."""
    sigma2 = 3.7
    result = poly_second_moment([0.0, 1.0], sigma2)
    assert result == pytest.approx(sigma2)


# ---------------------------------------------------------------------------
# bs_implied_vol
# ---------------------------------------------------------------------------

def test_bs_implied_vol_roundtrip():
    """Implied vol recovered from a BS price must reproduce the input sigma."""
    S0, K, T, r, sigma = 1.0, 1.0, 1.0, 0.0, 0.2

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    recovered = bs_implied_vol(price, S0, K, T, r)
    assert abs(recovered - sigma) < 1e-6


# ---------------------------------------------------------------------------
# build_pp_conv
# ---------------------------------------------------------------------------

def test_build_pp_conv_length():
    """Output vector must have exactly M+1 elements."""
    p_coeffs = [1, 0, 0, 0, 0]  # p(x) = 1  (degree-4 polynomial with only constant term)
    M = 10
    pp = build_pp_conv(p_coeffs, M)

    assert len(pp) == M + 1
    # Constant term of p(x)^2 equals p[0]^2 = 1
    assert pp[0] == pytest.approx(1.0)
