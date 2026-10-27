import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vol_swap import laplace_transform_riccati, vol_swap_rate

# Shared simple parameters for tests
_SIMPLE_PARAMS = dict(
    a=0,
    b=-5,
    c=7,
    p_coeffs=[1],
    xi0_func=lambda t: 0.025,
)


# ---------------------------------------------------------------------------
# laplace_transform_riccati
# ---------------------------------------------------------------------------

def test_laplace_transform_bounds():
    """For real u > 0 the magnitude of the Laplace transform must not exceed 1."""
    F = laplace_transform_riccati(u_val=1.0, T=1.0, M=10, **_SIMPLE_PARAMS)
    assert 0 <= abs(F) <= 1 + 1e-6


def test_laplace_at_u_zero():
    """Laplace transform at u≈0 must approach 1 (total probability mass)."""
    F = laplace_transform_riccati(u_val=1e-6, T=1.0, M=10, **_SIMPLE_PARAMS)
    assert abs(F - 1.0) < 0.01


# ---------------------------------------------------------------------------
# vol_swap_rate
# ---------------------------------------------------------------------------

def test_vol_swap_positive():
    """Vol swap rate must be strictly positive."""
    rate = vol_swap_rate(T=1.0, M=10, **_SIMPLE_PARAMS)
    assert rate > 0


def test_vol_swap_reasonable_range():
    """With xi0=0.025 the rate should lie in (0.05, 0.5); naive ref: sqrt(0.025)≈0.158."""
    rate = vol_swap_rate(T=1.0, M=10, **_SIMPLE_PARAMS)
    assert 0.05 < rate < 0.5, (
        f"Vol swap rate {rate:.4f} outside expected range (0.05, 0.5)"
    )
