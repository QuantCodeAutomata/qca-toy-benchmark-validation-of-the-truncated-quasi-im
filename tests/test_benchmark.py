import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import exact_benchmark, solve_benchmark_quasi_implicit, solve_benchmark_rk4


# ---------------------------------------------------------------------------
# exact_benchmark
# ---------------------------------------------------------------------------

def test_exact_benchmark_at_zero():
    """Exact solution at t=0 must equal 1 (initial condition)."""
    val = exact_benchmark([0.0])[0]
    assert abs(val - 1.0) < 1e-6


def test_exact_benchmark_monotone_decreasing():
    """Exact solution is monotonically non-increasing and stays in (0, 1]."""
    t_values = [0.1, 0.5, 1.0, 2.0]
    vals = exact_benchmark(t_values)

    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1], (
            f"Not non-increasing at i={i}: vals[{i}]={vals[i]}, vals[{i+1}]={vals[i+1]}"
        )
    assert all(0 < v <= 1 for v in vals)


def test_exact_benchmark_bounded():
    """Exact solution satisfies 0 <= val <= 1 for all t in [0, 3]."""
    t_values = np.linspace(0, 3, 10)
    vals = exact_benchmark(t_values)
    assert np.all(vals >= 0), "Some exact benchmark values are negative"
    assert np.all(vals <= 1), "Some exact benchmark values exceed 1"


# ---------------------------------------------------------------------------
# solve_benchmark_quasi_implicit
# ---------------------------------------------------------------------------

def test_quasi_implicit_at_zero():
    """First returned value (t=0 IC) must be exp(psi_0(0)) = exp(0) = 1.0."""
    t_arr, phi_arr = solve_benchmark_quasi_implicit(M=10, T=0.01, n_steps=5)
    assert abs(phi_arr[0] - 1.0) < 1e-10


def test_quasi_implicit_convergence():
    """Approximation error at T=1 must decrease monotonically with M."""
    exact_val = exact_benchmark([1.0])[0]

    errors = {}
    for M in [10, 20, 40]:
        _, phi = solve_benchmark_quasi_implicit(M=M, T=1.0, n_steps=100)
        errors[M] = abs(phi[-1] - exact_val)

    assert errors[40] < errors[10], (
        f"Error did not decrease: M=10 err={errors[10]:.6f}, M=40 err={errors[40]:.6f}"
    )


def test_quasi_implicit_stability():
    """All returned values for T=2 must be finite and within a reasonable range."""
    _, psi0_values = solve_benchmark_quasi_implicit(M=40, T=2.0, n_steps=100)

    assert np.all(np.isfinite(psi0_values)), "NaN or Inf in quasi-implicit output"
    assert np.all(psi0_values >= 0), "Quasi-implicit output contains negative values"
    assert np.all(psi0_values <= 2.0), "Quasi-implicit output exceeds expected upper bound"


def test_rk4_runs():
    """RK4 solver completes without error and returns real-valued output."""
    t_arr, phi_arr = solve_benchmark_rk4(M=10, T=1.0, n_steps=100)
    assert np.all(np.isreal(phi_arr)), "RK4 output contains complex values"
    assert np.all(np.isfinite(phi_arr)), "RK4 output contains NaN or Inf"


def test_quasi_vs_exact_small_T():
    """For small T=0.1 the quasi-implicit result must be within 1% of exact."""
    T = 0.1
    M = 20
    n_steps = 100

    _, phi = solve_benchmark_quasi_implicit(M=M, T=T, n_steps=n_steps)
    exact_val = exact_benchmark([T])[0]

    relative_error = abs(phi[-1] - exact_val) / exact_val
    assert relative_error < 0.01, (
        f"Relative error {relative_error:.4%} exceeds 1% tolerance"
    )
