"""
exp_1 benchmark validation script.

Validates the quasi-implicit Riccati solver against the exactly computable
expectation E[exp(-W_t^4/4!)] for a standard Brownian motion W_t.

Produces:
    results/exp1_level_plot.png  — solution curves vs exact
    results/exp1_error_plot.png  — absolute errors on log scale
    results/RESULTS.md           — appended summary of exp_1 results
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import (
    exact_benchmark,
    solve_benchmark_quasi_implicit,
    solve_benchmark_rk4,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T = 2.0
n_steps = 100
M_list = [10, 20, 40]
t_values = np.linspace(0.0, T, 200)

results_dir = Path(__file__).parent.parent / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Exact benchmark
# ---------------------------------------------------------------------------
print("Computing exact benchmark...")
exact = exact_benchmark(t_values)
print(f"  exact range: [{exact.min():.6f}, {exact.max():.6f}]")

# ---------------------------------------------------------------------------
# Quasi-implicit solver for each M
# ---------------------------------------------------------------------------
qi_results = {}
for M in M_list:
    print(f"Solving quasi-implicit M={M}...")
    t_grid, psi0_vals = solve_benchmark_quasi_implicit(M, T, n_steps)
    # Interpolate solver output (101 pts) onto the dense t_values (200 pts)
    qi_interp = np.interp(t_values, t_grid, psi0_vals)
    qi_results[M] = qi_interp
    print(f"  final value at T={T}: {qi_interp[-1]:.8f}  exact: {exact[-1]:.8f}")

# ---------------------------------------------------------------------------
# RK4 solver for M=20
# ---------------------------------------------------------------------------
M_rk4 = 20
print(f"Solving RK4 M={M_rk4}...")
t_grid_rk4, psi0_rk4 = solve_benchmark_rk4(M_rk4, T, n_steps)
rk4_interp = np.interp(t_values, t_grid_rk4, psi0_rk4)
rk4_has_nan = not np.all(np.isfinite(rk4_interp))
if rk4_has_nan:
    print(f"  WARNING: RK4 M={M_rk4} produced non-finite values (instability).")
else:
    print(f"  final value at T={T}: {rk4_interp[-1]:.8f}  exact: {exact[-1]:.8f}")

# ---------------------------------------------------------------------------
# Absolute errors (skip t=0 where both methods are exact by construction)
# ---------------------------------------------------------------------------
t_err = t_values[1:]
exact_err = exact[1:]

qi_errors = {}
for M in M_list:
    qi_errors[M] = np.abs(qi_results[M][1:] - exact_err)

rk4_error = np.abs(rk4_interp[1:] - exact_err)

# ---------------------------------------------------------------------------
# Level plot
# ---------------------------------------------------------------------------
colors_qi = {10: 'tab:blue', 20: 'tab:green', 40: 'tab:red'}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_values, exact, color='black', lw=2, label='Exact')
for M in M_list:
    ax.plot(
        t_values,
        qi_results[M],
        color=colors_qi[M],
        lw=1.5,
        linestyle='--',
        label=f'QI M={M}',
    )
if np.all(np.isfinite(rk4_interp)):
    ax.plot(
        t_values,
        rk4_interp,
        color='darkorange',
        lw=1.5,
        linestyle=':',
        label=f'RK4 M={M_rk4}',
    )
else:
    finite_mask = np.isfinite(rk4_interp)
    if finite_mask.any():
        ax.plot(
            t_values[finite_mask],
            rk4_interp[finite_mask],
            color='darkorange',
            lw=1.5,
            linestyle=':',
            label=f'RK4 M={M_rk4} (partial)',
        )
ax.set_xlabel('t')
ax.set_ylabel(r'$\mathbb{E}[\exp(-W_t^4/4!)]$')
ax.set_title('exp_1 Benchmark: Quasi-Implicit vs Exact')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(results_dir / 'exp1_level_plot.png', dpi=150)
plt.close(fig)
print("Saved results/exp1_level_plot.png")

# ---------------------------------------------------------------------------
# Error plot (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for M in M_list:
    err = qi_errors[M]
    ax.semilogy(t_err, err, color=colors_qi[M], lw=1.5, linestyle='--', label=f'QI M={M}')

rk4_err_finite = np.where(np.isfinite(rk4_error), rk4_error, np.nan)
ax.semilogy(t_err, rk4_err_finite, color='darkorange', lw=1.5, linestyle=':', label=f'RK4 M={M_rk4}')

ax.set_xlabel('t')
ax.set_ylabel('Absolute error')
ax.set_title('exp_1 Benchmark: Absolute Errors')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(results_dir / 'exp1_error_plot.png', dpi=150)
plt.close(fig)
print("Saved results/exp1_error_plot.png")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n=== exp_1 Summary Statistics ===")
print(f"{'Method':<20} {'MaxErr':>12} {'MeanErr':>12} {'FinalErr':>12}")
print("-" * 60)
for M in M_list:
    err = qi_errors[M]
    final_err = float(np.abs(qi_results[M][-1] - exact[-1]))
    print(f"{'QI M='+str(M):<20} {err.max():>12.4e} {err.mean():>12.4e} {final_err:>12.4e}")

rk4_err_clean = rk4_err_finite[np.isfinite(rk4_err_finite)]
rk4_max = float(np.nanmax(rk4_err_finite)) if len(rk4_err_clean) > 0 else float('nan')
rk4_mean = float(np.nanmean(rk4_err_finite)) if len(rk4_err_clean) > 0 else float('nan')
rk4_final = float(np.abs(rk4_interp[-1] - exact[-1])) if np.isfinite(rk4_interp[-1]) else float('nan')
print(f"{'RK4 M='+str(M_rk4):<20} {rk4_max:>12.4e} {rk4_mean:>12.4e} {rk4_final:>12.4e}")

# ---------------------------------------------------------------------------
# Save results/RESULTS.md
# ---------------------------------------------------------------------------
results_md = results_dir / 'RESULTS.md'
with open(results_md, 'a') as f:
    f.write("\n## exp_1 Benchmark Results\n\n")
    f.write(
        f"Terminal time T={T}, n_steps={n_steps}, M values={M_list}.\n"
        f"Exact reference: E[exp(-W_t^4/4!)] via numerical integration (scipy.integrate.quad).\n\n"
    )
    f.write(f"| Method       | Max |error| | Mean |error| | Final |error| |\n")
    f.write(f"|:-------------|------------:|-------------:|-------------:|\n")
    for M in M_list:
        err = qi_errors[M]
        fe = float(np.abs(qi_results[M][-1] - exact[-1]))
        f.write(f"| QI M={M:<5}   | {err.max():.4e}   | {err.mean():.4e}    | {fe:.4e}    |\n")
    f.write(
        f"| RK4 M={M_rk4:<5}  | {rk4_max:.4e}   | {rk4_mean:.4e}    | {rk4_final:.4e}    |\n\n"
    )
    if rk4_has_nan:
        f.write(
            f"Note: RK4 M={M_rk4} produced non-finite values — numerical instability "
            "expected for stiff systems with explicit integrators.\n\n"
        )
    f.write("Plots saved: `exp1_level_plot.png`, `exp1_error_plot.png`.\n")

print(f"\nResults appended to {results_md}")
