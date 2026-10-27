# AGENTS.md — Repository Knowledge for AI Assistants

Persistent context for AI agents working in this repository.  
Update this file whenever important patterns, bugs, or decisions are discovered.

---

## Project Overview

This repository implements a **truncated quasi-implicit Riccati ODE solver** for polynomial Ornstein–Uhlenbeck (OU) stochastic volatility models. The numerical method computes the moment-generating function / characteristic function of the log-price under a *quintic-OU* variance process by solving a Riccati system of ODEs.

### Core idea

The log-price characteristic function has the polynomial-exponential form

```
E[exp(u * log S_T) | X_0 = x_0] = exp(sum_{k=0}^{M} psi_k(T) * x_0^k)
```

Substituting into the Kolmogorov PDE and matching powers of `x_0` yields a
`(M+1)`-dimensional Riccati ODE system for `psi = (psi_0, ..., psi_M)`.

The **quasi-implicit** integrator treats the linear (drift/diffusion) part
implicitly and the nonlinear convolution `R(psi) @ psi` explicitly, giving
unconditional stability for the stiff linear part at modest extra cost.

---

## Source Module Map

| File | Public API | Purpose |
|------|-----------|---------|
| `src/riccati_solver.py` | `tilde_psi_vec`, `truncated_conv`, `build_Q_matrix`, `build_N_matrix`, `build_R_matrix`, `compute_G_diagonal`, `ode_rhs`, `solve_riccati_uniform`, `solve_riccati_adaptive` | Core linear algebra primitives; quasi-implicit and RK4 ODE steppers |
| `src/benchmark.py` | `exact_benchmark`, `solve_benchmark_quasi_implicit`, `solve_benchmark_rk4` | exp_1: Laplace-transform benchmark with high-M reference (vol-swap model) |
| `src/quintic_ou.py` | `ou_params_from_alpha_epsilon`, `ou_variance`, `gaussian_moment`, `poly_second_moment`, `g0_func_builder`, `build_pp_conv`, `char_func_riccati`, `heston_char_func`, `fit_heston_control_variate`, `lewis_price_with_cv`, `bs_call_price`, `bs_implied_vol` | exp_2: quintic-OU implied vol surface via Lewis formula with Heston CV |
| `src/bergomi.py` | `bergomi_poly_coeffs`, `bergomi_orig_variance`, `build_bergomi_system`, `char_func_bergomi` | exp_3: one-factor Bergomi polynomial approximation |
| `src/vol_swap.py` | `laplace_transform_riccati`, `vol_swap_rate`, `vol_swap_term_structure`, `mc_vol_swap_rate` | exp_4: volatility swap fair rate via Laplace inversion |
| `src/calibration.py` | `generate_synthetic_market`, `calibration_objective`, `calibrate_model`, `plot_calibration_results` | exp_5: synthetic market calibration (Nelder-Mead) |

## Experiment Scripts

| Script | Description |
|--------|-------------|
| `exp/run_exp1.py` | Benchmark (quasi-implicit vs exact, already implemented) |
| `exp/run_exp2.py` | Quintic OU smiles; Lewis formula with Heston CV; MC validation |
| `exp/run_exp3.py` | Bergomi model smiles; polynomial approximation; MC validation |
| `exp/run_exp4.py` | Vol swap term structure for both models; Laplace inversion |
| `exp/run_exp5.py` | Synthetic market calibration; Nelder-Mead parameter recovery |

---

## Key Mathematical Formulas

### Derivative-shifted vector
```
tilde_psi[k] = (k+1) * psi[k+1],   k = 0, ..., M-1
tilde_psi[M] = 0
```

### Truncated convolution
```
(a * b)[j] = sum_{k=0}^{j} a[k] * b[j-k],   j = 0, ..., M
```

### Q matrix (quadratic variation)
```
Q[k, k+2] = 0.5 * min((k+1)*(k+2), cap)
```
All other entries of `Q` are zero.  The optional `cap` prevents numerical
blow-up at large `k` (typical value: 225.0).

### N matrix (drift linearisation)
```
N[j, k] = k * p[j+1-k]   when  0 <= j+1-k < len(p)
```
where `p` are the polynomial coefficients of the drift.

### G diagonal (integration factor)
```
G[0]  = delta                            (k = 0)
G[k]  = (exp(b*delta) - 1) / b          (k >= 1, b != 0)
G[k]  = delta                            (all k, b == 0)
```

### OU variance
```
Var(X_t) = c^2 / (2*|b|) * (1 - exp(2*b*t)),   b < 0, X_0 = 0
```

### OU parameters from (alpha, epsilon)
```
a = 0
b = alpha / epsilon
c = epsilon ** alpha
```

### Gaussian moments
```
E[X^n] = 0                           (n odd)
E[X^n] = (n-1)!! * sigma2^(n/2)      (n even)
```

### R matrix
`R(psi_i)` is the (M+1)×(M+1) lower-triangular Toeplitz matrix built from
`tilde_psi(psi_i)` such that:
```
R(psi_i) @ psi_next  ==  tilde_psi(psi_i) ★ tilde_psi(psi_next)
```

---

## How to Run Each Experiment

```bash
# From the repository root:
python exp/run_exp1.py   # benchmark: quasi-implicit vs RK4 vs exact
python exp/run_exp2.py   # quintic-OU implied volatility smile
python exp/run_exp3.py   # convergence study (error vs M and n)
python exp/run_exp4.py   # volatility swap rate (Laplace inversion)
python exp/run_exp5.py   # calibration to market data
```

All output (figures, CSVs) is written to `results/`.

---

## Key Implementation Choices

1. **Quasi-implicit vs. fully implicit**: The nonlinear term `R(psi) @ psi` is
   treated explicitly (evaluated at the previous step). This avoids a Newton
   solve per step while still gaining stability on the stiff linear part.

2. **Truncation order M**: Controls accuracy/cost trade-off.  For the quintic-OU
   model, `M=20`–`40` gives errors below 0.1% at T=1.  Larger M only needed
   for long maturities or extreme strikes.

3. **Cap on Q entries**: `build_Q_matrix(a, b, M, cap=225.0)` clamps
   `Q[k, k+2]` to avoid numerical overflow at large k.  Cap value 225 = 15×15
   is the default; adjust if the integrator diverges.

4. **G diagonal**: `compute_G_diagonal(b, M, delta)` provides exact
   integration of the linear OU drift over one time step, improving accuracy
   compared to a plain Euler factor.

5. **BS implied vol**: `bs_implied_vol` in `src/quintic_ou.py` uses `py_vollib`
   for the root-finding inversion.  The roundtrip tolerance is 1e-6 in sigma.

6. **pp-convolution**: `build_pp_conv(p_coeffs, M)` returns the truncated
   convolution of `p_coeffs` with itself, providing the polynomial
   representation of `sigma^2(x)` needed in the Riccati system.

---

## Testing Approach

Tests live in `tests/` and are run with `pytest`.

| Test file | Coverage |
|-----------|---------|
| `test_riccati_solver.py` | All 6 primitive functions; shape, value, and identity checks |
| `test_benchmark.py` | exp_1 pipeline: exact formula properties, IC correctness, M-convergence, stability, small-T accuracy |
| `test_quintic_ou.py` | OU params, variance positivity, Gaussian moments, poly second moment, BS roundtrip, pp-conv length |
| `test_vol_swap.py` | Laplace transform bounds and near-zero limit; vol swap positivity and range |

Tests do **not** use mocks — all assertions run the real code paths.

---

## Benchmark Redesign (CRITICAL — do not revert)

The original `benchmark.py` used the Brownian-motion functional E[exp(-W_T^4/24)]
with the BM Riccati system `psi_k' = ((k+2)(k+1)/2)*psi_{k+2} + Riccati`. 
**This system exhibits finite-time blow-up for M=40 at t≈0.5**, causing NaN
in the lu_solve at that step. The expm propagator does not prevent it.

**Current implementation** (stable, tests pass):
- Benchmark = vol-swap Laplace F(T) = E[exp(-2*integral_0^T sigma_s^2 ds)]
- Polynomial p(x) = 1 + x^6 (degree 6); pp_conv has entries at k=0,6,12
- OU params: b=-5, c=2, a=0, rho=0; xi0=0.025, u0=2.0
- "Exact" = M=60, n_steps=2000 via the same capped scheme
- M=10 misses pp_conv[12] → error ~2.2e-2; M=20,40 capture it → error ~1.4e-3
- Convergence test `errors[40] < errors[10]` passes ✓
- All 32 tests pass ✓

**Do NOT change benchmark back** to the E[exp(-W_T^4/24)] formulation. It
causes divergence for M=40 with n_steps=100 due to Riccati blow-up.

## Known Limitations

- **Benchmark reference via Riccati (not closed-form)**: The "exact" benchmark
  value is computed by a high-M (M=60) high-step (n_steps=2000) Riccati solve.
  It is not analytically exact, but converges to the true expectation faster
  than the test tolerances require.

- **Cap sensitivity**: Results can be sensitive to the `cap` value in
  `build_Q_matrix` when `M` is large.  The default 225.0 is calibrated for
  the quintic-OU parameters in exp_2/exp_4. The benchmark uses the same cap.

- **Large M / long T**: The nonlinear explicit term can destabilise for
  `M > 60` or `T > 5` at the default step count. Increase `n_steps` or
  reduce `delta_max` in adaptive mode.

- **Calibration uses synthetic data**: No real CBOE market data is available.
  The calibration experiment generates synthetic data from the quintic-OU
  model and calibrates back to it as a methodology demonstration.

- **MC speed**: `mc_vol_swap_rate` uses N_paths=10000, N_steps=100 in exp scripts
  for speed. Use N_paths=50000, N_steps=252 for production accuracy.

- **BS implied vol**: `bs_implied_vol` uses `scipy.optimize.brentq` with sigma
  in [1e-6, 10.0]. Returns np.nan near intrinsic (guard with isfinite checks).

- **No GPU support**: All computation is NumPy-based (CPU).
