# Truncated Quasi-Implicit Riccati Solver for Polynomial OU Volatility Models

A Python implementation of the truncated quasi-implicit numerical solver for Riccati-type ODEs arising in quantitative finance, specifically for polynomial Ornstein–Uhlenbeck (OU) stochastic volatility models. The solver enables efficient pricing of European options, volatility swaps, and VIX futures under a quintic-OU variance process.

---

## Description

In stochastic volatility models where the variance factor follows a polynomial OU process, the moment-generating function (MGF) / characteristic function of the log-price satisfies a **Riccati ODE system** of the form

$$\dot{\psi}_k(t) = \sum_{j} A_{kj} \psi_j(t) + \sum_{j,l} B_{kjl} \psi_j(t)\,\psi_l(t), \quad k = 0, 1, \ldots, M,$$

where $M$ is the **truncation order** and $\psi = (\psi_0, \ldots, \psi_M)$ stores the polynomial coefficients of the exponent.

The **quasi-implicit** integrator treats the linear (drift/diffusion) part implicitly and the nonlinear (quadratic convolution) part explicitly, which substantially improves stability over explicit methods while avoiding the cost of a fully implicit Newton iteration at each step.

---

## Mathematical Overview

### Polynomial OU Variance Model

The variance factor $X_t$ follows an OU SDE:

$$dX_t = (a + b\,X_t)\,dt + c\,dW_t, \qquad X_0 = x_0,$$

with mean-reversion speed $-b > 0$, long-run mean $-a/b$, and diffusion $c$.  
The instantaneous variance of the log-price is $\sigma^2(X_t)$, where $\sigma$ is a polynomial (up to degree 5 in the quintic case).

### Characteristic Function via Riccati System

Define the extended affine ansatz

$$\mathbb{E}\!\left[e^{u\,\log S_T}\,\big|\,X_0 = x_0\right] = \exp\!\left(\sum_{k=0}^{M} \psi_k(T)\,x_0^k\right).$$

Applying Itô's formula and matching polynomial powers yields the truncated Riccati system solved by this library. Three key operator matrices are constructed by `src/riccati_solver.py`:

| Matrix | Role |
|--------|------|
| $Q$ | Quadratic variation: $Q_{k,k+2} = \tfrac{1}{2}(k+1)(k+2)$ |
| $N$ | Drift linearisation: $N_{j,k} = k\,p_{j+1-k}$ |
| $R(\psi)$ | Nonlinear convolution: encodes $\tilde{\psi} \star \tilde{\psi}$ |

where $\tilde{\psi}_k = (k+1)\psi_{k+1}$ (the "derivative-shifted" vector).

### Quasi-Implicit Step

For step size $\delta = T/n$ the update rule is

$$\left(I - \delta\,L\right)\psi^{(n+1)} = \psi^{(n)} + \delta\,N\!\left(\psi^{(n)}\right),$$

where $L$ is the linear part (involving $Q$, $N$) and $N(\psi^{(n)})$ is the explicit nonlinear contribution from $R(\psi^{(n)})$.

---

## Experiments

| ID | Script | Description |
|----|--------|-------------|
| **exp_1** | `exp/run_exp1.py` | Scalar Riccati benchmark: compares quasi-implicit, RK4, and exact closed-form solutions across truncation orders $M$ |
| **exp_2** | `exp/run_exp2.py` | Quintic-OU implied volatility surface: prices European calls and extracts the BS implied vol smile |
| **exp_3** | `exp/run_exp3.py` | Convergence study: error vs. $M$ and step-size $n$ across multiple maturities |
| **exp_4** | `exp/run_exp4.py` | Volatility swap fair rate: computes $\mathbb{E}[\sqrt{\langle \log S \rangle_T}]$ via Laplace inversion |
| **exp_5** | `exp/run_exp5.py` | Model calibration to market implied-vol data: fits $(b, c, \epsilon)$ by minimising the smile RMSE |

---

## Installation

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

Python 3.10+ is recommended. All dependencies are pure-Python or have pre-built wheels.

---

## Usage

Run each experiment from the repository root:

```bash
python exp/run_exp1.py   # benchmark: quasi-implicit vs RK4 vs exact
python exp/run_exp2.py   # implied volatility smile (quintic OU)
python exp/run_exp3.py   # convergence tables
python exp/run_exp4.py   # volatility swap rate
python exp/run_exp5.py   # calibration to market data
```

Figures and result tables are written to `results/`.

---

## Running Tests

```bash
pytest tests/
```

Or with verbose output:

```bash
pytest -v tests/
```

---

## Repository Structure

```
project/
├── src/
│   ├── riccati_solver.py   # Core: tilde_psi, truncated_conv, Q/N/R/G matrices,
│   │                       #        quasi-implicit and RK4 ODE integrators
│   ├── benchmark.py        # exp_1: exact closed-form + solver wrappers
│   ├── quintic_ou.py       # exp_2: OU params, Gaussian moments, BS implied vol,
│   │                       #        polynomial second moment, pp-convolution
│   └── vol_swap.py         # exp_4: Laplace transform via Riccati, vol-swap rate
├── exp/
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   ├── run_exp4.py
│   └── run_exp5.py
├── tests/
│   ├── __init__.py
│   ├── test_riccati_solver.py
│   ├── test_benchmark.py
│   ├── test_quintic_ou.py
│   └── test_vol_swap.py
├── data/                   # Market data (not tracked in git)
├── results/                # Generated figures and tables
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## Key Parameters

| Symbol | Description | Typical value |
|--------|-------------|---------------|
| $M$ | Polynomial truncation order | 10 – 40 |
| $b$ | OU mean-reversion speed ($b < 0$) | $\approx -31$ |
| $c$ | OU diffusion coefficient | $\approx 7$ |
| $a$ | OU drift constant | 0 |
| $\delta$ | Time step $T/n$ | $T/100$ |
| `cap` | Cap on $Q_{k,k+2}$ entries | 225.0 |
