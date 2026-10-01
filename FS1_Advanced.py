#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS1_Advanced.py
================================================================================
Advanced Fourier fitting methods — comparison harness for FS1.

Provides three solvers for the same trigonometric-basis least-squares problem:

    1. Normal equations (chunked, low memory, reference)
    2. QR least squares  (numerically stable, needs full design matrix)
    3. Bayesian Ridge    (posterior mean + posterior σ per coefficient)

Usage
-----
    from FS1_Advanced import compare_all, fit_qr, fit_bayesian

Output includes per-coefficient significance flags (|A| > k·σ) so that
noise-only harmonics can be identified and pruned.

--------------------------------------------------------------------------------
BAYESIAN RIDGE — NAMING CONVENTION
--------------------------------------------------------------------------------
The generative model is

    x  =  Φ·θ + ε,       ε  ~ N(0, 1/α_noise)          (observation noise)
    θ  ~ N(0, 1/λ_weight · I)                          (coefficient prior)

where

    α_noise   = noise precision   = 1 / σ²_noise
    λ_weight  = weight precision  = 1 / σ²_weights

scikit-learn stores these under unintuitive attribute names:

    model.alpha_   →   α_noise   (noise precision)
    model.lambda_  →   λ_weight  (weight precision)

This module renames them on output to avoid confusion.

Author : MJS · Jolotundo Observatory
Version: 1.1.0
================================================================================
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import qr as scipy_qr
from sklearn.linear_model import BayesianRidge

import FS1_FourierSeries as FS1
from FS1_FourierSeries import (
    OMEGA_A, OMEGA_D, CROSS_TERMS,
    N_ANNUAL, N_DIURNAL,
    _n_params,
    extract_annual, extract_diurnal, extract_cross,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1  FULL DESIGN MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_full_design(t: np.ndarray,
                      n_a: int = N_ANNUAL,
                      n_d: int = N_DIURNAL,
                      cx: List[Tuple[int, int]] = CROSS_TERMS
                      ) -> np.ndarray:
    """
    Construct the full trig design matrix Φ of shape (N, P).

    Memory requirement: 8·N·P bytes.  For N = 277,800 and P = 119,
    approximately 264 MB.  Caller must ensure this fits in RAM.

    Column layout matches FS1._build_chunk exactly:

        col 0               :  1  (DC term)
        col 1 + 2(k-1)      :  cos(k·ωA·t)         k = 1..n_a
        col 1 + 2(k-1) + 1  :  sin(k·ωA·t)
        col ...             :  cos(m·ωD·t), sin(m·ωD·t)     m = 1..n_d
        col ...             :  cos(Ω·t), sin(Ω·t)           Ω = k·ωA + m·ωD
    """
    P = _n_params(n_a, n_d, cx)
    N = len(t)
    Phi = np.empty((N, P), dtype=np.float64)

    Phi[:, 0] = 1.0
    col = 1

    for k in range(1, n_a + 1):
        arg = k * OMEGA_A * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for m in range(1, n_d + 1):
        arg = m * OMEGA_D * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for (k, m) in cx:
        arg = (k * OMEGA_A + m * OMEGA_D) * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    return Phi


# ══════════════════════════════════════════════════════════════════════════════
# §2  THREE SOLVERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_normal_eq(t: np.ndarray, x: np.ndarray,
                  n_a: int = N_ANNUAL,
                  n_d: int = N_DIURNAL,
                  cx: List[Tuple[int, int]] = CROSS_TERMS,
                  chunk: int = 8760) -> np.ndarray:
    """
    Reference solver — chunked normal equations.

    Accumulates  AtA = ΦᵀΦ  and  Aty = Φᵀx  one year at a time, then
    solves the P×P system.  Memory scales as O(P²) rather than O(N·P).

    Returns
    -------
    theta : np.ndarray, shape (P,)
    """
    P = _n_params(n_a, n_d, cx)
    AtA = np.zeros((P, P))
    Aty = np.zeros(P)

    for i0 in range(0, len(t), chunk):
        i1 = min(i0 + chunk, len(t))
        t_c, x_c = t[i0:i1], x[i0:i1]
        v = np.isfinite(x_c)
        if not v.any():
            continue
        t_v, x_v = t_c[v], x_c[v]
        Phi = FS1._build_chunk(t_v, n_a, n_d, cx, OMEGA_A, OMEGA_D)
        AtA += Phi.T @ Phi
        Aty += Phi.T @ x_v

    return np.linalg.solve(AtA, Aty)


def fit_qr(t: np.ndarray, x: np.ndarray,
           n_a: int = N_ANNUAL,
           n_d: int = N_DIURNAL,
           cx: List[Tuple[int, int]] = CROSS_TERMS
           ) -> Tuple[np.ndarray, Dict]:
    """
    QR least squares via Householder factorization.

    Solves  min ‖Φθ − x‖₂  as

        Φ = Q·R     →     θ = R⁻¹ · Qᵀ·x

    Condition number is computed from R only (an P×P matrix), which
    equals cond(Φ) up to round-off and is far cheaper than an SVD of
    the full N×P design matrix.

    Returns
    -------
    theta : np.ndarray, shape (P,)
    info  : dict with keys 'method', 'N', 'P', 'cond_R', 'rmse', 'time_s'
    """
    t_start = time.perf_counter()

    Phi = build_full_design(t, n_a, n_d, cx)
    valid = np.isfinite(x)
    Phi = Phi[valid]
    xv = x[valid]

    Q, R = scipy_qr(Phi, mode='economic', check_finite=False)
    theta = np.linalg.solve(R, Q.T @ xv)

    resid = xv - Phi @ theta
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    info = {
        'method': 'QR',
        'N': int(valid.sum()),
        'P': Phi.shape[1],
        'cond_R': float(np.linalg.cond(R)),
        'rmse': rmse,
        'time_s': time.perf_counter() - t_start,
    }
    return theta, info


def fit_bayesian(t: np.ndarray, x: np.ndarray,
                 n_a: int = N_ANNUAL,
                 n_d: int = N_DIURNAL,
                 cx: List[Tuple[int, int]] = CROSS_TERMS,
                 max_iter: int = 300,
                 tol: float = 1e-8
                 ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Bayesian Ridge regression.

    Generative model
    ----------------
        x  =  Φ·θ + ε,      ε  ~  N(0, 1/α_noise)
        θ  ~  N(0, 1/λ_weight · I)

    with α_noise (noise precision) and λ_weight (weight precision)
    estimated from the data via type-II maximum likelihood.

    Posterior
    ---------
        θ | x  ~  N(μ, Σ),      Σ = (α_noise · ΦᵀΦ + λ_weight · I)⁻¹

    The posterior mean μ is the returned point estimate; the diagonal
    of Σ gives the per-coefficient posterior variance.

    scikit-learn attribute mapping
    ------------------------------
        model.alpha_   =  α_noise   (noise precision)   → σ²_noise = 1/α
        model.lambda_  =  λ_weight  (weight precision)  → σ²_weights = 1/λ
        model.sigma_   =  Σ          (posterior covariance)

    Returns
    -------
    theta_mean : np.ndarray, shape (P,)
    theta_std  : np.ndarray, shape (P,)   sqrt(diag(Σ))
    info       : dict with keys:
                    'method', 'N', 'P',
                    'noise_precision', 'weight_precision',
                    'sigma_noise', 'sigma_weights',
                    'n_iter', 'time_s'
    """
    t_start = time.perf_counter()

    Phi = build_full_design(t, n_a, n_d, cx)
    valid = np.isfinite(x)
    Phi = Phi[valid]
    xv = x[valid]

    model = BayesianRidge(
        max_iter=max_iter,
        tol=tol,
        compute_score=True,
        fit_intercept=False,   # DC column already in Φ
    )
    model.fit(Phi, xv)

    theta_mean = model.coef_
    theta_std  = np.sqrt(np.diag(model.sigma_))

    alpha_noise  = float(model.alpha_)    # noise precision
    lambda_weight = float(model.lambda_)   # weight precision

    info = {
        'method': 'BayesianRidge',
        'N': int(valid.sum()),
        'P': Phi.shape[1],
        'noise_precision':  alpha_noise,
        'weight_precision': lambda_weight,
        'sigma_noise':      1.0 / math.sqrt(alpha_noise),
        'sigma_weights':    1.0 / math.sqrt(lambda_weight),
        'n_iter': int(len(model.scores_)),
        'time_s': time.perf_counter() - t_start,
    }
    return theta_mean, theta_std, info


# ══════════════════════════════════════════════════════════════════════════════
# §3  COMPARISON HARNESS
# ══════════════════════════════════════════════════════════════════════════════

def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative difference, guarded against zero denominators."""
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    return float(np.max(np.abs(a - b) / denom))


def _posterior_std_per_term(theta_std: np.ndarray
                            ) -> Tuple[Dict, Dict, Dict]:
    """
    Map the flat posterior σ vector to per-amplitude σ per term.

    For a term with coefficients (a, b) in (cos, sin) form, the
    amplitude is A = √(a² + b²).  When a and b are independent with
    equal variance σ², first-order propagation gives

        σ(A) ≈ σ

    (the (a/A)² + (b/A)² = 1 normalization is exact for the variance).
    Because the design matrix here is nearly orthogonal, σ(a) = σ(b)
    is an excellent approximation.

    Returns
    -------
    (std_ann, std_dia, std_cross) — each dict mapping index/key to σ(A).
    """
    std_ann = {}
    for k in range(1, N_ANNUAL + 1):
        c0 = 1 + 2 * (k - 1)
        std_ann[k] = float(math.hypot(theta_std[c0], theta_std[c0 + 1]))

    std_dia = {}
    for m in range(1, N_DIURNAL + 1):
        c0 = 1 + 2 * N_ANNUAL + 2 * (m - 1)
        std_dia[m] = float(math.hypot(theta_std[c0], theta_std[c0 + 1]))

    std_cross = {}
    for i, (k, m) in enumerate(CROSS_TERMS):
        c0 = 1 + 2 * N_ANNUAL + 2 * N_DIURNAL + 2 * i
        std_cross[(k, m)] = float(math.hypot(theta_std[c0], theta_std[c0 + 1]))

    return std_ann, std_dia, std_cross


def compare_all(vk: str, t: np.ndarray, x: np.ndarray,
                sigma_threshold: float = 2.0) -> Dict:
    """
    Run all three solvers on a single variable and produce a full
    comparison report.

    Parameters
    ----------
    vk : str
        Variable key, e.g. 'TEMP'.
    t : np.ndarray
        Time coordinates (Julian hours from epoch).
    x : np.ndarray
        Observations (already scaled to physical units).
    sigma_threshold : float
        Significance cutoff in units of posterior σ (default 2.0 ≈ 95 %
        credible interval).

    Returns
    -------
    report : dict with fitted θ from each solver, agreement metrics,
             posterior σ mapping, and significance counts.
    """
    print(f"\n{'═' * 78}")
    print(f"  COMPARISON — {vk}")
    print(f"{'═' * 78}")

    # ---- 1/3 Normal equations ----
    print("  [1/3] Normal equations …", end=' ', flush=True)
    t0 = time.perf_counter()
    theta_ne = fit_normal_eq(t, x)
    print(f"done   ({time.perf_counter() - t0:.2f}s)")

    # ---- 2/3 QR ----
    print("  [2/3] QR least squares …", end=' ', flush=True)
    theta_qr, info_qr = fit_qr(t, x)
    print(f"done   (cond(R)={info_qr['cond_R']:.3e}, "
          f"{info_qr['time_s']:.2f}s)")

    # ---- 3/3 Bayesian Ridge ----
    print("  [3/3] Bayesian Ridge …", end=' ', flush=True)
    theta_br, theta_std, info_br = fit_bayesian(t, x)
    print(f"done   (α_noise={info_br['noise_precision']:.4e}, "
          f"λ_weight={info_br['weight_precision']:.4e}, "
          f"{info_br['n_iter']} iters, "
          f"{info_br['time_s']:.2f}s)")

    # ---- Agreement between solvers ----
    print(f"\n  Solver agreement (max relative diff):")
    print(f"    ‖θ_NE − θ_QR‖ :  {_rel_diff(theta_ne, theta_qr):.3e}")
    print(f"    ‖θ_NE − θ_BR‖ :  {_rel_diff(theta_ne, theta_br):.3e}")
    print(f"    ‖θ_QR − θ_BR‖ :  {_rel_diff(theta_qr, theta_br):.3e}")

    # ---- Structured coefficients (QR as canonical) ----
    ann  = extract_annual(theta_qr, N_ANNUAL)
    dia  = extract_diurnal(theta_qr, N_ANNUAL, N_DIURNAL)
    cros = extract_cross(theta_qr, N_ANNUAL, N_DIURNAL, CROSS_TERMS)

    std_ann, std_dia, std_cross = _posterior_std_per_term(theta_std)

    # ---- Significance table: ANNUAL ----
    print(f"\n  ANNUAL — significant if A > {sigma_threshold}σ")
    print(f"  {'k':>3}  {'A (QR)':>12}  {'σ (Bayes)':>12}  "
          f"{'A/σ':>8}  {'sig?':>5}")
    print("  " + "─" * (78 - 4))

    n_sig_ann = 0
    for t_ in ann:
        k = t_['k']
        s = std_ann[k]
        ratio = t_['A'] / s if s > 0 else float('inf')
        sig = ratio > sigma_threshold
        if sig:
            n_sig_ann += 1
        print(f"  {k:>3}  {t_['A']:>12.6f}  {s:>12.6f}  "
              f"{ratio:>8.2f}  {'YES' if sig else 'no':>5}")
    print(f"\n  → {n_sig_ann} of {N_ANNUAL} annual harmonics significant")

    # ---- Significance table: DIURNAL ----
    print(f"\n  DIURNAL — significant if A > {sigma_threshold}σ")
    print(f"  {'m':>3}  {'A (QR)':>12}  {'σ (Bayes)':>12}  "
          f"{'A/σ':>8}  {'sig?':>5}")
    print("  " + "─" * (78 - 4))

    n_sig_dia = 0
    for t_ in dia:
        m = t_['m']
        s = std_dia[m]
        ratio = t_['A'] / s if s > 0 else float('inf')
        sig = ratio > sigma_threshold
        if sig:
            n_sig_dia += 1
        print(f"  {m:>3}  {t_['A']:>12.6f}  {s:>12.6f}  "
              f"{ratio:>8.2f}  {'YES' if sig else 'no':>5}")
    print(f"\n  → {n_sig_dia} of {N_DIURNAL} diurnal harmonics significant")

    # ---- Significance table: CROSS (top-15) ----
    print(f"\n  CROSS — top-15 by amplitude")
    print(f"  {'(k,m)':>8}  {'A (QR)':>12}  {'σ (Bayes)':>12}  "
          f"{'A/σ':>8}  {'sig?':>5}")
    print("  " + "─" * (78 - 4))

    n_sig_cr = 0
    for t_ in cros[:15]:
        key = (t_['k'], t_['m'])
        s = std_cross[key]
        ratio = t_['A'] / s if s > 0 else float('inf')
        sig = ratio > sigma_threshold
        if sig:
            n_sig_cr += 1
        km = f"({t_['k']:+d},{t_['m']:+d})"
        print(f"  {km:>8}  {t_['A']:>12.6f}  {s:>12.6f}  "
              f"{ratio:>8.2f}  {'YES' if sig else 'no':>5}")

    n_sig_cr_total = sum(
        1 for t_ in cros
        if t_['A'] / std_cross[(t_['k'], t_['m'])] > sigma_threshold
    )
    n_total_sig = n_sig_ann + n_sig_dia + n_sig_cr_total
    P_total = _n_params(N_ANNUAL, N_DIURNAL, CROSS_TERMS)

    print(f"\n  TOTAL significant: {n_total_sig} / {P_total}")

    return {
        'theta_ne': theta_ne,
        'theta_qr': theta_qr,
        'theta_br': theta_br,
        'theta_std': theta_std,
        'info_qr': info_qr,
        'info_br': info_br,
        'n_sig_annual': n_sig_ann,
        'n_sig_diurnal': n_sig_dia,
        'n_sig_cross': n_sig_cr_total,
        'n_sig_total': n_total_sig,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="FS1 Advanced — compare OLS vs QR vs Bayesian Ridge")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--var", default="TEMP",
                    help="Variable key (TEMP, RH, SW, …)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Significance threshold in units of posterior σ")
    args = ap.parse_args()

    # ---- Load data exactly as FS1 does ----
    print("Loading data …")
    fdict, data_dir = FS1._resolve_files(args.data_dir)
    W_A, W_B, d_A, d_B = FS1.idw_weights(
        FS1.PHI_T, FS1.LAM_T, FS1.PHI_A, FS1.LAM_A,
        FS1.PHI_B, FS1.LAM_B, FS1.IDW_POWER)
    sta_a = FS1.load_station(fdict["A"])
    sta_b = FS1.load_station(fdict["B"])
    cols = sorted(set(sta_a.columns) | set(sta_b.columns))
    merged = FS1.idw_merge(sta_a, sta_b, W_A, W_B, cols)
    t_jh = FS1.datetime_to_j2000h(merged.index)

    vd = FS1.VARS[args.var]
    col = vd['col']
    if col not in merged.columns:
        raise SystemExit(f"Column '{col}' not found")
    x = merged[col].values.astype(np.float64) * vd['scale']

    print(f"  N = {len(x):,} h   variable = {vd['name']}   "
          f"unit = {vd['unit']}")

    # ---- Run comparison ----
    report = compare_all(args.var, t_jh, x, sigma_threshold=args.sigma)

    # ---- Cross-check vs FS1.fit_variable ----
    print(f"\n{'─' * 78}")
    print(f"  Reference cross-check vs FS1.fit_variable")
    print(f"{'─' * 78}")
    theta_ref, mse_ref, N_ref = FS1.fit_variable(
        t_jh, x, FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS)
    diff = float(np.max(np.abs(theta_ref - report['theta_qr'])))
    print(f"  max|θ_FS1 − θ_QR| = {diff:.6e}")
    print(f"  →  {'IDENTICAL' if diff < 1e-6 else 'DIFFERENT'}")


if __name__ == "__main__":
    main()