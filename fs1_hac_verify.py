#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_hac_verify.py
================================================================================
HAC (Newey-West) standard errors for FS1 Fourier coefficients.

Motivation
----------
The Bayesian Ridge implementation in FS1_Batch_Bayes.py assumes i.i.d.
residuals (ε ~ N(0, 1/α)). For hourly meteorological data this is severely
violated: the lag-1 autocorrelation of residuals is typically ρ₁ ≈ 0.95–0.99.

Consequence:
    σ_posterior(BR)  <<  σ_true
    → thousands of harmonics appear significant (A/σ > 10) that are in
      reality pure noise.

This module recomputes coefficient uncertainties using the
Newey-West (1987) HAC estimator with Bartlett kernel:

    V(θ̂) = (ΦᵀΦ)⁻¹ · Ŝ · (ΦᵀΦ)⁻¹
    Ŝ    = Γ₀ + Σ_{j=1}^{L} wⱼ · (Γⱼ + Γⱼᵀ)
    Γⱼ   = Σₜ ûₜ ûₜ₋ⱼᵀ          (ûₜ = Φₜ · εₜ  — the score matrix)
    wⱼ   = 1 − j/(L+1)          (Bartlett kernel)

Per-harmonic σ(A) is derived from V(θ̂) by first-order propagation:

    σ²(A) = (a/A)²·V_aa + (b/A)²·V_bb + 2·(a/A)·(b/A)·V_ab

Output
------
Side-by-side comparison of:
    • Bayesian Ridge σ      (i.i.d. assumption)
    • HAC σ                 (autocorrelation-corrected)
    • Significance counts at 2σ
    • Per-term breakdown for the most affected variables

Runtime
-------
Roughly 60–180 s per variable, dominated by the L matrix products in
the HAC sum. With 14 variables, expect 20–40 minutes total.

Author : MJS · Jolotundo Observatory
Version: 1.0.0
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np

import FS1_FourierSeries as FS1
from FS1_Advanced import fit_bayesian, build_full_design
from fs1_verify_pc1 import load_merged


# ══════════════════════════════════════════════════════════════════════════════
# §1  HAC COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def hac_covariance(Phi: np.ndarray,
                   resid: np.ndarray,
                   maxlags: int = 72) -> np.ndarray:
    """
    Newey-West (1987) HAC covariance matrix with Bartlett kernel.

    Parameters
    ----------
    Phi    : (N, P) design matrix
    resid  : (N,)   OLS residuals
    maxlags: int    bandwidth (default 72 h = 3 days)

    Returns
    -------
    V : (P, P) asymptotic covariance of θ̂

    Notes
    -----
    For tropical hourly data with strong diurnal + synoptic-scale
    autocorrelation, maxlags ∈ [48, 168] is standard. Below 48 the
    variance is underestimated; above 168 the estimator becomes noisy
    due to finite-sample bias of the Bartlett kernel.
    """
    N, P = Phi.shape
    u = Phi * resid[:, None]                     # (N, P) score matrix

    XtX = Phi.T @ Phi
    XtX_inv = np.linalg.pinv(XtX)                # guard against rank loss

    S = u.T @ u                                  # Γ₀

    L = min(maxlags, N - 1)
    for j in range(1, L + 1):
        w = 1.0 - j / (maxlags + 1.0)            # Bartlett weight
        G_j = u[j:].T @ u[:-j]                   # Γⱼ
        S += w * (G_j + G_j.T)

    return XtX_inv @ S @ XtX_inv


def sigma_per_amplitude(theta: np.ndarray,
                        V: np.ndarray,
                        pairs: List[Tuple[int, int]]) -> List[float]:
    """
    First-order propagation from (cos, sin) coefficients to amplitude σ.

    For A = √(a² + b²):
        σ²(A) = (a/A)²·V_aa + (b/A)²·V_bb + 2·(a/A)·(b/A)·V_ab

    If A ≈ 0, returns σ(a) as fallback.
    """
    out = []
    for (ia, ib) in pairs:
        a, b = float(theta[ia]), float(theta[ib])
        A = math.hypot(a, b)
        if A < 1e-30:
            out.append(math.sqrt(max(V[ia, ia], V[ib, ib], 0.0)))
            continue
        va, vb = a / A, b / A
        var = (va ** 2) * V[ia, ia] + (vb ** 2) * V[ib, ib] \
              + 2.0 * va * vb * V[ia, ib]
        out.append(math.sqrt(max(var, 0.0)))
    return out


def collect_pairs(n_a: int, n_d: int, cx: List[Tuple[int, int]]
                  ) -> Tuple[List[Tuple[int, int]],
                             List[Tuple[int, int]],
                             List[Tuple[int, int]]]:
    """
    Column-index pairs for annual / diurnal / cross terms in θ.
    Order matches extract_annual / extract_diurnal / extract_cross.
    """
    ann = [(1 + 2 * (k - 1), 1 + 2 * (k - 1) + 1)
           for k in range(1, n_a + 1)]
    off_d = 1 + 2 * n_a
    dia = [(off_d + 2 * (m - 1), off_d + 2 * (m - 1) + 1)
           for m in range(1, n_d + 1)]
    off_c = off_d + 2 * n_d
    cross = [(off_c + 2 * i, off_c + 2 * i + 1)
             for i in range(len(cx))]
    return ann, dia, cross


# ══════════════════════════════════════════════════════════════════════════════
# §2  SINGLE-VARIABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze(vk: str, vd: Dict, t_jh: np.ndarray, x: np.ndarray,
            maxlags: int = 72, sigma_threshold: float = 2.0) -> Dict:
    """
    Compare Bayesian Ridge σ with HAC σ for one variable.
    """
    t0 = time.perf_counter()

    # ---- OLS via FS1 (chunked, low memory) ----
    theta_ols, _, _ = FS1.fit_variable(
        t_jh, x, FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS
    )

    # ---- Full design matrix (needed for HAC scores) ----
    valid = np.isfinite(x)
    t_v, x_v = t_jh[valid], x[valid]
    Phi = build_full_design(t_v, FS1.N_ANNUAL, FS1.N_DIURNAL,
                            FS1.CROSS_TERMS)

    # ---- Residuals ----
    resid = x_v - Phi @ theta_ols
    sigma_resid = float(np.std(resid, ddof=0))
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])

    # ---- HAC covariance ----
    V_hac = hac_covariance(Phi, resid, maxlags=maxlags)

    # ---- Bayesian Ridge for comparison ----
    theta_br, theta_std_br, info_br = fit_bayesian(t_jh, x)

    # ---- Column pairs ----
    ann_pairs, dia_pairs, cross_pairs = collect_pairs(
        FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS
    )

    # ---- Per-amplitude σ: BR (hypot of consecutive coef σs) ----
    def br_sigma(pairs):
        return [math.hypot(theta_std_br[i], theta_std_br[i + 1])
                for (i, _) in pairs]

    sig_ann_br   = br_sigma(ann_pairs)
    sig_dia_br   = br_sigma(dia_pairs)
    sig_cross_br = br_sigma(cross_pairs)

    # ---- Per-amplitude σ: HAC via first-order propagation ----
    sig_ann_hac   = sigma_per_amplitude(theta_ols, V_hac, ann_pairs)
    sig_dia_hac   = sigma_per_amplitude(theta_ols, V_hac, dia_pairs)
    sig_cross_hac = sigma_per_amplitude(theta_ols, V_hac, cross_pairs)

    # ---- Amplitudes ----
    ann  = FS1.extract_annual(theta_ols, FS1.N_ANNUAL)
    dia  = FS1.extract_diurnal(theta_ols, FS1.N_ANNUAL, FS1.N_DIURNAL)
    cros = FS1.extract_cross(theta_ols, FS1.N_ANNUAL, FS1.N_DIURNAL,
                              FS1.CROSS_TERMS)

    # ---- Significance counts ----
    def count_sig(terms, sigmas):
        return sum(1 for t, s in zip(terms, sigmas)
                   if s > 0 and t["A"] > sigma_threshold * s)

    n_sig_br  = (count_sig(ann,  sig_ann_br) +
                 count_sig(dia,  sig_dia_br) +
                 count_sig(cros, sig_cross_br))
    n_sig_hac = (count_sig(ann,  sig_ann_hac) +
                 count_sig(dia,  sig_dia_hac) +
                 count_sig(cros, sig_cross_hac))

    # Representative σ (annual k = 1)
    sig_rep_br  = sig_ann_br[0]  if sig_ann_br  else float("nan")
    sig_rep_hac = sig_ann_hac[0] if sig_ann_hac else float("nan")

    return {
        "vk": vk, "sym": vd["sym"], "unit": vd["unit"],
        "N": int(valid.sum()), "P": int(Phi.shape[1]),
        "maxlags": maxlags,
        "sigma_resid": sigma_resid,
        "rho1": rho1,
        "sigma_rep_br": sig_rep_br,
        "sigma_rep_hac": sig_rep_hac,
        "sigma_ratio": sig_rep_hac / sig_rep_br if sig_rep_br > 0 else float("nan"),
        "n_sig_br": n_sig_br,
        "n_sig_hac": n_sig_hac,
        "n_total": len(ann) + len(dia) + len(cros),
        # Per-term breakdown for detailed tables
        "ann":  [{"k": t["k"],  "A": t["A"], "phi": t["phi"],
                  "sig_br":  sig_ann_br[i],  "sig_hac": sig_ann_hac[i]}
                 for i, t in enumerate(ann)],
        "dia":  [{"m": t["m"],  "A": t["A"], "phi": t["phi"],
                  "sig_br":  sig_dia_br[i],  "sig_hac": sig_dia_hac[i]}
                 for i, t in enumerate(dia)],
        "cross": [{"k": t["k"], "m": t["m"], "A": t["A"], "phi": t["phi"],
                   "sig_br":  sig_cross_br[i], "sig_hac": sig_cross_hac[i]}
                  for i, t in enumerate(cros)],
        "time_s": time.perf_counter() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §3  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict], maxlags: int, threshold: float):
    W = 120
    print()
    print("=" * W)
    print(f"  HAC vs BAYESIAN RIDGE — UNCERTAINTY COMPARISON "
          f"(maxlags = {maxlags} h, threshold = {threshold}σ)")
    print("=" * W)

    print(f"\n  {'var':<8}{'sym':<7}{'ρ₁':>7}"
          f"{'σ_BR':>13}{'σ_HAC':>13}{'ratio':>8}"
          f"  {'sig_BR':>10}{'sig_HAC':>10}{'Δ':>9}")
    print("  " + "─" * (W - 4))

    for r in results:
        d = r["n_sig_hac"] - r["n_sig_br"]
        print(f"  {r['vk']:<8}{r['sym']:<7}"
              f"{r['rho1']:>7.3f}"
              f"{r['sigma_rep_br']:>13.6f}"
              f"{r['sigma_rep_hac']:>13.6f}"
              f"{r['sigma_ratio']:>7.2f}× "
              f"{r['n_sig_br']:>5}/{r['n_total']:<4}"
              f"{r['n_sig_hac']:>5}/{r['n_total']:<4}"
              f"{d:>+9}")

    print(f"\n  Legend:")
    print(f"    ρ₁      = lag-1 autocorrelation of OLS residuals")
    print(f"    σ_BR    = representative posterior σ from Bayesian Ridge (i.i.d.)")
    print(f"    σ_HAC   = autocorrelation-corrected σ (annual k = 1)")
    print(f"    ratio   = σ_HAC / σ_BR  (>1 ⇒ BR underestimates)")
    print(f"    sig     = # terms with |A| > {threshold}σ")

    ratios = [r["sigma_ratio"] for r in results
              if np.isfinite(r["sigma_ratio"])]
    if ratios:
        print(f"\n  σ inflation factor: "
              f"median = {np.median(ratios):.2f}×   "
              f"range = {min(ratios):.2f}× – {max(ratios):.2f}×")

    tot_br  = sum(r["n_sig_br"]  for r in results)
    tot_hac = sum(r["n_sig_hac"] for r in results)
    tot_all = sum(r["n_total"]   for r in results)
    print(f"  Significance reduction: {tot_br} → {tot_hac} "
          f"(out of {tot_all} possible; "
          f"{100*(1 - tot_hac/max(tot_br,1)):+.1f}%)")
    print()


def print_detail(r: Dict, n_show: int = 10):
    """
    Show which terms flip from significant to non-significant under
    HAC. Sorted by absolute σ inflation.
    """
    W = 100
    print()
    print("=" * W)
    print(f"  DETAIL — {r['vk']}  ({r['sym']}, {r['unit']})   "
          f"N={r['N']:,}   ρ₁={r['rho1']:.4f}   "
          f"σ_BR={r['sigma_rep_br']:.6f}   σ_HAC={r['sigma_rep_hac']:.6f}")
    print("=" * W)

    # ----- Annual -----
    print(f"\n  ANNUAL  (top {n_show} by A)")
    print(f"  {'k':>3}  {'A':>12}  {'σ_BR':>11}  {'σ_HAC':>11}  "
          f"{'A/σ_BR':>8}  {'A/σ_HAC':>9}  flag")
    print("  " + "─" * (W - 4))
    for t in sorted(r["ann"], key=lambda x: -x["A"])[:n_show]:
        rb = t["A"] / t["sig_br"]  if t["sig_br"]  > 0 else float("inf")
        rh = t["A"] / t["sig_hac"] if t["sig_hac"] > 0 else float("inf")
        flag = "✓✓" if rh > 2 else ("✓" if rh > 1 else "··")
        print(f"  {t['k']:>3}  {t['A']:>12.6f}  {t['sig_br']:>11.6f}  "
              f"{t['sig_hac']:>11.6f}  {rb:>8.2f}  {rh:>9.2f}  {flag}")

    # ----- Diurnal -----
    print(f"\n  DIURNAL  (top {min(n_show, 11)} by A)")
    print(f"  {'m':>3}  {'A':>12}  {'σ_BR':>11}  {'σ_HAC':>11}  "
          f"{'A/σ_BR':>8}  {'A/σ_HAC':>9}  flag")
    print("  " + "─" * (W - 4))
    for t in sorted(r["dia"], key=lambda x: -x["A"])[:n_show]:
        rb = t["A"] / t["sig_br"]  if t["sig_br"]  > 0 else float("inf")
        rh = t["A"] / t["sig_hac"] if t["sig_hac"] > 0 else float("inf")
        flag = "✓✓" if rh > 2 else ("✓" if rh > 1 else "··")
        print(f"  {t['m']:>3}  {t['A']:>12.6f}  {t['sig_br']:>11.6f}  "
              f"{t['sig_hac']:>11.6f}  {rb:>8.2f}  {rh:>9.2f}  {flag}")

    # ----- Cross -----
    print(f"\n  CROSS  (top {n_show} by A)")
    print(f"  {'(k,m)':>8}  {'A':>12}  {'σ_BR':>11}  {'σ_HAC':>11}  "
          f"{'A/σ_BR':>8}  {'A/σ_HAC':>9}  flag")
    print("  " + "─" * (W - 4))
    for t in r["cross"][:n_show]:
        rb = t["A"] / t["sig_br"]  if t["sig_br"]  > 0 else float("inf")
        rh = t["A"] / t["sig_hac"] if t["sig_hac"] > 0 else float("inf")
        flag = "✓✓" if rh > 2 else ("✓" if rh > 1 else "··")
        km = f"({t['k']:+d},{t['m']:+d})"
        print(f"  {km:>8}  {t['A']:>12.6f}  {t['sig_br']:>11.6f}  "
              f"{t['sig_hac']:>11.6f}  {rb:>8.2f}  {rh:>9.2f}  {flag}")

    print("\n  flag:  ✓✓ still significant at 2σ_HAC  ·  "
          "✓ significant at 1σ_HAC  ·  ·· now below 1σ_HAC")


# ══════════════════════════════════════════════════════════════════════════════
# §4  SANITY CHECK (optional)
# ══════════════════════════════════════════════════════════════════════════════

def _sanity_check():
    """
    Validate the HAC implementation on synthetic AR(1) data.
    Confirms that HAC inflates σ correctly.
    """
    print("  Sanity check on synthetic AR(1) noise …", end=" ", flush=True)
    rng = np.random.default_rng(42)
    N = 8000
    rho = 0.90
    eps = rng.standard_normal(N)
    noise = np.zeros(N)
    for i in range(1, N):
        noise[i] = rho * noise[i - 1] + eps[i]

    t = np.arange(N) / 24.0
    Phi = np.column_stack([np.ones(N),
                           np.cos(2 * np.pi * t),
                           np.sin(2 * np.pi * t)])
    y = Phi @ np.array([0.5, 1.0, -0.3]) + noise

    theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    resid = y - Phi @ theta

    # Naive OLS
    s2 = float(resid @ resid) / (N - 3)
    V_ols = s2 * np.linalg.inv(Phi.T @ Phi)

    # HAC
    V_hac = hac_covariance(Phi, resid, maxlags=72)

    # Theoretical inflation for AR(1):
    #   var inflation = (1+ρ)/(1−ρ)  for the DC term
    theo = math.sqrt((1 + rho) / (1 - rho))
    emp_dc = math.sqrt(V_hac[0, 0] / V_ols[0, 0])
    print(f"done")
    print(f"    Theoretical σ inflation (DC):  {theo:.3f}×")
    print(f"    HAC empirical  σ inflation:    {emp_dc:.3f}×")
    print(f"    Agreement: "
          f"{'OK ✓' if abs(emp_dc/theo - 1) < 0.15 else 'POOR ✗'}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §5  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="HAC standard errors for FS1 Fourier coefficients")
    ap.add_argument("--maxlags", type=int, default=72,
                    help="Newey-West bandwidth in hours (default 72)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Significance threshold in σ units (default 2)")
    ap.add_argument("--vars", default="",
                    help="Comma-separated variables (default: all)")
    ap.add_argument("--detail", default="",
                    help="Comma-separated vars for detailed tables")
    ap.add_argument("--out", default="fs1_hac_verify.json")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    if not args.skip_sanity:
        print("=" * 78)
        _sanity_check()

    print("Loading merged hourly data …", flush=True)
    merged = load_merged()
    print(f"  {len(merged):,} hourly records")

    t_jh = FS1.datetime_to_j2000h(merged.index)

    var_keys = (
        [v.strip() for v in args.vars.split(",") if v.strip()]
        or list(FS1.VARS.keys())
    )
    detail_keys = set(v.strip() for v in args.detail.split(",") if v.strip())

    results = []
    for vk in var_keys:
        vd = FS1.VARS[vk]
        col = vd["col"]
        if col not in merged.columns:
            print(f"  [{vk:<6}] column missing — skip")
            continue

        x = merged[col].values.astype(np.float64) * vd["scale"]
        print(f"  [{vk:<6}] {vd['sym']:<6} fitting …", end=" ", flush=True)

        r = analyze(vk, vd, t_jh, x,
                    maxlags=args.maxlags,
                    sigma_threshold=args.sigma)
        results.append(r)

        print(f"{r['time_s']:>5.1f}s   "
              f"ρ₁={r['rho1']:>5.3f}   "
              f"σ ratio={r['sigma_ratio']:>5.2f}×   "
              f"sig {r['n_sig_br']} → {r['n_sig_hac']}")

    print_summary(results, args.maxlags, args.sigma)

    # Detailed tables for selected variables
    for r in results:
        if r["vk"] in detail_keys:
            print_detail(r, n_show=10)

    # ---- JSON export ----
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "maxlags": args.maxlags,
                "sigma_threshold": args.sigma,
                "N_annual": FS1.N_ANNUAL,
                "N_diurnal": FS1.N_DIURNAL,
                "N_cross": len(FS1.CROSS_TERMS),
                "estimator": "Newey-West (1987), Bartlett kernel",
            },
            "series": results,
        }, fh, indent=2, default=_ser)
    print(f"  JSON → {args.out}\n")


if __name__ == "__main__":
    main()