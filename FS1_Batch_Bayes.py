#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS1_Batch_Bayes.py
================================================================================
Batch Bayesian Ridge analysis for all 14 meteorological variables.

Uses Bayesian Ridge throughout — no shortcuts, no analytical σ approximation.

For each variable:
    - fit θ via chunked normal equations (initial guess, fast)
    - fit θ via Bayesian Ridge → posterior mean + posterior σ
    - count harmonics with |A| > 2σ_posterior
    - report R² progression + per-harmonic significance

Runtime: ~35 s per variable × 14 ≈ 8 minutes total.

Author : MJS · Jolotundo Observatory
Version: 1.0.0
================================================================================
"""

from __future__ import annotations

import json
import math
import time
from typing import Dict, List

import numpy as np

import FS1_FourierSeries as FS1
from FS1_Advanced import fit_normal_eq, fit_bayesian


# ══════════════════════════════════════════════════════════════════════════════
# §1  SINGLE-VARIABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bayes(vk: str, vd: Dict,
                  t_jh: np.ndarray,
                  x: np.ndarray,
                  sigma_threshold: float = 2.0) -> Dict:
    """
    Fit one variable with Bayesian Ridge and count significant harmonics
    using the *posterior* σ returned by the model.

    Posterior σ per amplitude is derived from the posterior covariance
    of the (cos, sin) coefficient pair:

        σ(Aₖ) = √(σ²(aₖ) + σ²(bₖ))

    where σ(aₖ) and σ(bₖ) are the diagonal entries of the posterior
    covariance matrix Σ = (β·ΦᵀΦ + α·I)⁻¹.
    """
    t0 = time.perf_counter()

    # ---- Bayesian Ridge fit ----
    theta_mean, theta_std, info_br = fit_bayesian(t_jh, x)

    mean_val = float(theta_mean[0])

    # ---- Extract structured coefficients from posterior mean ----
    ann  = FS1.extract_annual(theta_mean, FS1.N_ANNUAL)
    dia  = FS1.extract_diurnal(theta_mean, FS1.N_ANNUAL, FS1.N_DIURNAL)
    cros = FS1.extract_cross(theta_mean, FS1.N_ANNUAL, FS1.N_DIURNAL,
                              FS1.CROSS_TERMS)

    # ---- Fit statistics ----
    fit = FS1.incremental_r2(t_jh, x, theta_mean,
                              FS1.N_ANNUAL, FS1.N_DIURNAL,
                              FS1.CROSS_TERMS)

    # ---- Posterior σ per amplitude term ----
    # Annual: pair (aₖ, bₖ) at columns 1 + 2(k-1), 1 + 2(k-1) + 1
    std_ann = {}
    for k in range(1, FS1.N_ANNUAL + 1):
        c0 = 1 + 2 * (k - 1)
        s = math.sqrt(theta_std[c0] ** 2 + theta_std[c0 + 1] ** 2)
        std_ann[k] = s

    std_dia = {}
    for m in range(1, FS1.N_DIURNAL + 1):
        c0 = 1 + 2 * FS1.N_ANNUAL + 2 * (m - 1)
        s = math.sqrt(theta_std[c0] ** 2 + theta_std[c0 + 1] ** 2)
        std_dia[m] = s

    std_cros = {}
    for i, (k, m) in enumerate(FS1.CROSS_TERMS):
        c0 = 1 + 2 * FS1.N_ANNUAL + 2 * FS1.N_DIURNAL + 2 * i
        s = math.sqrt(theta_std[c0] ** 2 + theta_std[c0 + 1] ** 2)
        std_cros[(k, m)] = s

    # ---- Significance counts ----
    n_sig_ann = sum(1 for t in ann  if t["A"] > sigma_threshold * std_ann[t["k"]])
    n_sig_dia = sum(1 for t in dia  if t["A"] > sigma_threshold * std_dia[t["m"]])
    n_sig_cr  = sum(1 for t in cros
                    if t["A"] > sigma_threshold * std_cros[(t["k"], t["m"])])

    # Representative σ (use annual k=1 as canonical — should be ~uniform)
    sigma_rep = std_ann[1]

    return {
        "vk": vk,
        "name": vd["name"],
        "sym": vd["sym"],
        "unit": vd["unit"],
        "mean": mean_val,
        "sigma_resid": fit["rmse_full"],
        "sigma_posterior": sigma_rep,
        "noise_precision":  info_br["noise_precision"],
        "weight_precision": info_br["weight_precision"],
        "sigma_noise":      info_br["sigma_noise"],
        "sigma_weights":    info_br["sigma_weights"],
        "R2_annual":  fit["r2_annual"],
        "R2_diurnal": fit["r2_diurnal"],
        "R2_full":    fit["r2_full"],
        "rmse_full":  fit["rmse_full"],
        "A_ann1": ann[0]["A"],
        "A_dia1": dia[0]["A"],
        "A_cross1": cros[0]["A"] if cros else 0.0,
        "n_sig_annual":  n_sig_ann,
        "n_sig_diurnal": n_sig_dia,
        "n_sig_cross":   n_sig_cr,
        "n_sig_total":   n_sig_ann + n_sig_dia + n_sig_cr,
        "n_annual":  FS1.N_ANNUAL,
        "n_diurnal": FS1.N_DIURNAL,
        "n_cross":   len(FS1.CROSS_TERMS),
        # Per-term details (for JSON export)
        "annual_terms": [
            {"k": t["k"], "A": t["A"], "phi": t["phi"],
             "sigma": std_ann[t["k"]], "A_over_sigma": t["A"] / std_ann[t["k"]]}
            for t in ann
        ],
        "diurnal_terms": [
            {"m": t["m"], "A": t["A"], "phi": t["phi"],
             "sigma": std_dia[t["m"]], "A_over_sigma": t["A"] / std_dia[t["m"]]}
            for t in dia
        ],
        "cross_terms": [
            {"k": t["k"], "m": t["m"], "A": t["A"], "phi": t["phi"],
             "sigma": std_cros[(t["k"], t["m"])],
             "A_over_sigma": t["A"] / std_cros[(t["k"], t["m"])]}
            for t in cros
        ],
        "time_s": time.perf_counter() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §2  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(data_dir: str = None):
    print("Loading data …", flush=True)
    fdict, base = FS1._resolve_files(data_dir)
    W_A, W_B, d_A, d_B = FS1.idw_weights(
        FS1.PHI_T, FS1.LAM_T, FS1.PHI_A, FS1.LAM_A,
        FS1.PHI_B, FS1.LAM_B, FS1.IDW_POWER)
    print(f"  IDW:  W_A = {W_A:.6f}  (d_A = {d_A:.4f} km)")
    print(f"        W_B = {W_B:.6f}  (d_B = {d_B:.4f} km)")

    sta_a = FS1.load_station(fdict["A"])
    sta_b = FS1.load_station(fdict["B"])
    cols = sorted(set(sta_a.columns) | set(sta_b.columns))
    merged = FS1.idw_merge(sta_a, sta_b, W_A, W_B, cols)

    t_jh = FS1.datetime_to_j2000h(merged.index)
    print(f"  Merged: {len(merged):,} h")
    return t_jh, merged


# ══════════════════════════════════════════════════════════════════════════════
# §3  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict], sigma: float):
    W = 118
    print()
    print("=" * W)
    print(f"  BAYESIAN RIDGE SUMMARY — 14 VARIABLES  (threshold = {sigma}σ)")
    print("=" * W)

    # ---- Fit quality ----
    print()
    print("  FIT QUALITY & POSTERIOR")
    print(f"  {'var':<8} {'sym':<6} {'R²_ann':>8} {'R²_dia':>8} {'R²_full':>8} "
          f"{'σ_resid':>11} {'σ_post':>11} {'noise_prec':>11} {'weight_prec':>11}")
    print("  " + "─" * (W - 4))
    for r in results:
        print(f"  {r['vk']:<8} {r['sym']:<6} "
              f"{r['R2_annual']:>8.4f} "
              f"{r['R2_diurnal']:>8.4f} "
              f"{r['R2_full']:>8.4f} "
              f"{r['sigma_resid']:>11.6f} "
              f"{r['sigma_posterior']:>11.6f} "
              f"{r['noise_precision']:>11.3e} "
              f"{r['weight_precision']:>11.3e}")

    # ---- Significance ----
    print()
    print(f"  SIGNIFICANT HARMONICS (A > {sigma}σ_posterior)")
    print(f"  {'var':<8} {'sym':<6} "
          f"{'ann':>8} {'dia':>8} {'cross':>8} {'total':>8} {'%':>6}")
    print("  " + "─" * (W - 4))
    for r in results:
        total_terms = r['n_annual'] + r['n_diurnal'] + r['n_cross']
        pct = 100.0 * r['n_sig_total'] / total_terms
        print(f"  {r['vk']:<8} {r['sym']:<6} "
              f"{r['n_sig_annual']:>3}/{r['n_annual']:<3} "
              f"{r['n_sig_diurnal']:>3}/{r['n_diurnal']:<3} "
              f"{r['n_sig_cross']:>3}/{r['n_cross']:<3} "
              f"{r['n_sig_total']:>6} "
              f"{pct:>5.1f}%")

    # ---- Ranking ----
    print()
    print("  RANKING BY STRUCTUREDNESS")
    for i, r in enumerate(sorted(results, key=lambda r: -r['n_sig_total']), 1):
        bar = "█" * int(r['n_sig_total'] / 2)
        print(f"  {i:>2}. {r['vk']:<8} {r['sym']:<6} "
              f"{r['n_sig_total']:>3}  {bar}")

    total_time = sum(r['time_s'] for r in results)
    print()
    print(f"  Total analysis time: {total_time:.1f} s "
          f"(avg {total_time / len(results):.1f} s/variable)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §4  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", default="fourier_batch_bayes.json")
    args = ap.parse_args()

    t_jh, merged = load_all_data(args.data_dir)

    print(f"\nFitting {len(FS1.VARS)} variables with Bayesian Ridge …")
    print(f"(estimated {len(FS1.VARS) * 35 / 60:.1f} minutes)\n")

    results = []
    t_all = time.perf_counter()

    for i, (vk, vd) in enumerate(FS1.VARS.items(), 1):
        col = vd['col']
        if col not in merged.columns:
            print(f"  [{i:>2}/{len(FS1.VARS)}] {vk:<8} — missing, skip")
            continue

        x = merged[col].values.astype(np.float64) * vd['scale']

        print(f"  [{i:>2}/{len(FS1.VARS)}] {vk:<8} {vd['sym']:<6} "
              f"fitting …", end=" ", flush=True)

        r = analyze_bayes(vk, vd, t_jh, x, args.sigma)
        results.append(r)

        print(f"{r['time_s']:>5.1f}s  R²={r['R2_full']:.4f}  "
              f"sig={r['n_sig_total']:>3}  "
              f"σ_post={r['sigma_posterior']:.6f}")

    total = time.perf_counter() - t_all
    print(f"\n  All done in {total:.1f}s.")

    print_summary(results, args.sigma)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer)
                  else str(o))
    print(f"  JSON → {args.out}\n")


if __name__ == "__main__":
    main()