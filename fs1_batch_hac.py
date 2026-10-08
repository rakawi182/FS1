#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_batch_hac.py
================================================================================
Regenerate the batch coefficient database with HAC (Newey-West) uncertainties.

Motivation
----------
The original fourier_batch_bayes.json stored σ from Bayesian Ridge, which
assumes i.i.d. residuals. For hourly tropical data, ρ₁ ≈ 0.32–0.998 and this
assumption fails badly. The HAC run (fs1_hac_verify.py) showed:

    • σ inflation factor: median 3.46×, range 1.35×–5.40×
    • The inflation is NOT uniform across frequency blocks:
         annual  → inflates (low-frequency red noise)
         diurnal → often deflates (high-frequency white noise)
      Bayesian Ridge cannot capture this because it fits a single α_noise
      for the entire spectrum.

Worse, the "representative σ" used to count significant harmonics in
FS1_Batch_Bayes.py was `std_ann[1]` — an annual σ applied uniformly to
diurnal and cross terms. That miscounts significance in both directions.

This script
-----------
    • computes per-term σ via Newey-West HAC (Bartlett kernel, default L=72 h)
    • counts significant terms with the correct per-term σ
    • preserves BR σ side-by-side for comparison
    • writes a JSON with the same top-level schema as fourier_batch_bayes.json
      so fs1_multivariate.py can consume either file unchanged

Runtime
-------
~40 s per variable × 14 ≈ 10 min on Android/Termux.

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
from fs1_hac_verify import (
    hac_covariance,
    sigma_per_amplitude,
    collect_pairs,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1  PER-VARIABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_full(vk: str, vd: Dict,
                 t_jh: np.ndarray, x: np.ndarray,
                 maxlags: int = 72,
                 sigma_threshold: float = 2.0) -> Dict:
    """
    Fit one variable with OLS, compute HAC per-term σ, and preserve
    Bayesian Ridge per-term σ for comparison.

    Returns a record matching the fourier_batch_bayes.json schema, with
    additional `*_hac` fields and per-term `sigma_hac` / `A_over_sigma_hac`.
    """
    t0 = time.perf_counter()

    # ---- 1. OLS via FS1 (chunked, low memory) ----
    theta_ols, _, _ = FS1.fit_variable(
        t_jh, x,
        FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS,
    )
    mean_val = float(theta_ols[0])

    # ---- 2. Full design matrix + residuals ----
    valid = np.isfinite(x)
    t_v, x_v = t_jh[valid], x[valid]
    Phi = build_full_design(t_v, FS1.N_ANNUAL, FS1.N_DIURNAL,
                            FS1.CROSS_TERMS)

    resid = x_v - Phi @ theta_ols
    sigma_resid = float(np.std(resid, ddof=0))
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])

    # ---- 3. HAC covariance ----
    V_hac = hac_covariance(Phi, resid, maxlags=maxlags)

    # ---- 4. Bayesian Ridge for comparison ----
    theta_br, theta_std_br, info_br = fit_bayesian(t_jh, x)

    # ---- 5. Column pairs + per-amplitude σ ----
    ann_pairs, dia_pairs, cross_pairs = collect_pairs(
        FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS
    )

    def br_sigma(pairs):
        return [math.hypot(theta_std_br[i], theta_std_br[i + 1])
                for (i, _) in pairs]

    sig_ann_br   = br_sigma(ann_pairs)
    sig_dia_br   = br_sigma(dia_pairs)
    sig_cross_br = br_sigma(cross_pairs)

    sig_ann_hac   = sigma_per_amplitude(theta_ols, V_hac, ann_pairs)
    sig_dia_hac   = sigma_per_amplitude(theta_ols, V_hac, dia_pairs)
    sig_cross_hac = sigma_per_amplitude(theta_ols, V_hac, cross_pairs)

    # ---- 6. Amplitudes + phases ----
    ann  = FS1.extract_annual(theta_ols, FS1.N_ANNUAL)
    dia  = FS1.extract_diurnal(theta_ols, FS1.N_ANNUAL, FS1.N_DIURNAL)
    cros = FS1.extract_cross(theta_ols, FS1.N_ANNUAL, FS1.N_DIURNAL,
                              FS1.CROSS_TERMS)

    # ---- 7. Nested R² ----
    fit = FS1.incremental_r2(
        t_jh, x, theta_ols,
        FS1.N_ANNUAL, FS1.N_DIURNAL, FS1.CROSS_TERMS,
    )

    # ---- 8. Build per-term records ----
    def make_term(term, key_names, s_br, s_hac):
        rec = {k: term[k] for k in key_names}
        rec["A"]   = term["A"]
        rec["phi"] = term["phi"]
        rec["sigma_br"]  = s_br
        rec["sigma_hac"] = s_hac
        rec["A_over_sigma_br"]  = term["A"] / s_br  if s_br  > 0 else float("inf")
        rec["A_over_sigma_hac"] = term["A"] / s_hac if s_hac > 0 else float("inf")
        rec["sig_br"]  = bool(term["A"] > sigma_threshold * s_br)
        rec["sig_hac"] = bool(term["A"] > sigma_threshold * s_hac)
        return rec

    ann_terms   = [make_term(t, ("k",),   sb, sh)
                   for t, sb, sh in zip(ann,  sig_ann_br,   sig_ann_hac)]
    dia_terms   = [make_term(t, ("m",),   sb, sh)
                   for t, sb, sh in zip(dia,  sig_dia_br,   sig_dia_hac)]
    cross_terms = [make_term(t, ("k", "m"), sb, sh)
                   for t, sb, sh in zip(cros, sig_cross_br, sig_cross_hac)]

    # ---- 9. Significance counts ----
    n_sig_ann_hac   = sum(1 for r in ann_terms   if r["sig_hac"])
    n_sig_dia_hac   = sum(1 for r in dia_terms   if r["sig_hac"])
    n_sig_cross_hac = sum(1 for r in cross_terms if r["sig_hac"])

    n_sig_ann_br   = sum(1 for r in ann_terms   if r["sig_br"])
    n_sig_dia_br   = sum(1 for r in dia_terms   if r["sig_br"])
    n_sig_cross_br = sum(1 for r in cross_terms if r["sig_br"])

    # Representative σ (annual k = 1)
    sig_rep_br  = sig_ann_br[0]  if sig_ann_br  else float("nan")
    sig_rep_hac = sig_ann_hac[0] if sig_ann_hac else float("nan")

    return {
        # ---- identity ----
        "vk":   vk,
        "name": vd["name"],
        "sym":  vd["sym"],
        "unit": vd["unit"],

        # ---- provenance ----
        "N": int(valid.sum()),
        "P": int(Phi.shape[1]),
        "maxlags": maxlags,
        "sigma_threshold": sigma_threshold,
        "rho1": rho1,

        # ---- DC + fit quality ----
        "mean": mean_val,
        "sigma_resid": sigma_resid,
        "rmse_full":   fit["rmse_full"],

        # ---- representative uncertainties ----
        "sigma_posterior_br": sig_rep_br,
        "sigma_hac_rep":      sig_rep_hac,
        "sigma_hac_over_br":  sig_rep_hac / sig_rep_br
                              if sig_rep_br > 0 else float("nan"),

        # ---- BR hyperparameters (kept for provenance) ----
        "noise_precision":  info_br["noise_precision"],
        "weight_precision": info_br["weight_precision"],
        "sigma_noise":      info_br["sigma_noise"],
        "sigma_weights":    info_br["sigma_weights"],

        # ---- nested R² ----
        "R2_annual":  fit["r2_annual"],
        "R2_diurnal": fit["r2_diurnal"],
        "R2_full":    fit["r2_full"],

        # ---- first-harmonic amplitudes (convenience) ----
        "A_ann1":   ann_terms[0]["A"]   if ann_terms   else 0.0,
        "A_dia1":   dia_terms[0]["A"]   if dia_terms   else 0.0,
        "A_cross1": cross_terms[0]["A"] if cross_terms else 0.0,

        # ---- significance counts: HAC (canonical) ----
        "n_sig_annual_hac":  n_sig_ann_hac,
        "n_sig_diurnal_hac": n_sig_dia_hac,
        "n_sig_cross_hac":   n_sig_cross_hac,
        "n_sig_total_hac":   n_sig_ann_hac + n_sig_dia_hac + n_sig_cross_hac,

        # ---- significance counts: BR (reference) ----
        "n_sig_annual_br":  n_sig_ann_br,
        "n_sig_diurnal_br": n_sig_dia_br,
        "n_sig_cross_br":   n_sig_cross_br,
        "n_sig_total_br":   n_sig_ann_br + n_sig_dia_br + n_sig_cross_br,

        # ---- block sizes ----
        "n_annual":  FS1.N_ANNUAL,
        "n_diurnal": FS1.N_DIURNAL,
        "n_cross":   len(FS1.CROSS_TERMS),

        # ---- per-term coefficients ----
        "annual_terms":  ann_terms,
        "diurnal_terms": dia_terms,
        "cross_terms":   cross_terms,

        "time_s": time.perf_counter() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §2  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict], maxlags: int, threshold: float):
    W = 126
    print()
    print("=" * W)
    print(f"  BATCH HAC SUMMARY  "
          f"(Newey-West, L = {maxlags} h, threshold = {threshold}σ)")
    print("=" * W)

    print(f"\n  {'var':<8}{'sym':<7}{'ρ₁':>7}"
          f"{'σ_BR':>13}{'σ_HAC':>13}{'ratio':>8}"
          f"   {'ann':>10}{'dia':>10}{'cross':>10}{'total':>10}")
    print("  " + "─" * (W - 4))

    for r in results:
        n_a, n_d, n_c = r["n_annual"], r["n_diurnal"], r["n_cross"]
        print(f"  {r['vk']:<8}{r['sym']:<7}"
              f"{r['rho1']:>7.3f}"
              f"{r['sigma_posterior_br']:>13.6f}"
              f"{r['sigma_hac_rep']:>13.6f}"
              f"{r['sigma_hac_over_br']:>7.2f}×"
              f"   "
              f"{r['n_sig_annual_hac']:>3}/{n_a:<3} "
              f"{r['n_sig_diurnal_hac']:>3}/{n_d:<3} "
              f"{r['n_sig_cross_hac']:>3}/{n_c:<3} "
              f"{r['n_sig_total_hac']:>4}/{r['n_sig_total_hac'] + (n_a + n_d + n_c) - r['n_sig_total_hac']:<4}")

    # ---- Comparison table: BR vs HAC significance ----
    print(f"\n  SIGNIFICANCE COUNT SHIFT (BR → HAC)")
    print(f"  {'var':<8}{'annual':>14}{'diurnal':>14}{'cross':>14}{'total':>14}")
    print("  " + "─" * (W - 4))
    for r in results:
        def fmt(br, hac):
            d = hac - br
            sign = "+" if d > 0 else ""
            return f"{br:>3} → {hac:>3} ({sign}{d})"
        print(f"  {r['vk']:<8}"
              f"{fmt(r['n_sig_annual_br'],  r['n_sig_annual_hac']):>14}"
              f"{fmt(r['n_sig_diurnal_br'], r['n_sig_diurnal_hac']):>14}"
              f"{fmt(r['n_sig_cross_br'],   r['n_sig_cross_hac']):>14}"
              f"{fmt(r['n_sig_total_br'],   r['n_sig_total_hac']):>14}")

    ratios = [r["sigma_hac_over_br"] for r in results
              if np.isfinite(r["sigma_hac_over_br"])]
    if ratios:
        print(f"\n  σ inflation factor: "
              f"median = {np.median(ratios):.2f}×   "
              f"range = {min(ratios):.2f}× – {max(ratios):.2f}×")

    total_time = sum(r["time_s"] for r in results)
    print(f"\n  Total runtime: {total_time:.1f} s "
          f"(avg {total_time/len(results):.1f} s/variable)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §3  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Batch HAC — regenerate coefficient DB with "
                    "Newey-West uncertainties")
    ap.add_argument("--maxlags", type=int, default=72,
                    help="HAC bandwidth in hours (default 72)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Significance threshold in σ units (default 2)")
    ap.add_argument("--vars", default="",
                    help="Comma-separated variables (default: all)")
    ap.add_argument("--out", default="fourier_batch_hac.json")
    args = ap.parse_args()

    print("=" * 78)
    print("  FS1 BATCH HAC — coefficient DB with Newey-West uncertainties")
    print("=" * 78)

    print("\nLoading merged hourly data …", flush=True)
    merged = load_merged()
    print(f"  {len(merged):,} hourly records")

    t_jh = FS1.datetime_to_j2000h(merged.index)

    var_keys = (
        [v.strip() for v in args.vars.split(",") if v.strip()]
        or list(FS1.VARS.keys())
    )

    print(f"\nFitting {len(var_keys)} variables "
          f"(~{len(var_keys) * 40 / 60:.0f} min) …\n")

    results: List[Dict] = []
    t_all = time.perf_counter()

    for i, vk in enumerate(var_keys, 1):
        vd = FS1.VARS[vk]
        col = vd["col"]
        if col not in merged.columns:
            print(f"  [{i:>2}/{len(var_keys)}] {vk:<7} — column missing, skip")
            continue

        x = merged[col].values.astype(np.float64) * vd["scale"]

        print(f"  [{i:>2}/{len(var_keys)}] {vk:<7} {vd['sym']:<6} fitting …",
              end=" ", flush=True)

        r = analyze_full(vk, vd, t_jh, x,
                         maxlags=args.maxlags,
                         sigma_threshold=args.sigma)
        results.append(r)

        print(f"{r['time_s']:>5.1f}s   "
              f"ρ₁={r['rho1']:>5.3f}   "
              f"σ×{r['sigma_hac_over_br']:>4.2f}   "
              f"sig {r['n_sig_total_br']:>2} → {r['n_sig_total_hac']:>2}")

    total = time.perf_counter() - t_all
    print(f"\n  All done in {total:.1f}s.")

    print_summary(results, args.maxlags, args.sigma)

    # ---- JSON export (list schema, compatible with fourier_batch_bayes.json) ----
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return str(o)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=_ser)

    import os
    print(f"  JSON → {args.out}  "
          f"({os.path.getsize(args.out) // 1024} KB)\n")


if __name__ == "__main__":
    main()