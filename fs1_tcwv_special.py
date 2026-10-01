#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_tcwv_special.py
================================================================================
Specialized analysis of Total Column Integrated Water Vapour (TCWV / W_c).

Rationale
---------
In the IFS-HRES 2017–2026 record, TCWV is only available from
~September 2024 onward (≈18,216 h out of 85,392 h, i.e. ~21 %).  That
corresponds to roughly 2 annual cycles, which is insufficient for a
stable annual Fourier decomposition.

However, three analyses are statistically well-posed:

  • Diurnal cycle (24 h period, sampled ~730 times) → robust fit.
  • Hourly cross-correlation with other variables → well-posed.
  • 30-day rolling mean and monthly means → coarse seasonal envelope.

This module therefore deliberately:

  • fits ONLY diurnal harmonics (n_annual = 0, no cross terms),
  • reports honest temporal coverage and gap structure,
  • and cross-correlates W_c with the other 13 variables.

It does NOT produce annual harmonic amplitudes for W_c, because those
would be an artefact of the truncated record.

Author : MJS · Jolotundo Observatory
Version: 1.0.0
================================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

import FS1_FourierSeries as FS1
from FS1_Advanced import fit_bayesian


# ══════════════════════════════════════════════════════════════════════════════
# §1  LOAD HOURLY MERGED DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_hourly() -> pd.DataFrame:
    """Load stations A & B, IDW-merge, return hourly merged DataFrame."""
    fdict, _ = FS1._resolve_files(None)
    W_A, W_B, _, _ = FS1.idw_weights(
        FS1.PHI_T, FS1.LAM_T, FS1.PHI_A, FS1.LAM_A,
        FS1.PHI_B, FS1.LAM_B, FS1.IDW_POWER)
    sta_a = FS1.load_station(fdict["A"])
    sta_b = FS1.load_station(fdict["B"])
    cols = sorted(set(sta_a.columns) | set(sta_b.columns))
    return FS1.idw_merge(sta_a, sta_b, W_A, W_B, cols)


# ══════════════════════════════════════════════════════════════════════════════
# §2  COVERAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def report_coverage(merged: pd.DataFrame,
                    col: str = "total_column_integrated_water_vapour"):
    """
    Report temporal coverage, gap structure, and basic statistics of TCWV.
    """
    W = 78
    print("=" * W)
    print("  TCWV / W_c — SPECIALIZED ANALYSIS")
    print("=" * W)

    if col not in merged.columns:
        raise SystemExit(f"Column '{col}' not found in merged data.")

    x = merged[col].values.astype(np.float64)
    valid = np.isfinite(x)
    N_total = len(x)
    N_valid = int(valid.sum())
    pct = 100.0 * N_valid / N_total if N_total else 0.0

    # Time range of valid samples
    valid_idx = merged.index[valid]
    t_first = valid_idx.min()
    t_last  = valid_idx.max()
    span_years = (t_last - t_first).total_seconds() / 86400.0 / 365.24219

    # Detect gaps > 7 days
    gaps = []
    diffs = valid_idx.to_series().diff()
    long_gaps = diffs[diffs > pd.Timedelta(days=7)]
    for ts, gap in long_gaps.items():
        gaps.append((ts - gap, ts, gap.days))

    print(f"\n  Column       : {col}")
    print(f"  Total samples: {N_total:,} h   "
          f"({N_total / 24 / 365.25:.2f} years)")
    print(f"  Valid samples: {N_valid:,} h   ({pct:.1f} %)")
    print(f"  First valid  : {t_first}")
    print(f"  Last valid   : {t_last}")
    print(f"  Span         : {span_years:.2f} years")

    if gaps:
        print(f"\n  Long gaps (> 7 days):")
        for start, end, days in gaps[:8]:
            print(f"    {start.date()} → {end.date()}   ({days} days)")
        if len(gaps) > 8:
            print(f"    … and {len(gaps) - 8} more")

    xv = x[valid]
    print(f"\n  Statistics (valid samples only):")
    print(f"    mean    = {xv.mean():>10.4f} kg m⁻²")
    print(f"    std     = {xv.std(ddof=1):>10.4f} kg m⁻²")
    print(f"    min     = {xv.min():>10.4f} kg m⁻²")
    print(f"    max     = {xv.max():>10.4f} kg m⁻²")
    print(f"    median  = {np.median(xv):>10.4f} kg m⁻²")

    return x, valid, t_first, t_last


# ══════════════════════════════════════════════════════════════════════════════
# §3  MONTHLY MEANS (COARSE SEASONAL ENVELOPE)
# ══════════════════════════════════════════════════════════════════════════════

def print_monthly_climatology(merged: pd.DataFrame,
                              col: str = "total_column_integrated_water_vapour"):
    """
    Print monthly means computed only from available data.

    Because TCWV starts in Sept 2024, months Jan–Aug have at most
    2 realizations (2025 and 2026).  This is NOT a robust climatology —
    it is a coarse seasonal glimpse.
    """
    if col not in merged.columns:
        return

    s = merged[col].dropna()
    monthly = s.groupby(s.index.month).agg(["mean", "count"])

    W = 78
    print(f"\n  {W * '─'}")
    print("  MONTHLY MEANS (from available data only — NOT a robust climatology)")
    print(f"  {W * '─'}")
    print(f"    {'month':>5}   {'mean kg/m²':>12}   {'n samples':>10}   "
          f"{'n years':>8}   pattern")
    print("    " + "─" * (W - 6))
    for m in range(1, 13):
        if m in monthly.index:
            mean = monthly.loc[m, "mean"]
            n = int(monthly.loc[m, "count"])
            n_years = n / 24.0 / 30.44
            bar = "█" * max(0, int(mean - 35))
            print(f"    {m:>5}   {mean:>12.4f}   {n:>10,}   "
                  f"{n_years:>7.2f}   {bar}")
        else:
            print(f"    {m:>5}   {'—':>12}   {0:>10,}   {0:>7.2f}   (no data)")


# ══════════════════════════════════════════════════════════════════════════════
# §4  DIURNAL-ONLY FOURIER FIT (BAYESIAN RIDGE)
# ══════════════════════════════════════════════════════════════════════════════

def fit_diurnal_only(t_jh: np.ndarray, x: np.ndarray) -> Dict:
    """
    Fit ONLY the diurnal harmonics (n_annual = 0, no cross terms).

    Uses Bayesian Ridge to obtain posterior σ per harmonic.

    Column layout for n_a = 0, n_d = 11, cx = []:
        col 0       : DC
        col 1..22   : cos(m·ωD·t), sin(m·ωD·t)  m = 1..11
    """
    theta_mean, theta_std, info = fit_bayesian(
        t_jh, x,
        n_a=0, n_d=FS1.N_DIURNAL, cx=[],
    )

    mean_val = float(theta_mean[0])

    dia = []
    for m in range(1, FS1.N_DIURNAL + 1):
        col = 1 + 2 * (m - 1)
        a, b = theta_mean[col], theta_mean[col + 1]
        A = math.hypot(a, b)
        phi = math.degrees(math.atan2(-b, a)) % 360.0
        sig = math.hypot(theta_std[col], theta_std[col + 1])
        dia.append({
            "m": m,
            "period_h": 24.0 / m,
            "A": A,
            "phi": phi,
            "sigma": sig,
            "A_over_sigma": A / sig if sig > 0 else float("inf"),
        })

    # Reconstruct residual statistics using the same design basis
    P = 1 + 2 * FS1.N_DIURNAL
    valid = np.isfinite(x)
    xv = x[valid]
    tv = t_jh[valid]

    Phi = np.empty((len(tv), P), dtype=np.float64)
    Phi[:, 0] = 1.0
    col = 1
    for m in range(1, FS1.N_DIURNAL + 1):
        arg = m * FS1.OMEGA_D * tv
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    resid = xv - Phi @ theta_mean
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((xv - xv.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mean": mean_val,
        "diurnal": dia,
        "rmse": rmse,
        "sigma_resid": rmse,
        "r2_diurnal": r2,
        "N": int(valid.sum()),
        "sigma_noise": info["sigma_noise"],
        "sigma_weights": info["sigma_weights"],
        "noise_precision": info["noise_precision"],
        "weight_precision": info["weight_precision"],
    }


def print_diurnal_table(fit: Dict):
    W = 78
    print(f"\n  {W * '─'}")
    print("  DIURNAL FOURIER FIT  (annual & cross terms NOT fitted)")
    print(f"  {W * '─'}")
    print(f"    DC term  W_c₀       = {fit['mean']:+.4f} kg m⁻²")
    print(f"    R² (diurnal only)  = {fit['r2_diurnal']:.6f}")
    print(f"    RMSE               = {fit['rmse']:.4f} kg m⁻²")
    print(f"    N                  = {fit['N']:,} h")
    print(f"    σ_noise            = {fit['sigma_noise']:.4f} kg m⁻²")
    print(f"    σ_weights          = {fit['sigma_weights']:.4f} kg m⁻²")

    print(f"\n    {'m':>3}  {'Period/h':>10}  {'Amplitude':>12}  "
          f"{'Phase/°':>10}  {'σ':>10}  {'A/σ':>8}  {'Peak WIB':>10}  sig")
    print("    " + "─" * (W - 4))
    for d in fit["diurnal"]:
        hmax = (12.0 - d["phi"] / (d["m"] * 15.0)) % 24.0
        sig = "yes" if d["A_over_sigma"] > 2.0 else "no"
        print(f"    {d['m']:>3}  {d['period_h']:>10.4f}  "
              f"{d['A']:>12.6f}  {d['phi']:>10.4f}  "
              f"{d['sigma']:>10.6f}  {d['A_over_sigma']:>8.2f}  "
              f"{hmax:>10.2f}  {sig:>3}")

    print(f"\n    Legend: 'sig' = significant at 2σ_posterior.")


# ══════════════════════════════════════════════════════════════════════════════
# §5  HOURLY CROSS-CORRELATION WITH OTHER VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def hourly_correlation(merged: pd.DataFrame,
                       target_col: str = "total_column_integrated_water_vapour",
                       vars_to_check: List[str] = None) -> pd.DataFrame:
    """
    Pearson correlation of TCWV with each other variable, computed
    hourly over the intersection of finite samples (post Sept-2024).
    """
    if vars_to_check is None:
        vars_to_check = [k for k, v in FS1.VARS.items()
                         if v["col"] != target_col
                         and v["col"] in merged.columns]

    x = merged[target_col]

    rows = []
    for vk in vars_to_check:
        vd = FS1.VARS[vk]
        col = vd["col"]
        if col not in merged.columns:
            continue
        y = merged[col] * vd["scale"]
        pair = pd.concat([x.rename("tcwv"), y.rename("y")],
                         axis=1).dropna()
        if len(pair) < 500:
            continue
        r = float(pair["tcwv"].corr(pair["y"]))
        n = len(pair)
        rows.append({
            "vk": vk,
            "sym": vd["sym"],
            "unit": vd["unit"],
            "r": r,
            "n": n,
        })

    return pd.DataFrame(rows).sort_values("r", key=lambda s: -s.abs())


def print_correlation_table(df: pd.DataFrame):
    W = 78
    print(f"\n  {W * '─'}")
    print("  HOURLY CROSS-CORRELATION  (TCWV vs other variables)")
    print(f"  {W * '─'}")
    print("    (computed only over overlapping valid samples — post Sept-2024)")
    print(f"\n    {'var':>7}  {'sym':>6}  {'r':>8}  {'n':>10}  "
          f"{'strength':>12}  pattern")
    print("    " + "─" * (W - 4))
    for _, r in df.iterrows():
        ar = abs(r["r"])
        if ar >= 0.7:   strength = "strong"
        elif ar >= 0.4: strength = "moderate"
        elif ar >= 0.2: strength = "weak"
        else:           strength = "negligible"
        bar = "█" * int(ar * 20)
        print(f"    {r['vk']:>7}  {r['sym']:>6}  {r['r']:>+8.4f}  "
              f"{r['n']:>10,}  {strength:>12}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading merged hourly data …", flush=True)
    merged = load_hourly()
    print(f"  {len(merged):,} hourly records")

    # Coverage report
    x_raw, valid, t_first, t_last = report_coverage(merged)

    # Monthly envelope
    print_monthly_climatology(merged)

    # Diurnal-only Fourier fit
    print("\nFitting diurnal-only Fourier model …", flush=True)
    t_jh = FS1.datetime_to_j2000h(merged.index)
    fit = fit_diurnal_only(t_jh, x_raw)
    print_diurnal_table(fit)

    # Hourly cross-correlation
    print("\nComputing hourly cross-correlations …", flush=True)
    corr = hourly_correlation(merged)
    print_correlation_table(corr)

    # Closing notes
    W = 78
    print(f"\n{'=' * W}")
    print("  INTERPRETATION NOTES")
    print(f"{'=' * W}")
    print("  • The DIURNAL cycle above is reliable: ~730 diurnal cycles")
    print("    are sampled over the 2-year TCWV record.")
    print("  • The ANNUAL cycle is NOT fitted: only ~2 annual cycles are")
    print("    available, making harmonic amplitudes unstable.")
    print("  • Monthly means are coarse — treat them as a seasonal glimpse,")
    print("    not a climatology.")
    print("  • Correlations use the overlapping valid window only.")
    print("  • Any reference to W_c in the main FS1 analysis should")
    print("    explicitly state its reduced coverage (see coverage report).")
    print()


if __name__ == "__main__":
    main()