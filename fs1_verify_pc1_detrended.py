#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_verify_pc1_detrended.py
================================================================================
PC1 verification with seasonal detrending.

Rationale
---------
The previous test (split by raw G) conflates two drivers:

    (a) seasonal cycle      — June vs January
    (b) day-to-day weather  — clear vs cloudy within the same month

To isolate (b), we compute for every day the anomaly of G from its
monthly climatology:

    ΔG[d] = G[d] − mean{ G[all days in same calendar month] }

Then we split days by ΔG (top / bottom quartile) and compare the other
variables.  Any systematic shift in the low-ΔG vs high-ΔG groups cannot
be caused by the seasonal cycle — it is a pure day-to-day effect.

Data coverage note
------------------
Same as fs1_verify_pc1: Wc (TCWV) is excluded from the analysis list
because it is only available from ~Sept 2024 in the IFS-HRES record.

Author : MJS · Jolotundo Observatory
Version: 1.1.0
================================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

import FS1_FourierSeries as FS1
from fs1_verify_pc1 import load_merged, build_daily, cohens_d


# ══════════════════════════════════════════════════════════════════════════════
# §1  SEASONAL DETRENDING
# ══════════════════════════════════════════════════════════════════════════════

def detrend_seasonal(daily: pd.DataFrame,
                     vars_to_detrend: List[str] = None) -> pd.DataFrame:
    """
    Remove the seasonal climatology from each variable.

    For each variable X and each calendar month m ∈ {1..12}:

        climatology[X, m]  =  mean of X over all days whose month == m
        anomaly[X, t]      =  X[t] − climatology[X, month(t)]

    This is the standard "monthly anomaly" method used in climate science.
    It removes the annual cycle while preserving day-to-day variability.

    Parameters
    ----------
    daily : pd.DataFrame
        Daily means, indexed by date.
    vars_to_detrend : list of str, optional
        Columns to detrend.  Default: all numeric columns.

    Returns
    -------
    pd.DataFrame
        Same shape and index as `daily`, all columns detrended.
    """
    if vars_to_detrend is None:
        vars_to_detrend = daily.select_dtypes(include=[np.number]).columns.tolist()

    months = daily.index.month
    anomalies = pd.DataFrame(index=daily.index)

    for v in vars_to_detrend:
        if v not in daily.columns:
            continue
        x = daily[v].values.astype(np.float64)
        clim = np.zeros(13, dtype=np.float64)      # index 1..12
        for m in range(1, 13):
            mask = (months == m)
            if mask.any():
                clim[m] = np.nanmean(x[mask])
        anomalies[v] = x - clim[months]

    return anomalies


# ══════════════════════════════════════════════════════════════════════════════
# §2  SPLIT & COMPARE (same as previous, but on anomalies)
# ══════════════════════════════════════════════════════════════════════════════

def quartile_split_anomaly(anom: pd.DataFrame, key: str = "G",
                           q: float = 0.25) -> Dict[str, pd.DataFrame]:
    v = anom[key].values
    q_low  = np.quantile(v, q)
    q_high = np.quantile(v, 1.0 - q)
    return {
        "low":  anom[anom[key] <= q_low],
        "mid":  anom[(anom[key] > q_low) & (anom[key] < q_high)],
        "high": anom[anom[key] >= q_high],
    }


def compare_groups(low: pd.DataFrame, high: pd.DataFrame,
                   vars_to_check: List[str]) -> pd.DataFrame:
    rows = []
    for v in vars_to_check:
        if v not in low.columns or v not in high.columns:
            continue
        a = low[v].dropna().values
        b = high[v].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue

        d = cohens_d(b, a)
        direction = "↑" if d > 0.2 else ("↓" if d < -0.2 else "·")

        rows.append({
            "var":        v,
            "mean_low":   a.mean(),
            "mean_high":  b.mean(),
            "diff":       b.mean() - a.mean(),
            "sigma":      math.sqrt(((len(a)-1)*a.var(ddof=1) +
                                     (len(b)-1)*b.var(ddof=1)) /
                                    (len(a) + len(b) - 2)),
            "cohens_d":   d,
            "direction":  direction,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# §3  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, n_low: int, n_high: int,
                 dG_low: float, dG_high: float):
    W = 94
    print("=" * W)
    print("  PC1 VERIFICATION — DETRENDED (seasonal cycle removed)")
    print("=" * W)

    if df.empty:
        print("\n  [No variable had sufficient data in both groups.]")
        return

    print(f"\n  Split by G anomaly  (ΔG = G − monthly climatology)")
    print(f"    low-ΔG  days :  {n_low:>5}   mean ΔG = {dG_low:>+8.2f} W/m²")
    print(f"    high-ΔG days :  {n_high:>5}   mean ΔG = {dG_high:>+8.2f} W/m²")
    print(f"    spread between groups:  {dG_high - dG_low:>8.2f} W/m²")

    print(f"\n  {'var':>6}  {'low':>12}  {'high':>12}  {'diff':>12}  "
          f"{'Cohen d':>9}  {'dir':>4}  {'interpretation':>16}")
    print("  " + "─" * (W - 4))

    df_sorted = df.sort_values("cohens_d", key=lambda s: -s.abs())

    for _, r in df_sorted.iterrows():
        d = r["cohens_d"]
        if abs(d) >= 0.8:      interp = "large"
        elif abs(d) >= 0.5:    interp = "medium"
        elif abs(d) >= 0.2:    interp = "small"
        else:                  interp = "negligible"

        print(f"  {r['var']:>6}  "
              f"{r['mean_low']:>12.4f}  "
              f"{r['mean_high']:>12.4f}  "
              f"{r['diff']:>+12.4f}  "
              f"{d:>+9.3f}  "
              f"{r['direction']:>4}  "
              f"{interp:>16}")

    print("\n  Legend: ↑ = high-ΔG mean larger   ↓ = high-ΔG mean smaller")


def print_hypothesis_check(result: pd.DataFrame):
    print(f"\n  {'─' * (94 - 2)}")
    print(f"  PC1 HYPOTHESIS CHECK (detrended)")
    print(f"  {'─' * (94 - 2)}")
    print(f"  Same expected signs as before:")
    print(f"    ↑ T₂, e_d, ET₀, S_h     (Cluster 2 — radiative)")
    print(f"    ↓ T_d, U₂, θ, N_c       (Cluster 1 — humid)")
    print()

    if result.empty:
        print("  [No comparison possible — not enough data.]")
        return

    expected_sign = {
        "T2": +1, "VPD": +1, "ET0": +1, "Sh": +1,
        "Td": -1, "U2": -1, "SM07": -1, "SM728": -1, "Nc": -1,
    }
    correct = 0
    total = 0
    for v, sign in expected_sign.items():
        row = result[result["var"] == v]
        if row.empty:
            continue
        d = row["cohens_d"].values[0]
        total += 1
        ok = (sign * d > 0.2)
        if ok:
            correct += 1
            print(f"    ✓ {v:>6}  d = {d:+.3f}")
        else:
            print(f"    ✗ {v:>6}  d = {d:+.3f}   (expected sign {sign:+d})")

    print(f"\n  → {correct} / {total} variables match the PC1 hypothesis "
          f"(detrended)")


# ══════════════════════════════════════════════════════════════════════════════
# §4  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading merged hourly data …", flush=True)
    df = load_merged()
    print(f"  {len(df):,} hourly records")

    print("Aggregating to daily means …", flush=True)
    daily = build_daily(df)
    print(f"  {len(daily):,} daily records")

    # ---- Detrend ----
    print("\nRemoving seasonal cycle (monthly climatology) …", flush=True)
    # Wc (TCWV) is excluded — only available from ~Sept 2024.
    vars_all = ["G", "T2", "Td", "U2", "VPD", "ET0", "Nc",
                "SM07", "SM728", "Sh", "P", "v10"]
    anom = detrend_seasonal(daily, vars_all)

    # ---- Split ----
    groups = quartile_split_anomaly(anom, key="G", q=0.25)
    low, high = groups["low"], groups["high"]

    # ---- Compare ----
    result = compare_groups(low, high, vars_all[1:])  # skip G itself

    print_report(result,
                 n_low=len(low), n_high=len(high),
                 dG_low=low["G"].mean(), dG_high=high["G"].mean())

    print_hypothesis_check(result)

    # ---- Comparison table with previous (raw-G) result ----
    print(f"\n  {'─' * (94 - 2)}")
    print(f"  COMPARISON: RAW vs DETRENDED COHEN'S d")
    print(f"  {'─' * (94 - 2)}")
    print(f"  {'var':>6}  {'raw d':>10}  {'detrended d':>12}  {'ratio':>8}")
    print("  " + "─" * (94 - 4))

    # Hardcoded raw values from a previous 2017–2026 run for reference.
    raw_d = {
        "ET0": +4.482, "U2": -3.082, "VPD": +3.063,
        "Sh": +2.445, "T2": +2.316, "SM07": -2.203, "Td": -1.999,
        "SM728": -1.990, "Nc": -1.877, "P": -1.771, "v10": +0.273,
    }

    for v in ["ET0", "VPD", "U2", "Sh", "T2", "SM07", "Td",
              "SM728", "Nc", "P", "v10"]:
        row = result[result["var"] == v]
        if row.empty:
            continue
        d_det = row["cohens_d"].values[0]
        d_raw = raw_d.get(v, float("nan"))
        ratio = d_det / d_raw if abs(d_raw) > 1e-9 else float("nan")
        print(f"  {v:>6}  {d_raw:>+10.3f}  {d_det:>+12.3f}  {ratio:>8.2f}")


if __name__ == "__main__":
    main()