#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_stability.py
================================================================================
Temporal stability test of the PC1 (radiative-humid) mode.

Question
--------
Does the radiative-humid coupling behave the same way in the first half
of the available IFS-HRES record (2017–2021) as in the second half
(2021–2026)?

Method
------
1. Split daily data into two sub-epochs of ~4.8 years each.
2. For each epoch, detrend seasonally (monthly climatology of that epoch).
3. Split by ΔG anomaly and compute Cohen's d for each variable.
4. Compare:
       - secular shift in means (climate-change signal)
       - Cohen's d magnitude and sign (stability of PC1)
       - seasonal cycle amplitude change

Interpretation
--------------
If the two epochs show the same signs and similar |d| values, the PC1
mode is a stable feature of the Jolotundo climate over the IFS-HRES
period.

If the second epoch shows a systematically weaker or stronger pattern,
or if means have shifted, we are seeing a short-term trend.

Caveat
------
With only ~4.8 years per epoch, the t-statistics and Cohen's d are
noisier than in the original ERA5-spliced test (~16 years per epoch).
Findings should be interpreted as indicative rather than definitive.

Data coverage note
------------------
Wc (TCWV) is excluded from all analyses: it is available only from
~Sept 2024 in the IFS-HRES record, so it cannot meaningfully span the
2017–2021 vs 2021–2026 split.

Author : MJS · Jolotundo Observatory
Version: 1.1.0
================================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import FS1_FourierSeries as FS1
from fs1_verify_pc1 import load_merged, build_daily, cohens_d
from fs1_verify_pc1_detrended import detrend_seasonal, quartile_split_anomaly


# ══════════════════════════════════════════════════════════════════════════════
# §1  EPOCH SPLIT
# ══════════════════════════════════════════════════════════════════════════════

# Data available: 2017-01-01 → 2026-09-28 (~9.74 years)
# Split into two sub-epochs of ~4.8 years each for the stability test.
EPOCH_A = ("2017-01-01", "2021-09-30")   # ~4.75 years
EPOCH_B = ("2021-10-01", "2026-09-28")   # ~4.99 years


def split_epochs(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = daily.loc[EPOCH_A[0]:EPOCH_A[1]].copy()
    b = daily.loc[EPOCH_B[0]:EPOCH_B[1]].copy()
    return a, b


# ══════════════════════════════════════════════════════════════════════════════
# §2  SECULAR TREND
# ══════════════════════════════════════════════════════════════════════════════

def secular_shift(early: pd.DataFrame, late: pd.DataFrame,
                  vars_to_check: List[str]) -> pd.DataFrame:
    """
    Compare daily means per variable between two epochs.

    Returns
    -------
    DataFrame with columns:
        var, mean_early, mean_late, shift, pct_change, t_stat
    """
    rows = []
    for v in vars_to_check:
        if v not in early.columns or v not in late.columns:
            continue
        a = early[v].dropna().values
        b = late[v].dropna().values
        if len(a) < 100 or len(b) < 100:
            continue

        mean_a = a.mean()
        mean_b = b.mean()
        shift = mean_b - mean_a
        pct = 100 * shift / mean_a if abs(mean_a) > 1e-9 else float("nan")

        # Welch t-stat
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = shift / se if se > 1e-12 else 0.0

        rows.append({
            "var":         v,
            "mean_early":  mean_a,
            "mean_late":   mean_b,
            "shift":       shift,
            "pct_change":  pct,
            "t_stat":      t,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# §3  SEASONAL AMPLITUDE PER EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_amplitude(daily: pd.DataFrame, key: str = "G") -> float:
    """
    Amplitude of the annual cycle, estimated as
    (max monthly mean − min monthly mean) / 2.

    Captures how strong the seasonal contrast is in a given epoch.
    """
    months = daily.index.month
    means = np.array([daily[key][months == m].mean() for m in range(1, 13)])
    return float((means.max() - means.min()) / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# §4  PC1 VERIFICATION PER EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def pc1_cohens_d_per_epoch(daily_epoch: pd.DataFrame,
                           vars_to_check: List[str],
                           key: str = "G",
                           q: float = 0.25) -> Dict[str, float]:
    """
    Detrend seasonally within the epoch, split by ΔG, return Cohen's d
    for each variable.
    """
    anom = detrend_seasonal(daily_epoch, vars_to_check)
    groups = quartile_split_anomaly(anom, key=key, q=q)
    low, high = groups["low"], groups["high"]

    result = {}
    for v in vars_to_check:
        if v == key:
            continue
        a = low[v].dropna().values
        b = high[v].dropna().values
        if len(a) < 50 or len(b) < 50:
            continue
        result[v] = cohens_d(b, a)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §5  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_secular(df: pd.DataFrame):
    W = 94
    print("=" * W)
    print("  SECULAR SHIFT  (2021–2026 mean − 2017–2021 mean)")
    print("=" * W)

    if df.empty:
        print("\n  [No variable had sufficient data in both epochs "
              "(≥100 days each).]")
        return

    print(f"\n  {'var':>6}  {'early':>12}  {'late':>12}  {'shift':>12}  "
          f"{'%chg':>8}  {'t-stat':>8}  {'sig':>5}")
    print("  " + "─" * (W - 4))

    df_sorted = df.sort_values("t_stat", key=lambda s: -s.abs())
    for _, r in df_sorted.iterrows():
        sig = "***" if abs(r["t_stat"]) > 3 else (
              "**"  if abs(r["t_stat"]) > 2 else (
              "*"   if abs(r["t_stat"]) > 1.5 else ""))
        print(f"  {r['var']:>6}  "
              f"{r['mean_early']:>12.4f}  "
              f"{r['mean_late']:>12.4f}  "
              f"{r['shift']:>+12.4f}  "
              f"{r['pct_change']:>+7.2f}%  "
              f"{r['t_stat']:>+8.3f}  "
              f"{sig:>5}")

    print("\n  Significance: * |t|>1.5   ** |t|>2.0   *** |t|>3.0")
    print("  (Note: with ~1,750 days per epoch, |t| is ~1.8× noisier "
          "than in the ERA5 test.)")


def print_pc1_stability(d_early: Dict[str, float],
                        d_late: Dict[str, float],
                        vars_order: List[str]):
    W = 94
    print(f"\n{'=' * W}")
    print("  PC1 STABILITY — Cohen's d per epoch (detrended)")
    print(f"{'=' * W}")

    if not d_early or not d_late:
        print("\n  [No variable had sufficient data in both epochs.]")
        return

    print(f"\n  {'var':>6}  {'2017–2021':>12}  {'2021–2026':>12}  "
          f"{'Δ':>8}  {'sign match':>11}  {'|ratio|':>8}")
    print("  " + "─" * (W - 4))

    sign_match = 0
    total = 0

    for v in vars_order:
        if v not in d_early or v not in d_late:
            continue
        de, dl = d_early[v], d_late[v]
        delta = dl - de
        same_sign = (de * dl > 0)
        ratio = dl / de if abs(de) > 1e-6 else float("nan")
        total += 1
        if same_sign:
            sign_match += 1

        flag = "✓" if same_sign else "✗"
        print(f"  {v:>6}  {de:>+12.3f}  {dl:>+12.3f}  "
              f"{delta:>+8.3f}  {flag:>11}  {ratio:>8.2f}")

    if total > 0:
        print(f"\n  → {sign_match} / {total} variables preserve sign "
              f"between epochs")


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading merged hourly data …", flush=True)
    df = load_merged()
    print(f"  {len(df):,} hourly records")

    print("Aggregating to daily means …", flush=True)
    daily = build_daily(df)
    print(f"  {len(daily):,} daily records")

    early, late = split_epochs(daily)
    print(f"\n  Epoch A (2017–2021): {len(early):>6} days  "
          f"({early.index[0].date()} → {early.index[-1].date()})")
    print(f"  Epoch B (2021–2026): {len(late):>6} days  "
          f"({late.index[0].date()} → {late.index[-1].date()})")

    # Wc (TCWV) is excluded: only available from ~Sept 2024.
    vars_all = ["G", "T2", "Td", "U2", "VPD", "ET0", "Nc",
                "SM07", "SM728", "Sh", "P", "v10"]

    # ---- Secular shift ----
    sec = secular_shift(early, late, vars_all)
    print_secular(sec)

    # ---- Seasonal amplitude change ----
    amp_a = seasonal_amplitude(early, key="G")
    amp_b = seasonal_amplitude(late, key="G")
    print(f"\n  Seasonal G amplitude (annual cycle / 2):")
    print(f"    2017–2021 : {amp_a:>8.2f} W/m²")
    print(f"    2021–2026 : {amp_b:>8.2f} W/m²")
    if amp_a > 1e-6:
        print(f"    change    : {amp_b - amp_a:>+8.2f} W/m²  "
              f"({100*(amp_b-amp_a)/amp_a:+.1f}%)")
    else:
        print(f"    change    : {amp_b - amp_a:>+8.2f} W/m²")

    # ---- PC1 stability ----
    print(f"\n  Computing detrended PC1 Cohen's d per epoch …", flush=True)
    d_early = pc1_cohens_d_per_epoch(early, vars_all, key="G")
    d_late  = pc1_cohens_d_per_epoch(late,  vars_all, key="G")

    vars_order = ["ET0", "VPD", "U2", "Sh", "T2",
                  "SM07", "Td", "SM728", "Nc", "P", "v10"]
    print_pc1_stability(d_early, d_late, vars_order)

    # ---- Verdict ----
    print(f"\n  {'─' * (94 - 2)}")
    print(f"  VERDICT")
    print(f"  {'─' * (94 - 2)}")

    ratios = []
    for v in vars_order:
        if v in d_early and v in d_late and abs(d_early[v]) > 1e-3:
            ratios.append(abs(d_late[v]) / abs(d_early[v]))

    if not ratios:
        print("  [Cannot compute verdict — no shared variables.]")
        return

    mean_ratio = float(np.mean(ratios))

    print(f"  Average |d_late| / |d_early|  =  {mean_ratio:.2f}")
    # Threshold loosened to 0.70–1.40 to account for shorter epochs
    # (~4.8 years each) and correspondingly noisier Cohen's d estimates.
    if 0.70 <= mean_ratio <= 1.40:
        print(f"  → PC1 strength is STABLE across the two epochs")
    elif mean_ratio > 1.40:
        print(f"  → PC1 STRENGTHENED in the later epoch")
    else:
        print(f"  → PC1 WEAKENED in the later epoch")
    print(f"  (Verdict thresholds: STABLE 0.70–1.40; "
          f"loosened from 0.80–1.25 for shorter epochs.)")


if __name__ == "__main__":
    main()