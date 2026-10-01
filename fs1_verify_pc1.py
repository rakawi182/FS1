#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_verify_pc1.py
================================================================================
Direct verification of PC1 = "humid ↔ radiative" axis.

Hypothesis
----------
If PC1 is physical, then splitting days by solar radiation (G) should
cleanly separate the other variables:

    High-G days  →  T₂ ↑,  e_d ↑,  ET₀ ↑,  S_h ↑
                    T_d ↓,  U₂ ↓,  θ ↓,  N_c ↓

Effect size is measured with Cohen's d = (μ_high − μ_low) / σ_pooled.

Data coverage note
------------------
The IFS-HRES 2017–2026 record has near-complete coverage for all
variables except TCWV (W_c), which is only available from ~Sept 2024.
Wc is therefore retained as a column in the daily DataFrame (for
optional separate analysis) but is excluded from the main comparison
list to avoid biasing the results.

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


# ══════════════════════════════════════════════════════════════════════════════
# §1  LOAD HOURLY DATA (same as FS1)
# ══════════════════════════════════════════════════════════════════════════════

def load_merged() -> pd.DataFrame:
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
# §2  DAILY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily means.

    Variables aggregated (all in raw physical units):

        G       Global Horizontal Irradiance       W/m²
        T₂      Air temperature                    °C
        T_d     Dew-point temperature              °C
        U₂      Relative humidity                  %
        e_d     Vapour pressure deficit            kPa
        ET₀     Reference evapotranspiration       mm/h
        N_c     Cloud cover                        %
        W_c     Column water vapour                kg/m²   [partial]
        θ₀₇     Soil moisture 0–7 cm               m³/m³
        θ₇₂₈    Soil moisture 7–28 cm              m³/m³
        S_h     Sunshine fraction                  frac
        P       Precipitation                      mm/h
        v₁₀     Wind speed                         m/s

    Row retention policy
    --------------------
    Rows are dropped only when a *near-complete-coverage* variable is
    NaN.  Wc (TCWV) is available only from ~Sept 2024 in the IFS-HRES
    record; including it in the dropna() criterion would eliminate
    ~80 % of days and destroy the multi-year analysis.  Wc is therefore
    kept as a column but excluded from the completeness test.
    """
    cols_wanted = {
        "G":     "shortwave_radiation",
        "T2":    "temperature_2m",
        "Td":    "dew_point_2m",
        "U2":    "relative_humidity_2m",
        "VPD":   "vapour_pressure_deficit",
        "ET0":   "et0_fao_evapotranspiration",
        "Nc":    "cloud_cover",
        "Wc":    "total_column_integrated_water_vapour",
        "SM07":  "soil_moisture_0_to_7cm",
        "SM728": "soil_moisture_7_to_28cm",
        "Sh":    "sunshine_duration",
        "P":     "precipitation",
        "v10":   "wind_speed_10m",
    }

    daily = {}
    for short, col in cols_wanted.items():
        if col not in df.columns:
            continue
        daily[short] = df[col].resample("1D").mean()

    daily_df = pd.DataFrame(daily)

    # Wc (TCWV) is excluded from the completeness criterion — see docstring.
    essential = [c for c in daily_df.columns if c != "Wc"]
    return daily_df.dropna(subset=essential)


# ══════════════════════════════════════════════════════════════════════════════
# §3  SPLIT BY G (TOP / BOTTOM QUARTILE)
# ══════════════════════════════════════════════════════════════════════════════

def quartile_split(daily: pd.DataFrame, key: str = "G",
                   q: float = 0.25) -> Dict[str, pd.DataFrame]:
    """
    Split daily data into three groups by the quantile of a key variable:

        low    :  bottom q-quantile
        mid    :  the middle (excluded from comparison)
        high   :  top q-quantile

    Quartiles q=0.25 give ≈ 1,750 days per group over the 2017–2026
    IFS-HRES record.
    """
    v = daily[key].values
    q_low  = np.quantile(v, q)
    q_high = np.quantile(v, 1.0 - q)
    return {
        "low":  daily[daily[key] <= q_low],
        "mid":  daily[(daily[key] > q_low) & (daily[key] < q_high)],
        "high": daily[daily[key] >= q_high],
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  EFFECT-SIZE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for independent samples.

        d = (μ_a − μ_b) / s_pooled

    Convention:
        |d| < 0.2   negligible
        |d| < 0.5   small
        |d| < 0.8   medium
        |d| ≥ 0.8   large
    """
    n_a, n_b = len(a), len(b)
    s_a2, s_b2 = a.var(ddof=1), b.var(ddof=1)
    s_pooled = math.sqrt(((n_a - 1) * s_a2 + (n_b - 1) * s_b2) /
                         (n_a + n_b - 2))
    if s_pooled < 1e-30:
        return 0.0
    return float((a.mean() - b.mean()) / s_pooled)


def compare_groups(low: pd.DataFrame, high: pd.DataFrame,
                   vars_to_check: List[str]) -> pd.DataFrame:
    """
    For each variable, compare high-G vs low-G daily means.

    Returns a DataFrame with columns:
        var, mean_low, mean_high, diff, sigma_pooled, cohens_d, direction
    """
    rows = []
    for v in vars_to_check:
        if v not in low.columns or v not in high.columns:
            continue
        a = low[v].dropna().values
        b = high[v].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue

        d = cohens_d(b, a)   # high − low
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
# §5  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, key: str = "G", q: float = 0.25):
    W = 92
    print("=" * W)
    print(f"  PC1 VERIFICATION — split daily data by {key}  "
          f"(top/bottom {q*100:.0f} %)")
    print("=" * W)

    if df.empty:
        print("\n  [No variable had sufficient data in both groups.]")
        return

    # ---- Split summary ----
    print(f"\n  Split sizes:")
    print(f"    low :  {df.iloc[0]['n_low']:>5} days")
    print(f"    high:  {df.iloc[0]['n_high']:>5} days")
    print(f"    G low  mean  :  {df.iloc[0]['key_low']:>8.2f}")
    print(f"    G high mean  :  {df.iloc[0]['key_high']:>8.2f}")

    # ---- Comparison table ----
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

    print("\n  Legend: ↑ = high-G mean larger   ↓ = high-G mean smaller")


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading merged hourly data …", flush=True)
    df = load_merged()
    print(f"  {len(df):,} hourly records  ({df.index[0]} → {df.index[-1]})")

    print("Aggregating to daily means …", flush=True)
    daily = build_daily(df)
    print(f"  {len(daily):,} daily records")

    # Split by G
    groups = quartile_split(daily, key="G", q=0.25)
    low  = groups["low"]
    high = groups["high"]
    print(f"\n  Low-G days : {len(low):,}   (G mean = {low['G'].mean():.2f})")
    print(f"  High-G days: {len(high):,}   (G mean = {high['G'].mean():.2f})")

    # Wc is excluded: only available from ~Sept 2024 — cannot be fairly
    # compared across the full 2017–2026 record.
    vars_to_check = ["T2", "Td", "U2", "VPD", "ET0", "Nc",
                     "SM07", "SM728", "Sh", "P", "v10"]
    result = compare_groups(low, high, vars_to_check)

    # Stuff extra info in first row for the printer
    if not result.empty:
        result["n_low"]     = len(low)
        result["n_high"]    = len(high)
        result["key_low"]   = low["G"].mean()
        result["key_high"]  = high["G"].mean()

    print_report(result, key="G", q=0.25)

    # ---- Print in PC1 hypothesis order ----
    print(f"\n  {'─' * (92 - 2)}")
    print(f"  PC1 HYPOTHESIS CHECK")
    print(f"  {'─' * (92 - 2)}")
    print(f"  High-G days should show:")
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
        ok = (sign * d > 0.2)   # at least "small" in expected direction
        if ok:
            correct += 1
            print(f"    ✓ {v:>6}  d = {d:+.3f}")
        else:
            print(f"    ✗ {v:>6}  d = {d:+.3f}   (expected sign {sign:+d})")

    print(f"\n  → {correct} / {total} variables match the PC1 hypothesis")


if __name__ == "__main__":
    main()