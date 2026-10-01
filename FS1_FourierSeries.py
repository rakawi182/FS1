#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FS1_FourierSeries.py
================================================================================
Analytical Fourier Series of Hourly Meteorological Variables
at IDW-Interpolated Target Coordinates (MJS Obsv, East Java)

--------------------------------------------------------------------------------
SCIENTIFIC CONTEXT
--------------------------------------------------------------------------------
Hourly meteorological fields at a target location are decomposed into a
finite Fourier series.  Because tropical atmospheric variability is dominated
by two orthogonal cycles — the annual (seasonal) cycle and the diurnal
(day–night) cycle — a trig-basis design matrix that includes both, plus
their cross-modulations, yields a compact analytical description of the
signal:

    X(t) = X̄₀  +  Σ[k=1..K] Aₖ cos(k·ωA·t + φₖ)        [annual]
               +  Σ[m=1..M] Bₘ cos(m·ωD·t + ψₘ)         [diurnal]
               +  Σ[n]      Cₙ cos(Ωₙ·t + χₙ)            [cross]

where
    ωA  = 2π / T_tropical                              (annual)
    ωD  = 2π / T_day                                   (diurnal)
    Ωₙ  = k·ωA + m·ωD                                  (cross-frequency)

The cross-frequency terms are **not independent oscillations**.  They
describe the modulation of the diurnal cycle (amplitude and phase) by the
annual cycle.  The (k, +1) and (k, −1) pairs are sidebands around ωD,
and their coefficients carry the amplitude-modulation index and the
phase-modulation index of the diurnal cycle.

--------------------------------------------------------------------------------
TIME CONVENTION
--------------------------------------------------------------------------------
The time coordinate is

    t  =  Julian hours elapsed from 2000-01-01 12:00:00 WIB
       =  (hours since J2000.0 reference, in civil time UTC+7)

Input data are timestamped in **WIB (UTC+7)**.  All output times
(day-of-year maxima, peak hour-of-day, etc.) are therefore expressed in
WIB.  No TT/TAI conversion is applied; the analysis is purely about the
shape of the signal, not about ephemeris-grade absolute timing.

Nominal label:  Epoch = J2000.0  ≈  JD 2451545.000 (civil, WIB).

--------------------------------------------------------------------------------
DATA PIPELINE
--------------------------------------------------------------------------------
    Station A (IFS-HRES 2017–2026, WIB)
    Station B (IFS-HRES 2017–2026, WIB)
        │
        ▼  IDW with inverse-square distance weighting
    Target T (MJS Obsv, Φ = −7.5220, λ = 112.5661, h = 28 m)
        │
        ▼  Ordinary Least Squares on the trig basis
    Coefficient vector θ  →  {Aₖ, φₖ, Bₘ, ψₘ, Cₙ, χₙ}

--------------------------------------------------------------------------------
NUMERICAL NOTES
--------------------------------------------------------------------------------
1.  Basis construction is incremental, year-by-year (chunk size = 8760 h),
    so memory usage is O(chunk × P) rather than O(N × P).  N ≈ 8.5 × 10⁴ h.

2.  Normal equations are accumulated in float64.  A condition-number
    safeguard (cond > 1e12) triggers a rank-truncated least-squares fallback.

3.  m = 12 diurnal is deliberately excluded.  12·ωD = π rad/h is the
    Nyquist frequency for hourly sampling; cos(πt) and sin(πt) are both
    constant on integer t, producing a rank-deficient column pair that
    yields non-physical amplitudes (~10⁸).  The 2-hour cycle is aliased
    in hourly data and has no atmospheric interpretation.

4.  Partial-R² values (annual-only, annual+diurnal) are computed by
    projecting the *full* θ onto leading column blocks.  Because the
    basis is near-orthogonal over ~10 years (frequency separations ≫ 1/T),
    this is a good approximation of the true nested R².  Rigorous values
    would require refitting the reduced model.

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------
    fourier_series.json   Structured coefficient database (machine-readable)
    fourier_series.txt    IMCCE-style ASCII tables (human-readable)

--------------------------------------------------------------------------------
AUTHOR
--------------------------------------------------------------------------------
    MJS · Jolotundo Observatory · East Java, Indonesia
    Version 1.2.0
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 1.1  Geodetic constants (degrees, metres) ─────────────────────────────────

PHI_T = -7.5220       # Target latitude  (MJS Obsv)
LAM_T = 112.5661      # Target longitude
H_T   = 28.0          # Target orthometric height

PHI_A = -7.5571175    # Station A latitude
LAM_A = 112.55735
H_A   = 28.0

PHI_B = -7.486819     # Station B latitude
LAM_B = 112.53821
H_B   = 28.0

IDW_POWER = 2         # inverse-distance exponent (Shepard's p = 2)

# ── 1.2 Data source ──────────────────────────────────────────────────────────

DEFAULT_BASE_DIR = "/storage/emulated/0/Download/openmeteo_data"

FILE_NAMES: Dict[str, Dict[str, str]] = {
    "A": {
        # Hanya gunakan file IFS-HRES 2017-2026 (sesuai screenshot)
        "hres": "ecmwf_ifs9_hourly2017-2026_-7.56S112.56E28m.csv",
    },
    "B": {
        # Hanya gunakan file IFS-HRES 2017-2026 (sesuai screenshot)
        "hres": "ecmwf_ifs9_hourly2017-2026_-7.49S112.54E28m.csv",
    },
}

JSON_OUT = "fourier_series.json"
TXT_OUT  = "fourier_series.txt"

# ── 1.3  Temporal reference ───────────────────────────────────────────────────
#
# The reference instant for t = 0 is 2000-01-01 12:00:00 WIB
# (civil noon of the reference day).  This is *labelled* J2000.0 in the
# traditional JD sense, but t is computed against a naïve pandas Timestamp
# rather than a Julian Day — the analysis is invariant under this choice.

J2000_JD        = 2451545.0
J2000_REF_STR   = "2000-01-01 12:00:00"
J2000_REF_TS    = pd.Timestamp(J2000_REF_STR)

# Tropical year (days) — governs the seasonal cycle of insolation.
T_YR_D  = 365.24219
T_YR_H  = T_YR_D * 24.0                 # 8765.81256 h
T_DAY_H = 24.0

# Angular frequencies (rad · h⁻¹)
OMEGA_A = 2.0 * math.pi / T_YR_H        # 7.167829867 × 10⁻⁴
OMEGA_D = 2.0 * math.pi / T_DAY_H       # 2.617993878 × 10⁻¹

# ── 1.4  Series order ─────────────────────────────────────────────────────────

# Annual harmonics k·ωA.  Nyquist at hourly sampling is π rad/h ≈
# 4383 · ωA, so 18 harmonics are far below aliasing.
N_ANNUAL = 18

# Diurnal harmonics m·ωD.  m = 12 excluded — see §NUMERICAL NOTES above.
N_DIURNAL = 11

# Cross-frequency pairs (k, m) with |Ω| = |k·ωA + m·ωD| ≥ 1e-9.
_CX_RAW: List[Tuple[int, int]] = [
    (1, +1), (1, +2), (1, +3), (1, +4), (1, +5), (1, +6),
    (2, +1), (2, +2), (2, +3), (2, +4),
    (3, +1), (3, +2), (3, +3),
    (4, +1), (4, +2),
    (1, -1), (1, -2), (1, -3), (1, -4), (1, -5), (1, -6),
    (2, -1), (2, -2), (2, -3), (2, -4),
    (3, -1), (3, -2), (3, -3),
    (4, -1), (4, -2),
]
CROSS_TERMS: List[Tuple[int, int]] = [
    (k, m) for (k, m) in _CX_RAW
    if abs(k * OMEGA_A + m * OMEGA_D) > 1.0e-9
]
CROSS_AMP_MIN = 0.005    # display threshold for ASCII cross-terms table

# ── 1.5  Variable registry ────────────────────────────────────────────────────

VARS: Dict[str, Dict] = {
    "TEMP":  {"col": "temperature_2m",
              "unit": "°C",     "sym": "T₂",   "name": "Air Temperature at 2 m",
              "scale": 1.0},
    "TDEW":  {"col": "dew_point_2m",
              "unit": "°C",     "sym": "T_d",  "name": "Dew-Point Temperature at 2 m",
              "scale": 1.0},
    "RH":    {"col": "relative_humidity_2m",
              "unit": "%",      "sym": "U₂",   "name": "Relative Humidity at 2 m",
              "scale": 1.0},
    "VPD":   {"col": "vapour_pressure_deficit",
              "unit": "kPa",    "sym": "e_d",  "name": "Vapour Pressure Deficit at 2 m",
              "scale": 1.0},
    "PSURF": {"col": "surface_pressure",
              "unit": "hPa",    "sym": "P_s",  "name": "Surface Pressure",
              "scale": 1.0},
    "CLOUD": {"col": "cloud_cover",
              "unit": "%",      "sym": "N_c",  "name": "Total Cloud Cover",
              "scale": 1.0},
    "SW":    {"col": "shortwave_radiation",
              "unit": "W m⁻²",  "sym": "G",    "name": "Global Horizontal Irradiance",
              "scale": 1.0},
    "WIND":  {"col": "wind_speed_10m",
              "unit": "m s⁻¹",  "sym": "v₁₀",  "name": "Wind Speed at 10 m",
              "scale": 1.0 / 3.6},
    "ET0":   {"col": "et0_fao_evapotranspiration",
              "unit": "mm h⁻¹", "sym": "ET₀",  "name": "FAO-56 Reference Evapotranspiration",
              "scale": 1.0},
    "PREC":  {"col": "precipitation",
              "unit": "mm h⁻¹", "sym": "P",    "name": "Total Precipitation",
              "scale": 1.0},
    "TCWV":  {"col": "total_column_integrated_water_vapour",
              "unit": "kg m⁻²", "sym": "W_c",  "name": "Total Column Integrated Water Vapour",
              "scale": 1.0},
    "SMSH":  {"col": "soil_moisture_0_to_7cm",
              "unit": "m³ m⁻³", "sym": "θ₀₇",  "name": "Soil Moisture 0–7 cm",
              "scale": 1.0},
    "SMDP":  {"col": "soil_moisture_7_to_28cm",
              "unit": "m³ m⁻³", "sym": "θ₇₂₈", "name": "Soil Moisture 7–28 cm",
              "scale": 1.0},
    "SUN":   {"col": "sunshine_duration",
              "unit": "frac",   "sym": "S_h",  "name": "Sunshine Fraction (0–1)",
              "scale": 1.0 / 3600.0},
}

W = 120   # ASCII table width


# ══════════════════════════════════════════════════════════════════════════════
# §2  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_files(cli_dir: Optional[str] = None) -> Tuple[Dict[str, Dict[str, str]], str]:
    """
    Locate the directory containing the four input CSV files.

    Resolution order
    ----------------
    1. `--data-dir` CLI argument
    2. `$MJS_DATA_DIR` environment variable
    3. `DEFAULT_BASE_DIR`  (Android default path)
    4. `$HOME/Download/openmeteo_data`

    Returns
    -------
    (files, base_dir)
        files    : dict {station: {source: path}} for the two CSVs
        base_dir : resolved absolute directory

    Raises
    ------
    FileNotFoundError
        if none of the candidate directories contains both CSVs
    """
    home = os.path.expanduser("~")
    candidates = [
        cli_dir,
        os.environ.get("MJS_DATA_DIR"),
        DEFAULT_BASE_DIR,
        os.path.join(home, "Download", "openmeteo_data"),
    ]

    tried: List[str] = []
    seen: set = set()

    for base in candidates:
        if not base:
            continue
        base = os.path.abspath(base)
        if base in seen:
            continue
        seen.add(base)

        files = {
            "A": {k: os.path.join(base, n) for k, n in FILE_NAMES["A"].items()},
            "B": {k: os.path.join(base, n) for k, n in FILE_NAMES["B"].items()},
        }

        missing = [
            os.path.basename(p)
            for st in files.values() for p in st.values()
            if not os.path.isfile(p)
        ]
        if not missing:
            return files, base
        if os.path.isdir(base):
            tried.append(f"  {base}\n    missing: " + ", ".join(missing))

    msg = "No directory contains the two required CSV files.\n"
    if tried:
        msg += "\nDirectories inspected:\n" + "\n".join(tried) + "\n"
    else:
        msg += "\nNo candidate directory exists.\n"
    msg += (
        "\nResolutions:\n"
        f"  • Place CSVs in:  {DEFAULT_BASE_DIR}\n"
        "  • Or run with:    python FS1_FourierSeries.py --data-dir /path/to/csv\n"
        "  • Or set env:     export MJS_DATA_DIR=/path/to/csv\n"
    )
    raise FileNotFoundError(msg)


def _read_csv(path: str) -> pd.DataFrame:
    """
    Read an Open-Meteo / ERA5 CSV with optional metadata header.

    The raw files produced by the Open-Meteo historical API begin with
    several metadata lines (coordinates, elevation, timezone) before the
    data header.  This function scans for the first line beginning with
    ``time`` and parses the table from there.

    Column names are stripped of their unit suffix
    (``"temperature_2m (°C)"`` → ``"temperature_2m"``).

    Timestamps are assumed to be in **WIB (UTC+7)** and are kept naïve
    (no tz-aware conversion), consistent with the analysis frame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, sorted ascending.
    """
    with open(path, encoding="utf-8") as fh:
        header_row = None
        for i, line in enumerate(fh):
            if line.lstrip().startswith("time"):
                header_row = i
                break

    if header_row is None:
        raise ValueError(f"No 'time,...' header found in {path}")

    df = pd.read_csv(path, skiprows=header_row, low_memory=False)
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def load_station(files: Dict[str, str]) -> pd.DataFrame:
    """
    Load a single station record from an IFS-HRES CSV (2017–2026).

    Only the ``hres`` file is read; no ERA5 splice is performed.
    This eliminates the model-transition step-jump at the
    ERA5 → IFS boundary that would otherwise contaminate the Fourier
    phase estimate with a spurious low-frequency component.

    Parameters
    ----------
    files : dict
        ``{"hres": path}`` for one station.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted station record.
    """
    hres = _read_csv(files["hres"])
    return hres.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# §3  IDW INTERPOLATION
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(phi1: float, lam1: float, phi2: float, lam2: float) -> float:
    """
    Great-circle distance (kilometres) between two geodetic points.

    Uses the haversine formula with a spherical Earth of radius 6371 km.
    For station separations ≲ 5 km this agrees with the ellipsoidal
    geodesic to well within 0.1 %.

    Parameters
    ----------
    phi1, lam1 : float
        Latitude and longitude of the first point, in degrees.
    phi2, lam2 : float
        Latitude and longitude of the second point, in degrees.

    Returns
    -------
    float
        Surface distance in kilometres.
    """
    R = 6371.0
    p1, p2 = math.radians(phi1), math.radians(phi2)
    dp = p2 - p1
    dl = math.radians(lam2 - lam1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(max(0.0, a)))


def idw_weights(phi_t: float, lam_t: float,
                phi_a: float, lam_a: float,
                phi_b: float, lam_b: float,
                p: float = 2.0) -> Tuple[float, float, float, float]:
    """
    Inverse-distance weighting coefficients for two stations.

    Formula
    -------
        W_i = d_i^(−p) / Σ_j d_j^(−p)

    with ``p = 2`` (Shepard's classic exponent) giving an inverse-square
    decay.  Because both distances are O(1 km) here, the weights are
    smooth and the interpolation is stable.

    Returns
    -------
    (W_A, W_B, d_A_km, d_B_km)
    """
    d_a = haversine_km(phi_t, lam_t, phi_a, lam_a)
    d_b = haversine_km(phi_t, lam_t, phi_b, lam_b)
    w_a, w_b = d_a ** (-p), d_b ** (-p)
    s = w_a + w_b
    return w_a / s, w_b / s, d_a, d_b


def idw_merge(sta_a: pd.DataFrame, sta_b: pd.DataFrame,
              W_A: float, W_B: float,
              cols: List[str]) -> pd.DataFrame:
    """
    Merge two station frames onto their overlapping hourly grid using
    precomputed IDW weights.

    Time span is the intersection of both stations' records.
    Missing hours within either station are reindexed to NaN, and the
    linear combination propagates NaN if *either* input is NaN.  When
    one station is entirely missing a column, the other station is used
    unchanged.

    Parameters
    ----------
    sta_a, sta_b : pd.DataFrame
        Station records indexed by timestamp.
    W_A, W_B : float
        IDW weights summing to 1.
    cols : list of str
        Columns to interpolate.

    Returns
    -------
    pd.DataFrame
        IDW-interpolated target record on the intersection grid.
    """
    t_start = max(sta_a.index.min(), sta_b.index.min())
    t_end   = min(sta_a.index.max(), sta_b.index.max())
    idx     = pd.date_range(t_start, t_end, freq="1h")

    a = sta_a.reindex(idx)
    b = sta_b.reindex(idx)
    out = pd.DataFrame(index=idx)

    for col in cols:
        in_a, in_b = (col in a.columns), (col in b.columns)
        if in_a and in_b:
            out[col] = W_A * a[col].values + W_B * b[col].values
        elif in_a:
            out[col] = a[col].values
        elif in_b:
            out[col] = b[col].values

    return out


# ══════════════════════════════════════════════════════════════════════════════
# §4  TIME COORDINATE
# ══════════════════════════════════════════════════════════════════════════════

def datetime_to_j2000h(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert a naïve timestamp index (WIB) to Julian hours from
    the reference epoch.

    Reference
    ---------
        t = 0  ⇔  2000-01-01 12:00:00 WIB

    The returned array is in **hours**, float64, monotonically increasing
    at 1 h per sample for regularly-sampled input.

    Parameters
    ----------
    dt_index : pd.DatetimeIndex
        Naïve timestamps assumed to be in WIB.

    Returns
    -------
    np.ndarray, shape (N,)
        Julian hours from reference epoch.
    """
    return (dt_index - J2000_REF_TS).total_seconds().values / 3600.0


# ══════════════════════════════════════════════════════════════════════════════
# §5  FOURIER DESIGN MATRIX AND LEAST-SQUARES FIT
# ══════════════════════════════════════════════════════════════════════════════

def _n_params(n_a: int, n_d: int, cx: List[Tuple[int, int]]) -> int:
    """Total number of free parameters: 1 DC + 2·n_a + 2·n_d + 2·len(cx)."""
    return 1 + 2 * n_a + 2 * n_d + 2 * len(cx)


def _build_chunk(t: np.ndarray, n_a: int, n_d: int,
                 cx: List[Tuple[int, int]],
                 oa: float, od: float) -> np.ndarray:
    """
    Build the trigonometric design matrix Φ for a chunk of samples.

    Column layout
    -------------
        col 0                     :  1
        cols 1 … 2·n_a            :  cos(k·oa·t), sin(k·oa·t)     k = 1..n_a
        next 2·n_d                :  cos(m·od·t), sin(m·od·t)     m = 1..n_d
        next 2·len(cx)            :  cos(Ω·t),     sin(Ω·t)        Ω = k·oa + m·od

    Parameters
    ----------
    t : np.ndarray
        Time coordinates (Julian hours from epoch).
    n_a, n_d : int
        Number of annual and diurnal harmonics.
    cx : list of (int, int)
        Cross-frequency index pairs.
    oa, od : float
        Angular frequencies ωA and ωD in rad · h⁻¹.

    Returns
    -------
    np.ndarray, shape (len(t), n_params)
    """
    P = _n_params(n_a, n_d, cx)
    Phi = np.empty((len(t), P), dtype=np.float64)
    Phi[:, 0] = 1.0
    col = 1

    for k in range(1, n_a + 1):
        arg = k * oa * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for m in range(1, n_d + 1):
        arg = m * od * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for (k, m) in cx:
        arg = (k * oa + m * od) * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    return Phi


def fit_variable(t_jh: np.ndarray, X: np.ndarray,
                 n_a: int, n_d: int,
                 cx: List[Tuple[int, int]],
                 chunk: int = 8760
                 ) -> Tuple[np.ndarray, float, int]:
    """
    Ordinary least squares fit of a single variable on the trig basis.

    Accumulates the normal equations AtA and Aty chunk-by-chunk to keep
    peak memory bounded, then solves for θ.  A condition-number check
    (cond > 1e12) falls back to rank-truncated least squares.

    Parameters
    ----------
    t_jh : np.ndarray
        Time coordinates (Julian hours from epoch).
    X : np.ndarray
        Observations; NaNs are excluded sample-wise.
    n_a, n_d : int
        Number of annual / diurnal harmonics.
    cx : list of (int, int)
        Cross-frequency index pairs.
    chunk : int
        Maximum samples per accumulation block (default one year).

    Returns
    -------
    theta : np.ndarray, shape (P,)
        Fitted coefficients in the column order of `_build_chunk`.
    mse : float
        Mean squared residual (σ²_residual).
    N_ok : int
        Number of finite samples used.

    Raises
    ------
    ValueError
        If fewer finite samples are available than free parameters.
    """
    P = _n_params(n_a, n_d, cx)
    AtA = np.zeros((P, P))
    Aty = np.zeros(P)
    N_ok = 0
    SSyy = 0.0

    for i0 in range(0, len(t_jh), chunk):
        i1 = min(i0 + chunk, len(t_jh))
        t_c, x_c = t_jh[i0:i1], X[i0:i1]
        v = np.isfinite(x_c)
        if not v.any():
            continue
        t_v, x_v = t_c[v], x_c[v]
        Phi = _build_chunk(t_v, n_a, n_d, cx, OMEGA_A, OMEGA_D)
        AtA += Phi.T @ Phi
        Aty += Phi.T @ x_v
        N_ok += int(v.sum())
        SSyy += float(np.dot(x_v, x_v))

    if N_ok < P + 1:
        raise ValueError(f"Insufficient data: N={N_ok} < P+1={P+1}")

    cond = np.linalg.cond(AtA)
    if not np.isfinite(cond) or cond > 1e12:
        theta = np.linalg.lstsq(AtA, Aty, rcond=1e-10)[0]
        warnings.warn(
            f"AtA ill-conditioned (cond={cond:.2e}); using "
            f"rank-truncated least-squares. Check for duplicate or "
            f"Nyquist frequencies.",
            RuntimeWarning,
        )
    else:
        try:
            theta = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(AtA, Aty, rcond=None)[0]

    SS_res = max(SSyy - float(np.dot(theta, Aty)), 0.0)
    mse = SS_res / max(N_ok - P, 1)
    return theta, mse, N_ok


# ══════════════════════════════════════════════════════════════════════════════
# §6  COEFFICIENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_term(a: float, b: float) -> Tuple[float, float, float, float]:
    """
    Convert a (cos, sin) pair to (amplitude, phase, C_real, C_imag).

    Given a term written as

        a·cos(ω t) + b·sin(ω t)

    the equivalent single-cosine form is

        A·cos(ω t + φ),   A = √(a² + b²),   φ = atan2(−b, a)

    with φ ∈ [0°, 360°).  The complex representation used for downstream
    FFT-like manipulation is

        C = a − i·b

    so C_real = a and C_imag = −b.

    Returns
    -------
    (A, φ_deg, C_real, C_imag)
    """
    A = math.hypot(a, b)
    phi = math.degrees(math.atan2(-b, a)) % 360.0
    return A, phi, a, -b


def extract_annual(theta: np.ndarray, n_a: int) -> List[Dict]:
    """
    Extract annual harmonic coefficients from θ.

    For each k = 1..n_a, reads the (cos, sin) pair starting at column
    ``1 + 2·(k − 1)`` and returns a dict with amplitude, phase, physical
    period in hours and days, and the corresponding (C_real, C_imag).
    """
    out: List[Dict] = []
    for k in range(1, n_a + 1):
        col = 1 + 2 * (k - 1)
        A, phi, cr, ci = _extract_term(theta[col], theta[col + 1])
        w = k * OMEGA_A
        out.append({
            "k": k,
            "omega": w,
            "period_h": 2 * math.pi / w,
            "period_d": 2 * math.pi / w / 24.0,
            "A": A, "phi": phi,
            "c_re": cr, "c_im": ci,
        })
    return out


def extract_diurnal(theta: np.ndarray, n_a: int, n_d: int) -> List[Dict]:
    """
    Extract diurnal harmonic coefficients from θ.

    Diurnal block begins at column ``1 + 2·n_a``.  The physical period
    is exactly 24/m hours for each harmonic m.
    """
    off = 1 + 2 * n_a
    out: List[Dict] = []
    for m in range(1, n_d + 1):
        col = off + 2 * (m - 1)
        A, phi, cr, ci = _extract_term(theta[col], theta[col + 1])
        w = m * OMEGA_D
        out.append({
            "m": m,
            "omega": w,
            "period_h": 2 * math.pi / w,
            "A": A, "phi": phi,
            "c_re": cr, "c_im": ci,
        })
    return out


def extract_cross(theta: np.ndarray,
                  n_a: int, n_d: int,
                  cx: List[Tuple[int, int]]) -> List[Dict]:
    """
    Extract cross-frequency coefficients from θ.

    Physical interpretation
    -----------------------
    A cross term Ω = k·ωA + m·ωD can be viewed as a beat between the
    annual and diurnal cycles.  Since ωA ≪ ωD, the (k, ±1) family
    represents the amplitude- and phase-modulation of the m = 1 diurnal
    cycle by the k-th annual harmonic.  The reported period |2π/Ω| is a
    *physical* positive number; the sign of Ω is preserved in the `omega`
    field for reconstruction only.

    Returns
    -------
    List of dicts sorted by descending amplitude.
    """
    off = 1 + 2 * n_a + 2 * n_d
    out: List[Dict] = []
    for i, (k, m) in enumerate(cx):
        col = off + 2 * i
        A, phi, cr, ci = _extract_term(theta[col], theta[col + 1])
        w = k * OMEGA_A + m * OMEGA_D
        out.append({
            "k": k, "m": m,
            "omega": w,
            "period_h": abs(2 * math.pi / w),
            "A": A, "phi": phi,
            "c_re": cr, "c_im": ci,
        })
    return sorted(out, key=lambda x: -x["A"])


def incremental_r2(t_jh: np.ndarray, X: np.ndarray,
                   theta: np.ndarray,
                   n_a: int, n_d: int,
                   cx: List[Tuple[int, int]],
                   chunk: int = 8760) -> Dict[str, float]:
    """
    Nested R² and RMSE for three model tiers:

        1. annual only                     (1 + 2·n_a parameters)
        2. annual + diurnal                (+ 2·n_d)
        3. annual + diurnal + cross        (+ 2·len(cx))

    The three tiers are evaluated by projecting the *full* θ onto
    leading column blocks.  Because the basis is near-orthogonal over
    three decades of hourly samples, this is an accurate proxy for the
    true nested least-squares R².  Rigorous values would require
    refitting the reduced models.

    Also returns σ_total (standard deviation of the target) and the
    full-model residual RMSE, which together define the coefficient of
    determination for each tier.
    """
    groups = {
        "annual":  1 + 2 * n_a,
        "diurnal": 1 + 2 * n_a + 2 * n_d,
        "cross":   1 + 2 * n_a + 2 * n_d + 2 * len(cx),
    }
    valid = np.isfinite(X)
    xv = X[valid]
    xm = float(xv.mean())
    SS_tot = float(np.sum((xv - xm) ** 2))
    N_ok = int(valid.sum())

    if SS_tot < 1e-30:
        return {g: 0.0 for g in groups} | {
            "N": N_ok, "sigma_total": 0.0,
            "rmse_full": 0.0, "rmse_annual": 0.0, "rmse_diurnal": 0.0,
        }

    SS_res = {g: 0.0 for g in groups}
    for i0 in range(0, len(t_jh), chunk):
        i1 = min(i0 + chunk, len(t_jh))
        t_c, x_c = t_jh[i0:i1], X[i0:i1]
        v = np.isfinite(x_c)
        if not v.any():
            continue
        t_v, x_v = t_c[v], x_c[v]
        Phi = _build_chunk(t_v, n_a, n_d, cx, OMEGA_A, OMEGA_D)
        for g, pmax in groups.items():
            xh = Phi[:, :pmax] @ theta[:pmax]
            SS_res[g] += float(np.sum((x_v - xh) ** 2))

    return {
        "r2_annual":   max(0.0, 1.0 - SS_res["annual"]  / SS_tot),
        "r2_diurnal":  max(0.0, 1.0 - SS_res["diurnal"] / SS_tot),
        "r2_full":     max(0.0, 1.0 - SS_res["cross"]   / SS_tot),
        "rmse_annual":  math.sqrt(SS_res["annual"]  / N_ok),
        "rmse_diurnal": math.sqrt(SS_res["diurnal"] / N_ok),
        "rmse_full":    math.sqrt(SS_res["cross"]   / N_ok),
        "sigma_total":  math.sqrt(SS_tot / N_ok),
        "N":            N_ok,
    }


def augment_r2_cumulative(terms: List[Dict], sigma_total: float) -> None:
    """
    Annotate each term in-place with its incremental and cumulative
    contribution to the total variance.

    For a sinusoid of amplitude A, the variance is A²/2, so the
    fractional contribution to the signal variance σ²_total is

        ΔR²_k = A²_k / (2·σ²_total)

    Terms are accumulated in their list order to yield R²_cum.  This is
    a *diagnostic* aid, not part of the fit itself.
    """
    var = sigma_total ** 2 if sigma_total > 0 else 1.0
    cumul = 0.0
    for t in terms:
        dr2 = t["A"] ** 2 / (2.0 * var)
        cumul += dr2
        t["dR2"]    = min(dr2, 1.0)
        t["R2_cum"] = min(cumul, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# §7  TIME OF EXTREMUM
# ══════════════════════════════════════════════════════════════════════════════

def phase_to_doy(phi_deg: float, k: int) -> float:
    """
    Day-of-year of the annual-harmonic maximum.

    The k-th annual term is A·cos(k·ωA·t + φ).  Its maximum occurs when
    the argument is a multiple of 2π:

        t_peak = (−φ mod 2π) / (k·ωA)     [hours since epoch]

    Converted to day-of-year using the tropical year (365.24219 d).
    Returns a value in [1, 366.24) with day 1 ≈ Jan 1.

    Notes
    -----
    The reference epoch 2000-01-01 12:00 lies at day 1.5 of the
    Gregorian year, so the returned value carries a systematic +0.5 d
    offset relative to "true day-of-year".  This offset is consistent
    across all annual terms and is ignored for interpretation.
    """
    phi = math.radians(phi_deg)
    w = k * OMEGA_A
    t_h = (-phi % (2.0 * math.pi)) / w
    return ((t_h / T_YR_H) % 1.0) * T_YR_D + 1.0


def phase_to_hour_wib(phi_deg: float, m: int) -> float:
    """
    Hour-of-day (WIB) of the m-th diurnal harmonic maximum.

    The m-th diurnal term is B·cos(m·ωD·t + ψ).  With ωD = 15°/h,

        t_peak = −ψ / (m · 15)  hours since epoch

    Since the epoch is at 12:00 WIB, the *hour-of-day* is

        h_peak = (12 + t_peak) mod 24
               = (12 − ψ/(m · 15)) mod 24

    Returns a float in [0, 24).
    """
    return (12.0 - phi_deg / (m * 15.0)) % 24.0


# ══════════════════════════════════════════════════════════════════════════════
# §8  ASCII FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def _dms(deg: float) -> str:
    """Format a signed angle in degrees as D°M′S″ with hemisphere-neutral sign."""
    sign = "+" if deg >= 0 else "−"
    deg = abs(deg)
    d = int(deg)
    rem = (deg - d) * 60
    m = int(rem)
    s = (rem - m) * 60
    return f"{sign}{d:02d}°{m:02d}′{s:04.1f}″"


def _sep(c: str = "─") -> str:
    """Return a horizontal separator line of width W."""
    return c * W


def format_header(vk: str, vd: Dict, meta: Dict) -> str:
    """Block heading: variable identity, target & station geometry, sources."""
    return "\n".join([
        _sep("═"),
        f"  {vd['name']}   [{vd['unit']}]   symbol: {vd['sym']}",
        _sep("─"),
        f"  Target    Φ_T = {_dms(PHI_T)}   λ_T = {_dms(LAM_T)}   h = {H_T:.0f} m MSL",
        f"  Station A Φ_A = {_dms(PHI_A)}   λ_A = {_dms(LAM_A)}   "
        f"h = {H_A:.0f} m   d_A = {meta['d_A']:.4f} km   W_A = {meta['W_A']:.6f}",
        f"  Station B Φ_B = {_dms(PHI_B)}   λ_B = {_dms(LAM_B)}   "
        f"h = {H_B:.0f} m   d_B = {meta['d_B']:.4f} km   W_B = {meta['W_B']:.6f}",
        f"  Source    IFS-HRES / Open-Meteo 2017–2026 (single-model)",
        f"  Data      N = {meta['N']:,} h   Δt = 1 h   "
        f"{meta['t_start']}  →  {meta['t_end']}  (WIB, UTC+7)",
        f"  Epoch     2000-01-01 12:00:00 WIB  "
        f"(nominal J2000.0 = JD {J2000_JD:.1f})",
    ])


def format_statistics(mean: float, vd: Dict, fit: Dict) -> str:
    """Grand mean, total σ, and full-model residual σ."""
    return "\n".join([
        _sep("─"),
        f"  {vd['sym']}₀  =  {mean:+.6f}  {vd['unit']}   (grand mean)",
        f"  σ_total    =  ±{fit['sigma_total']:.6f}  {vd['unit']}",
        f"  σ_residual =  ±{fit['rmse_full']:.6f}  {vd['unit']}",
    ])


def format_annual_table(terms: List[Dict], vd: Dict) -> str:
    """Annual harmonics: period, amplitude, phase, day-of-year maximum."""
    L = [
        _sep("─"),
        f"  ANNUAL HARMONICS   ωA = {OMEGA_A:.9e} rad h⁻¹   "
        f"T_A = {T_YR_H:.5f} h = {T_YR_D:.5f} d",
        f"  {vd['sym']}(t) = {vd['sym']}₀ + Σ[k=1..{N_ANNUAL}] Aₖ cos(k·ωA·t + φₖ)",
        "",
        f"   {'k':>3}   {'Period /h':>12}   {'Period /d':>10}"
        f"   {'Amplitude':>14}   {'Phase φₖ /°':>12}"
        f"   {'C_real':>12}   {'C_imag':>12}"
        f"   {'ΔR²':>7}   {'R²_cum':>7}   {'Day-of-yr max':>13}",
        "   " + "─" * (W - 3),
    ]
    for t in terms:
        doy = phase_to_doy(t["phi"], t["k"])
        L.append(
            f"   {t['k']:>3}   {t['period_h']:>12.4f}   {t['period_d']:>10.4f}"
            f"   {t['A']:>14.6f}   {t['phi']:>12.4f}"
            f"   {t['c_re']:>+12.6f}   {t['c_im']:>+12.6f}"
            f"   {t.get('dR2', 0):>7.4f}   {t.get('R2_cum', 0):>7.4f}"
            f"   {doy:>11.2f}"
        )
    return "\n".join(L)


def format_diurnal_table(terms: List[Dict], vd: Dict) -> str:
    """Diurnal harmonics: period, amplitude, phase, peak hour of day (WIB)."""
    L = [
        _sep("─"),
        f"  DIURNAL HARMONICS   ωD = {OMEGA_D:.9e} rad h⁻¹   "
        f"T_D = {T_DAY_H:.6f} h",
        f"  {vd['sym']}(t) = {vd['sym']}₀ + Σ[m=1..{N_DIURNAL}] Bₘ cos(m·ωD·t + ψₘ)",
        "",
        f"   {'m':>3}   {'Period /h':>12}"
        f"   {'Amplitude':>14}   {'Phase ψₘ /°':>12}"
        f"   {'C_real':>12}   {'C_imag':>12}"
        f"   {'ΔR²':>7}   {'R²_cum':>7}   {'Peak WIB-h':>10}",
        "   " + "─" * (W - 3),
    ]
    for t in terms:
        hmax = phase_to_hour_wib(t["phi"], t["m"])
        L.append(
            f"   {t['m']:>3}   {t['period_h']:>12.6f}"
            f"   {t['A']:>14.6f}   {t['phi']:>12.4f}"
            f"   {t['c_re']:>+12.6f}   {t['c_im']:>+12.6f}"
            f"   {t.get('dR2', 0):>7.4f}   {t.get('R2_cum', 0):>7.4f}"
            f"   {hmax:>10.2f}"
        )
    return "\n".join(L)


def format_cross_table(terms: List[Dict], vd: Dict, amp_min: float) -> str:
    """
    Cross-frequency terms above the display threshold.

    Sideband interpretation: pairs (k, ±1) around ωD.
    Physical period reported as |2π/Ω|.
    """
    sig = [t for t in terms if t["A"] >= amp_min]
    L = [
        _sep("─"),
        f"  CROSS-FREQUENCY TERMS   Ω_{{k,m}} = k·ωA + m·ωD   "
        f"(|C| ≥ {amp_min:.3f} {vd['unit']}; total fitted: {len(terms)})",
        "  Period is |2π/Ω|.  Pairs (k, ±1) are annual sidebands around ωD",
        "  recording amplitude/phase modulation of the diurnal cycle.",
        "",
        f"   {'(k,m)':>8}   {'Ω rad h⁻¹':>14}"
        f"   {'|Period| /h':>11}"
        f"   {'Amplitude':>12}   {'Phase χ /°':>12}"
        f"   {'C_real':>12}   {'C_imag':>12}",
        "   " + "─" * (W - 3),
    ]
    for t in sig:
        km = f"({t['k']:+d},{t['m']:+d})"
        L.append(
            f"   {km:>8}   {t['omega']:>14.9e}"
            f"   {t['period_h']:>11.4f}"
            f"   {t['A']:>12.6f}   {t['phi']:>12.4f}"
            f"   {t['c_re']:>+12.6f}   {t['c_im']:>+12.6f}"
        )
    if not sig:
        L.append(f"   (no terms ≥ {amp_min:.3f})")
    return "\n".join(L)


def format_convergence(fit: Dict, vd: Dict) -> str:
    """Nested R² / RMSE summary across the three model tiers."""
    n_cx = len(CROSS_TERMS)
    P = _n_params(N_ANNUAL, N_DIURNAL, CROSS_TERMS)
    L = [
        _sep("─"),
        f"  CONVERGENCE   N = {fit['N']:,} h   P = {P} "
        f"(1 DC + 2×{N_ANNUAL} annual + 2×{N_DIURNAL} diurnal + 2×{n_cx} cross)",
        "",
        f"  {'Component':<36} {'N_pars':>7}  {'R²':>8}  {'RMSE':>12}  {'ΔR²':>12}",
        "  " + "─" * (W - 4),
    ]
    r0 = fit["r2_annual"]
    L.append(f"  {'Annual harmonics':<36} {1 + 2 * N_ANNUAL:>7}  "
             f"{r0:>8.6f}  {fit['rmse_annual']:>10.6f} {vd['unit']:>2}  {'—':>12}")
    L.append(f"  {'+ Diurnal harmonics':<36} {2 * N_DIURNAL:>7}  "
             f"{fit['r2_diurnal']:>8.6f}  {fit['rmse_diurnal']:>10.6f} "
             f"{vd['unit']:>2}  {fit['r2_diurnal'] - r0:>+12.6f}")
    L.append(f"  {'+ Cross-freq. terms':<36} {2 * n_cx:>7}  "
             f"{fit['r2_full']:>8.6f}  {fit['rmse_full']:>10.6f} "
             f"{vd['unit']:>2}  {fit['r2_full'] - fit['r2_diurnal']:>+12.6f}")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
# §9  JSON OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_json(results: Dict[str, Dict], meta: Dict) -> Dict:
    """
    Assemble the machine-readable coefficient database.

    The top-level ``meta`` block captures the analysis geometry, epoch
    convention, series orders, and provenance notes.  Each entry under
    ``series`` holds the mean, σ values, nested R², and the full list of
    annual / diurnal / cross coefficients with their (A, φ, C_real, C_imag)
    tuples.
    """
    out = {
        "meta": {
            "module": "FS1_FourierSeries",
            "series_root": "MJS",
            "version": "1.2.0",
            "target": {
                "name": "MJS Obsv",
                "phi_deg": PHI_T, "lambda_deg": LAM_T, "h_m": H_T,
                "phi_dms": _dms(PHI_T), "lambda_dms": _dms(LAM_T),
            },
            "stations": {
                "A": {"phi": PHI_A, "lambda": LAM_A, "h": H_A,
                      "d_km": meta["d_A"], "W": meta["W_A"]},
                "B": {"phi": PHI_B, "lambda": LAM_B, "h": H_B,
                      "d_km": meta["d_B"], "W": meta["W_B"]},
            },
            "epoch": "2000-01-01 12:00:00 WIB (nominal J2000.0)",
            "epoch_jd": J2000_JD,
            "timezone": "WIB (UTC+7)",
            "t_unit": "Julian hours from the reference epoch",
            "omega_A_rad_h": OMEGA_A,
            "omega_D_rad_h": OMEGA_D,
            "T_yr_h": T_YR_H,
            "T_day_h": T_DAY_H,
            "idw_power": IDW_POWER,
            "N_annual": N_ANNUAL,
            "N_diurnal": N_DIURNAL,
            "N_cross": len(CROSS_TERMS),
            "N_params": _n_params(N_ANNUAL, N_DIURNAL, CROSS_TERMS),
            "N_hours": meta["N"],
            "t_start": meta["t_start"],
            "t_end": meta["t_end"],
            "data_dir": meta["data_dir"],
            "nyquist_note": (
                "m=12 diurnal excluded: 12·ωD = π rad/h is the Nyquist "
                "frequency at hourly sampling; columns are rank-deficient "
                "and amplitudes non-physical."
            ),
            "cross_note": (
                "Reported period = |2π/Ω|; signed Ω preserved for "
                "reconstruction. (k, ±1) pairs are annual sidebands around ωD."
            ),
            "timezone_note": (
                "All times are in WIB (UTC+7). Peak-hour values in the "
                "diurnal table are hour-of-day WIB."
            ),
        },
        "series": {},
    }

    def _r(v):
        return round(v, 9) if isinstance(v, float) else v

    for vk, vr in results.items():
        out["series"][vk] = {
            "series_id": f"{vk}1",
            "name": VARS[vk]["name"],
            "symbol": VARS[vk]["sym"],
            "unit": VARS[vk]["unit"],
            "mean": round(vr["mean"], 8),
            "sigma_total": round(vr["fit"]["sigma_total"], 8),
            "sigma_resid": round(vr["fit"]["rmse_full"], 8),
            "R2_annual":  round(vr["fit"]["r2_annual"], 6),
            "R2_diurnal": round(vr["fit"]["r2_diurnal"], 6),
            "R2_full":    round(vr["fit"]["r2_full"], 6),
            "annual":  [{k: _r(v) for k, v in t.items()} for t in vr["annual"]],
            "diurnal": [{k: _r(v) for k, v in t.items()} for t in vr["diurnal"]],
            "cross":   [{k: _r(v) for k, v in t.items()} for t in vr["cross"]],
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §10  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point.

    Steps
    -----
    1. Resolve the four input CSVs.
    2. Compute IDW weights and load both stations.
    3. Build the IDW-interpolated target record.
    4. Convert timestamps to Julian hours from epoch.
    5. For each requested variable:
         - fit the full model,
         - extract annual / diurnal / cross coefficients,
         - compute nested R² and RMSE,
         - emit ASCII and JSON blocks.
    6. Write outputs.

    Returns
    -------
    int
        Process exit status (0 on success, 2 on data error).
    """
    ap = argparse.ArgumentParser(
        description="FS1 — Analytical Fourier Series at MJS Obsv"
    )
    ap.add_argument("--data-dir", default=None,
                    help=f"CSV directory (default: {DEFAULT_BASE_DIR})")
    ap.add_argument("--out-json", default=JSON_OUT)
    ap.add_argument("--out-txt",  default=TXT_OUT)
    ap.add_argument("--vars", default="",
                    help="Comma-separated subset of variables (default: all)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    var_keys = (
        [v.strip() for v in args.vars.split(",") if v.strip()]
        or list(VARS.keys())
    )

    # ── Banner ────────────────────────────────────────────────────────────
    print("=" * W)
    print("  MJS · Analytical Fourier Series · MJS Obsv")
    print(f"  Φ_T = {_dms(PHI_T)}   λ_T = {_dms(LAM_T)}   h = {H_T:.0f} m MSL")
    print(f"  Epoch 2000-01-01 12:00:00 WIB   ωA = {OMEGA_A:.6e} rad h⁻¹")
    print(f"  N_annual = {N_ANNUAL}   N_diurnal = {N_DIURNAL} "
          f"(m=12 excluded: Nyquist)   N_cross = {len(CROSS_TERMS)}")
    print("=" * W)

    # ── Resolve & load ────────────────────────────────────────────────────
    try:
        fdict, data_dir = _resolve_files(args.data_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 2

    W_A, W_B, d_A, d_B = idw_weights(
        PHI_T, LAM_T, PHI_A, LAM_A, PHI_B, LAM_B, IDW_POWER
    )
    print(f"\n  Data dir: {data_dir}")
    print(f"  IDW (p={IDW_POWER}):  d_A = {d_A:.4f} km  →  W_A = {W_A:.6f}")
    print(f"               d_B = {d_B:.4f} km  →  W_B = {W_B:.6f}")

    print("\n  Loading station data…", end="", flush=True)
    try:
        sta_a = load_station(fdict["A"])
        sta_b = load_station(fdict["B"])
    except Exception as e:
        print(f"\n[ERROR] Cannot read CSVs: {e}", file=sys.stderr)
        return 2
    print(f"  A: {len(sta_a):,} h   B: {len(sta_b):,} h")

    all_cols = sorted(set(sta_a.columns) | set(sta_b.columns))
    merged = idw_merge(sta_a, sta_b, W_A, W_B, all_cols)
    N_tot = len(merged)
    t_start = merged.index.min().strftime("%Y-%m-%dT%H:%M:%S")
    t_end   = merged.index.max().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"  Merged: {N_tot:,} h   {t_start} → {t_end}")

    t_jh = datetime_to_j2000h(merged.index)
    print(f"  t₀ = {t_jh[0]:+.4f} h   t_N = {t_jh[-1]:+.4f} h")

    meta = {
        "W_A": W_A, "W_B": W_B, "d_A": d_A, "d_B": d_B,
        "N": N_tot, "t_start": t_start, "t_end": t_end,
        "data_dir": data_dir,
    }

    # ── Fit loop ──────────────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    txt_blocks: List[str] = [
        "MJS — Analytical Fourier Series | MJS Obsv\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"N = {N_tot:,} h   {t_start} – {t_end} (WIB)\n"
        f"Data dir: {data_dir}\n"
        f"Epoch 2000-01-01 12:00:00 WIB  (nominal J2000.0 = JD {J2000_JD:.1f})\n"
        f"N_annual = {N_ANNUAL}   N_diurnal = {N_DIURNAL} "
        f"(m=12 Nyquist excluded)   N_cross = {len(CROSS_TERMS)}\n"
    ]

    for vk in var_keys:
        vd = VARS[vk]
        col = vd["col"]
        if col not in merged.columns:
            if not args.quiet:
                print(f"  [{vk}] column '{col}' not found — skip")
            continue

        X = merged[col].values.astype(np.float64) * vd["scale"]
        n_nan = int(np.sum(~np.isfinite(X)))
        if not args.quiet:
            print(f"\n  [{vk}] {vd['name']}   N_valid={N_tot - n_nan:,}  "
                  f"(NaN={n_nan:,})  …", end="", flush=True)

        theta, _, N_ok = fit_variable(t_jh, X, N_ANNUAL, N_DIURNAL, CROSS_TERMS)
        mean_val = float(theta[0])

        ann  = extract_annual(theta, N_ANNUAL)
        dia  = extract_diurnal(theta, N_ANNUAL, N_DIURNAL)
        cros = extract_cross(theta, N_ANNUAL, N_DIURNAL, CROSS_TERMS)
        fit  = incremental_r2(t_jh, X, theta, N_ANNUAL, N_DIURNAL, CROSS_TERMS)

        augment_r2_cumulative(ann, fit["sigma_total"])
        augment_r2_cumulative(dia, fit["sigma_total"])

        if not args.quiet:
            print(f"  R²={fit['r2_full']:.4f}  "
                  f"RMSE={fit['rmse_full']:.4f} {vd['unit']}")

        results[vk] = {
            "mean": mean_val, "theta": theta,
            "annual": ann, "diurnal": dia, "cross": cros, "fit": fit,
        }

        # ASCII block
        block  = "\n\n" + format_header(vk, vd, meta) + "\n"
        block += format_statistics(mean_val, vd, fit) + "\n"
        block += format_annual_table(ann, vd)  + "\n"
        block += format_diurnal_table(dia, vd) + "\n"
        block += format_cross_table(cros, vd, CROSS_AMP_MIN) + "\n"
        block += format_convergence(fit, vd) + "\n"
        block += _sep("═")
        txt_blocks.append(block)

        # Console summary
        print(f"\n  ─── {vd['name']}   "
              f"{vd['sym']}₀ = {mean_val:+.4f} {vd['unit']} ───")
        print(f"  Annual:  k  {'Period/d':>10}  {'Amplitude':>12}  "
              f"{'Phase/°':>10}  {'ΔR²':>8}")
        for t in ann[:6]:
            print(f"           {t['k']:>2}  {t['period_d']:>10.3f}  "
                  f"{t['A']:>12.6f}  {t['phi']:>10.4f}  "
                  f"{t.get('dR2', 0):>8.5f}")
        print(f"  Diurnal: m  {'Period/h':>10}  {'Amplitude':>12}  "
              f"{'Phase/°':>10}  {'Peak WIB':>9}")
        for t in dia[:5]:
            hmax = phase_to_hour_wib(t["phi"], t["m"])
            print(f"           {t['m']:>2}  {t['period_h']:>10.3f}  "
                  f"{t['A']:>12.6f}  {t['phi']:>10.4f}  "
                  f"{hmax:>9.2f}")
        print("  Cross-freq top-3 (|Period| = 2π/|Ω|):")
        for t in cros[:3]:
            print(f"    ({t['k']:+d},{t['m']:+d})  |T|={t['period_h']:.4f}h  "
                  f"A={t['A']:.6f}  φ={t['phi']:.4f}°")
        print(f"  Convergence: R²_ann={fit['r2_annual']:.6f}  "
              f"R²_dia={fit['r2_diurnal']:.6f}  "
              f"R²_full={fit['r2_full']:.6f}  "
              f"RMSE={fit['rmse_full']:.4f} {vd['unit']}")

    print("\n" + "=" * W)

    # ── Output ────────────────────────────────────────────────────────────
    with open(args.out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt_blocks))
    print(f"  ASCII → {args.out_txt}  "
          f"({os.path.getsize(args.out_txt) // 1024} KB)")

    out_json = build_json(
        {k: {kk: vv for kk, vv in v.items() if kk != "theta"}
         for k, v in results.items()},
        meta,
    )

    def _ser(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=2, ensure_ascii=False, default=_ser)
    print(f"  JSON  → {args.out_json}  "
          f"({os.path.getsize(args.out_json) // 1024} KB)")
    print("  Done.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())