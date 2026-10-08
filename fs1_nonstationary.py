#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_nonstationary.py  (v2)
================================================================================
Uji stasionaritas koefisien Fourier tahunan sepanjang epoch 2017–2025.

================================================================================
1. RUMUSAN MASALAH
================================================================================
Model FS1 mengasumsikan amplitudo dan fasa setiap harmonik tahunan konstan
sepanjang record:

    X(t) = X̄₀ + Σ_k A_k·cos(k·ωA·t + φ_k) + (diurnal + cross)

Pertanyaan: apakah A_k dan φ_k benar-benar konstan dari tahun ke tahun,
atau terdapat tren/modulasi sistematis yang membuat model ini bias?

Kita fit ulang model per-tahun kalender, memperoleh A_k(Y) dan φ_k(Y)
untuk Y ∈ {2017, ..., 2025}, lalu uji apakah deret waktu ini memiliki
tren linear yang signifikan.

================================================================================
2. MASALAH STATISTIK: FIXED-EFFECTS vs RANDOM-EFFECTS
================================================================================
Estimasi per-tahun Â_k(Y) mengandung dua sumber variasi:

    Â_k(Y) = A_k^true(Y) + ε_Y
    A_k^true(Y) = μ + δ_Y

dengan:
    ε_Y ~ N(0, σ_Y²)   — ketidakpastian pelaporan (σ_Y dari HAC per-tahun)
    δ_Y ~ N(0, τ²)      — variasi antar-tahun yang sebenarnya

Maka:
    Var(Â_k(Y)) = τ² + σ_Y²

Jika kita mengabaikan τ² dan menggunakan weight fixed-effects w_Y = 1/σ_Y²,
standard error slope akan under-estimate oleh faktor ≈ τ/σ̄. Untuk SMDP
di dataset ini: τ ≈ 0.030, σ̄ ≈ 0.004 → faktor ≈ 7×. Inilah sebabnya
t-statistik FE lalu menghasilkan t ≈ −22 padahal permutation memberi
p ≈ 0.09.

Solusi: model RANDOM-EFFECTS (DerSimonian & Laird 1986):

    w_Y^{RE} = 1 / (σ_Y² + τ̂²)

dengan τ̂² dari statistik Cochran Q:

    Q = Σ_Y w_Y^{FE}·(Â_k(Y) − Â_k^{FE})²
    τ̂² = max(0, (Q − df) / (Σw^{FE} − Σ(w^{FE})²/Σw^{FE}))
    df = Y_count − 2

Referensi:
    DerSimonian R, Laird N. Control Clin Trials. 1986;7(3):177-188.
    Higgins JPT, Thompson SG. Stat Med. 2002;21(11):1539-1558.

================================================================================
3. UJI PERMUTASI DENGAN RESIDUAL TERSTANDARDISASI
================================================================================
Di bawah H₀ (tidak ada tren), residu Â_k(Y) − μ̂ memiliki magnitudo
∝ σ_total,Y sehingga tidak exchangeable. Permutasi naif menghasilkan
distribusi null yang miring.

Konstruksi benar:

    z_Y = (Â_k(Y) − μ̂) / σ_total,Y          [terstandardisasi]
    Â_perm(Y) = μ̂ + z_{π(Y)} · σ_total,Y    [rekonstruksi]

Distribusi null dari |t_perm| ≥ |t_obs| memberi nilai-p yang valid
terhadap heteroskedastis.

Referensi:
    Good P. Permutation, Parametric and Bootstrap Tests of Hypotheses.
    Springer; 2005.

================================================================================
4. STRUKTUR MODUL
================================================================================
    §1  Konfigurasi
    §2  Design matrix + fit per-tahun dengan HAC
    §3  Uji tren random-effects
    §4  Uji permutasi terstandardisasi
    §5  Sanity check (~10 detik, abort jika FP > 12%)
    §6  Driver per-variabel
    §7  Reporting
    §8  Main

Author : MJS · Jolotundo Observatory
Version: 2.0.0
================================================================================
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import FS1_FourierSeries as FS1
from fs1_verify_pc1 import load_merged
from fs1_hac_verify import hac_covariance


# ══════════════════════════════════════════════════════════════════════════════
# §1  KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════

N_ANN = 3
"""Jumlah harmonik tahunan yang diuji per-tahun (k = 1..3)."""

N_DIA = 3
"""Jumlah harmonik diurnal yang menjadi kontrol (m = 1..3)."""

CX_YEAR: List[Tuple[int, int]] = [
    (1,  1), (1, -1), (1,  2), (1, -2),
    (2,  1), (2, -1),
    (3,  1), (3, -1),
]
"""Cross-term signifikan (dari fourier_batch_hac.json)."""

HAC_L = 48
"""Bandwidth HAC (jam) untuk fit per-tahun."""

YEARS_FULL = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
"""Tahun yang dianalisis. 2026 dipotong karena berakhir September."""

SANITY_FP_MAX = 12.0
"""Ambang FP rate. Sanity check gagal jika melebihi ini."""


# ══════════════════════════════════════════════════════════════════════════════
# §2  DESIGN MATRIX DAN FIT PER-TAHUN
# ══════════════════════════════════════════════════════════════════════════════

def build_design(t: np.ndarray) -> np.ndarray:
    """
    Design matrix trigonometri untuk satu tahun.

    Kolom:
        col 0              : 1
        col 1 + 2(k−1)     : cos(k·ωA·t)       k = 1..N_ANN
        col 1 + 2(k−1) + 1 : sin(k·ωA·t)
        ...                : cos(m·ωD·t), sin(m·ωD·t)   m = 1..N_DIA
        ...                : cos(Ω·t),     sin(Ω·t)     Ω = k·ωA + m·ωD

    Returns
    -------
    Phi : (N, P), P = 1 + 2·N_ANN + 2·N_DIA + 2·|CX_YEAR| = 29.
    """
    P = 1 + 2 * N_ANN + 2 * N_DIA + 2 * len(CX_YEAR)
    Phi = np.empty((len(t), P), dtype=np.float64)
    Phi[:, 0] = 1.0

    col = 1
    for k in range(1, N_ANN + 1):
        arg = k * FS1.OMEGA_A * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for m in range(1, N_DIA + 1):
        arg = m * FS1.OMEGA_D * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    for (k, m) in CX_YEAR:
        arg = (k * FS1.OMEGA_A + m * FS1.OMEGA_D) * t
        Phi[:, col] = np.cos(arg); col += 1
        Phi[:, col] = np.sin(arg); col += 1

    return Phi


def phase_to_doy(phi_deg: float, k: int) -> float:
    """
    Fasa harmonik → hari-dalam-tahun puncak.

    Term: A_k·cos(k·ωA·t + φ_k). Puncak ketika argumen = 2πn:

        t_peak = (−φ_k mod 2π) / (k·ωA)      [jam dari epoch]

    Konversi ke DOY dengan T_YR_H dan T_YR_D. Offset +1 agar DOY ∈ [1, 366).
    Offset sistemik +0.5 d dari epoch konsisten antar tahun sehingga tidak
    memengaruhi uji tren.
    """
    phi = math.radians(phi_deg)
    w = k * FS1.OMEGA_A
    t_h = (-phi % (2.0 * math.pi)) / w
    return ((t_h / FS1.T_YR_H) % 1.0) * FS1.T_YR_D + 1.0


def fit_year(t_jh: np.ndarray, x: np.ndarray,
             year_mask: np.ndarray) -> Optional[Dict]:
    """
    Fit model trigonometri ke satu tahun kalender.

    Prosedur:
        1. Ekstrak sampel; buang NaN.
        2. Bangun design matrix.
        3. OLS via np.linalg.lstsq.
        4. Residual, ρ₁.
        5. Kovarians HAC (Newey-West, Bartlett, L = HAC_L).
        6. Untuk setiap k, hitung A_k, φ_k dan σ via propagasi orde pertama:

           σ²(A) = (a/A)²·V_aa + (b/A)²·V_bb + 2(a/A)(b/A)·V_ab
           σ(φ)  ≈ σ(A)/A         [radian]
           σ(DOY) = T_YR_D/(360·k)·σ(φ_deg)

    Returns
    -------
    dict atau None jika N < 2000.
    """
    t_v = t_jh[year_mask]
    x_v = x[year_mask]
    valid = np.isfinite(x_v)
    N = int(valid.sum())
    if N < 2000:
        return None

    t_v, x_v = t_v[valid], x_v[valid]
    Phi = build_design(t_v)

    theta, _, _, _ = np.linalg.lstsq(Phi, x_v, rcond=None)
    resid = x_v - Phi @ theta

    sigma_resid = float(np.std(resid, ddof=0))
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    V = hac_covariance(Phi, resid, maxlags=HAC_L)

    out = {'N': N, 'sigma_resid': sigma_resid, 'rho1': rho1}

    for k in range(1, N_ANN + 1):
        col = 1 + 2 * (k - 1)
        a, b = theta[col], theta[col + 1]
        A = math.hypot(a, b)
        phi_deg = math.degrees(math.atan2(-b, a)) % 360.0

        Va, Vb = V[col, col], V[col + 1, col + 1]
        Vab = V[col, col + 1]

        if A > 1e-30:
            va, vb = a / A, b / A
            var_A = va * va * Va + vb * vb * Vb + 2.0 * va * vb * Vab
        else:
            var_A = max(Va, Vb)
        sig_A = math.sqrt(max(var_A, 0.0))
        sig_phi_deg = math.degrees(sig_A / A) if A > 1e-30 else float('inf')

        out[f'A{k}']       = A
        out[f'phi{k}']     = phi_deg
        out[f'sig_A{k}']   = sig_A
        out[f'sig_phi{k}'] = sig_phi_deg
        out[f'doy{k}']     = phase_to_doy(phi_deg, k)
        out[f'sig_doy{k}'] = FS1.T_YR_D / (360.0 * k) * sig_phi_deg

    return out


# ══════════════════════════════════════════════════════════════════════════════
# §3  UJI TREN — RANDOM-EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_trend(years: np.ndarray, values: np.ndarray,
                   sigmas: np.ndarray) -> Optional[Dict]:
    """
    Regresi linear berbobot dengan komponen random-effects.

    Model: value_Y = α + β·Y + δ_Y + ε_Y, dengan δ_Y ~ N(0, τ²)
    dan ε_Y ~ N(0, σ_Y²). Estimator DerSimonian-Laird:

        1. FE fit dengan w_Y^{FE} = 1/σ_Y².
        2. Q = Σ w^{FE}·resid_FE².
        3. τ̂² = max(0, (Q − df)/(Σw^{FE} − Σ(w^{FE})²/Σw^{FE})).
        4. RE fit dengan w_Y^{RE} = 1/(σ_Y² + τ̂²).
        5. Kovarians = (Xᵀ W X)⁻¹.

    Returns
    -------
    dict berisi slope, se_slope, intercept, t_stat, mean, tau2, tau, Q, df.
    """
    years  = np.asarray(years,  dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    w0 = 1.0 / np.maximum(sigmas, 1e-30) ** 2

    X = np.column_stack([np.ones_like(years), years])

    try:
        beta_fe = np.linalg.solve(X.T @ (X * w0[:, None]), X.T @ (values * w0))
    except np.linalg.LinAlgError:
        return None

    Q = float(np.sum(w0 * (values - X @ beta_fe) ** 2))
    df = len(years) - 2
    sw, sw2 = float(w0.sum()), float((w0 ** 2).sum())

    if Q > df and (sw - sw2 / sw) > 0:
        tau2 = (Q - df) / (sw - sw2 / sw)
    else:
        tau2 = 0.0

    w = 1.0 / (sigmas ** 2 + tau2)

    try:
        XtWX = X.T @ (X * w[:, None])
        beta = np.linalg.solve(XtWX, X.T @ (values * w))
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None

    se = math.sqrt(max(cov[1, 1], 0.0))

    return {
        'slope':     float(beta[1]),
        'se_slope':  float(se),
        'intercept': float(beta[0]),
        't_stat':    float(beta[1] / se) if se > 0 else 0.0,
        'mean':      float(np.average(values, weights=w)),
        'tau2':      float(tau2),
        'tau':       float(math.sqrt(tau2)),
        'Q':         float(Q),
        'df':        int(df),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  UJI PERMUTASI — RESIDUAL TERSTANDARDISASI
# ══════════════════════════════════════════════════════════════════════════════

def permutation_p(years: np.ndarray, values: np.ndarray,
                  sigmas: np.ndarray, observed_t: float,
                  n_perm: int = 5000, seed: Optional[int] = 42) -> float:
    """
    Nilai-p dari uji permutasi residual terstandardisasi.

    Prosedur:
        1. μ̂ = weighted mean dengan w = 1/σ².
        2. z_Y = (value_Y − μ̂) / σ_Y.
        3. Permutasi z, rekonstruksi dengan σ_Y asli.
        4. p = proporsi |t_perm| ≥ |t_obs|.

    Distribusi null valid terhadap heteroskedastis karena σ_Y dipertahankan
    di posisi semula.

    Parameters
    ----------
    seed : int atau None
        None untuk sanity check (variasi antar trial).
    """
    years  = np.asarray(years,  dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)

    w = 1.0 / np.maximum(sigmas, 1e-30) ** 2
    ybar = float(np.sum(w * values) / np.sum(w))
    z = (values - ybar) / sigmas

    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_perm):
        v_perm = ybar + z[rng.permutation(len(values))] * sigmas
        r = weighted_trend(years, v_perm, sigmas)
        if r is not None and abs(r['t_stat']) >= abs(observed_t):
            count += 1

    return count / n_perm


# ══════════════════════════════════════════════════════════════════════════════
# §5  SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(n_trials: int = 100, n_perm: int = 200) -> None:
    """
    Uji mesin trend-test pada data sintetis tanpa tren.

    Konstruksi: A_Y = μ + N(0, σ_Y² + τ²) untuk Y ∈ YEARS_FULL.
    Di bawah H₀, FP rate seharusnya ≈ 5%. Dengan 100 trial, resolusi
    ±2% — cukup untuk membedakan 5% dari 12%.

    Runtime: ~10 detik di Termux.
    Abort dengan sys.exit(1) jika FP > SANITY_FP_MAX.
    """
    print("Sanity check …", end=" ", flush=True)
    t0 = time.perf_counter()

    years  = np.array(YEARS_FULL, dtype=float)
    sigmas = np.array([0.05, 0.06, 0.055, 0.07, 0.05, 0.06, 0.05, 0.07, 0.06])
    tau_true = 0.15
    rng = np.random.default_rng(0)
    n_flag = 0

    for _ in range(n_trials):
        A = 1.0 + rng.standard_normal(len(years)) * np.sqrt(sigmas**2 + tau_true**2)
        tr = weighted_trend(years, A, sigmas)
        if tr is None:
            continue
        p = permutation_p(years, A, sigmas, tr['t_stat'],
                          n_perm=n_perm, seed=None)
        n_flag += (p < 0.05)

    fp = 100.0 * n_flag / n_trials
    print(f"FP {fp:.1f}%  ({time.perf_counter() - t0:.1f}s)")

    if fp > SANITY_FP_MAX:
        print(f"  ✗ GAGAL: FP {fp:.1f}% > {SANITY_FP_MAX}%. Abort.\n")
        sys.exit(1)

    print("  ✓ PASS\n")


# ══════════════════════════════════════════════════════════════════════════════
# §6  DRIVER PER-VARIABEL
# ══════════════════════════════════════════════════════════════════════════════

def run_variable(vk: str, vd: Dict, t_jh: np.ndarray, x: np.ndarray,
                 years_of_sample: np.ndarray, n_perm: int) -> Optional[Dict]:
    """
    Analisis non-stasionaritas untuk satu variabel.

    Untuk setiap Y ∈ YEARS_FULL: fit per-tahun. Lalu untuk setiap
    k = 1..N_ANN:
        - uji tren A_k(Y) vs Y
        - uji tren DOY_k(Y) vs Y
    Plus tren σ_resid(Y).

    Returns
    -------
    dict atau None jika < 6 tahun berhasil difit.
    """
    by_year: List[Dict] = []

    for Y in YEARS_FULL:
        fit = fit_year(t_jh, x, years_of_sample == Y)
        if fit is None:
            continue
        fit['year'] = Y
        by_year.append(fit)

    if len(by_year) < 6:
        return None

    years_arr = np.array([f['year'] for f in by_year], dtype=float)
    trends: Dict[str, Optional[Dict]] = {}

    for k in [1, 2, 3]:
        # Tren amplitudo
        A    = np.array([f[f'A{k}']     for f in by_year])
        sigA = np.array([f[f'sig_A{k}'] for f in by_year])
        tr_A = weighted_trend(years_arr, A, sigA)
        if tr_A is not None:
            tr_A['p_perm'] = permutation_p(years_arr, A, sigA,
                                           tr_A['t_stat'], n_perm=n_perm)
        trends[f'A{k}'] = tr_A

        # Tren fasa (DOY)
        doy    = np.array([f[f'doy{k}']     for f in by_year])
        sigdoy = np.array([f[f'sig_doy{k}'] for f in by_year])
        tr_D = weighted_trend(years_arr, doy, sigdoy)
        if tr_D is not None:
            tr_D['p_perm'] = permutation_p(years_arr, doy, sigdoy,
                                           tr_D['t_stat'], n_perm=n_perm)
        trends[f'DOY{k}'] = tr_D

    # Tren σ_resid
    rs    = np.array([f['sigma_resid'] for f in by_year])
    N_arr = np.array([f['N']           for f in by_year])
    sig_rs = rs / np.sqrt(2.0 * N_arr)
    tr_rs = weighted_trend(years_arr, rs, sig_rs)
    if tr_rs is not None:
        tr_rs['p_perm'] = permutation_p(years_arr, rs, sig_rs,
                                        tr_rs['t_stat'], n_perm=n_perm)
    trends['sigma_resid'] = tr_rs

    return {
        'vk':   vk,
        'name': vd['name'],
        'sym':  vd['sym'],
        'unit': vd['unit'],
        'by_year': by_year,
        'trends':  trends,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §7  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict]) -> None:
    """Tabel utama: trend amplitudo dan DOY per harmonik per variabel."""
    W = 132
    print()
    print("=" * W)
    print("  NON-STATIONARITY OF ANNUAL HARMONICS (2017–2025)  [random-effects]")
    print("=" * W)

    print(f"\n  {'var':<7}{'k':>3}"
          f"{'A_mean':>10}{'τ':>9}"
          f"{'ΔA/dec':>10}{'t(A)':>8}{'p_A':>8}  "
          f"{'DOY_mean':>10}{'ΔDOY/dec':>11}{'t(DOY)':>8}{'p_DOY':>8}")
    print("  " + "─" * (W - 4))

    for r in results:
        for k in [1, 2, 3]:
            tr_A = r['trends'].get(f'A{k}')
            tr_D = r['trends'].get(f'DOY{k}')
            if tr_A is None:
                continue

            A_mean = tr_A['mean']
            dA_pct = (100.0 * tr_A['slope'] * 10.0 / A_mean
                      if A_mean > 1e-30 else 0.0)
            flag = "*" if tr_A['p_perm'] < 0.05 else " "

            if tr_D is not None:
                doy_str = f"{tr_D['mean']:>10.2f}"
                dD_str  = f"{tr_D['slope'] * 10:>+11.2f}"
                tD_str  = f"{tr_D['t_stat']:>+8.2f}"
                pD_str  = f"{tr_D['p_perm']:>8.4f}"
            else:
                doy_str = dD_str = tD_str = pD_str = " " * 10

            print(f"  {r['vk']:<7}{k:>3}"
                  f"{A_mean:>10.4f}{tr_A['tau']:>9.4f}"
                  f"{dA_pct:>+9.1f}%{tr_A['t_stat']:>+7.2f}"
                  f"{tr_A['p_perm']:>8.4f}{flag} "
                  f"{doy_str}{dD_str}{tD_str}{pD_str}")

        tr_r = r['trends'].get('sigma_resid')
        if tr_r is not None:
            rs = tr_r['mean']
            drs_pct = (100.0 * tr_r['slope'] * 10 / rs
                       if rs > 1e-30 else 0.0)
            flag = "*" if tr_r['p_perm'] < 0.05 else " "
            print(f"  {r['vk']:<7}{'-':>3}"
                  f"{'σ_res':>10}{tr_r['tau']:>9.4f}"
                  f"{drs_pct:>+9.1f}%{tr_r['t_stat']:>+7.2f}"
                  f"{tr_r['p_perm']:>8.4f}{flag}")
        print()

    print("  * = permutation p < 0.05")
    print("  τ  = between-year std (random-effects component)")
    print()


def print_flagged(results: List[Dict]) -> None:
    """Ringkasan tren signifikan (p_perm < 0.05), diurut |t| menurun."""
    flagged = []
    for r in results:
        for k in [1, 2, 3]:
            tr = r['trends'].get(f'A{k}')
            if tr and tr['p_perm'] < 0.05:
                flagged.append((r['vk'], f'A{k}', tr['t_stat'],
                                tr['slope'] * 10, tr['mean'], 'amplitude'))
            td = r['trends'].get(f'DOY{k}')
            if td and td['p_perm'] < 0.05:
                flagged.append((r['vk'], f'DOY{k}', td['t_stat'],
                                td['slope'] * 10, td['mean'], 'phase'))
        tr = r['trends'].get('sigma_resid')
        if tr and tr['p_perm'] < 0.05:
            flagged.append((r['vk'], 'σ_resid', tr['t_stat'],
                            tr['slope'] * 10, tr['mean'], 'variance'))

    print("=" * 78)
    print(f"  FLAGGED TRENDS (p_perm < 0.05):  {len(flagged)}")
    print("=" * 78)

    if not flagged:
        print("\n  Tidak ada tren signifikan yang terdeteksi.\n")
        return

    print(f"\n  {'var':<7}{'term':>10}{'t':>8}"
          f"{'Δ/decade':>12}{'mean':>12}  {'type':>10}")
    print("  " + "─" * 74)

    for vk, term, t, delta, mean, kind in sorted(flagged, key=lambda x: -abs(x[2])):
        print(f"  {vk:<7}{term:>10}{t:>+8.2f}"
              f"{delta:>+12.4f}{mean:>12.4f}  {kind:>10}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §8  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point. Urutan:
        1. Sanity check (~10 s; abort jika FP > 12%).
        2. Load data.
        3. Loop per-variabel.
        4. Tabel utama + flagged.
        5. JSON export.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", default="",
                    help="Comma-separated variable keys (default: all)")
    ap.add_argument("--out", default="fs1_nonstationary.json")
    ap.add_argument("--n-perm", type=int, default=5000,
                    help="Permutations per test (default 5000)")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Skip sanity check (JANGAN untuk produksi)")
    args = ap.parse_args()

    print("=" * 78)
    print("  FS1 NON-STATIONARITY v2")
    print("=" * 78)

    if not args.skip_sanity:
        sanity_check(n_trials=100, n_perm=200)

    print("Loading merged hourly data …", flush=True)
    merged = load_merged()
    print(f"  {len(merged):,} hourly records")

    t_jh = FS1.datetime_to_j2000h(merged.index)
    years_of_sample = merged.index.year.values

    var_keys = ([v.strip() for v in args.vars.split(",") if v.strip()]
                or list(FS1.VARS.keys()))

    print(f"\nFitting {len(var_keys)} vars × {len(YEARS_FULL)} years …\n")

    results: List[Dict] = []
    t_all = time.perf_counter()

    for i, vk in enumerate(var_keys, 1):
        vd = FS1.VARS[vk]
        col = vd['col']
        if col not in merged.columns:
            print(f"  [{i:>2}/{len(var_keys)}] {vk:<7} missing, skip")
            continue

        x = merged[col].values.astype(np.float64) * vd['scale']
        print(f"  [{i:>2}/{len(var_keys)}] {vk:<7} {vd['sym']:<6} fitting …",
              end=" ", flush=True)
        t0 = time.perf_counter()

        r = run_variable(vk, vd, t_jh, x, years_of_sample, n_perm=args.n_perm)
        if r is None:
            print("skip (too few years)")
            continue

        results.append(r)

        tr1 = r['trends']['A1']
        print(f"{time.perf_counter() - t0:>5.1f}s   "
              f"t(A1)={tr1['t_stat']:>+5.2f}  "
              f"p={tr1['p_perm']:.4f}  "
              f"τ={tr1['tau']:.4f}")

    print(f"\n  Total runtime: {time.perf_counter() - t_all:.1f}s")

    print_summary(results)
    print_flagged(results)

    def _ser(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray):                return o.tolist()
        if isinstance(o, float) and not math.isfinite(o): return None
        return str(o)

    payload = {
        "meta": {
            "years":       YEARS_FULL,
            "n_annual":    N_ANN,
            "n_diurnal":   N_DIA,
            "cross_terms": CX_YEAR,
            "hac_lags":    HAC_L,
            "n_perm":      args.n_perm,
            "estimator":   "random-effects (DerSimonian-Laird)",
            "permutation": "standardized residual z=(v-ȳ)/σ",
            "version":     "2.0.0",
        },
        "series": results,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_ser)

    import os
    print(f"  JSON → {args.out}  "
          f"({os.path.getsize(args.out) // 1024} KB)\n")


if __name__ == "__main__":
    main()