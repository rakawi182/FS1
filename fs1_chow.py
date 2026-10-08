#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_chow.py  —  Uji Chow breakpoint untuk 3 series flagged.
================================================================================
Input   : fs1_nonstationary.json (dari fs1_nonstationary.py v2)
Output  : tsc_chow.json  +  tabel di layar

Untuk setiap series (SMSH A₁, SMDP A₁, SMSH σ_resid):

  H₀ :  A(Y) = α + β·Y              [satu garis, drift]
  H₁(τ): A(Y) = α₁ + β₁·Y, Y ≤ τ
              = α₂ + β₂·Y, Y > τ    [dua segmen, patah di τ]

Statistik: F_Chow(τ) = ((SSE₀ − SSE₁)/2) / (SSE₁/(n−4))
          dibobot dengan 1/σ_Y² (weighted least squares).

Breakpoint optimal: τ* = argmax_τ F_Chow(τ), τ ∈ {2020, 2021, 2022, 2023}.

Nilai-p: permutation max-F.
  1. Fit H₀ ke data.
  2. Ambil z_i = resid_i · √w_i.
  3. Untuk 5000 iterasi: permutasi z, rekonstruksi y, hitung max_τ F.
  4. p = P(max_τ F_perm ≥ max_τ F_obs).

Ini mengontrol false-discovery atas pencarian τ. Tidak pakai tabel F
karena n=9 terlalu kecil untuk asimptotik.
================================================================================
"""

import json
import math
import numpy as np


CANDIDATE_BREAKS = [2020, 2021, 2022, 2023]
N_PERM = 5000
SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Weighted least squares: y = a + b·x
# ──────────────────────────────────────────────────────────────────────────────

def wls(x, y, w):
    X = np.column_stack([np.ones_like(x), x])
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y * w)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return None
    yhat = X @ beta
    sse = float(np.sum(w * (y - yhat) ** 2))
    return beta, sse, yhat


# ──────────────────────────────────────────────────────────────────────────────
# Chow F-statistic untuk satu kandidat τ
# ──────────────────────────────────────────────────────────────────────────────

def chow_F(x, y, w, tau):
    """
    F-statistik Chow untuk break di τ.
    Model nol  : 2 parameter (α, β)  → df_resid = n - 2
    Model alt  : 4 parameter         → df_resid = n - 4
    df1 = 2, df2 = n - 4.
    """
    fit0 = wls(x, y, w)
    if fit0 is None:
        return 0.0, None
    _, sse0, _ = fit0

    mask_L = x <= tau
    mask_R = x > tau
    if mask_L.sum() < 3 or mask_R.sum() < 3:
        return 0.0, None

    fitL = wls(x[mask_L], y[mask_L], w[mask_L])
    fitR = wls(x[mask_R], y[mask_R], w[mask_R])
    if fitL is None or fitR is None:
        return 0.0, None

    _, sseL, _ = fitL
    _, sseR, _ = fitR
    sse1 = sseL + sseR

    n = len(x)
    df2 = n - 4
    if df2 <= 0 or sse1 <= 0:
        return 0.0, None

    F = ((sse0 - sse1) / 2.0) / (sse1 / df2)
    return float(F), (fitL[0], fitR[0], sseL, sseR)


# ──────────────────────────────────────────────────────────────────────────────
# Uji Chow dengan permutation max-F
# ──────────────────────────────────────────────────────────────────────────────

def chow_test(x, y, w, n_perm=N_PERM, seed=SEED):
    # Observed: F untuk setiap τ, ambil yang terbesar
    F_tau = {}
    fits_tau = {}
    for tau in CANDIDATE_BREAKS:
        F, fit = chow_F(x, y, w, tau)
        F_tau[tau] = F
        fits_tau[tau] = fit

    tau_star = max(F_tau, key=F_tau.get)
    F_obs = F_tau[tau_star]

    # Permutation null: residual terstandardisasi dari H₀
    fit0 = wls(x, y, w)
    beta0, _, yhat0 = fit0
    resid0 = y - yhat0
    z = resid0 * np.sqrt(w)          # terstandardisasi
    inv_sqrt_w = 1.0 / np.sqrt(w)

    rng = np.random.default_rng(seed)
    n_exceed = 0
    for _ in range(n_perm):
        zp = z[rng.permutation(len(z))]
        y_perm = yhat0 + zp * inv_sqrt_w
        F_max = 0.0
        for tau in CANDIDATE_BREAKS:
            F, _ = chow_F(x, y_perm, w, tau)
            if F > F_max:
                F_max = F
        if F_max >= F_obs:
            n_exceed += 1

    p_perm = n_exceed / n_perm
    return {
        'tau_star': int(tau_star),
        'F_obs': F_obs,
        'p_perm': p_perm,
        'F_tau': {int(k): v for k, v in F_tau.items()},
        'fit_obs': fits_tau[tau_star],
        'beta0': beta0.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Ekstraksi series dari fs1_nonstationary.json
# ──────────────────────────────────────────────────────────────────────────────

def load_series(path, vk, field):
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    for s in data['series']:
        if s['vk'] != vk:
            continue
        years = np.array([y['year'] for y in s['by_year']], dtype=float)
        vals = np.array([y[field] for y in s['by_year']], dtype=float)
        if field == 'A1':
            sig = np.array([y['sig_A1'] for y in s['by_year']], dtype=float)
        elif field == 'sigma_resid':
            N_arr = np.array([y['N'] for y in s['by_year']], dtype=float)
            sig = vals / np.sqrt(2.0 * N_arr)
        else:
            raise ValueError(f"field tidak didukung: {field}")
        return years, vals, sig
    raise SystemExit(f"series {vk} tidak ditemukan")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cases = [
        ("SMDP", "A1",          "θ₇₂₈  amplitude annual k=1"),
        ("SMSH", "A1",          "θ₀₇  amplitude annual k=1"),
        ("SMSH", "sigma_resid", "θ₀₇  residual σ per tahun"),
    ]

    print("=" * 100)
    print("  UJI CHOW BREAKPOINT — drift vs step")
    print("=" * 100)
    print(f"  Kandidat τ: {CANDIDATE_BREAKS}    Permutasi: {N_PERM}")
    print()

    results = []

    for vk, field, label in cases:
        years, vals, sig = load_series("fs1_nonstationary.json", vk, field)
        w = 1.0 / np.maximum(sig, 1e-30) ** 2

        r = chow_test(years, vals, w)
        r['vk'] = vk
        r['field'] = field
        r['label'] = label
        r['years'] = years.tolist()
        r['values'] = vals.tolist()
        r['sigmas'] = sig.tolist()

        results.append(r)

        # ── Ringkasan per series ─────────────────────────────────────
        print("─" * 100)
        print(f"  {vk:<6} {field:<12} {label}")
        print("─" * 100)
        print(f"  Data:")
        for Y, v, s in zip(years, vals, sig):
            print(f"    {int(Y)}  value = {v:>12.6f}   σ = {s:>10.6f}")

        print(f"\n  F_Chow per kandidat τ:")
        for tau in CANDIDATE_BREAKS:
            marker = "  ← τ*" if tau == r['tau_star'] else ""
            print(f"    τ = {tau}   F = {r['F_tau'][tau]:>8.3f}{marker}")

        print(f"\n  τ* = {r['tau_star']}   F_obs = {r['F_obs']:.3f}   p_perm = {r['p_perm']:.4f}")

        # Fit dua segmen pada τ*
        fit = r['fit_obs']
        if fit is not None:
            betaL, betaR, _, _ = fit
            tau = r['tau_star']
            nL = int((years <= tau).sum())
            nR = int((years > tau).sum())
            print(f"\n  Fit dua segmen di τ* = {tau}:")
            print(f"    Segmen 1 ({int(years[0])}–{tau}, n={nL}):  "
                  f"A = {betaL[0]:+.6f} {betaL[1]:+.6f}·Y   "
                  f"(slope = {betaL[1]:+.6f} /tahun)")
            print(f"    Segmen 2 ({tau+1}–{int(years[-1])}, n={nR}):  "
                  f"A = {betaR[0]:+.6f} {betaR[1]:+.6f}·Y   "
                  f"(slope = {betaR[1]:+.6f} /tahun)")
            print(f"\n  Fit satu garis (H₀):")
            b0 = r['beta0']
            print(f"    A = {b0[0]:+.6f} {b0[1]:+.6f}·Y   "
                  f"(slope = {b0[1]:+.6f} /tahun)")
        print()

    # ── Ringkasan akhir ─────────────────────────────────────────────
    print("=" * 100)
    print("  RINGKASAN")
    print("=" * 100)
    print(f"\n  {'series':<20}{'τ*':>6}{'F_obs':>10}{'p_perm':>10}   "
          f"{'slope S1':>12}{'slope S2':>12}   {'verdict':>12}")
    print("  " + "─" * 90)

    for r in results:
        fit = r['fit_obs']
        if fit is None:
            continue
        betaL, betaR, _, _ = fit
        s1, s2 = betaL[1], betaR[1]
        sig_flag = "*" if r['p_perm'] < 0.05 else " "
        if r['p_perm'] < 0.05 and abs(s1) < 0.005 and abs(s2) < 0.005:
            verdict = "step"
        elif r['p_perm'] < 0.05:
            verdict = "drift+step"
        else:
            verdict = "tidak sig"

        label = f"{r['vk']}.{r['field']}"
        print(f"  {label:<20}{r['tau_star']:>6}{r['F_obs']:>10.3f}"
              f"{r['p_perm']:>10.4f}{sig_flag} "
              f"{s1:>+12.6f}{s2:>+12.6f}   {verdict:>12}")

    print("\n  * = p_perm < 0.05")
    print("  step    : kedua slope ≈ 0, perbedaan hanya di intercept (lompatan)")
    print("  drift   : slope signifikan di kedua segmen dengan tanda sama")
    print()

    # ── JSON ────────────────────────────────────────────────────────
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return str(o)

    with open("tsc_chow.json", "w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "candidate_breaks": CANDIDATE_BREAKS,
                "n_perm": N_PERM,
                "seed": SEED,
                "test": "Chow F (weighted), permutation max-F",
            },
            "results": results,
        }, fh, indent=2, ensure_ascii=False, default=_ser)

    print("  JSON → tsc_chow.json\n")


if __name__ == "__main__":
    main()