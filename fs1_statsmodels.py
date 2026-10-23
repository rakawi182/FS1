#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_statsmodels.py  ·  FS1 Fourier-series inference, statsmodels edition
(Standalone Run Edition - Auto-Batching with INSTANT SAVE)
"""
import json
import math
import os
import time
import warnings
import gc

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import solve_toeplitz
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acovf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── KONFIGURASI PATH (HARDCODED) ─────────────────────────────────────────────
DATA_DIR = "/storage/emulated/0/Download/openmeteo_data/"
OUT_JSON = "/storage/emulated/0/Download/Meteo_JOBSV/fourier/fs1_statsmodels.json"
HTML_IN = "/storage/emulated/0/Download/Meteo_JOBSV/fourier/FS1SpectralCore.html"
HTML_OUT = "/storage/emulated/0/Download/Meteo_JOBSV/fourier/FS1SpectralCore_v2.html"

TARGET = (-7.5220, 112.5661)
STATIONS = {
    "A": (-7.5571175, 112.55735, "ecmwf_ifs9_hourly2017-2026_-7.56S112.56E28m.csv"),
    "B": (-7.486819, 112.53821, "ecmwf_ifs9_hourly2017-2026_-7.49S112.54E28m.csv"),
}
REF = pd.Timestamp("2000-01-01 12:00:00")
TY = 365.24219 * 24.0
WA, WD = 2 * math.pi / TY, 2 * math.pi / 24.0
N_A, N_D = 18, 11
CROSS = ([(1, m) for m in (1, 2, 3, 4, 5, 6)] + [(2, m) for m in (1, 2, 3, 4)] +
         [(3, m) for m in (1, 2, 3)] + [(4, m) for m in (1, 2)])
CROSS = CROSS + [(k, -m) for k, m in CROSS]
TERMS = ([(k, 0) for k in range(1, N_A + 1)] + [(0, m) for m in range(1, N_D + 1)] + CROSS)
P = 1 + 2 * len(TERMS)

VARS = {
    "TEMP": ("temperature_2m", 1), "TDEW": ("dew_point_2m", 1),
    "RH": ("relative_humidity_2m", 1), "VPD": ("vapour_pressure_deficit", 1),
    "PSURF": ("surface_pressure", 1), "CLOUD": ("cloud_cover", 1),
    "SW": ("shortwave_radiation", 1), "WIND": ("wind_speed_10m", 1 / 3.6),
    "ET0": ("et0_fao_evapotranspiration", 1), "PREC": ("precipitation", 1),
    "TCWV": ("total_column_integrated_water_vapour", 1),
    "SMSH": ("soil_moisture_0_to_7cm", 1), "SMDP": ("soil_moisture_7_to_28cm", 1),
    "SUN": ("sunshine_duration", 1 / 3600.0),
}

BATCHES = [
    ["TEMP", "TDEW", "RH", "VPD", "PSURF", "WIND"],   # Batch 1: Meteorologi Dasar
    ["CLOUD", "SW", "ET0", "SUN", "TCWV"],             # Batch 2: Radiasi & Awan
    ["PREC", "SMSH", "SMDP"]                           # Batch 3: Tanah & Hujan
]

ALPHA_FDR, PMAX = 0.05, 48


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        hdr = next(i for i, ln in enumerate(fh) if ln.lstrip().startswith("time"))
    df = pd.read_csv(path, skiprows=hdr, low_memory=False)
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def haversine(p1, l1, p2, l2):
    p1, p2, dl = map(math.radians, (p1, p2, l2 - l1))
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(a))


def load_target(data_dir, vars_to_load):
    st = {k: read_csv(os.path.join(data_dir, v[2])) for k, v in STATIONS.items()}
    d = {k: haversine(*TARGET, v[0], v[1]) for k, v in STATIONS.items()}
    w = {k: d[k] ** -2 for k in d}
    s = sum(w.values())
    w = {k: w[k] / s for k in w}
    idx = pd.date_range(max(x.index.min() for x in st.values()),
                        min(x.index.max() for x in st.values()), freq="1h")
    a, b = st["A"].reindex(idx), st["B"].reindex(idx)
    out = {}
    for vid in vars_to_load:
        col, sc = VARS[vid]
        if col in a and col in b:
            out[vid] = (w["A"] * a[col].values + w["B"] * b[col].values) * sc
        elif col in a:
            out[vid] = a[col].values * sc
        elif col in b:
            out[vid] = b[col].values * sc
    return idx, out


def design(t):
    X = np.empty((len(t), P))
    X[:, 0] = 1.0
    for i, (k, m) in enumerate(TERMS):
        arg = (k * WA + m * WD) * t
        X[:, 1 + 2 * i], X[:, 2 + 2 * i] = np.cos(arg), np.sin(arg)
    return X


def hac_fit(y, X, w, L):
    return sm.WLS(np.where(w > 0, y, 0.0), X, weights=w).fit(
        cov_type="HAC", cov_kwds={"maxlags": L, "kernel": "bartlett", "use_correction": True})


def longest_run(a):
    f = np.isfinite(a).astype(np.int8)
    ix = np.flatnonzero(np.diff(np.r_[0, f, 0]))
    s, e = ix[::2], ix[1::2]
    j = int(np.argmax(e - s))
    return a[s[j]:e[j]]


def select_ar(acov, n):
    best = (np.inf, 1, None)
    for p in range(1, PMAX + 1):
        phi = solve_toeplitz(acov[:p], acov[1:p + 1])
        s2 = acov[0] - phi @ acov[1:p + 1]
        bic = n * math.log(max(s2, 1e-300)) + p * math.log(n)
        if bic < best[0]:
            best = (bic, p, phi)
    return best[2], best[1]


def whiten(a, phi):
    out = a.copy()
    for i, ph in enumerate(phi, 1):
        out[i:] -= ph * a[:-i]
    out[:len(phi)] = np.nan
    return out


def amp_phase(a, b, S):
    A = math.hypot(a, b)
    th = np.array([a, b])
    W = float(th @ np.linalg.solve(S, th))
    u = th / A
    wv = np.array([b, -a]) / A ** 2
    return (A, math.degrees(math.atan2(-b, a)) % 360.0,
            math.sqrt(float(u @ S @ u)), math.degrees(math.sqrt(float(wv @ S @ wv))),
            W, math.exp(-W / 2.0))


def analyse(y, t, X, L_OLS=168, L_GLS=48):
    ok = np.isfinite(y)
    n = int(ok.sum())
    w = ok.astype(float)

    f = hac_fit(y, X, w, L_OLS)
    th, C_ols = np.asarray(f.params), np.asarray(f.cov_params())
    u = np.where(ok, np.where(ok, y, 0.0) - X @ th, np.nan)
    rs = math.sqrt(np.nansum(u ** 2) / (n - P))

    yo, Xo = y[ok], X[ok]
    r2 = [float(sm.OLS(yo, Xo[:, :c]).fit().rsquared) for c in (1 + 2 * N_A, 1 + 2 * (N_A + N_D), P)]

    ac = acovf(u, adjusted=False, demean=False, fft=True, missing="conservative", nlag=PMAX)
    phi, p = select_ar(ac, n)
    yw, Xw = whiten(np.where(ok, y, np.nan), phi), whiten(X, phi)
    okw = np.isfinite(yw)
    fg = hac_fit(yw, np.nan_to_num(Xw), okw.astype(float), L_GLS)
    C_g = np.asarray(fg.cov_params())
    uw = np.where(okw, np.where(okw, yw, 0.0) - np.nan_to_num(Xw) @ np.asarray(fg.params), np.nan)

    td = (t - t[ok].mean()) / (8766.0 * 10.0)
    Xt = np.column_stack([X, td])
    ft = hac_fit(yw, np.nan_to_num(whiten(Xt, phi)), okw.astype(float), L_GLS)
    tr_b, tr_p = float(ft.params[-1]), float(ft.pvalues[-1])

    XtX = Xo.T @ Xo
    C_iid = rs ** 2 * np.linalg.pinv(XtX)

    rows, ratios = [], []
    for i, (k, m) in enumerate(TERMS):
        ix = [1 + 2 * i, 2 + 2 * i]
        Bo, Bg, Bi = (C[np.ix_(ix, ix)] for C in (C_ols, C_g, C_iid))
        S = Bo if np.trace(Bo) >= np.trace(Bg) else Bg
        A, ph, sA, sph, W, pv = amp_phase(th[ix[0]], th[ix[1]], S)
        ratios.append(math.sqrt(np.trace(S) / np.trace(Bi)))
        rows.append([k, m, A, ph, sA, sph, W, pv])
    pv = np.array([r[7] for r in rows])
    rej, q, _, _ = multipletests(pv, alpha=ALPHA_FDR, method="fdr_bh")

    seg = longest_run(u)
    lb = lambda x: float(acorr_ljungbox(x, lags=[24], return_df=True).loc[24, "lb_pvalue"])
    return {
        "n": n, "mean": float(yo.mean()), "sd": float(yo.std(ddof=1)), "rs": rs,
        "r2a": r2[0], "r2d": r2[1], "r2f": r2[2],
        "ar_p": p, "rho1": float(ac[1] / ac[0]), "dw": float(durbin_watson(seg)),
        "lb_pre": lb(seg), "lb_post": lb(longest_run(uw)),
        "infl": float(np.median(ratios)), "n_sig": int(rej.sum()),
        "trend_dec": tr_b, "trend_p": tr_p,
        "terms": [r + [float(qq), bool(rj)] for r, qq, rj in zip(rows, q, rej)],
    }


def inject(html_in, res, html_out):
    s = open(html_in, encoding="utf-8").read()
    i0 = s.index("const V=[") + len("const V=")
    i1 = s.index("];", i0) + 1
    V = json.loads(s[i0:i1])
    for v in V:
        r = res.get(v["id"])
        if not r:
            continue
        new = {(a, b): (A, ph, sA, q) for a, b, A, ph, sA, _, _, _, q, _ in r["terms"]}
        old = {(t[0], t[1]): t for t in v["T"]}
        dA = max(abs(new[k][0] - o[2]) / max(o[2], 1e-9) for k, o in old.items() if k in new and o[2] > 1e-3)
        print(f"  {v['id']:6s} max rel.dA vs lama = {dA:.3e}")
        v.update(mean=round(r["mean"], 6), sd=round(r["sd"], 5), rs=round(r["rs"], 5),
                 r2a=round(r["r2a"], 4), r2d=round(r["r2d"], 4), r2f=round(r["r2f"], 4),
                 sp=round(float(np.median([t[4] for t in r["terms"]])), 6),
                 ar=[r["ar_p"], round(r["rho1"], 4)], dw=round(r["dw"], 3),
                 ns=r["n_sig"], infl=round(r["infl"], 2),
                 tr=[float("%.4g" % r["trend_dec"]), float("%.3g" % r["trend_p"])])
        v["T"] = [[k, m, round(new[(k, m)][0], 6), round(new[(k, m)][1], 3),
                   float("%.4g" % new[(k, m)][2]),
                   round(min(-math.log10(max(new[(k, m)][3], 1e-99)), 99), 2)] for (k, m) in old]
    blob = ",\n".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) for v in V)
    open(html_out, "w", encoding="utf-8").write(s[:i0] + "[\n" + blob + "\n" + s[i1:])


def main():
    print("=== MEMULAI PROSES ANALISIS FOURIER (AUTO-BATCH & INSTANT SAVE) ===")
    
    all_res = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, 'r') as f:
                all_res = json.load(f).get("series", {})
            print(f"Menemukan progres sebelumnya. {len(all_res)} variabel sudah selesai.")
            print(f"Variabel yang sudah ada: {', '.join(all_res.keys())}")
        except Exception:
            pass

    for b_idx, batch_vars in enumerate(BATCHES):
        vars_to_process = [v for v in batch_vars if v not in all_res]
        if not vars_to_process:
            print(f"\n[Batch {b_idx+1}] Dilewati (semua sudah selesai).")
            continue

        print(f"\n--- MEMPROSES BATCH {b_idx+1} ---")
        
        idx, data = load_target(DATA_DIR, vars_to_process)
        t = (idx - REF).total_seconds().values / 3600.0
        X = design(t)
        
        for vid in vars_to_process:
            t0 = time.perf_counter()
            all_res[vid] = analyse(data[vid], t, X, 168, 48)
            r = all_res[vid]
            print(f"{vid:6s} N={r['n']:6d} AR({r['ar_p']:2d}) rho1={r['rho1']:.3f} DW={r['dw']:.3f} "
                  f"infl={r['infl']:.2f}x sig={r['n_sig']}/{len(TERMS)} R2f={r['r2f']:.4f} "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)
            
            # --- PERBAIKAN: SIMPAN JSON SETIAP SELESAI 1 VARIABEL ---
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as f_out:
                json.dump({"meta": {"lags_ols": 168, "lags_gls": 48, "fdr": ALPHA_FDR, "terms": len(TERMS)}, "series": all_res}, f_out)
            print(f"  -> [Tersimpan ke JSON]")
            
            # Bersihkan data variabel ini dari RAM
            del data[vid]
            gc.collect()
            
        del idx, t, X
        gc.collect()
        print(f"[Batch {b_idx+1} Selesai dan RAM dibersihkan.]")
        time.sleep(1)

    print("\n=== SEMUA BATCH SELESAI. MENYUNTIKKAN KE HTML ===")
    try:
        inject(HTML_IN, all_res, HTML_OUT)
        print(f"SUKSES BESAR! File HTML baru tersimpan di:\n{HTML_OUT}")
    except Exception as e:
        print(f"Gagal menyuntikkan ke HTML: {e}")

if __name__ == "__main__":
    main()