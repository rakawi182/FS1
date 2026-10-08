#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_multivariate_hac.py
================================================================================
Multivariate structure of the HAC-filtered Fourier coefficient matrix.

Perbedaan terhadap fs1_multivariate.py
--------------------------------------
fs1_multivariate.py memakai semua 118 koefisien per variabel. Dari HAC kita
tahu ~40-70% di antaranya adalah noise (A < 2σ_HAC). Noise ini mengisi
ruang vektor dengan arah acak yang mengaburkan cluster dan PCA loading.

Modul ini membuang koefisien yang tidak lolos uji 2σ_HAC, menyisakan
hanya koefisien yang benar-benar nyata secara statistik.

Input   : fourier_batch_hac.json  (dari FS1_Batch_HAC.py)
Output  : tiga heat-map korelasi + dendrogram + PCA (ke layar)
          fs1_multivariate_hac.json (ringkasan numerik)

TCWV dikecualikan — hanya ~2 tahun coverage, tidak stabil untuk
perbandingan antar-variabel.

Author : MJS · Jolotundo Observatory
Version: 1.0.0
================================================================================
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════════════
# §1  KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════

EXCLUDE_VK = {"TCWV"}      # hanya 2 tahun, tidak stabil
SIGMA_THRESHOLD = 2.0      # uji signifikansi HAC
N_CLUSTERS = 4
N_PCA_COMPS = 8


# ══════════════════════════════════════════════════════════════════════════════
# §2  LOAD & BUILD COEFFICIENT MATRIX (HAC-FILTERED)
# ══════════════════════════════════════════════════════════════════════════════

def load_coefficients(path: str = "fourier_batch_hac.json"):
    """
    Baca JSON HAC, ubah setiap term (A, φ) → (c_re, c_im), buang yang
    tidak lolos uji 2σ_HAC.

    Nilai A dan σ dari JSON adalah per-amplitudo. Konversi ke komponen
    (cos, sin):

        c_re = A · cos(φ)
        c_im = A · sin(φ)

    Flag `sig_hac` di JSON sudah dihitung dengan threshold 2σ. Kita
    pakai flag itu langsung, tidak menghitung ulang.

    Returns
    -------
    data, names, syms, M_ann, M_dia, M_cross, n_sig_per_var
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    data = [r for r in raw if r["vk"] not in EXCLUDE_VK]
    if not data:
        raise SystemExit(f"Tidak ada data setelah exclude {EXCLUDE_VK}")

    n = len(data)
    names = [r["vk"] for r in data]
    syms  = [r["sym"] for r in data]

    # Kolom tetap sama panjang (18·2, 11·2, 30·2) untuk semua variabel;
    # yang berbeda adalah koefisien mana yang non-nol.
    M_ann   = np.zeros((n, 18 * 2))
    M_dia   = np.zeros((n, 11 * 2))
    M_cross = np.zeros((n, 30 * 2))

    n_sig_per_var = []

    for i, r in enumerate(data):
        n_sig = 0
        for j, t in enumerate(r["annual_terms"]):
            if not t.get("sig_hac", False):
                continue
            phi = math.radians(t["phi"])
            M_ann[i, 2*j]     = t["A"] * math.cos(phi)
            M_ann[i, 2*j + 1] = t["A"] * math.sin(phi)
            n_sig += 1

        for j, t in enumerate(r["diurnal_terms"]):
            if not t.get("sig_hac", False):
                continue
            phi = math.radians(t["phi"])
            M_dia[i, 2*j]     = t["A"] * math.cos(phi)
            M_dia[i, 2*j + 1] = t["A"] * math.sin(phi)
            n_sig += 1

        for j, t in enumerate(r["cross_terms"]):
            if not t.get("sig_hac", False):
                continue
            phi = math.radians(t["phi"])
            M_cross[i, 2*j]     = t["A"] * math.cos(phi)
            M_cross[i, 2*j + 1] = t["A"] * math.sin(phi)
            n_sig += 1

        n_sig_per_var.append(n_sig)

    return data, names, syms, M_ann, M_dia, M_cross, n_sig_per_var


# ══════════════════════════════════════════════════════════════════════════════
# §3  COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(M: np.ndarray) -> np.ndarray:
    """
    Cosine similarity antar baris M. Baris yang seluruhnya nol
    menghasilkan NaN → di-set ke 0 (variabel tidak punya koefisien
    signifikan di blok ini).
    """
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    Mn = M / norms
    C = Mn @ Mn.T
    # Baris yang benar-benar nol → similarity dengan semua = 0
    zero = (np.linalg.norm(M, axis=1) < 1e-30)
    for i in np.where(zero)[0]:
        C[i, :] = 0.0
        C[:, i] = 0.0
    return C


# ══════════════════════════════════════════════════════════════════════════════
# §4  HEAT MAP
# ══════════════════════════════════════════════════════════════════════════════

def print_heatmap(C: np.ndarray, syms: List[str], title: str) -> None:
    n = len(syms)
    print(f"\n  {title}")
    print("  " + "─" * (6 + 4 * n))
    print("  " + " " * 6 + "".join(f"{s:>4}" for s in syms))

    def cell(v):
        if v >= 0.9:  return " ██ "
        if v >= 0.7:  return " ▓▓ "
        if v >= 0.5:  return " ▒▒ "
        if v >= 0.3:  return " ░░ "
        if v >= 0.1:  return " ·· "
        if v >= -0.1: return "    "
        if v >= -0.3: return " -- "
        if v >= -0.5: return " == "
        return " ## "

    for i, s in enumerate(syms):
        row = f"  {s:>4}  "
        for j in range(n):
            row += cell(C[i, j])
        print(row)


def print_legend() -> None:
    print("\n  Legend: ██ ≥0.9  ▓▓ ≥0.7  ▒▒ ≥0.5  ░░ ≥0.3  ·· ≥0.1")
    print("          (blank) ≥−0.1  -- ≤−0.3  == ≤−0.5  ## < −0.5")


# ══════════════════════════════════════════════════════════════════════════════
# §5  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def cluster_variables(C: np.ndarray, n_clusters: int = 4):
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


def print_clusters(labels: np.ndarray, syms: List[str]) -> None:
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    print(f"\n  CLUSTER (average linkage pada 1 − cos_sim, HAC-filtered)")
    print("  " + "─" * 78)
    for lab in sorted(groups.keys()):
        members = groups[lab]
        member_str = ", ".join(syms[i] for i in members)
        print(f"    Cluster {lab}:  {member_str}")


# ══════════════════════════════════════════════════════════════════════════════
# §6  PCA
# ══════════════════════════════════════════════════════════════════════════════

def pca_report(M: np.ndarray, syms: List[str]) -> None:
    n_var = M.shape[0]
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Ms = (M - mean) / std

    n_comp = min(N_PCA_COMPS, n_var - 1)
    if n_comp < 1:
        print("\n  [PCA butuh ≥2 variabel]")
        return

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Ms)
    var_ratio = pca.explained_variance_ratio_
    cum = np.cumsum(var_ratio)

    print(f"\n  PCA pada matriks koefisien HAC-filtered  (shape {M.shape})")
    print("  " + "─" * 78)
    print(f"  {'PC':>4}  {'Var %':>8}  {'Cum %':>8}   Top-4 beban variabel")
    print("  " + "─" * 78)

    for i in range(n_comp):
        order = np.argsort(-np.abs(scores[:, i]))
        top = ", ".join(syms[j] for j in order[:4])
        print(f"  {i+1:>4}  {var_ratio[i]*100:>7.2f}%  "
              f"{cum[i]*100:>7.2f}%   {top}")

    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"\n  → {n_90} komponen untuk 90% varians")
    print(f"  → {n_95} komponen untuk 95% varians")


# ══════════════════════════════════════════════════════════════════════════════
# §7  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    W = 78
    print("=" * W)
    print("  FS1 MULTIVARIATE — HAC-FILTERED COEFFICIENTS")
    print("=" * W)

    data, names, syms, M_ann, M_dia, M_cross, n_sig = load_coefficients()

    print(f"\n  Variabel  : {len(data)}  (excluded: {sorted(EXCLUDE_VK)})")
    print(f"  Threshold : |A| > {SIGMA_THRESHOLD}σ_HAC")
    print(f"\n  {'var':<8}{'sym':<7}{'n_sig / 59'}")
    print("  " + "─" * 40)
    for r, ns in zip(data, n_sig):
        pct = 100.0 * ns / 59.0
        print(f"  {r['vk']:<8}{r['sym']:<7}{ns:>3} / 59   ({pct:>5.1f}%)")

    M = np.hstack([M_ann, M_dia, M_cross])

    C_total = cosine_sim(M)
    C_ann   = cosine_sim(M_ann)
    C_dia   = cosine_sim(M_dia)
    C_cross = cosine_sim(M_cross)

    print_heatmap(C_total, syms, "TOTAL CORRELATION (HAC-filtered)")
    print_heatmap(C_ann,   syms, "ANNUAL CORRELATION")
    print_heatmap(C_dia,   syms, "DIURNAL CORRELATION")
    print_heatmap(C_cross, syms, "CROSS CORRELATION")
    print_legend()

    _, labels = cluster_variables(C_total, n_clusters=N_CLUSTERS)
    print_clusters(labels, syms)

    pca_report(M, syms)

    # ---- JSON export (ringkas) ----
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    out = {
        "meta": {
            "source": "fourier_batch_hac.json",
            "sigma_threshold": SIGMA_THRESHOLD,
            "exclude": sorted(EXCLUDE_VK),
            "n_vars": len(data),
        },
        "significance": {
            r["vk"]: {"sym": r["sym"], "n_sig": int(ns), "n_total": 59}
            for r, ns in zip(data, n_sig)
        },
        "cosine_total": C_total.tolist(),
        "cosine_annual": C_ann.tolist(),
        "cosine_diurnal": C_dia.tolist(),
        "cosine_cross": C_cross.tolist(),
        "cluster_labels": {r["vk"]: int(l)
                           for r, l in zip(data, labels)},
    }

    with open("fs1_multivariate_hac.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=_ser)

    import os
    print(f"\n  JSON → fs1_multivariate_hac.json  "
          f"({os.path.getsize('fs1_multivariate_hac.json') // 1024} KB)\n")


if __name__ == "__main__":
    main()