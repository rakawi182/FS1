#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs1_multivariate.py
================================================================================
Multivariate structure of the Fourier coefficient matrix.

Loads fourier_batch_bayes.json and analyses the joint structure:

    1. Correlation matrix between variables (cosine similarity of
       coefficient vectors, separately for annual / diurnal / cross).
    2. Hierarchical clustering (average linkage) on 1 − correlation.
    3. PCA on the harmonic-coefficient matrix
       (DC excluded — no structural information).

Goal: identify latent modes shared across variables.

Data coverage note
------------------
TCWV (W_c) is EXCLUDED from this analysis. In the IFS-HRES 2017–2026
record, W_c is only available from ~Sept 2024 (~21 % of the period),
so its Fourier coefficients are fitted on ~2 effective annual cycles
and are statistically unstable. Including it would inject an
anomalously large σ_noise into the correlation structure and bias
both the hierarchical clustering and the PCA loadings.

This reduces the working set from 14 to 13 variables.

Author : MJS · Jolotundo Observatory
Version: 1.1.0
================================================================================
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════════════
# §0  EXCLUSION LIST
# ══════════════════════════════════════════════════════════════════════════════

# Variable keys to exclude from the multivariate analysis.
#   W_c : TCWV — only ~21 % temporal coverage (from ~Sept 2024),
#         Fourier coefficients unreliable for cross-variable comparison.
EXCLUDE_VK = {"TCWV"}


# ══════════════════════════════════════════════════════════════════════════════
# §1  LOAD & BUILD COEFFICIENT MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def load_coefficients(path: str = "fourier_batch_bayes.json"):
    """
    Read the batch JSON and rebuild the full coefficient matrix.

    Each term is stored as (A, φ) and converted back to (c_re, c_im):

        c_re = A · cos(φ)
        c_im = A · sin(φ)

    This is a bijection: (A, φ) ↔ (c_re, c_im).  The complex form is
    linear, so it composes naturally under PCA / cosine similarity.

    Variables in EXCLUDE_VK are silently skipped.

    Returns
    -------
    data, names, syms, M_ann, M_dia, M_cross
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Filter out excluded variables
    data = [r for r in raw if r["vk"] not in EXCLUDE_VK]

    if not data:
        raise SystemExit(
            f"[ERROR] No variables left after excluding {EXCLUDE_VK}. "
            f"Check the JSON content."
        )

    n = len(data)
    names = [r["vk"] for r in data]
    syms  = [r["sym"] for r in data]

    M_ann   = np.zeros((n, 18 * 2))
    M_dia   = np.zeros((n, 11 * 2))
    M_cross = np.zeros((n, 30 * 2))

    for i, r in enumerate(data):
        for j, t in enumerate(r["annual_terms"]):
            phi = math.radians(t["phi"])
            M_ann[i, 2*j]     = t["A"] * math.cos(phi)
            M_ann[i, 2*j + 1] = t["A"] * math.sin(phi)
        for j, t in enumerate(r["diurnal_terms"]):
            phi = math.radians(t["phi"])
            M_dia[i, 2*j]     = t["A"] * math.cos(phi)
            M_dia[i, 2*j + 1] = t["A"] * math.sin(phi)
        for j, t in enumerate(r["cross_terms"]):
            phi = math.radians(t["phi"])
            M_cross[i, 2*j]     = t["A"] * math.cos(phi)
            M_cross[i, 2*j + 1] = t["A"] * math.sin(phi)

    return data, names, syms, M_ann, M_dia, M_cross


# ══════════════════════════════════════════════════════════════════════════════
# §2  COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(M: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between rows of M.

    C[i, j] = (M[i] · M[j]) / (‖M[i]‖ · ‖M[j]‖)

    Values in [-1, 1]:
        +1  →  identical spectral shape
         0  →  orthogonal (90° phase apart or disjoint bands)
        -1  →  anti-phase
    """
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    Mn = M / norms
    return Mn @ Mn.T


# ══════════════════════════════════════════════════════════════════════════════
# §3  ASCII HEAT MAP
# ══════════════════════════════════════════════════════════════════════════════

def print_heatmap(C: np.ndarray, syms: List[str], title: str):
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


def print_legend():
    print("\n  Legend: ██ ≥0.9  ▓▓ ≥0.7  ▒▒ ≥0.5  ░░ ≥0.3  ·· ≥0.1")
    print("          (blank) ≥-0.1  -- ≤-0.3  == ≤-0.5  ## < -0.5")


# ══════════════════════════════════════════════════════════════════════════════
# §4  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def cluster_variables(C: np.ndarray, n_clusters: int = 4):
    """
    Average-linkage clustering on distance d = 1 − C.

    Returns
    -------
    Z      : linkage matrix
    labels : int cluster assignment per variable (1..n_clusters)
    """
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


def print_clusters(labels: np.ndarray, names: List[str], syms: List[str]):
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    print(f"\n  CLUSTERS (average linkage on 1 − cos_sim)")
    print("  " + "─" * 78)
    for lab in sorted(groups.keys()):
        members = groups[lab]
        member_str = ", ".join(f"{syms[i]}" for i in members)
        print(f"    Cluster {lab}:  {member_str}")


# ══════════════════════════════════════════════════════════════════════════════
# §5  PCA
# ══════════════════════════════════════════════════════════════════════════════

def pca_report(M: np.ndarray, syms: List[str]):
    """
    PCA on the coefficient matrix M (n_var × n_coeff).

    Columns are z-scored first, so each coefficient position contributes
    equally.  With n_var = 13 samples the rank is 12, so at most 12
    components are meaningful; we report the leading 8.

    Parameters
    ----------
    M    : (n_var, P) coefficient matrix
    syms : list of variable symbols (length n_var)
    """
    n_var = M.shape[0]

    mean = M.mean(axis=0)
    std  = M.std(axis=0)
    std  = np.where(std < 1e-12, 1.0, std)
    Ms = (M - mean) / std

    n_comp = min(8, n_var - 1)
    if n_comp < 1:
        print("\n  [PCA skipped — need ≥2 variables.]")
        return None, None, None

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Ms)
    var_ratio = pca.explained_variance_ratio_
    cum = np.cumsum(var_ratio)

    print(f"\n  PCA ON COEFFICIENT MATRIX  (shape {M.shape})")
    print("  " + "─" * 78)
    print(f"  {'PC':>4}  {'Var %':>8}  {'Cum %':>8}   Top-4 variables")
    print("  " + "─" * 78)

    for i in range(n_comp):
        order = np.argsort(-np.abs(scores[:, i]))
        top = ", ".join(f"{syms[j]}" for j in order[:4])
        print(f"  {i+1:>4}  {var_ratio[i]*100:>7.2f}%  "
              f"{cum[i]*100:>7.2f}%   {top}")

    # Effective dimensionality
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"\n  → {n_90} components explain 90 % of variance")
    print(f"  → {n_95} components explain 95 % of variance")

    return pca, scores, var_ratio


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    W = 78
    print("=" * W)
    print("  FS1 MULTIVARIATE STRUCTURE")
    print("=" * W)

    data, names, syms, M_ann, M_dia, M_cross = load_coefficients()

    n_var = len(data)
    print(f"\n  Loaded {n_var} variables (excluded: {sorted(EXCLUDE_VK)})")
    print(f"  Variables: {', '.join(syms)}")

    M = np.hstack([M_ann, M_dia, M_cross])   # (n_var, 118)

    # Correlation matrices
    C_total = cosine_sim(M)
    C_ann   = cosine_sim(M_ann)
    C_dia   = cosine_sim(M_dia)
    C_cross = cosine_sim(M_cross)

    print_heatmap(C_total, syms, "TOTAL CORRELATION (59 terms)")
    print_heatmap(C_ann, syms, "ANNUAL CORRELATION (18 terms)")
    print_heatmap(C_dia, syms, "DIURNAL CORRELATION (11 terms)")
    print_heatmap(C_cross, syms, "CROSS CORRELATION (30 terms)")
    print_legend()

    # Clustering
    _, labels = cluster_variables(C_total, n_clusters=4)
    print_clusters(labels, names, syms)

    # PCA
    pca_report(M, syms)


if __name__ == "__main__":
    main()