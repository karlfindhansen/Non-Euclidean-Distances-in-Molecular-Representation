"""
Controlled Head-to-Head Benchmark: Separation & Prediction on QM9

One joint-stratified sample (gap × heavy-atom count, N≈1200).
Distance matrices built and cached once — PH's ~38-min cost is paid once.
Every representation scored on both tasks under one shared protocol:

  Separation  — kNN property-tercile agreement + Rsep in ambient distance space
                (property-labelled, so PH has no home-field advantage)
  Prediction  — KRR R²/MAE on gap and mu (10 repeated stratified 80/20 splits)

Primary comparison: ALL methods → Laplacian kernel (median bandwidth, inner-CV α).
  This is the only kernel both flat and curved metrics admit; the flat-vs-curved
  comparison is finally clean.
Secondary table  : flat methods also with their best-fit kernel (Tanimoto direct,
  cosine-RBF, Euclidean-RBF). CLEARLY FLAGGED — not the controlled comparison.

Method rows (descriptor × geometry):
  2D floor      : Morgan–Tanimoto distance, ChemProp–cosine distance
  SOAP / MACE   : Euclidean (avg), W₁, W₂, REMatch (α=0.1 fixed),
                  Grassmann geodesic/chordal, Riemann AIRM/LE, PH-descriptor
  Reference     : PH on 3D coordinates — topology detector, not a competitor

Statistics: 10 repeated stratified 80/20 splits (identical across all methods).
  Report mean ± 95% CI; paired Wilcoxon signed-rank test for every "A > B" claim.
Size analysis : every metric reported for ≤6 heavy atoms vs 7–9 heavy atoms —
  turns N≪D covariance underdetermination into a measured curve.

Usage
─────
  python seperation_and_prediction.py --smoke            # N=300, 1 seed, sanity
  python seperation_and_prediction.py                    # full run
  python seperation_and_prediction.py --n 1500 --n-seeds 10 --descriptors soap mace
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from umap import UMAP

# ── path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR  = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.non_euclidean import Grassmann, Riemann, PersistentHomology  # noqa: E402
from src.optimal_transport import REMatch, Wasserstein  # noqa: E402
from src.datasets import QM9Dataset  # noqa: E402
from regression_task import (  # noqa: E402
    _take_rows,
    _laplacian_kernel_from_distance,
    _rbf_kernel_from_distance,
    _median_beta_from_distances,
    _sanitize_distance_matrix,
    _pooled_descriptor_matrix,
)

warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Display labels for each internal matrix key
DISPLAY_NAMES: Dict[str, str] = {
    "morgan_tanimoto":    "Morgan (Tanimoto)",
    "chemprop_cosine":    "ChemProp (Cosine)",
    "soap_euclidean":     "SOAP · Euclidean",
    "soap_w1":            "SOAP · W₁",
    "soap_w2":            "SOAP · W₂",
    "soap_rematch":       "SOAP · REMatch",
    "soap_grass_geo":     "SOAP · Grassmann geodesic",
    "soap_grass_chd":     "SOAP · Grassmann chordal",
    "soap_riemann_airm":  "SOAP · Riemann AIRM",
    "soap_riemann_le":    "SOAP · Riemann LE",
    "soap_ph":            "SOAP · PH-descriptor",
    "mace_euclidean":     "MACE · Euclidean",
    "mace_w1":            "MACE · W₁",
    "mace_w2":            "MACE · W₂",
    "mace_rematch":       "MACE · REMatch",
    "mace_grass_geo":     "MACE · Grassmann geodesic",
    "mace_grass_chd":     "MACE · Grassmann chordal",
    "mace_riemann_airm":  "MACE · Riemann AIRM",
    "mace_riemann_le":    "MACE · Riemann LE",
    "mace_ph":            "MACE · PH-descriptor",
    "ph_coords":          "PH-coords [topology ref]",
}

# Methods grouped for table display
BLOCK_2D     = ["morgan_tanimoto", "chemprop_cosine"]
BLOCK_FLAT   = ["soap_euclidean",  "mace_euclidean"]
BLOCK_CURVED = ["soap_grass_geo",  "soap_grass_chd",
                "soap_riemann_airm", "soap_riemann_le",
                "mace_grass_geo",  "mace_grass_chd",
                "mace_riemann_airm", "mace_riemann_le"]
BLOCK_OT     = ["soap_w1", "soap_w2", "soap_rematch",
                "mace_w1", "mace_w2", "mace_rematch"]
BLOCK_PH     = ["soap_ph", "mace_ph", "ph_coords"]


# =============================================================================
# A.  Stratified sample draw (gap × heavy-atom count)
# =============================================================================

def _heavy_atom_count(df: pl.DataFrame) -> np.ndarray:
    if "atomic_numbers" in df.columns:
        return np.array([
            sum(1 for z in row if z > 1)
            for row in df["atomic_numbers"].to_list()
        ])
    counts = []
    for formula in df["formula"].to_list():
        total = sum(
            int(m.group(2) or 1)
            for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula)
            if m.group(1) != "H"
        )
        counts.append(total)
    return np.array(counts)


def draw_stratified_sample(
    df_base:      pl.DataFrame,
    n_target:     int = 1200,
    size_bins:    Tuple[int, int] = (6, 9),
    gap_quantiles: Tuple[float, float] = (1 / 3, 2 / 3),
    seed:         int = 0,
) -> pl.DataFrame:
    """
    Joint-stratified draw on gap × heavy-atom size bin.
    size_bins = (b1, b2) → three size strata: ≤b1, b1+1…b2, >b2.

    Allocation: equal-per-stratum (NOT proportionate to QM9 prevalence).
    This deliberately flattens QM9's imbalance (dominated by 8–9 heavy-atom
    molecules) to improve coverage of the small-molecule size bin.  The sample
    is NOT representative of the QM9 distribution; headline prediction metrics
    should be interpreted conditional on this balanced draw.
    Note: QM9 has ≤9 heavy atoms, so the >b2 stratum is near-empty by
    construction and any size-curve analysis is limited to ≤6 vs 7–9.
    """
    df = df_base.filter(pl.col("gap").is_finite())
    n_heavy = _heavy_atom_count(df)
    gap_arr = df["gap"].to_numpy()

    q1, q2 = np.quantile(gap_arr, gap_quantiles)
    gap_tercile = np.where(gap_arr <= q1, 0, np.where(gap_arr <= q2, 1, 2))
    size_bin    = np.where(n_heavy <= size_bins[0], 0,
                           np.where(n_heavy <= size_bins[1], 1, 2))
    stratum = gap_tercile * 3 + size_bin

    df = df.with_columns([
        pl.Series("n_heavy",     n_heavy.tolist()),
        pl.Series("gap_tercile", gap_tercile.tolist()),
        pl.Series("size_bin",    size_bin.tolist()),
        pl.Series("_stratum",    stratum.tolist()),
    ])

    unique_strata = np.unique(stratum)
    n_strata      = len(unique_strata)
    per_stratum   = max(1, n_target // n_strata)

    rng = np.random.default_rng(seed)
    pieces: List[pl.DataFrame] = []
    for s in unique_strata:
        sub = df.filter(pl.col("_stratum") == int(s))
        k   = min(per_stratum, sub.height)
        idx = rng.choice(sub.height, size=k, replace=False)
        pieces.append(sub[idx.tolist()])

    result = pl.concat(pieces).drop("_stratum")
    logger.info(
        f"Stratified sample: {result.height} molecules "
        f"({n_strata} strata, ~{per_stratum} each)  "
        f"size bins: ≤{size_bins[0]}={result.filter(pl.col('size_bin')==0).height}, "
        f"{size_bins[0]+1}–{size_bins[1]}={result.filter(pl.col('size_bin')==1).height}, "
        f">{size_bins[1]}={result.filter(pl.col('size_bin')==2).height}"
    )
    return result


# =============================================================================
# B.  2-D distance matrices
# =============================================================================

def _tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AB    = A @ B.T
    A_sq  = np.einsum("ij,ij->i", A, A)
    B_sq  = np.einsum("ij,ij->i", B, B)
    denom = A_sq[:, None] + B_sq[None, :] - AB
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(denom > 0, AB / denom, 0.0)
    return np.clip(K, 0.0, None)


def _morgan_tanimoto_distance(df: pl.DataFrame) -> np.ndarray:
    X = _pooled_descriptor_matrix(df, "morgan_fingerprint")
    D = _sanitize_distance_matrix(np.maximum(1.0 - _tanimoto_kernel(X, X), 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def _chemprop_cosine_distance(df: pl.DataFrame) -> np.ndarray:
    X = _pooled_descriptor_matrix(df, "chemprop_embedding")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    D = _sanitize_distance_matrix(np.maximum(1.0 - X @ X.T, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


# =============================================================================
# C.  Per-descriptor non-Euclidean matrix suite
# =============================================================================

def _build_descriptor_matrices(
    df:           pl.DataFrame,
    descriptor:   str,
    rematch_alpha: float,
    grassmann_k:  int,
) -> Dict[str, np.ndarray]:
    """Build the full geometry suite for one descriptor (soap or mace)."""
    d = descriptor
    emb_col = f"{d}_embedding"
    mat_col = f"{d}_matrix"

    if emb_col not in df.columns or mat_col not in df.columns:
        logger.warning(f"  {descriptor.upper()} columns missing — skipping.")
        return {}

    X_avg   = np.vstack(df[emb_col].to_list())
    d_euclid = squareform(pdist(X_avg, metric="euclidean"))

    wasser  = Wasserstein()
    d_w1    = wasser.distance_matrix(df, descriptor=d, metric="euclidean")
    d_w2    = wasser.distance_matrix(df, descriptor=d, metric="sqeuclidean")

    rematch = REMatch()
    d_rm    = rematch.distance_matrix(df, descriptor=d, alpha=rematch_alpha)
    if d_rm is None:
        logger.warning(f"  REMatch returned None for {descriptor} — substituting zeros.")
        d_rm = np.zeros_like(d_euclid)

    riemann = Riemann()
    d_airm  = riemann.distance_matrix(df, d, distance_type="affine-invariant")
    d_le    = riemann.distance_matrix(df, d, distance_type="log-euclidean")

    g       = Grassmann()
    d_ggeo  = g.distance_matrix(df, d, distance_type="geodesic", k=grassmann_k)
    d_gchd  = g.distance_matrix(df, d, distance_type="chordal",  k=grassmann_k)

    d_ph    = PersistentHomology.distance_matrix(df, descriptor=d, metric="sliced_wasserstein")

    return {
        f"{d}_euclidean":    d_euclid,
        f"{d}_w1":           d_w1,
        f"{d}_w2":           d_w2,
        f"{d}_rematch":      d_rm,
        f"{d}_riemann_airm": d_airm,
        f"{d}_riemann_le":   d_le,
        f"{d}_grass_geo":    d_ggeo,
        f"{d}_grass_chd":    d_gchd,
        f"{d}_ph":           d_ph,
    }


# =============================================================================
# D.  Caching — full-sample matrix store (keyed on molecule IDs, not splits)
# =============================================================================

def _mol_hash(df: pl.DataFrame) -> str:
    if "mol_id" in df.columns:
        ids_str = "_".join(str(x) for x in sorted(df["mol_id"].to_list()))
    else:
        ids_str = "_".join(str(x) for x in sorted(df["smiles"].to_list()))
    return hashlib.sha256(ids_str.encode()).hexdigest()[:16]


def _cache_path(cache_dir: str, mol_hash: str, key: str) -> Path:
    safe = re.sub(r'[^a-z0-9_]', '_', key.lower())
    return Path(cache_dir) / f"{mol_hash}_{safe}.npz"


def _load_cached(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.load(path)["D"]
        except Exception:
            pass
    return None


def _save_cached(D: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, D=D)


def build_all_matrices(
    df:            pl.DataFrame,
    descriptors:   Sequence[str] = ("soap",),
    rematch_alpha: float = 0.1,
    grassmann_k:   int   = 3,
    cache_dir:     str   = ".cache/sep_pred_matrices",
) -> Dict[str, np.ndarray]:
    """
    Compute (or load from cache) every distance matrix in the benchmark suite.
    PH matrices are expensive (~38 min); the cache means they are paid once.
    """
    mh = _mol_hash(df)
    matrices: Dict[str, np.ndarray] = {}

    def _get(key: str, compute_fn) -> np.ndarray:
        cp = _cache_path(cache_dir, mh, key)
        D  = _load_cached(cp)
        if D is not None:
            logger.info(f"  [cache hit ] {key}")
            return D
        t0 = time.perf_counter()
        D  = compute_fn()
        _save_cached(D, cp)
        logger.info(f"  [computed  ] {key}  ({time.perf_counter() - t0:.1f} s)")
        return D

    # 2-D baselines (only if columns present)
    if "morgan_fingerprint" in df.columns:
        matrices["morgan_tanimoto"] = _get("morgan_tanimoto",
                                           lambda: _morgan_tanimoto_distance(df))
    if "chemprop_embedding" in df.columns:
        matrices["chemprop_cosine"] = _get("chemprop_cosine",
                                           lambda: _chemprop_cosine_distance(df))

    # 3-D geometry suites
    for desc in descriptors:
        logger.info(f"\n  Building {desc.upper()} matrices...")
        suite = {}
        for key, compute_fn in _descriptor_compute_fns(df, desc, rematch_alpha, grassmann_k):
            suite[key] = _get(key, compute_fn)
        matrices.update(suite)

    # PH on 3D coordinates — topology reference
    matrices["ph_coords"] = _get(
        "ph_coords",
        lambda: PersistentHomology.distance_matrix(
            df, descriptor="coordinates", metric="sliced_wasserstein"
        ),
    )

    return matrices


def _descriptor_compute_fns(
    df: pl.DataFrame, desc: str, rematch_alpha: float, grassmann_k: int
) -> List[Tuple[str, Any]]:
    """Return (key, compute_fn) pairs for a single descriptor."""
    emb = f"{desc}_embedding"
    if emb not in df.columns:
        return []

    n = df.height

    def euclid():
        X = np.vstack(df[emb].to_list())
        return squareform(pdist(X, metric="euclidean"))

    def make_w1():
        return Wasserstein().distance_matrix(df, desc, metric="euclidean")

    def make_w2():
        return Wasserstein().distance_matrix(df, desc, metric="sqeuclidean")

    def make_rematch():
        d = REMatch().distance_matrix(df, desc, alpha=rematch_alpha)
        return d if d is not None else np.zeros((n, n))

    def make_airm():
        return Riemann().distance_matrix(df, desc, distance_type="affine-invariant")

    def make_le():
        return Riemann().distance_matrix(df, desc, distance_type="log-euclidean")

    def make_geo():
        return Grassmann().distance_matrix(df, desc, distance_type="geodesic", k=grassmann_k)

    def make_chd():
        return Grassmann().distance_matrix(df, desc, distance_type="chordal", k=grassmann_k)

    def make_ph():
        return PersistentHomology.distance_matrix(df, descriptor=desc, metric="sliced_wasserstein")

    return [
        (f"{desc}_euclidean",    euclid),
        (f"{desc}_w1",           make_w1),
        (f"{desc}_w2",           make_w2),
        (f"{desc}_rematch",      make_rematch),
        (f"{desc}_riemann_airm", make_airm),
        (f"{desc}_riemann_le",   make_le),
        (f"{desc}_grass_geo",    make_geo),
        (f"{desc}_grass_chd",    make_chd),
        (f"{desc}_ph",           make_ph),
    ]


# =============================================================================
# E.  Separation metrics — property-aligned, no ring labels
# =============================================================================

def _property_tercile_labels(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= q1, 0, np.where(values <= q2, 1, 2))


def knn_tercile_agreement(D: np.ndarray, tercile_labels: np.ndarray, k: int = 10) -> float:
    """
    Fraction of a molecule's k nearest neighbors (in ambient D) sharing its
    property tercile. Average over all molecules. Range [0,1]; random baseline ≈ 1/3.
    """
    n = len(D)
    scores = np.empty(n)
    for i in range(n):
        d_row = D[i].copy()
        d_row[i] = np.inf
        nn = np.argsort(d_row)[:k]
        scores[i] = np.mean(tercile_labels[nn] == tercile_labels[i])
    return float(scores.mean())


def compute_rsep(D: np.ndarray, tercile_labels: np.ndarray) -> Dict[str, float]:
    """Mean intra-tercile dist / mean inter-tercile dist (lower = better separated)."""
    classes = np.unique(tercile_labels)
    intra = []
    for cls in classes:
        idx = np.where(tercile_labels == cls)[0]
        if len(idx) > 1:
            sub  = D[np.ix_(idx, idx)]
            triu = np.triu_indices(len(idx), k=1)
            intra.append(np.mean(sub[triu]))
    mean_intra = float(np.mean(intra)) if intra else 0.0

    inter = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ia = np.where(tercile_labels == classes[i])[0]
            ib = np.where(tercile_labels == classes[j])[0]
            inter.append(np.mean(D[np.ix_(ia, ib)]))
    mean_inter = float(np.mean(inter)) if inter else 1.0

    return {"intra": mean_intra, "inter": mean_inter, "rsep": mean_intra / mean_inter}


def run_separation_arm(
    matrices:    Dict[str, np.ndarray],
    df:          pl.DataFrame,
    target_cols: Sequence[str],
    k:           int,
    output_dir:  str,
    umap_seed:   int = 42,
) -> pl.DataFrame:
    """
    Scores every matrix on property-tercile kNN agreement and Rsep.
    UMAP is generated as illustration only (single seed, not headline metric).
    Returns a long-format DataFrame: (method, target, knn_agreement, rsep, intra, inter).
    """
    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict] = []

    for target in target_cols:
        if target not in df.columns:
            continue
        vals   = df[target].to_numpy().astype(np.float64)
        labels = _property_tercile_labels(vals)
        label_names = np.where(labels == 0, f"Low {target}",
                     np.where(labels == 1, f"Mid {target}", f"High {target}"))

        palette = {
            f"Low {target}":  "#4a7c59",
            f"Mid {target}":  "#dda15e",
            f"High {target}": "#bc6c25",
        }

        for key, D in matrices.items():
            knn  = knn_tercile_agreement(D, labels, k=k)
            sep  = compute_rsep(D, labels)
            rows.append({
                "method":  key,
                "display": DISPLAY_NAMES.get(key, key),
                "target":  target,
                "knn_agreement": knn,
                "rsep":          sep["rsep"],
                "intra":         sep["intra"],
                "inter":         sep["inter"],
            })

            # UMAP illustration (one per method × target)
            _save_umap_illustration(
                D, label_names, palette,
                title=f"{DISPLAY_NAMES.get(key, key)} — {target}",
                path=os.path.join(output_dir, f"umap_{key}_{target}.png"),
                seed=umap_seed,
            )

    df_sep = pl.DataFrame(rows)
    df_sep.write_csv(os.path.join(output_dir, "separation_ambient.csv"))
    _print_separation_table(df_sep, target_cols)
    return df_sep


def _save_umap_illustration(
    D:          np.ndarray,
    labels:     np.ndarray,
    palette:    Dict[str, str],
    title:      str,
    path:       str,
    seed:       int,
) -> None:
    try:
        reducer = UMAP(n_neighbors=15, metric="precomputed", random_state=seed)
        coords  = reducer.fit_transform(D)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    plt.style.use("seaborn-v0_8-whitegrid")
    import pandas as pd
    df_plot = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    for lbl, color in palette.items():
        sub = df_plot[df_plot["label"] == lbl]
        ax.scatter(sub["x"], sub["y"], c=color, s=18, alpha=0.7,
                   linewidths=0.3, edgecolors="#2d3436", label=lbl, zorder=5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.legend(fontsize=7, frameon=True, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _print_separation_table(df_sep: pl.DataFrame, target_cols: Sequence[str]) -> None:
    W = 95
    print("\n" + "=" * W)
    print("  SEPARATION — Ambient kNN tercile agreement and Rsep  (property-defined labels)")
    print("  kNN agreement is a LOCAL view of Similar-Property; KRR R² is the GLOBAL view.")
    print("  They operationalize the SAME criterion — divergence between them is the finding,")
    print("  not an inconsistency.  Random baseline for kNN agreement ≈ 0.333.")
    print("=" * W)
    for target in target_cols:
        sub = df_sep.filter(pl.col("target") == target).sort("knn_agreement", descending=True)
        print(f"\n  Target: {target}")
        print(f"  {'Method':<40} | {'kNN agree ↑':>12} | {'Rsep ↓':>10} | {'Intra':>8} | {'Inter':>8}")
        print("  " + "-" * (W - 2))
        for row in sub.iter_rows(named=True):
            print(f"  {row['display']:<40} | {row['knn_agreement']:>12.4f} "
                  f"| {row['rsep']:>10.4f} | {row['intra']:>8.4f} | {row['inter']:>8.4f}")
    print("=" * W + "\n")


# =============================================================================
# F.  KRR fitting — Laplacian primary, best-kernel secondary
# =============================================================================

def _fit_krr_laplacian(
    D_full:    np.ndarray,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    alpha_grid: Sequence[float],
    cv:        int,
    seed:      int,
) -> Dict[str, Any]:
    """Primary controlled comparison: Laplacian kernel, median bandwidth."""
    D_tr  = D_full[np.ix_(train_idx, train_idx)]
    D_te  = D_full[np.ix_(test_idx,  train_idx)]
    beta  = _median_beta_from_distances(D_tr)
    K_tr  = _laplacian_kernel_from_distance(D_tr, beta)
    K_te  = _laplacian_kernel_from_distance(D_te, beta)
    return _inner_cv_krr(K_tr, K_te, y_train, y_test, alpha_grid, cv, seed, beta)


def _fit_krr_rbf(
    D_full:    np.ndarray,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    alpha_grid: Sequence[float],
    cv:        int,
    seed:      int,
) -> Dict[str, Any]:
    """Secondary best-kernel for flat methods: RBF from distance."""
    D_tr  = D_full[np.ix_(train_idx, train_idx)]
    D_te  = D_full[np.ix_(test_idx,  train_idx)]
    beta  = _median_beta_from_distances(D_tr, squared=True)
    K_tr  = _rbf_kernel_from_distance(D_tr, beta)
    K_te  = _rbf_kernel_from_distance(D_te, beta)
    return _inner_cv_krr(K_tr, K_te, y_train, y_test, alpha_grid, cv, seed, beta)


def _fit_krr_tanimoto_direct(
    df_full:   pl.DataFrame,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    alpha_grid: Sequence[float],
    cv:        int,
    seed:      int,
) -> Dict[str, Any]:
    """Secondary best-kernel for Morgan: Tanimoto kernel directly (not distance)."""
    X     = _pooled_descriptor_matrix(df_full, "morgan_fingerprint")
    X_tr  = X[train_idx]
    X_te  = X[test_idx]
    K_tr  = _tanimoto_kernel(X_tr, X_tr)
    K_tr  = (K_tr + K_tr.T) / 2
    K_te  = _tanimoto_kernel(X_te, X_tr)
    return _inner_cv_krr(K_tr, K_te, y_train, y_test, alpha_grid, cv, seed, beta=None)


def _inner_cv_krr(
    K_tr:      np.ndarray,
    K_te:      np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    alpha_grid: Sequence[float],
    cv:        int,
    seed:      int,
    beta:      Optional[float],
) -> Dict[str, Any]:
    splitter   = KFold(n_splits=cv, shuffle=True, random_state=seed)
    best_alpha, best_cv = None, np.inf
    for a in alpha_grid:
        fold_rmse = []
        for i_tr, i_val in splitter.split(K_tr):
            m = KernelRidge(alpha=float(a), kernel="precomputed")
            m.fit(K_tr[np.ix_(i_tr, i_tr)], y_train[i_tr])
            fold_rmse.append(float(np.sqrt(mean_squared_error(
                y_train[i_val], m.predict(K_tr[np.ix_(i_val, i_tr)])
            ))))
        cv_rmse = float(np.mean(fold_rmse))
        if cv_rmse < best_cv:
            best_cv, best_alpha = cv_rmse, a

    m = KernelRidge(alpha=best_alpha, kernel="precomputed")
    m.fit(K_tr, y_train)
    pred = m.predict(K_te)
    return {
        "test_r2":    float(r2_score(y_test, pred)),
        "test_mae":   float(mean_absolute_error(y_test, pred)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_test, pred))),
        "best_alpha": best_alpha,
        "best_beta":  beta,
        "cv_rmse":    best_cv,
    }


# =============================================================================
# G.  Prediction arm — primary (Laplacian) + secondary (best-kernel)
# =============================================================================

def _stratified_split(
    df: pl.DataFrame, seed: int, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split on gap_tercile × size_bin stratum."""
    if "gap_tercile" in df.columns and "size_bin" in df.columns:
        strat = (df["gap_tercile"].to_numpy() * 3 + df["size_bin"].to_numpy()).tolist()
        splitter = StratifiedKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
        idx = np.arange(df.height)
        train_idx, test_idx = next(splitter.split(idx, strat))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(df.height), test_size=test_size, random_state=seed, shuffle=True,
        )
    return train_idx, test_idx


def run_prediction_arm(
    df:          pl.DataFrame,
    matrices:    Dict[str, np.ndarray],
    targets:     Sequence[str],
    seeds:       Sequence[int],
    alpha_grid:  Sequence[float],
    cv:          int,
    output_dir:  str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Primary (Laplacian-only) and secondary (best-kernel flat) prediction tables.
    Returns (primary_df, secondary_df).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Secondary methods: best-fit kernel per flat/tangent method.
    # 2D floor: Tanimoto direct / cosine-RBF.
    # Flat 3D baselines: Euclidean-RBF.
    # Tangent-space / flat-embedding manifold methods (LE, chordal): also RBF.
    #   These CAN take RBF because their distances live in a flat space;
    #   AIRM and geodesic Grassmann cannot (curved, no inner product).
    # Including LE and chordal here is the key move to close the kernel-asymmetry
    # confound: if curved AIRM/geodesic beats LE/chordal even with RBF, the gain
    # is geometric, not a kernel availability artefact.
    SECONDARY: Dict[str, str] = {
        "morgan_tanimoto":   "tanimoto_direct",
        "chemprop_cosine":   "rbf",
        "soap_euclidean":    "rbf",
        "mace_euclidean":    "rbf",
        "soap_riemann_le":   "rbf",   # log-Euclidean tangent space is flat
        "mace_riemann_le":   "rbf",
        "soap_grass_chd":    "rbf",   # chordal embedding is flat
        "mace_grass_chd":    "rbf",
    }

    primary_rows:   List[Dict] = []
    secondary_rows: List[Dict] = []

    for seed in seeds:
        logger.info(f"\n  [Prediction] seed={seed}")
        train_idx, test_idx = _stratified_split(df, seed)
        for target in targets:
            if target not in df.columns:
                continue
            y_full  = df[target].to_numpy().astype(np.float64)
            y_train = y_full[train_idx]
            y_test  = y_full[test_idx]

            # ── Primary: Laplacian for all methods ─────────────────────────
            for key, D in matrices.items():
                t0 = time.perf_counter()
                try:
                    res = _fit_krr_laplacian(
                        D, train_idx, test_idx, y_train, y_test,
                        alpha_grid, cv, seed,
                    )
                    primary_rows.append({
                        "method": key, "display": DISPLAY_NAMES.get(key, key),
                        "target": target, "seed": seed, "kernel": "laplacian",
                        "status": "ok", "total_seconds": time.perf_counter() - t0,
                        **res,
                    })
                except Exception as e:
                    logger.warning(f"    {key} [{target}] FAILED: {e}")
                    primary_rows.append({
                        "method": key, "display": DISPLAY_NAMES.get(key, key),
                        "target": target, "seed": seed, "kernel": "laplacian",
                        "status": "failed", "test_r2": None, "test_mae": None,
                    })

            # ── Secondary: best-fit kernels for flat methods ────────────────
            for key, ktype in SECONDARY.items():
                if key not in matrices:
                    continue
                D = matrices[key]
                t0 = time.perf_counter()
                try:
                    if ktype == "tanimoto_direct" and "morgan_fingerprint" in df.columns:
                        res = _fit_krr_tanimoto_direct(
                            df, train_idx, test_idx, y_train, y_test,
                            alpha_grid, cv, seed,
                        )
                    else:
                        res = _fit_krr_rbf(
                            D, train_idx, test_idx, y_train, y_test,
                            alpha_grid, cv, seed,
                        )
                    secondary_rows.append({
                        "method": key, "display": DISPLAY_NAMES.get(key, key),
                        "target": target, "seed": seed, "kernel": ktype,
                        "status": "ok", "total_seconds": time.perf_counter() - t0,
                        **res,
                    })
                except Exception as e:
                    logger.warning(f"    {key}[secondary, {target}] FAILED: {e}")

    primary_df   = pl.DataFrame(primary_rows)
    secondary_df = pl.DataFrame(secondary_rows)
    primary_df.write_csv(os.path.join(output_dir, "prediction_primary.csv"))
    secondary_df.write_csv(os.path.join(output_dir, "prediction_secondary.csv"))

    for target in targets:
        _print_prediction_table(primary_df, secondary_df, target, seeds)

    return primary_df, secondary_df


def _print_prediction_table(
    primary_df:   pl.DataFrame,
    secondary_df: pl.DataFrame,
    target:       str,
    seeds:        Sequence[int],
) -> None:
    n = len(seeds)
    z95 = 1.96
    W = 95

    print(f"\n{'=' * W}")
    print(f"  PREDICTION — Target: {target.upper()}  (primary: Laplacian kernel, {n} splits)")
    print(f"  ± 95% CI is split-variance only (fixed {n}-molecule sample, varying 80/20 split).")
    print("  Repeated splits of the same data are correlated → p-values are anti-conservative.")
    print("  Interpret as: 'reliable across splits of this sample', not as population inference.")
    print(f"{'=' * W}")
    print(f"  {'Method':<40} | {'R² (↑)':>18} | {'MAE (↓)':>16}")
    print("  " + "-" * (W - 2))

    sub = primary_df.filter((pl.col("target") == target) & (pl.col("status") == "ok"))
    summary = (
        sub.group_by(["method", "display"])
        .agg([
            pl.col("test_r2").mean().alias("r2_mean"),
            pl.col("test_r2").std().alias("r2_std"),
            pl.col("test_mae").mean().alias("mae_mean"),
            pl.col("test_mae").std().alias("mae_std"),
            pl.col("test_r2").count().alias("n"),
        ])
        .sort("r2_mean", descending=True, nulls_last=True)
    )
    for row in summary.iter_rows(named=True):
        ci_r2  = z95 * (row["r2_std"]  or 0) / np.sqrt(max(row["n"], 1))
        ci_mae = z95 * (row["mae_std"] or 0) / np.sqrt(max(row["n"], 1))
        print(f"  {row['display']:<40} | {row['r2_mean']:>8.4f} ± {ci_r2:<7.4f} "
              f"| {row['mae_mean']:>7.4f} ± {ci_mae:<6.4f}")

    print("\n  [Secondary — best-fit kernel per flat method, NOT the controlled comparison]")
    sub2 = secondary_df.filter((pl.col("target") == target) & (pl.col("status") == "ok"))
    if sub2.height > 0:
        sum2 = (
            sub2.group_by(["method", "display", "kernel"])
            .agg([
                pl.col("test_r2").mean().alias("r2_mean"),
                pl.col("test_r2").std().alias("r2_std"),
                pl.col("test_r2").count().alias("n"),
            ])
            .sort("r2_mean", descending=True, nulls_last=True)
        )
        for row in sum2.iter_rows(named=True):
            ci = z95 * (row["r2_std"] or 0) / np.sqrt(max(row["n"], 1))
            print(f"  {row['display']:<40} [{row['kernel']}]  R²={row['r2_mean']:.4f} ± {ci:.4f}")
    print("=" * W)


# =============================================================================
# H.  Heavy-atom bin analysis
# =============================================================================

def report_by_size_bin(
    df:          pl.DataFrame,
    primary_df:  pl.DataFrame,
    sep_df:      pl.DataFrame,
    matrices:    Dict[str, np.ndarray],
    targets:     Sequence[str],
    seeds:       Sequence[int],
    alpha_grid:  Sequence[float],
    cv:          int,
    k_sep:       int,
    output_dir:  str,
) -> pl.DataFrame:
    """
    Re-runs prediction and separation restricted to ≤6 vs 7–9 heavy atoms.

    LIMITATION: QM9 has at most 9 heavy atoms, so the >9 size bin is empty by
    construction and every bin sits in the N≪D regime (D≈252 for SOAP).  This
    analysis shows whether performance *varies* across the limited QM9 size range,
    not whether curved geometry pays off when N approaches D.  For that question
    you need QMugs / a GDB-13/17 slice.  Report results here as "limited size
    range (≤6 vs 7–9 heavy atoms)", not as a resolved N≪D curve.
    Returns combined results DataFrame.
    """
    if "size_bin" not in df.columns:
        logger.warning("n_heavy / size_bin not available — skipping size-bin analysis.")
        return pl.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    bin_labels = {0: "small (≤6 heavy)", 1: "medium (7–9 heavy)", 2: "large (>9 heavy)"}
    all_rows: List[Dict] = []

    for bin_val, bin_name in bin_labels.items():
        idx  = np.where(df["size_bin"].to_numpy() == bin_val)[0]
        if len(idx) < 20:
            continue
        df_sub = _take_rows(df, idx)
        logger.info(f"\n  Size bin '{bin_name}': N={len(idx)}")

        # Sub-matrices
        sub_mats = {k: D[np.ix_(idx, idx)] for k, D in matrices.items()}

        # Separation on sub-set
        for target in targets:
            if target not in df_sub.columns:
                continue
            vals   = df_sub[target].to_numpy().astype(np.float64)
            labels = _property_tercile_labels(vals)
            for key, D in sub_mats.items():
                knn = knn_tercile_agreement(D, labels, k=min(k_sep, len(idx) - 1))
                sep = compute_rsep(D, labels)
                all_rows.append({
                    "size_bin": bin_name, "method": key,
                    "display": DISPLAY_NAMES.get(key, key),
                    "target": target, "task": "separation",
                    "knn_agreement": knn, "rsep": sep["rsep"],
                })

        # Prediction on sub-set (1 seed for speed; report without CI)
        for seed in seeds[:3]:
            if len(idx) < 25:
                continue
            tr_idx, te_idx = _stratified_split(df_sub, seed)
            for target in targets:
                if target not in df_sub.columns:
                    continue
                y_full  = df_sub[target].to_numpy().astype(np.float64)
                y_train = y_full[tr_idx]
                y_test  = y_full[te_idx]
                for key, D in sub_mats.items():
                    try:
                        res = _fit_krr_laplacian(
                            D, tr_idx, te_idx, y_train, y_test,
                            alpha_grid, cv, seed,
                        )
                        all_rows.append({
                            "size_bin": bin_name, "method": key,
                            "display": DISPLAY_NAMES.get(key, key),
                            "target": target, "seed": seed, "task": "prediction",
                            **res,
                        })
                    except Exception:
                        pass

    result = pl.DataFrame(all_rows)
    result.write_csv(os.path.join(output_dir, "by_size_bin.csv"))
    _print_size_bin_table(result, targets)
    return result


def _print_size_bin_table(df: pl.DataFrame, targets: Sequence[str]) -> None:
    if df.is_empty():
        return
    W = 95
    print("\n" + "=" * W)
    print("  SIZE-BIN ANALYSIS  (Laplacian, primary metric)")
    print("=" * W)
    for target in targets:
        print(f"\n  Target: {target}")
        for task, col, arrow in [("separation", "knn_agreement", "↑"), ("prediction", "test_r2", "↑")]:
            sub = df.filter((pl.col("target") == target) & (pl.col("task") == task))
            if sub.is_empty():
                continue
            agg = (
                sub.group_by(["method", "display", "size_bin"])
                .agg(pl.col(col).mean().alias("score"))
                .pivot(values="score", index=["method", "display"], on="size_bin")
            )
            print(f"\n    {task.capitalize()} ({col} {arrow})")
            cols = [c for c in agg.columns if c not in ("method", "display")]
            header = f"    {'Method':<40} " + "".join(f"| {c:<20} " for c in cols)
            print(header)
            print("    " + "-" * (len(header) - 4))
            for row in agg.sort("display").iter_rows(named=True):
                line = f"    {row['display']:<40} "
                for c in cols:
                    v = row.get(c)
                    line += f"| {v if v is not None else float('nan'):<20.4f} "
                print(line)
    print("=" * W + "\n")


# =============================================================================
# I.  Paired significance — Wilcoxon signed-rank test
# =============================================================================

def paired_wilcoxon_table(
    primary_df: pl.DataFrame,
    target:     str,
    metric:     str = "test_r2",
    alpha:      float = 0.05,
    output_dir: str = ".",
) -> pl.DataFrame:
    """
    For every ordered pair (A, B) where mean(A) > mean(B), run Wilcoxon signed-rank
    on the per-seed paired differences. Reports p-value and significance flag.

    CAVEAT: seeds are repeated splits of one fixed molecule sample, so observations
    are correlated. The resulting p-values are anti-conservative — they understate
    the true uncertainty. Interpret as "A is reliably > B across splits of this
    sample," not as a population significance claim.
    """
    sub = primary_df.filter(
        (pl.col("target") == target) & (pl.col("status") == "ok")
    )
    methods = sub["method"].unique().to_list()

    pivot = sub.pivot(values=metric, index="seed", on="method", aggregate_function="mean")
    rows: List[Dict] = []
    for a in methods:
        for b in methods:
            if a == b or a not in pivot.columns or b not in pivot.columns:
                continue
            va = pivot[a].drop_nulls().to_numpy()
            vb = pivot[b].drop_nulls().to_numpy()
            if len(va) < 5 or len(vb) < 5 or np.mean(va) <= np.mean(vb):
                continue
            n = min(len(va), len(vb))
            try:
                stat, p = scipy_wilcoxon(va[:n], vb[:n], alternative="greater")
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "method_A": a, "display_A": DISPLAY_NAMES.get(a, a),
                "method_B": b, "display_B": DISPLAY_NAMES.get(b, b),
                "mean_A":   float(np.mean(va)),
                "mean_B":   float(np.mean(vb)),
                "delta":    float(np.mean(va) - np.mean(vb)),
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
                "p_value":       float(p)    if not np.isnan(p)    else None,
                "significant":   bool(p < alpha) if not np.isnan(p) else False,
            })

    result = pl.DataFrame(rows).sort("delta", descending=True, nulls_last=True)
    out = os.path.join(output_dir, f"paired_wilcoxon_{target}.csv")
    result.write_csv(out)

    sig = result.filter(pl.col("significant"))
    if sig.height > 0:
        print(f"\n  Significant pairs ({metric}, {target}, p<{alpha}):")
        for row in sig.iter_rows(named=True):
            print(f"    {row['display_A']}  >  {row['display_B']}  "
                  f"Δ={row['delta']:.4f}  p={row['p_value']:.4f}")
    else:
        print(f"\n  No significant pairs found for {metric}, {target} at α={alpha}.")

    return result


# =============================================================================
# J.  Pipeline
# =============================================================================

def run_pipeline(
    df_base:       pl.DataFrame,
    n_target:      int            = 1200,
    descriptors:   Sequence[str]  = ("soap",),
    targets:       Sequence[str]  = ("gap", "mu"),
    seeds:         Sequence[int]  = tuple(range(10)),
    alpha_grid:    Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0),
    cv:            int            = 5,
    rematch_alpha: float          = 0.1,
    grassmann_k:   int            = 3,
    k_sep:         int            = 10,
    umap_seed:     int            = 42,
    cache_dir:     str            = ".cache/sep_pred_matrices",
    output_dir:    str            = "results/qm9/sep_pred_benchmark",
    smoke:         bool           = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if smoke:
        seeds    = seeds[:1]
        n_target = min(n_target, 300)
        logger.info(f"SMOKE MODE — N={n_target}, 1 seed.")

    # ── 1. Draw sample ────────────────────────────────────────────────────────
    logger.info(f"\n[1/5] Drawing stratified sample (N≈{n_target})...")
    df = draw_stratified_sample(df_base, n_target=n_target)
    valid_targets = [t for t in targets if t in df.columns]
    if valid_targets:
        mask = pl.lit(True)
        for t in valid_targets:
            mask = mask & pl.col(t).is_finite()
        df = df.filter(mask)
        logger.info(f"After target-finite filter: {df.height} molecules.")

    if df.height < 50:
        raise ValueError(f"Only {df.height} molecules after filtering — raise --n.")

    # ── 2. Build / load distance matrices ────────────────────────────────────
    logger.info(f"\n[2/5] Building distance matrices (cached at {cache_dir})...")
    t0       = time.perf_counter()
    matrices = build_all_matrices(
        df, descriptors=descriptors,
        rematch_alpha=rematch_alpha, grassmann_k=grassmann_k,
        cache_dir=cache_dir,
    )
    logger.info(f"  Matrices ready in {time.perf_counter() - t0:.1f} s  "
                f"({len(matrices)} total)")

    # ── 3. Separation arm ─────────────────────────────────────────────────────
    logger.info(f"\n[3/5] Running separation arm (kNN k={k_sep})...")
    sep_dir = os.path.join(output_dir, "separation")
    sep_df  = run_separation_arm(
        matrices, df, valid_targets, k=k_sep,
        output_dir=sep_dir, umap_seed=umap_seed,
    )

    # ── 4. Prediction arm ─────────────────────────────────────────────────────
    logger.info(f"\n[4/5] Running prediction arm ({len(seeds)} seeds × {len(valid_targets)} targets)...")
    pred_dir   = os.path.join(output_dir, "prediction")
    primary_df, secondary_df = run_prediction_arm(
        df, matrices, valid_targets, seeds,
        alpha_grid=alpha_grid, cv=cv, output_dir=pred_dir,
    )

    # ── 5. Paired significance ─────────────────────────────────────────────────
    logger.info("\n[5/5] Paired Wilcoxon significance tests...")
    sig_dir = os.path.join(output_dir, "significance")
    os.makedirs(sig_dir, exist_ok=True)
    for target in valid_targets:
        paired_wilcoxon_table(primary_df, target=target, output_dir=sig_dir)

    # ── 6. Size-bin analysis (non-smoke) ──────────────────────────────────────
    if not smoke and "size_bin" in df.columns:
        logger.info("\n[+] Running size-bin analysis...")
        bin_dir = os.path.join(output_dir, "by_size_bin")
        report_by_size_bin(
            df, primary_df, sep_df, matrices,
            valid_targets, seeds, alpha_grid, cv, k_sep, bin_dir,
        )

    logger.info(f"\nAll outputs saved to: {output_dir}")


# =============================================================================
# K.  CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Separation + Prediction Benchmark (QM9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit",          type=int,   default=2000,
                        help="QM9 rows to load.")
    parser.add_argument("--n",              type=int,   default=1200,
                        help="Target stratified sample size.")
    parser.add_argument("--descriptors",    nargs="+",  default=["soap", "mace"],
                        help="3D descriptors (soap, mace). 2D added if columns exist.")
    parser.add_argument("--targets",        nargs="+",  default=["gap", "mu"],
                        help="Prediction targets.")
    parser.add_argument("--n-seeds",        type=int,   default=10,
                        help="Number of repeated 80/20 splits.")
    parser.add_argument("--cv",             type=int,   default=5,
                        help="Inner KFold CV folds.")
    parser.add_argument("--rematch-alpha",  type=float, default=0.1,
                        help="Fixed REMatch α (not tuned on labels).")
    parser.add_argument("--grassmann-k",    type=int,   default=3,
                        help="Grassmann subspace rank k.")
    parser.add_argument("--k-sep",          type=int,   default=10,
                        help="kNN k for tercile-agreement separation metric.")
    parser.add_argument("--cache-dir",      type=str,   default=".cache/sep_pred_matrices")
    parser.add_argument("--out-dir",        type=str,   default="results/qm9/sep_pred_benchmark")
    parser.add_argument("--smoke",          action="store_true",
                        help="N=300, 1 seed — quick sanity check.")
    args = parser.parse_args()

    descriptors = list(dict.fromkeys(args.descriptors))  # deduplicate, preserve order
    seeds       = list(range(args.n_seeds))

    logger.info(f"Loading QM9 (limit={args.limit}, descriptors={descriptors})...")
    qm9 = QM9Dataset(limit=args.limit, descriptors=descriptors)
    df  = qm9.load()
    logger.info(f"Loaded {df.height} molecules.")

    run_pipeline(
        df_base       = df,
        n_target      = args.n,
        descriptors   = descriptors,
        targets       = args.targets,
        seeds         = seeds,
        cv            = args.cv,
        rematch_alpha = args.rematch_alpha,
        grassmann_k   = args.grassmann_k,
        k_sep         = args.k_sep,
        cache_dir     = args.cache_dir,
        output_dir    = args.out_dir,
        smoke         = args.smoke,
    )

if __name__ == "__main__":
    main()
