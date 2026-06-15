"""
Three-Block Regression Benchmark: Geometric Sensitivity Study on QM9
Master's Thesis

Predicts molecular properties via Kernel Ridge Regression, isolating the
effect of the geometry / distance framework from the molecular representation.

Targets (looped):
  gap              — HOMO-LUMO gap (eV)        : electronic
  geometric_strain — Valence Angle Strain Index : angular / geometric
  mu               — dipole moment (Debye)      : directional (MACE equivariance)

╔══════════════╦════════════════════════════════╦════════════════════════════════════╗
║ Block        ║ Representation                 ║ Distance / Kernel                  ║
╠══════════════╬════════════════════════════════╬════════════════════════════════════╣
║ 1 Reference  ║ Morgan fingerprints (2D)       ║ Tanimoto, Euclidean                ║
║              ║ ChemProp GNN embeddings (2D)   ║ Cosine, Euclidean                  ║
╠══════════════╬════════════════════════════════╬════════════════════════════════════╣
║ 2 Control    ║ Trivial mean predictor         ║ —                                  ║
║              ║ Atom composition {H,C,N,O,F}   ║ RBF-KRR                            ║
╠══════════════╬════════════════════════════════╬════════════════════════════════════╣
║ 3 Controlled ║ SOAP (per-atom matrix)         ║ Euclidean (avg), Euclidean (linear)║
║              ║ MACE (per-atom matrix)         ║ Riemann (AI, LE, tangent),         ║
║              ║                                ║ Grassmann (geodesic, chordal,      ║
║              ║                                ║ isometric), Wasserstein-1/2,        ║
║              ║                                ║ Average kernel, REMatch,            ║
║              ║                                ║ PH × {coords, descriptor-space}    ║
║              ║                                ║   × {bottleneck, sliced-W}          ║
╚══════════════╩════════════════════════════════╩════════════════════════════════════╝

Fair comparison invariants
──────────────────────────
• All methods AND all targets see the SAME molecules. The subset is filtered ONCE,
  up front, to rows where every benchmarked target is finite (and geometric_strain
  ≥ 0 when it is a target). This is essential: per-target filtering would change the
  molecule set, silently breaking both the distance-matrix cache (keyed on mol_ids)
  and the cross-method / cross-target comparability.
• For every seed, ALL methods use the IDENTICAL 80/20 train/test split. The split
  depends only on the seed, so within a seed the molecule set + ordering is fixed
  across all three targets, and the expensive N×N matrices are computed ONCE per
  seed and reused for every target.
• Significance is assessed via the Friedman omnibus test (seeds as blocks, methods
  as treatments) on test_rmse, run per target. See the honesty note below.

Statistical-significance honesty note
─────────────────────────────────────
Using SEEDS as Friedman blocks tests *reproducibility across random splits* — it is
NOT the multi-dataset critical-difference setup the Friedman/Nemenyi procedure was
designed for. With a handful of seed-blocks and ~14 Block-3 methods, the Nemenyi
critical difference is very wide and almost no pair will separate. A non-significant
post-hoc here is therefore NOT evidence that two methods are equivalent. We report
the Friedman omnibus statistic and treat any pairwise post-hoc as exploratory only.
Default seed count is 10 to give the omnibus some power; raise it if compute allows.

Compute note
────────────
Affine-Invariant Riemannian distance was 259–308 s PER N×N matrix at N=500. At
N=2000 across many seeds × {SOAP, MACE} it dominates wall-time even with the cache
(the matrix is reused across targets within a seed, but recomputed once per seed
because each seed permutes the split). ALWAYS run `--smoke` first: it runs a single
seed across all targets, reports per-method timing, and extrapolates the full sweep
before you commit to it. If the estimate is unacceptable, drop the worst offenders
via DISABLE_METHODS (e.g. "{desc}_riemann_affine_invariant").

Usage
─────
  python regression_benchmark.py --smoke         # time one seed, extrapolate, stop
  python regression_benchmark.py                 # full sweep (defaults below)
  python regression_benchmark.py --seeds 42 123 456 --targets gap mu
  python regression_benchmark.py --sample-size 2000 --n-seeds 10

Outputs  (all in OUT_DIR; OUT_DIR/timing for --smoke)
──────────────────────────
  raw_{target}.csv          — one row per (method, seed)
  summary_{target}.csv      — mean ± std across seeds, sorted by test_rmse_mean
  summary_block_{b}_{target}.csv — per-block subset of summary (for LaTeX tables)
  timing_per_method.csv     — (smoke only) per-method seconds + full-sweep estimate
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from loguru import logger
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# ── path setup — must precede local imports ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR  = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regression_task import (  # noqa: E402
    DistanceMatrixCache,
    MethodSpec,
    _fit_one_method,
    _take_rows,
    _pooled_descriptor_matrix,
    _sanitize_distance_matrix,
    _laplacian_kernel_from_distance,
    _rbf_kernel_from_distance,
    _median_beta_from_distances,
    _build_vector_kernel,
    _build_linear_kernel,
    get_regression_methods,
)
from src.datasets import QM9Dataset  # noqa: E402

warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. Kernel / distance helpers for 2D molecular representations
# =============================================================================

def _tanimoto_kernel_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Generalised Tanimoto (Jaccard) kernel for real-valued vectors.
    Known to be PSD for count/binary fingerprints (Ralaivola et al., 2005).
    K(a,b) = (a·b) / (‖a‖² + ‖b‖² − a·b)
    """
    AB = A @ B.T
    A_sq = np.einsum("ij,ij->i", A, A)
    B_sq = np.einsum("ij,ij->i", B, B)
    denom = A_sq[:, None] + B_sq[None, :] - AB
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(denom > 0, AB / denom, 0.0)
    return np.clip(K, 0.0, None)


def _tanimoto_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D_ij = 1 − T(i, j).  Diagonal is 0 for identical inputs."""
    D = 1.0 - _tanimoto_kernel_matrix(A, B)
    D = _sanitize_distance_matrix(np.maximum(D, 0.0))
    if A is B or (A.shape == B.shape and np.array_equal(A, B)):
        np.fill_diagonal(D, 0.0)
    return D


def _cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D_ij = 1 − cos(i, j).  Diagonal is 0 for identical inputs."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    D = _sanitize_distance_matrix(np.maximum(1.0 - A_n @ B_n.T, 0.0))
    if A is B or (A.shape == B.shape and np.array_equal(A, B)):
        np.fill_diagonal(D, 0.0)
    return D


def _build_tanimoto_laplacian(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float], column: str
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = _pooled_descriptor_matrix(train_df, column)
    X_te = _pooled_descriptor_matrix(test_df, column)
    D_tr = _tanimoto_distance_matrix(X_tr, X_tr)
    D_te = _tanimoto_distance_matrix(X_te, X_tr)
    b = _median_beta_from_distances(D_tr) if beta is None else float(beta)
    return _laplacian_kernel_from_distance(D_tr, b), _laplacian_kernel_from_distance(D_te, b)


def _build_tanimoto_direct(
    train_df: pl.DataFrame, test_df: pl.DataFrame, column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the Tanimoto kernel directly as a precomputed kernel for KRR."""
    X_tr = _pooled_descriptor_matrix(train_df, column)
    X_te = _pooled_descriptor_matrix(test_df, column)
    K_tr = _tanimoto_kernel_matrix(X_tr, X_tr)
    K_te = _tanimoto_kernel_matrix(X_te, X_tr)
    K_tr = (K_tr + K_tr.T) / 2.0
    return K_tr, K_te


def _build_cosine_laplacian(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float], column: str
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = _pooled_descriptor_matrix(train_df, column)
    X_te = _pooled_descriptor_matrix(test_df, column)
    D_tr = _cosine_distance_matrix(X_tr, X_tr)
    D_te = _cosine_distance_matrix(X_te, X_tr)
    b = _median_beta_from_distances(D_tr) if beta is None else float(beta)
    return _laplacian_kernel_from_distance(D_tr, b), _laplacian_kernel_from_distance(D_te, b)


def _build_cosine_rbf(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float], column: str
) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = _pooled_descriptor_matrix(train_df, column)
    X_te = _pooled_descriptor_matrix(test_df, column)
    D_tr = _cosine_distance_matrix(X_tr, X_tr)
    D_te = _cosine_distance_matrix(X_te, X_tr)
    b = _median_beta_from_distances(D_tr, squared=True) if beta is None else float(beta)
    return _rbf_kernel_from_distance(D_tr, b), _rbf_kernel_from_distance(D_te, b)


# =============================================================================
# 2. Block 1 — Reference methods (2D baselines)
# =============================================================================

def get_reference_methods(
    has_morgan: bool = True,
    has_chemprop: bool = True,
) -> List[MethodSpec]:
    """
    2D reference baselines.  These use 1D embedding vectors (not per-atom matrices)
    and the chemically standard distance metrics for each representation.

    NOTE: each block uses a distinct column variable (morgan_col / chemprop_col).
    Do NOT reuse one `col` name across blocks — the builder lambdas capture it by
    reference, so a reassigned `col` silently routes every method to the last value.
    """
    methods: List[MethodSpec] = []

    if has_morgan:
        morgan_col = "morgan_fingerprint"
        methods.append(MethodSpec(
            name="morgan_tanimoto_direct",
            kind="kernel",
            builder=lambda tr, te, _b, **_: _build_tanimoto_direct(tr, te, morgan_col),
            beta_grid=[None],
            notes="Morgan r=2 fingerprints with Tanimoto kernel (precomputed). "
                  "Chemically canonical; equivalent to KRR on Tanimoto similarity.",
        ))
        methods.append(MethodSpec(
            name="morgan_tanimoto_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_tanimoto_laplacian(tr, te, b, morgan_col),
            notes="Morgan fingerprints + Tanimoto distance → Laplacian kernel.",
        ))
        methods.append(MethodSpec(
            name="morgan_euclidean_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_kernel(tr, te, b, column=morgan_col, kernel_type="laplacian"),
            notes="Morgan fingerprints + Euclidean distance → Laplacian kernel (Tanimoto ablation).",
        ))
        methods.append(MethodSpec(
            name="morgan_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_linear_kernel(tr, te, b, column=morgan_col),
            beta_grid=[None],
            notes="Morgan fingerprints + linear (dot-product) kernel.",
        ))

    if has_chemprop:
        chemprop_col = "chemprop_embedding"
        methods.append(MethodSpec(
            name="chemprop_cosine_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_cosine_laplacian(tr, te, b, chemprop_col),
            notes="ChemProp GNN embedding + Cosine distance → Laplacian kernel.",
        ))
        methods.append(MethodSpec(
            name="chemprop_cosine_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_cosine_rbf(tr, te, b, chemprop_col),
            notes="ChemProp GNN embedding + Cosine distance → RBF kernel.",
        ))
        methods.append(MethodSpec(
            name="chemprop_euclidean_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_kernel(tr, te, b, column=chemprop_col, kernel_type="laplacian"),
            notes="ChemProp GNN embedding + Euclidean distance → Laplacian (Cosine ablation).",
        ))
        methods.append(MethodSpec(
            name="chemprop_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_linear_kernel(tr, te, b, column=chemprop_col),
            beta_grid=[None],
            notes="ChemProp GNN embedding + linear (dot-product) kernel.",
        ))

    return methods


# =============================================================================
# 3. Block 2 — Controls (performance floor for geometry claims)
# =============================================================================

_QM9_SPECIES = (1, 6, 7, 8, 9)  # H, C, N, O, F


def _composition_matrix(df: pl.DataFrame) -> np.ndarray:
    rows = []
    for zs in df["atomic_numbers"].to_list():
        zs = np.asarray(zs)
        rows.append([int((zs == z).sum()) for z in _QM9_SPECIES])
    return np.asarray(rows, dtype=np.float64)


def evaluate_controls(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    alpha_grid: Sequence[float],
    gamma_grid: Sequence[float] = (0.01, 0.1, 1.0),
    cv: int = 5,
) -> List[Dict[str, Any]]:
    """
    Two mandatory controls for a representative-sample regression benchmark:
      baseline_mean        : predict the training mean (trivial floor).
      baseline_composition : RBF-KRR on element counts {H,C,N,O,F} only.

    Both are evaluated on the same train/test split passed in, so they line up
    with every other method in the merged results table.
    """
    rows: List[Dict[str, Any]] = []

    # 1) Mean predictor (trivial floor) ──────────────────────────────────────
    t0 = time.perf_counter()
    pred = np.full(y_test.shape, float(y_train.mean()))
    row = _metrics_row("baseline_mean", y_test, pred, seed, time.perf_counter() - t0)
    rows.append(row)

    # 2) Composition-only RBF-KRR ─────────────────────────────────────────────
    t0 = time.perf_counter()
    X_tr = _composition_matrix(train_df)
    X_te = _composition_matrix(test_df)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    X_tr_n, X_te_n = (X_tr - mu) / sd, (X_te - mu) / sd

    best: Optional[Tuple[float, float, float]] = None
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for a in alpha_grid:
        for g in gamma_grid:
            fold_mse = []
            for i_tr, i_val in splitter.split(X_tr_n):
                m = KernelRidge(alpha=float(a), kernel="rbf", gamma=float(g))
                m.fit(X_tr_n[i_tr], y_train[i_tr])
                fold_mse.append(mean_squared_error(y_train[i_val], m.predict(X_tr_n[i_val])))
            score = float(np.mean(fold_mse))
            if best is None or score < best[0]:
                best = (score, float(a), float(g))

    _, a_best, g_best = best
    final_m = KernelRidge(alpha=a_best, kernel="rbf", gamma=g_best)
    final_m.fit(X_tr_n, y_train)
    pred = final_m.predict(X_te_n)
    row = _metrics_row("baseline_composition", y_test, pred, seed, time.perf_counter() - t0)
    row["best_alpha"] = a_best
    row["best_beta"] = g_best
    rows.append(row)

    return rows


def _metrics_row(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, seed: int, secs: float
) -> Dict[str, Any]:
    return {
        "method": name,
        "kind": "control",
        "best_alpha": None,
        "best_beta": None,
        "cv_rmse": None,
        "total_seconds": float(secs),
        "train_rmse": None,
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_r2": float(r2_score(y_true, y_pred)),
        "status": "ok",
        "seed": seed,
    }


# =============================================================================
# 4. Run a list of MethodSpec objects (shared by Blocks 1 and 3)
# =============================================================================

def run_method_list(
    methods: List[MethodSpec],
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: Sequence[float],
    cv: int,
    seed: int,
    cache: Optional[DistanceMatrixCache],
) -> List[Dict[str, Any]]:
    """Evaluate each MethodSpec via inner-CV KRR.  One result dict per method."""
    rows: List[Dict[str, Any]] = []
    for spec in methods:
        if not spec.enabled:
            continue
        try:
            res = _fit_one_method(
                spec, train_df, test_df, y_train, y_test,
                alpha_grid=alpha_grid, cv=cv, random_state=seed,
                seed=seed, cache=cache,
            )
            res.pop("model", None)
            res.pop("y_test_pred", None)
            res["seed"] = seed
            rows.append({"status": "ok", **res})
        except Exception as e:
            logger.error(f"  Method '{spec.name}' failed: {e}")
            rows.append({
                "method": spec.name, "kind": spec.kind,
                "seed": seed, "status": "failed", "error": str(e),
                "test_rmse": None, "test_mae": None, "test_r2": None,
            })
    return rows


def _filter_disabled(methods: List[MethodSpec], disabled: set) -> List[MethodSpec]:
    """Drop any MethodSpec whose name is in the disabled set (compute control)."""
    if not disabled:
        return methods
    kept = [m for m in methods if m.name not in disabled]
    dropped = sorted({m.name for m in methods} & disabled)
    if dropped:
        logger.warning(f"  DISABLE_METHODS active — skipping: {dropped}")
    return kept


# =============================================================================
# 5. Friedman omnibus significance test (seeds as blocks) — honest reporting
# =============================================================================

def friedman_across_methods(raw: pl.DataFrame, target: str) -> None:
    """
    Friedman omnibus across methods with SEEDS as blocks, on test_rmse.

    HONESTY: seeds-as-blocks tests reproducibility across random splits, not the
    multi-dataset critical-difference setup Friedman/Nemenyi was designed for.
    With few seed-blocks relative to the number of methods, post-hoc Nemenyi is
    underpowered and a non-significant pair is NOT evidence of equivalence. We
    always print the omnibus; Nemenyi is printed only as an exploratory aid.
    """
    from scipy.stats import friedmanchisquare

    ok = raw.filter(pl.col("status") == "ok")
    wide = ok.pivot(values="test_rmse", index="seed", on="method", aggregate_function="mean")
    methods = [c for c in wide.columns if c != "seed"]
    mat = wide.select(methods).to_numpy()

    n_blocks, n_methods = mat.shape
    if np.isnan(mat).any() or n_blocks < 3:
        logger.warning(
            f"[{target}] Friedman skipped: need ≥3 complete seed-blocks with no NaNs "
            f"(got {n_blocks} blocks, NaN present={np.isnan(mat).any()})."
        )
        return

    stat, p = friedmanchisquare(*[mat[:, j] for j in range(n_methods)])
    logger.info(
        f"[{target}] Friedman omnibus: {n_methods} methods × {n_blocks} seed-blocks  "
        f"→  chi²={stat:.3f}, p={p:.4g}"
    )

    # Power caveat — make the underpowering explicit and quantitative.
    if n_blocks < 2 * n_methods:
        logger.warning(
            f"[{target}] POST-HOC UNDERPOWERED: {n_blocks} seed-blocks vs {n_methods} methods. "
            f"The Nemenyi critical difference will be wide; absence of significant pairs is "
            f"NOT evidence of equality. Report the omnibus above; treat pairwise results as "
            f"exploratory only. (Seeds-as-blocks measures split reproducibility, not the "
            f"multi-dataset CD-diagram setup the test was designed for.)"
        )

    try:
        import scikit_posthocs as sp  # type: ignore
        nem = sp.posthoc_nemenyi_friedman(mat)
        nem.index = methods
        nem.columns = methods
        logger.info(f"[{target}] Nemenyi post-hoc p-values (EXPLORATORY — see caveat):")
        print(nem.round(3))
    except Exception:
        logger.info(
            f"[{target}] Install scikit-posthocs for an (exploratory) Nemenyi post-hoc + "
            f"critical-difference diagram."
        )


# =============================================================================
# 6. Per-target aggregation / output
# =============================================================================

def _write_target_outputs(target: str, target_rows: List[Dict[str, Any]], out_dir: str) -> None:
    raw_df = pl.DataFrame(target_rows)
    raw_path = os.path.join(out_dir, f"raw_{target}.csv")
    raw_df.write_csv(raw_path)
    logger.info(f"[{target}] Saved raw results → {raw_path}")

    ok_df = raw_df.filter(pl.col("status") == "ok")
    summary = (
        ok_df.group_by(["block", "method", "kind"])
        .agg([
            pl.col("test_rmse").mean().alias("test_rmse_mean"),
            pl.col("test_rmse").std().alias("test_rmse_std"),
            pl.col("test_mae").mean().alias("test_mae_mean"),
            pl.col("test_mae").std().alias("test_mae_std"),
            pl.col("test_r2").mean().alias("test_r2_mean"),
            pl.col("test_r2").std().alias("test_r2_std"),
            pl.col("total_seconds").mean().alias("time_sec_mean"),
            pl.col("seed").n_unique().alias("n_seeds"),
        ])
        .sort("test_rmse_mean", nulls_last=True)
    )
    summary_path = os.path.join(out_dir, f"summary_{target}.csv")
    summary.write_csv(summary_path)
    logger.info(f"[{target}] Saved summary  → {summary_path}")

    block_order = {"control": 0, "reference": 1, "controlled": 2}
    for block_name, _ in sorted(block_order.items(), key=lambda x: x[1]):
        block_df = summary.filter(pl.col("block") == block_name)
        if block_df.is_empty():
            continue
        block_df.write_csv(os.path.join(out_dir, f"summary_block_{block_name}_{target}.csv"))

        logger.info(f"\n{'─' * 72}")
        logger.info(f"  [{target}]  BLOCK: {block_name.upper()}")
        logger.info(f"{'─' * 72}")
        logger.info(f"  {'Method':<55s}  {'RMSE (mean±std)':>20s}  {'R²':>8s}")
        for row in block_df.iter_rows(named=True):
            rmse_s = f"{row['test_rmse_mean']:.4f}±{row['test_rmse_std'] or 0:.4f}"
            r2_s   = f"{row['test_r2_mean']:.3f}"
            logger.info(f"  {row['method']:<55s}  {rmse_s:>20s}  {r2_s:>8s}")

    # Omnibus significance across Block-3 methods, per target
    block3_raw = raw_df.filter(pl.col("block") == "controlled")
    if block3_raw.height > 0:
        logger.info(f"\n[{target}] Friedman significance test (Block 3 controlled methods):")
        try:
            friedman_across_methods(block3_raw, target)
        except Exception as e:
            logger.warning(f"[{target}] Significance test skipped: {e}")

    failed_df = raw_df.filter(pl.col("status") != "ok")
    if failed_df.height > 0:
        failed_df.write_csv(os.path.join(out_dir, f"failed_methods_{target}.csv"))
        failed_names = failed_df["method"].unique().to_list()
        logger.warning(
            f"[{target}] {failed_df.height} failures across {len(failed_names)} methods: {failed_names}"
        )


def _print_timing_report(all_rows: List[Dict[str, Any]], n_full_seeds: int, out_dir: str) -> None:
    """Smoke-mode timing: per-method seconds + extrapolation to the full sweep."""
    df = pl.DataFrame(all_rows).filter(pl.col("status") == "ok")
    if df.is_empty():
        logger.warning("Timing report skipped — no successful rows.")
        return

    # Average per-method time across whatever targets ran in the single seed.
    per_method = (
        df.group_by(["block", "method"])
        .agg(pl.col("total_seconds").mean().alias("sec_per_seed"))
        .sort("sec_per_seed", descending=True)
    )
    per_method.write_csv(os.path.join(out_dir, "timing_per_method.csv"))

    one_seed_total = float(df["total_seconds"].sum())
    est_full = one_seed_total * n_full_seeds

    logger.info(f"\n{'=' * 72}")
    logger.info("  SMOKE / TIMING REPORT (single seed, all targets)")
    logger.info(f"{'=' * 72}")
    logger.info(f"  {'Method':<55s}  {'sec (this seed)':>16s}")
    for row in per_method.iter_rows(named=True):
        logger.info(f"  {row['method']:<55s}  {row['sec_per_seed']:>16.2f}")
    logger.info(f"{'─' * 72}")
    logger.info(f"  Summed method time, 1 seed (all targets) : {one_seed_total:>10.1f} s")
    logger.info(
        f"  ESTIMATED full sweep ({n_full_seeds} seeds)       : "
        f"{est_full:>10.1f} s  (~{est_full / 60:.1f} min, ~{est_full / 3600:.2f} h)"
    )
    logger.info(
        "  NOTE: extrapolation is roughly linear in seeds because each seed permutes "
        "the split and recomputes its N×N matrices once (then reuses them across targets)."
    )
    logger.info(
        "  If this is too slow, set DISABLE_METHODS (e.g. the affine-invariant Riemannian "
        "method) and re-run --smoke."
    )
    logger.info(f"  Per-method timing written → {os.path.join(out_dir, 'timing_per_method.csv')}")


# =============================================================================
# 7. Main driver
# =============================================================================

def main() -> None:
    # ── Defaults ───────────────────────────────────────────────────────────
    DEFAULT_SEEDS = [42, 123, 456, 789, 1011]  # 10 seeds

    parser = argparse.ArgumentParser(description="Three-block QM9 regression benchmark.")
    parser.add_argument("--targets", nargs="+",
                        default=["gap", "geometric_strain", "mu"],
                        help="Target columns to loop over (electronic / geometric / directional).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit list of split seeds. Overrides --n-seeds.")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds to take from the default pool (if --seeds unset).")
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="QM9 stratified sample size.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run ONE seed across all targets, report timing + a full-sweep "
                             "estimate, then stop. Always do this before the full sweep.")
    args = parser.parse_args()

    # ── Configuration ──────────────────────────────────────────────────────
    DESCRIPTORS_3D   = ["soap", "mace"]   # 3D per-atom descriptors (Block 3)
    DESCRIPTORS_2D   = ["onehot", "transformer", "morgan", "chemprop"]  # 2D vector descriptors
    ALL_DESCRIPTORS  = DESCRIPTORS_3D + DESCRIPTORS_2D

    TARGETS          = list(args.targets)
    SAMPLE_SIZE      = args.sample_size
    ALPHA_GRID       = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0)
    CV_FOLDS         = 5

    GRASSMANN_K      = 3
    REMATCH_ALPHA    = 0.1

    # Compute-control switch. Add method names here to skip them everywhere,
    # e.g. {"soap_riemann_affine_invariant", "mace_riemann_affine_invariant"}.
    DISABLE_METHODS: set = set()

    if args.seeds is not None:
        SPLIT_SEEDS = list(args.seeds)
    else:
        SPLIT_SEEDS = DEFAULT_SEEDS[: max(1, args.n_seeds)]

    n_full_seeds = len(SPLIT_SEEDS)

    USE_CACHE        = True
    CACHE_DIR        = ".cache/regression_benchmark_matrices"
    base_out         = f"results/qm9/regression_benchmark/n_{SAMPLE_SIZE}"
    OUT_DIR          = os.path.join(base_out, "timing") if args.smoke else base_out
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke:
        SPLIT_SEEDS = SPLIT_SEEDS[:1]
        logger.info(
            f"SMOKE MODE: running 1 seed ({SPLIT_SEEDS[0]}) across targets {TARGETS} to time "
            f"the sweep. Full run would use {n_full_seeds} seeds."
        )

    logger.info(
        f"Config: targets={TARGETS}  seeds={SPLIT_SEEDS}  sample_size={SAMPLE_SIZE}  "
        f"smoke={args.smoke}"
    )

    logger.info(f"Loading QM9 stratified sample (limit={SAMPLE_SIZE}, all descriptors)...")
    qm9 = QM9Dataset(limit=SAMPLE_SIZE, descriptors=ALL_DESCRIPTORS)
    subset = qm9.load(force_process=False)
    logger.info(f"Loaded {subset.height} molecules.")

    # ── Validate target columns ────────────────────────────────────────────
    missing_targets = [t for t in TARGETS if t not in subset.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in dataset: {missing_targets}")

    n_before = subset.height
    mask = pl.lit(True)
    for t in TARGETS:
        mask = mask & pl.col(t).is_finite()
    if "geometric_strain" in TARGETS:
        mask = mask & (pl.col("geometric_strain") >= 0)
    subset = subset.filter(mask)
    logger.info(
        f"Filtered to molecules valid for ALL targets {TARGETS}: "
        f"{subset.height}/{n_before} retained "
        f"(this fixed set is shared across every target and method)."
    )
    if subset.height < 50:
        raise ValueError(f"Too few molecules after joint-target filtering: {subset.height}.")

    # ── Detect which 2D descriptors are available ──────────────────────────
    has_morgan   = "morgan_fingerprint"  in subset.columns
    has_chemprop = "chemprop_embedding"  in subset.columns
    has_atom_z   = "atomic_numbers"      in subset.columns
    logger.info(
        f"Available: morgan={has_morgan}, chemprop={has_chemprop}, atomic_numbers={has_atom_z}"
    )

    cache = DistanceMatrixCache(CACHE_DIR) if USE_CACHE else None

    # Pre-extract target vectors once.
    y_by_target = {t: subset[t].to_numpy().astype(np.float64) for t in TARGETS}

    # Accumulate rows per target across all seeds.
    rows_by_target: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TARGETS}

    wall_start = time.perf_counter()

    # ── Seed-outer / target-inner: split depends only on seed, so the N×N
    #    matrices are computed once per seed and reused across all targets. ──
    for seed in SPLIT_SEEDS:
        logger.info(f"\n{'=' * 72}")
        logger.info(f"  Seed {seed}")
        logger.info(f"{'=' * 72}")

        train_idx, test_idx = train_test_split(
            np.arange(subset.height), test_size=0.2, random_state=seed, shuffle=True,
        )
        train_df = _take_rows(subset, train_idx)
        test_df  = _take_rows(subset, test_idx)
        logger.info(f"  Train: {len(train_idx)}  |  Test: {len(test_idx)}")

        for target in TARGETS:
            y_full  = y_by_target[target]
            y_train = y_full[train_idx]
            y_test  = y_full[test_idx]
            logger.info(f"\n  ── Target: {target.upper()}  (seed {seed}) ──")

            # ── Block 2: Controls ──────────────────────────────────────────
            logger.info("  [Block 2 — Control]")
            if has_atom_z:
                ctrl_rows = evaluate_controls(
                    train_df, test_df, y_train, y_test,
                    seed=seed, alpha_grid=ALPHA_GRID, cv=CV_FOLDS,
                )
                for r in ctrl_rows:
                    r["block"] = "control"; r["target"] = target
                    rows_by_target[target].append(r)
                    logger.info(f"    {r['method']:<40s}  test_rmse={r['test_rmse']:.4f}")
            else:
                logger.warning("  atomic_numbers missing — composition control skipped.")
                t0 = time.perf_counter()
                pred = np.full(y_test.shape, float(y_train.mean()))
                r = _metrics_row("baseline_mean", y_test, pred, seed, time.perf_counter() - t0)
                r["block"] = "control"; r["target"] = target
                rows_by_target[target].append(r)

            # ── Block 1: Reference (2D baselines) ──────────────────────────
            if has_morgan or has_chemprop:
                logger.info("  [Block 1 — Reference]")
                ref_methods = _filter_disabled(
                    get_reference_methods(has_morgan=has_morgan, has_chemprop=has_chemprop),
                    DISABLE_METHODS,
                )
                ref_rows = run_method_list(
                    ref_methods, train_df, test_df, y_train, y_test,
                    alpha_grid=ALPHA_GRID, cv=CV_FOLDS, seed=seed, cache=None,
                )
                for r in ref_rows:
                    r["block"] = "reference"; r["target"] = target
                    rows_by_target[target].append(r)
                    if r["status"] == "ok":
                        logger.info(f"    {r['method']:<40s}  test_rmse={r.get('test_rmse', '?'):.4f}")
                    else:
                        logger.warning(f"    {r['method']:<40s}  FAILED: {r.get('error')}")

            # ── Block 3: Controlled (SOAP & MACE × geometry suite) ─────────
            logger.info("  [Block 3 — Controlled]")
            for desc in DESCRIPTORS_3D:
                emb_col = f"{desc}_embedding"
                mat_col = f"{desc}_matrix"
                if emb_col not in subset.columns or mat_col not in subset.columns:
                    logger.warning(f"    {desc}: missing columns — skipping.")
                    continue

                logger.info(f"    Descriptor: {desc.upper()}")
                controlled_methods = _filter_disabled(
                    get_regression_methods(
                        descriptor=desc, grassmann_k=GRASSMANN_K, rematch_alpha=REMATCH_ALPHA,
                    ),
                    DISABLE_METHODS,
                )
                c_rows = run_method_list(
                    controlled_methods, train_df, test_df, y_train, y_test,
                    alpha_grid=ALPHA_GRID, cv=CV_FOLDS, seed=seed, cache=cache,
                )
                for r in c_rows:
                    r["block"] = "controlled"; r["target"] = target
                    rows_by_target[target].append(r)
                    if r["status"] == "ok":
                        logger.info(f"      {r['method']:<50s}  test_rmse={r.get('test_rmse', '?'):.4f}")
                    else:
                        logger.warning(f"      {r['method']:<50s}  FAILED: {r.get('error')}")

        if not args.smoke:
            seed_dir = os.path.join(OUT_DIR, "seeds")
            os.makedirs(seed_dir, exist_ok=True)
            seed_rows = [
                r for rows in rows_by_target.values()
                for r in rows if r.get("seed") == seed
            ]
            pl.DataFrame(seed_rows).write_csv(
                os.path.join(seed_dir, f"raw_seed_{seed}.csv")
            )
            logger.info(
                f"  [checkpoint] seed {seed}: wrote {len(seed_rows)} rows "
                f"→ seeds/raw_seed_{seed}.csv"
            )

    wall_total = time.perf_counter() - wall_start
    logger.info(f"\nWall time for run: {wall_total:.1f} s ({wall_total/60:.1f} min)")

    # ── Smoke mode: timing report + extrapolation, then stop ───────────────
    if args.smoke:
        flat = [r for rows in rows_by_target.values() for r in rows]
        _print_timing_report(flat, n_full_seeds=n_full_seeds, out_dir=OUT_DIR)
        logger.info(
            "\nSMOKE COMPLETE. Review the estimate above, set DISABLE_METHODS if needed, "
            "then launch the full sweep without --smoke."
        )
        return

    # ── Full sweep: write per-target outputs + per-target significance ─────
    for target in TARGETS:
        logger.info(f"\n{'#' * 72}")
        logger.info(f"  AGGREGATING TARGET: {target.upper()}")
        logger.info(f"{'#' * 72}")
        _write_target_outputs(target, rows_by_target[target], OUT_DIR)

    # Combined raw across all targets (handy for cross-target plots).
    combined = pl.DataFrame([r for rows in rows_by_target.values() for r in rows])
    combined.write_csv(os.path.join(OUT_DIR, "raw_all_targets.csv"))

    logger.info(f"\nAll results saved to: {OUT_DIR}")
    logger.info(
        "Reminder for the write-up: report std ACROSS SEEDS (as done here), not across "
        "CV folds, and present the Friedman omnibus with the post-hoc framed as exploratory."
    )


if __name__ == "__main__":
    main()