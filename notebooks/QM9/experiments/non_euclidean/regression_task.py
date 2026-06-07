from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Adjust REPO_ROOT as needed to resolve imports from src
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.non_euclidean import (  # noqa: E402
    Grassmann,
    PersistentHomology,
    Riemann,
)

from src.optimal_transport import Wasserstein, REMatch # noqa: E402

ArrayPair = Tuple[np.ndarray, np.ndarray]
KernelBuilder = Callable[[pl.DataFrame, pl.DataFrame, Optional[float]], ArrayPair]

class MethodSpec:
    def __init__(
        self,
        name: str,
        kind: str,
        builder: KernelBuilder,
        beta_grid: Optional[Sequence[float]] = None,
        enabled: bool = True,
        notes: str = ""
    ):
        self.name = name
        self.kind = kind
        self.builder = builder
        self.beta_grid = beta_grid
        self.enabled = enabled
        self.notes = notes

# --- Data Cleaning and Polars Utilities ---

def _as_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)

def _take_rows(df: pl.DataFrame, indices: Sequence[int]) -> pl.DataFrame:
    return df[np.asarray(indices, dtype=np.int64).tolist()]

def _clean_regression_df(df: Any, target_col: str, descriptor: str) -> pl.DataFrame:
    df = _as_polars(df)
    required_cols = {target_col, f"{descriptor}_embedding", f"{descriptor}_matrix"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for {descriptor}: {missing}")

    cleaned = df.drop_nulls(list(required_cols)).filter(pl.col(target_col).is_not_nan())
    if cleaned.height < 10:
        raise ValueError(f"Insufficient valid rows after dropping nulls for {descriptor}.")
    return cleaned

def _target_array(df: pl.DataFrame, target_col: str) -> np.ndarray:
    y = np.asarray(df[target_col].to_list(), dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError(f"Target column '{target_col}' contains non-finite values.")
    return y

def _pooled_descriptor_matrix(df: pl.DataFrame, column: str) -> np.ndarray:
    rows = [np.asarray(val, dtype=np.float64).ravel() for val in df[column].to_list()]
    return np.vstack(rows)

# --- Matrix Math & Kernel Functions ---

def _sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return D
    finite = np.isfinite(D)
    if not finite.all():
        replacement = np.nanmax(D[finite]) if finite.any() else 0.0
        D = np.nan_to_num(D, nan=replacement, posinf=replacement, neginf=0.0)
    return np.maximum(D, 0.0)

def _laplacian_kernel_from_distance(D: np.ndarray, beta: float) -> np.ndarray:
    D = _sanitize_distance_matrix(D)
    K = np.exp(-float(beta) * D)
    return np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)

def _rbf_kernel_from_distance(D: np.ndarray, beta: float) -> np.ndarray:
    D = _sanitize_distance_matrix(D)
    K = np.exp(-float(beta) * (D ** 2))
    return np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)

def _median_beta_from_distances(D: np.ndarray, squared: bool = False) -> float:
    D = _sanitize_distance_matrix(D)
    values = D[np.triu_indices_from(D, k=1)] if D.ndim == 2 and D.shape[0] == D.shape[1] else D.ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    median_val = np.median(values)
    return 1.0 / float(median_val ** 2) if squared and median_val > 0 else 1.0 / float(median_val)

# --- Memory-Safe Builders (Leakage-Free) ---

def _build_distance_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float],
    distance_matrix_fn: Callable[[pl.DataFrame], np.ndarray], kernel_type: str = "laplacian"
) -> ArrayPair:
    full_df = pl.concat([train_df, test_df], how="vertical")
    D_full = np.asarray(distance_matrix_fn(full_df), dtype=np.float64)
    
    n_train = train_df.height
    D_train = _sanitize_distance_matrix(D_full[:n_train, :n_train])
    D_test = _sanitize_distance_matrix(D_full[n_train:, :n_train])
    del D_full  # RAM safety for large N
    
    is_squared = (kernel_type.lower() == "rbf")
    beta_val = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
    
    if is_squared:
        return _rbf_kernel_from_distance(D_train, beta_val), _rbf_kernel_from_distance(D_test, beta_val)
    return _laplacian_kernel_from_distance(D_train, beta_val), _laplacian_kernel_from_distance(D_test, beta_val)

def _build_projection_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float],
    projection_fn: Callable[[pl.DataFrame], np.ndarray], kernel_type: str = "laplacian"
) -> ArrayPair:
    X_train = projection_fn(train_df)
    X_test = projection_fn(test_df)
    
    D_train = _sanitize_distance_matrix(cdist(X_train, X_train, metric="euclidean"))
    D_test = _sanitize_distance_matrix(cdist(X_test, X_train, metric="euclidean"))
    
    is_squared = (kernel_type.lower() == "rbf")
    beta_val = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
    
    if is_squared:
        return _rbf_kernel_from_distance(D_train, beta_val), _rbf_kernel_from_distance(D_test, beta_val)
    return _laplacian_kernel_from_distance(D_train, beta_val), _laplacian_kernel_from_distance(D_test, beta_val)

def _build_vector_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float],
    column: str, kernel_type: str = "laplacian"
) -> ArrayPair:
    X_train = _pooled_descriptor_matrix(train_df, column=column)
    X_test = _pooled_descriptor_matrix(test_df, column=column)
    
    D_train = _sanitize_distance_matrix(cdist(X_train, X_train, metric="euclidean"))
    D_test = _sanitize_distance_matrix(cdist(X_test, X_train, metric="euclidean"))
    
    is_squared = (kernel_type.lower() == "rbf")
    beta_val = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
    
    if is_squared:
        return _rbf_kernel_from_distance(D_train, beta_val), _rbf_kernel_from_distance(D_test, beta_val)
    return _laplacian_kernel_from_distance(D_train, beta_val), _laplacian_kernel_from_distance(D_test, beta_val)

# --- Linear (Dot Product) Builders for Geometric Ablation ---

def _build_linear_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float], column: str
) -> ArrayPair:
    """Builds a pure linear dot-product kernel: K = X @ X^T. Beta is ignored."""
    X_train = _pooled_descriptor_matrix(train_df, column=column)
    X_test = _pooled_descriptor_matrix(test_df, column=column)
    
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    return K_train, K_test

def _build_projection_linear_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float],
    projection_fn: Callable[[pl.DataFrame], np.ndarray]
) -> ArrayPair:
    """Builds a pure linear dot-product kernel for manifold projections. Beta is ignored."""
    X_train = projection_fn(train_df)
    X_test = projection_fn(test_df)
    
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    return K_train, K_test

# --- Dynamic Method Factory ---

def get_regression_methods(
    descriptor: str,
    grassmann_k: int = 3,
    rematch_alpha: float = 0.1,
) -> List[MethodSpec]:
    
    return [
        # --- Pure Linear Baselines (Ablation) ---
        MethodSpec(
            name=f"{descriptor}_avg_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_linear_kernel(tr, te, b, column=f"{descriptor}_embedding"),
            beta_grid=[None],
            notes=f"Averaged {descriptor.upper()} descriptor with pure Linear dot-product kernel.",
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_tangent_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_linear_kernel(
                tr, te, b, lambda df: Riemann.vectorized_spd_matrices(df, descriptor=descriptor, pca=False)
            ),
            beta_grid=[None],
            notes="Log-Euclidean Tangent mapping with pure Linear dot-product kernel.",
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_logeuclidean_dist_laplacian",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="log-euclidean", pca=False), "laplacian"
            ),
            notes="Direct Log-Euclidean distance matrix. Should mathematically match tangent_laplacian."
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_projection_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_linear_kernel(
                tr, te, b, lambda df: Grassmann.get_projection_features(df, descriptor=descriptor, k=grassmann_k, vectorization_type='isometric')
            ),
            beta_grid=[None],
            notes="Grassmann Isometric subspace mapping with pure Linear dot-product kernel.",
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_projection_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_kernel(
                tr, te, b, lambda df: Grassmann.get_projection_features(df, descriptor=descriptor, k=grassmann_k, vectorization_type='isometric'), "laplacian"
            ),
            notes="Euclidean distance applied to Grassmann projection."
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_chordal_dist_laplacian",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: Grassmann.distance_matrix(df=df, descriptor=descriptor, k=grassmann_k, distance_type="chordal"), "laplacian"
            ),
            notes="Direct Chordal distance matrix. Should mathematically match projection_laplacian."
        ),
        
        # --- Non-Linear Baselines (RBF / Laplacian) ---
        MethodSpec(
            name=f"{descriptor}_avg_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_kernel(tr, te, b, column=f"{descriptor}_embedding", kernel_type="laplacian"),
        ),
        MethodSpec(
            name=f"{descriptor}_avg_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_kernel(tr, te, b, column=f"{descriptor}_embedding", kernel_type="rbf"),
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_tangent_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_kernel(
                tr, te, b, lambda df: Riemann.vectorized_spd_matrices(df, descriptor=descriptor, pca=False), "laplacian"
            ),
        ),
        
        MethodSpec(
            name=f"{descriptor}_rematch_direct",
            kind="kernel",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: REMatch.kernel_matrix(df=df, descriptor=descriptor, alpha=rematch_alpha), "laplacian"
            ),
            beta_grid=[None], 
            notes="REMatch Entropic OT Kernel"
        ),
        MethodSpec(
            name=f"{descriptor}_wasserstein_w1",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, 
                lambda df: Wasserstein.distance_matrix(df, descriptor=descriptor, metric="euclidean"), 
                "laplacian"
            ),
            notes="Earth Mover's Distance (W1) with standard Euclidean ground cost."
        ),
        MethodSpec(
            name=f"{descriptor}_wasserstein_w2",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, 
                lambda df: np.sqrt(Wasserstein.distance_matrix(df, descriptor=descriptor, metric="sqeuclidean")), 
                "laplacian"
            ),
            notes="Wasserstein-2 (W2) distance requiring sqrt of the sqeuclidean EMD cost."
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_geodesic",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: Grassmann.distance_matrix(df=df, descriptor=descriptor, k=grassmann_k, distance_type="geodesic"), "laplacian"
            ),
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_affine_invariant",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="affine-invariant", pca=False), "laplacian"
            ),
        ),
        MethodSpec(
            name=f"{descriptor}_wasserstein_w1",
            kind="distance",
            builder=lambda tr, te, b: _build_distance_kernel(
                tr, te, b, lambda df: Wasserstein.distance_matrix(df, descriptor=descriptor, metric="euclidean"), "laplacian"
            ),
        ),
    ]

# --- Evaluation Core ---

def _cv_score_precomputed_krr(K_train: np.ndarray, y_train: np.ndarray, alpha: float, cv: int, random_state: int) -> float:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for inner_train_idx, inner_val_idx in splitter.split(K_train):
        K_inner_train = K_train[np.ix_(inner_train_idx, inner_train_idx)]
        K_inner_val = K_train[np.ix_(inner_val_idx, inner_train_idx)]
        model = KernelRidge(alpha=float(alpha), kernel="precomputed")
        model.fit(K_inner_train, y_train[inner_train_idx])
        pred = model.predict(K_inner_val)
        scores.append(mean_squared_error(y_train[inner_val_idx], pred))
    return float(np.mean(scores))

def _fit_one_method(
    method: MethodSpec, train_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_test: np.ndarray, alpha_grid: Sequence[float],
    beta_grid: Optional[Sequence[float]], cv: int, random_state: int
) -> Dict[str, Any]:
    
    logger.info(f"Evaluating regression method: {method.name}")
    method_started = time.perf_counter()
    candidate_betas = list(method.beta_grid) if method.beta_grid is not None else [None]
    
    kernel_cache: Dict[Any, ArrayPair] = {}
    best: Optional[Dict[str, Any]] = None

    for beta in candidate_betas:
        try:
            if beta not in kernel_cache:
                K_train, K_test = method.builder(train_df, test_df, beta)
                K_train = np.asarray(K_train, dtype=np.float64)
                K_train = (K_train + K_train.T) / 2.0  # Symmetrize
                kernel_cache[beta] = (K_train, np.asarray(K_test, dtype=np.float64))
            
            K_train, K_test = kernel_cache[beta]

            for alpha in alpha_grid:
                cv_mse = _cv_score_precomputed_krr(K_train, y_train, float(alpha), cv, random_state)
                if best is None or cv_mse < best["cv_mse"]:
                    best = {
                        "alpha": float(alpha), "beta": None if beta is None else float(beta),
                        "cv_mse": cv_mse, "K_train": K_train, "K_test": K_test
                    }
        except Exception as e:
            warnings.warn(f"Skipping beta={beta} for '{method.name}': {e}")
            continue

    if best is None:
        raise RuntimeError(f"No valid hyperparameters found for '{method.name}'.")

    model = KernelRidge(alpha=best["alpha"], kernel="precomputed")
    model.fit(best["K_train"], y_train)
    
    pred_train = model.predict(best["K_train"])
    pred_test = model.predict(best["K_test"])
    
    return {
        "method": method.name,
        "kind": method.kind,
        "best_alpha": best["alpha"],
        "best_beta": best["beta"],
        "cv_rmse": float(np.sqrt(best["cv_mse"])),
        "total_seconds": float(time.perf_counter() - method_started),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "model": model,
        "y_test_pred": pred_test,
    }

def compare_non_euclidean_regression(
    df: Any, descriptor: str, target_col: str,
    alpha_grid: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0),
    test_size: float = 0.2, cv: int = 5, random_state: int = 40,
) -> Dict[str, Any]:
    
    cleaned = _clean_regression_df(df, target_col, descriptor)
    y = _target_array(cleaned, target_col)
    
    train_idx, test_idx = train_test_split(
        np.arange(cleaned.height), test_size=test_size, random_state=random_state, shuffle=True
    )
    
    train_df, test_df = _take_rows(cleaned, train_idx), _take_rows(cleaned, test_idx)
    y_train, y_test = y[train_idx], y[test_idx]
    specs = get_regression_methods(descriptor=descriptor)

    rows, fitted, predictions = [], {}, {}

    for spec in specs:
        try:
            res = _fit_one_method(spec, train_df, test_df, y_train, y_test, alpha_grid, None, cv, random_state)
            fitted[spec.name] = res.pop("model")
            predictions[spec.name] = res.pop("y_test_pred")
            rows.append({"status": "ok", **res})
        except Exception as e:
            logger.error(f"Method '{spec.name}' failed: {e}")
            rows.append({"method": spec.name, "status": "failed", "error": str(e)})

    return {
        "results": pl.DataFrame(rows).sort("test_rmse", nulls_last=True),
        "models": fitted,
        "predictions": predictions,
        "y_test": y_test,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }

if __name__ == '__main__':
    from src.datasets import QM9Dataset

    n = 100
    qm9 = QM9Dataset(limit=n, descriptors=["soap", "mace"])
    df = qm9.load()
    
    output_dir = "results/qm9/regression"
    os.makedirs(output_dir, exist_ok=True)
    
    for target in ["gap", "mu", "cv"]:
        logger.info(f"=== Starting benchmark for target: {target.upper()} ===")
        for desc in ["soap", "mace"]:
            logger.info(f"--- Evaluating descriptor: {desc.upper()} ---")
            
            comparison = compare_non_euclidean_regression(df, descriptor=desc, target_col=target)
            
            comparison["results"].write_csv(os.path.join(output_dir, f"results_{target}_{desc}_{n}.csv"))
            
            artifacts = {
                "models": comparison["models"],
                "predictions": comparison["predictions"],
                "y_test": comparison["y_test"],
                "train_indices": comparison["train_indices"],
                "test_indices": comparison["test_indices"]
            }
            with open(os.path.join(output_dir, f"artifacts_{target}_{desc}_{n}.pkl"), "wb") as f:
                pickle.dump(artifacts, f)