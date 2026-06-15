from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import hashlib
import json
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

warnings.filterwarnings("ignore", message="Singular matrix in solving dual problem")

ArrayPair = Tuple[np.ndarray, np.ndarray]
KernelBuilder = Callable[[pl.DataFrame, pl.DataFrame, Optional[float]], ArrayPair]

class DistanceMatrixCache:
    def __init__(self, cache_dir: str = ".cache/distance_matrices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_mol_ids_key(self, df: pl.DataFrame) -> str:
        """Generate a hash key from the mol_ids in the dataframe."""
        if "mol_id" not in df.columns:
            return "unknown"
        mol_ids = sorted(df["mol_id"].to_list())
        mol_ids_str = "_".join(str(mid) for mid in mol_ids)
        return hashlib.sha256(mol_ids_str.encode()).hexdigest()[:16]

    def _get_cache_key(self, seed: int, df: pl.DataFrame, method_name: str) -> Tuple[str, Path]:
        """Generate cache key and path for a specific seed, dataframe, and method."""
        mol_ids_key = self._get_mol_ids_key(df)
        cache_key = f"seed_{seed}_molids_{mol_ids_key}_{method_name}"
        cache_path = self.cache_dir / f"{cache_key}.npz"
        return cache_key, cache_path

    def get_metadata_path(self, seed: int, df: pl.DataFrame, method_name: str) -> Path:
        """Get the metadata file path (stores mol_ids for validation)."""
        _, cache_path = self._get_cache_key(seed, df, method_name)
        return cache_path.with_suffix(".json")

    def save(self, D_full: np.ndarray, seed: int, df: pl.DataFrame, method_name: str) -> Path:
        """Save distance matrix to cache."""
        _, cache_path = self._get_cache_key(seed, df, method_name)
        
        np.savez_compressed(cache_path, D_full=D_full)
        
        metadata_path = self.get_metadata_path(seed, df, method_name)
        metadata = {
            "seed": seed,
            "method": method_name,
            "mol_ids": df["mol_id"].to_list() if "mol_id" in df.columns else [],
            "n_samples": df.height,
            "cached_at": time.time()
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"Cached distance matrix to {cache_path}")
        return cache_path

    def load(self, seed: int, df: pl.DataFrame, method_name: str) -> Optional[np.ndarray]:
        """Load distance matrix from cache if it exists and is valid."""
        _, cache_path = self._get_cache_key(seed, df, method_name)
        metadata_path = self.get_metadata_path(seed, df, method_name)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            current_mol_ids = sorted(df["mol_id"].to_list()) if "mol_id" in df.columns else []
            cached_mol_ids = sorted(metadata.get("mol_ids", []))
            
            if current_mol_ids != cached_mol_ids:
                logger.warning(
                    f"Mol IDs mismatch for seed={seed}, method={method_name}. "
                    f"Recalculating distance matrix."
                )
                return None
            
            if metadata.get("n_samples") != df.height:
                logger.warning(
                    f"Sample count mismatch for seed={seed}, method={method_name}. "
                    f"Recalculating distance matrix."
                )
                return None
            
            loaded_data = np.load(cache_path)
            D_full = loaded_data["D_full"]
            logger.info(f"Loaded cached distance matrix from {cache_path}")
            return D_full
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def clear(self):
        """Clear all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared cache directory {self.cache_dir}")

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

_RBF_LS_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)   # x median-heuristic length-scale, inner-CV tuned

def _build_vector_rbf_tuned(train_df, test_df, mult, column):
    X_tr = _pooled_descriptor_matrix(train_df, column=column)
    X_te = _pooled_descriptor_matrix(test_df, column=column)
    D_tr = _sanitize_distance_matrix(cdist(X_tr, X_tr, metric="euclidean"))
    D_te = _sanitize_distance_matrix(cdist(X_te, X_tr, metric="euclidean"))
    beta = _median_beta_from_distances(D_tr, squared=True) * (1.0 if mult is None else float(mult))
    return _rbf_kernel_from_distance(D_tr, beta), _rbf_kernel_from_distance(D_te, beta)

def _build_projection_rbf_tuned(train_df, test_df, mult, projection_fn):
    X_tr, X_te = projection_fn(train_df), projection_fn(test_df)
    D_tr = _sanitize_distance_matrix(cdist(X_tr, X_tr, metric="euclidean"))
    D_te = _sanitize_distance_matrix(cdist(X_te, X_tr, metric="euclidean"))
    beta = _median_beta_from_distances(D_tr, squared=True) * (1.0 if mult is None else float(mult))
    return _rbf_kernel_from_distance(D_tr, beta), _rbf_kernel_from_distance(D_te, beta)

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

# --- Memory & Disk-Safe Builders ---

def _build_distance_kernel(
    train_df: pl.DataFrame, 
    test_df: pl.DataFrame, 
    beta: Optional[float], 
    distance_matrix_fn: Callable[[pl.DataFrame], np.ndarray],
    kernel_type: str = "laplacian",
    seed: Optional[int] = None,
    cache: Optional[DistanceMatrixCache] = None,
    method_name: Optional[str] = None,
) -> ArrayPair:
    n_train = train_df.height
    full_df = pl.concat([train_df, test_df], how="vertical")
    
    D_full = None
    if seed is not None and cache is not None and method_name is not None:
        D_full = cache.load(seed, full_df, method_name)
    
    if D_full is None:
        D_full = np.asarray(distance_matrix_fn(full_df), dtype=np.float64)
        if seed is not None and cache is not None and method_name is not None:
            cache.save(D_full, seed, full_df, method_name)
    
    D_train = _sanitize_distance_matrix(D_full[:n_train, :n_train])
    D_test = _sanitize_distance_matrix(D_full[n_train:, :n_train])
    
    is_squared = (kernel_type.lower() == "rbf")
    beta_val = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
    
    if is_squared:
        return _rbf_kernel_from_distance(D_train, beta_val), _rbf_kernel_from_distance(D_test, beta_val)
    return _laplacian_kernel_from_distance(D_train, beta_val), _laplacian_kernel_from_distance(D_test, beta_val)

def _sanitize_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Sanitize kernel matrix for numerical stability."""
    K = np.asarray(K, dtype=np.float64)
    if K.size == 0:
        return K
    
    finite = np.isfinite(K)
    if not finite.all():
        logger.warning(f"Kernel matrix contains {(~finite).sum()} non-finite values")
        replacement = np.nanmax(K[finite]) if finite.any() else 1.0
        K = np.nan_to_num(K, nan=replacement, posinf=replacement, neginf=replacement)
    
    if K.ndim == 2 and K.shape[0] == K.shape[1]:
        K = (K + K.T) / 2.0
    
    K = np.maximum(K, 1e-10)
    return K

def _build_direct_kernel(
    train_df: pl.DataFrame, 
    test_df: pl.DataFrame, 
    kernel_matrix_fn: Callable[[pl.DataFrame], np.ndarray],
    seed: Optional[int] = None,
    cache: Optional[DistanceMatrixCache] = None,
    method_name: Optional[str] = None,
) -> ArrayPair:
    n_train = train_df.height
    full_df = pl.concat([train_df, test_df], how="vertical")
    
    K_full = None
    if seed is not None and cache is not None and method_name is not None:
        K_full = cache.load(seed, full_df, method_name)
    
    if K_full is None:
        K_full = np.asarray(kernel_matrix_fn(full_df), dtype=np.float64)
        K_full = _sanitize_kernel_matrix(K_full)
        
        if seed is not None and cache is not None and method_name is not None:
            cache.save(K_full, seed, full_df, method_name)
    
    K_train = _sanitize_kernel_matrix(K_full[:n_train, :n_train])
    K_test = _sanitize_kernel_matrix(K_full[n_train:, :n_train])
    
    return K_train, K_test

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

# --- Linear (Dot Product) Builders ---

def _build_linear_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float], column: str
) -> ArrayPair:
    X_train = _pooled_descriptor_matrix(train_df, column=column)
    X_test = _pooled_descriptor_matrix(test_df, column=column)
    
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    return K_train, K_test

def _build_projection_linear_kernel(
    train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float],
    projection_fn: Callable[[pl.DataFrame], np.ndarray]
) -> ArrayPair:
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
            name=f"{descriptor}_covariance_flat_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_linear_kernel(
                tr, te, b,
                lambda df: Riemann.flat_vectorized_spd_matrices(df, descriptor=descriptor, pca=False)
            ),
            beta_grid=[None],
            notes="CONTROL: flat (Frobenius) vectorization of the SAME OAS covariance as "
                "riemann_tangent_linear, WITHOUT the matrix log. Isolates second-moment from "
                "curvature. R2(tangent_linear) - R2(this) = pure curvature gain.",
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
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="log-euclidean", pca=False), "laplacian", **kw
            ),
            notes="Direct Log-Euclidean distance matrix. Should mathematically match tangent_laplacian."
        ),
        MethodSpec(
            name=f"{descriptor}_covariance_euclidean_dist_laplacian",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b,
                lambda df: Riemann.distance_matrix(
                    df=df, descriptor=descriptor, distance_type="euclidean", pca=False),
                "laplacian", **kw
            ),
            notes="CONTROL: flat Frobenius distance on the SAME OAS covariance ('euclid'). "
                "Curvature ablation vs riemann_logeuclidean_dist_laplacian; gives a kernelised "
                "(non-linear) cross-check of the linear ladder.",
        ),
        MethodSpec(
            name=f"{descriptor}_spd_scalar_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_linear_kernel(
                tr, te, b, lambda df: Riemann.scalar_logeuclidean(df, descriptor=descriptor, pca=False)),
            beta_grid=[None],
            notes="ABLATION (flaw 3): single scalar tr(log C)/D. Collapse floor for the SPD rep.",
        ),
        MethodSpec(
            name=f"{descriptor}_spd_scalar_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_rbf_tuned(
                tr, te, b, lambda df: Riemann.scalar_logeuclidean(df, descriptor=descriptor, pca=False)),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="Scalar collapse floor under tuned RBF.",
        ),
        MethodSpec(
            name=f"{descriptor}_spd_diagonal_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_rbf_tuned(
                tr, te, b, lambda df: Riemann.diagonal_logeuclidean(df, descriptor=descriptor, pca=False)),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="Diagonal block under tuned RBF.",
        ),
        MethodSpec(
            name=f"{descriptor}_spd_diagonal_linear",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_linear_kernel(
                tr, te, b, lambda df: Riemann.diagonal_logeuclidean(df, descriptor=descriptor, pca=False)),
            beta_grid=[None],
            notes="ABLATION (flaw 3): diag(log C) only, off-diagonals removed. "
                "R2(riemann_tangent_linear) - R2(this) = pure cross-channel covariance contribution.",
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
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b,
                lambda df: Grassmann.distance_matrix(
                    df=df, descriptor=descriptor, k=grassmann_k, distance_type="chordal"),
                "laplacian", **kw
            ),
            notes="FLAT rung: extrinsic chordal distance ||sin Theta||_2 — straight-line distance of "
                "the projectors P=UU^T in Euclidean matrix space. Grassmann analog of the Frobenius "
                "(covariance_euclidean) SPD rung. Flat-vs-curved partner of grassmann_geodesic.",
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_geodesic_dist_laplacian",   # renamed for parity
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b,
                lambda df: Grassmann.distance_matrix(
                    df=df, descriptor=descriptor, k=grassmann_k, distance_type="geodesic"),
                "laplacian", **kw
            ),
            notes="CURVED rung: intrinsic geodesic ||Theta||_2 (principal-angle arc length) on G(k,D). "
                "Identical subspaces and kernel to the chordal rung; only the metric curvature differs. "
                "R2(this) - R2(chordal) = pure Grassmann curvature gain.",
        ),
        MethodSpec(
            name=f"{descriptor}_avg_laplacian",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_kernel(tr, te, b, column=f"{descriptor}_embedding", kernel_type="laplacian"),
        ),
        MethodSpec(
            name=f"{descriptor}_avg_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_vector_rbf_tuned(tr, te, b, column=f"{descriptor}_embedding"),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="STEELMAN baseline: the universal nonlinear function of the first moment. Any geometry "
                "claim must beat THIS, not avg_linear.",
        ),
        MethodSpec(
            name=f"{descriptor}_covariance_flat_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_rbf_tuned(
                tr, te, b, lambda df: Riemann.flat_vectorized_spd_matrices(df, descriptor=descriptor, pca=False)),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="Flat covariance, tuned RBF. Pairs with avg_rbf (Δ_info) and tangent_rbf (Δ_curvature).",
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_tangent_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_rbf_tuned(
                tr, te, b, lambda df: Riemann.vectorized_spd_matrices(df, descriptor=descriptor, pca=False)),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="Log-Euclidean tangent, tuned RBF. Curvature claim must survive HERE, not only linear.",
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_projection_rbf",
            kind="vector",
            builder=lambda tr, te, b: _build_projection_rbf_tuned(
                tr, te, b, lambda df: Grassmann.get_projection_features(
                    df, descriptor=descriptor, k=grassmann_k, vectorization_type='isometric')),
            beta_grid=_RBF_LS_MULTIPLIERS,
            notes="Grassmann chordal embedding, tuned RBF.",
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
            builder=lambda tr, te, b, **kw: _build_direct_kernel(
                tr, te, 
                kernel_matrix_fn=lambda df: REMatch.kernel_matrix(df=df, descriptor=descriptor, alpha=rematch_alpha),
                **kw
            ),
            beta_grid=[None], 
            notes="REMatch Entropic OT Kernel"
        ),
        MethodSpec(
            name=f"{descriptor}_wasserstein_w1",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: Wasserstein.distance_matrix(df, descriptor=descriptor, metric="euclidean"), "laplacian", **kw
            ),
            notes="Earth Mover's Distance (W1) with standard Euclidean ground cost."
        ),
        MethodSpec(
            name=f"{descriptor}_wasserstein_w2",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: np.sqrt(Wasserstein.distance_matrix(df, descriptor=descriptor, metric="sqeuclidean")), "laplacian", **kw
            ),
            notes="Wasserstein-2 (W2) distance requiring sqrt of the sqeuclidean EMD cost."
        ),
        MethodSpec(
            name=f"{descriptor}_grassmann_geodesic",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: Grassmann.distance_matrix(df=df, descriptor=descriptor, k=grassmann_k, distance_type="geodesic"), "laplacian", **kw
            ),
        ),
        MethodSpec(
            name=f"{descriptor}_riemann_affine_invariant",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="affine-invariant", pca=False), "laplacian", **kw
            ),
        ),

        # --- Persistent Homology (Topological Space Elements) ---
        MethodSpec(
            name=f"{descriptor}_ph_coordinates_bottleneck",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: PersistentHomology.distance_matrix(df=df, descriptor="coordinates", metric="bottleneck"), "laplacian", **kw
            ),
            notes="Persistent Homology over raw Cartesian coordinates using Bottleneck metric."
        ),
        MethodSpec(
            name=f"{descriptor}_ph_coordinates_sliced_wasserstein",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: PersistentHomology.distance_matrix(df=df, descriptor="coordinates", metric="sliced-wasserstein"), "laplacian", **kw
            ),
            notes="Persistent Homology over raw Cartesian coordinates using Sliced-Wasserstein metric."
        ),
        MethodSpec(
            name=f"{descriptor}_ph_descriptor_space_bottleneck",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: PersistentHomology.distance_matrix(df=df, descriptor=descriptor, metric="bottleneck"), "laplacian", **kw
            ),
            notes="Persistent Homology over unpooled descriptor space matrix distributions using Bottleneck metric."
        ),
        MethodSpec(
            name=f"{descriptor}_ph_descriptor_space_sliced_wasserstein",
            kind="distance",
            builder=lambda tr, te, b, **kw: _build_distance_kernel(
                tr, te, b, lambda df: PersistentHomology.distance_matrix(df=df, descriptor=descriptor, metric="sliced-wasserstein"), "laplacian", **kw
            ),
            notes="Persistent Homology over unpooled descriptor space matrix distributions using Sliced-Wasserstein metric."
        ),
    ]

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
    cv: int, random_state: int,
    seed: Optional[int] = None,
    cache: Optional[DistanceMatrixCache] = None,
) -> Dict[str, Any]:
    
    logger.info(f"Evaluating regression method: {method.name}")
    method_started = time.perf_counter()
    candidate_betas = list(method.beta_grid) if method.beta_grid is not None else [None]
    
    best: Optional[Dict[str, Any]] = None

    for beta in candidate_betas:
        try:
            if method.kind in ("distance", "kernel"):
                K_train, K_test = method.builder(train_df, test_df, beta, seed=seed, cache=cache, method_name=method.name)
            else:
                K_train, K_test = method.builder(train_df, test_df, beta)
            K_train = np.asarray(K_train, dtype=np.float64)
            K_train = (K_train + K_train.T) / 2.0
            K_test = np.asarray(K_test, dtype=np.float64)

            for alpha in alpha_grid:
                cv_mse = _cv_score_precomputed_krr(K_train, y_train, float(alpha), cv, random_state)
                if best is None or cv_mse < best["cv_mse"]:
                    best = {
                        "alpha": float(alpha), "beta": None if beta is None else float(beta),
                        "cv_mse": cv_mse, "K_train": K_train, "K_test": K_test
                    }
        except Exception as e:
            logger.error(f"Error computing kernel for '{method.name}' with beta={beta}: {e}")
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
    cache: Optional[DistanceMatrixCache] = None,
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
            res = _fit_one_method(
                spec, train_df, test_df, y_train, y_test, alpha_grid, cv, random_state,
                seed=random_state, cache=cache
            )
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

    n = 2000
    descriptores = ["soap"]
    qm9 = QM9Dataset(limit=n, descriptors=descriptores)
    df = qm9.load()
    df = df.filter((pl.col("geometric_strain") >= 0) & (pl.col("geometric_strain").is_finite()))
    n = df.height

    output_dir = "results/qm9/regression"
    os.makedirs(output_dir, exist_ok=True)
    
    USE_SAVED_MATRICES = True
    CACHE_DIR_PATH = ".cache/distance_matrices"
    
    SEEDS = [42, 123, 456]
    all_targets = ["gap", "mu", "cv", "geometric_strain", "u0", "A", "B", "C"]
    target_g = ["geometric_strain", "gap", "mu"]
    lim = f"n_{n}"
    path = os.path.join(output_dir, lim)
    os.makedirs(path, exist_ok=True)

    logger.info(f"Starting regression benchmarks on QM9 with {n} samples. Results will be saved to {path}.")

    for target in target_g:
        logger.info(f"=== Starting benchmark for target: {target.upper()} ===")
        for desc in descriptores:
            summary_path = os.path.join(path, f"results_summary_{target}_{desc}.csv")
            if os.path.exists(summary_path):
                logger.info(f" -> Summary already exists at {summary_path}, skipping...")
                continue
            logger.info(f"--- Evaluating descriptor: {desc.upper()} ---")
            
            cache = DistanceMatrixCache(CACHE_DIR_PATH) if USE_SAVED_MATRICES else None
            all_results = []
            all_artifacts = {}
            
            for seed in SEEDS:
                logger.info(f"--- Running Seed: {seed} ---")
                
                comparison = compare_non_euclidean_regression(
                    df, descriptor=desc, target_col=target,
                    random_state=seed, cache=cache
                )
                
                res_df = comparison["results"].with_columns(pl.lit(seed).alias("seed"))
                all_results.append(res_df)
                
                all_artifacts[seed] = {
                    "models": comparison["models"],
                    "predictions": comparison["predictions"],
                    "y_test": comparison["y_test"],
                    "train_indices": comparison["train_indices"],
                    "test_indices": comparison["test_indices"]
                }
            
            full_results_df = pl.concat(all_results)
            ok_runs = full_results_df.filter(pl.col("status") == "ok")
            
            summary_df = ok_runs.group_by(["method", "kind"]).agg([
                pl.col("test_rmse").mean().alias("test_rmse_mean"),
                pl.col("test_rmse").std().alias("test_rmse_std"),
                pl.col("test_mae").mean().alias("test_mae_mean"),
                pl.col("test_mae").std().alias("test_mae_std"),
                pl.col("test_r2").mean().alias("test_r2_mean"),
                pl.col("test_r2").std().alias("test_r2_std"),
                pl.col("total_seconds").mean().alias("time_mean_sec")
            ]).sort("test_rmse_mean", nulls_last=True)
            
            full_results_df.write_csv(os.path.join(path, f"results_raw_{target}_{desc}.csv"))
            summary_df.write_csv(os.path.join(path, f"results_summary_{target}_{desc}.csv"))