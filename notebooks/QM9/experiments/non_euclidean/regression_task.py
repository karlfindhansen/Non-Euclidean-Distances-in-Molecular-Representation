from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar
import warnings

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.non_euclidean import (  # noqa: E402
    Grassmann,
    PersistentHomology,
    REMatch,
    Riemann,
    Wasserstein,
)


ArrayPair = Tuple[np.ndarray, np.ndarray]
KernelBuilder = Callable[[pl.DataFrame, pl.DataFrame, Optional[float]], ArrayPair]
T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    builder: KernelBuilder
    beta_grid: Optional[Sequence[float]] = None
    enabled: bool = True
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _effective_n_jobs(n_jobs: Optional[int], n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0.")

    cpu_count = os.cpu_count() or 1
    if n_jobs < 0:
        resolved = cpu_count + 1 + n_jobs
    else:
        resolved = n_jobs
    return max(1, min(int(resolved), int(n_tasks)))


def _parallel_map_ordered(
    items: Sequence[T],
    fn: Callable[[T], U],
    n_jobs: Optional[int],
) -> List[U]:
    workers = _effective_n_jobs(n_jobs, len(items))
    if workers == 1:
        return [fn(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        return [future.result() for future in futures]


def _as_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    try:
        return pl.from_pandas(df)
    except Exception as e:
        raise TypeError("df must be a Polars DataFrame or pandas-convertible dataframe.") from e


def _take_rows(df: pl.DataFrame, indices: Sequence[int]) -> pl.DataFrame:
    return df[np.asarray(indices, dtype=np.int64).tolist()]


def _clean_regression_df(
    df: Any,
    target_col: str,
    descriptor: str = "soap",
) -> pl.DataFrame:
    df = _as_polars(df)
    required_cols = {target_col, f"{descriptor}_embedding", f"{descriptor}_matrix"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Run QM9Dataset(..., descriptors=['{descriptor}']) "
            f"or qm9.add_{descriptor}() before calling this function."
        )

    cleaned = df.drop_nulls(list(required_cols))
    cleaned = cleaned.filter(pl.col(target_col).is_not_nan())
    if cleaned.height < 10:
        raise ValueError(
            f"Need at least 10 valid rows after dropping null {descriptor} descriptors and target values; "
            f"got {cleaned.height}."
        )
    return cleaned


def _target_array(df: pl.DataFrame, target_col: str) -> np.ndarray:
    y = np.asarray(df[target_col].to_list(), dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError(f"Target column '{target_col}' contains non-finite values.")
    return y


def _pooled_descriptor_matrix(df: pl.DataFrame, column: str = "soap_embedding") -> np.ndarray:
    rows = []
    for idx, value in enumerate(df[column].to_list()):
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid pooled descriptor at row {idx} in column '{column}'.")
        rows.append(arr)
    return np.vstack(rows)


def _sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return D
    finite = np.isfinite(D)
    if not finite.all():
        replacement = np.nanmax(D[finite]) if finite.any() else 0.0
        D = np.nan_to_num(D, nan=replacement, posinf=replacement, neginf=0.0)
    D = np.maximum(D, 0.0)
    return D


def _laplacian_kernel_from_distance(D: np.ndarray, beta: float) -> np.ndarray:
    D = _sanitize_distance_matrix(D)
    K = np.exp(-float(beta) * D)
    return np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)


def _rbf_kernel_from_distance(D: np.ndarray, beta: float) -> np.ndarray:
    """Computes Gaussian / Radial Basis Function kernel mapping: exp(-beta * D^2)"""
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
    if squared:
        return 1.0 / float(median_val ** 2) if median_val > 0 else 1.0
    return 1.0 / float(median_val)


def _vector_kernels(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    beta: Optional[float],
    column: str = "soap_embedding",
    kernel_type: str = "laplacian",
) -> ArrayPair:
    X_train = _pooled_descriptor_matrix(train_df, column=column)
    X_test = _pooled_descriptor_matrix(test_df, column=column)
    D_train = cdist(X_train, X_train, metric="euclidean")
    D_test = cdist(X_test, X_train, metric="euclidean")
    
    is_squared = (kernel_type.lower() == "rbf")
    beta = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
    
    if is_squared:
        return (
            _rbf_kernel_from_distance(D_train, beta),
            _rbf_kernel_from_distance(D_test, beta),
        )
    return (
        _laplacian_kernel_from_distance(D_train, beta),
        _laplacian_kernel_from_distance(D_test, beta),
    )


def _train_test_concat(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([train_df, test_df], how="vertical")


def _slice_train_test_from_square(matrix: np.ndarray, n_train: int) -> ArrayPair:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[:n_train, :n_train], matrix[n_train:, :n_train]


def _cached_source_distance_kernel_builder(
    distance_matrix_fn: Callable[[pl.DataFrame], np.ndarray],
    kernel_type: str = "laplacian"
) -> KernelBuilder:
    cache: Dict[Tuple[int, int], ArrayPair] = {}

    def _builder(train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float]) -> ArrayPair:
        cache_key = (id(train_df), id(test_df))
        if cache_key not in cache:
            D_full = np.asarray(distance_matrix_fn(_train_test_concat(train_df, test_df)), dtype=np.float64)
            D_train, D_test = _slice_train_test_from_square(D_full, train_df.height)
            cache[cache_key] = (_sanitize_distance_matrix(D_train), _sanitize_distance_matrix(D_test))

        D_train, D_test = cache[cache_key]
        is_squared = (kernel_type.lower() == "rbf")
        beta_value = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
        
        if is_squared:
            return (
                _rbf_kernel_from_distance(D_train, beta_value),
                _rbf_kernel_from_distance(D_test, beta_value),
            )
        return (
            _laplacian_kernel_from_distance(D_train, beta_value),
            _laplacian_kernel_from_distance(D_test, beta_value),
        )

    return _builder


def _cached_source_kernel_builder(
    kernel_matrix_fn: Callable[[pl.DataFrame], Optional[np.ndarray]],
) -> KernelBuilder:
    cache: Dict[Tuple[int, int], ArrayPair] = {}

    def _builder(train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float]) -> ArrayPair:
        cache_key = (id(train_df), id(test_df))
        if cache_key not in cache:
            K_full = kernel_matrix_fn(_train_test_concat(train_df, test_df))
            if K_full is None:
                raise ValueError("Source kernel function returned None.")
            K_train, K_test = _slice_train_test_from_square(np.asarray(K_full, dtype=np.float64), train_df.height)
            cache[cache_key] = (np.nan_to_num(K_train), np.nan_to_num(K_test))
        return cache[cache_key]

    return _builder


def _cached_riemann_tangent_kernel_builder(
    descriptor: str = "soap", 
    pca: bool = False,
    kernel_type: str = "laplacian"
) -> KernelBuilder:
    cache: Dict[Tuple[int, int], ArrayPair] = {}

    def _builder(train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float]) -> ArrayPair:
        cache_key = (id(train_df), id(test_df))
        if cache_key not in cache:
            X_full = Riemann.vectorized_spd_matrices(
                _train_test_concat(train_df, test_df),
                descriptor=descriptor,
                pca=pca,
            )
            X_train = X_full[: train_df.height]
            X_test = X_full[train_df.height:]
            D_train = cdist(X_train, X_train, metric="euclidean")
            D_test = cdist(X_test, X_train, metric="euclidean")
            cache[cache_key] = (_sanitize_distance_matrix(D_train), _sanitize_distance_matrix(D_test))

        D_train, D_test = cache[cache_key]
        is_squared = (kernel_type.lower() == "rbf")
        beta_value = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
        
        if is_squared:
            return (
                _rbf_kernel_from_distance(D_train, beta_value),
                _rbf_kernel_from_distance(D_test, beta_value),
            )
        return (
            _laplacian_kernel_from_distance(D_train, beta_value),
            _laplacian_kernel_from_distance(D_test, beta_value),
        )

    return _builder


def _cached_grassmann_projection_kernel_builder(
    descriptor: str = "soap", 
    k: int = 3,
    kernel_type: str = "laplacian"
) -> KernelBuilder:
    cache: Dict[Tuple[int, int], ArrayPair] = {}

    def _builder(train_df: pl.DataFrame, test_df: pl.DataFrame, beta: Optional[float]) -> ArrayPair:
        cache_key = (id(train_df), id(test_df))
        if cache_key not in cache:
            X_full = Grassmann.get_projection_features(
                _train_test_concat(train_df, test_df),
                descriptor=descriptor,
                k=k,
                vectorization_type='isometric'
            )
            X_train = X_full[: train_df.height]
            X_test = X_full[train_df.height:]
            D_train = cdist(X_train, X_train, metric="euclidean")
            D_test = cdist(X_test, X_train, metric="euclidean")
            cache[cache_key] = (_sanitize_distance_matrix(D_train), _sanitize_distance_matrix(D_test))

        D_train, D_test = cache[cache_key]
        is_squared = (kernel_type.lower() == "rbf")
        beta_value = _median_beta_from_distances(D_train, squared=is_squared) if beta is None else float(beta)
        
        if is_squared:
            return (
                _rbf_kernel_from_distance(D_train, beta_value),
                _rbf_kernel_from_distance(D_test, beta_value),
            )
        return (
            _laplacian_kernel_from_distance(D_train, beta_value),
            _laplacian_kernel_from_distance(D_test, beta_value),
        )

    return _builder


def default_soap_regression_methods(
    descriptor: str = "soap",
    grassmann_k: int = 3,
    rematch_alpha: float = 0.1,
    include_persistent_homology: bool = False,
) -> List[MethodSpec]:
    return [
        MethodSpec(
            name="avg_soap_laplacian",
            kind="vector",
            builder=lambda train, test, beta: _vector_kernels(
                train, test, beta, column=f"{descriptor}_embedding", kernel_type="laplacian"
            ),
            notes="Averaged SOAP descriptor with Laplacian kernel.",
        ),
        MethodSpec(
            name="avg_soap_rbf",
            kind="vector",
            builder=lambda train, test, beta: _vector_kernels(
                train, test, beta, column=f"{descriptor}_embedding", kernel_type="rbf"
            ),
            notes="Averaged SOAP descriptor with Gaussian RBF kernel.",
        ),
        MethodSpec(
            name="wasserstein_w1",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Wasserstein.distance_matrix(df, descriptor=descriptor, metric="euclidean"),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="wasserstein_w2",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Wasserstein.distance_matrix(df, descriptor=descriptor, metric="sqeuclidean"),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="rematch_direct",
            kind="kernel",
            builder=_cached_source_kernel_builder(
                lambda df: REMatch.kernel_matrix(df=df, descriptor=descriptor, alpha=rematch_alpha)
            ),
            beta_grid=[None],
        ),
        MethodSpec(
            name="grassmann_geodesic",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Grassmann.distance_matrix(df=df, descriptor=descriptor, k=grassmann_k, distance_type="geodesic"),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="grassmann_chordal",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Grassmann.distance_matrix(df=df, descriptor=descriptor, k=grassmann_k, distance_type="chordal"),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="grassmann_projection_laplacian",
            kind="vector",
            builder=_cached_grassmann_projection_kernel_builder(descriptor=descriptor, k=grassmann_k, kernel_type="laplacian"),
            notes="Isometric projector vectorization mapping using Laplacian metric decay.",
        ),
        MethodSpec(
            name="grassmann_projection_rbf",
            kind="vector",
            builder=_cached_grassmann_projection_kernel_builder(descriptor=descriptor, k=grassmann_k, kernel_type="rbf"),
            notes="Isometric projector vectorization mapping using Gaussian/RBF metric decay.",
        ),
        MethodSpec(
            name="riemann_affine_invariant",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="affine-invariant", pca=False),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="riemann_log_euclidean",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: Riemann.distance_matrix(df=df, descriptor=descriptor, distance_type="log-euclidean", pca=False),
                kernel_type="laplacian"
            ),
        ),
        MethodSpec(
            name="riemann_tangent_laplacian",
            kind="vector",
            builder=_cached_riemann_tangent_kernel_builder(descriptor=descriptor, pca=False, kernel_type="laplacian"),
        ),
        MethodSpec(
            name="riemann_tangent_rbf",
            kind="vector",
            builder=_cached_riemann_tangent_kernel_builder(descriptor=descriptor, pca=False, kernel_type="rbf"),
            notes="Log-Euclidean Tangent vectorization mapping using Gaussian/RBF metric decay.",
        ),
        MethodSpec(
            name="persistent_homology_bottleneck",
            kind="distance",
            builder=_cached_source_distance_kernel_builder(
                lambda df: PersistentHomology.distance_matrix(df=df, descriptor=descriptor, metric="bottleneck"),
                kernel_type="laplacian"
            ),
            enabled=include_persistent_homology,
        ),
    ]


def _cv_score_precomputed_krr(
    K_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    cv: int,
    random_state: int,
) -> float:
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
    method: MethodSpec,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: Sequence[float],
    beta_grid: Optional[Sequence[float]],
    cv: int,
    random_state: int,
) -> Dict[str, Any]:
    logger.info(f"Evaluating regression method: {method.name}")
    method_started = time.perf_counter()
    profile_rows: List[Dict[str, Any]] = []

    if method.beta_grid is not None:
        candidate_betas = list(method.beta_grid)
    else:
        candidate_betas = [None]  # Activates auto-median scaling closures inside our custom builders

    kernel_cache: Dict[Any, ArrayPair] = {}
    best: Optional[Dict[str, Any]] = None

    for beta in candidate_betas:
        try:
            if beta not in kernel_cache:
                kernel_started = time.perf_counter()
                K_train, K_test = method.builder(train_df, test_df, beta)
                K_train = np.asarray(K_train, dtype=np.float64)
                K_test = np.asarray(K_test, dtype=np.float64)
                if K_train.shape != (len(y_train), len(y_train)):
                    raise ValueError(f"K_train shape mismatch: {K_train.shape}.")
                if K_test.shape != (len(y_test), len(y_train)):
                    raise ValueError(f"K_test shape mismatch: {K_test.shape}.")
                K_train = (K_train + K_train.T) / 2.0
                kernel_cache[beta] = (K_train, K_test)
                profile_rows.append(
                    {
                        "method": method.name,
                        "stage": "kernel_build",
                        "beta": None if beta is None else float(beta),
                        "alpha": None,
                        "seconds": time.perf_counter() - kernel_started,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }
                )
            else:
                K_train, K_test = kernel_cache[beta]

            for alpha in alpha_grid:
                cv_started = time.perf_counter()
                cv_mse = _cv_score_precomputed_krr(
                    K_train=K_train,
                    y_train=y_train,
                    alpha=float(alpha),
                    cv=cv,
                    random_state=random_state,
                )
                profile_rows.append(
                    {
                        "method": method.name,
                        "stage": "cross_validation",
                        "beta": None if beta is None else float(beta),
                        "alpha": float(alpha),
                        "seconds": time.perf_counter() - cv_started,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }
                )
                if best is None or cv_mse < best["cv_mse"]:
                    best = {
                        "alpha": float(alpha),
                        "beta": None if beta is None else float(beta),
                        "cv_mse": cv_mse,
                        "K_train": K_train,
                        "K_test": K_test,
                    }
        except Exception as e:
            warnings.warn(f"Skipping beta={beta} for method '{method.name}': {e}")
            continue

    if best is None:
        raise RuntimeError(f"No valid hyperparameter setting found for method '{method.name}'.")

    model = KernelRidge(alpha=best["alpha"], kernel="precomputed")
    final_fit_started = time.perf_counter()
    model.fit(best["K_train"], y_train)
    profile_rows.append(
        {
            "method": method.name,
            "stage": "final_fit",
            "beta": best["beta"],
            "alpha": best["alpha"],
            "seconds": time.perf_counter() - final_fit_started,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
    )
    predict_started = time.perf_counter()
    pred_train = model.predict(best["K_train"])
    pred_test = model.predict(best["K_test"])
    profile_rows.append(
        {
            "method": method.name,
            "stage": "final_predict",
            "beta": best["beta"],
            "alpha": best["alpha"],
            "seconds": time.perf_counter() - predict_started,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
    )
    total_seconds = time.perf_counter() - method_started
    profile_rows.append(
        {
            "method": method.name,
            "stage": "method_total",
            "beta": best["beta"],
            "alpha": best["alpha"],
            "seconds": total_seconds,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
    )

    return {
        "method": method.name,
        "kind": method.kind,
        "best_alpha": best["alpha"],
        "best_beta": best["beta"],
        "cv_rmse": float(np.sqrt(best["cv_mse"])),
        "total_seconds": float(total_seconds),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "notes": method.notes,
        "model": model,
        "y_test_pred": pred_test,
        "profile_rows": profile_rows,
    }


def compare_soap_non_euclidean_regression(
    df: Any,
    target_col: str = "gap",
    descriptor: str = "soap",
    methods: Optional[Sequence[MethodSpec]] = None,
    alpha_grid: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0),
    beta_grid: Optional[Sequence[float]] = None,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 40,
    grassmann_k: int = 3,
    rematch_alpha: float = 0.1,
    include_persistent_homology: bool = False,
) -> Dict[str, Any]:
    """Compare averaged SOAP against non-Euclidean representations with both Laplacian and Gaussian RBF options."""
    cleaned = _clean_regression_df(df, target_col=target_col, descriptor=descriptor)
    y = _target_array(cleaned, target_col=target_col)
    indices = np.arange(cleaned.height)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = _take_rows(cleaned, train_idx)
    test_df = _take_rows(cleaned, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    specs = list(methods) if methods is not None else default_soap_regression_methods(
        descriptor=descriptor,
        grassmann_k=grassmann_k,
        rematch_alpha=rematch_alpha,
        include_persistent_homology=include_persistent_homology,
    )
    specs = [spec for spec in specs if spec.enabled]

    if cv < 2 or cv > len(y_train):
        raise ValueError(f"cv must be between 2 and n_train={len(y_train)}; got {cv}.")

    rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for spec in specs:
        try:
            result = _fit_one_method(
                method=spec,
                train_df=train_df,
                test_df=test_df,
                y_train=y_train,
                y_test=y_test,
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                cv=cv,
                random_state=random_state,
            )
        except Exception as e:
            warnings.warn(f"Method '{spec.name}' failed and was skipped: {e}")
            rows.append(
                {
                    "method": spec.name,
                    "kind": spec.kind,
                    "status": "failed",
                    "error": str(e),
                }
            )
            continue

        fitted[spec.name] = result.pop("model")
        predictions[spec.name] = result.pop("y_test_pred")
        profile_rows.extend(result.pop("profile_rows"))
        rows.append({"status": "ok", **result})

    results = pl.DataFrame(rows)
    if "test_rmse" in results.columns:
        results = results.sort("test_rmse", nulls_last=True)

    profile = pl.DataFrame(profile_rows)
    if not profile.is_empty():
        profile = profile.sort(["method", "stage", "beta", "alpha"], nulls_last=True)

    return {
        "results": results,
        "profile": profile,
        "models": fitted,
        "predictions": predictions,
        "y_test": y_test,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "methods": specs,
    }


if __name__ == '__main__':
    import os
    from src.datasets import QM9Dataset

    n = 100
    qm9 = QM9Dataset(
        limit=n,
        descriptors=["soap"],
    )
    df = qm9.load()
    
    comparison = compare_soap_non_euclidean_regression(df, target_col="gap")
    
    output_dir = "results/qm9/regression"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison["results"].write_csv(os.path.join(output_dir, f"results_{n}.csv"))
    comparison["profile"].write_csv(os.path.join(output_dir, f"profile_{n}.csv"))

    import pickle
    artifacts = {
        "models": comparison["models"],
        "predictions": comparison["predictions"],
        "y_test": comparison["y_test"],
        "train_indices": comparison["train_indices"],
        "test_indices": comparison["test_indices"]
    }
    with open(os.path.join(output_dir, f"artifacts_{n}.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    total_seconds = comparison["profile"]["seconds"].sum()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Successfully saved all outputs to: {output_dir}")