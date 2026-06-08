from typing import Dict, List, Sequence, Optional, Any
import warnings

import numpy as np
import ot
import persim
import polars as pl

from ase import Atoms
from dscribe.kernels import (
    REMatchKernel as DScribeREMatchKernel,
    AverageKernel as DScribeAverageKernel,
)
from loguru import logger
from pyriemann.utils.distance import pairwise_distance
from ripser import ripser
from sklearn.covariance import oas
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import cdist
from tqdm import tqdm


def _descriptor_matrix_column(descriptor: str) -> str:
    key = str(descriptor).strip().lower()
    mapping = {
        "soap": "soap_matrix",
        "soap_matrix": "soap_matrix",
        "acsf": "acsf_matrix",
        "acsf_matrix": "acsf_matrix",
        "mace": "mace_matrix",
        "mace_matrix": "mace_matrix",
    }
    column = mapping.get(key)
    if column is None:
        raise ValueError(
            f"Unknown descriptor '{descriptor}'. Expected one of: soap, acsf, mace."
        )
    return column


def _feature_matrices_from_df(
    df: Any,
    descriptor: str,
) -> List[np.ndarray]:
    """
    Extract per-structure atom-wise descriptor matrices from a dataframe column such as
    `soap_matrix`, `acsf_matrix`, or `mace_matrix`.
    """
    if df is None:
        raise ValueError("A dataframe must be provided.")

    column_name = _descriptor_matrix_column(descriptor)
    logger.info(f"Using column: {column_name} from df")

    if isinstance(df, pl.DataFrame):
        if column_name not in df.columns:
            raise ValueError(
                f"Dataframe is missing required descriptor column '{column_name}'."
            )
        values = df[column_name].to_list()
    else:
        try:
            values = df[column_name].to_list()
        except Exception as e:
            raise ValueError(
                f"Could not extract descriptor column '{column_name}' from dataframe-like input."
            ) from e

    matrices: List[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64) if value is not None else np.empty((0, 0), dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        matrices.append(arr)
    return matrices


class REMatch:
    """
    Computes DScribe REMatch kernel distances between atom-wise descriptor matrices.
    """

    @classmethod
    def _clean_descriptor_matrices(
        cls,
        raw_matrices: Sequence[np.ndarray],
        tol: float = 1e-3,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        cleaned: List[np.ndarray] = []
        normalized_count = 0

        for idx, matrix in enumerate(raw_matrices):
            X = np.asarray(matrix, dtype=np.float64)

            if X.ndim != 2:
                raise ValueError(f"Descriptor matrix {idx} is not 2D: shape={X.shape}")
            if X.size == 0:
                raise ValueError(f"Descriptor matrix {idx} is empty.")
            if not np.isfinite(X).all():
                raise ValueError(f"Non-finite values in descriptor matrix {idx}.")

            if normalize:
                norms = np.linalg.norm(X, axis=1)
                already_normalized = np.all(np.abs(norms - 1.0) < tol)

                if already_normalized:
                    X_norm = X
                    normalized_count += 1
                else:
                    X_norm = X / (norms[:, None] + 1e-12)
            else:
                X_norm = X

            if not np.isfinite(X_norm).all():
                raise ValueError(f"NaN/inf after normalization in descriptor matrix {idx}.")

            cleaned.append(X_norm)

        if normalize and normalized_count == len(cleaned):
            logger.info("All descriptor matrices are already normalized.")
        elif normalize and normalized_count > 0:
            logger.info(
                f"{normalized_count}/{len(cleaned)} descriptor matrices were already normalized."
            )

        return cleaned

    @classmethod
    def kernel_matrix(
        cls,
        df: Any,
        descriptor: str = "soap",
        metric: str = "linear",
        alpha: float = 0.1,
        tol: float = 1e-3,
        normalize: bool = True,
        threshold: Optional[float] = 10e-12,
        gamma: Optional[float] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Computes the regularized REMatch transport kernel matrix directly.

        Returns None if DScribe's REMatchKernel produces non-finite values.
        """
        raw_matrices = _feature_matrices_from_df(df, descriptor)
        if not raw_matrices:
            return np.array([])

        cleaned = cls._clean_descriptor_matrices(
            raw_matrices=raw_matrices,
            tol=tol,
            normalize=normalize,
        )

        kernel_kwargs = {
            "metric": metric,
            "alpha": alpha,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "kernel_params": kernel_params,
        }
        if threshold is not None:
            kernel_kwargs["threshold"] = threshold

        logger.info(
            f"Computing REMatch regularized transport kernel | Features: {descriptor} | "
            f"Metric: {metric} | alpha: {alpha}"
        )

        kernel = DScribeREMatchKernel(**kernel_kwargs)
        K = kernel.create(cleaned)

        if not np.isfinite(K).all():
            logger.warning("DScribe REMatch returned NaN/inf. Aborting calculation.")
            return None

        return K

    @classmethod
    def distance_matrix(
        cls,
        df: Any,
        descriptor: str = "soap",
        metric: str = "linear",
        alpha: float = 0.1,
        tol: float = 1e-3,
        normalize: bool = True,
        threshold: Optional[float] = 10e-12,
        gamma: Optional[float] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Computes a REMatch kernel distance matrix from atom-wise descriptor matrices.

        Uses DScribe's REMatchKernel and converts the resulting similarity kernel K
        to distances via sqrt(K_ii + K_jj - 2 K_ij). Returns None if the kernel
        computation fails or produces non-finite values.
        """
        K = cls.kernel_matrix(
            df=df,
            descriptor=descriptor,
            metric=metric,
            alpha=alpha,
            tol=tol,
            normalize=normalize,
            threshold=threshold,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
        )

        if K is None:
            return None

        diag = np.diag(K)
        dist_sq = diag[:, None] + diag[None, :] - 2.0 * K
        dist_sq = np.clip(dist_sq, 0.0, None)
        dist_matrix = np.sqrt(dist_sq)

        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

        return dist_matrix


class Average:
    """
    Computes DScribe Average kernel distances between atom-wise descriptor matrices.
    """

    @classmethod
    def _clean_descriptor_matrices(
        cls,
        raw_matrices: Sequence[np.ndarray],
        tol: float = 1e-3,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        cleaned: List[np.ndarray] = []
        normalized_count = 0

        for idx, matrix in enumerate(raw_matrices):
            X = np.asarray(matrix, dtype=np.float64)

            if X.ndim != 2:
                raise ValueError(f"Descriptor matrix {idx} is not 2D: shape={X.shape}")
            if X.size == 0:
                raise ValueError(f"Descriptor matrix {idx} is empty.")
            if not np.isfinite(X).all():
                raise ValueError(f"Non-finite values in descriptor matrix {idx}.")

            if normalize:
                norms = np.linalg.norm(X, axis=1)
                already_normalized = np.all(np.abs(norms - 1.0) < tol)

                if already_normalized:
                    X_norm = X
                    normalized_count += 1
                else:
                    X_norm = X / (norms[:, None] + 1e-12)
            else:
                X_norm = X

            if not np.isfinite(X_norm).all():
                raise ValueError(f"NaN/inf after normalization in descriptor matrix {idx}.")

            cleaned.append(X_norm)

        if normalize and normalized_count == len(cleaned):
            logger.info("All descriptor matrices are already normalized.")
        elif normalize and normalized_count > 0:
            logger.info(
                f"{normalized_count}/{len(cleaned)} descriptor matrices were already normalized."
            )

        return cleaned

    @classmethod
    def kernel_matrix(
        cls,
        df: Any,
        descriptor: str = "soap",
        metric: str = "linear",
        tol: float = 1e-3,
        normalize: bool = True,
        normalize_kernel: bool = True,
        gamma: Optional[float] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Computes the Average kernel matrix directly.
        """
        raw_matrices = _feature_matrices_from_df(df, descriptor)
        if not raw_matrices:
            return np.array([])

        cleaned = cls._clean_descriptor_matrices(
            raw_matrices=raw_matrices,
            tol=tol,
            normalize=normalize,
        )

        kernel_kwargs = {
            "metric": metric,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "kernel_params": kernel_params,
            "normalize_kernel": normalize_kernel,
        }

        logger.info(
            f"Computing Average kernel | Features: {descriptor} | Metric: {metric}"
        )

        kernel = DScribeAverageKernel(**kernel_kwargs)
        K = kernel.create(cleaned)

        if not np.isfinite(K).all():
            logger.warning("DScribe AverageKernel returned NaN/inf. Aborting calculation.")
            return None

        return K

    @classmethod
    def distance_matrix(
        cls,
        df: Any,
        descriptor: str = "soap",
        metric: str = "linear",
        tol: float = 1e-3,
        normalize: bool = True,
        normalize_kernel: bool = True,
        gamma: Optional[float] = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Computes an Average kernel distance matrix from atom-wise descriptor matrices.
        """
        K = cls.kernel_matrix(
            df=df,
            descriptor=descriptor,
            metric=metric,
            tol=tol,
            normalize=normalize,
            normalize_kernel=normalize_kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
        )

        if K is None:
            return None

        diag = np.diag(K)
        dist_sq = diag[:, None] + diag[None, :] - 2.0 * K
        dist_sq = np.clip(dist_sq, 0.0, None)
        dist_matrix = np.sqrt(dist_sq)

        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

        return dist_matrix


class Wasserstein:
    """
    Computes the Earth Mover's Distance (Wasserstein-1) between molecules.
    Treats each molecule as a uniform distribution of atomic feature vectors.
    """

    @classmethod
    def distance_matrix(
        cls, 
        df: Any,
        descriptor: str = 'soap',
        metric: str = 'sqeuclidean',
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise Wasserstein distance matrix sequentially.
        Optimized to minimize inner-loop Python overhead.
        """
        # 1. Extract raw feature matrices (N_atoms, D_features)
        raw_matrices = _feature_matrices_from_df(df, descriptor)
        
        num_items = len(raw_matrices)
        if num_items == 0:
            return np.array([])
            
        dist_matrix = np.zeros((num_items, num_items))
        logger.info(f"Computing Wasserstein distance matrix sequentially | Features: {descriptor} | Metric: {metric}")

        # 2. PRE-OPTIMIZATION: Pre-compute the uniform mass weights for all molecules
        # Weights: Each atom is 1/N of the molecule's total "mass"
        # Pre-calculating this saves N^2 array allocations inside the loop.
        weights = [np.ones(X.shape[0]) / X.shape[0] for X in raw_matrices]
        
        # 3. Sequential nested loop calculation (Upper triangle only, mirrored to lower)
        for i in tqdm(range(num_items), desc="Wasserstein distances", unit="row"):
            X_i = np.asarray(raw_matrices[i])
            w_i = weights[i]
            
            for j in range(i + 1, num_items):
                X_j = np.asarray(raw_matrices[j])
                w_j = weights[j]

                try:
                    # Compute the Cost Matrix (Distances between all atoms in A and B)
                    # M[a, b] is the cost to move atom 'a' to 'b' in feature space
                    M = ot.dist(X_i, X_j, metric=metric)

                    # Solve the Optimal Transport problem directly using POT's C-backend
                    d = float(ot.emd2(w_i, w_j, M))
                    
                    dist_matrix[i, j] = dist_matrix[j, i] = d
                    
                except Exception as e:
                    logger.warning(f"Distance calculation failed for pair ({i}, {j}): {e}")
                    dist_matrix[i, j] = dist_matrix[j, i] = np.nan

        # 4. Fill failed calculations if any numerical instabilities occurred
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")
            dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix