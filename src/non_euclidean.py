from typing import Dict, List, Sequence, Optional, Any
import warnings

import numpy as np
import ot
import persim
import polars as pl

from ase import Atoms
from dscribe.kernels import REMatchKernel as DScribeREMatchKernel
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
        threshold: Optional[float] = None,
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

        # If DScribe outputs non-finite matrices, drop out early and return None
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
        threshold: Optional[float] = None,
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


class PersistentHomology:
    """
    Computes topological features (persistence diagrams) either from physical 3D coordinates
    or high-dimensional feature spaces (e.g., SOAP matrices treated as point clouds).
    """

    @classmethod
    def _get_persistence_diagrams(
        cls,
        df: Any,
        descriptor: str = 'soap',
        max_homology_dim: int = 2
    ) -> List[Dict[int, np.ndarray]]:
        """
        Maps inputs to persistence diagrams. Dynamically switches between physical coordinate 
        filtration or high-dimensional feature space filtration based on the descriptor parameter.
        """
        diagrams = []
        
        for row in df.iter_rows(named=True):
            
            # --- PATH A: Physical Cartesian Coordinate Space ---
            if descriptor.lower() == 'coordinates':
                nums = row["atomic_numbers"]
                coords = np.array(row["coordinates"], dtype=np.float64)
                
                if len(coords) == 0:
                    diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
                    continue
                    
                frame = Atoms(numbers=nums, positions=coords)
                #is_material = any(row.get("material_id")) 
                dist_mat = frame.get_all_distances(mic=False)
                
            # --- PATH B: High-Dimensional Feature Space (e.g., SOAP) ---
            else:
                # Dynamically match column name syntax
                col_name = "soap_matrix" if descriptor.lower() == "soap" else descriptor
                X = np.asarray(row[col_name], dtype=np.float64)
                
                if len(X) == 0:
                    diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
                    continue
                    
                # Compute distances between atom feature vectors within the single molecule
                dist_mat = cdist(X, X, metric='euclidean')

            # --- Extract Persistence Topology ---
            raw_dgms = ripser(dist_mat, maxdim=max_homology_dim, distance_matrix=True)["dgms"]
            
            # Pre-filter infinite features for inner-loop speed
            formatted_dgm = {}
            for d in range(max_homology_dim + 1):
                dgm_layer = np.asarray(raw_dgms[d])
                if d == 0:
                    finite_mask = np.isfinite(dgm_layer[:,1])
                    dgm_layer = dgm_layer[finite_mask]

                formatted_dgm[d] = dgm_layer
                
            diagrams.append(formatted_dgm)
            
        return diagrams

    @classmethod
    def distance_matrix(
        cls,
        df: Any,
        descriptor: str = 'soap',
        metric: str = "bottleneck",
        max_homology_dim: int = 2,
        homology_dims: Sequence[int] = (0, 1, 2),
        sw_projections: int = 50
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise persistent homology distance matrix.
        Allows explicit selection of the underlying input representation via the 'descriptor' argument.
        """
        metric_key = metric.lower()
        valid_metrics = {"bottleneck", "b", "sliced-wasserstein", "sliced_wasserstein", "sw"}
        if metric_key not in valid_metrics:
            raise ValueError(f"Unknown metric: '{metric}'. Must be one of {valid_metrics}.")

        # Generate persistence representations based on requested input domain
        dgms = cls._get_persistence_diagrams(df=df, descriptor=descriptor, max_homology_dim=max_homology_dim)
        
        num_items = len(dgms)
        dist_matrix = np.zeros((num_items, num_items))
        
        logger.info(f"Computing PH distance matrix | Domain: {descriptor} | Metric: {metric} | Max Dim: {max_homology_dim}")
        
        # Upper-triangular matrix computation loop
        for i in tqdm(range(num_items), desc="Persistence distances", unit="row"):
            dgm_i = dgms[i]
            
            for j in range(i + 1, num_items):
                dgm_j = dgms[j]
                total_dist = 0.0
                
                for d in homology_dims:
                    p1 = dgm_i.get(d, np.empty((0, 2)))
                    p2 = dgm_j.get(d, np.empty((0, 2)))
                    
                    if len(p1) == 0 and len(p2) == 0:
                        continue
                    
                    if metric_key in {"bottleneck", "b"}:
                        total_dist += persim.bottleneck(p1, p2)
                    else:
                        total_dist += persim.sliced_wasserstein(p1, p2, M=sw_projections)
                
                dist_matrix[i, j] = dist_matrix[j, i] = float(total_dist)

        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")
            dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix

import numpy as np
from typing import Any, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Grassmann:
    """
    Handles molecular representation on Grassmann Manifolds G(k, D).
    Represents each molecule as a k-dimensional subspace in R^D (feature space).
    """

    @classmethod
    def _get_uk_bases(
        cls,
        df: Any,
        descriptor: str = 'soap',
        k: int = 3, 
    ) -> List[np.ndarray]:
        """
        Maps 3D atomic coordinates to an orthonormal basis in R^D (feature space) via SVD.
        """
        bases = []
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        for X in raw_matrices:
            X = np.asarray(X, dtype=np.float64)
            
            # SVD on X (N x D): U is (N, M), S is (M,), Vh is (M, D)
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            basis = vh.T[:, :k]
            bases.append(basis)
        
        return bases

    @classmethod
    def get_projection_features(
        cls,
        df: Any,
        descriptor: str = 'soap',
        k: int = 3,
        vectorization_type: str = 'isometric'
    ) -> np.ndarray:
        """
        Computes the orthogonal projector matrix P = U @ U.T for each molecule 
        and maps them into a flat ambient feature matrix X_Grassmann.

        Parameters:
        -----------
        df : Any
            The input dataframe containing molecular structures.
        descriptor : str
            The atomic descriptor type (e.g., 'soap').
        k : int
            The subspace dimension.
        vectorization_type : str
            'flat': Flattens the full matrix to a vector of size D^2.
            'isometric': Extracts the upper triangle and scales off-diagonals by sqrt(2)
                         to perfectly preserve the Chordal distance metric isometry.

        Returns:
        --------
        np.ndarray
            A 2D feature matrix of shape (num_molecules, feature_dim).
        """
        bases = cls._get_uk_bases(df=df, descriptor=descriptor, k=k)
        if not bases:
            return np.array([[]])

        num_items = len(bases)
        D = bases[0].shape[0]  # Ambient descriptor feature dimension
        
        logger.info(f"Extracting Grassmann projection features | Method: {vectorization_type} | D: {D} | k: {k}")

        if vectorization_type == 'flat':
            feature_matrix = np.zeros((num_items, D * D))
            for idx, U in enumerate(tqdm(bases, desc="Projector flat vectorization")):
                P = U @ U.T
                feature_matrix[idx] = P.ravel()
                
        elif vectorization_type == 'isometric':
            tri_len = D * (D + 1) // 2
            feature_matrix = np.zeros((num_items, tri_len))
            
            # Precompute upper triangular indices once
            iu = np.triu_indices(D)
            off_diagonal_mask = (iu[0] != iu[1])
            
            for idx, U in enumerate(tqdm(bases, desc="Projector isometric vectorization")):
                P = U @ U.T
                
                # Extract the upper triangular vector
                vec = P[iu]
                
                # Scale off-diagonal components to preserve Frobenius/Chordal distance metrics
                vec[off_diagonal_mask] *= np.sqrt(2)
                
                feature_matrix[idx] = vec
        else:
            raise ValueError("Unknown vectorization_type. Choose either 'flat' or 'isometric'.")

        return feature_matrix

    @classmethod
    def distance_matrix(
        cls, 
        df: Any,
        descriptor: str = 'soap',
        distance_type: str = "geodesic",
        k: int = 3, 
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise distance matrix on the Grassmann Manifold.
        """
        valid_distances = {"geodesic", "chordal", "projection"}
        distance_type = distance_type.lower()
        if distance_type not in valid_distances:
            raise ValueError(f"Unknown distance_type: '{distance_type}'. Must be one of {valid_distances}.")

        # Generate Subspaces
        bases = cls._get_uk_bases(
            df=df,
            descriptor=descriptor,
            k=k, 
        )
        
        num_items = len(bases)
        dist_matrix = np.zeros((num_items, num_items))
        
        logger.info(f"Computing Grassmann distance matrix | Features: {descriptor} | Distance: {distance_type} | k: {k}")
        
        # Pre-calculate transposes to save N^2 transpose operations
        bases_T = [U.T for U in bases]
        
        # Sequential nested loop calculation
        for i in tqdm(range(num_items), desc="Grassmann distances", unit="row"):
            U1_T = bases_T[i]
            for j in range(i + 1, num_items):
                
                # 1. Inner Product: Yields a tiny (k x k) matrix
                core_matrix = U1_T @ bases[j]
                
                # 2. Pure NumPy SVD: compute_uv=False skips calculating the vectors, only gets singular values
                s = np.linalg.svd(core_matrix, compute_uv=False)
                
                # 3. Math: Singular values of U1^T U2 represent cos(theta).
                # Clip to [0.0, 1.0] to prevent floating point errors from crashing arccos
                angles = np.arccos(np.clip(s, 0.0, 1.0))
                
                # 4. Inline Distance Calculation
                if distance_type == "geodesic":
                    d = float(np.linalg.norm(angles))
                elif distance_type == "chordal":
                    d = float(np.linalg.norm(np.sin(angles)))
                else: # projection
                    d = float(np.max(np.sin(angles)))
                    
                dist_matrix[i, j] = dist_matrix[j, i] = d

        # Fill failed calculations
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")
            dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix

class Riemann:
    """
    Handles molecular representation on the Riemannian Manifold.
    Supports global descriptors by converting them into SPD covariance matrices.
    """

    @classmethod
    def get_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca=True,
    ) -> np.ndarray:
        
        # 1. Obtain raw feature matrices directly from df
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        # 2. PCA Reduction
        if pca:
            n_pca = df['num_atoms'].min() - 2
            raw_matrices = cls.matrix_pca(n_pca, raw_matrices)

        # 3. Build SPD Matrices (Empirical Covariance)
        spd_matrices = []
        for X in raw_matrices:
            X = np.asarray(X)
            C, _ = oas(X, assume_centered=False)
            spd_matrices.append(C)

        for idx, C in enumerate(spd_matrices):
            if not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Matrix at index {idx} failed symmetry validation")
            
            eigvals = np.linalg.eigvalsh(C)
            min_eig = eigvals.min()
            if min_eig <= 0:
                raise ValueError(f"Matrix at index {idx} has eigenvalue lower than 0")

        # Pyriemann expects a 3D array of shape (N_matrices, n_channels, n_channels)
        return np.array(spd_matrices)

    @classmethod
    def log_euclidean_vectorize(
        cls,
        spd_matrices: np.ndarray,
        eig_floor: float = 1e-12,
        warn_threshold: float = 1e-6,
    ) -> np.ndarray:
        """
        Computes Log-Euclidean vectors from a tensor of SPD matrices.

        Off-diagonal entries are weighted by sqrt(2) so Euclidean dot products
        between the flattened upper triangles preserve the Frobenius inner
        product of the symmetric matrix logarithms.
        """
        spd_matrices = np.asarray(spd_matrices, dtype=np.float64)
        if spd_matrices.ndim != 3 or spd_matrices.shape[1] != spd_matrices.shape[2]:
            raise ValueError(
                "spd_matrices must have shape (n_molecules, d, d) with square matrices."
            )

        _, d, _ = spd_matrices.shape
        triu_idx = np.triu_indices(d)
        weight_matrix = np.where(np.eye(d, dtype=bool), 1.0, np.sqrt(2.0))

        vectorized_dataset = []
        min_eigenvalues = []

        for idx, C in enumerate(spd_matrices):
            if not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Matrix at index {idx} failed symmetry validation.")

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            min_eigenvalues.append(float(eigenvalues.min()))

            eigenvalues = np.clip(eigenvalues, a_min=eig_floor, a_max=None)
            log_C = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
            weighted_log_C = log_C * weight_matrix
            vectorized_dataset.append(weighted_log_C[triu_idx])

        global_min_eig = min(min_eigenvalues) if min_eigenvalues else np.nan
        logger.info(
            f"Smallest eigenvalue across SPD dataset: {global_min_eig:.6e}"
        )
        if global_min_eig < warn_threshold:
            logger.warning(
                "Extremely small eigenvalues detected. Verify OAS scaling or centering."
            )
        else:
            logger.info("Minimum eigenvalue looks structurally stable.")

        return np.asarray(vectorized_dataset, dtype=np.float64)

    @classmethod
    def vectorized_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = True,
        eig_floor: float = 1e-12,
        warn_threshold: float = 1e-6,
    ) -> np.ndarray:
        """
        Builds SPD covariance matrices from the dataframe and returns their
        Log-Euclidean vectorized representation.
        """
        spd_matrices = cls.get_spd_matrices(
            df=df,
            descriptor=descriptor,
            pca=pca,
        )
        return cls.log_euclidean_vectorize(
            spd_matrices,
            eig_floor=eig_floor,
            warn_threshold=warn_threshold,
        )

    @classmethod
    def matrix_pca(cls, n_pca, raw_matrices):
        if n_pca is not None:
            logger.info(f"Applying PCA to reduce feature dimension to {n_pca}...")
            
            # Stack all atoms from all molecules into one giant 2D matrix
            stacked_features = np.vstack(raw_matrices)
            pca = PCA(n_components=n_pca)
            stacked_reduced = pca.fit_transform(stacked_features)
            
            # Unstack back into the original list of (N_atoms, n_pca) matrices
            reduced_matrices = []
            current_idx = 0
            for X in raw_matrices:
                n_atoms = X.shape[0]
                reduced_matrices.append(stacked_reduced[current_idx : current_idx + n_atoms, :])
                current_idx += n_atoms

            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f} (cumulative for {n_pca} components)")
                
            raw_matrices = reduced_matrices

        return raw_matrices

    @classmethod
    def distance_matrix(
        cls,
        df: Any,
        descriptor: str = 'soap',
        distance_type: str = "affine-invariant",
        pca : bool = False,
    ) -> np.ndarray:
        
        metric_map = {
            "affine-invariant": "riemann",
            "log-euclidean": "logeuclid",
            "euclidean": "euclid"
        }
        pyriemann_metric = metric_map.get(distance_type.lower())
        if not pyriemann_metric:
            raise ValueError(f"Unknown distance_type: '{distance_type}'. Must be one of {list(metric_map.keys())}.")

        logger.info(f"Computing Riemann distance matrix | Features: {descriptor} | Distance: {distance_type}")

        # 1. Build SPD matrices
        spd_matrices = cls.get_spd_matrices(
            df=df,
            descriptor=descriptor,
            pca=pca,
        )
        
        # 2. Compute Distances
        logger.info(f"Computing {distance_type} distances...")
        dist_matrix = pairwise_distance(spd_matrices, metric=pyriemann_metric)
        np.fill_diagonal(dist_matrix, 0)

        # Fallback for severe numerical instability
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")

        return dist_matrix
