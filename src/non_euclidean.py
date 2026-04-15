from typing import Dict, List, Sequence, Literal, Optional
import signal

import numpy as np
import ot
import persim

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from loguru import logger
from scipy.linalg import logm, eigvalsh
from pymatgen.core import Element
from ripser import ripser
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def _compute_invariant_feature_matrix(frame: Atoms, cutoff: float = 1.8) -> np.ndarray:
    """
    Maps a molecule to a D x N matrix of invariant physical features.
    D is the fixed ambient dimension. N is the number of atoms.
    """
    features = []
    # Center of mass acts as an invariant spatial anchor
    com = frame.get_center_of_mass()

    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)

    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    for i, atom in enumerate(frame):
        z = atom.number
        rad = covalent_radii[z]
        el = Element.from_Z(z)
        en = el.X if el.X else 0.0
        mendeleev = el.mendeleev_no if el.mendeleev_no else 0
        ion_en = el.ionization_energy if el.ionization_energy else 0.0

        mass = atom.mass

        # Geometric invariance: distance to center of mass
        dist_to_com = np.linalg.norm(atom.position - com)

        coord = len(set(neighbors[i]))

        if coord > 0:
            avg_neighbor_z = np.mean(neighbors[i])
            avg_neighbor_dist = np.mean(distances[i])
        else:
            avg_neighbor_z = 0
            avg_neighbor_dist = 0

        feat_vector = [
            coord,
            mendeleev,
            ion_en,
        ]

        features.append(feat_vector)
    #logger.info(f"Computed invariant feature matrix consisting of coordination, mendeleev, and ionization energy for frame with {len(frame)} atoms.")
    return np.array(features).T


def _compute_feature_matrices(
    frames: Sequence[Atoms],
    normalized: bool = True,
) -> List[np.ndarray]:
    """
    Builds invariant feature matrices for all frames.

    If normalized=True, applies a global StandardScaler across all atoms
    in the dataset (fit on the stacked per-atom features).
    """
    raw_matrices = [_compute_invariant_feature_matrix(f) for f in frames]

    if not normalized:
        return raw_matrices

    if not raw_matrices:
        return raw_matrices

    stacked = np.vstack([m.T for m in raw_matrices if m.size > 0]) if raw_matrices else np.empty((0, 0))

    if stacked.size == 0:
        return raw_matrices

    scaler = StandardScaler().fit(stacked)

    scaled_matrices = []
    for raw in raw_matrices:
        if raw.size == 0:
            scaled_matrices.append(raw)
            continue
        scaled_matrices.append(scaler.transform(raw.T).T)

    return scaled_matrices

def _compute_soap_feature_matrices(frames: Sequence[Atoms]):
    """
    Builds invariant soap feature matrices for all frames.
    """
    features = []
    for frame in frames:
        features.append(frame.soap)
    
    return features


def _pairwise_distance_matrix(
    n: int,
    pair_fn,
    desc: str
) -> np.ndarray:
    dist_matrix = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc=desc, unit="pair") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                d = pair_fn(i, j)
                dist_matrix[i, j] = dist_matrix[j, i] = d
                pbar.update(1)

    return dist_matrix

class Wasserstein:
    """
    Computes the Earth Mover's Distance (Wasserstein-1) between molecules.
    Treats each molecule as a distribution of atomic feature vectors.
    """

    @staticmethod
    def compute_feature_distance(feat_i: np.ndarray, feat_j: np.ndarray, metric: str = 'sqeuclidean') -> float:
        """
        Computes EMD between two matrices of shape (D, N_atoms).
        """
        # Transpose to (N_atoms, D) for the POT (ot) library
        pos_i = feat_i.T
        pos_j = feat_j.T

        # 1. Assign weights (Uniform: each atom is 1/N of the molecule's 'mass')
        weights_i = np.ones(pos_i.shape[0]) / pos_i.shape[0]
        weights_j = np.ones(pos_j.shape[0]) / pos_j.shape[0]

        # 2. Compute the Cost Matrix (Distances between all atoms in A and B)
        # M[a, b] is the cost to move atom 'a' to 'b' in feature space
        M = ot.dist(pos_i, pos_j, metric=metric)

        # 3. Solve the Optimal Transport problem
        # We use emd2 to get the scalar distance value
        distance = ot.emd2(weights_i, weights_j, M)
        
        return float(distance)

    @classmethod
    def distance_matrix(
        cls, 
        frames: Optional[Sequence] = None, 
        feature_matrices: Optional[Sequence[np.ndarray]] = None,
        feature_type: str = 'invariant',
        metric: str = 'sqeuclidean'
    ) -> np.ndarray:
        
        # Step 1: Extract the correct feature matrices
        if feature_matrices is not None:
            raw_matrices = feature_matrices
        else:
            if feature_type == 'invariant':
                # Returns list of (3, N_atoms)
                raw_matrices = _compute_feature_matrices(frames, normalized=True)
            elif feature_type == 'soap':
                # If global, returns (1, 252). If atomic, returns (N, 252)
                soap_list = _compute_soap_feature_matrices(frames)
                raw_matrices = [np.atleast_2d(s).T if s.ndim == 1 else s.T for s in soap_list]
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

        n = len(raw_matrices)
        logger.info(f"Computing Wasserstein distance matrix | Features: {feature_type}")

        # Step 2: Pairwise distance calculation
        def pair_fn(i, j):
            return cls.compute_feature_distance(raw_matrices[i], raw_matrices[j], metric=metric)

        dist_matrix = _pairwise_distance_matrix(
            n=n,
            pair_fn=pair_fn,
            desc=f"Wasserstein ({feature_type})",
        )
        
        return dist_matrix


class PersistentHomology:
    """
    Computes topological features (persistence diagrams) from 3D atomic point clouds 
    and evaluates the structural similarities between different frames using 
    Bottleneck or Sliced Wasserstein distances.
    """

    @staticmethod
    def _compute_ripser(distance_matrix: np.ndarray, max_dim: int) -> Dict[int, np.ndarray]:
        """Calculates persistence diagrams up to max_dim from a precomputed distance matrix."""
        # distance_matrix=True is strictly required so ripser doesn't treat the input as a point cloud
        dgms = ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)["dgms"]
        
        # Format output into a dictionary mapping homology dimension to its (birth, death) array
        return {d: np.asarray(dgms[d]) for d in range(max_dim + 1)}

    @classmethod
    def compute_persistence_diagrams(
        cls, frames: Sequence[Atoms], max_homology_dim: int = 2
    ) -> List[Dict[int, np.ndarray]]:
        """Generates persistence diagrams for a sequence of molecular frames."""
        
        logger.info(
            f"Computing persistence diagrams for {len(frames)} frames "
            f"(max_homology_dim={max_homology_dim})."
        )
        
        diagrams = []

        for frame in tqdm(frames, desc="Persistence diagrams", unit="frame"):
            # Handle edge case of an empty simulation frame to prevent Ripser crashes
            if len(frame) == 0:
                diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
                continue
            
            # Use Minimum Image Convention (MIC) to ensure bonds across periodic 
            # cell boundaries are calculated at their true shortest distance
            dist_mat = frame.get_all_distances(mic=True)
            diagrams.append(cls._compute_ripser(dist_mat, max_homology_dim))
 
        logger.success("Finished persistence diagram computation.")
        return diagrams
    
    @staticmethod
    def distance(
        dgm1: Dict[int, np.ndarray],
        dgm2: Dict[int, np.ndarray],
        metric: str = "bottleneck",
        dims: Sequence[int] = (0, 1, 2),
        sw_projections: int = 50
    ) -> float:
        """
        Computes the total topological distance between two diagrams across 
        specified homology dimensions (e.g., 0=components, 1=loops, 2=voids).
        """
        metric_key = metric.lower()
        if metric_key not in {"bottleneck", "b", "sliced-wasserstein", "sliced_wasserstein", "sw"}:
            logger.error(f"Unknown persistence metric '{metric}'.")
            raise ValueError(
                "metric must be one of: ['bottleneck', 'b', 'sliced-wasserstein', 'sliced_wasserstein', 'sw']"
            )

        total_dist = 0.0
        
        for d in dims:
            # Safely fetch the diagrams for dimension `d`, defaulting to empty if missing
            p1, p2 = dgm1.get(d, np.empty((0, 2))), dgm2.get(d, np.empty((0, 2)))

            if len(p1) == 0 and len(p2) == 0:
                continue
            
            # Filter out features with infinite death times (essential classes) 
            # since distance metrics require finite bounds to compute properly
            if len(p1) > 0:
                p1 = p1[np.isfinite(p1[:, 1])]
            if len(p2) > 0:
                p2 = p2[np.isfinite(p2[:, 1])]
            
            # Accumulate the calculated distance for this dimension
            if metric_key in {"bottleneck", "b"}:
                total_dist += persim.bottleneck(p1, p2)
            else:
                total_dist += persim.sliced_wasserstein(p1, p2, M=sw_projections)
                
        return float(total_dist)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        metric: str = "bottleneck",
        max_homology_dim: int = 2,
        homology_dims: Sequence[int] = (0, 1, 2)
    ) -> np.ndarray:
        """
        Generates a symmetric pairwise distance matrix comparing the topological 
        features of all molecular frames in the sequence.
        """
        logger.info(
            f"Computing persistent homology distance matrix for {len(frames)} frames "
            f"(metric='{metric}', max_homology_dim={max_homology_dim}, "
            f"dims={tuple(homology_dims)})."
        )
        
        # Precompute all diagrams
        dgms = cls.compute_persistence_diagrams(frames, max_homology_dim)
        n = len(dgms)
        dist_mat = _pairwise_distance_matrix(
            n=n,
            pair_fn=lambda i, j: cls.distance(dgms[i], dgms[j], metric=metric, dims=homology_dims),
            desc="Persistence distances",
        )
                    
        logger.success("Finished persistent homology distance matrix computation.")
        return dist_mat

class Grassmann:
    """
    Handles molecular representation on Grassmann Manifolds G(k, n).
    Represents each molecule as a k-dimensional subspace in R^n (atom-space).
    """

    @classmethod
    def _get_uk_bases(
        cls,
        frames: Optional[Sequence['Atoms']], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd",
        features : Literal['soap', 'invariant'] = 'invariant',
        normalized: bool = True,
        precomputed_feature_matrices: Optional[Sequence[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Maps 3D atomic coordinates to an orthonormal basis in R^n.
        """
        bases = []
        
        # Extract raw (or normalized) features for ALL frames
        if precomputed_feature_matrices is not None:
            raw_matrices = precomputed_feature_matrices
        else:
            if frames is None:
                raise ValueError("Must provide either 'frames' or 'precomputed_matrices'.")
            if features == 'invariant':
                raw_matrices = _compute_feature_matrices(frames, normalized=normalized)
            elif features == 'soap':
                raw_matrices = _compute_soap_feature_matrices(frames, normalized=normalized)
            else:
                raise ValueError(f"Unknown feature type: {features}")

        for raw_feat in raw_matrices:
            if method.lower() == "qr":
                # QR decomposition of feature matrix
                q, _ = np.linalg.qr(raw_feat)
                basis = q[:, :k]
            else:
                # SVD gives a basis ordered by structural variance
                u, _, _ = np.linalg.svd(raw_feat, full_matrices=False)
                basis = u[:, :k]
                
            bases.append(basis)
        
        return bases

    @staticmethod
    def _distance(U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Computes the Geodesic (arc-length) distance on the Grassmannian.
        Calculated as the L2 norm of the principal angles between subspaces.
        Includes numerical safeguards for ill-conditioned subspaces.
        """
        try:
            # Ensure bases are properly orthonormal (re-orthogonalize via QR)
            U1_safe, _ = np.linalg.qr(U1)
            U2_safe, _ = np.linalg.qr(U2)
            U1_safe = U1_safe[:, :U1.shape[1]]
            U2_safe = U2_safe[:, :U2.shape[1]]
            
            # Compute principal angles with default tolerance
            angles = subspace_angles(U1_safe, U2_safe)
            
            # Clip small numerical errors
            angles = np.clip(angles, 0, np.pi / 2)
            return float(np.linalg.norm(angles))
        except Exception as e:
            # Fallback: compute distance via singular values of U1^T @ U2
            logger.debug(f"Grassmann distance computation fell back to SVD method: {e}")
            try:
                _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
                # Distance from principal angles via singular values
                s_clipped = np.clip(s, -1.0, 1.0)
                angles = np.arccos(s_clipped)
                return float(np.linalg.norm(angles))
            except Exception as e2:
                logger.warning(f"All Grassmann distance methods failed: {e2}. Returning max distance.")
                return float(np.pi / 2)

    @classmethod
    def distance_matrix(
        cls, 
        frames: Optional[Sequence['Atoms']] = None, 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd",
        features : Literal['soap', 'invariant'] = 'invariant',
        normalized: bool = True,
        precomputed_feature_matrices: Optional[Sequence[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise distance matrix for a molecular trajectory.
        """
        num_items = len(precomputed_feature_matrices) if precomputed_feature_matrices is not None else len(frames)
        
        # Precompute bases
        bases = cls._get_uk_bases(
            frames=frames, 
            k=k, 
            method=method, 
            features=features, 
            normalized=normalized,
            precomputed_feature_matrices=precomputed_feature_matrices
        )
        
        # Initialize an empty symmetric matrix
        dist_matrix = np.zeros((num_items, num_items))
        logger.info(f"Computing Grassmann distance matrix for {num_items} items (k={k}, method='{method}', features='{features}', normalized={normalized}).")
        
        # Compute pairwise distances (upper triangle)
        for i in tqdm(range(num_items), desc="Grassmann distances", unit="pair"):
            for j in range(i + 1, num_items):
                d = cls._distance(bases[i], bases[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d # Matrix is symmetric
                
        return dist_matrix


class Riemann:
    """
    Handles molecular representation on the Riemannian Manifold.
    Supports both atomic invariant feature matrices and global descriptors (SOAP).
    """

    @classmethod
    def _get_spd_matrices(
        cls,
        frames=None,
        feature_matrices=None,
        feature_type: str = 'invariant',
        regularization: float = 1e-3,
        n_pca: int = 30
    ) -> np.ndarray:
        """
        Step 1: Extract features and convert to SPD (covariance) matrices.
        """
        # 1. Obtain raw feature matrices (List of D x N matrices)
        if feature_matrices is not None:
            raw_matrices = feature_matrices
        elif frames is not None:
            if feature_type == 'invariant':
                # Returns list of (3, N_atoms)
                raw_matrices = _compute_feature_matrices(frames, normalized=True)
            elif feature_type == 'soap':
                # User has global vectors (1000, 252), convert to list of (252, 1)
                soap_list = _compute_soap_feature_matrices(frames)
                raw_matrices = [np.atleast_2d(s).T if s.ndim == 1 else s for s in soap_list]
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
        else:
            raise ValueError("Must provide 'frames' or 'feature_matrices'.")

        # 2. Optional PCA reduction
        if feature_type == 'soap' and n_pca is not None:
            # Flatten to (N_samples, D) for PCA
            flat_features = np.array([m.flatten() for m in raw_matrices])
            pca = PCA(n_components=n_pca)
            reduced = pca.fit_transform(flat_features)
            # Reshape back to list of (n_pca, 1)
            raw_matrices = [r.reshape(-1, 1) for r in reduced]

        # 3. Build SPD Matrices (Covariance)
        spd_matrices = []
        for X in raw_matrices:
            # X shape is (D, N)
            # C = (X @ X.T) / N -> shape (D, D)
            C = (X @ X.T) / X.shape[1]
            
            # Regularization to ensure it is strictly Positive Definite
            C += np.eye(C.shape[0]) * regularization
            spd_matrices.append(C)

        return np.array(spd_matrices)

    @staticmethod
    def _log_spd(C: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-9, None)
        return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

    @classmethod
    def distance_matrix(
        cls,
        frames=None,
        feature_matrices=None,
        feature_type: str = 'invariant',
        metric: str = "log-euclidean",
        regularization: float = 1e-3,
        n_pca: int = None
    ) -> np.ndarray:
        
        logger.info(f"Computing Riemann distance matrix | Features: {feature_type} | Metric: {metric}")

        # Build SPD matrices
        spd_matrices = cls._get_spd_matrices(
            frames=frames, 
            feature_matrices=feature_matrices,
            feature_type=feature_type,
            regularization=regularization,
            n_pca=n_pca
        )
        
        n = len(spd_matrices)
        dist_matrix = np.zeros((n, n))

        if metric.lower() == "log-euclidean":
            # Log-Euclidean: Compute logs once (O(n * d^3))
            log_mats = np.array([cls._log_spd(C) for C in tqdm(spd_matrices, desc="Matrix Logs")])
            
            for i in tqdm(range(n), desc="Computing Distances"):
                for j in range(i + 1, n):
                    d = np.linalg.norm(log_mats[i] - log_mats[j], ord="fro")
                    dist_matrix[i, j] = dist_matrix[j, i] = d

        else:
            # Affine-Invariant: Solve generalized eigenvalue for every pair (O(n^2 * d^3))
            for i in tqdm(range(n), desc="Computing Distances"):
                for j in range(i + 1, n):
                    try:
                        evs = eigvalsh(spd_matrices[i], spd_matrices[j])
                        d = np.sqrt(np.sum(np.log(np.clip(evs, 1e-9, None))**2))
                    except:
                        d = np.nan
                    dist_matrix[i, j] = dist_matrix[j, i] = d
            
            # Fill failed calculations with max distance
            if np.isnan(dist_matrix).any():
                dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix