from typing import Dict, List, Literal, Sequence

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

        # D = 8 fixed, invariant features. 
        # You can expand this to include SOAP or local coordination.
        feat_vector = [
            z,
            rad,
            en,
            mass,
            dist_to_com,
            #coord,
            #avg_neighbor_z,
            #avg_neighbor_dist
        ]

        features.append(feat_vector)

    return np.array(features).T


def _compute_feature_matrices(
    frames: Sequence[Atoms],
    normalized: bool = False,
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
    Computes the Earth Mover's Distance (Wasserstein-1) between molecules,
    using atomic masses as the distribution weights.
    """

    @staticmethod
    def compute_distance(frame_i: Atoms, frame_j: Atoms, metric: str = 'sqeuclidean') -> float:
        pos_i = np.asarray(frame_i.get_positions(), dtype=np.float64)
        pos_j = np.asarray(frame_j.get_positions(), dtype=np.float64)

        weights_i = np.asarray(frame_i.get_masses(), dtype=np.float64)
        weights_j = np.asarray(frame_j.get_masses(), dtype=np.float64)
        
        weights_i /= weights_i.sum()
        weights_j /= weights_j.sum()

        M = ot.dist(pos_i, pos_j, metric=metric)

        distance = ot.emd2(weights_i, weights_j, M)
        
        return float(distance)

    @classmethod
    def distance_matrix(cls, frames: Sequence[Atoms], metric: str = 'euclidean') -> np.ndarray:
        logger.info(
            f"Computing Wasserstein distance matrix for {len(frames)} frames "
            f"(ground metric='{metric}')."
        )
        n = len(frames)
        dist_matrix = _pairwise_distance_matrix(
            n=n,
            pair_fn=lambda i, j: cls.compute_distance(frames[i], frames[j], metric=metric),
            desc="Wasserstein distances",
        )
        logger.success("Finished Wasserstein distance matrix computation.")
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
        frames: Sequence['Atoms'], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd",
        normalized: bool = False,
    ) -> np.ndarray:
        """
        Maps 3D atomic coordinates to an orthonormal basis in R^n.
        """
        bases = []
        
        # Extract raw (or normalized) features for ALL frames
        raw_matrices = _compute_feature_matrices(frames, normalized=normalized)

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
        
        return np.array(bases)

    @staticmethod
    def _distance(U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Computes the Geodesic (arc-length) distance on the Grassmannian.
        Calculated as the L2 norm of the principal angles between subspaces.
        """
        # Principal angles represent the 'rotations' needed to align two subspaces
        angles = subspace_angles(U1, U2)
        return float(np.linalg.norm(angles))

    @classmethod
    def distance_matrix(
        cls, 
        frames: Sequence['Atoms'], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd",
        normalized: bool = False,
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise distance matrix for a molecular trajectory.
        """
        logger.info(
            f"Computing Grassmann distance matrix for {len(frames)} frames "
            f"(k={k}, method='{method}', normalized={normalized})."
        )
        
        # Precompute bases
        bases = cls._get_uk_bases(frames, k=k, method=method, normalized=normalized)
        num_frames = len(bases)
        dist_matrix = _pairwise_distance_matrix(
            n=num_frames,
            pair_fn=lambda i, j: cls._distance(bases[i], bases[j]),
            desc="Grassmann distances",
        )
        
        logger.success("Finished Grassmann distance matrix computation.")
        return dist_matrix

class Riemann:
    """
    Handles molecular representation on the Riemannian Manifold of SPD matrices.

    Each molecule is represented as a covariance matrix of invariant atomic
    feature vectors. Distances are computed using either:

    - Log-Euclidean metric
    - Affine-Invariant Riemannian metric
    """

    @classmethod
    def _get_spd_matrices(
        cls,
        frames,
        regularization: float = 1e-6,
        normalized: bool = False,
    ) -> np.ndarray:
        """
        Converts frames into SPD covariance matrices.
        """
        # Extract invariant atomic features (optionally normalized)
        raw_matrices = _compute_feature_matrices(frames, normalized=normalized)

        spd_matrices = []

        for raw_feat in raw_matrices:
            X = raw_feat

            C = (X @ X.T) / X.shape[1]
            C += np.eye(C.shape[0]) * regularization

            spd_matrices.append(C)

        return np.array(spd_matrices)

    # ---------------------------------------------------------

    @staticmethod
    def _log_spd(C: np.ndarray) -> np.ndarray:
        """
        Computes matrix logarithm of SPD matrix using eigen-decomposition.
        """
        eigvals, eigvecs = np.linalg.eigh(C)

        eigvals = np.clip(eigvals, 1e-12, None)

        log_eigvals = np.log(eigvals)

        return eigvecs @ np.diag(log_eigvals) @ eigvecs.T

    # ---------------------------------------------------------

    @classmethod
    def _log_spd_batch(cls, spd_matrices: np.ndarray) -> np.ndarray:
        """
        Precompute log-SPD matrices for all frames.
        """
        log_mats = []

        for C in tqdm(spd_matrices, desc="Computing log-SPD matrices"):
            log_mats.append(cls._log_spd(C))

        return np.array(log_mats)

    # ---------------------------------------------------------

    @staticmethod
    def _distance_log_euclidean(logC1: np.ndarray, logC2: np.ndarray) -> float:
        """
        Log-Euclidean Riemannian distance.

        d(A,B) = || log(A) - log(B) ||_F
        """
        return float(np.linalg.norm(logC1 - logC2, ord="fro"))

    # ---------------------------------------------------------

    @staticmethod
    def _distance_affine_invariant(C1: np.ndarray, C2: np.ndarray) -> float:
        """
        Affine-Invariant Riemannian distance.

        d(A,B) = sqrt( sum( log(lambda_i)^2 ) )
        where lambda_i are generalized eigenvalues.
        """

        eigvals = eigvalsh(C1, C2)
        eigvals = np.clip(eigvals, 1e-12, None)

        return float(np.sqrt(np.sum(np.log(eigvals) ** 2)))

    # ---------------------------------------------------------

    @classmethod
    def distance_matrix(
        cls,
        frames,
        metric: str = "log-euclidean",
        regularization: float = 1e-6,
        normalized: bool = False,
    ) -> np.ndarray:
        """
        Computes pairwise Riemannian distance matrix for frames.
        """

        logger.info(
            f"Computing Riemannian distance matrix for {len(frames)} frames "
            f"(metric='{metric}', normalized={normalized})"
        )

        metric_key = metric.lower()

        if metric_key not in {"log-euclidean", "affine-invariant"}:
            raise ValueError(
                "metric must be one of: ['log-euclidean', 'affine-invariant']"
            )

        # Step 1: Build SPD matrices
        spd_matrices = cls._get_spd_matrices(
            frames,
            regularization=regularization,
            normalized=normalized,
        )
        n = len(spd_matrices)

        dist_matrix = np.zeros((n, n))

        # -------------------------------------------------
        # LOG-EUCLIDEAN METRIC
        # -------------------------------------------------

        if metric_key == "log-euclidean":

            log_mats = cls._log_spd_batch(spd_matrices)

            total_pairs = n * (n - 1) // 2

            with tqdm(total=total_pairs, desc="Riemann distances", unit="pair") as pbar:

                for i in range(n):
                    for j in range(i + 1, n):

                        d = cls._distance_log_euclidean(log_mats[i], log_mats[j])

                        dist_matrix[i, j] = dist_matrix[j, i] = d

                        pbar.update(1)

        # -------------------------------------------------
        # AFFINE-INVARIANT METRIC
        # -------------------------------------------------

        else:

            total_pairs = n * (n - 1) // 2

            with tqdm(total=total_pairs, desc="Riemann distances", unit="pair") as pbar:

                for i in range(n):
                    for j in range(i + 1, n):

                        d = cls._distance_affine_invariant(
                            spd_matrices[i], spd_matrices[j]
                        )

                        dist_matrix[i, j] = dist_matrix[j, i] = d

                        pbar.update(1)

        logger.success("Finished Riemannian distance matrix computation.")

        return dist_matrix
