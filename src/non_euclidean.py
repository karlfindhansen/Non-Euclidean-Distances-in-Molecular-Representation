from typing import Sequence
from scipy.linalg import subspace_angles
from pyriemann.utils.distance import distance_riemann, distance_logeuclid
from ase import Atoms
import persim
from ripser import ripser
from typing import Dict, List, Literal
import numpy as np
from tqdm import tqdm
from loguru import logger
import ot

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
    def distance_matrix(cls, frames: Sequence[Atoms], metric: str = 'sqeuclidean') -> np.ndarray:
        logger.info(
            f"Computing Wasserstein distance matrix for {len(frames)} frames "
            f"(ground metric='{metric}')."
        )
        n = len(frames)
        dist_matrix = np.zeros((n, n))
        total_pairs = n * (n - 1) // 2

        with tqdm(total=total_pairs, desc="Wasserstein distances", unit="pair") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    d = cls.compute_distance(frames[i], frames[j], metric=metric)
                    dist_matrix[i, j] = dist_matrix[j, i] = d
                    pbar.update(1)
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
        dist_mat = np.zeros((n, n))
        
        # Only compute the upper triangle to halve the required calculations
        total_pairs = n * (n - 1) // 2

        with tqdm(total=total_pairs, desc="Persistence distances", unit="pair") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    d = cls.distance(dgms[i], dgms[j], metric=metric, dims=homology_dims)
                    # Mirror the upper triangle to the lower triangle
                    dist_mat[i, j] = dist_mat[j, i] = d
                    pbar.update(1)
                    
        logger.success("Finished persistent homology distance matrix computation.")
        return dist_mat

class Grassmann:
    """
    Handles molecular representation on Grassmann Manifolds G(k, n).
    Represents each molecule as a k-dimensional subspace in R^n (atom-space).
    """

    @staticmethod
    def _get_uk_bases(
        frames: Sequence['Atoms'], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd"
    ) -> np.ndarray:
        """
        Maps 3D atomic coordinates to an orthonormal basis in R^n.
        """
        bases = []
        # Ambient dimension n must be constant across the dataset for manifold comparison
        max_atoms = max(len(f) for f in frames)
        
        for frame in frames:
            coords = frame.get_positions(invariant=False)
            n_atoms = len(coords)

            # Translation invariance: move geometric centroid to origin
            centered = coords - np.mean(coords, axis=0)

            # Zero-padding ensures all molecules live in the same R^max_atoms space
            # Missing atoms are treated as having zero variance in those dimensions
            padded = np.zeros((max_atoms, 3))
            padded[:n_atoms, :] = centered

            # Rotational invariance: The Gram Matrix (n x n) captures relative.
            # distances/angles between atoms, independent of 3D orientation.
            gram = padded @ padded.T
            
            if method.lower() == "qr":
                # QR decomposition of Gram matrix
                q, _ = np.linalg.qr(gram)
                basis = q[:, :k]
            else:
                # SVD gives a basis ordered by structural variance
                u, _, _ = np.linalg.svd(gram, full_matrices=False)
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
        method: Literal["qr", "svd"] = "svd"
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise distance matrix for a molecular trajectory.
        """
        logger.info(
            f"Computing Grassmann distance matrix for {len(frames)} frames "
            f"(k={k}, method='{method}')."
        )
        
        # Precompute bases
        bases = cls._get_uk_bases(frames, k=k, method=method)
        num_frames = len(bases)
        dist_matrix = np.zeros((num_frames, num_frames))
        total_pairs = num_frames * (num_frames - 1) // 2

        with tqdm(total=total_pairs, desc="Grassmann distances", unit="pair") as pbar:
            for i in range(num_frames):
                for j in range(i + 1, num_frames):
                    # Compute distance once per unique pair
                    dist = cls._distance(bases[i], bases[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
                    pbar.update(1)
                    
        logger.success("Finished Grassmann distance matrix computation.")
        return dist_matrix

class Riemann:

    _METRICS = {
        "log-euclidean":    distance_logeuclid,
        "affine-invariant": distance_riemann,
    }

    @staticmethod
    def compute_covariance_matrices(frames: Sequence[Atoms]) -> np.ndarray:
        covs = []
        for idx, frame in enumerate(frames):
            positions = frame.get_positions()
            charge = frame.get_initial_charges()

            positions = np.hstack((positions, charge[:, np.newaxis]))

            cov = np.cov(positions, rowvar=False)   
            cov = (cov + cov.T) / 2
            cov += np.eye(cov.shape[0]) * 1e-6   

            covs.append(cov)
        return np.array(covs)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        metric_type: str = "log-euclidean",
    ) -> np.ndarray:
        logger.info(
            f"Computing Riemannian distance matrix for {len(frames)} frames "
            f"(metric_type='{metric_type}')."
        )
        key = metric_type.strip().lower().replace("_", "-").replace(" ", "-")
        metric_fn = cls._METRICS.get(key)
        if metric_fn is None:
            logger.error(f"Unknown Riemann metric_type '{metric_type}'.")
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                f"Choose from: {list(cls._METRICS)}."
            )

        covs = cls.compute_covariance_matrices(frames)
        n = len(covs)
        dist_matrix = np.zeros((n, n))
        total_pairs = n * (n - 1) // 2

        with tqdm(total=total_pairs, desc="Riemannian distances", unit="pair") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    d = metric_fn(covs[i], covs[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = d
                    pbar.update(1)
        logger.success("Finished Riemannian distance matrix computation.")

        return dist_matrix
