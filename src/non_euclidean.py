from typing import Sequence
from scipy.linalg import subspace_angles
from pyriemann.utils.distance import distance_riemann, distance_logeuclid
from ase import Atoms
import persim
import gudhi as gd
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

        for i in range(n):
            for j in range(i + 1, n):
                d = cls.compute_distance(frames[i], frames[j], metric=metric)
                dist_matrix[i, j] = dist_matrix[j, i] = d
        logger.debug("Finished Wasserstein distance matrix computation.")
        return dist_matrix

class PersistentHomology:
    """
    Task 6.1: Persistence diagrams from 3D point clouds of atoms.
    Task 6.2: Diagram distances (Bottleneck or Sliced Wasserstein).
    """

    @staticmethod
    def _compute_ripser(points: np.ndarray, max_dim: int) -> Dict[int, np.ndarray]:
        dgms = ripser(points, maxdim=max_dim)["dgms"]
        return {d: np.asarray(dgms[d]) for d in range(max_dim + 1)}

    @staticmethod
    def _compute_gudhi(points: np.ndarray, max_dim: int) -> Dict[int, np.ndarray]:
        rips = gd.RipsComplex(points=points).create_simplex_tree(max_dimension=max_dim + 1)
        rips.persistence()
        return {d: np.asarray(rips.persistence_intervals_in_dimension(d)) for d in range(max_dim + 1)}

    @classmethod
    def compute_persistence_diagrams(
        cls, frames: Sequence[Atoms], max_homology_dim: int = 2, backend: str = "ripser"
    ) -> List[Dict[int, np.ndarray]]:
        backend_key = backend.lower()
        if backend_key not in {"ripser", "gudhi"}:
            logger.error(f"Unknown persistence backend '{backend}'.")
            raise ValueError("backend must be one of: ['ripser', 'gudhi']")

        logger.info(
            f"Computing persistence diagrams for {len(frames)} frames "
            f"(backend='{backend_key}', max_homology_dim={max_homology_dim})."
        )
        compute_fn = cls._compute_ripser if backend_key == "ripser" else cls._compute_gudhi
        diagrams = []

        for frame in frames:
            pts = frame.get_positions()
            if len(pts) == 0:
                diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
            else:
                diagrams.append(compute_fn(pts, max_homology_dim))
        logger.debug("Finished persistence diagram computation.")
        return diagrams

    @staticmethod
    def distance(
        dgm1: Dict[int, np.ndarray],
        dgm2: Dict[int, np.ndarray],
        metric: str = "bottleneck",
        dims: Sequence[int] = (0, 1, 2),
        sw_projections: int = 50
    ) -> float:
        """Computes total distance across specified homology dimensions."""
        metric_key = metric.lower()
        if metric_key not in {"bottleneck", "b", "sliced-wasserstein", "sliced_wasserstein", "sw"}:
            logger.error(f"Unknown persistence metric '{metric}'.")
            raise ValueError(
                "metric must be one of: ['bottleneck', 'b', 'sliced-wasserstein', 'sliced_wasserstein', 'sw']"
            )

        total_dist = 0.0
        
        for d in dims:
            p1, p2 = dgm1.get(d, np.empty((0, 2))), dgm2.get(d, np.empty((0, 2)))
            
            if metric_key in {"bottleneck", "b"}:
                total_dist += persim.bottleneck(p1, p2)
            else:
                total_dist += persim.sliced_wasserstein(p1, p2, M=sw_projections)
                
        return float(total_dist)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        backend: str = "ripser",
        metric: str = "bottleneck",
        max_homology_dim: int = 2,
        homology_dims: Sequence[int] = (0, 1, 2)
    ) -> np.ndarray:
        """Generates the pairwise distance matrix for a set of molecular frames."""
        logger.info(
            f"Computing persistent homology distance matrix for {len(frames)} frames "
            f"(backend='{backend}', metric='{metric}', max_homology_dim={max_homology_dim}, "
            f"dims={tuple(homology_dims)})."
        )
        dgms = cls.compute_persistence_diagrams(frames, max_homology_dim, backend)
        n = len(dgms)
        dist_mat = np.zeros((n, n))

        for i in tqdm(range(n), desc="Computing Persistence Diagram Distance Matrix"):
            for j in range(i + 1, n):
                d = cls.distance(dgms[i], dgms[j], metric=metric, dims=homology_dims)
                dist_mat[i, j] = dist_mat[j, i] = d
        logger.debug("Finished persistent homology distance matrix computation.")
        return dist_mat

class Grassmann:
    """
    Task 7.1: Grassmann Manifold (Subspaces)
    Supports both QR and SVD for orthonormal basis generation.
    """

    @staticmethod
    def get_uk_bases(
        frames: Sequence[Atoms], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd"
    ) -> np.ndarray:
        """
        Computes an orthonormal basis for each frame.
        
        Args:
            frames: Sequence of ASE Atoms objects.
            k: Dimension of the subspace (usually 3 for 3D coordinates).
            method: 'qr' for fast decomposition, 'svd' for principal component basis.
        """
        bases = []
        max_atoms = max(len(f) for f in frames)
        for frame in frames:
            coords = frame.get_positions()
            n_atoms = len(coords)
            centered = coords - np.mean(coords, axis=0)

            padded = np.zeros((max_atoms, 3))
            padded[:n_atoms, :] = centered
            
            if method.lower() == "qr":
                q, _ = np.linalg.qr(padded)
                basis = q[:, :k]
            else:
                u, s, _ = np.linalg.svd(padded, full_matrices=False)
                basis = u[:, :k]
                
            bases.append(basis)
        
        return np.array(bases)

    @staticmethod
    def distance(U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Step C: Compute Grassmann Distance using Principal Angles.
        d = sqrt(sum(theta_i^2))
        """
        angles = subspace_angles(U1, U2)
        return float(np.linalg.norm(angles))

    @classmethod
    def distance_matrix(
        cls, 
        frames: Sequence[Atoms], 
        k: int = 3, 
        method: Literal["qr", "svd"] = "svd"
    ) -> np.ndarray:
        """
        Generates the pairwise Grassmann distance matrix.
        """
        logger.info(
            f"Computing Grassmann distance matrix for {len(frames)} frames "
            f"(k={k}, method='{method}')."
        )
        bases = cls.get_uk_bases(frames, k=k, method=method)
        num_frames = len(bases)
        dist_matrix = np.zeros((num_frames, num_frames))

        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                dist = cls.distance(bases[i], bases[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        logger.debug("Finished Grassmann distance matrix computation.")
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

        for i in range(n):
            for j in range(i + 1, n):

                d = metric_fn(covs[i], covs[j])

                dist_matrix[i, j] = dist_matrix[j, i] = d
        logger.success("Finished Riemannian distance matrix computation.")

        return dist_matrix
