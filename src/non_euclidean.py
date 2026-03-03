from typing import Sequence
from importlib import import_module
from scipy.linalg import subspace_angles
from pyriemann.utils.distance import distance_riemann, distance_logeuclid
from ase import Atoms
import persim
import gudhi as gd
from ripser import ripser
from typing import Dict, List, Literal
import numpy as np

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
        for frame in frames:
            coords = frame.get_positions()
            centered = coords - np.mean(coords, axis=0)
            
            if method.lower() == "qr":
                q, _ = np.linalg.qr(centered)
                basis = q[:, :k]
            else:
                u, s, _ = np.linalg.svd(centered, full_matrices=False)
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
        bases = cls.get_uk_bases(frames, k=k, method=method)
        num_frames = len(bases)
        dist_matrix = np.zeros((num_frames, num_frames))

        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                dist = cls.distance(bases[i], bases[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
                
        return dist_matrix

class Riemann:

    _METRICS = {
        "log-euclidean":    distance_logeuclid,
        "affine-invariant": distance_riemann,
    }

    @staticmethod
    def compute_covariance_matrices(frames: Sequence[Atoms]) -> np.ndarray:

        covs = []
        for frame in frames:
            positions = frame.get_positions()       
            cov = np.cov(positions, rowvar=False)   
            cov += np.eye(cov.shape[0]) * 1e-6     
            covs.append(cov)
        return np.array(covs)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        metric_type: str = "log-euclidean",
    ) -> np.ndarray:

        key = metric_type.strip().lower().replace("_", "-").replace(" ", "-")
        metric_fn = cls._METRICS.get(key)
        if metric_fn is None:
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
        
        compute_fn = cls._compute_ripser if backend.lower() == "ripser" else cls._compute_gudhi
        diagrams = []

        for frame in frames:
            pts = frame.get_positions()
            if len(pts) == 0:
                diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
            else:
                diagrams.append(compute_fn(pts, max_homology_dim))
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
        total_dist = 0.0
        
        for d in dims:
            p1, p2 = dgm1.get(d, np.empty((0, 2))), dgm2.get(d, np.empty((0, 2)))
            
            if metric.lower() in ["bottleneck", "b"]:
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
        dgms = cls.compute_persistence_diagrams(frames, max_homology_dim, backend)
        n = len(dgms)
        dist_mat = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = cls.distance(dgms[i], dgms[j], metric=metric, dims=homology_dims)
                dist_mat[i, j] = dist_mat[j, i] = d
                
        return dist_mat