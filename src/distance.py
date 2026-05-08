import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from loguru import logger
from typing import Literal

class DistanceCalculator:
    """
    Computes and caches pairwise distance matrices.
    """
    
    def get_matrix(
        self, 
        data_series: pl.Series, 
        metric: Literal['jaccard', 'euclidean', 'cosine', 'soap_kernel', 'hamming'], 
    ) -> np.ndarray:
        """
        Computes the distance matrix.
        """

        return self._compute_and_save(data_series, metric)

    def _compute_and_save(self, series: pl.Series, metric: str) -> np.ndarray:
        
        data_list = series.to_list()
        
        if not data_list:
            raise ValueError("Input series is empty.")

        # Determine dtype based on metric
        dtype = bool if metric == 'jaccard' else np.float32
        data_array = np.array(data_list, dtype=dtype)

        try:
            if metric == "soap_kernel":
                # SOAP kernel distance: 1 - normalized dot product
                zeta = 1.0  # Optional: change to 2.0 or 4.0 to sharpen the similarity
                
                # 1. Normalize the power spectrum vectors
                norms = np.linalg.norm(data_array, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized = data_array / norms
                
                # 2. Compute the base rotationally-averaged overlap (dot product)
                base_kernel = normalized @ normalized.T
                
                # 3. Apply the zeta exponent (with clipping to prevent NaNs on tiny negatives)
                kernel_matrix = base_kernel ** zeta if zeta != 1.0 else base_kernel
                    
                # 4. Compute the formal Kernel Distance: D = sqrt(2 - 2K)
                dist_sq = 2.0 - 2.0 * kernel_matrix
                dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
                dist_matrix = np.sqrt(dist_sq)
                
                # Force exact 0s on the diagonal and ensure strict symmetry
                np.fill_diagonal(dist_matrix, 0.0)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

            else:
                condensed = pdist(data_array, metric=metric)
                dist_matrix = squareform(condensed)
            return dist_matrix
        
        except Exception as e:
            logger.error(f"Matrix computation failed: {e}")
            raise
