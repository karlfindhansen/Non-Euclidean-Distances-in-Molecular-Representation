import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from loguru import logger
from typing import Literal

# Import dscribe kernels
from dscribe.kernels import AverageKernel, REMatchKernel

class DistanceCalculator:
    """
    Computes and caches pairwise distance matrices.
    """
    
    def get_matrix(
        self, 
        data_series: pl.Series, 
        metric: Literal['jaccard', 'euclidean', 'cosine', 'soap_kernel', 'hamming', 'average_kernel', 'rematch_kernel'], 
    ) -> np.ndarray:
        """
        Computes the distance matrix.
        """
        return self._compute_and_save(data_series, metric)

    def _compute_and_save(self, series: pl.Series, metric: str) -> np.ndarray:
        
        data_list = series.to_list()
        
        if not data_list:
            raise ValueError("Input series is empty.")

        # --- DScribe Kernels Implementation ---
        if metric in ("average_kernel", "rematch_kernel"):
            formatted_data = []
            for item in data_list:
                arr = np.array(item, dtype=np.float32)
                
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                
                formatted_data.append(arr)
            
            try:
                if metric == "average_kernel":
                    kernel = AverageKernel(metric="linear")
                else:
                    kernel = REMatchKernel(metric="linear", alpha=1.0, threshold=1e-6)
                
                kernel_matrix = kernel.create(formatted_data)
                
                # Safely convert the similarity matrix to a distance matrix using D = sqrt(K_ii + K_jj - 2K_ij)
                diag = np.diag(kernel_matrix)
                dist_sq = diag[:, None] + diag[None, :] - 2.0 * kernel_matrix
                dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
                dist_matrix = np.sqrt(dist_sq)
                
                # Eliminate floating point artifacts on diagonals and force strict symmetry
                np.fill_diagonal(dist_matrix, 0.0)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
                return dist_matrix
                
            except Exception as e:
                logger.error(f"DScribe Kernel matrix computation failed: {e}")
                raise

        dtype = bool if metric == 'jaccard' else np.float32
        data_array = np.array(data_list, dtype=dtype)

        try:
            if metric == "soap_kernel":
                # SOAP kernel distance: 1 - normalized dot product
                zeta = 1.0 
                
                norms = np.linalg.norm(data_array, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized = data_array / norms
                
                base_kernel = normalized @ normalized.T
                
                kernel_matrix = base_kernel ** zeta if zeta != 1.0 else base_kernel
                    
                dist_sq = 2.0 - 2.0 * kernel_matrix
                dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
                dist_matrix = np.sqrt(dist_sq)
                
                np.fill_diagonal(dist_matrix, 0.0)
                dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

            else:
                condensed = pdist(data_array, metric=metric)
                dist_matrix = squareform(condensed)
            return dist_matrix
        
        except Exception as e:
            logger.error(f"Matrix computation failed: {e}")
            raise