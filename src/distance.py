import os
import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from loguru import logger
from typing import Literal

class DistanceCalculator:
    """
    Computes and caches pairwise distance matrices.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def get_matrix(
        self, 
        data_series: pl.Series, 
        metric: Literal['jaccard', 'euclidean', 'cosine', 'soap_kernel', 'hamming'], 
        filename: str
    ) -> np.ndarray:
        """
        Retrieves or computes the distance matrix.
        """
        file_path = os.path.join(self.cache_dir, filename)

        if os.path.exists(file_path):
            logger.info(f"Loading cached distance matrix from {file_path}")
            return np.load(file_path)

        return self._compute_and_save(data_series, metric, file_path)

    def _compute_and_save(self, series: pl.Series, metric: str, path: str) -> np.ndarray:
        logger.info(f"Computing {metric} distance matrix...")
        
        data_list = series.to_list()
        
        if not data_list:
            raise ValueError("Input series is empty.")

        # Determine dtype based on metric
        dtype = bool if metric == 'jaccard' else np.float32
        data_array = np.array(data_list, dtype=dtype)

        try:
            if metric == "soap_kernel":
                # SOAP kernel distance: 1 - normalized dot product
                norms = np.linalg.norm(data_array, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized = data_array / norms
                kernel = normalized @ normalized.T
                matrix = 1.0 - kernel
            else:
                condensed = pdist(data_array, metric=metric)
                matrix = squareform(condensed)
            np.save(path, matrix)
            logger.success(f"Saved distance matrix to {path}")
            return matrix
        except Exception as e:
            logger.error(f"Matrix computation failed: {e}")
            raise
