from src.datasets import QM9Dataset
from src.non_euclidean import _feature_matrices_from_df

import numpy as np
import logging
from typing import Any
from sklearn.decomposition import PCA
from sklearn.covariance import oas, ledoit_wolf, empirical_covariance
from sklearn.metrics import pairwise_distances
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class Riemann:
    
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
    def get_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = True,
        estimator: str = 'oas'
    ) -> np.ndarray:
        
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        if pca:
            n_pca = df['num_atoms'].min() - 2
            raw_matrices = cls.matrix_pca(n_pca, raw_matrices)

        spd_matrices = []
        for X in raw_matrices:
            X = np.asarray(X)
            
            # Select covariance estimator
            if estimator == 'oas':
                C, _ = oas(X, assume_centered=False)
            elif estimator == 'ledoit_wolf':
                C, _ = ledoit_wolf(X, assume_centered=False)
            elif estimator == 'empirical':
                C = empirical_covariance(X, assume_centered=False)
            else:
                raise ValueError(f"Unknown estimator: {estimator}")
                
            spd_matrices.append(C)

        for idx, C in enumerate(spd_matrices):
            if not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Matrix at index {idx} failed symmetry validation")
            
            eigvals = np.linalg.eigvalsh(C)
            min_eig = eigvals.min()
            if min_eig <= 0:
                logger.warning(f"Matrix {idx} has eigenvalue <= 0 ({min_eig}) using {estimator}")

        return np.array(spd_matrices)

    @classmethod
    def vectorized_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = True,
        estimator: str = 'oas',
        eig_floor: float = 1e-12,
        warn_threshold: float = 1e-6,
    ) -> np.ndarray:
        spd_matrices = cls.get_spd_matrices(df, descriptor, pca, estimator)
        return cls.log_euclidean_vectorize(spd_matrices, eig_floor, warn_threshold)
    
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

def run_covariance_experiment(df: pl.DataFrame, target_property: str = 'gap') -> pl.DataFrame:
    """
    Evaluates different covariance estimators for SPD manifold generation
    using a Polars DataFrame.
    """
    configs = [
        {"estimator": "empirical", "pca": True},
        {"estimator": "empirical", "pca": False},
        {"estimator": "ledoit_wolf", "pca": True},
        {"estimator": "ledoit_wolf", "pca": False},
        {"estimator": "oas", "pca": True},
        {"estimator": "oas", "pca": False},
    ]

    # Polars syntax to extract the target column as a 1D NumPy array
    y = df.get_column(target_property).to_numpy()
    results = []

    for conf in configs:
        est = conf["estimator"]
        use_pca = conf["pca"]
        print(f"\n--- Running: Estimator={est.upper()} | PCA={use_pca} ---")
        
        try:
            # 1. Generate SPD Matrices
            spd_mats = Riemann.get_spd_matrices(
                df=df, descriptor="soap", pca=use_pca, estimator=est
            )
            
            # 2. Evaluate Numerical Stability (Condition Number)
            condition_numbers = []
            for C in spd_mats:
                eigvals = np.linalg.eigvalsh(C)
                cond = eigvals.max() / max(eigvals.min(), 1e-15)
                condition_numbers.append(cond)
                
            median_cond = np.median(condition_numbers)
            print(f"Median Condition Number: {median_cond:.2e}")

            # 3. Vectorize via Log-Euclidean mapping
            X_vec = Riemann.log_euclidean_vectorize(spd_mats)
            
            # 4. Property Prediction Task
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            print(f"Prediction MAE: {mae:.4f} | R2: {r2:.4f}")
            
            results.append({
                "Estimator": est,
                "PCA": use_pca,
                "Median Condition No.": median_cond,
                "MAE": mae,
                "R2": r2,
                "Status": "Success"
            })
            
        except Exception as e:
            print(f"FAILED: {str(e)}")
            results.append({
                "Estimator": est,
                "PCA": use_pca,
                "Median Condition No.": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "Status": f"Failed: {type(e).__name__}"
            })

    return pl.DataFrame(results)

if __name__ == "__main__":
    qm9 = QM9Dataset(limit=1000, descriptors=["soap"])
    df = qm9.load()
    summary_df = run_covariance_experiment(df, target_property='gap')
    print("\nFinal Summary:\n")
    print(summary_df)