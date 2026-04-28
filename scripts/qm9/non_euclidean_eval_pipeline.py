from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import kmedoids

from loguru import logger
from scipy.linalg import logm, expm, sqrtm, inv
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.datasets import QM9Dataset
from src.non_euclidean import Grassmann, Riemann
from scripts.materials_project.euclidean_evaluation_pipeline import (
    INVARIANT_FEATURES_QM9,
    _build_qm9_frames_from_df,
    _iter_invariant_combinations,
    _screen_correlated_invariant_features,
    build_invariant_matrix,
    get_overall_chemical_coherence,
)


DEFAULT_DBSCAN_EPS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
DEFAULT_DBSCAN_MIN_SAMPLES = [2, 3, 5, 8, 10]
DEFAULT_AP_PREFERENCE_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75]
DEFAULT_AP_DAMPING_VALUES = [0.60, 0.70, 0.80, 0.90]

# ==============================================================================
# TRUE K-MEANS MANIFOLD IMPLEMENTATIONS
# ==============================================================================

def log_euclidean_kmeans(spd_matrices: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard K-Means using the Log-Euclidean metric.
    """
    n_samples, d, _ = spd_matrices.shape
    
    # Map to tangent space using existing function
    log_mats = Riemann._log_spd_batch(spd_matrices)
    flattened_logs = log_mats.reshape(n_samples, -1)
    
    # Run Euclidean K-Means on the tangent space vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(flattened_logs)
    
    # Map centroids back to the SPD manifold (Exponential map)
    centroids_spd = []
    for center in kmeans.cluster_centers_:
        center_matrix = center.reshape(d, d)
        eigvals, eigvecs = np.linalg.eigh(center_matrix)
        exp_C = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
        centroids_spd.append(exp_C)
        
    return labels, np.array(centroids_spd)


def affine_invariant_frechet_mean(spd_matrices: Sequence[np.ndarray], max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Computes the Fréchet mean of a set of SPD matrices under the Affine-Invariant metric.
    """
    M = np.mean(spd_matrices, axis=0)
    
    for _ in range(max_iter):
        M_half = sqrtm(M)
        M_inv_half = inv(M_half)
        
        tangent_vectors = []
        for C in spd_matrices:
            whitened_C = M_inv_half @ C @ M_inv_half
            whitened_C = (whitened_C + whitened_C.T) / 2.0 
            tangent_vectors.append(logm(whitened_C))
            
        mean_tangent = np.mean(tangent_vectors, axis=0)
        mean_tangent = (mean_tangent + mean_tangent.T) / 2.0 
        
        if np.linalg.norm(mean_tangent, ord='fro') < tol:
            break
            
        M = M_half @ expm(mean_tangent) @ M_half
        M = (M + M.T) / 2.0 
        
    return M


def grassmann_karcher_mean(subspaces: Sequence[np.ndarray], max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
    """
    Computes the Karcher mean of a list of orthonormal bases (subspaces) on the Grassmannian.
    """
    M = subspaces[0]
    
    for _ in range(max_iter):
        gradient = np.zeros_like(M)
        
        for U in subspaces:
            U_proj = M.T @ U
            U_orth = U - M @ U_proj
            
            try:
                direction_matrix = U_orth @ np.linalg.inv(U_proj)
            except np.linalg.LinAlgError:
                continue 
                
            X, S, Yh = np.linalg.svd(direction_matrix, full_matrices=False)
            tangent_vec = X @ np.diag(np.arctan(S)) @ Yh
            gradient += tangent_vec
            
        gradient /= len(subspaces)
        if np.linalg.norm(gradient) < tol:
            break
            
        U_grad, S_grad, Vh_grad = np.linalg.svd(gradient, full_matrices=False)
        cos_S = np.diag(np.cos(S_grad))
        sin_S = np.diag(np.sin(S_grad))
        
        M = M @ Vh_grad.T @ cos_S + U_grad @ sin_S
        M = M @ Vh_grad 
        M, _ = np.linalg.qr(M)
        
    return M


def riemannian_kmeans(
    items: Sequence[np.ndarray], 
    n_clusters: int, 
    manifold: str, 
    max_iter: int = 30,
    random_state: int = 42
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Custom K-Means loop utilizing Fréchet/Karcher means.
    """
    np.random.seed(random_state)
    n = len(items)
    
    initial_indices = np.random.choice(n, n_clusters, replace=False)
    centroids = [items[i] for i in initial_indices]
    labels = np.zeros(n, dtype=int)
    
    if manifold == "riemann":
        dist_fn = Riemann._distance_affine_invariant
        mean_fn = affine_invariant_frechet_mean
    elif manifold == "grassmann":
        dist_fn = Grassmann._distance
        mean_fn = grassmann_karcher_mean
    else:
        raise ValueError("Manifold must be 'riemann' or 'grassmann'")

    for iteration in range(max_iter):
        old_labels = labels.copy()
        
        for i, item in enumerate(items):
            distances = [dist_fn(item, c) for c in centroids]
            labels[i] = np.argmin(distances)
            
        if np.array_equal(labels, old_labels) and iteration > 0:
            break
            
        for k in range(n_clusters):
            cluster_items = [items[j] for j in range(n) if labels[j] == k]
            if len(cluster_items) == 0:
                centroids[k] = items[np.random.randint(0, n)]
                continue
            centroids[k] = mean_fn(cluster_items)
            
    return labels, centroids

# ==============================================================================
# PIPELINE HELPER FUNCTIONS
# ==============================================================================

def _normalize_feature_matrices(feature_matrices: Sequence[np.ndarray]) -> List[np.ndarray]:
    if not feature_matrices:
        return []

    stacked_parts = [mat.T for mat in feature_matrices if mat.size > 0]
    if not stacked_parts:
        return list(feature_matrices)

    stacked = np.vstack(stacked_parts)
    scaler = StandardScaler().fit(stacked)

    normalized: List[np.ndarray] = []
    for mat in feature_matrices:
        if mat.size == 0:
            normalized.append(mat)
            continue
        normalized.append(scaler.transform(mat.T).T)

    return normalized


def _nonzero_upper_triangle(dist_matrix: np.ndarray) -> np.ndarray:
    tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    tri = tri[np.isfinite(tri)]
    return tri[tri > 0]


def _gaussian_affinity_from_distance(
    dist_matrix: np.ndarray,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    non_zero = _nonzero_upper_triangle(dist_matrix)

    if sigma is None:
        sigma = float(np.median(non_zero)) if non_zero.size else 1.0

    sigma = max(float(sigma), 1e-12)
    affinity = np.exp(-(dist_matrix.astype(np.float64) ** 2) / (2.0 * sigma**2))
    np.fill_diagonal(affinity, 1.0)
    return affinity, sigma


def _num_valid_clusters(labels: np.ndarray) -> int:
    return len(set(labels) - {-1})


def _safe_silhouette(dist_matrix: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None

    try:
        return float(silhouette_score(dist_matrix, labels, metric="precomputed"))
    except Exception:
        return None


def _distance_profile_features(dist_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(dist_matrix, dtype=np.float64)


def _safe_davies_bouldin(dist_matrix: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None

    try:
        return float(davies_bouldin_score(_distance_profile_features(dist_matrix), labels))
    except Exception:
        return None


def _spectral_eigengap_heuristic(
    affinity: np.ndarray,
    k_range: Sequence[int],
) -> Tuple[Optional[int], Optional[float]]:
    if not k_range:
        return None, None

    degree = np.sum(affinity, axis=1)
    safe_degree = np.clip(degree, 1e-12, None)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(safe_degree))
    laplacian = np.eye(affinity.shape[0]) - inv_sqrt_degree @ affinity @ inv_sqrt_degree

    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
    except Exception:
        return None, None

    candidate_ks = sorted({int(k) for k in k_range if 1 <= int(k) < len(eigenvalues)})
    if not candidate_ks:
        return None, None

    best_k = None
    best_gap = -np.inf
    for k in candidate_ks:
        gap = float(eigenvalues[k] - eigenvalues[k - 1])
        if gap > best_gap:
            best_gap = gap
            best_k = k

    return best_k, best_gap if np.isfinite(best_gap) else None

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def _compute_manifold_distance_matrix(
    manifold: str,
    feature_matrices: Sequence[np.ndarray],
    grassmann_k: int = 3,
    grassmann_method: str = "svd",
    riemann_metric: str = "log-euclidean",
    riemann_regularization: float = 1e-6,
) -> np.ndarray:
    manifold_key = manifold.lower()

    if manifold_key == "grassmann":
        return Grassmann.distance_matrix(
            feature_matrices=feature_matrices,
            k=grassmann_k,
            method=grassmann_method,
        )

    if manifold_key == "riemann":
        return Riemann.distance_matrix(
            precomputed_feature_matrices=feature_matrices,
            metric=riemann_metric,
            regularization=riemann_regularization,
        )

    raise ValueError("manifold must be one of: ['grassmann', 'riemann']")


def _evaluate_kmeans(
    items: Sequence[np.ndarray],
    dist_matrix: np.ndarray,
    k_range: Sequence[int],
    manifold: str,
    metric: str = "log-euclidean",
) -> Optional[Dict[str, float]]:
    best_result = None

    for k in k_range:
        try:
            if manifold == "riemann" and metric == "log-euclidean":
                labels, _ = log_euclidean_kmeans(np.array(items), int(k))
            else:
                labels, _ = riemannian_kmeans(items, int(k), manifold=manifold)
        except Exception as e:
            logger.debug(f"True K-Means failed for k={k} on {manifold}: {e}")
            continue

        silhouette = _safe_silhouette(dist_matrix, labels)
        davies_bouldin = _safe_davies_bouldin(dist_matrix, labels)
        if silhouette is None or davies_bouldin is None:
            continue

        result = {
            "Optimal k": int(k),
            "Num Clusters": int(len(np.unique(labels))),
            "Silhouette": silhouette,
            "Davies-Bouldin": davies_bouldin,
            "labels": labels,
        }

        if best_result is None or result["Silhouette"] > best_result["Silhouette"]:
            best_result = result

    return best_result


def _evaluate_spectral(
    dist_matrix: np.ndarray,
    affinity: np.ndarray,
    k_range: Sequence[int],
) -> Optional[Dict[str, float]]:
    best_result = None
    eigengap_k, eigengap_value = _spectral_eigengap_heuristic(affinity, k_range)

    for k in k_range:
        try:
            model = SpectralClustering(
                n_clusters=int(k),
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            labels = model.fit_predict(affinity)
        except Exception:
            continue

        silhouette = _safe_silhouette(dist_matrix, labels)
        davies_bouldin = _safe_davies_bouldin(dist_matrix, labels)
        if silhouette is None or davies_bouldin is None:
            continue

        result = {
            "Optimal k": int(k),
            "Num Clusters": int(len(np.unique(labels))),
            "Silhouette": silhouette,
            "Davies-Bouldin": davies_bouldin,
            "Eigengap Suggested k": eigengap_k,
            "Eigengap": eigengap_value,
            "labels": labels,
        }

        if best_result is None or result["Silhouette"] > best_result["Silhouette"]:
            best_result = result

    return best_result


def _evaluate_kmedoids(
    dist_matrix: np.ndarray,
    k_range: Sequence[int],
) -> Optional[Dict[str, float]]:
    best_result = None

    for k in k_range:
        try:
            model = kmedoids.KMedoids(
                n_clusters=int(k),
                metric="precomputed",
                random_state=42,
            )
            labels = model.fit_predict(dist_matrix)
        except Exception:
            continue

        silhouette = _safe_silhouette(dist_matrix, labels)
        davies_bouldin = _safe_davies_bouldin(dist_matrix, labels)
        if silhouette is None or davies_bouldin is None:
            continue

        medoid_indices = getattr(model, "medoid_indices_", [])
        inertia = float(sum(dist_matrix[i, medoid_indices[labels[i]]] for i in range(len(labels))))

        result = {
            "Optimal k": int(k),
            "Num Clusters": int(len(np.unique(labels))),
            "Inertia": inertia,
            "Silhouette": silhouette,
            "Davies-Bouldin": davies_bouldin,
            "labels": labels,
        }

        if best_result is None or result["Silhouette"] > best_result["Silhouette"]:
            best_result = result

    return best_result


def _evaluate_dbscan(
    dist_matrix: np.ndarray,
    eps_values: Sequence[float],
    min_samples_values: Sequence[int],
) -> Optional[Dict[str, float]]:
    best_result = None

    for min_samples in min_samples_values:
        for eps in eps_values:
            try:
                model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed")
                labels = model.fit_predict(dist_matrix)
            except Exception:
                continue

            n_clusters = _num_valid_clusters(labels)
            noise_ratio = float(np.mean(labels == -1))
            if n_clusters < 2 or noise_ratio > 0.50:
                continue

            silhouette = _safe_silhouette(dist_matrix, labels)
            davies_bouldin = _safe_davies_bouldin(dist_matrix, labels)
            if silhouette is None or davies_bouldin is None:
                continue

            result = {
                "Optimal eps": float(eps),
                "Optimal min_samples": int(min_samples),
                "Num Clusters": int(n_clusters),
                "Noise Ratio": noise_ratio,
                "Silhouette": silhouette,
                "Davies-Bouldin": davies_bouldin,
                "labels": labels,
            }

            if best_result is None or result["Silhouette"] > best_result["Silhouette"]:
                best_result = result

    return best_result


def _evaluate_affinity_propagation(
    dist_matrix: np.ndarray,
    affinity: np.ndarray,
    preference_quantiles: Sequence[float],
    damping_values: Sequence[float],
) -> Optional[Dict[str, float]]:
    best_result = None
    off_diag = affinity[np.triu_indices_from(affinity, k=1)]
    off_diag = off_diag[np.isfinite(off_diag)]

    if off_diag.size == 0:
        off_diag = np.array([0.5], dtype=float)

    for quantile in preference_quantiles:
        preference = float(np.quantile(off_diag, quantile))

        for damping in damping_values:
            try:
                model = AffinityPropagation(
                    affinity="precomputed",
                    preference=preference,
                    damping=float(damping),
                    random_state=42,
                    max_iter=500,
                    convergence_iter=30,
                )
                labels = model.fit_predict(affinity)
            except Exception:
                continue

            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                continue

            silhouette = _safe_silhouette(dist_matrix, labels)
            davies_bouldin = _safe_davies_bouldin(dist_matrix, labels)
            if silhouette is None or davies_bouldin is None:
                continue

            result = {
                "Preference Quantile": float(quantile),
                "Preference": preference,
                "Damping": float(damping),
                "Num Clusters": int(n_clusters),
                "Silhouette": silhouette,
                "Davies-Bouldin": davies_bouldin,
                "labels": labels,
            }

            if best_result is None or result["Silhouette"] > best_result["Silhouette"]:
                best_result = result

    return best_result

# ==============================================================================
# PLOTTING AND SAVING
# ==============================================================================

def _plot_top_results(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    annotation_columns: Sequence[str],
    top_n: int = 20,
) -> None:
    top_df = results_df.head(top_n).copy()
    if top_df.empty:
        return

    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_df,
        x="Silhouette",
        y="Combination",
        hue="Combination",
        palette="viridis",
        legend=False,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Silhouette Score (Higher is Better)")
    plt.ylabel("Feature Combination")

    for index, (_, row) in enumerate(top_df.iterrows()):
        parts = []
        for column in annotation_columns:
            if column not in row or pd.isna(row[column]):
                continue

            value = row[column]
            if isinstance(value, (int, np.integer)):
                parts.append(f"{column}={int(value)}")
            elif isinstance(value, (float, np.floating)):
                parts.append(f"{column}={float(value):.3f}")
            else:
                parts.append(f"{column}={value}")

        if parts:
            plt.text(
                float(row["Silhouette"]) + 0.005,
                index,
                " | ".join(parts),
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_ranked_results(
    results: List[Dict[str, float]],
    output_dir: Path,
    method_name: str,
    title: str,
    annotation_columns: Sequence[str],
) -> None:
    if not results:
        logger.warning(f"No valid results produced for {method_name}.")
        return

    results_df = pd.DataFrame(results).sort_values(by="Silhouette", ascending=False)
    valid_models = results_df[results_df["Silhouette"] > 0.2]
    if valid_models.empty:
        logger.warning(
            f"No {method_name} combinations met the silhouette threshold. Falling back to full ranking."
        )
        valid_models = results_df

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ablation_results_full_{method_name}.csv"
    valid_models.to_csv(csv_path, index=False)
    logger.success(f"Saved {method_name} results to {csv_path}")

    plot_path = output_dir / f"ablation_study_top20_{method_name}.png"
    _plot_top_results(
        valid_models,
        output_path=plot_path,
        title=title,
        annotation_columns=annotation_columns,
    )
    logger.success(f"Saved {method_name} top-result plot to {plot_path}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def evaluate_non_euclidean_combinations(
    df: pl.DataFrame,
    manifold: str,
    features: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: int = 20,
    eps_values: Optional[Sequence[float]] = None,
    min_samples_values: Optional[Sequence[int]] = None,
    ap_preference_quantiles: Optional[Sequence[float]] = None,
    ap_damping_values: Optional[Sequence[float]] = None,
    normalize_features: bool = True,
    correlation_threshold: float = 0.85,
    output_base_dir: str = "figures/qm9/non_euclidean",
    grassmann_k: int = 3,
    grassmann_method: str = "svd",
    riemann_metric: str = "log-euclidean",
    riemann_regularization: float = 1e-6,
    qm9_seed: int = 40,
    qm9_invariant: bool = True,
) -> None:
    if features is None:
        features = list(INVARIANT_FEATURES_QM9)

    if eps_values is None:
        eps_values = DEFAULT_DBSCAN_EPS
    if min_samples_values is None:
        min_samples_values = DEFAULT_DBSCAN_MIN_SAMPLES
    if ap_preference_quantiles is None:
        ap_preference_quantiles = DEFAULT_AP_PREFERENCE_QUANTILES
    if ap_damping_values is None:
        ap_damping_values = DEFAULT_AP_DAMPING_VALUES

    k_range = [k for k in range(k_min, k_max + 1) if 2 <= k < len(df)]
    if not k_range:
        raise ValueError("No valid k values available for clustering.")

    logger.info(f"--- {manifold.upper()} NON-EUCLIDEAN EVALUATION ---")
    logger.info("Building QM9 frames once for all feature combinations.")
    frames = _build_qm9_frames_from_df(df, seed=qm9_seed, invariant=qm9_invariant)

    if len(frames) != len(df):
        raise ValueError(
            "Number of embedded QM9 frames does not match dataframe rows. "
            "Please inspect failed molecule embeddings before running the evaluation pipeline."
        )

    logger.info("Screening correlated invariant features before combination search.")
    screened_features, dropped_features = _screen_correlated_invariant_features(
        df,
        features,
        threshold=correlation_threshold,
    )
    logger.info(f"Dropped highly correlated features: {dropped_features}")
    logger.info(f"Retained features: {screened_features}")

    combinations_to_test = _iter_invariant_combinations(screened_features)
    logger.info(
        f"Evaluating {len(combinations_to_test)} invariant combinations for {manifold} distances."
    )

    spectral_results: List[Dict[str, float]] = []
    dbscan_results: List[Dict[str, float]] = []
    affinity_results: List[Dict[str, float]] = []
    kmedoids_results: List[Dict[str, float]] = []
    kmeans_results: List[Dict[str, float]] = []

    combo_iter = tqdm(combinations_to_test, desc=f"{manifold.title()} combinations", unit="combo")
    for feature_keys in combo_iter:
        combo_name = " + ".join(feature_keys)

        feature_matrices = build_invariant_matrix(
            df,
            aggregated=False,
            feature_keys=feature_keys,
            frames=frames,
        )

        if normalize_features:
            feature_matrices = _normalize_feature_matrices(feature_matrices)

        # 1. Retrieve the actual manifold representations for True K-Means
        if manifold.lower() == "grassmann":
            items = Grassmann._get_uk_bases(
                frames=None,
                k=grassmann_k,
                method=grassmann_method,
                precomputed_feature_matrices=feature_matrices
            )
        elif manifold.lower() == "riemann":
            items = Riemann._get_spd_matrices(
                frames=None,
                regularization=riemann_regularization,
                precomputed_feature_matrices=feature_matrices
            )
        else:
            items = []

        # 2. Compute the shared distance matrix for K-Medoids, DBSCAN, Spectral, etc.
        dist_matrix = _compute_manifold_distance_matrix(
            manifold=manifold,
            feature_matrices=feature_matrices,
            grassmann_k=grassmann_k,
            grassmann_method=grassmann_method,
            riemann_metric=riemann_metric,
            riemann_regularization=riemann_regularization,
        )
        affinity_matrix, sigma = _gaussian_affinity_from_distance(dist_matrix)

        # --- TRUE K-MEANS ---
        kmeans_result = _evaluate_kmeans(items, dist_matrix, k_range, manifold.lower(), riemann_metric)
        if kmeans_result is not None:
            labels = kmeans_result.pop("labels")
            _, avg_coherence = get_overall_chemical_coherence(df, labels)
            kmeans_results.append(
                {
                    "Combination": combo_name,
                    "Feature Count": len(feature_keys),
                    "Overall Chemical Coherence": avg_coherence,
                    **kmeans_result,
                }
            )

        # --- SPECTRAL ---
        spectral_result = _evaluate_spectral(dist_matrix, affinity_matrix, k_range)
        if spectral_result is not None:
            labels = spectral_result.pop("labels")
            _, avg_coherence = get_overall_chemical_coherence(df, labels)
            spectral_results.append(
                {
                    "Combination": combo_name,
                    "Feature Count": len(feature_keys),
                    "Sigma": sigma,
                    "Overall Chemical Coherence": avg_coherence,
                    **spectral_result,
                }
            )

        # --- DBSCAN ---
        dbscan_result = _evaluate_dbscan(dist_matrix, eps_values, min_samples_values)
        if dbscan_result is not None:
            labels = dbscan_result.pop("labels")
            _, avg_coherence = get_overall_chemical_coherence(df, labels)
            dbscan_results.append(
                {
                    "Combination": combo_name,
                    "Feature Count": len(feature_keys),
                    "Overall Chemical Coherence": avg_coherence,
                    **dbscan_result,
                }
            )

        # --- AFFINITY PROPAGATION ---
        affinity_result = _evaluate_affinity_propagation(
            dist_matrix,
            affinity_matrix,
            preference_quantiles=ap_preference_quantiles,
            damping_values=ap_damping_values,
        )
        if affinity_result is not None:
            labels = affinity_result.pop("labels")
            _, avg_coherence = get_overall_chemical_coherence(df, labels)
            affinity_results.append(
                {
                    "Combination": combo_name,
                    "Feature Count": len(feature_keys),
                    "Sigma": sigma,
                    "Overall Chemical Coherence": avg_coherence,
                    **affinity_result,
                }
            )

        # --- K-MEDOIDS ---
        kmedoids_result = _evaluate_kmedoids(dist_matrix, k_range)
        if kmedoids_result is not None:
            labels = kmedoids_result.pop("labels")
            _, avg_coherence = get_overall_chemical_coherence(df, labels)
            kmedoids_results.append(
                {
                    "Combination": combo_name,
                    "Feature Count": len(feature_keys),
                    "Overall Chemical Coherence": avg_coherence,
                    **kmedoids_result,
                }
            )

    manifold_output_dir = Path(output_base_dir) / manifold.lower() / "invariant_features"

    _save_ranked_results(
        kmeans_results,
        output_dir=manifold_output_dir / "kmeans",
        method_name="kmeans",
        title=f"Top Feature Combinations by Silhouette Score\n(True K-Means, {manifold.title()})",
        annotation_columns=["Optimal k", "Davies-Bouldin"],
    )
    _save_ranked_results(
        spectral_results,
        output_dir=manifold_output_dir / "spectral",
        method_name="spectral",
        title=f"Top Feature Combinations by Silhouette Score\n(Spectral, {manifold.title()})",
        annotation_columns=["Optimal k", "Eigengap Suggested k", "Eigengap", "Davies-Bouldin"],
    )
    _save_ranked_results(
        dbscan_results,
        output_dir=manifold_output_dir / "dbscan",
        method_name="dbscan",
        title=f"Top Feature Combinations by Silhouette Score\n(DBSCAN, {manifold.title()})",
        annotation_columns=["Optimal eps", "Optimal min_samples", "Davies-Bouldin", "Noise Ratio"],
    )
    _save_ranked_results(
        affinity_results,
        output_dir=manifold_output_dir / "affinity_propagation",
        method_name="affinity_propagation",
        title=f"Top Feature Combinations by Silhouette Score\n(Affinity Propagation, {manifold.title()})",
        annotation_columns=["Preference Quantile", "Damping", "Davies-Bouldin", "Sigma"],
    )
    _save_ranked_results(
        kmedoids_results,
        output_dir=manifold_output_dir / "kmedoids",
        method_name="kmedoids",
        title=f"Top Feature Combinations by Silhouette Score\n(K-Medoids, {manifold.title()})",
        annotation_columns=["Optimal k", "Davies-Bouldin", "Inertia"],
    )



if __name__ == "__main__":
    qm9 = QM9Dataset(
        limit=400,
        stratify_by=["num_atoms", "gap"],
        sampling_strategy="stratified",
    )
    df = qm9.load()

    output_base_dir = "figures/qm9/non_euclidean"

    evaluate_non_euclidean_combinations(
        df,
        manifold="grassmann",
        k_min=2,
        k_max=20,
        output_base_dir=output_base_dir,
    )

    evaluate_non_euclidean_combinations(
        df,
        manifold="riemann",
        k_min=2,
        k_max=20,
        output_base_dir=output_base_dir,
    )
