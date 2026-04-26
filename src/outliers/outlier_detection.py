import polars as pl
import numpy as np
from hdbscan import HDBSCAN

from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def evaluate_hdbscan(dist_matrix: np.ndarray, mcs: int, ms: int, method: str = 'eom') -> dict:
    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric='precomputed',
        cluster_selection_method=method
    )
    
    labels = clusterer.fit_predict(dist_matrix)
    outlier_scores = clusterer.outlier_scores_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = np.mean(labels == -1)
    
    if n_clusters > 0:
        avg_persistence = np.mean(clusterer.cluster_persistence_)
    else:
        avg_persistence = 0.0
    
    return {
        "mcs": mcs,
        "ms": ms,
        "clusters": n_clusters,
        "noise": noise_frac,
        "persistence": avg_persistence,
        "labels": labels,
        "outlier_scores": outlier_scores
    }

def hdbscan_outliers(
    df: pl.DataFrame,
    dist_matrix: np.ndarray,
) -> pl.DataFrame:
    dist_matrix = dist_matrix.astype(np.float64)
    n_samples = dist_matrix.shape[0]

    # 1. Define the search space boundaries
    # Don't let the max cluster size exceed 10% of the data, capped at 500
    max_mcs = min(1000, max(20, n_samples // 10)) 
    min_mcs = 2
    
    # 2. Generate ~12 min_cluster_size candidates geometrically
    # This samples densely at lower values and sparsely at higher values
    raw_mcs = np.geomspace(min_mcs, max_mcs, num=30)
    mcs_candidates = sorted(list(set(raw_mcs.astype(int))))

    # 3. Build the combinations
    combinations = []
    for mcs in mcs_candidates:
        ms_candidates = [
            1,
            max(2, int(mcs * 0.25)),
            max(3, int(mcs * 0.50)),
            max(4, int(mcs * 0.75)),
            mcs
        ]
        # Deduplicate and add to combinations
        for ms in sorted(list(set(ms_candidates))):
            combinations.append((mcs, ms))

    print(f"Starting HDBSCAN Auto-Tuning over {len(combinations)} combinations for N={n_samples}...")

    results = []
    with tqdm(total=len(combinations), desc="🔍 Evaluating HDBSCAN", unit="cfg") as pbar:
        for mcs, ms in combinations:
            res = evaluate_hdbscan(dist_matrix, mcs, ms)
            
            # Update bar description with live stats
            pbar.set_postfix({
                "mcs": mcs, 
                "ms": ms, 
                "clusters": res["clusters"]
            })
            
            # Logic: Only keep results that meet your quality thresholds
            # Adjust these based on your specific QM9 distribution needs
            if res["noise"] <= 0.25:
                results.append(res)
            
            pbar.update(1)

    # 5. Select the best parameters
    # Primary: Maximize persistence (cluster stability)
    # Secondary: Minimize noise fraction
    results_sorted = sorted(results, key=lambda x: (-x["persistence"], x["noise"]))
    best_run = results_sorted[0]

    best_mcs = best_run["mcs"]
    best_ms = best_run["ms"]
    best_labels = best_run["labels"]
    best_scores = best_run["outlier_scores"]

    # 6. Logging and Summary
    unique_labels, counts = np.unique(best_labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    
    print("\n--- Auto-Tuning Results ---")
    print(f"Selected params: min_cluster_size={best_mcs}, min_samples={best_ms}")
    print(f"Metrics: clusters={best_run['clusters']}, noise={best_run['noise']:.4f}, persistence={best_run['persistence']:.3f}")
    print(f"HDBSCAN — {len(unique_labels)} distinct labels: {label_summary}")

    return df.with_columns([
        pl.Series("hdbscan_label", best_labels),
        pl.Series("hdbscan_score", best_scores)
    ])
def lof_outliers(df: pl.DataFrame, dist_matrix: np.ndarray) -> pl.DataFrame:
    lof = LocalOutlierFactor(n_neighbors=20, metric="precomputed")
    labels = lof.fit_predict(dist_matrix)
    scores = -lof.negative_outlier_factor_ 

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    print(f"LOF — {len(unique_labels)} distinct: {label_summary}")

    return df.with_columns([
        pl.Series("lof_label", labels),
        pl.Series("lof_score", scores),
    ])

def knn_outliers(df: pl.DataFrame, dist_matrix: np.ndarray, k: int = 20) -> pl.DataFrame:
    """
    Detects outliers based on the average distance to the k-nearest neighbors.
    Higher scores indicate a higher likelihood of being an outlier.
    """
    knn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    knn.fit(dist_matrix)
    
    distances, _ = knn.kneighbors(dist_matrix)
    
    scores = np.mean(distances, axis=1)
    
    threshold = np.percentile(scores, 90)
    labels = np.where(scores > threshold, -1, 1)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    print(f"k-NN — {len(unique_labels)} distinct: {label_summary}")

    return df.with_columns([
        pl.Series("knn_label", labels),
        pl.Series("knn_score", scores),
    ])

