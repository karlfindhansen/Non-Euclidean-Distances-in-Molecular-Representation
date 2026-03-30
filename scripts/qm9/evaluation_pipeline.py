import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from umap import UMAP
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram

from src.datasets import QM9Dataset

QM9_PROPS = ["logp", "tpsa", "mol_weight", "homo", "lumo"]

# ---------------------------------------------------------------------------
# 1. QM9 Coherence & Evaluation Functions
# ---------------------------------------------------------------------------

def _functional_consistency(values: List) -> float:
    """Compute the max normalized frequency of functional_groups in a cluster."""
    if not values:
        return 0.0
    normalized = []
    for v in values:
        if v is None:
            normalized.append(None)
        elif isinstance(v, list):
            normalized.append(tuple(v))
        else:
            normalized.append(v)
    counts = Counter(normalized)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(c / total for c in counts.values())

def get_overall_qm9_coherence(df: pl.DataFrame, labels: np.ndarray) -> Tuple[Dict[str, float], float]:
    """Compute functional consistency and property cohesion for QM9 clusters."""
    if len(labels) != len(df):
        raise ValueError("labels length must match dataframe length")

    df = df.with_columns(pl.Series(name="labels", values=labels))

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.select(QM9_PROPS).to_numpy())
    df_scaled = df.with_columns(
        [pl.Series(name, scaled_values[:, i]) for i, name in enumerate(QM9_PROPS)]
    )

    results = {f: 0.0 for f in QM9_PROPS}
    results["functional_consistency"] = 0.0

    cluster_sizes = df_scaled.group_by("labels").agg(pl.len().alias("count"))
    total_count = float(cluster_sizes["count"].sum())

    for label in df_scaled["labels"].unique().to_list():
        cluster_df = df_scaled.filter(pl.col("labels") == label)
        size = cluster_df.height
        if size <= 1:
            continue

        func_score = _functional_consistency(cluster_df["functional_groups"].to_list())

        for prop in QM9_PROPS:
            # fill_null(0.0) handles single-item edge cases if they sneak through
            std = float(cluster_df.select(pl.col(prop).std().fill_null(0.0)).item())
            cohesion = 1.0 / (1.0 + std)
            results[prop] += cohesion * size

        results["functional_consistency"] += func_score * size

    if total_count > 0:
        for key in results:
            results[key] /= total_count

    average_coherence = float(np.mean(list(results.values())))
    return results, average_coherence

# ---------------------------------------------------------------------------
# 2. General Clustering & Reduction Utilities
# ---------------------------------------------------------------------------

def get_reducers():
    return {
        'tsne': TSNE(n_components=3), 
        'pca': PCA(n_components=3), 
        'umap': UMAP(n_components=3), 
        'isomap': Isomap(n_components=3)
    }

def hierachial_clustering(dist_matrix, n_clusters, linkage='complete'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='precomputed')
    return model.fit_predict(dist_matrix)

def _soap_embeddings(df: pl.DataFrame):
    soap_array = np.array(df["soap_embedding"].to_list())
    reducers = get_reducers()
    return {
        "soap_pca": reducers["pca"].fit_transform(soap_array),
        "soap_tsne": reducers["tsne"].fit_transform(soap_array),
        "soap_umap": reducers["umap"].fit_transform(soap_array),
        "soap_isomap": reducers["isomap"].fit_transform(soap_array),
    }

def _soap_distance_matrices(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        name: squareform(pdist(emb, metric="euclidean"))
        for name, emb in embeddings.items()
    }

def _invert_db_score(db_score: float) -> float:
    return 1.0 / (1.0 + db_score)

def _normalized(values: np.ndarray) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    if np.isclose(vmin, vmax):
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)

def _kneedle_elbow(ks: np.ndarray, values: np.ndarray) -> int:
    if len(ks) < 3:
        return int(ks[-1])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    distances = np.abs(y - x)
    idx = int(np.argmax(distances))
    return int(ks[idx])

# ---------------------------------------------------------------------------
# 3. Plotting Sub-Routines
# ---------------------------------------------------------------------------

def plot_evaluation(res, out_path: Optional[Path] = None, title: str = "Clustering Evaluation", dpi: int = 300):
    n = len(res["sil"])
    x_range = np.arange(2, n + 2)
    
    # Base metrics present in both Hierarchical and KMeans
    metrics = [
        ('sil', 'Silhouette Score', 'max'),
        ('ch', 'Calinski-Harabasz Index', 'max'),
        ('db', 'Davies-Bouldin Index', 'min'),
    ]
    
    # Dynamically add Inertia if it was passed in the results dictionary (KMeans only)
    if "inertia" in res:
        metrics.append(('inertia', 'Inertia (SSE)', 'elbow'))

    num_metrics = len(metrics)
    
    # Dynamically scale the width of the figure based on the number of plots (3 or 4)
    plt.figure(figsize=(6 * num_metrics, 5))

    for i, (key, m_title, goal) in enumerate(metrics, 1):
        plt.subplot(1, num_metrics, i)
        data = np.array(res[key][:n])
        plt.plot(x_range, data, label=m_title)
        
        # Determine the best point based on the metric's specific goal
        if goal == 'elbow':
            # For inertia, the absolute minimum is always the highest K. 
            # We use your elbow detection function instead to find the true "best" K.
            best_x = _kneedle_elbow(x_range, data)
            best_idx = np.where(x_range == best_x)[0][0]
            best_y = data[best_idx]
            annot_text = f'Elbow: {best_x}'
        else:
            best_idx = np.argmax(data) if goal == 'max' else np.argmin(data)
            best_x, best_y = x_range[best_idx], data[best_idx]
            annot_text = f'Best: {best_x}'

        # Add the point and the text
        plt.scatter(best_x, best_y, color='red', zorder=5)
        plt.annotate(
            annot_text, 
            xy=(best_x, best_y), 
            xytext=(best_x + 1, best_y + (np.max(data) - np.min(data)) * 0.05), # Offset text slightly
            arrowprops=dict(arrowstyle='->', color='black'), 
            fontsize=10, 
            fontweight='bold'
        )
        
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title(m_title)
        plt.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
        logger.info(f"Saved evaluation plot to {out_path}")
    plt.close()

def _plot_radar_series(ax, labels, values, series_label=None):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate([values, values[:1]])
    angles = np.concatenate([angles, angles[:1]])
    ax.plot(angles, values, linewidth=2, label=series_label)
    ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])

def _plot_radar_pair(title, left, right, out_path: Optional[Path] = None, dpi: int = 300):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"polar": True})
    fig.suptitle(title, fontsize=14)
    for ax, (name, labels, series) in zip(axes, [left, right]):
        for series_label, values in series.items():
            _plot_radar_series(ax, labels, values, series_label=series_label)
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()

def _plot_radar_quad(title, quad_items, out_path: Optional[Path] = None, dpi: int = 300):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"polar": True})
    axes = axes.ravel()
    fig.suptitle(title, fontsize=14)
    for ax, (name, labels, series) in zip(axes, quad_items):
        for series_label, values in series.items():
            _plot_radar_series(ax, labels, values, series_label=series_label)
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()

# ---------------------------------------------------------------------------
# 4. Radar & Evaluator Data Builders
# ---------------------------------------------------------------------------

def _qm9_feature_order() -> list:
    return QM9_PROPS + ["functional_consistency"]

def _build_eval_radar_values(sil: float, ch: float, db_inv: float) -> Tuple[list, np.ndarray]:
    return ["silhouette", "calinski_harabasz", "davies_bouldin_inv"], np.array([sil, ch, db_inv], dtype=float)

def _build_qm9_radar_values(chem_scores: Dict[str, float]) -> Tuple[list, np.ndarray]:
    labels = _qm9_feature_order()
    values = [chem_scores[k] for k in _qm9_feature_order()]
    return labels, np.array(values, dtype=float)

def _kmeans_scores(embeddings: np.ndarray, k_range: range):
    silhouettes, ch_scores, db_scores, inertias = [], [], [], []
    labels_by_k = {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(embeddings)
        labels_by_k[k] = labels
        silhouettes.append(silhouette_score(embeddings, labels))
        ch_scores.append(calinski_harabasz_score(embeddings, labels))
        db_scores.append(davies_bouldin_score(embeddings, labels))
        inertias.append(model.inertia_)
    return np.array(silhouettes), np.array(ch_scores), np.array(db_scores), np.array(inertias), labels_by_k

def _hierarchical_eval_curves(embeddings: np.ndarray, dist_matrix: np.ndarray, linkage: str, k_range: range):
    res = {"sil": [], "ch": [], "db": []}
    labels_by_k = {}
    for k in k_range:
        labels = hierachial_clustering(dist_matrix, k, linkage=linkage)
        labels_by_k[k] = labels
        res["sil"].append(silhouette_score(embeddings, labels))
        res["ch"].append(calinski_harabasz_score(embeddings, labels))
        res["db"].append(davies_bouldin_score(embeddings, labels))
    return res, labels_by_k

# ---------------------------------------------------------------------------
# 5. Main Plotting Master Functions
# ---------------------------------------------------------------------------

def plot_hierarchical_radar_plots(df: pl.DataFrame, k_min: int = 2, k_max: int = 20, output_dir: Path = Path("figures/qm9/clustering/hierarchical/soap_reduced")):
    embeddings = _soap_embeddings(df)
    dist_mats = _soap_distance_matrices(embeddings)
    k_range = range(k_min, k_max + 1)
    linkage_outputs = {}

    for linkage in ["average", "complete"]:
        logger.info(f"Computing hierarchical metrics for linkage='{linkage}'...")
        eval_series, chem_series, eval_triplets, k_by_name = {}, {}, {}, {}

        for name, emb in embeddings.items():
            res, labels_by_k = _hierarchical_eval_curves(emb, dist_mats[name], linkage, k_range)
            db_array = np.array(res["db"])
            best_idx = int(np.argmin(db_array))
            k = list(k_range)[best_idx]
            labels = labels_by_k[k]
            
            plot_evaluation(
                res, out_path=output_dir / f"evaluation/{name}_{linkage}_evaluation.png",
                title=f"{name} ({linkage}) - evaluation"
            )

            eval_triplets[name] = (
                silhouette_score(emb, labels), 
                calinski_harabasz_score(emb, labels), 
                _invert_db_score(db_array[best_idx])
            )
            k_by_name[name] = k

            chem_scores, _ = get_overall_qm9_coherence(df, labels)
            chem_labels, chem_values = _build_qm9_radar_values(chem_scores)
            chem_series[f"{name} (k={k})"] = chem_values

        sils, chs, dbs = (np.array([v[i] for v in eval_triplets.values()]) for i in range(3))
        sil_norm, ch_norm, db_norm = _normalized(sils), _normalized(chs), _normalized(dbs)

        for (name, _), s_n, c_n, d_n in zip(embeddings.items(), sil_norm, ch_norm, db_norm):
            eval_labels, eval_values = _build_eval_radar_values(float(s_n), float(c_n), float(d_n))
            eval_series[f"{name} (k={k_by_name[name]})"] = eval_values

        linkage_outputs[linkage] = {
            "eval": (f"Hierarchical {linkage} - metrics", eval_labels, eval_series),
            "chem": (f"Hierarchical {linkage} - chemical", chem_labels, chem_series),
        }

    _plot_radar_quad(
        "Hierarchical - complete vs average",
        [linkage_outputs["complete"]["eval"], linkage_outputs["complete"]["chem"],
         linkage_outputs["average"]["eval"], linkage_outputs["average"]["chem"]],
        out_path=output_dir / "hierarchical_radar_quad.png",
    )

def plot_kmeans_radar_plots(df: pl.DataFrame, k_min: int = 2, k_max: int = 20, output_dir: Path = Path("figures/qm9/clustering/kmeans/soap_reduced")):
    embeddings = _soap_embeddings(df)
    k_range = range(k_min, k_max + 1)
    eval_series, chem_series, eval_triplets = {}, {}, {}
    chem_labels, eval_labels = None, None

    for name, emb in embeddings.items():
        logger.info(f"Computing kmeans metrics for {name}...")
        silhouettes, ch_scores, db_scores, inertias, labels_by_k = _kmeans_scores(emb, k_range)
        
        # ---> ADDED INERTIA HERE <---
        res = {
            "sil": silhouettes.tolist(), 
            "ch": ch_scores.tolist(), 
            "db": db_scores.tolist(),
            "inertia": inertias.tolist() 
        }
        
        plot_evaluation(
            res,
            out_path=output_dir / f"evaluation/{name}_kmeans_evaluation.png",
            title=f"{name} (kmeans) - evaluation"
        )

        k_elbow = _kneedle_elbow(np.array(list(k_range)), inertias)
        labels_in = labels_by_k[k_elbow]
        
        eval_triplets[f"{name} (inertia k={k_elbow})"] = (
            silhouette_score(emb, labels_in), 
            calinski_harabasz_score(emb, labels_in), 
            _invert_db_score(davies_bouldin_score(emb, labels_in))
        )
        
        chem_scores_in, _ = get_overall_qm9_coherence(df, labels_in)
        chem_labels, chem_values = _build_qm9_radar_values(chem_scores_in)
        chem_series[f"{name} (inertia k={k_elbow})"] = chem_values

    if eval_triplets:
        sils, chs, dbs = (np.array([v[i] for v in eval_triplets.values()]) for i in range(3))
        sil_norm, ch_norm, db_norm = _normalized(sils), _normalized(chs), _normalized(dbs)

        for (series_label, _), s_n, c_n, d_n in zip(eval_triplets.items(), sil_norm, ch_norm, db_norm):
            eval_labels, eval_values = _build_eval_radar_values(float(s_n), float(c_n), float(d_n))
            eval_series[series_label] = eval_values

    _plot_radar_pair(
        "KMeans - metrics and chemical cohesion",
        ("KMeans - evaluation metrics", eval_labels, eval_series),
        ("KMeans - chemical cohesion", chem_labels, chem_series),
        out_path=output_dir / "kmeans_radar_pair.png",
    )

def plot_hierarchical_dendrograms(df: pl.DataFrame, linkage_method: str = "average", output_dir: Path = Path("figures/qm9/clustering/hierarchical/dendrograms/soap_reduced")):
    embeddings = _soap_embeddings(df)
    dist_mats = _soap_distance_matrices(embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    fig.suptitle(f"Hierarchical Dendrograms (linkage={linkage_method})", fontsize=14)

    for ax, (name, dist_matrix) in zip(axes, dist_mats.items()):
        condensed = squareform(dist_matrix, checks=False)
        Z = scipy_linkage(condensed, method=linkage_method)
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    out_path = output_dir / f"dendrograms_{linkage_method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved dendrograms to {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# 6. Execution Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure add_soap=True so the _soap_embeddings function has data to work with!
    qm9 = QM9Dataset(
        limit=10_000, 
        sampling_strategy="stratified", 
        stratify_by=["num_atoms", "gap"]
    )
    df = qm9.load()
    logger.info(f"Loaded QM9 with {len(df)} molecules.")

    # Paths for QM9 figures
    hierarchical_dir = Path("figures/qm9/clustering/hierarchical/soap_reduced")
    dendrogram_dir = hierarchical_dir / "dendrograms"
    kmeans_dir = Path("figures/qm9/clustering/kmeans/soap_reduced")

    logger.info("Generating hierarchical radar plots for QM9...")
    plot_hierarchical_radar_plots(df, k_min=2, k_max=20, output_dir=hierarchical_dir)
    
    logger.info("Generating kmeans radar plots for QM9...")
    plot_kmeans_radar_plots(df, k_min=2, k_max=20, output_dir=kmeans_dir)
    
    logger.info("Generating hierarchical dendrograms for QM9...")
    plot_hierarchical_dendrograms(df, linkage_method="average", output_dir=dendrogram_dir)
    plot_hierarchical_dendrograms(df, linkage_method="complete", output_dir=dendrogram_dir)