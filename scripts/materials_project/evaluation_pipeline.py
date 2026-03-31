from typing import Optional, Dict
from pathlib import Path

import hashlib
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Element
from loguru import logger
from kmedoids import KMedoids

from src.datasets import MaterialsProject, QM9Dataset
from src.non_euclidean import _compute_invariant_feature_matrix as invariant_matrix
from src.non_euclidean import _compute_soap_feature_matrices as soap_matrix
from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology
from src.helper_functions import create_chemiscope_viewer


def plot_evaluation(
    res,
    out_path: Optional[Path] = None,
    title: str = "Clustering Evaluation",
    dpi: int = 300,
):
    n = len(res["sil"])
    x_range = np.arange(2, n + 2)
    
    # Define metrics and whether we want the max or min
    metrics = [
        ('sil', 'Silhouette Score', 'max'),
        ('ch', 'Calinski-Harabasz Index', 'max'),
        ('db', 'Davies-Bouldin Index', 'min'),
    ]

    plt.figure(figsize=(20, 5))

    for i, (key, title, goal) in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        data = np.array(res[key][:n])
        plt.plot(x_range, data, label=title)
        
        # Find the best index based on the goal (max or min)
        if goal == 'max':
            best_idx = np.argmax(data)
        else:
            best_idx = np.argmin(data)
            
        best_x = x_range[best_idx]
        best_y = data[best_idx]

        # Add the point and the text
        plt.scatter(best_x, best_y, color='red', zorder=5)
        plt.annotate(
            f'Best: {best_x}', 
            xy=(best_x, best_y), 
            xytext=(best_x + 1, best_y),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10,
            fontweight='bold'
        )

        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
        logger.info(f"Saved evaluation plot to {out_path}")
    plt.close()

def run_evaluation(dist_matrix, linkage='complete'):
    res = {'sil': [], 'ch': [], 'db': []}
    for i in tqdm(range(2, 100), desc='Clustering'):
        labels = hierachial_clustering(dist_matrix, i, linkage=linkage)
        sil, ch, db = evaluation(dist_matrix, labels)
        res['sil'].append(sil)
        res['ch'].append(ch)
        res['db'].append(db)
        
    plot_evaluation(res)
    return res

def evaluation(dist_matrix, labels):
    sil = silhouette_score(dist_matrix, labels, metric='precomputed')
    ch = calinski_harabasz_score(dist_matrix, labels)
    db = davies_bouldin_score(dist_matrix, labels)
    return sil, ch, db

def get_overall_chemical_coherence(df, labels):

    df = df.with_columns(pl.Series(name='labels', values=labels))

    continuous_features = [
        'band_gap',
        'density',
        'energy_per_atom',
        'formation_energy_per_atom',
        'volume',
        'energy_above_hull'
    ]

    discrete_feature = "is_metal"

    # -------------------------
    # 1. Standardize continuous features
    # -------------------------

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.select(continuous_features).to_numpy())

    df_scaled = df.with_columns(
        [
            pl.Series(name, scaled_values[:, i])
            for i, name in enumerate(continuous_features)
        ]
    )

    # -------------------------
    # 2. Compute cluster sizes
    # -------------------------

    cluster_sizes = df_scaled.group_by("labels").agg(pl.len().alias("count"))

    # -------------------------
    # 3. Within-cluster std for continuous features
    # -------------------------

    cluster_std = (
        df_scaled
        .group_by("labels")
        .agg([
            pl.col(f).std().fill_null(0.0).alias(f) 
            for f in continuous_features
        ])
    )

    # convert std → coherence
    cluster_coherence = cluster_std.with_columns(
        [(1 / (1 + pl.col(f))).alias(f) for f in continuous_features]
    )

    # -------------------------
    # 4. Discrete feature coherence
    # -------------------------

    metal_coherence = (
        df_scaled
        .group_by("labels")
        .agg(
            pl.col(discrete_feature).mean().alias("metal_fraction")
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("metal_fraction"),
                1 - pl.col("metal_fraction")
            ).alias(discrete_feature)
        )
        .select(["labels", discrete_feature])
    )

    # -------------------------
    # 5. Combine all scores
    # -------------------------

    cluster_scores = (
        cluster_coherence
        .join(metal_coherence, on="labels")
        .join(cluster_sizes, on="labels")
    )

    # -------------------------
    # 6. Weighted average across clusters
    # -------------------------

    weights = cluster_scores["count"].to_numpy()

    results = {}

    for feature in continuous_features + [discrete_feature]:

        values = cluster_scores[feature].to_numpy()
        results[feature] = float(np.average(values, weights=weights))

    # average coherence across all features
    average_coherence = float(np.mean(list(results.values())))

    return results, average_coherence

def _compute_invariant_feature_matrix_materials(
    frame: Atoms, 
    cutoff: float = 3.0, 
    aggregated: bool = False,
    feature_keys: Optional[list] = ["z", "en", "coord", "avg_neighbor_dist", "vol_per_atom"]
) -> np.ndarray:
    
    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)
    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    features = []
    vol_per_atom = frame.get_volume() / len(frame)

    # 10 Available Features : ""z", "en", "coord", "avg_neighbor_dist", "vol_per_atom", "mass", "rad", "mendeleev", "group", "row""

    for i, atom in enumerate(frame):
        z = atom.number
        el = Element.from_Z(z)
        
        # Safely extract properties
        en = el.X if getattr(el, 'X', None) else 0.0
        rad = el.atomic_radius if getattr(el, 'atomic_radius', None) else 0.0
        mass = float(el.atomic_mass)
        mendeleev = el.mendeleev_no
        group = el.group
        row = el.row
        
        coord = len(neighbors[i])
        if coord > 0:
            avg_neighbor_dist = float(np.mean(distances[i]))
        else:
            avg_neighbor_dist = 0.0

        # Pool of all 10 features
        feature_pool = {
            "z": z,
            "en": en,
            "coord": coord,
            "avg_neighbor_dist": avg_neighbor_dist,
            "vol_per_atom": vol_per_atom,
            "mass": mass,
            "rad": rad,
            "mendeleev": mendeleev,
            "group": group,
            "row": row
        }

        feat_vector = [feature_pool[k] for k in feature_keys]
        features.append(feat_vector)

    if aggregated:
        atom_matrix = np.array(features).T 
        mean_features = np.mean(atom_matrix, axis=1)
        std_features = np.std(atom_matrix, axis=1)
        return np.concatenate([mean_features, std_features])
        
    return np.array(features).T

def evaluate_invariant_combinations(df: pl.DataFrame, linkage: str = "average", k_min: int = 2, k_max: int = 20):
    """
    Tests every combination of 10 invariant features (1023 total), 
    finds the optimal k via DB score, and records the chemical coherence.
    """
    all_features = [
        "z", "en", "coord", "avg_neighbor_dist", "vol_per_atom",
        "mass", "rad", "mendeleev", "group", "row"
    ]
    
    # Generate all combinations of length 1 to 10
    combinations_to_test = []
    for r in range(1, len(all_features) + 1):
        for combo in itertools.combinations(all_features, r):
            combinations_to_test.append(list(combo))

    logger.info(f"Generated {len(combinations_to_test)} feature combinations to test.")
    
    results = []
    k_range = range(k_min, k_max + 1)

    # Wrap in tqdm for a massive progress bar
    for feature_keys in tqdm(combinations_to_test, desc="Evaluating Combinations"):
        combo_name = " + ".join(feature_keys)
        
        feature_matrix = build_invariant_matrix(df, aggregated=True, feature_keys=feature_keys)
        feature_matrix = np.array(feature_matrix)
        
        dist_matrix = squareform(pdist(feature_matrix, metric='euclidean'))
        
        best_k = None
        best_db = np.inf
        best_labels = None
        
        for k in k_range:
            labels = hierachial_clustering(dist_matrix, k, linkage=linkage)
            db = davies_bouldin_score(feature_matrix, labels)
            if db < best_db:
                best_db = db
                best_k = k
                best_labels = labels
                
        chem_scores, avg_coherence = get_overall_chemical_coherence(df, best_labels)
        
        results.append({
            "Combination": combo_name,
            "Feature Count": len(feature_keys),
            "Optimal k": best_k,
            "DB Score": best_db,
            "Overall Chemical Coherence": avg_coherence
        })

    # Sort results to find the winners
    results_df = pd.DataFrame(results).sort_values(by="Overall Chemical Coherence", ascending=False)
    
    # Save the full 1023 row dataframe to a CSV for your records
    csv_path = Path(f"figures/materials/clustering/hierarchical/ablation_results_full_{linkage}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    logger.success(f"Saved full ablation results (CSV) to {csv_path}")
    
    # Plot ONLY the top 20 for readability
    top_20 = results_df.head(20)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_20, 
        x="Overall Chemical Coherence", 
        y="Combination", 
        hue="Combination",
        palette="viridis",
        legend=False
    )
    plt.title(f"Top 20 Feature Combinations by Chemical Coherence\n(Hierarchical, linkage={linkage})", fontsize=14)
    plt.xlabel("Overall Chemical Coherence (Higher is Better)")
    plt.ylabel("Feature Combination")
    
    for index, row in enumerate(top_20.itertuples()):
        plt.text(row._5 + 0.005, index, f"Best k={row._3}", va='center', color='black', fontsize=9)

    plt.tight_layout()
    out_path = Path(f"figures/materials/clustering/hierarchical/ablation_study_top20_{linkage}.png")
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 ablation plot to {out_path}")

def build_invariant_matrix(df, cutoff: float = 3.0, aggregated: bool = False, feature_keys: Optional[list] = ["z", "en", "coord", "avg_neighbor_dist", "vol_per_atom"]) -> list:
    """
    Iterates through the materials dataframe, converts JSON structures to ASE Atoms, 
    and computes the D x N invariant feature matrix for each material.
    """
    adaptor = AseAtomsAdaptor()
    invariant_matrices = []
    
    for struct_json in df["raw_structure"]:
        # 1. Reconstruct the Pymatgen Structure
        struct = Structure.from_dict(json.loads(struct_json))
        
        # 2. Convert to ASE Atoms (this automatically preserves periodic boundary conditions)
        atoms = adaptor.get_atoms(struct)
        
        # 3. Compute the D x N invariant matrix
        matrix = _compute_invariant_feature_matrix_materials(atoms, cutoff=cutoff, aggregated=aggregated, feature_keys=feature_keys)
        
        invariant_matrices.append(matrix)
        
    return invariant_matrices

def hierachial_clustering(dist_matrix, n_clusters, linkage='complete'):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='precomputed')
    labels = hierarchical_cluster.fit_predict(dist_matrix)
    return labels

def kmedoids_clustering(dist_matrix, n_clusters):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed')
    labels = kmedoids.fit_predict(dist_matrix)
    return labels

def get_ase_frames(df: pl.DataFrame) -> list:
    """Extracts raw JSON structures from the dataframe and converts them to ASE Atoms."""
    adaptor = AseAtomsAdaptor()
    frames = []
    
    for struct_json in df["raw_structure"]:
        struct = Structure.from_dict(json.loads(struct_json))
        atoms = adaptor.get_atoms(struct)
        frames.append(atoms)
        
    return frames

def get_reducers():
    tsne = TSNE(n_components=3)
    pca = PCA(n_components=3)
    umap = UMAP(n_components=3)
    isomap = Isomap(n_components=3)
    return {'tsne': tsne, 'pca': pca, 'umap': umap, 'isomap': isomap}

def _invert_db_score(db_score: float) -> float:
    """
    Convert Davies-Bouldin score to a [0, 1] coherence-like score (higher is better).
    """
    return 1.0 / (1.0 + db_score)

def _kneedle_elbow(ks: np.ndarray, values: np.ndarray) -> int:
    """
    Simple elbow detection using maximum distance to line.
    Assumes values are decreasing (e.g., inertia).
    """
    if len(ks) < 3:
        return int(ks[-1])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    line = x
    distances = np.abs(y - line)
    idx = int(np.argmax(distances))
    return int(ks[idx])

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
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={"polar": True})
    
    fig.suptitle(title, fontsize=16, y=1.05) 
    
    for ax, (name, labels, series) in zip(axes, [left, right]):
        for series_label, values in series.items():
            _plot_radar_series(ax, labels, values, series_label=series_label)
            
        ax.set_title(name, fontsize=12, pad=25) 
        
        ax.tick_params(axis='x', pad=20, labelsize=9)
        
        ax.legend(
            loc="upper right", 
            bbox_to_anchor=(1.35, 1.15),
            fontsize=9, 
            frameon=False
        )
        
    plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {out_path}")
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
        logger.info(f"Saved plot to {out_path}")
    plt.close()

def plot_invariant_aggregated_dendrograms(
    df: pl.DataFrame,
    linkage_method: str = "average",
    output_dir: Path = Path("figures/materials/clustering/hierarchical/dendrograms/invariant"),
):
    """Plots a single, standalone dendrogram for the invariant_aggregated distance matrix."""
    dist_mats = get_distance_matrices_non_euclidean(df)
    
    if "invariant_aggregated" not in dist_mats:
        logger.error("invariant_aggregated distance matrix not found!")
        return
        
    dist_matrix = dist_mats["invariant_aggregated"]

    plt.figure(figsize=(10, 6))
    plt.title(f"Hierarchical Dendrogram - Invariant Aggregated (linkage={linkage_method})", fontsize=14)

    condensed = squareform(dist_matrix, checks=False)
    Z = scipy_linkage(condensed, method=linkage_method)
    dendrogram(Z, no_labels=True, color_threshold=None)
    
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()

    out_path = output_dir / f"dendrogram_invariant_aggregated_{linkage_method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved invariant aggregated dendrogram to {out_path}")
    plt.close()

def _soap_embeddings(df: pl.DataFrame):
    soap_array = np.array(df["soap_embedding"].to_list())
    reducers = get_reducers()
    return {
        "soap_raw": soap_array,
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

def _chemical_feature_order() -> list:
    return [
        "band_gap",
        "density",
        "energy_per_atom",
        "formation_energy_per_atom",
        "volume",
        "energy_above_hull",
        "is_metal",
    ]

def _build_eval_radar_values(sil: float, ch: float, db_inv: float) -> (list, np.ndarray):
    labels = ["silhouette", "calinski_harabasz", "davies_bouldin_inv"]
    values = [sil, ch, db_inv]
    return labels, np.array(values, dtype=float)

def _build_chem_radar_values(chem_scores: Dict[str, float]) -> (list, np.ndarray):
    labels = _chemical_feature_order()
    values = [chem_scores[k] for k in _chemical_feature_order()]
    return labels, np.array(values, dtype=float)

def _select_k_by_db(
    embeddings: np.ndarray,
    dist_matrix: np.ndarray,
    linkage: str,
    k_range: range,
) -> (int, float, np.ndarray):
    best_k = None
    best_db = np.inf
    best_labels = None
    for k in k_range:
        labels = hierachial_clustering(dist_matrix, k, linkage=linkage)
        db = davies_bouldin_score(embeddings, labels)
        if db < best_db:
            best_db = db
            best_k = k
            best_labels = labels
    return best_k, best_db, best_labels

def _kmeans_scores(embeddings: np.ndarray, k_range: range):
    silhouettes = []
    ch_scores = []
    db_scores = []
    inertias = []
    labels_by_k = {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(embeddings)
        labels_by_k[k] = labels
        silhouettes.append(silhouette_score(embeddings, labels))
        ch_scores.append(calinski_harabasz_score(embeddings, labels))
        db_scores.append(davies_bouldin_score(embeddings, labels))
        inertias.append(model.inertia_)
    return (
        np.array(silhouettes),
        np.array(ch_scores),
        np.array(db_scores),
        np.array(inertias),
        labels_by_k,
    )

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

def _normalized(values: np.ndarray) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    if np.isclose(vmin, vmax):
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)

def plot_hierarchical_radar_plots(
    df: pl.DataFrame,
    k_min: int = 2,
    k_max: int = 20,
    output_dir: Path = Path("figures/materials/clustering/hierarchical/soap_reduced"),
):
    embeddings = _soap_embeddings(df)
    dist_mats = _soap_distance_matrices(embeddings)
    k_range = range(k_min, k_max + 1)

    linkage_outputs = {}
    for linkage in ["average", "complete"]:
        logger.info(f"Computing hierarchical metrics for linkage='{linkage}'...")
        eval_series = {}
        chem_series = {}
        eval_triplets = {}
        k_by_name = {}

        for name, emb in embeddings.items():
            res, labels_by_k = _hierarchical_eval_curves(emb, dist_mats[name], linkage, k_range)
            db_array = np.array(res["db"])
            best_idx = int(np.argmin(db_array))
            k = list(k_range)[best_idx]
            labels = labels_by_k[k]
            db = db_array[best_idx]

            eval_output_path = output_dir / f"evaluation/{name}_{linkage}_evaluation.png"
            plot_evaluation(
                res,
                out_path=eval_output_path,
                title=f"{name} ({linkage}) - evaluation",
                dpi=300,
            )

            sil = silhouette_score(emb, labels)
            ch = calinski_harabasz_score(emb, labels)
            db_inv = _invert_db_score(db)
            eval_triplets[name] = (sil, ch, db_inv)
            k_by_name[name] = k

            chem_scores, _ = get_overall_chemical_coherence(df, labels)
            chem_labels, chem_values = _build_chem_radar_values(chem_scores)
            chem_series[f"{name} (k={k})"] = chem_values

        sils = np.array([v[0] for v in eval_triplets.values()])
        chs = np.array([v[1] for v in eval_triplets.values()])
        dbs = np.array([v[2] for v in eval_triplets.values()])
        sil_norm = _normalized(sils)
        ch_norm = _normalized(chs)
        db_norm = _normalized(dbs)

        for (name, _), s_n, c_n, d_n in zip(embeddings.items(), sil_norm, ch_norm, db_norm):
            k = k_by_name[name]
            eval_labels, eval_values = _build_eval_radar_values(float(s_n), float(c_n), float(d_n))
            eval_series[f"{name} (k={k})"] = eval_values

        linkage_outputs[linkage] = {
            "eval": (f"Hierarchical {linkage} - metrics", eval_labels, eval_series),
            "chem": (f"Hierarchical {linkage} - chemical", chem_labels, chem_series),
        }

    # Plot 1: Metrics (Complete vs. Average)
    _plot_radar_pair(
        "Hierarchical Clustering - Evaluation Metrics",
        linkage_outputs["complete"]["eval"],
        linkage_outputs["average"]["eval"],
        out_path=output_dir / "hierarchical_radar_metrics.png",
        dpi=300,
    )

    # Plot 2: Chemical Features (Complete vs. Average)
    _plot_radar_pair(
        "Hierarchical Clustering - Chemical Cohesion",
        linkage_outputs["complete"]["chem"],
        linkage_outputs["average"]["chem"],
        out_path=output_dir / "hierarchical_radar_chemical.png",
        dpi=300,
    )

def plot_kmeans_radar_plots(
    df: pl.DataFrame,
    k_min: int = 2,
    k_max: int = 20,
    output_dir: Path = Path("figures/materials/clustering/kmeans/soap_reduced"),
):
    embeddings = _soap_embeddings(df)
    k_range = range(k_min, k_max + 1)
    eval_series = {}
    chem_series = {}

    eval_triplets = {}
    chem_labels = None
    eval_labels = None
    for name, emb in embeddings.items():
        logger.info(f"Computing kmeans metrics for {name}...")
        silhouettes, ch_scores, db_scores, inertias, labels_by_k = _kmeans_scores(emb, k_range)
        k_list = list(k_range)

        res = {"sil": silhouettes.tolist(), "ch": ch_scores.tolist(), "db": db_scores.tolist()}
        eval_plot_path = output_dir / f"evaluation/{name}_kmeans_evaluation.png"
        plot_evaluation(
            res,
            out_path=eval_plot_path,
            title=f"{name} (kmeans) - evaluation",
            dpi=300,
        )

        k_db = k_list[int(np.argmin(db_scores))]
        labels_in = labels_by_k[k_db]
        sil_i = silhouette_score(emb, labels_in)
        ch_i = calinski_harabasz_score(emb, labels_in)
        db_i = davies_bouldin_score(emb, labels_in)
        eval_triplets[f"{name} (db k={k_db})"] = (sil_i, ch_i, _invert_db_score(db_i))
        chem_scores_in, _ = get_overall_chemical_coherence(df, labels_in)
        chem_labels, chem_values = _build_chem_radar_values(chem_scores_in)
        chem_series[f"{name} (db k={k_db})"] = chem_values

    if eval_triplets:
        sils = np.array([v[0] for v in eval_triplets.values()])
        chs = np.array([v[1] for v in eval_triplets.values()])
        dbs = np.array([v[2] for v in eval_triplets.values()])
        sil_norm = _normalized(sils)
        ch_norm = _normalized(chs)
        db_norm = _normalized(dbs)

        for (series_label, _), s_n, c_n, d_n in zip(eval_triplets.items(), sil_norm, ch_norm, db_norm):
            eval_labels, eval_values = _build_eval_radar_values(float(s_n), float(c_n), float(d_n))
            eval_series[series_label] = eval_values

    _plot_radar_pair(
        "KMeans - metrics and chemical cohesion",
        ("KMeans - evaluation metrics", eval_labels, eval_series),
        ("KMeans - chemical cohesion", chem_labels, chem_series),
        out_path=output_dir / "kmeans_radar_pair.png",
        dpi=300,
    )

def plot_hierarchical_dendrograms(
    df: pl.DataFrame,
    linkage_method: str = "average",
    output_dir: Path = Path("figures/materials/clustering/hierarchical/dendrograms/soap_reduced"),
):
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
    out_path = output_dir / f"dendrograms/dendrograms_{linkage_method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved dendrograms to {out_path}")
    plt.close()

def _distance_cache_key(df: pl.DataFrame, cache_tag: Optional[str] = None) -> str:
    """
    Build a stable cache key from dataset identity (prefer material_id) + row count.
    """
    base = cache_tag or "default"
    if "material_id" in df.columns:
        ids = df["material_id"].cast(pl.Utf8).to_list()
        joined = "|".join(ids)
        digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    else:
        digest = hashlib.md5(str(len(df)).encode("utf-8")).hexdigest()
    return f"{base}_{len(df)}_{digest[:12]}"


def get_distance_matrices_soap(
    df: pl.DataFrame,
    soap_cache_dir: str = "data/Materials Project/soap_distances",
    cache_tag: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute (or load) SOAP-based distance matrices and return them in a dict.
    """
    cache_key = _distance_cache_key(df, cache_tag=cache_tag)
    soap_root = Path(soap_cache_dir)
    soap_root.mkdir(parents=True, exist_ok=True)
    soap_cache_path = soap_root / f"{cache_key}.npz"

    if soap_cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached SOAP distances from {soap_cache_path}...")
        cached_soap = np.load(soap_cache_path, allow_pickle=False)
        return {k: cached_soap[k] for k in cached_soap.files}

    soap_array = np.array(df['soap_embedding'].to_list())

    reducers = get_reducers()
    soap_pca = reducers['pca'].fit_transform(soap_array)
    soap_tsne = reducers['tsne'].fit_transform(soap_array)
    soap_umap = reducers['umap'].fit_transform(soap_array)
    soap_isomap = reducers['isomap'].fit_transform(soap_array)

    distance_raw_soap_matrix = squareform(pdist(soap_array, metric='euclidean'))
    distance_pca_soap_matrix = squareform(pdist(soap_pca, metric='euclidean'))
    distance_tsne_soap_matrix = squareform(pdist(soap_tsne, metric='euclidean'))
    distance_umap_soap_matrix = squareform(pdist(soap_umap, metric='euclidean'))
    distance_isomap_soap_matrix = squareform(pdist(soap_isomap, metric='euclidean'))

    soap_distances = {
        "soap_raw": distance_raw_soap_matrix,
        "soap_pca": distance_pca_soap_matrix,
        "soap_tsne": distance_tsne_soap_matrix,
        "soap_umap": distance_umap_soap_matrix,
        "soap_isomap": distance_isomap_soap_matrix,
    }

    np.savez_compressed(soap_cache_path, **soap_distances)
    logger.success(
        f"Cached {len(soap_distances)} SOAP distances to {soap_cache_path}."
    )

    return soap_distances


def get_distance_matrices_non_euclidean(
    df: pl.DataFrame,
    invariant_cache_dir: str = "data/Materials Project/invariant_distances",
    cache_tag: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute (or load) invariant/non-euclidean distance matrices and return them in a dict.
    """
    cache_key = _distance_cache_key(df, cache_tag=cache_tag)
    inv_root = Path(invariant_cache_dir)
    inv_root.mkdir(parents=True, exist_ok=True)
    inv_cache_path = inv_root / f"{cache_key}.npz"

    if inv_cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached invariant distances from {inv_cache_path}...")
        cached_inv = np.load(inv_cache_path, allow_pickle=False)
        return {k: cached_inv[k] for k in cached_inv.files}

    frames = get_ase_frames(df)
    precomputed_feature_matrices = build_invariant_matrix(df, aggregated=False)
    precomputed_feature_matrices_average_aggregated = build_invariant_matrix(df, aggregated=True, feature_keys=["en", "avg_neighbor_dist", "rad", "group"])
    precomputed_feature_matrices_complete_aggregated = build_invariant_matrix(df, aggregated=True, feature_keys=["en", "z", "mendeleev", "group", "row"])

    distance_invariant_average_matrix_aggregated = squareform(
        pdist(precomputed_feature_matrices_average_aggregated, metric='euclidean')
    )
    distance_invariant_complete_matrix_aggregated = squareform(
        pdist(precomputed_feature_matrices_complete_aggregated, metric='euclidean')
    )
    
    distance_invariant_matrix_riemann = Riemann.distance_matrix(
        precomputed_feature_matrices=precomputed_feature_matrices,
        metric='affine-invariant'
    )
    distance_invariant_matrix_grassmann = Grassmann.distance_matrix(
        precomputed_feature_matrices=precomputed_feature_matrices
    )
    distance_invariant_matrix_wasserstein = Wasserstein.distance_matrix(
        precomputed_feature_matrices=precomputed_feature_matrices
    )
    #distance_topological_bottleneck_matrix = PersistentHomology.distance_matrix(frames=frames, metric="bottleneck", max_homology_dim=2)
    #distance_topological_sliced_wasserstein_matrix = PersistentHomology.distance_matrix(frames=frames, metric="sliced-wasserstein", max_homology_dim=2)

    invariant_distances = {
        "invariant_avg_aggregated": distance_invariant_average_matrix_aggregated,
        "invariant_complete_aggregated": distance_invariant_complete_matrix_aggregated,
        "invariant_riemann": distance_invariant_matrix_riemann,
        "invariant_grassmann": distance_invariant_matrix_grassmann,
        "invariant_wasserstein": distance_invariant_matrix_wasserstein,
       # "topological_bottleneck": distance_topological_bottleneck_matrix,
       # "topological_sliced_wasserstein": distance_topological_sliced_wasserstein_matrix
    }

    np.savez_compressed(inv_cache_path, **invariant_distances)
    logger.success(
        f"Cached {len(invariant_distances)} invariant distances to {inv_cache_path}."
    )

    return invariant_distances


def get_distance_matrices(
    df: pl.DataFrame,
    soap_cache_dir: str = "data/Materials Project/soap_distances",
    invariant_cache_dir: str = "data/Materials Project/invariant_distances",
    cache_tag: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Backwards-compatible wrapper returning all distances in a single dict.
    """
    distances = {}
    distances.update(
        get_distance_matrices_soap(
            df,
            soap_cache_dir=soap_cache_dir,
            cache_tag=cache_tag,
            force_recompute=force_recompute,
        )
    )
    distances.update(
        get_distance_matrices_non_euclidean(
            df,
            invariant_cache_dir=invariant_cache_dir,
            cache_tag=cache_tag,
            force_recompute=force_recompute,
        )
    )
    return distances


if __name__ == '__main__':

    mp = MaterialsProject(add_soap=True, add_acsf=False, stratify_on=["band_gap", "energy_above_hull"], sampling_strategy="stratified")
    df = mp.load(limit=250)

    hierarchical_dir = Path("figures/materials/clustering/hierarchical/soap_reduced")
    dendrogram_dir = hierarchical_dir / "dendrograms"
    kmeans_dir = Path("figures/materials/clustering/kmeans/soap_reduced")
    
    invariant_dendrogram_dir = Path("figures/materials/clustering/hierarchical/dendrograms/invariant")

    #evaluate_invariant_combinations(df, linkage="average", k_min=2, k_max=20)
    #evaluate_invariant_combinations(df, linkage="complete", k_min=2, k_max=20)

    logger.info("Generating Invariant Aggregated hierarchical dendrograms...")
    plot_invariant_aggregated_dendrograms(df, linkage_method="average", output_dir=invariant_dendrogram_dir)
    plot_invariant_aggregated_dendrograms(df, linkage_method="complete", output_dir=invariant_dendrogram_dir)

    logger.info("Generating hierarchical radar plots...")
    plot_hierarchical_radar_plots(df, k_min=2, k_max=20, output_dir=hierarchical_dir)
    
    logger.info("Generating kmeans radar plots...")
    plot_kmeans_radar_plots(df, k_min=2, k_max=20, output_dir=kmeans_dir)
    
    logger.info("Generating SOAP hierarchical dendrograms...")
    plot_hierarchical_dendrograms(df, linkage_method="average", output_dir=dendrogram_dir)
    plot_hierarchical_dendrograms(df, linkage_method="complete", output_dir=dendrogram_dir)
    
    logger.info("Generating Invariant Aggregated hierarchical dendrograms...")
    plot_invariant_aggregated_dendrograms(df, linkage_method="average", output_dir=invariant_dendrogram_dir)
    plot_invariant_aggregated_dendrograms(df, linkage_method="complete", output_dir=invariant_dendrogram_dir)
