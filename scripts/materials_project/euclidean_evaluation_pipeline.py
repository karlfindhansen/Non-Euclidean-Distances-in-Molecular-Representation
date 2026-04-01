from typing import Optional, Dict, List, Tuple
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
from pymatgen.core import Structure, Element
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import neighbor_list
from loguru import logger

from src.datasets import MaterialsProject


INVARIANT_FEATURES = [
    "z", "en", "coord", "avg_neighbor_dist", "vol_per_atom", "mass", "rad",
    "mendeleev", "group", "row", "std_neighbor_dist", "min_neighbor_dist",
    "max_neighbor_dist", "ea", "ion_en", "vdw_rad", "melt_pt", "mol_vol",
]


def get_overall_chemical_coherence(df, labels):
    df = df.with_columns(pl.Series(name="labels", values=labels))

    continuous_features = [
        "band_gap",
        "density",
        "energy_per_atom",
        "formation_energy_per_atom",
        "volume",
        "energy_above_hull",
    ]

    discrete_feature = "is_metal"

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.select(continuous_features).to_numpy())

    df_scaled = df.with_columns(
        [
            pl.Series(name, scaled_values[:, i])
            for i, name in enumerate(continuous_features)
        ]
    )

    cluster_sizes = df_scaled.group_by("labels").agg(pl.len().alias("count"))

    cluster_std = (
        df_scaled
        .group_by("labels")
        .agg([
            pl.col(f).std().fill_null(0.0).alias(f)
            for f in continuous_features
        ])
    )

    cluster_coherence = cluster_std.with_columns(
        [(1 / (1 + pl.col(f))).alias(f) for f in continuous_features]
    )

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

    cluster_scores = (
        cluster_coherence
        .join(metal_coherence, on="labels")
        .join(cluster_sizes, on="labels")
    )

    weights = cluster_scores["count"].to_numpy()

    results = {}

    for feature in continuous_features + [discrete_feature]:
        values = cluster_scores[feature].to_numpy()
        results[feature] = float(np.average(values, weights=weights))

    average_coherence = float(np.mean(list(results.values())))

    return results, average_coherence


def _compute_invariant_feature_matrix_materials(
    frame: Atoms,
    cutoff: float = 3.0,
    aggregated: bool = False,
    feature_keys: Optional[list] = None,
) -> np.ndarray:
    if feature_keys is None:
        feature_keys = INVARIANT_FEATURES

    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)
    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    features = []
    vol_per_atom = frame.get_volume() / len(frame)

    for i, atom in enumerate(frame):
        z = atom.number
        el = Element.from_Z(z)

        en = el.X if getattr(el, "X", None) else 0.0
        rad = el.atomic_radius if getattr(el, "atomic_radius", None) else 0.0
        mass = float(el.atomic_mass)
        mendeleev = el.mendeleev_no if getattr(el, "mendeleev_no", None) else 0
        group = el.group if getattr(el, "group", None) else 0
        row = el.row if getattr(el, "row", None) else 0

        ea = getattr(el, "electron_affinity", 0.0) or 0.0
        ion_list = getattr(el, "ionization_energies", [])
        ion_en = ion_list[0] if ion_list else 0.0

        vdw_rad = getattr(el, "van_der_waals_radius", 0.0) or 0.0
        melt_pt = getattr(el, "melting_point", 0.0) or 0.0
        mol_vol = getattr(el, "molar_volume", 0.0) or 0.0

        coord = len(neighbors[i])
        if coord > 0:
            dist_array = distances[i]
            avg_neighbor_dist = float(np.mean(dist_array))
            std_neighbor_dist = float(np.std(dist_array))
            min_neighbor_dist = float(np.min(dist_array))
            max_neighbor_dist = float(np.max(dist_array))
        else:
            avg_neighbor_dist = 0.0
            std_neighbor_dist = 0.0
            min_neighbor_dist = 0.0
            max_neighbor_dist = 0.0

        feature_pool = {
            "z": z, "en": en, "coord": coord, "avg_neighbor_dist": avg_neighbor_dist,
            "vol_per_atom": vol_per_atom, "mass": mass, "rad": rad,
            "mendeleev": mendeleev, "group": group, "row": row,
            "std_neighbor_dist": std_neighbor_dist,
            "min_neighbor_dist": min_neighbor_dist,
            "max_neighbor_dist": max_neighbor_dist,
            "ea": ea,
            "ion_en": ion_en,
            "vdw_rad": vdw_rad,
            "melt_pt": melt_pt,
            "mol_vol": mol_vol,
        }

        feat_vector = [feature_pool[k] for k in feature_keys]
        features.append(feat_vector)

    if aggregated:
        atom_matrix = np.array(features).T
        mean_features = np.mean(atom_matrix, axis=1)
        std_features = np.std(atom_matrix, axis=1)
        return np.concatenate([mean_features, std_features])

    return np.array(features).T


def build_invariant_matrix(
    df: pl.DataFrame,
    cutoff: float = 3.0,
    aggregated: bool = False,
    feature_keys: Optional[list] = None,
) -> list:
    """
    Iterates through the materials dataframe, converts JSON structures to ASE Atoms,
    and computes the D x N invariant feature matrix for each material.
    """
    if feature_keys is None:
        feature_keys = INVARIANT_FEATURES

    adaptor = AseAtomsAdaptor()
    invariant_matrices = []

    for struct_json in df["raw_structure"]:
        struct = Structure.from_dict(json.loads(struct_json))
        atoms = adaptor.get_atoms(struct)
        matrix = _compute_invariant_feature_matrix_materials(
            atoms,
            cutoff=cutoff,
            aggregated=aggregated,
            feature_keys=feature_keys,
        )
        invariant_matrices.append(matrix)

    return invariant_matrices


def _screen_correlated_invariant_features(
    df: pl.DataFrame,
    features: List[str],
    threshold: float = 0.85,
) -> Tuple[List[str], List[str]]:
    full_matrix = np.array(build_invariant_matrix(df, aggregated=True, feature_keys=features))
    mean_matrix = full_matrix[:, :len(features)]

    df_corr = pd.DataFrame(mean_matrix, columns=features)
    corr_matrix = df_corr.corr(method="spearman").abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    screened_features = [f for f in features if f not in to_drop]
    return screened_features, to_drop


def _iter_invariant_combinations(features: List[str]) -> List[List[str]]:
    combinations_to_test = []
    for r in range(1, len(features) + 1):
        for combo in itertools.combinations(features, r):
            combinations_to_test.append(list(combo))
    return combinations_to_test


def _scaled_matrix(matrix: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def _scaled_invariant_matrix(df: pl.DataFrame, feature_keys: List[str]) -> np.ndarray:
    raw_matrix = build_invariant_matrix(df, aggregated=True, feature_keys=feature_keys)
    return _scaled_matrix(np.array(raw_matrix))


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


def _soap_embeddings(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    soap_array = np.array(df["soap_embedding"].to_list())
    reducers = get_reducers()
    return {
        "soap_raw": soap_array,
        "soap_pca": reducers["pca"].fit_transform(soap_array),
        "soap_tsne": reducers["tsne"].fit_transform(soap_array),
        "soap_umap": reducers["umap"].fit_transform(soap_array),
        "soap_isomap": reducers["isomap"].fit_transform(soap_array),
    }


def _soap_items(df: pl.DataFrame) -> List[Tuple[str, np.ndarray]]:
    embeddings = _soap_embeddings(df)
    return list(embeddings.items())


def evaluate_hierarchical_combinations(
    df: pl.DataFrame,
    linkage: str = "average",
    k_min: int = 2,
    k_max: int = 20,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using hierarchical clustering. Selects k by max silhouette and reports cophenetic corr.
    """
    results = []
    k_range = range(k_min, k_max + 1)

    if mode == "soap":
        items = _soap_items(df)
        output_dir = Path("figures/materials/clustering/hierarchical/soap_reduced")
        logger.info("--- HIERARCHICAL (SOAP) ---")
        for name, emb in tqdm(items, desc="Evaluating Hierarchical (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)
            dist_condensed = pdist(scaled_matrix, metric="euclidean")
            Z = scipy_linkage(dist_condensed, method=linkage)

            coph_corr, _ = cophenet(Z, dist_condensed)

            best_k = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None

            for k in k_range:
                labels = fcluster(Z, k, criterion="maxclust")
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(scaled_matrix, labels)
                    db = davies_bouldin_score(scaled_matrix, labels)
                    if sil > best_sil:
                        best_sil = sil
                        best_db = db
                        best_k = k
                        best_labels = labels

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Cophenetic Corr": coph_corr,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = INVARIANT_FEATURES

        logger.info("--- HIERARCHICAL STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced hierarchical search space to {len(combinations_to_test)} combinations.")

        output_dir = Path("figures/materials/clustering/hierarchical")
        logger.info("--- HIERARCHICAL STAGE 2 & 3: Statistical Evaluation & k Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating Hierarchical"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)
            dist_condensed = pdist(scaled_matrix, metric="euclidean")
            Z = scipy_linkage(dist_condensed, method=linkage)

            coph_corr, _ = cophenet(Z, dist_condensed)

            best_k = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None

            for k in k_range:
                labels = fcluster(Z, k, criterion="maxclust")
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(scaled_matrix, labels)
                    db = davies_bouldin_score(scaled_matrix, labels)
                    if sil > best_sil:
                        best_sil = sil
                        best_db = db
                        best_k = k
                        best_labels = labels

            if best_labels is None:
                logger.warning(f"Combination [{combo_name}] produced zero variance or invalid clusters. Skipping.")
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": len(feature_keys),
                "Optimal k": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Cophenetic Corr": coph_corr,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No hierarchical results produced.")
        return

    valid_models = results_df[results_df["Cophenetic Corr"] > 0.8]
    valid_models = valid_models[valid_models["Silhouette"] > 0.6]
    valid_models = valid_models.sort_values(by="Silhouette", ascending=False)

    if valid_models.empty:
        logger.warning("No hierarchical combinations met the thresholds. Falling back to simple sorting.")
        valid_models = results_df.sort_values(by="Silhouette", ascending=False)

    csv_path = output_dir / f"ablation_results_full_{linkage}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_models.to_csv(csv_path, index=False)
    logger.success(f"Saved hierarchical results to {csv_path}")

    top_20 = valid_models.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_20,
        x="Silhouette",
        y="Combination",
        hue="Combination",
        palette="viridis",
        legend=False,
    )
    title_suffix = "SOAP" if mode == "soap" else "Invariant"
    plt.title(f"Top Feature Combinations by Silhouette Score\n(Hierarchical {title_suffix}, linkage={linkage})", fontsize=14)
    plt.xlabel("Silhouette Score (Higher is Better)")
    plt.ylabel("Feature Combination")

    for index, (_, row) in enumerate(top_20.iterrows()):
        annot_text = (
            f"k={int(row['Optimal k'])} | DB: {row['Davies-Bouldin']:.2f} | "
            f"Coph: {row['Cophenetic Corr']:.2f}"
        )
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / f"ablation_study_top20_{linkage}.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 hierarchical ablation plot to {out_path}")


def evaluate_kmeans_combinations(
    df: pl.DataFrame,
    k_min: int = 2,
    k_max: int = 20,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using K-Means clustering with elbow-based k selection.
    """
    results = []
    k_range = list(range(k_min, k_max + 1))

    if mode == "soap":
        items = _soap_items(df)
        output_dir = Path("figures/materials/clustering/kmeans/soap_reduced")
        logger.info("--- KMEANS (SOAP) ---")
        for name, emb in tqdm(items, desc="Evaluating KMeans (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)

            inertias = []
            silhouettes = []
            labels_by_k = {}

            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(scaled_matrix)
                labels_by_k[k] = labels
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(scaled_matrix, labels))

            best_k = _kneedle_elbow(np.array(k_range), np.array(inertias))
            best_labels = labels_by_k[best_k]
            best_sil = silhouettes[k_range.index(best_k)]
            best_db = davies_bouldin_score(scaled_matrix, best_labels)

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k (Elbow)": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = INVARIANT_FEATURES

        logger.info("--- KMEANS STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced KMeans search space to {len(combinations_to_test)} combinations.")

        output_dir = Path("figures/materials/clustering/kmeans")
        logger.info("--- KMEANS STAGE 2 & 3: Statistical Evaluation & Auto-Elbow ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating KMeans"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)

            inertias = []
            silhouettes = []
            labels_by_k = {}

            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(scaled_matrix)
                labels_by_k[k] = labels
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(scaled_matrix, labels))

            best_k = _kneedle_elbow(np.array(k_range), np.array(inertias))
            best_labels = labels_by_k[best_k]
            best_sil = silhouettes[k_range.index(best_k)]
            best_db = davies_bouldin_score(scaled_matrix, best_labels)

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": len(feature_keys),
                "Optimal k (Elbow)": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No KMeans results produced.")
        return

    valid_models = results_df[results_df["Silhouette"] > 0.6].sort_values(by="Silhouette", ascending=False)
    if valid_models.empty:
        logger.warning("No KMeans combinations met the Silhouette threshold. Falling back to simple sorting.")
        valid_models = results_df.sort_values(by="Silhouette", ascending=False)

    csv_path = output_dir / "ablation_results_full_kmeans.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_models.to_csv(csv_path, index=False)
    logger.success(f"Saved KMeans results to {csv_path}")

    top_20 = valid_models.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_20,
        x="Silhouette",
        y="Combination",
        hue="Combination",
        palette="magma",
        legend=False,
    )
    title_suffix = "SOAP" if mode == "soap" else "Invariant"
    plt.title(f"Top Feature Combinations by Silhouette Score\n(KMeans {title_suffix})", fontsize=14)
    plt.xlabel("Silhouette Score (Higher is Better)")
    plt.ylabel("Feature Combination")

    for index, (_, row) in enumerate(top_20.iterrows()):
        annot_text = f"k={int(row['Optimal k (Elbow)'])} | DB: {row['Davies-Bouldin']:.2f}"
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "ablation_study_top20_kmeans.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 KMeans ablation plot to {out_path}")


def evaluate_spectral_combinations(
    df: pl.DataFrame,
    k_min: int = 2,
    k_max: int = 20,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using Spectral Clustering. Selects k by max silhouette.
    """
    results = []
    k_range = list(range(k_min, k_max + 1))

    if mode == "soap":
        items = _soap_items(df)
        output_dir = Path("figures/materials/clustering/spectral/soap_reduced")
        logger.info("--- SPECTRAL (SOAP) ---")
        for name, emb in tqdm(items, desc="Evaluating Spectral (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)

            best_k = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None

            for k in k_range:
                spectral = SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    assign_labels="kmeans",
                    random_state=42,
                )

                try:
                    labels = spectral.fit_predict(scaled_matrix)
                    if len(np.unique(labels)) > 1:
                        sil = silhouette_score(scaled_matrix, labels)
                        db = davies_bouldin_score(scaled_matrix, labels)

                        if sil > best_sil:
                            best_sil = sil
                            best_db = db
                            best_k = k
                            best_labels = labels
                except Exception:
                    continue

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = INVARIANT_FEATURES

        logger.info("--- SPECTRAL STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced Spectral search space to {len(combinations_to_test)} combinations.")

        output_dir = Path("figures/materials/clustering/spectral")
        logger.info("--- SPECTRAL STAGE 2 & 3: Statistical Evaluation & k Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating Spectral"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)

            best_k = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None

            for k in k_range:
                spectral = SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    assign_labels="kmeans",
                    random_state=42,
                )

                try:
                    labels = spectral.fit_predict(scaled_matrix)
                    if len(np.unique(labels)) > 1:
                        sil = silhouette_score(scaled_matrix, labels)
                        db = davies_bouldin_score(scaled_matrix, labels)

                        if sil > best_sil:
                            best_sil = sil
                            best_db = db
                            best_k = k
                            best_labels = labels
                except Exception:
                    continue

            if best_labels is None:
                logger.warning(f"Combination [{combo_name}] produced invalid graph components. Skipping.")
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": len(feature_keys),
                "Optimal k": best_k,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No Spectral results produced.")
        return

    valid_models = results_df[results_df["Silhouette"] > 0.6].sort_values(by="Silhouette", ascending=False)
    if valid_models.empty:
        logger.warning("No Spectral combinations met the Silhouette threshold. Falling back to simple sorting.")
        valid_models = results_df.sort_values(by="Silhouette", ascending=False)

    csv_path = output_dir / "ablation_results_full_spectral.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_models.to_csv(csv_path, index=False)
    logger.success(f"Saved Spectral results to {csv_path}")

    top_20 = valid_models.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_20,
        x="Silhouette",
        y="Combination",
        hue="Combination",
        palette="crest",
        legend=False,
    )
    title_suffix = "SOAP" if mode == "soap" else "Invariant"
    plt.title(f"Top Feature Combinations by Silhouette Score\n(Spectral {title_suffix})", fontsize=14)
    plt.xlabel("Silhouette Score (Higher is Better)")
    plt.ylabel("Feature Combination")

    for index, (_, row) in enumerate(top_20.iterrows()):
        annot_text = f"k={int(row['Optimal k'])} | DB: {row['Davies-Bouldin']:.2f}"
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "ablation_study_top20_spectral.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 Spectral ablation plot to {out_path}")


def evaluate_dbscan_combinations(
    df: pl.DataFrame,
    eps_values: Optional[List[float]] = None,
    min_samples: int = 5,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using DBSCAN and eps sweep.
    """
    if eps_values is None:
        eps_values = [0.25, 0.5, 0.75, 1.0, 1.25]

    results = []

    if mode == "soap":
        items = _soap_items(df)
        output_dir = Path("figures/materials/clustering/dbscan/soap_reduced")
        logger.info("--- DBSCAN (SOAP) ---")

        for name, emb in tqdm(items, desc="Evaluating DBSCAN (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)

            best_eps = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None
            best_noise_ratio = 1.0
            best_n_clusters = 0

            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(scaled_matrix)

                noise_ratio = np.sum(labels == -1) / len(labels)
                valid_clusters = set(labels) - {-1}
                n_clusters = len(valid_clusters)

                if n_clusters >= 2 and noise_ratio <= 0.50:
                    sil = silhouette_score(scaled_matrix, labels)
                    db = davies_bouldin_score(scaled_matrix, labels)

                    if sil > best_sil:
                        best_sil = sil
                        best_db = db
                        best_eps = eps
                        best_labels = labels
                        best_noise_ratio = noise_ratio
                        best_n_clusters = n_clusters

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal eps": best_eps,
                "Num Clusters": best_n_clusters,
                "Noise Ratio": best_noise_ratio,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = INVARIANT_FEATURES

        logger.info("--- DBSCAN STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced DBSCAN search space to {len(combinations_to_test)} combinations.")

        output_dir = Path("figures/materials/clustering/dbscan")
        logger.info("--- DBSCAN STAGE 2 & 3: Statistical Evaluation & Eps Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating DBSCAN"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)

            best_eps = None
            best_db = np.inf
            best_sil = -1.0
            best_labels = None
            best_noise_ratio = 1.0
            best_n_clusters = 0

            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(scaled_matrix)

                noise_ratio = np.sum(labels == -1) / len(labels)
                valid_clusters = set(labels) - {-1}
                n_clusters = len(valid_clusters)

                if n_clusters >= 2 and noise_ratio <= 0.50:
                    sil = silhouette_score(scaled_matrix, labels)
                    db = davies_bouldin_score(scaled_matrix, labels)

                    if sil > best_sil:
                        best_sil = sil
                        best_db = db
                        best_eps = eps
                        best_labels = labels
                        best_noise_ratio = noise_ratio
                        best_n_clusters = n_clusters

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": len(feature_keys),
                "Optimal eps": best_eps,
                "Num Clusters": best_n_clusters,
                "Noise Ratio": best_noise_ratio,
                "Davies-Bouldin": best_db,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.error("No DBSCAN combinations yielded valid clusters with <50% noise. Try adjusting eps_values or min_samples.")
        return

    valid_models = results_df[results_df["Silhouette"] > 0.4].sort_values(by="Silhouette", ascending=False)

    if valid_models.empty:
        logger.warning("No DBSCAN combinations met the Silhouette threshold. Falling back to simple sorting.")
        valid_models = results_df.sort_values(by="Silhouette", ascending=False)

    csv_path = output_dir / "ablation_results_full_dbscan.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_models.to_csv(csv_path, index=False)
    logger.success(f"Saved DBSCAN results to {csv_path}")

    top_20 = valid_models.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=top_20,
        x="Silhouette",
        y="Combination",
        hue="Combination",
        palette="flare",
        legend=False,
    )
    title_suffix = "SOAP" if mode == "soap" else "Invariant"
    plt.title(f"Top Feature Combinations by Silhouette Score\n(DBSCAN {title_suffix})", fontsize=14)
    plt.xlabel("Silhouette Score (Higher is Better)")
    plt.ylabel("Feature Combination")

    for index, (_, row) in enumerate(top_20.iterrows()):
        annot_text = f"eps={row['Optimal eps']} | k={int(row['Num Clusters'])} | Noise: {row['Noise Ratio']:.1%}"
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "ablation_study_top20_dbscan.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 DBSCAN ablation plot to {out_path}")


def get_reducers():
    tsne = TSNE(n_components=3)
    pca = PCA(n_components=3)
    umap = UMAP(n_components=3)
    isomap = Isomap(n_components=3)
    return {"tsne": tsne, "pca": pca, "umap": umap, "isomap": isomap}


def _soap_distance_matrices(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        name: squareform(pdist(emb, metric="euclidean"))
        for name, emb in embeddings.items()
    }


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

    soap_array = np.array(df["soap_embedding"].to_list())

    reducers = get_reducers()
    soap_pca = reducers["pca"].fit_transform(soap_array)
    soap_tsne = reducers["tsne"].fit_transform(soap_array)
    soap_umap = reducers["umap"].fit_transform(soap_array)
    soap_isomap = reducers["isomap"].fit_transform(soap_array)

    distance_raw_soap_matrix = squareform(pdist(soap_array, metric="euclidean"))
    distance_pca_soap_matrix = squareform(pdist(soap_pca, metric="euclidean"))
    distance_tsne_soap_matrix = squareform(pdist(soap_tsne, metric="euclidean"))
    distance_umap_soap_matrix = squareform(pdist(soap_umap, metric="euclidean"))
    distance_isomap_soap_matrix = squareform(pdist(soap_isomap, metric="euclidean"))

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


if __name__ == "__main__":
    mp = MaterialsProject(
        add_soap=True,
        add_acsf=False,
        stratify_on=["band_gap", "energy_above_hull"],
        sampling_strategy="stratified",
    )
    df = mp.load(limit=150)

    # Invariant-only evaluations
    # evaluate_hierarchical_combinations(df, linkage="average", k_min=2, k_max=20)
    # evaluate_hierarchical_combinations(df, linkage="complete", k_min=2, k_max=20)
    # evaluate_kmeans_combinations(df, k_min=2, k_max=20)
    # evaluate_spectral_combinations(df, k_min=2, k_max=20)
    # evaluate_dbscan_combinations(df, min_samples=3)

    # SOAP evaluations using the same functions
    evaluate_hierarchical_combinations(df, linkage="average", k_min=2, k_max=20, mode="soap")
    evaluate_hierarchical_combinations(df, linkage="complete", k_min=2, k_max=20, mode="soap")
    evaluate_kmeans_combinations(df, k_min=2, k_max=20, mode="soap")
    evaluate_spectral_combinations(df, k_min=2, k_max=20, mode="soap")
    evaluate_dbscan_combinations(df, min_samples=3, mode="soap")
