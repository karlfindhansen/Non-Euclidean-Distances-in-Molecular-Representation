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

from hdbscan.validity import validity_index
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Element
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from umap import UMAP
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import neighbor_list
from loguru import logger

from src.datasets import MaterialsProject, QM9Dataset


INVARIANT_FEATURES_MATERIALS = [
    "z", "en", "coord", "avg_neighbor_dist", "vol_per_atom", "mass", "rad",
    "mendeleev", "std_neighbor_dist", "min_neighbor_dist",
    "max_neighbor_dist", "ea", "ion_en", "vdw_rad", "melt_pt", "mol_vol",
]
INVARIANT_FEATURES_QM9 = [
    "z", "en", "coord", "avg_neighbor_dist", "mass", "rad",
    "mendeleev", "std_neighbor_dist", "min_neighbor_dist",
    "max_neighbor_dist", "ea", "ion_en", "vdw_rad",
]

CHEMICAL_INVARIANT_FEATURES = {
    "z",
    "mass",
    "rad",
    "mendeleev",
    "vdw_rad",
    "melt_pt",
    "mol_vol",
}
ELECTRONIC_INVARIANT_FEATURES = {
    "en",
    "ea",
    "ion_en",
}
STRUCTURAL_INVARIANT_FEATURES = {
    "coord",
    "avg_neighbor_dist",
    "std_neighbor_dist",
    "min_neighbor_dist",
    "max_neighbor_dist",
    "vol_per_atom",
}

MAX_SOAP_N_COMPONENTS = 50
# Backward compatible alias (defaults to materials).
INVARIANT_FEATURES = INVARIANT_FEATURES_MATERIALS


def _detect_dataset_kind(df: pl.DataFrame) -> str:
    if "raw_structure" in df.columns or "material_id" in df.columns:
        return "materials"
    if "mol_id" in df.columns or "canonical_smiles" in df.columns or "smiles" in df.columns:
        return "qm9"
    return "unknown"


def _default_invariant_features(df: pl.DataFrame) -> List[str]:
    kind = _detect_dataset_kind(df)
    if kind == "materials":
        return INVARIANT_FEATURES_MATERIALS
    if kind == "qm9":
        return INVARIANT_FEATURES_QM9
    return INVARIANT_FEATURES


_QM9_FRAMES_CACHE: Dict[str, List[Atoms]] = {}


def _qm9_frames_cache_key(df: pl.DataFrame) -> str:
    if "mol_id" in df.columns:
        ids = df["mol_id"].cast(pl.Utf8).to_list()
        joined = "|".join(ids)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()
    if "canonical_smiles" in df.columns:
        smiles = df["canonical_smiles"].cast(pl.Utf8).to_list()
        joined = "|".join(smiles)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()
    if "smiles" in df.columns:
        smiles = df["smiles"].cast(pl.Utf8).to_list()
        joined = "|".join(smiles)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()
    return hashlib.md5(str(len(df)).encode("utf-8")).hexdigest()

def _compute_gap_statistic(X: np.ndarray, k: int, orig_inertia: float, n_refs: int = 5) -> float:
    """Computes the Gap Statistic comparing original inertia to uniform reference distributions."""
    if orig_inertia == 0:
        return 0.0
    ref_dispersions = []
    min_vals, max_vals = np.min(X, axis=0), np.max(X, axis=0)
    for _ in range(n_refs):
        random_data = np.random.uniform(min_vals, max_vals, size=X.shape)
        ref_kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(random_data)
        ref_dispersions.append(ref_kmeans.inertia_)
    return np.log(np.mean(ref_dispersions)) - np.log(orig_inertia)

def _compute_ncut(X: np.ndarray, labels: np.ndarray, n_neighbors: int = 10) -> float:
    """Computes the Normalized Cut (NCut) using a k-NN affinity graph."""
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
    A = 0.5 * (A + A.T) # Make symmetric
    degree = np.sum(A, axis=1)
    
    ncut = 0.0
    unique_labels = set(labels) - {-1} # exclude noise if present
    for c in unique_labels:
        mask = (labels == c)
        cut = np.sum(A[mask][:, ~mask])
        vol = np.sum(degree[mask])
        if vol > 0:
            ncut += cut / vol
    return ncut


def _infer_coherence_features(df: pl.DataFrame) -> Tuple[List[str], Optional[str]]:
    materials_cont = [
        "band_gap",
        "density",
        "energy_per_atom",
        "formation_energy_per_atom",
        "volume",
        "energy_above_hull",
    ]
    if all(c in df.columns for c in materials_cont) and "is_metal" in df.columns:
        return materials_cont, "is_metal"

    qm9_cont_default = [
        "gap", "homo", "lumo", "u0", "u", "h", "g", "cv", "mu", "alpha", "zpve", "r2"
    ]
    qm9_cont = [c for c in qm9_cont_default if c in df.columns]
    if qm9_cont:
        return qm9_cont, None

    return [], None


def get_overall_chemical_coherence(
    df,
    labels,
    continuous_features: Optional[List[str]] = None,
    discrete_feature: Optional[str] = None,
):
    df = df.with_columns(pl.Series(name="labels", values=labels))

    if continuous_features is None and discrete_feature is None:
        continuous_features, discrete_feature = _infer_coherence_features(df)

    if not continuous_features and not discrete_feature:
        return {}, float("nan")

    scaler = StandardScaler()
    if continuous_features:
        scaled_values = scaler.fit_transform(df.select(continuous_features).to_numpy())
    else:
        scaled_values = np.empty((df.height, 0))

    df_scaled = df.with_columns(
        [
            pl.Series(name, scaled_values[:, i])
            for i, name in enumerate(continuous_features)
        ]
    )

    cluster_sizes = df_scaled.group_by("labels").agg(pl.len().alias("count"))

    if continuous_features:
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
    else:
        cluster_coherence = df_scaled.select("labels").unique()

    if discrete_feature:
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
    else:
        metal_coherence = None

    cluster_scores = (
        cluster_coherence
        .join(metal_coherence, on="labels") if metal_coherence is not None else cluster_coherence
        .join(cluster_sizes, on="labels")
    )

    weights = cluster_scores["count"].to_numpy()

    results = {}

    for feature in continuous_features + ([discrete_feature] if discrete_feature else []):
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
    if len(frame) == 0:
        vol_per_atom = 0.0
    else:
        try:
            vol_per_atom = float(frame.get_volume()) / len(frame)
        except Exception:
            vol_per_atom = 0.0

    for i, atom in enumerate(frame):
        z = atom.number
        el = Element.from_Z(z)

        en = el.X if getattr(el, "X", None) else 0.0
        rad = el.atomic_radius if getattr(el, "atomic_radius", None) else 0.0
        mass = float(el.atomic_mass)
        mendeleev = el.mendeleev_no if getattr(el, "mendeleev_no", None) else 0

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
            "mendeleev": mendeleev,
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


def _build_qm9_frames_from_df(
    df: pl.DataFrame,
    seed: int = 40,
    invariant: bool = True,
) -> List[Atoms]:
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    if smiles_col not in df.columns:
        logger.warning("QM9 embedding requires a smiles column.")

    frames: List[Atoms] = []
    for row in df.iter_rows(named=True):
        smiles = row[smiles_col]
        mol_id = row.get("mol_id")
        mol = QM9Dataset._embed_molecule(
            smiles=smiles,
            seed=seed,
            invariant=invariant,
        )
        if mol is None:
            logger.warning(f"QM9 embedding failed for mol_id={mol_id} smiles={smiles}")
            continue

        conf = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = conf.GetPositions()
        charges = np.array(
            [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
            dtype=np.float64,
        )

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_initial_charges(charges)
        atoms.arrays["partial_charge"] = charges
        atoms.arrays["mass"] = atoms.get_masses()
        if mol_id is not None:
            atoms.info["mol_id"] = mol_id
        atoms.info["smiles"] = smiles
        frames.append(atoms)

    return frames


def build_invariant_matrix(
    df: pl.DataFrame,
    cutoff: float = 3.0,
    aggregated: bool = False,
    feature_keys: Optional[list] = None,
    frames: Optional[List[Atoms]] = None,
    qm9_seed: int = 40,
    qm9_invariant: bool = True,
    cache_qm9_frames: bool = True,
) -> list:
    """
    Iterates through the materials/QM9 dataframe, converts structures (or embeds molecules)
    to ASE Atoms, and computes the D x N invariant feature matrix for each item.
    """
    if feature_keys is None:
        feature_keys = _default_invariant_features(df)

    invariant_matrices = []
    kind = _detect_dataset_kind(df)

    if frames is None and kind == "qm9":
        cache_key = _qm9_frames_cache_key(df)
        if cache_qm9_frames and cache_key in _QM9_FRAMES_CACHE:
            frames = _QM9_FRAMES_CACHE[cache_key]
        else:
            frames = _build_qm9_frames_from_df(
                df,
                seed=qm9_seed,
                invariant=qm9_invariant,
            )
            if cache_qm9_frames:
                _QM9_FRAMES_CACHE[cache_key] = frames

    if frames is not None:
        for atoms in frames:
            matrix = _compute_invariant_feature_matrix_materials(
                atoms,
                cutoff=cutoff,
                aggregated=aggregated,
                feature_keys=feature_keys,
            )
            invariant_matrices.append(matrix)
    else:
        adaptor = AseAtomsAdaptor()
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
    def _has_required_categories(combo: tuple) -> bool:
        combo_set = set(combo)
        has_chemical = any(f in CHEMICAL_INVARIANT_FEATURES for f in combo_set)
        has_electronic = any(f in ELECTRONIC_INVARIANT_FEATURES for f in combo_set)
        has_structural = any(f in STRUCTURAL_INVARIANT_FEATURES for f in combo_set)
        return has_chemical and has_electronic and has_structural

    combinations_to_test = []
    for r in range(1, len(features) + 1):
        for combo in itertools.combinations(features, r):
            if _has_required_categories(combo):
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


def _valid_soap_component_counts(n_samples: int, candidates: Optional[List[int]] = None) -> List[int]:
    raw_candidates = candidates if candidates is not None else [i for i in range(2, MAX_SOAP_N_COMPONENTS + 1)]
    cleaned = sorted(
        {
            int(c)
            for c in raw_candidates
            if isinstance(c, (int, np.integer)) and 2 <= int(c) <= MAX_SOAP_N_COMPONENTS
        }
    )
    if n_samples <= 2:
        return []
    return [c for c in cleaned if c < n_samples]


def _soap_embeddings(
    df: pl.DataFrame,
    n_components_list: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    soap_array = np.array(df["soap_embedding"].to_list())
    embeddings: Dict[str, np.ndarray] = {"soap_raw": soap_array}

    valid_components = _valid_soap_component_counts(
        n_samples=len(soap_array),
        candidates=n_components_list,
    )

    for n_components in valid_components:
        reducers = get_reducers(n_components=n_components)
        for reducer_name, reducer in reducers.items():
            key = f"soap_{reducer_name}_c{n_components}"
            try:
                reduced = reducer.fit_transform(soap_array)
                embeddings[key] = reduced
                # Backward-compatible aliases for legacy consumers.
                if n_components == 3:
                    embeddings[f"soap_{reducer_name}"] = reduced
            except Exception as e:
                logger.warning(
                    f"Skipping SOAP reducer '{reducer_name}' at n_components={n_components}: {e}"
                )

    return embeddings


def _soap_items(
    df: pl.DataFrame,
    n_components_list: Optional[List[int]] = None,
) -> List[Tuple[str, np.ndarray]]:
    embeddings = _soap_embeddings(df, n_components_list=n_components_list)
    return list(embeddings.items())


def evaluate_hierarchical_combinations(
    df: pl.DataFrame,
    linkage: str = "average",
    k_min: int = 2,
    k_max: int = 20,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
    output_base_dir: str = "figures/materials/clustering",
    soap_n_components: Optional[List[int]] = None,
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using hierarchical clustering. Selects k by max silhouette and reports cophenetic corr.
    """
    results = []
    k_range = range(k_min, k_max + 1)

    if mode == "soap":
        items = _soap_items(df, n_components_list=soap_n_components)
        output_dir = Path(output_base_dir) / "hierarchical" / "soap_reduced"
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
            features = _default_invariant_features(df)

        logger.info("--- HIERARCHICAL STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced hierarchical search space to {len(combinations_to_test)} combinations.")

        output_dir = Path(output_base_dir) / "hierarchical" / "invariant_features"
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

    valid_models = results_df[results_df["Cophenetic Corr"] > 0.6]
    valid_models = valid_models[valid_models["Silhouette"] > 0.4]
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
    output_base_dir: str = "figures/materials/clustering",
    soap_n_components: Optional[List[int]] = None,
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using K-Means clustering with Davies-Bouldin-based k selection.
    """
    results = []
    k_range = list(range(k_min, k_max + 1))

    if mode == "soap":
        items = _soap_items(df, n_components_list=soap_n_components)
        output_dir = Path(output_base_dir) / "kmeans" / "soap_reduced"
        logger.info("--- KMEANS (SOAP) ---")
        for name, emb in tqdm(items, desc="Evaluating KMeans (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)

            silhouettes = []
            ch_scores = []
            db_scores = []
            inertias = []
            gap_scores = []
            labels_by_k = {}

            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(scaled_matrix)
                labels_by_k[k] = labels
                
                sil = silhouette_score(scaled_matrix, labels)
                silhouettes.append(sil)
                ch_scores.append(calinski_harabasz_score(scaled_matrix, labels))
                db_scores.append(davies_bouldin_score(scaled_matrix, labels))
                inertias.append(kmeans.inertia_)
                gap_scores.append(_compute_gap_statistic(scaled_matrix, k, kmeans.inertia_))

            best_k_sil = k_range[int(np.argmax(silhouettes))]
            best_k_gap = k_range[int(np.argmax(gap_scores))]

            best_labels = labels_by_k[best_k_sil] 
            
            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k (Silhouette)": best_k_sil,
                "Optimal k (Gap)": best_k_gap,
                "Gap Statistic": gap_scores[k_range.index(best_k_gap)],
                "Davies-Bouldin": db_scores[k_range.index(best_k_sil)],
                "Silhouette": silhouettes[k_range.index(best_k_sil)],
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = _default_invariant_features(df)

        logger.info("--- KMEANS STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced KMeans search space to {len(combinations_to_test)} combinations.")

        output_dir = Path(output_base_dir) / "kmeans" / "invariant_features"
        logger.info("--- KMEANS STAGE 2 & 3: Statistical Evaluation & DB Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating KMeans"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)

            silhouettes = []
            ch_scores = []
            db_scores = []
            inertias = []
            gap_scores = []
            labels_by_k = {}

            for k in k_range:
                
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(scaled_matrix)

                if labels is None or len(np.unique(labels)) < 2:
                    logger.warning(f"Combination [{combo_name}] with k={k} produced zero variance or invalid clusters. Skipping this k.")
                    continue

                labels_by_k[k] = labels
                
                sil = silhouette_score(scaled_matrix, labels)
                silhouettes.append(sil)
                ch_scores.append(calinski_harabasz_score(scaled_matrix, labels))
                db_scores.append(davies_bouldin_score(scaled_matrix, labels))
                inertias.append(kmeans.inertia_)
                gap_scores.append(_compute_gap_statistic(scaled_matrix, k, kmeans.inertia_))

            if not silhouettes:
                logger.warning(f"Combination [{combo_name}] produced no valid clusterings across all k. Skipping.")
                continue

            best_k_sil = k_range[int(np.argmax(silhouettes))]
            best_k_gap = k_range[int(np.argmax(gap_scores))]
            best_k_db = k_range[int(np.argmin(db_scores))]

            best_labels = labels_by_k[best_k_sil] 
            
            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k (Silhouette)": best_k_sil,
                "Optimal k (Gap)": best_k_gap,
                "Gap Statistic": gap_scores[k_range.index(best_k_gap)],
                "Davies-Bouldin": db_scores[k_range.index(best_k_sil)],
                "Optimal k (DB)": best_k_db,
                "Silhouette": silhouettes[k_range.index(best_k_sil)],
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No KMeans results produced.")
        return

    valid_models = results_df[results_df["Silhouette"] > 0.4].sort_values(by="Silhouette", ascending=False)
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

    k_col = "Optimal k (DB)" if "Optimal k (DB)" in top_20.columns else "Optimal k (Silhouette)"
    for index, (_, row) in enumerate(top_20.iterrows()):
        annot_text = f"k={int(row[k_col])} | DB: {row['Davies-Bouldin']:.2f}"
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
    output_base_dir: str = "figures/materials/clustering",
    soap_n_components: Optional[List[int]] = None,
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using Spectral Clustering. Selects k by max silhouette.
    """
    results = []
    k_range = list(range(k_min, k_max + 1))

    if mode == "soap":
        items = _soap_items(df, n_components_list=soap_n_components)
        output_dir = Path(output_base_dir) / "spectral" / "soap_reduced"
        logger.info("--- SPECTRAL (SOAP) ---")
        for name, emb in tqdm(items, desc="Evaluating Spectral (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)

            best_k = None
            best_ncut = np.inf
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
                        current_ncut = _compute_ncut(scaled_matrix, labels)

                        if current_ncut < best_ncut:
                            best_ncut = current_ncut
                            best_sil = sil
                            best_k = k
                            best_labels = labels
                except Exception:
                    continue

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name, # or 'name' for SOAP
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k (NCut)": best_k,
                "NCut Score": best_ncut,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = _default_invariant_features(df)

        logger.info("--- SPECTRAL STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced Spectral search space to {len(combinations_to_test)} combinations.")

        output_dir = Path(output_base_dir) / "spectral" / "invariant_features"
        logger.info("--- SPECTRAL STAGE 2 & 3: Statistical Evaluation & k Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating Spectral"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)

            best_k = None
            best_ncut = np.inf
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
                        current_ncut = _compute_ncut(scaled_matrix, labels)

                        if current_ncut < best_ncut:
                            best_ncut = current_ncut
                            best_sil = sil
                            best_k = k
                            best_labels = labels
                except Exception:
                    continue

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal k (NCut)": best_k,
                "NCut Score": best_ncut,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No Spectral results produced.")
        return

    valid_models = results_df[results_df["Silhouette"] > 0.4].sort_values(by="Silhouette", ascending=False)
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
        annot_text = f"k={int(row['Optimal k (NCut)'])} | NCut: {row['NCut Score']:.3f}"
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "ablation_study_top20_spectral.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 Spectral ablation plot to {out_path}")


def evaluate_dbscan_combinations(
    df: pl.DataFrame,
    eps_values: Optional[List[float]] = None,
    min_samples_values: Optional[List[int]] = None,
    features: Optional[List[str]] = None,
    mode: str = "invariant",
    output_base_dir: str = "figures/materials/clustering",
    soap_n_components: Optional[List[int]] = None,
):
    """
    Tests invariant feature combinations (mode="invariant") or SOAP projections (mode="soap")
    using DBSCAN and eps/min_samples sweep.
    """
    if eps_values is None:
        eps_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    if min_samples_values is None:
        min_samples_values = [2, 3, 5, 8, 10],
    else:
        min_samples_values = sorted(set(min_samples_values))

    results = []

    if mode == "soap":
        items = _soap_items(df, n_components_list=soap_n_components)
        output_dir = Path(output_base_dir) / "dbscan" / "soap_reduced"
        logger.info("--- DBSCAN (SOAP) ---")

        for name, emb in tqdm(items, desc="Evaluating DBSCAN (SOAP)"):
            scaled_matrix = _scaled_matrix(emb)
            rng = np.random.default_rng(42) 
            noise = rng.normal(0, 1e-8, scaled_matrix.shape)
            scaled_matrix = scaled_matrix + noise

            best_eps = None
            best_dbcv = -np.inf
            best_sil = -1.0
            best_labels = None
            best_min_samples = None
            best_noise_ratio = 1.0
            best_n_clusters = 0

            for min_samples_candidate in min_samples_values:
                for eps in eps_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples_candidate)
                    labels = dbscan.fit_predict(scaled_matrix)

                    noise_ratio = np.sum(labels == -1) / len(labels)
                    valid_clusters = set(labels) - {-1}
                    n_clusters = len(valid_clusters)

                    if n_clusters >= 2 and noise_ratio <= 0.50:
                        sil = silhouette_score(scaled_matrix, labels)
                        try:
                            current_dbcv = validity_index(scaled_matrix, labels)
                        except ValueError as e:
                            logger.warning(f"Error occurred while computing DBCV for combination {name}: {e}")
                            continue

                        if current_dbcv > best_dbcv:
                            best_dbcv = current_dbcv
                            best_sil = sil
                            best_eps = eps
                            best_labels = labels
                            best_min_samples = min_samples_candidate
                            best_noise_ratio = noise_ratio
                            best_n_clusters = n_clusters

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal eps": best_eps,
                "Optimal min_samples": best_min_samples,
                "Num Clusters": best_n_clusters,
                "Noise Ratio": best_noise_ratio,
                "DBCV Score": best_dbcv,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })
    else:
        if features is None:
            features = _default_invariant_features(df)

        logger.info("--- DBSCAN STAGE 1: Preliminary Feature Screening ---")
        screened_features, dropped = _screen_correlated_invariant_features(df, features)
        logger.info(f"Dropped highly correlated features: {dropped}")
        logger.info(f"Retained features for combinations: {screened_features}")

        combinations_to_test = _iter_invariant_combinations(screened_features)
        logger.info(f"Reduced DBSCAN search space to {len(combinations_to_test)} combinations.")

        output_dir = Path(output_base_dir) / "dbscan" / "invariant_features"
        logger.info("--- DBSCAN STAGE 2 & 3: Statistical Evaluation & Eps Selection ---")

        for feature_keys in tqdm(combinations_to_test, desc="Evaluating DBSCAN"):
            combo_name = " + ".join(feature_keys)

            scaled_matrix = _scaled_invariant_matrix(df, feature_keys)
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 1e-8, scaled_matrix.shape)
            scaled_matrix = scaled_matrix + noise

            best_eps = None
            best_dbcv = -np.inf
            best_sil = -1.0
            best_labels = None
            best_min_samples = None
            best_noise_ratio = 1.0
            best_n_clusters = 0

            for min_samples_candidate in min_samples_values:
                for eps in eps_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples_candidate)
                    labels = dbscan.fit_predict(scaled_matrix)

                    noise_ratio = np.sum(labels == -1) / len(labels)
                    valid_clusters = set(labels) - {-1}
                    n_clusters = len(valid_clusters)

                    if n_clusters >= 2 and noise_ratio <= 0.50:
                        sil = silhouette_score(scaled_matrix, labels)
                        current_dbcv = validity_index(scaled_matrix, labels)

                        if current_dbcv > best_dbcv:
                            best_dbcv = current_dbcv
                            best_sil = sil
                            best_eps = eps
                            best_labels = labels
                            best_min_samples = min_samples_candidate
                            best_noise_ratio = noise_ratio
                            best_n_clusters = n_clusters

            if best_labels is None:
                continue

            _, avg_coherence = get_overall_chemical_coherence(df, best_labels)

            results.append({
                "Combination": combo_name,
                "Feature Count": scaled_matrix.shape[1],
                "Optimal eps": best_eps,
                "Optimal min_samples": best_min_samples,
                "Num Clusters": best_n_clusters,
                "Noise Ratio": best_noise_ratio,
                "DBCV Score": best_dbcv,
                "Silhouette": best_sil,
                "Overall Chemical Coherence": avg_coherence,
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.error("No DBSCAN combinations yielded valid clusters with <50% noise. Try adjusting eps_values or min_samples_values.")
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
        annot_text = (
            f"eps={row['Optimal eps']} | min_samples={int(row['Optimal min_samples'])} "
            f"| k={int(row['Num Clusters'])} | Noise: {row['Noise Ratio']:.1%}"
        )
        plt.text(row["Silhouette"] + 0.005, index, annot_text, va="center", color="black", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "ablation_study_top20_dbscan.png"
    plt.savefig(out_path, dpi=300)
    logger.success(f"Saved top 20 DBSCAN ablation plot to {out_path}")


def get_reducers(n_components: int = 10):
    """
    Reducers safe for downstream K-Means / DBSCAN clustering.
    Notice the default n_components is higher here!
    """
    pca = PCA(n_components=n_components)
    # Polynomial kernel matches the math of the SOAP power spectrum
    kpca = KernelPCA(n_components=n_components, kernel='poly', fit_inverse_transform=True)
    umap_reducer = UMAP(n_components=n_components, metric='euclidean')
    isomap = Isomap(n_components=n_components)
    
    return {"pca": pca, "kpca": kpca, "umap": umap_reducer, "isomap": isomap}


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
    elif "mol_id" in df.columns:
        ids = df["mol_id"].cast(pl.Utf8).to_list()
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
    df = mp.load(limit=1000)

    # Invariant-only evaluations
    # evaluate_hierarchical_combinations(df, linkage="average", k_min=2, k_max=20)
    # evaluate_hierarchical_combinations(df, linkage="complete", k_min=2, k_max=20)
    # evaluate_kmeans_combinations(df, k_min=2, k_max=20)
    # evaluate_spectral_combinations(df, k_min=2, k_max=20)
    # evaluate_dbscan_combinations(df, min_samples_values=[2, 3, 5, 8, 10])

    # SOAP evaluations using the same functions
    # evaluate_hierarchical_combinations(df, linkage="average", k_min=2, k_max=20, mode="soap")
    # evaluate_hierarchical_combinations(df, linkage="complete", k_min=2, k_max=20, mode="soap")
    # evaluate_kmeans_combinations(df, k_min=2, k_max=20, mode="soap")
    # evaluate_spectral_combinations(df, k_min=2, k_max=20, mode="soap")
    evaluate_dbscan_combinations(
        df,
        min_samples_values=[2, 3, 5, 8, 10],
        mode="soap",
    )
