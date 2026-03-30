import json
from kmedoids import KMedoids
from pathlib import Path
import hashlib
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import unique_labels
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Element
from loguru import logger

from src.datasets import MaterialsProject
from src.non_euclidean import _compute_invariant_feature_matrix as invariant_matrix
from src.non_euclidean import _compute_soap_feature_matrices as soap_matrix
from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology
from src.helper_functions import create_chemiscope_viewer

def plot_evaluation(res):
    n = 95
    x_range = np.arange(2, n + 2)
    
    # Define metrics and whether we want the max or min
    metrics = [
        ('sil', 'Silhouette Score', 'max'),
        ('ch', 'Calinski-Harabasz Index', 'max'),
        ('db', 'Davies-Bouldin Index', 'min'),
    ]

    plt.figure(figsize=(20, 5))

    for i, (key, title, goal) in enumerate(metrics, 1):
        plt.subplot(1, 4, i)
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

    plt.tight_layout()
    plt.savefig('figures/materials/clustering/evaluation_soap_matrix.png', dpi=300)
    plt.show()

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

def _compute_invariant_feature_matrix_materials(frame: Atoms, cutoff: float = 3.0, aggregated=False) -> np.ndarray:
    """
    Maps a periodic material to a D x N matrix of invariant physical features.
    D is the fixed ambient dimension. N is the number of atoms in the unit/super cell.
    """
    # 1. Periodic neighbor list calculation
    # "ijd" returns: center atom index, neighbor atom index, and distance
    # ASE automatically respects the periodic boundary conditions (pbc=True) of the frame
    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)

    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    features = []
    
    # 2. Global invariant: Volume per atom
    # Replaces the Center of Mass distance, providing a scale-invariant packing metric
    vol_per_atom = frame.get_volume() / len(frame)

    # 3. Iterate over atoms to build the invariant matrix
    for i, atom in enumerate(frame):
        z = atom.number
        el = Element.from_Z(z)
        
        # Safely extract elemental properties (some noble gases lack Pauling electronegativity)
        en = el.X if getattr(el, 'X', None) else 0.0
        rad = el.atomic_radius if getattr(el, 'atomic_radius', None) else 0.0
        mass = atom.mass

        # 4. Local geometric invariants
        coord = len(neighbors[i])

        if coord > 0:
            avg_neighbor_z = float(np.mean(neighbors[i]))
            avg_neighbor_dist = float(np.mean(distances[i]))
        else:
            avg_neighbor_z = 0.0
            avg_neighbor_dist = 0.0

        # Assemble D-dimensional feature vector (D=8)
        feat_vector = [
            z,                  # Fundamental chemistry
            en,                 # Bonding behavior
            coord,              # Local geometry type
            avg_neighbor_dist,  # Local bond strength/size
            vol_per_atom        # Global crystal packing
        ]

        features.append(feat_vector)

    if aggregated:
        # Create the D x N matrix
        atom_matrix = np.array(features).T 

        # 5. AGGREGATION STEP: Collapse N atoms into fixed statistical features

        # Calculate the mean and standard deviation across the atoms (axis=1)
        mean_features = np.mean(atom_matrix, axis=1)
        std_features = np.std(atom_matrix, axis=1)
        
        # You can also add min/max if you want to capture the extremes!
        # min_features = np.min(atom_matrix, axis=1)
        # max_features = np.max(atom_matrix, axis=1)
        
        # Concatenate into a flat, fixed-length 1D array (Length: 16)
        crystal_features = np.concatenate([mean_features, std_features])
        return crystal_features
        
    # Return transposed to match D x N shape requirements
    return np.array(features).T

def build_invariant_matrix(df, cutoff: float = 3.0, aggregated: bool = False) -> list:
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
        matrix = _compute_invariant_feature_matrix_materials(atoms, cutoff=cutoff, aggregated=aggregated)
        
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


def get_distance_matrices(
    df: pl.DataFrame,
    soap_cache_dir: str = "data/Materials Project/soap_distances",
    invariant_cache_dir: str = "data/Materials Project/invariant_distances",
    cache_tag: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute (or load) a suite of distance matrices and return them in a dict.
    """
    cache_key = _distance_cache_key(df, cache_tag=cache_tag)
    soap_root = Path(soap_cache_dir)
    inv_root = Path(invariant_cache_dir)
    soap_root.mkdir(parents=True, exist_ok=True)
    inv_root.mkdir(parents=True, exist_ok=True)
    soap_cache_path = soap_root / f"{cache_key}.npz"
    inv_cache_path = inv_root / f"{cache_key}.npz"

    if soap_cache_path.exists() and inv_cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached SOAP distances from {soap_cache_path}...")
        logger.info(f"Loading cached invariant distances from {inv_cache_path}...")
        cached_soap = np.load(soap_cache_path, allow_pickle=False)
        cached_inv = np.load(inv_cache_path, allow_pickle=False)
        distances = {k: cached_soap[k] for k in cached_soap.files}
        distances.update({k: cached_inv[k] for k in cached_inv.files})
        return distances

    frames = get_ase_frames(df)
    soap_array = np.array(df['soap_embedding'].to_list())

    reducers = get_reducers()
    soap_pca = reducers['pca'].fit_transform(soap_array)
    soap_tsne = reducers['tsne'].fit_transform(soap_array)
    soap_umap = reducers['umap'].fit_transform(soap_array)
    soap_isomap = reducers['isomap'].fit_transform(soap_array)

    distance_pca_soap_matrix = squareform(pdist(soap_pca, metric='euclidean'))
    distance_tsne_soap_matrix = squareform(pdist(soap_tsne, metric='euclidean'))
    distance_umap_soap_matrix = squareform(pdist(soap_umap, metric='euclidean'))
    distance_isomap_soap_matrix = squareform(pdist(soap_isomap, metric='euclidean'))

    precomputed_feature_matrices = build_invariant_matrix(df, aggregated=False)
    precomputed_feature_matrices_aggregated = build_invariant_matrix(df, aggregated=True)

    distance_invariant_matrix_aggregated = squareform(pdist(precomputed_feature_matrices_aggregated, metric='euclidean'))
    
    distance_invariant_matrix_riemann = Riemann.distance_matrix(precomputed_feature_matrices=precomputed_feature_matrices,metric='affine-invariant')
    distance_invariant_matrix_grassmann = Grassmann.distance_matrix(precomputed_feature_matrices=precomputed_feature_matrices)
    distance_invariant_matrix_wasserstein = Wasserstein.distance_matrix(precomputed_feature_matrices=precomputed_feature_matrices)
    #distance_topological_bottleneck_matrix = PersistentHomology.distance_matrix(frames=frames, metric="bottleneck", max_homology_dim=2)
    #distance_topological_sliced_wasserstein_matrix = PersistentHomology.distance_matrix(frames=frames, metric="sliced-wasserstein", max_homology_dim=2)

    soap_distances = {
        "soap_pca": distance_pca_soap_matrix,
        "soap_tsne": distance_tsne_soap_matrix,
        "soap_umap": distance_umap_soap_matrix,
        "soap_isomap": distance_isomap_soap_matrix,
    }
    invariant_distances = {
        "invariant_aggregated": distance_invariant_matrix_aggregated,
        "invariant_riemann": distance_invariant_matrix_riemann,
        "invariant_grassmann": distance_invariant_matrix_grassmann,
        "invariant_wasserstein":distance_invariant_matrix_wasserstein,
       # "topological_bottleneck": distance_topological_bottleneck_matrix,
       # "topological_sliced_wasserstein": distance_topological_sliced_wasserstein_matrix
    }

    np.savez_compressed(soap_cache_path, **soap_distances)
    np.savez_compressed(inv_cache_path, **invariant_distances)
    logger.success(
        f"Cached {len(soap_distances)} SOAP distances to {soap_cache_path} "
        f"and {len(invariant_distances)} invariant distances to {inv_cache_path}."
    )

    distances = {}
    distances.update(soap_distances)
    distances.update(invariant_distances)
    return distances


if __name__ == '__main__':

    mp = MaterialsProject(add_soap=True, add_acsf=False)
    df = mp.load(limit=1000)
    distance_matrices = get_distance_matrices(df, cache_tag="initial_run", force_recompute=False)
    
    soap_matrix = np.array(df['soap_embedding'].to_list())
    distance_soap_matrix = squareform(pdist(soap_matrix, metric='euclidean'))
    run_evaluation(distance_soap_matrix) # best number of clusters : 4
    hier_labels = hierachial_clustering(distance_soap_matrix, 4)
    df = df.with_columns(pl.Series(name='hier_labels', values=hier_labels))
    chemical_coherence = get_overall_chemical_coherence(df, hier_labels)

    materials_invariant_matrix = build_invariant_matrix(df, cutoff=3.0)
    #grassmann_dist_matrix = Grassmann.distance_matrix(precomputed_feature_matrices=materials_invariant_matrix)
    #kmedoids_labels = kmedoids_clustering(grassmann_dist_matrix, 4)

    
