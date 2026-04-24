import os
import chemiscope
import json
import webbrowser
from io import StringIO
import kmedoids
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import selfies as sf

from pathlib import Path
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
from pymatgen.core import Structure
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.manifold import TSNE, Isomap, MDS
from geomstats.learning.pca import TangentPCA
from sklearn.decomposition import PCA, KernelPCA
from typing import Sequence, Optional, Iterable
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from umap import UMAP
from ase import Atoms

from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_structures(df, mol_id_list = None):
    # Extract both the IDs and the SMILES strings
    if mol_id_list is None:
        mol_ids = df["mol_id"].to_list()
    else:
        mol_ids = mol_id_list
        df = df.filter(pl.col("mol_id").is_in(mol_id_list))
    
    smiles_list = df["canonical_smiles"].to_list()
    
    structures = []  
    valid_indices = [] # We need to track this (see warning below)
    
    for i, (mol_id, s) in enumerate(zip(mol_ids, smiles_list)):
        if s is None:
            print(f"Skipping {mol_id}: SMILES is missing.")
            continue
            
        mol = Chem.MolFromSmiles(s)
        if mol is None: 
            print(f"Skipping {mol_id}: RDKit could not parse SMILES.")
            continue
            
        mol = Chem.AddHs(mol)

        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0: 
            print(f"Skipping {mol_id}: Failed to generate 3D conformer.")
            continue

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
        
        atoms = Atoms(numbers=numbers, positions=positions)
        structures.append(atoms)
        valid_indices.append(i)
        
    return structures, valid_indices


def align_frames_to_dist_matrix(
    frames: Sequence[Atoms],
    dist_matrix: Optional[np.ndarray] = None,
    return_matrix: bool = False,
) -> list[Atoms]:
    """
    Ensures frames match a precomputed pairwise distance matrix size.
    If more frames are provided than the matrix size, truncate deterministically.
    """
    aligned_frames = list(frames)
    if dist_matrix is None:
        return aligned_frames if not return_matrix else (aligned_frames, dist_matrix)

    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n_frames = len(aligned_frames)
    n_matrix = int(dist_matrix.shape[0])
    if n_frames != n_matrix:
        min_n = min(n_frames, n_matrix)
        logger.warning(
            "Aligning frames and dist_matrix sizes: "
            f"frames={n_frames}, dist_matrix={n_matrix}. Using first {min_n} entries."
        )
        aligned_frames = aligned_frames[:min_n]
        dist_matrix = dist_matrix[:min_n, :min_n]

    return aligned_frames if not return_matrix else (aligned_frames, dist_matrix)


def plot_molecules_with_py3dmol(
    molecules: Sequence[Atoms],
    n: Optional[int] = None,
    width: int = 320,
    height: int = 320,
    show_atom_labels: bool = False,
):
    """
    Show the first ``n`` QM9 molecules from ``QM9Dataset.get_molecules()`` next to each
    other using ``py3Dmol``.

    Parameters
    ----------
    molecules
        Sequence of ASE ``Atoms`` objects returned by ``QM9Dataset.get_molecules()``.
    n
        Optional number of molecules from ``molecules`` to plot. If ``None``, plot all.
    width
        Width in pixels for each molecule panel.
    height
        Height in pixels for each molecule panel.
    show_atom_labels
        If ``True``, show element labels on atoms.
    """
    if n is not None and n <= 0:
        raise ValueError("n must be a positive integer when provided.")

    selected_molecules = list(molecules if n is None else molecules[:n])
    if not selected_molecules:
        raise ValueError("No molecules were selected for plotting.")

    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "py3Dmol support requires the `py3Dmol` package to be available."
        ) from exc

    viewer = py3Dmol.view(
        viewergrid=(1, len(selected_molecules)),
        width=width * len(selected_molecules),
        height=height,
    )

    for idx, molecule in enumerate(selected_molecules):
        if not isinstance(molecule, Atoms):
            raise TypeError(
                f"Expected ASE Atoms objects from get_molecules(), got {type(molecule).__name__} at index {idx}."
            )

        pdb_buffer = StringIO()
        write(pdb_buffer, molecule, format="proteindatabank")
        viewer.addModel(pdb_buffer.getvalue(), "pdb", viewer=(0, idx))
        viewer.setStyle(
            {"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}},
            viewer=(0, idx),
        )
        if show_atom_labels:
            viewer.addPropertyLabels(
                "elem",
                "",
                {"fontSize": 10, "showBackground": False, "alignment": "center"},
                viewer=(0, idx),
            )
        viewer.zoomTo(viewer=(0, idx))

    return viewer


def plot_molecules_with_pymol(
    frames: Sequence[Atoms],
    n: int,
    width: int = 320,
    height: int = 320,
    show_atom_labels: bool = False,
):
    """Backward-compatible alias that now uses ``py3Dmol``."""
    return plot_molecules_with_py3dmol(
        molecules=frames,
        n=n,
        width=width,
        height=height,
        show_atom_labels=show_atom_labels,
    )


def plot_atoms_with_pymol(
    frames: Sequence[Atoms],
    n: int,
    width: int = 320,
    height: int = 320,
    show_atom_labels: bool = False,
):
    """Backward-compatible alias that now uses ``py3Dmol``."""
    return plot_molecules_with_py3dmol(
        molecules=frames,
        n=n,
        width=width,
        height=height,
        show_atom_labels=show_atom_labels,
    )


def _project_distance_matrix(
    dist_matrix: np.ndarray,
    projection_method: str = "PCA",
    random_state: int = 42,
) -> np.ndarray:
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    method = projection_method.strip().lower()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(dist_matrix)

    if method == "kpca":
        d2 = np.square(dist_matrix.astype(np.float64))
        non_zero = d2[d2 > 0]
        gamma = 1.0 / np.median(non_zero) if non_zero.size else 1.0
        kernel = np.exp(-gamma * d2)
        reducer = KernelPCA(n_components=2, kernel="precomputed", random_state=random_state)
        return reducer.fit_transform(kernel)

    if method in {"tsne", "t-sne"}:
        reducer = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=random_state,
        )
        return reducer.fit_transform(dist_matrix)

    if method == "umap":
        reducer = UMAP(n_components=2, metric="precomputed", random_state=random_state)
        return reducer.fit_transform(dist_matrix)

    if method == "isomap":
        reducer = Isomap(n_components=2, metric="precomputed")
        return reducer.fit_transform(dist_matrix)

    if method == "mds":
        reducer = MDS(n_components=2, metric=True, dissimilarity="precomputed", random_state=random_state)
        return reducer.fit_transform(dist_matrix)

    raise ValueError(
        "Unsupported projection_method. Use one of: "
        "'PCA', 'KPCA', 't-SNE', 'UMAP', 'ISOMAP', 'MDS'."
    )


def plot_distance_matrix_projection(
    dist_matrix: np.ndarray,
    fingerprint: str,
    distance_metric: str,
    projection_method: str = "PCA",
    dataset_name: str = "qm9",
    labels: Optional[Sequence] = None,
    clustering_method: Optional[str] = None,
    title: Optional[str] = None,
    output_base_dir: str = "figures",
    point_size: int = 55,
    alpha: float = 0.9,
    random_state: int = 42,
):
    """
    Project a distance matrix to 2D, create a publication-friendly scatter plot,
    and save it under ``figures/<dataset>/clustering/<distance_metric>/<fingerprint>/``.

    Parameters
    ----------
    dist_matrix
        Square pairwise distance matrix.
    fingerprint
        Fingerprint/descriptor name used to generate the distance matrix.
    distance_metric
        Distance metric used to generate the distance matrix.
    projection_method
        2D projection method. Defaults to ``PCA``.
    dataset_name
        Dataset name used in the output folder path.
    labels
        Optional labels for coloring points.
    clustering_method
        Optional name of the clustering method used to generate ``labels``.
    title
        Optional custom plot title.
    output_base_dir
        Root output directory. Defaults to ``figures``.
    point_size
        Scatter marker size.
    alpha
        Marker opacity.
    random_state
        Random seed for stochastic projection methods.
    """
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n_samples = dist_matrix.shape[0]
    if labels is not None and len(labels) != n_samples:
        raise ValueError("labels must have the same length as dist_matrix.")

    coords = _project_distance_matrix(
        dist_matrix=dist_matrix,
        projection_method=projection_method,
        random_state=random_state,
    )

    projection_label = projection_method.upper() if projection_method.lower() != "t-sne" else "t-SNE"
    fingerprint_slug = str(fingerprint).strip().lower().replace(" ", "_")
    metric_slug = str(distance_metric).strip().lower().replace(" ", "_")
    dataset_slug = str(dataset_name).strip().lower().replace(" ", "_")
    clustering_slug = (
        str(clustering_method).strip().lower().replace(" ", "_")
        if clustering_method is not None
        else None
    )

    output_dir = Path(output_base_dir) / dataset_slug / "clustering" / metric_slug / fingerprint_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = (
        f"{projection_label.lower().replace('-', '_')}_{clustering_slug}_projection"
        if clustering_slug
        else f"{projection_label.lower().replace('-', '_')}_projection"
    )
    output_path = output_dir / f"{filename_prefix}.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))

    if labels is None:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            alpha=alpha,
            c="#1f77b4",
            edgecolors="white",
            linewidths=0.6,
        )
    else:
        labels_arr = np.asarray(labels)
        unique_labels = np.unique(labels_arr)
        cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))

        for idx, label in enumerate(unique_labels):
            mask = labels_arr == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=alpha,
                color=cmap(idx),
                edgecolors="white",
                linewidths=0.6,
                label=str(label),
            )
        if len(unique_labels) <= 20:
            ax.legend(title="Label", frameon=True, loc="best")

    ax.set_title(
        title
        or f"{dataset_slug.upper()} {projection_label} Projection\n"
           f"Fingerprint: {fingerprint} | Distance: {distance_metric}"
           + (
               f" | Clustering: {clustering_method}"
               if clustering_method is not None
               else ""
           ),
        fontsize=14,
    )
    ax.set_xlabel(f"{projection_label} Component 1", fontsize=12)
    ax.set_ylabel(f"{projection_label} Component 2", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {projection_label} projection plot to {output_path}")

    return {
        "coords": coords,
        "figure_path": output_path,
        "output_dir": output_dir,
        "clustering_method": clustering_method,
    }

def get_distances(frames, frames_ph=None, dataset = 'QM9', include_ph=True):
    expected_n = len(frames)
    data_dir = f'data/{dataset}/distance_matrices_n{expected_n}'
    os.makedirs(data_dir, exist_ok=True)
    print(os.listdir(data_dir))

    matrix_tasks = {
        'grassmann': {
            'path': f'{data_dir}/dist_matrix_grassmann.npy',
            'compute': lambda: Grassmann.distance_matrix(frames)
        },
        'euclidean_riemann': {
            'path': f'{data_dir}/dist_matrix_euclidean_riemann.npy',
            'compute': lambda: Riemann.distance_matrix(frames, metric='log-euclidean')
        },
        'affine_riemann': {
            'path': f'{data_dir}/dist_matrix_affine_riemann.npy',
            'compute': lambda: Riemann.distance_matrix(frames, metric='affine-invariant')
        },
        'wasserstein': {
            'path': f'{data_dir}/dist_matrix_wasserstein.npy',
            'compute': lambda: Wasserstein.distance_matrix(frames)
        },
        'ph_bottleneck': {
            'path': f'{data_dir}/persistent_dist_matrix_bottleneck.npy',
            'compute': lambda: PersistentHomology.distance_matrix(frames, metric="bottleneck")
        },
        'ph_sliced_wasserstein': {
            'path': f'{data_dir}/persistent_dist_matrix_sw.npy',
            'compute': lambda: PersistentHomology.distance_matrix(frames, metric="sliced_wasserstein")
        }
    }

    matrices = {}

    for name, task in matrix_tasks.items():
        file_path = task['path']
        
        if os.path.exists(file_path):
            logger.info(f"Loading {name} distance matrix...")
            matrices[name] = np.load(file_path)
        else:
            if not include_ph and name.startswith('ph_'):
                logger.warning("Skipping PH task")
                continue
            logger.info(f"Computing {name} distance matrix...")
            matrices[name] = task['compute']()
            np.save(file_path, matrices[name])

        mat = np.asarray(matrices[name])
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{name} distance matrix must be square. Got shape {mat.shape}."
            )
        if mat.shape[0] != expected_n:
            raise ValueError(
                f"{name} distance matrix size mismatch: expected ({expected_n}, {expected_n}), got {mat.shape}."
            )

    logger.success("✓ All distance matrices are ready!")
    
    return matrices


def find_best_kmedoids_k(
    dist_matrix: np.ndarray,
    k_range: Iterable[int] = range(2, 15),
    random_state: int = 42,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate K-Medoids clustering over a range of k using:
    - inertia (sum of distances to assigned medoids)
    - silhouette score (precomputed distances)
    - Calinski-Harabasz score (on feature_matrix if provided, else on dist_matrix)
    """
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n = dist_matrix.shape[0]
    k_list = [int(k) for k in k_range if 2 <= int(k) <= n - 1]
    if not k_list:
        raise ValueError("k_range must contain values in [2, n-1].")

    results = {"k": [], "inertia": [], "silhouette": [], "ch": []}

    use_features_for_ch = feature_matrix is not None
    if feature_matrix is None:
        logger.warning(
            "feature_matrix is None; Calinski-Harabasz will be computed on dist_matrix "
            "(treated as features). Provide feature_matrix for a more meaningful CH score."
        )

    for k in k_list:
        model = kmedoids.KMedoids(n_clusters=k, metric="precomputed", random_state=random_state)
        labels = model.fit_predict(dist_matrix)
        medoid_indices = model.medoid_indices_

        inertia = float(
            sum(dist_matrix[i, medoid_indices[labels[i]]] for i in range(n))
        )

        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

        if use_features_for_ch:
            if feature_matrix.shape[0] != n:
                raise ValueError(
                    "feature_matrix must have the same number of rows as dist_matrix."
                )
            ch = float(calinski_harabasz_score(feature_matrix, labels))
        else:
            ch = float(calinski_harabasz_score(dist_matrix, labels))

        results["k"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil)
        results["ch"].append(ch)

    best_k = {
        "inertia": results["k"][int(np.argmin(results["inertia"]))],
        "silhouette": results["k"][int(np.argmax(results["silhouette"]))],
        "ch": results["k"][int(np.argmax(results["ch"]))],
    }

    return {"results": results, "best_k": best_k}


def _gaussian_affinity_from_distance(dist_matrix: np.ndarray) -> np.ndarray:
    d2 = np.square(np.asarray(dist_matrix, dtype=np.float64))
    non_zero = d2[d2 > 0]
    gamma = 1.0 / np.median(non_zero) if non_zero.size else 1.0
    return np.exp(-gamma * d2)


def _safe_silhouette_from_precomputed_distance(
    dist_matrix: np.ndarray,
    labels: Sequence,
) -> Optional[float]:
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) < 2:
        return None
    if len(unique_labels) >= len(labels):
        return None
    try:
        return float(silhouette_score(dist_matrix, labels, metric="precomputed"))
    except Exception:
        return None


def _safe_cluster_quality_scores(
    dist_matrix: np.ndarray,
    labels: Sequence,
) -> dict[str, Optional[float]]:
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {"silhouette": None, "db_index": None, "ch_index": None}

    feature_matrix = np.asarray(dist_matrix, dtype=np.float64)

    try:
        silhouette = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
    except Exception:
        silhouette = None

    try:
        db_index = float(davies_bouldin_score(feature_matrix, labels))
    except Exception:
        db_index = None

    try:
        ch_index = float(calinski_harabasz_score(feature_matrix, labels))
    except Exception:
        ch_index = None

    return {
        "silhouette": silhouette,
        "db_index": db_index,
        "ch_index": ch_index,
    }


def evaluate_distance_matrix_clustering_sweep(
    dist_matrix: np.ndarray,
    fingerprint: str,
    distance_metric: str,
    dataset_name: str = "qm9",
    k_range: Iterable[int] = range(2, 21),
    eps_values: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0),
    min_samples_values: Sequence[int] = (2, 3, 5, 8, 10),
    hierarchical_linkage: str = "average",
    spectral_assign_labels: str = "kmeans",
    output_base_dir: str = "figures",
    title: Optional[str] = None,
    random_state: int = 42,
):
    """
    Sweep K-Medoids, hierarchical, spectral clustering, and DBSCAN over a distance
    matrix and plot silhouette, Davies-Bouldin, and Calinski-Harabasz against the
    number of clusters (2..20 by default).

    DBSCAN is swept over ``eps_values`` and ``min_samples_values``. Its curve shows the
    best silhouette score achieved for each produced cluster count.
    """
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n = dist_matrix.shape[0]
    k_list = [int(k) for k in k_range if 2 <= int(k) <= min(20, n - 1)]
    if not k_list:
        raise ValueError("k_range must contain at least one value in [2, min(20, n-1)].")

    fingerprint_slug = str(fingerprint).strip().lower().replace(" ", "_")
    metric_slug = str(distance_metric).strip().lower().replace(" ", "_")
    dataset_slug = str(dataset_name).strip().lower().replace(" ", "_")
    output_dir = Path(output_base_dir) / dataset_slug / "clustering" / metric_slug / fingerprint_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    affinity = _gaussian_affinity_from_distance(dist_matrix)

    results = {
        "kmedoids": [],
        "hierarchical": [],
        "spectral": [],
        "dbscan": [],
    }
    dbscan_grid_results = []

    for k in tqdm(k_list, desc="Evaluation k"):
        try:
            model = kmedoids.KMedoids(n_clusters=k, metric="precomputed", random_state=random_state)
            labels = model.fit_predict(dist_matrix)
            scores = _safe_cluster_quality_scores(dist_matrix, labels)
            results["kmedoids"].append({"k": k, **scores})
        except Exception as exc:
            logger.warning(f"K-Medoids failed for k={k}: {exc}")
            results["kmedoids"].append({"k": k, "silhouette": None, "db_index": None, "ch_index": None})

        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage=hierarchical_linkage,
            )
            labels = model.fit_predict(dist_matrix)
            scores = _safe_cluster_quality_scores(dist_matrix, labels)
            results["hierarchical"].append({"k": k, **scores})
        except Exception as exc:
            logger.warning(f"Hierarchical clustering failed for k={k}: {exc}")
            results["hierarchical"].append({"k": k, "silhouette": None, "db_index": None, "ch_index": None})

        try:
            model = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels=spectral_assign_labels,
                random_state=random_state,
            )
            labels = model.fit_predict(affinity)
            scores = _safe_cluster_quality_scores(dist_matrix, labels)
            results["spectral"].append({"k": k, **scores})
        except Exception as exc:
            logger.warning(f"Spectral clustering failed for k={k}: {exc}")
            results["spectral"].append({"k": k, "silhouette": None, "db_index": None, "ch_index": None})

    best_dbscan_by_k = {}
    for eps in tqdm(eps_values, desc="Evaluating epsilon and min samples"):
        for min_samples in min_samples_values:
            try:
                model = DBSCAN(
                    eps=float(eps),
                    min_samples=int(min_samples),
                    metric="precomputed",
                )
                labels = model.fit_predict(dist_matrix)
                unique_clusters = sorted({int(x) for x in labels if int(x) != -1})
                n_clusters = len(unique_clusters)
                if n_clusters < 2 or n_clusters > 20:
                    continue

                scores = _safe_cluster_quality_scores(dist_matrix, labels)
                if scores["silhouette"] is None:
                    continue

                current = best_dbscan_by_k.get(n_clusters)
                candidate = {
                    "k": n_clusters,
                    **scores,
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "noise_ratio": float(np.mean(np.asarray(labels) == -1)),
                }
                dbscan_grid_results.append(candidate.copy())
                if current is None or scores["silhouette"] > current["silhouette"]:
                    best_dbscan_by_k[n_clusters] = candidate
            except Exception as exc:
                logger.warning(
                    f"DBSCAN failed for eps={eps}, min_samples={min_samples}: {exc}"
                )

    for k in k_list:
        entry = best_dbscan_by_k.get(
            k,
            {
                "k": k,
                "silhouette": None,
                "db_index": None,
                "ch_index": None,
                "eps": None,
                "min_samples": None,
                "noise_ratio": None,
            },
        )
        results["dbscan"].append(entry)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    plot_specs = {
        "kmedoids": {"label": "K-Medoids", "color": "#1f77b4", "marker": "o"},
        "hierarchical": {"label": f"Hierarchical ({hierarchical_linkage})", "color": "#ff7f0e", "marker": "s"},
        "spectral": {"label": "Spectral", "color": "#2ca02c", "marker": "^"},
        "dbscan": {"label": "DBSCAN (best per cluster count)", "color": "#d62728", "marker": "D"},
    }

    metric_specs = [
        ("silhouette", "Silhouette score"),
        ("db_index", "Davies-Bouldin index"),
        ("ch_index", "Calinski-Harabasz index"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        for method_name, entries in results.items():
            xs = [entry["k"] for entry in entries if entry.get(metric_key) is not None]
            ys = [entry[metric_key] for entry in entries if entry.get(metric_key) is not None]
            if xs:
                spec = plot_specs[method_name]
                ax.plot(
                    xs,
                    ys,
                    label=spec["label"],
                    color=spec["color"],
                    marker=spec["marker"],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlim(min(k_list), max(k_list))
        ax.set_xticks(k_list)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, loc="best")

    axes[0].set_title(
        title
        or (
            f"{dataset_slug.upper()} Clustering Sweep\n"
            f"Fingerprint: {fingerprint} | Distance: {distance_metric}"
        ),
        fontsize=14,
    )
    for ax in axes:
        ax.set_xlabel("Number of clusters", fontsize=12)

    fig.tight_layout()
    plot_path = output_dir / "clustering_sweep_2_to_20_clusters.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    dbscan_plot_path = output_dir / "dbscan_parameter_sweep.png"
    if dbscan_grid_results:
        fig_dbscan, axes_dbscan = plt.subplots(1, 4, figsize=(24, 6), sharex=False)
        min_samples_sorted = sorted({entry["min_samples"] for entry in dbscan_grid_results})
        cmap = plt.get_cmap("tab10", max(len(min_samples_sorted), 1))

        dbscan_metric_specs = [
            ("silhouette", "Silhouette score"),
            ("db_index", "Davies-Bouldin index"),
            ("ch_index", "Calinski-Harabasz index"),
            ("k", "Number of clusters"),
        ]

        for ax, (metric_key, metric_label) in zip(axes_dbscan, dbscan_metric_specs):
            for idx, min_samples in enumerate(min_samples_sorted):
                entries = sorted(
                    [entry for entry in dbscan_grid_results if entry["min_samples"] == min_samples],
                    key=lambda entry: entry["eps"],
                )
                xs = [entry["eps"] for entry in entries if entry.get(metric_key) is not None]
                ys = [entry[metric_key] for entry in entries if entry.get(metric_key) is not None]
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        label=f"min_samples={min_samples}",
                        color=cmap(idx),
                        marker="o",
                        linewidth=2,
                        markersize=5,
                    )

            ax.set_xlabel("eps", fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=True, loc="best")

        axes_dbscan[0].set_title(
            f"{dataset_slug.upper()} DBSCAN Parameter Sweep\n"
            f"Fingerprint: {fingerprint} | Distance: {distance_metric}",
            fontsize=14,
        )
        fig_dbscan.tight_layout()
        fig_dbscan.savefig(dbscan_plot_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        logger.warning("No valid DBSCAN parameter combinations produced clusterings to plot.")
        dbscan_plot_path = None

    results_path = output_dir / "clustering_sweep_2_to_20_clusters.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "fingerprint": fingerprint,
                "distance_metric": distance_metric,
                "hierarchical_linkage": hierarchical_linkage,
                "spectral_assign_labels": spectral_assign_labels,
                "k_values": k_list,
                "eps_values": list(eps_values),
                "min_samples_values": list(min_samples_values),
                "results": results,
                "dbscan_grid_results": dbscan_grid_results,
            },
            f,
            indent=2,
        )

    logger.success(f"Saved clustering sweep plot to {plot_path}")
    if dbscan_plot_path is not None:
        logger.success(f"Saved DBSCAN evaluation plot to {dbscan_plot_path}")
    logger.success(f"Saved clustering sweep results to {results_path}")

    return {
        "results": results,
        "dbscan_grid_results": dbscan_grid_results,
        "plot_path": plot_path,
        "dbscan_plot_path": dbscan_plot_path,
        "results_path": results_path,
        "output_dir": output_dir,
    }

def _open_in_browser(path_or_url):
    try:
        if not path_or_url:
            return
        if isinstance(path_or_url, str) and (
            path_or_url.startswith("http://") or path_or_url.startswith("https://")
        ):
            webbrowser.open(path_or_url)
            return
        webbrowser.open(Path(path_or_url).resolve().as_uri())
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")


def _build_chemiscope_frames(df: pl.DataFrame, qm9_seed: int = 40, qm9_invariant: bool = True):
    """Build ASE frames for either materials (raw_structure) or QM9 (smiles)."""
    if "raw_structure" in df.columns:
        frames = []
        adaptor = AseAtomsAdaptor()
        for struct_json in df["raw_structure"]:
            struct = Structure.from_dict(json.loads(struct_json))
            frames.append(adaptor.get_atoms(struct))
        return frames, list(range(df.height)), "materials"

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles" if "smiles" in df.columns else None
    if smiles_col is None:
        raise ValueError("DataFrame must contain either 'raw_structure' or a SMILES column.")

    # Local import avoids circular imports at module load time.
    from src.datasets import QM9Dataset

    frames = []
    valid_indices = []
    for i, row in enumerate(df.iter_rows(named=True)):
        smiles = row.get(smiles_col)
        mol_id = row.get("mol_id")
        if not smiles:
            continue

        mol = QM9Dataset._embed_molecule(smiles=smiles, seed=qm9_seed, invariant=qm9_invariant)
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
        valid_indices.append(i)

    return frames, valid_indices, "qm9"


def create_chemiscope_viewer(df, dist_matrix, labels, reduction_method='t-SNE'):
    print("Running " + reduction_method + " dimensionality reduction...")

    dist_matrix = np.asarray(dist_matrix)
    labels = np.asarray(labels)
    if labels.shape[0] != df.height:
        raise ValueError(
            f"labels length ({labels.shape[0]}) must match dataframe rows ({df.height})."
        )

    print("Converting structures/molecules to ASE Atoms for Chemiscope...")
    frames, valid_indices, dataset_kind = _build_chemiscope_frames(df)
    if not frames:
        raise ValueError("No valid structures/molecules could be converted for Chemiscope.")

    # Keep dataframe/labels/distances aligned with successfully built frames.
    if len(valid_indices) != df.height:
        df = df[valid_indices]
        labels = labels[valid_indices]
        if dist_matrix.ndim == 2:
            if dist_matrix.shape[0] == dist_matrix.shape[1]:
                dist_matrix = dist_matrix[np.ix_(valid_indices, valid_indices)]
            else:
                dist_matrix = dist_matrix[valid_indices]

    if dist_matrix.shape[0] != len(frames):
        min_n = min(dist_matrix.shape[0], len(frames), labels.shape[0], df.height)
        logger.warning(
            "Chemiscope alignment mismatch; truncating to first "
            f"{min_n} entries (frames={len(frames)}, dist={dist_matrix.shape}, labels={labels.shape[0]}, df={df.height})."
        )
        frames = frames[:min_n]
        labels = labels[:min_n]
        df = df.head(min_n)
        if dist_matrix.ndim == 2 and dist_matrix.shape[0] == dist_matrix.shape[1]:
            dist_matrix = dist_matrix[:min_n, :min_n]
        else:
            dist_matrix = dist_matrix[:min_n]

    if reduction_method == 't-SNE':
        if dist_matrix.shape[1] != dist_matrix.shape[0]:
            tsne = TSNE(n_components=2, random_state=42)
        else:
            tsne = TSNE(n_components=2, metric='precomputed',init='random', random_state=42)
        coords = tsne.fit_transform(dist_matrix)
    elif reduction_method in ['UMAP', 'umap']:
        if dist_matrix.shape[1] != dist_matrix.shape[0]:
            reducer = UMAP(metric='euclidean', random_state=42)
        else:
            reducer = UMAP(metric='precomputed', random_state=42)
        coords = reducer.fit_transform(dist_matrix)
    elif reduction_method == 'PCA':
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(dist_matrix)
    elif reduction_method in ['KPCA', 'kpca']:
        if dist_matrix.shape[1] == dist_matrix.shape[0]:
            # Distances are converted to an RBF affinity for precomputed-kernel KPCA.
            d2 = np.square(dist_matrix.astype(np.float64))
            non_zero = d2[d2 > 0]
            gamma = 1.0 / np.median(non_zero) if non_zero.size else 1.0
            kernel = np.exp(-gamma * d2)
            kpca = KernelPCA(n_components=2, kernel='precomputed', random_state=42)
            coords = kpca.fit_transform(kernel)
        else:
            kpca = KernelPCA(n_components=2, random_state=42)
            coords = kpca.fit_transform(dist_matrix)
    elif reduction_method == 'ISOMAP':
        isomap = Isomap(n_components=2, metric='precomputed')
        coords = isomap.fit_transform(dist_matrix)
    elif reduction_method == 'MDS':
        mds = MDS(n_components=2, metric='precomputed', random_state=42)
        coords = mds.fit_transform(dist_matrix)
    elif reduction_method == 'PGA':
        pga = TangentPCA(n_components=2, space="metric")
        coords = pga.fit_transform(dist_matrix)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")

    print("Assembling properties for Chemiscope...")

    if "fraction_csp1" in df.columns:
        df = df.with_columns(
            pl.format(
                "{} / {} / {}", 
                pl.col("fraction_csp1").round(2), 
                pl.col("fraction_csp2").round(2), 
                pl.col("fraction_csp3").round(2)
            ).alias("sp_ratio_set")
        )

    properties = {
        f"{reduction_method}_1": coords[:, 0],
        f"{reduction_method}_2": coords[:, 1],
        "Cluster": labels.astype(int),
    }

    if dataset_kind == "materials":
        materials_cols = {
            "Formula": "formula_pretty",
            "Band_Gap": "band_gap",
            "Energy_per_Atom": "energy_per_atom",
            "Is_Metal": "is_metal",
            "crystal_system": "crystal_system",
            "Density": "density",
            "Space_Group": "space_group",
            "energy_above_hull": "energy_above_hull",
            "formation_energy_per_atom": "formation_energy_per_atom",
            "volume": "volume",
            "num_sites": "num_sites",
            "max_en_diff": "max_en_diff",
            "avg_bond_length": "avg_bond_length",
            "max_bond_length": "max_bond_length",
            "material_id": "material_id",
        }
        for prop_name, col_name in materials_cols.items():
            if col_name in df.columns:
                # FIX: Handle null values safely for Chemiscope
                if df[col_name].dtype in pl.NUMERIC_DTYPES:
                    properties[prop_name] = df[col_name].fill_null(float('nan')).to_list()
                else:
                    properties[prop_name] = df[col_name].fill_null("N/A").to_list()

    else:
        qm9_cols = {
            "mol_id": "mol_id",
            "Formula": "formula",
            #"smiles": "canonical_smiles" if "canonical_smiles" in df.columns else "smiles",
            #"sefiles": "selfies",
            "num_atoms": "num_atoms",
            "structure_class": "structure_class",
            "functional_groups": "functional_groups",
            "hybridization_ratio": "sp_ratio_set",
            #"scaffold": "scaffold",
            "outlier_type": "outlier_category",
            "hdbscan_label": "hdbscan_label",
            "lof_label": "lof_label",
            "knn_label": "knn_label",
        }
        for prop_name, col_name in qm9_cols.items():
            if col_name in df.columns:
                # FIX: Replace SMILES/SELFIES with their lengths directly
                if prop_name in ["smiles", "sefiles"]:
                    values = df[col_name].to_list()
                    properties[prop_name + "_length"] = [
                        len(v) if isinstance(v, str) else 0 for v in values
                    ]
                # FIX: Properly label non-outliers instead of crashing on 'None'
                elif col_name == "outlier_category":
                    properties[prop_name] = df[col_name].fill_null("Native QM9").to_list()
                # FIX: Safety net for any other numerical or categorical nulls
                elif df[col_name].dtype in pl.NUMERIC_DTYPES:
                    properties[prop_name] = df[col_name].fill_null(float('nan')).to_list()
                else:
                    properties[prop_name] = df[col_name].fill_null("N/A").to_list()



    settings = {
        "map": {
            "x": {"property": f"{reduction_method}_1"},
            "y": {"property": f"{reduction_method}_2"},
            "color": {"property": "Cluster"},
            "size": {"factor": 35}
        },
        "structure": [{"keepOrientation": True}],
    }
    if dataset_kind == "materials":
        settings["structure"][0]["supercell"] = [2, 2, 2]

    print("Generating Chemiscope widget...")
    title_prefix = "Materials Project" if dataset_kind == "materials" else "QM9"
    output_prefix = "materials" if dataset_kind == "materials" else "qm9"

    if hasattr(chemiscope, "write_html"):
        output_html = f"data/chemiscope/{output_prefix}_{reduction_method}_clustering.html"
        chemiscope.write_html(
            output_html,
            frames=frames,
            properties=properties,
            settings=settings,
            title=f"{title_prefix} - {reduction_method} Clustering",
        )
        print(f"Saved interactive viewer to: {output_html}")
        _open_in_browser(output_html)
        return chemiscope.show(frames=frames, properties=properties, settings=settings)

    output_json = f"{output_prefix}_{reduction_method}_clustering.json"
    if not hasattr(chemiscope, "write_input"):
        raise AttributeError(
            "chemiscope does not provide write_html or write_input; "
            "please upgrade/downgrade chemiscope to a supported version."
        )

    chemiscope.write_input(
        output_json,
        structures=frames,
        properties=properties,
        settings=settings,
        metadata={"name": f"{title_prefix} - {reduction_method} Clustering"},
    )
    print(f"Saved Chemiscope input to: {output_json}")
    viewer = chemiscope.show_input(output_json)
    viewer_url = getattr(viewer, "url", None)
    if viewer_url:
        _open_in_browser(viewer_url)
    else:
        print(
            "If the viewer does not open automatically, run "
            f"`chemiscope show {output_prefix}_{reduction_method}_clustering.json`."
        )
    return viewer

def average_numeric_by_cluster(df: pl.DataFrame, labels_col="cluster_label") -> pl.DataFrame:
    """
    Groups a Polars DataFrame by 'cluster_label' and returns 
    the mean of all numeric columns along with the count of elements.
    Includes the 'token_to_atom_ratio' to measure syntactic complexity,
    and calculates categorical enrichment for scaffolds, material properties,
    and outlier categories.
    """
    if "raw_token_count" in df.columns and "structure_class" in df.columns:
        agg_exprs = [
            pl.len().alias("count"),
            # Calculate the mean of the individual ratios
            (pl.col("raw_token_count") / pl.col("num_atoms")).mean().alias("token_to_atom_ratio"),
            pl.col(pl.NUMERIC_DTYPES).mean(),
        ]

        agg_exprs.extend([
            (pl.col("structure_class").eq("Aliphatic Ring").mean() * 100).alias("pct_aliphatic_ring"),
            (pl.col("structure_class").eq("Aromatic").mean() * 100).alias("pct_aromatic"),
            (pl.col("structure_class").eq("Acyclic").mean() * 100).alias("pct_acyclic"),
        ])
    else:
        agg_exprs = [pl.len().alias("count"), pl.col(pl.NUMERIC_DTYPES).mean()]

    # Add metal percentage if the column exists in the dataset
    if "is_metal" in df.columns:
        agg_exprs.append(
            (pl.col("is_metal").cast(pl.Float64).mean() * 100).alias("pct_metal")
        )

    if "selfies" in df.columns:
        agg_exprs.append(
            pl.col("selfies")
            .drop_nulls()
            .map_elements(sf.len_selfies, return_dtype=pl.UInt32)
            .mean()
            .alias("avg_len_selfies")
        )

    # Scaffold Enrichment Metrics (Chemical Taxonomy Baseline)
    scaffold_columns = {
        "scaffold_smiles": ("unique_scaffolds", "top_scaffold", "top_scaffold_pct"),
        "generic_scaffold": (
            "unique_generic_scaffolds",
            "top_generic_scaffold",
            "top_generic_scaffold_pct",
        ),
    }
    for scaffold_col, (unique_alias, top_alias, pct_alias) in scaffold_columns.items():
        if scaffold_col in df.columns:
            # Get the most frequent scaffold in the cluster (first item handles ties)
            top_scaffold_expr = pl.col(scaffold_col).drop_nulls().mode().first()

            agg_exprs.extend([
                # 1. Total unique scaffolds in this cluster (Lower = More chemically pure)
                pl.col(scaffold_col).n_unique().alias(unique_alias),

                # 2. The dominant scaffold representation
                top_scaffold_expr.alias(top_alias),

                # 3. Purity/Enrichment: What % of the cluster matches this exact top scaffold?
                ((pl.col(scaffold_col) == top_scaffold_expr).sum() / pl.len() * 100).alias(pct_alias)
            ])

    # Materials Informatics Metrics (Crystallography & Formulas)
    for cat_col in ["crystal_system", "space_group", "anonymized_formula", "pearson_symbol"]:
        if cat_col in df.columns:
            top_expr = pl.col(cat_col).drop_nulls().mode().first()
            
            agg_exprs.extend([
                # 1. Total unique categories in this cluster
                pl.col(cat_col).n_unique().alias(f"unique_{cat_col}s"),
                
                # 2. The most dominant category
                top_expr.alias(f"top_{cat_col}"),
                
                # 3. Purity/Enrichment: What % of the cluster matches this exact top category?
                ((pl.col(cat_col) == top_expr).sum() / pl.len() * 100).alias(f"top_{cat_col}_pct")
            ])

    # Outlier Category Enrichment
    if "outlier_category" in df.columns:
        top_outlier_expr = pl.col("outlier_category").drop_nulls().mode().first()
        
        agg_exprs.extend([
            # 1. Total unique outlier categories in this cluster
            pl.col("outlier_category").n_unique().alias("unique_outlier_categories"),
            
            # 2. The dominant outlier category
            top_outlier_expr.alias("top_outlier_category"),
            
            # 3. Purity/Enrichment: What % of the cluster matches this exact outlier category?
            ((pl.col("outlier_category") == top_outlier_expr).sum() / pl.len() * 100).alias("top_outlier_category_pct")
        ])

    # Group by the cluster label and apply all aggregations
    summary = df.group_by(labels_col).agg(agg_exprs).sort(labels_col)

    with pl.Config(
        tbl_cols=-1,           # Show all columns
        tbl_rows=-1,           # Show all rows
        tbl_width_chars=1000,  # Increase total table width
        fmt_str_lengths=100,   # Allow strings to be visible
        float_precision=4      # Control decimal space
    ):
        print(summary)
        
    return summary

def benchmark_functional_groups(df, cluster_col="labels_hier", fr_prefix="fr_"):
    """
    Evaluates which functional groups dominate each cluster.
    Assumes df is a Polars or Pandas DataFrame containing fragment counts.
    """
    # Convert to pandas for easier dynamic column manipulation if using Polars
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
        
    # Isolate functional group columns
    fg_cols = [col for col in df.columns if col.startswith(fr_prefix)]
    
    results = []
    
    # Group by the cluster labels
    for cluster_id, group in df.groupby(cluster_col):
        cluster_size = len(group)
        
        # Calculate what percentage of molecules in this cluster have >0 of each FG
        fg_presence_pct = (group[fg_cols] > 0).mean() * 100
        
        # Get the top 3 most prevalent functional groups in this cluster
        top_fgs = fg_presence_pct.nlargest(3)
        
        results.append({
            "Cluster": cluster_id,
            "Size": cluster_size,
            "Top_FG_1": f"{top_fgs.index[0]} ({top_fgs.iloc[0]:.1f}%)",
            "Top_FG_2": f"{top_fgs.index[1]} ({top_fgs.iloc[1]:.1f}%)",
            "Top_FG_3": f"{top_fgs.index[2]} ({top_fgs.iloc[2]:.1f}%)"
        })
        
    return results

if __name__ == '__main__':
    from src.datasets import QM9Dataset
    dataset = QM9Dataset()
    df = dataset.load()
    frames = dataset.get_molecules()
    matrices = get_distances(frames)
    
