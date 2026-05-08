from typing import Dict, List, Sequence, Literal, Optional, Any
import json
from joblib import Parallel, delayed
import os
import re

import numpy as np
import ot
import persim
import polars as pl

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from loguru import logger
from pymatgen.core import Element
from pyriemann.utils.distance import pairwise_distance
from ripser import ripser
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import StandardScaler

def plot_grassmann_scree(
    df: Any,
    descriptor: str = 'soap',
    k: Optional[int] = None,
    cumulative_thresholds: tuple = (0.8, 0.9, 0.95),
    show_individual: bool = False,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (9, 6),
) -> Dict[str, Any]:
    """
    Plot a scree curve for choosing the Grassmann subspace dimension (k).
    Moved outside the Grassmann distance calculation class to enforce separation of concerns.
    """
    import matplotlib.pyplot as plt

    raw_matrices = _feature_matrices_from_df(df, descriptor)
    if not raw_matrices:
        raise ValueError("No feature matrices were provided.")

    variance_ratios = []
    singular_values = []
    max_components = 0
    
    for idx, matrix in enumerate(raw_matrices):
        X = np.asarray(matrix, dtype=np.float64)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        if X.size == 0:
            continue

        _, s, _ = np.linalg.svd(X, full_matrices=False)
        variances = s ** 2
        total_variance = float(np.sum(variances))
        
        if total_variance <= 0:
            continue

        ratios = variances / total_variance
        variance_ratios.append(ratios)
        singular_values.append(s)
        max_components = max(max_components, len(ratios))

    if not variance_ratios:
        raise ValueError("Could not compute variance ratios.")

    plot_k = k or max_components
    plot_k = min(plot_k, max_components)
    
    padded_ratios = np.zeros((len(variance_ratios), plot_k), dtype=np.float64)
    for idx, ratios in enumerate(variance_ratios):
        n = min(plot_k, len(ratios))
        padded_ratios[idx, :n] = ratios[:n]

    mean_variance_ratio = np.mean(padded_ratios, axis=0)
    cumulative_variance_ratio = np.cumsum(mean_variance_ratio)
    components = np.arange(1, plot_k + 1)

    threshold_k: Dict[float, Optional[int]] = {}
    for threshold in cumulative_thresholds:
        reached = np.where(cumulative_variance_ratio >= threshold)[0]
        threshold_k[threshold] = int(reached[0] + 1) if reached.size else None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.bar(
        components,
        mean_variance_ratio,
        width=0.75,
        color="#4C78A8",
        alpha=0.85,
        label="Mean variance per vector",
    )
    ax1.set_xlabel("Number of Grassmann vectors (k)")
    ax1.set_ylabel("Mean explained variance ratio")
    
    if plot_k <= 30:
        ax1.set_xticks(components)
    else:
        tick_positions = np.unique(np.linspace(1, plot_k, num=10, dtype=int))
        ax1.set_xticks(tick_positions)
        
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    
    if show_individual:
        for ratios in variance_ratios:
            individual = np.zeros(plot_k, dtype=np.float64)
            n = min(plot_k, len(ratios))
            individual[:n] = ratios[:n]
            ax2.plot(components, np.cumsum(individual), color="#9ecae9", alpha=0.25, linewidth=1)

    ax2.plot(
        components,
        cumulative_variance_ratio,
        color="#F58518",
        marker="o",
        linewidth=2,
        label="Cumulative mean variance",
    )
    ax2.set_ylabel("Cumulative explained variance ratio")
    ax2.set_ylim(0, min(1.05, max(1.0, float(cumulative_variance_ratio[-1]) * 1.05)))

    for threshold, selected_k in threshold_k.items():
        ax2.axhline(threshold, color="#666666", linestyle="--", linewidth=1, alpha=0.45)
        if selected_k is not None:
            ax2.axvline(selected_k, color="#666666", linestyle=":", linewidth=1, alpha=0.45)
            ax2.text(
                selected_k,
                threshold,
                f" k={selected_k} ({threshold:.0%})",
                va="bottom",
                ha="left",
                fontsize=9,
                color="#444444",
            )

    ax1.set_title(title or f"Grassmann Scree Plot ({len(variance_ratios)} structures)")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Grassmann scree plot to {save_path}")
        
    if show:
        plt.show()

    return {
        "fig": fig,
        "mean_variance_ratio": mean_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio,
        "threshold_k": threshold_k,
    }

def _ensure_feature_matrix_d_by_n(
    matrix: np.ndarray,
    input_orientation: Literal["rowwise", "columnwise"] = "rowwise",
) -> np.ndarray:
    """
    Normalize an input feature matrix to shape (D, N_samples_per_structure).
    `rowwise` expects (N, D) and converts to (D, N).
    `columnwise` expects (D, N) and preserves that convention.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.size == 0:
        return arr.reshape(0, 0)

    if input_orientation == "rowwise":
        return arr.T
    if input_orientation == "columnwise":
        return arr

    raise ValueError(f"Unknown input_orientation '{input_orientation}'.")


def _ensure_feature_matrix_n_by_d(
    matrix: np.ndarray,
    input_orientation: Literal["rowwise", "columnwise"] = "rowwise",
) -> np.ndarray:
    """
    Normalize an input feature matrix to shape (N_samples_per_structure, D).
    `rowwise` expects (N, D) and preserves that convention.
    `columnwise` expects (D, N) and converts to (N, D).
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.size == 0:
        return arr.reshape(0, 0)

    if input_orientation == "rowwise":
        return arr
    if input_orientation == "columnwise":
        return arr.T

    raise ValueError(f"Unknown input_orientation '{input_orientation}'.")


def _feature_input_orientation(feature_type: str) -> Literal["rowwise", "columnwise"]:
    """
    External invariant features are historically passed as (D, N).
    All structure-level descriptor vectors such as SOAP/ACSF/MACE are treated as (N, D).
    """
    return "columnwise" if str(feature_type).lower() == "invariant" else "rowwise"


def _coerce_external_feature_collection(
    feature_matrices: Sequence[np.ndarray] | np.ndarray,
) -> List[np.ndarray]:
    """
    Accept either:
    - a sequence of per-structure feature matrices/vectors, or
    - a single 2D (N_structures, D_features) descriptor matrix.
    """
    if isinstance(feature_matrices, np.ndarray):
        arr = np.asarray(feature_matrices, dtype=np.float64)
        if arr.ndim == 0:
            return [arr.reshape(1, 1)]
        if arr.ndim == 1:
            return [arr.reshape(1, -1)]
        if arr.ndim == 2:
            return [row.reshape(1, -1) for row in arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(
            "feature_matrices must be a 1D/2D/3D array or a sequence of per-structure matrices."
        )

    matrices = list(feature_matrices)
    if not matrices:
        return []

    first = np.asarray(matrices[0])
    if first.ndim <= 1:
        return [np.asarray(matrix, dtype=np.float64).reshape(1, -1) for matrix in matrices]

    return [np.asarray(matrix, dtype=np.float64) for matrix in matrices]


def _descriptor_matrix_column(descriptor: str) -> str:
    key = str(descriptor).strip().lower()
    mapping = {
        "soap": "soap_matrix",
        "soap_matrix": "soap_matrix",
        "acsf": "acsf_matrix",
        "acsf_matrix": "acsf_matrix",
        "mace": "mace_matrix",
        "mace_matrix": "mace_matrix",
    }
    column = mapping.get(key)
    if column is None:
        raise ValueError(
            f"Unknown descriptor '{descriptor}'. Expected one of: soap, acsf, mace."
        )
    return column


def _is_dataframe_like(value: Any) -> bool:
    return isinstance(value, pl.DataFrame) or (
        value is not None
        and hasattr(value, "columns")
        and hasattr(value, "__getitem__")
    )


def _normalize_distance_matrix_inputs(
    frames: Any,
    df: Any,
    descriptor: str,
) -> tuple[Any, Any, str]:
    """
    Allows notebook-friendly calls such as:
        Grassmann.distance_matrix(df, descriptor="mace")
        Grassmann.distance_matrix(df, "mace")

    The second form arrives as `df="mace"` because `frames` is the first
    positional parameter, so normalize it before any cache or feature handling.
    """
    if not _is_dataframe_like(frames):
        return frames, df, descriptor

    if isinstance(df, str):
        descriptor = df
    elif df is not None:
        raise ValueError(
            "When passing a dataframe as the first argument, the second positional "
            "argument must be the descriptor string."
        )

    return None, frames, descriptor


def _feature_matrices_from_df(
    df: Any,
    descriptor: str,
) -> List[np.ndarray]:
    """
    Extract per-structure atom-wise descriptor matrices from a dataframe column such as
    `soap_matrix`, `acsf_matrix`, or `mace_matrix`.
    """
    if df is None:
        raise ValueError("A dataframe must be provided.")

    column_name = _descriptor_matrix_column(descriptor)
    logger.info(f"Using column: {column_name} from df")

    if isinstance(df, pl.DataFrame):
        if column_name not in df.columns:
            raise ValueError(
                f"Dataframe is missing required descriptor column '{column_name}'."
            )
        values = df[column_name].to_list()
    else:
        try:
            values = df[column_name].to_list()
        except Exception as e:
            raise ValueError(
                f"Could not extract descriptor column '{column_name}' from dataframe-like input."
            ) from e

    matrices: List[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64) if value is not None else np.empty((0, 0), dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        matrices.append(arr)
    return matrices


def _id_column_from_df(df: Any) -> str:
    if df is None:
        raise ValueError("A dataframe must be provided.")

    if isinstance(df, pl.DataFrame):
        columns = set(df.columns)
    else:
        columns = set(getattr(df, "columns", []))

    if "mol_id" in columns:
        return "mol_id"
    if "material_id" in columns:
        return "material_id"

    raise ValueError(
        "Dataframe must contain either a 'mol_id' or 'material_id' column "
        "for distance-matrix caching."
    )


def _mol_ids_from_df(df: Any) -> List[str]:
    id_column = _id_column_from_df(df)

    if isinstance(df, pl.DataFrame):
        return [str(mol_id) for mol_id in df[id_column].to_list()]

    try:
        values = df[id_column].to_list()
    except Exception as e:
        raise ValueError(f"Could not extract '{id_column}' from dataframe-like input.") from e
    return [str(mol_id) for mol_id in values]


def _mol_ids_from_frames(frames: Sequence[Atoms]) -> List[str]:
    mol_ids: List[str] = []
    for idx, frame in enumerate(frames):
        mol_id = frame.info.get("mol_id", frame.info.get("material_id"))
        if mol_id is None:
            raise ValueError(
                f"Frame at index {idx} is missing info['mol_id'] or info['material_id']; "
                "this is required for persistent-homology caching."
            )
        mol_ids.append(str(mol_id))
    return mol_ids


def _maybe_mol_ids_from_frames(frames: Sequence[Atoms] | None) -> Optional[List[str]]:
    if frames is None:
        return None

    mol_ids: List[str] = []
    for frame in frames:
        mol_id = frame.info.get("mol_id", frame.info.get("material_id"))
        if mol_id is None:
            return None
        mol_ids.append(str(mol_id))
    return mol_ids


def _default_non_euclidean_cache_dir(dataset: str = "QM9") -> str:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, "data", dataset, "non_euclidean_cache")


def _non_euclidean_cache_dir_for_df(df: Any) -> str:
    id_column = _id_column_from_df(df)
    dataset = "Materials Project" if id_column == "material_id" else "QM9"
    return _default_non_euclidean_cache_dir(dataset)


def _non_euclidean_cache_dir_for_frames(frames: Sequence[Atoms]) -> str:
    has_material_ids = any(frame.info.get("material_id") is not None for frame in frames)
    dataset = "Materials Project" if has_material_ids else "QM9"
    return _default_non_euclidean_cache_dir(dataset)


def _cache_key_payload(
    method_name: str,
    mol_ids: Sequence[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": method_name,
        "mol_ids": list(mol_ids),
        "params": params,
    }


def _sanitize_cache_token(value: Any) -> str:
    token = str(value).strip().lower()
    token = token.replace(" ", "-").replace("_", "-")
    token = re.sub(r"[^a-z0-9.-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or "value"


def _dataset_label_from_ids(mol_ids: Sequence[str]) -> str:
    ids = [str(mol_id).strip().lower() for mol_id in mol_ids]
    if any(mol_id.startswith("qm9_") for mol_id in ids):
        return "QM9"
    if any(
        mol_id.startswith(("mp-", "mvc-", "material", "synthetic-"))
        for mol_id in ids
    ):
        return "Materials Project"
    return "Materials Project"


def _log_distance_dataset_from_ids(method_name: str, mol_ids: Sequence[str] | None) -> None:
    if mol_ids is None:
        return
    logger.info(
        f"Using {_dataset_label_from_ids(mol_ids)} ids for {method_name} "
        f"distance matrix (n={len(mol_ids)})."
    )


def _cache_file_stem(
    method_name: str,
    mol_ids: Sequence[str],
    params: Dict[str, Any],
) -> str:
    descriptor = _sanitize_cache_token(params.get("descriptor", "unknown"))
    dataset = _sanitize_cache_token(_dataset_label_from_ids(mol_ids))
    stem_parts = [method_name, dataset, f"n{len(mol_ids)}", descriptor]

    if method_name == "wasserstein":
        stem_parts.append(_sanitize_cache_token(params.get("metric", "sqeuclidean")))
    elif method_name == "grassmann":
        stem_parts.append(f"k{int(params.get('k', 0))}")
        stem_parts.append(_sanitize_cache_token(params.get("method", "svd")))
        stem_parts.append(_sanitize_cache_token(params.get("vector_side", "left")))
        stem_parts.append("norm" if bool(params.get("normalized", True)) else "raw")
    elif method_name == "riemann":
        stem_parts.append(_sanitize_cache_token(params.get("metric", "affine-invariant")))
        n_pca = params.get("n_pca", None)
        stem_parts.append("nopca" if n_pca is None else f"pca{int(n_pca)}")
    elif method_name == "persistent_homology":
        stem_parts.append(_sanitize_cache_token(params.get("metric", "bottleneck")))
        stem_parts.append(f"maxdim{int(params.get('max_homology_dim', 2))}")
        dims = params.get("homology_dims", ())
        dims_token = "-".join(str(int(d)) for d in dims) if dims else "none"
        stem_parts.append(f"dims{dims_token}")

    return "_".join(stem_parts)


def _cache_paths(cache_dir: str, stem: str, index: Optional[int] = None) -> tuple[str, str]:
    suffix = "" if index is None else f"_{index}"
    return (
        os.path.join(cache_dir, f"{stem}{suffix}.npy"),
        os.path.join(cache_dir, f"{stem}{suffix}.json"),
    )


def _load_cached_distance_matrix(
    method_name: str,
    mol_ids: Sequence[str],
    params: Dict[str, Any],
    cache_dir: str,
    force_recalculate: bool = False,
) -> Optional[np.ndarray]:
    stem = _cache_file_stem(method_name, mol_ids, params)

    if force_recalculate or not os.path.isdir(cache_dir):
        return None

    for filename in sorted(os.listdir(cache_dir)):
        if not filename.startswith(stem) or not filename.endswith(".json"):
            continue

        meta_path = os.path.join(cache_dir, filename)
        matrix_path = meta_path[:-5] + ".npy"
        if not os.path.exists(matrix_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as e:
            logger.warning(f"Failed to read non-Euclidean cache metadata from {meta_path}: {e}")
            continue

        if metadata.get("mol_ids") != list(mol_ids):
            continue
        if metadata.get("params") != params:
            continue

        logger.info(f"Loading cached {method_name} distance matrix from {matrix_path}")
        return np.load(matrix_path)

    return None


def _save_cached_distance_matrix(
    matrix: np.ndarray,
    method_name: str,
    mol_ids: Sequence[str],
    params: Dict[str, Any],
    cache_dir: str,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    payload = _cache_key_payload(method_name, mol_ids, params)
    stem = _cache_file_stem(method_name, mol_ids, params)

    chosen_index: Optional[int] = None
    for candidate in [None] + list(range(1, 10_000)):
        matrix_path, meta_path = _cache_paths(cache_dir, stem, candidate)
        if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
            chosen_index = candidate
            break

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            chosen_index = candidate
            break

        if metadata.get("mol_ids") == list(mol_ids) and metadata.get("params") == params:
            chosen_index = candidate
            break

    matrix_path, meta_path = _cache_paths(cache_dir, stem, chosen_index)

    np.save(matrix_path, matrix)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.success(f"Saved cached {method_name} distance matrix to {matrix_path}")


def _normalize_external_feature_matrices_d_by_n(
    feature_matrices: Sequence[np.ndarray] | np.ndarray,
    feature_type: str,
) -> List[np.ndarray]:
    input_orientation = _feature_input_orientation(feature_type)
    return [
        _ensure_feature_matrix_d_by_n(matrix, input_orientation=input_orientation)
        for matrix in _coerce_external_feature_collection(feature_matrices)
    ]


def _normalize_external_feature_matrices_n_by_d(
    feature_matrices: Sequence[np.ndarray] | np.ndarray,
    feature_type: str,
) -> List[np.ndarray]:
    input_orientation = _feature_input_orientation(feature_type)
    return [
        _ensure_feature_matrix_n_by_d(matrix, input_orientation=input_orientation)
        for matrix in _coerce_external_feature_collection(feature_matrices)
    ]


def _compute_invariant_feature_matrix(frame: Atoms, cutoff: float = 1.8) -> np.ndarray:
    """
    Maps a molecule to a D x N matrix of invariant physical features.
    D is the fixed ambient dimension. N is the number of atoms.
    """
    features = []
    # Center of mass acts as an invariant spatial anchor
    com = frame.get_center_of_mass()

    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)

    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    for i, atom in enumerate(frame):
        z = atom.number
        rad = covalent_radii[z]
        el = Element.from_Z(z)
        en = el.X if el.X else 0.0
        mendeleev = el.mendeleev_no if el.mendeleev_no else 0
        ion_en = el.ionization_energy if el.ionization_energy else 0.0

        mass = atom.mass

        # Geometric invariance: distance to center of mass
        dist_to_com = np.linalg.norm(atom.position - com)

        coord = len(set(neighbors[i]))

        if coord > 0:
            avg_neighbor_z = np.mean(neighbors[i])
            avg_neighbor_dist = np.mean(distances[i])
        else:
            avg_neighbor_z = 0
            avg_neighbor_dist = 0

        feat_vector = [
            coord,
            mendeleev,
            ion_en,
        ]

        features.append(feat_vector)
    #logger.info(f"Computed invariant feature matrix consisting of coordination, mendeleev, and ionization energy for frame with {len(frame)} atoms.")
    return np.array(features).T


def _compute_feature_matrices(
    frames: Sequence[Atoms],
    normalized: bool = True,
) -> List[np.ndarray]:
    """
    Builds invariant feature matrices for all frames.

    If normalized=True, applies a global StandardScaler across all atoms
    in the dataset (fit on the stacked per-atom features).
    """
    raw_matrices = [_compute_invariant_feature_matrix(f) for f in frames]

    if not normalized:
        return raw_matrices

    if not raw_matrices:
        return raw_matrices

    stacked = np.vstack([m.T for m in raw_matrices if m.size > 0]) if raw_matrices else np.empty((0, 0))

    if stacked.size == 0:
        return raw_matrices

    scaler = StandardScaler().fit(stacked)

    scaled_matrices = []
    for raw in raw_matrices:
        if raw.size == 0:
            scaled_matrices.append(raw)
            continue
        scaled_matrices.append(scaler.transform(raw.T).T)

    return scaled_matrices

def _compute_soap_feature_matrices(frames: Sequence[Atoms]) -> List[np.ndarray]:
    """
    Builds invariant soap feature matrices for all frames.
    """
    return [
        _ensure_feature_matrix_d_by_n(frame.soap, input_orientation="rowwise")
        for frame in frames
    ]


def _pairwise_distance_matrix(
    n: int,
    pair_fn,
    desc: str
) -> np.ndarray:
    dist_matrix = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2

    with tqdm(total=total_pairs, desc=desc, unit="pair") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                d = pair_fn(i, j)
                dist_matrix[i, j] = dist_matrix[j, i] = d
                pbar.update(1)

    return dist_matrix

class Wasserstein:
    """
    Computes the Earth Mover's Distance (Wasserstein-1) between molecules.
    Treats each molecule as a uniform distribution of atomic feature vectors.
    """

    @classmethod
    def distance_matrix(
        cls, 
        df: Any,
        descriptor: str = 'soap',
        metric: str = 'sqeuclidean',
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise Wasserstein distance matrix sequentially.
        Optimized to minimize inner-loop Python overhead.
        """
        # 1. Extract raw feature matrices (N_atoms, D_features)
        raw_matrices = _feature_matrices_from_df(df, descriptor)
        
        num_items = len(raw_matrices)
        if num_items == 0:
            return np.array([])
            
        dist_matrix = np.zeros((num_items, num_items))
        logger.info(f"Computing Wasserstein distance matrix sequentially | Features: {descriptor} | Metric: {metric}")

        # 2. PRE-OPTIMIZATION: Pre-compute the uniform mass weights for all molecules
        # Weights: Each atom is 1/N of the molecule's total "mass"
        # Pre-calculating this saves N^2 array allocations inside the loop.
        weights = [np.ones(X.shape[0]) / X.shape[0] for X in raw_matrices]
        
        # 3. Sequential nested loop calculation (Upper triangle only, mirrored to lower)
        for i in tqdm(range(num_items), desc="Wasserstein distances", unit="row"):
            X_i = np.asarray(raw_matrices[i])
            w_i = weights[i]
            
            for j in range(i + 1, num_items):
                X_j = np.asarray(raw_matrices[j])
                w_j = weights[j]

                try:
                    # Compute the Cost Matrix (Distances between all atoms in A and B)
                    # M[a, b] is the cost to move atom 'a' to 'b' in feature space
                    M = ot.dist(X_i, X_j, metric=metric)

                    # Solve the Optimal Transport problem directly using POT's C-backend
                    d = float(ot.emd2(w_i, w_j, M))
                    
                    dist_matrix[i, j] = dist_matrix[j, i] = d
                    
                except Exception as e:
                    logger.warning(f"Distance calculation failed for pair ({i}, {j}): {e}")
                    dist_matrix[i, j] = dist_matrix[j, i] = np.nan

        # 4. Fill failed calculations if any numerical instabilities occurred
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")
            dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix

class PersistentHomology:
    """
    Computes topological features (persistence diagrams) from 3D atomic point clouds 
    and evaluates the structural similarities between different frames using 
    Bottleneck or Sliced Wasserstein distances.
    """

    @staticmethod
    def _compute_ripser(distance_matrix: np.ndarray, max_dim: int) -> Dict[int, np.ndarray]:
        """Calculates persistence diagrams up to max_dim from a precomputed distance matrix."""
        # distance_matrix=True is strictly required so ripser doesn't treat the input as a point cloud
        dgms = ripser(distance_matrix, maxdim=max_dim, distance_matrix=True)["dgms"]
        
        # Format output into a dictionary mapping homology dimension to its (birth, death) array
        return {d: np.asarray(dgms[d]) for d in range(max_dim + 1)}

    @classmethod
    def compute_persistence_diagrams(
        cls, frames: Sequence[Atoms], max_homology_dim: int = 2
    ) -> List[Dict[int, np.ndarray]]:
        """Generates persistence diagrams for a sequence of molecular frames."""
        
        logger.info(
            f"Computing persistence diagrams for {len(frames)} frames "
            f"(max_homology_dim={max_homology_dim})."
        )
        
        diagrams = []

        for frame in tqdm(frames, desc="Persistence diagrams", unit="frame"):
            # Handle edge case of an empty simulation frame to prevent Ripser crashes
            if len(frame) == 0:
                diagrams.append({d: np.empty((0, 2)) for d in range(max_homology_dim + 1)})
                continue
            
            # Use Minimum Image Convention (MIC) to ensure bonds across periodic 
            # cell boundaries are calculated at their true shortest distance
            dist_mat = frame.get_all_distances(mic=True)
            diagrams.append(cls._compute_ripser(dist_mat, max_homology_dim))
 
        logger.success("Finished persistence diagram computation.")
        return diagrams
    
    @staticmethod
    def distance(
        dgm1: Dict[int, np.ndarray],
        dgm2: Dict[int, np.ndarray],
        metric: str = "bottleneck",
        dims: Sequence[int] = (0, 1, 2),
        sw_projections: int = 50
    ) -> float:
        """
        Computes the total topological distance between two diagrams across 
        specified homology dimensions (e.g., 0=components, 1=loops, 2=voids).
        """
        metric_key = metric.lower()
        if metric_key not in {"bottleneck", "b", "sliced-wasserstein", "sliced_wasserstein", "sw"}:
            logger.error(f"Unknown persistence metric '{metric}'.")
            raise ValueError(
                "metric must be one of: ['bottleneck', 'b', 'sliced-wasserstein', 'sliced_wasserstein', 'sw']"
            )

        total_dist = 0.0
        
        for d in dims:
            # Safely fetch the diagrams for dimension `d`, defaulting to empty if missing
            p1, p2 = dgm1.get(d, np.empty((0, 2))), dgm2.get(d, np.empty((0, 2)))

            if len(p1) == 0 and len(p2) == 0:
                continue
            
            # Filter out features with infinite death times (essential classes) 
            # since distance metrics require finite bounds to compute properly
            if len(p1) > 0:
                p1 = p1[np.isfinite(p1[:, 1])]
            if len(p2) > 0:
                p2 = p2[np.isfinite(p2[:, 1])]
            
            # Accumulate the calculated distance for this dimension
            if metric_key in {"bottleneck", "b"}:
                total_dist += persim.bottleneck(p1, p2)
            else:
                total_dist += persim.sliced_wasserstein(p1, p2, M=sw_projections)
                
        return float(total_dist)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        metric: str = "bottleneck",
        max_homology_dim: int = 2,
        homology_dims: Sequence[int] = (0, 1, 2),
        cache_dir: Optional[str] = None,
        force_recalculate: bool = False,
    ) -> np.ndarray:
        """
        Generates a symmetric pairwise distance matrix comparing the topological 
        features of all molecular frames in the sequence.
        """
        mol_ids = _mol_ids_from_frames(frames)
        resolved_cache_dir = cache_dir or _non_euclidean_cache_dir_for_frames(frames)
        cache_params = {
            "metric": metric,
            "max_homology_dim": int(max_homology_dim),
            "homology_dims": [int(d) for d in homology_dims],
        }
        cached = _load_cached_distance_matrix(
            method_name="persistent_homology",
            mol_ids=mol_ids,
            params=cache_params,
            cache_dir=resolved_cache_dir,
            force_recalculate=force_recalculate,
        )
        if cached is not None:
            return cached

        _log_distance_dataset_from_ids("persistent homology", mol_ids)
        logger.info(
            f"Computing persistent homology distance matrix for {len(frames)} frames "
            f"(metric='{metric}', max_homology_dim={max_homology_dim}, "
            f"dims={tuple(homology_dims)})."
        )
        
        # Precompute all diagrams
        dgms = cls.compute_persistence_diagrams(frames, max_homology_dim)
        n = len(dgms)
        dist_mat = _pairwise_distance_matrix(
            n=n,
            pair_fn=lambda i, j: cls.distance(dgms[i], dgms[j], metric=metric, dims=homology_dims),
            desc="Persistence distances",
        )

        _save_cached_distance_matrix(
            dist_mat,
            method_name="persistent_homology",
            mol_ids=mol_ids,
            params=cache_params,
            cache_dir=resolved_cache_dir,
        )

        logger.success("Finished persistent homology distance matrix computation.")
        return dist_mat

class Grassmann:
    """
    Handles molecular representation on Grassmann Manifolds G(k, D).
    Represents each molecule as a k-dimensional subspace in R^D (feature space).
    """

    @classmethod
    def _get_uk_bases(
        cls,
        df: Any,
        descriptor: str = 'soap',
        k: int = 3, 
        vector_side: str = "right"
    ) -> List[np.ndarray]:
        """
        Maps 3D atomic coordinates to an orthonormal basis in R^D (feature space) via SVD.
        """
        vector_side = vector_side.strip().lower()
        if vector_side not in {"left", "right"}:
            raise ValueError("vector_side must be either 'left' or 'right'.")

        bases = []
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        for X in raw_matrices:
            X = np.asarray(X, dtype=np.float64)
            
            # SVD on X (N x D): U is (N, M), S is (M,), Vh is (M, D)
            u, _, vh = np.linalg.svd(X, full_matrices=False)
            basis = u[:, :k] if vector_side == "left" else vh.T[:, :k]
            bases.append(basis)
        
        return bases

    @classmethod
    def distance_matrix(
        cls, 
        df: Any,
        descriptor: str = 'soap',
        distance_type: str = "geodesic",
        k: int = 3, 
        vector_side: str = "right",
    ) -> np.ndarray:
        """
        Computes a symmetric pairwise distance matrix on the Grassmann Manifold.
        """
        valid_distances = {"geodesic", "chordal", "projection"}
        distance_type = distance_type.lower()
        if distance_type not in valid_distances:
            raise ValueError(f"Unknown distance_type: '{distance_type}'. Must be one of {valid_distances}.")

        # Generate Subspaces
        bases = cls._get_uk_bases(
            df=df,
            descriptor=descriptor,
            k=k, 
            vector_side=vector_side
        )
        
        num_items = len(bases)
        dist_matrix = np.zeros((num_items, num_items))
        
        logger.info(f"Computing Grassmann distance matrix | Features: {descriptor} | Distance: {distance_type} | k: {k}")
        
        # Pre-calculate transposes to save N^2 transpose operations
        bases_T = [U.T for U in bases]
        
        # Sequential nested loop calculation
        for i in tqdm(range(num_items), desc="Grassmann distances", unit="row"):
            U1_T = bases_T[i]
            for j in range(i + 1, num_items):
                
                # 1. Inner Product: Yields a tiny (k x k) matrix
                core_matrix = U1_T @ bases[j]
                
                # 2. Pure NumPy SVD: compute_uv=False skips calculating the vectors, only gets singular values
                s = np.linalg.svd(core_matrix, compute_uv=False)
                
                # 3. Math: Singular values of U1^T U2 represent cos(theta).
                # Clip to [0.0, 1.0] to prevent floating point errors (like 1.0000001) from crashing arccos
                angles = np.arccos(np.clip(s, 0.0, 1.0))
                
                # 4. Inline Distance Calculation
                if distance_type == "geodesic":
                    d = float(np.linalg.norm(angles))
                elif distance_type == "chordal":
                    d = float(np.linalg.norm(np.sin(angles)))
                else: # projection
                    d = float(np.max(np.sin(angles)))
                    
                dist_matrix[i, j] = dist_matrix[j, i] = d

        # Fill failed calculations
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")
            dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix))

        return dist_matrix

class Riemann:
    """
    Handles molecular representation on the Riemannian Manifold.
    Supports global descriptors by converting them into SPD covariance matrices.
    """

    @classmethod
    def _get_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        regularization: float = 1e-3,
        n_pca: Optional[int] = 30
    ) -> np.ndarray:
        
        # 1. Obtain raw feature matrices directly from df
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        # 2. PCA Reduction
        n_pca = df['num_atoms'].min() - 2
        raw_matrices = cls.matrix_pca(n_pca, raw_matrices)

        # 3. Build SPD Matrices (Empirical Covariance)
        spd_matrices = []
        for X in raw_matrices:
            X = np.asarray(X)
            # Compute covariance
            C = (X.T @ X) / X.shape[0]
            # Regularization to ensure strict positive-definiteness
            C += np.eye(C.shape[0]) * regularization
            spd_matrices.append(C)

        # Pyriemann expects a 3D array of shape (N_matrices, n_channels, n_channels)
        return np.array(spd_matrices)

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
    def distance_matrix(
        cls,
        df: Any,
        descriptor: str = 'soap',
        distance_type: str = "affine-invariant",
        regularization: float = 1e-3,
    ) -> np.ndarray:
        
        metric_map = {
            "affine-invariant": "riemann",
            "log-euclidean": "logeuclid",
            "euclidean": "euclid"
        }
        pyriemann_metric = metric_map.get(distance_type.lower())
        if not pyriemann_metric:
            raise ValueError(f"Unknown distance_type: '{distance_type}'. Must be one of {list(metric_map.keys())}.")

        logger.info(f"Computing Riemann distance matrix | Features: {descriptor} | Distance: {distance_type}")

        # 1. Build SPD matrices
        spd_matrices = cls._get_spd_matrices(
            df=df,
            descriptor=descriptor,
            regularization=regularization,
        )
        
        # 2. Compute Distances
        # This replaces the manual loops, generalized eigenvalue calculations, and joblib parallelization.
        logger.info(f"Computing {distance_type} distances...")
        dist_matrix = pairwise_distance(spd_matrices, metric=pyriemann_metric)

        # Fallback for severe numerical instability
        if np.isnan(dist_matrix).any():
            logger.warning("NaNs detected in distance matrix. Filling with maximum matrix distance.")

        return dist_matrix