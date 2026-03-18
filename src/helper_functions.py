from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import polars as pl
import numpy as np
from typing import Sequence, Optional, Iterable
from loguru import logger
import os

from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import kmedoids


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

def get_distances(frames, frames_ph=None, dataset = 'QM9', include_ph=True):
    data_dir = f'data/{dataset}'
    os.makedirs(data_dir, exist_ok=True)
    expected_n = len(frames)

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

if __name__ == '__main__':
    from src.datasets import QM9Dataset
    dataset = QM9Dataset()
    df = dataset.load()
    frames = dataset.get_positions()
    matrices = get_distances(frames)
    
