from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import polars as pl
import numpy as np
from typing import Sequence, Optional
from loguru import logger
import os

from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology


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

if __name__ == '__main__':
    from src.datasets import QM9Dataset
    dataset = QM9Dataset()
    df = dataset.load()
    frames = dataset.get_positions()
    matrices = get_distances(frames)
    
