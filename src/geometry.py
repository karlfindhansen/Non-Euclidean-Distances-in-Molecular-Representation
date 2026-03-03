import numpy as np
import os
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Sequence
from loguru import logger
import polars as pl

class GeometryPerturber:
    """
    Handles 3D geometry generation and perturbation (Stress Test).
    """
    def __init__(self, save_path: str):
        self.save_path = save_path

    def load_stress_test(
        self,
        save_path: str | None = None,
        mol_ids: Sequence[str] | None = None,
    ) -> List[Atoms]:
        """Loads existing stress test data."""
        target_path = save_path or self.save_path
        if not os.path.exists(target_path):
            logger.warning("No stress test file found at path.")
            return []
        try:
            frames = read(target_path, index=":")
            if mol_ids is None:
                return frames

            wanted_ids = set(mol_ids)
            filtered_frames = [frame for frame in frames if frame.info.get("mol_id") in wanted_ids]
            logger.info(
                f"Loaded {len(filtered_frames)} filtered frames "
                f"(requested mol_ids={len(wanted_ids)})."
            )
            return filtered_frames
        except Exception as e:
            logger.error(f"Failed to read stress test file: {e}")
            return []

    def generate_stress_test(
        self, 
        dataframe: pl.DataFrame, 
        num_molecules: int = 10, 
        mol_ids: Sequence[str] | None = None,
        perturbations: int = 20, 
        include_base: bool = True,
        max_rattle: float = 0.5, 
        seed: int = 40,
        rotated: bool = False,
        save_path: str | None = None,
    ) -> List[Atoms]:
        """Generates perturbed geometries and saves them."""
        logger.info(f"Generating Grassmann Stress Test (Seed={seed}, Rotated={rotated}, Max Rattle={max_rattle})...")
        
        if dataframe.is_empty():
            raise ValueError("DataFrame provided for geometry generation is empty.")

        target_path = save_path or self.save_path
        if mol_ids is not None:
            requested_ids = list(dict.fromkeys(mol_ids))
            sample_df = dataframe.filter(pl.col("mol_id").is_in(requested_ids))
            if sample_df.is_empty():
                raise ValueError("None of the provided mol_ids were found in the dataframe.")
            missing = sorted(set(requested_ids) - set(sample_df["mol_id"].to_list()))
            if missing:
                logger.warning(f"Ignoring {len(missing)} missing mol_ids: {missing}")
        else:
            available = len(dataframe)
            n_sample = min(available, num_molecules)
            sample_df = dataframe.sample(n=n_sample, seed=seed)

        rng = np.random.default_rng(seed)
        
        all_frames = []
        failed_count = 0

        for row in sample_df.iter_rows(named=True):
            mol_id = row['mol_id']
            smiles = row['canonical_smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol: raise ValueError("Invalid SMILES")
                
                mol = Chem.AddHs(mol)
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
                params = AllChem.ETKDG()
                params.randomSeed = seed
                
                if AllChem.EmbedMolecule(mol, params) == -1:
                    raise ValueError("Embedding failed")
                
                base_pos = mol.GetConformer().GetPositions()
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

                if include_base:
                    base_atoms = Atoms(symbols=symbols, positions=base_pos.copy(), charges=charges)
                    base_atoms.info.update({
                        'mol_id': mol_id,
                        'perturbation_idx': -1,
                        'frame_type': 'base',
                        'smiles': smiles,
                        'rotated': False
                    })
                    all_frames.append(base_atoms)
                
                for i in range(perturbations):
                    # Create new Atom object for this perturbation
                    pert_atoms = Atoms(symbols=symbols, positions=base_pos.copy(), charges=charges)
                    noise = rng.uniform(low=-max_rattle, high=max_rattle, size=base_pos.shape)
                    pert_atoms.positions += noise
                    
                    if rotated:
                        angle = rng.uniform(0.0, 360.0)
                        axis = rng.normal(size=3)
                        axis /= np.linalg.norm(axis)
                        pert_atoms.rotate(angle, axis)
                    
                    pert_atoms.info.update({
                        'mol_id': mol_id, 
                        'perturbation_idx': i, 
                        'frame_type': 'perturbed',
                        'smiles': smiles,
                        'rotated': rotated
                    })
                    all_frames.append(pert_atoms)

            except Exception as e:
                logger.debug(f"Skipping {mol_id}: {e}")
                failed_count += 1
                continue

        logger.info(f"Generated {len(all_frames)} frames. Failed molecules: {failed_count}")
        
        try:
            write(target_path, all_frames)
            logger.success(f"Saved stress test to {target_path}")
        except Exception as e:
            logger.error(f"Failed to save .xyz file: {e}")
            
        return all_frames

    def get_grassmann_distance_matrix(self, frames: Sequence[Atoms]) -> np.ndarray:
        """
        Computes a Grassmann distance matrix from ASE frames.
        """
        from src.features import Grassmann
        return Grassmann.distance_matrix(frames)
