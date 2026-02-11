import numpy as np
import os
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List
from loguru import logger
import polars as pl

class GeometryPerturber:
    """
    Handles 3D geometry generation and perturbation (Stress Test).
    """
    def __init__(self, save_path: str):
        self.save_path = save_path

    def load_stress_test(self) -> List[Atoms]:
        """Loads existing stress test data."""
        if not os.path.exists(self.save_path):
            logger.warning("No stress test file found at path.")
            return []
        try:
            return read(self.save_path, index=":")
        except Exception as e:
            logger.error(f"Failed to read stress test file: {e}")
            return []

    def generate_stress_test(
        self, 
        dataframe: pl.DataFrame, 
        num_molecules: int = 10, 
        perturbations: int = 20, 
        stdev: float = 0.1, 
        seed: int = 40
    ) -> List[Atoms]:
        """Generates perturbed geometries and saves them."""
        logger.info(f"Generating Grassmann Stress Test (Seed={seed})...")
        
        if dataframe.is_empty():
            raise ValueError("DataFrame provided for geometry generation is empty.")

        available = len(dataframe)
        n_sample = min(available, num_molecules)
        
        sample_df = dataframe.sample(n=n_sample, seed=seed)
        np.random.seed(seed)
        
        all_frames = []
        failed_count = 0

        for row in sample_df.iter_rows(named=True):
            mol_id = row['mol_id']
            smiles = row['canonical_smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol: raise ValueError("Invalid SMILES")
                
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDG()
                params.randomSeed = seed
                
                if AllChem.EmbedMolecule(mol, params) == -1:
                    raise ValueError("Embedding failed")
                
                base_pos = mol.GetConformer().GetPositions()
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

                for i in range(perturbations):
                    # Create new Atom object for this perturbation
                    pert_atoms = Atoms(symbols=symbols, positions=base_pos.copy())
                    noise = np.random.normal(0.0, stdev, base_pos.shape)
                    pert_atoms.positions += noise
                    
                    pert_atoms.info.update({
                        'mol_id': mol_id, 
                        'perturbation_idx': i, 
                        'smiles': smiles
                    })
                    all_frames.append(pert_atoms)

            except Exception as e:
                logger.debug(f"Skipping {mol_id}: {e}")
                failed_count += 1
                continue

        logger.info(f"Generated {len(all_frames)} frames. Failed molecules: {failed_count}")
        
        try:
            write(self.save_path, all_frames)
            logger.success(f"Saved stress test to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save .xyz file: {e}")
            
        return all_frames