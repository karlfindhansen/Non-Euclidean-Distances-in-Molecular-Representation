import numpy as np
import os
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Sequence
from loguru import logger
import polars as pl

def get_downstream_atoms(mol: Chem.Mol, fixed_idx: int, moving_idx: int) -> List[int]:
    """
    Finds all atoms attached to moving_idx, ensuring we don't traverse back 
    through fixed_idx. This defines the 'branch' of the molecule that must move.
    """
    visited = {fixed_idx}
    queue = [moving_idx]
    moving_group = []
    
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            moving_group.append(curr)
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    queue.append(nbr_idx)
    return moving_group

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
        max_bond_rattle: float = 0.05,  # Changed from max_rattle (in Angstroms)
        max_angle_rattle: float = 5.0,  # New parameter (in Degrees)
        seed: int = 40,
        rotated: bool = False,
        save_path: str | None = None,
    ) -> List[Atoms]:
        """Generates perturbed geometries using internal coordinates and saves them."""
        logger.info(f"Generating Stress Test (Seed={seed}, Max Bond Rattle={max_bond_rattle}Å, Max Angle Rattle={max_angle_rattle}°)...")
        
        if dataframe.is_empty():
            raise ValueError("DataFrame provided for geometry generation is empty.")

        target_path = save_path or getattr(self, 'save_path', 'stress_test.xyz')
        
        if mol_ids is not None:
            requested_ids = list(dict.fromkeys(mol_ids))
            sample_df = dataframe.filter(pl.col("mol_id").is_in(requested_ids))
            if sample_df.is_empty():
                raise ValueError("None of the provided mol_ids were found in the dataframe.")
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
                charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms() if atom.HasProp("_GasteigerCharge")]

                if len(charges) != mol.GetNumAtoms():
                    charges = [0.0] * mol.GetNumAtoms()

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
                    pert_atoms = Atoms(symbols=symbols, positions=base_pos.copy(), charges=charges)
                    
                    # 1. PERTURB BOND LENGTHS
                    for bond in mol.GetBonds():
                        # Skipping rings: expanding one bond in a ring forces another to break or distort wildly
                        if bond.IsInRing():
                            continue 
                            
                        a1 = bond.GetBeginAtomIdx()
                        a2 = bond.GetEndAtomIdx()
                        
                        # Get atoms that should move when we stretch the a1-a2 bond
                        moving_indices = get_downstream_atoms(mol, fixed_idx=a1, moving_idx=a2)
                        
                        # Create boolean mask for ASE
                        mask = [idx in moving_indices for idx in range(len(pert_atoms))]
                        
                        current_dist = pert_atoms.get_distance(a1, a2)
                        noise = rng.uniform(-max_bond_rattle, max_bond_rattle)
                        new_dist = max(0.5, current_dist + noise) # Prevent atoms from overlapping
                        
                        try:
                            pert_atoms.set_distance(a1, a2, new_dist, mask=mask)
                        except Exception:
                            pass # Safely ignore if geometry makes this impossible

                    # 2. PERTURB ANGLES
                    for atom in mol.GetAtoms():
                        if atom.IsInRing(): 
                            continue
                            
                        neighbors = atom.GetNeighbors()
                        if len(neighbors) >= 2:
                            # Just perturb the angle between the first two neighbors
                            a1 = neighbors[0].GetIdx()
                            center = atom.GetIdx()
                            a2 = neighbors[1].GetIdx()
                            
                            moving_indices = get_downstream_atoms(mol, fixed_idx=center, moving_idx=a2)
                            mask = [idx in moving_indices for idx in range(len(pert_atoms))]
                            
                            current_angle = pert_atoms.get_angle(a1, center, a2)
                            angle_noise = rng.uniform(-max_angle_rattle, max_angle_rattle)
                            # Clip to avoid flipping the molecule inside out (180 or 0 degrees)
                            new_angle = np.clip(current_angle + angle_noise, 15.0, 165.0) 
                            
                            try:
                                pert_atoms.set_angle(a1, center, a2, new_angle, mask=mask)
                            except Exception:
                                pass

                    # 3. ROTATE ENTIRE MOLECULE (If requested)
                    if rotated:
                        angle = rng.uniform(0.0, 360.0)
                        axis = rng.normal(size=3)
                        axis /= np.linalg.norm(axis)
                        pert_atoms.rotate(angle, axis, center='COM') # Better to rotate around Center of Mass
                    
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
