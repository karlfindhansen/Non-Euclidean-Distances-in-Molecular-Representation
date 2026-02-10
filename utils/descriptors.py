from dscribe.descriptors import SOAP, ACSF
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
import polars as pl
import numpy as np
from loguru import logger

class SOAPDescriptor:
    """
    Computes SOAP descriptors and integrates them directly into a QM9Loader.
    """

    def __init__(self, loader, r_cut=6.0, n_max=8, l_max=6, sigma=0.5):
        self.loader = loader
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        
        # QM9 elements
        self.species = ["C", "H", "O", "N", "F"]

        self.soap = SOAP(
            species=self.species,
            periodic=False,
            r_cut=self.r_cut,   
            n_max=self.n_max,    
            l_max=self.l_max,   
            sigma=self.sigma,
            average="inner"
        )

    def compute(self):
        """
        Main method: Checks the loader for data, computes SOAP, 
        and updates the loader's DataFrame in-place.
        """
        if self.loader.df.is_empty():
            logger.info("Loader is empty. triggering load_data()...")
            self.loader.load_data()

        logger.info(f"Computing SOAP (rcut={self.r_cut}, nmax={self.n_max})...")

        # Define the conversion logic using self.soap
        def _get_soap(smiles):
            if not smiles: return None
            try:
                # SMILES -> 3D Molecule
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                
                # Generate 3D coords
                params = AllChem.ETKDG()
                params.randomSeed = 42
                if AllChem.EmbedMolecule(mol, params) == -1:
                    return None
                
                # RDKit -> ASE
                atoms = Atoms(
                    symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                    positions=mol.GetConformer().GetPositions()
                )
                
                # ASE -> SOAP Vector using the stored instance
                return self.soap.create(atoms).flatten().tolist()
            except:
                return None

        # Apply to DataFrame
        self.loader.df = self.loader.df.with_columns(
            pl.col("canonical_smiles")
            .map_elements(_get_soap, return_dtype=pl.List(pl.Float64))
            .alias("soap_embedding")
        )
        
        logger.success("SOAP computed. Loader now contains 'soap_embedding' column.")


class ACSFDescriptor:
    """
    Computes Atom-Centered Symmetry Functions (ACSF) and integrates 
    them directly into a QM9Loader.
    """

    def __init__(self, loader, r_cut=6.0, species=None):
        """
        Args:
            loader: An instance of QM9Loader.
            r_cut: Cutoff radius (default: 6.0).
            species: List of atomic species (default: ["C", "H", "O", "N", "F"]).
        """
        self.loader = loader
        self.r_cut = r_cut
        self.species = species if species else ["C", "H", "O", "N", "F"]

        # Standard ACSF parameter settings for QM9-like organic molecules
        self.acsf = ACSF(
            species=self.species,
            r_cut=self.r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],  
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]],
            periodic=False,
        )

    def compute(self):
        """
        Main method: Checks the loader for data, computes ACSF, 
        averages atomic vectors to a global vector, and updates the loader.
        """
        # 1. Auto-load data if missing
        if self.loader.df.is_empty():
            logger.info("Loader is empty. Triggering load_data()...")
            self.loader.load_data()

        logger.info(f"Computing ACSF (rcut={self.r_cut})...")

        # 2. Define the conversion logic
        def _get_acsf(smiles):
            if not smiles: return None
            try:
                # SMILES -> 3D Molecule
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                
                # Generate 3D coords
                params = AllChem.ETKDG()
                params.randomSeed = 42
                if AllChem.EmbedMolecule(mol, params) == -1:
                    return None
                
                # RDKit -> ASE
                atoms = Atoms(
                    symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                    positions=mol.GetConformer().GetPositions()
                )
                
                # Calculate ACSF
                atomic_features = self.acsf.create(atoms)
                
                # Average atomic vectors to get Global Structure Vector
                global_feature = np.mean(atomic_features, axis=0)
                
                return global_feature.tolist()
            except Exception:
                return None

        # 3. Update the Loader's DataFrame directly
        self.loader.df = self.loader.df.with_columns(
            pl.col("canonical_smiles")
            .map_elements(_get_acsf, return_dtype=pl.List(pl.Float64))
            .alias("acsf_embedding")
        )
        
        logger.success("ACSF computed. Loader now contains 'acsf_embedding' column.")