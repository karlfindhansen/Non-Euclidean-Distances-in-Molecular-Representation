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

    FIXES:
    - Uses per-atom SOAP (average=None)
    - Explicit mean pooling across atoms (research-standard)
    - Removes flatten()
    - Adds geometry optimization for better physics
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
            sigma=self.sigma
        )

    def compute(self):
        """
        Computes SOAP features and updates loader DataFrame with a global vector
        per structure (mean of atomic SOAP vectors).
        """
        if self.loader.df.is_empty():
            logger.info("Loader is empty. triggering load_data()...")
            self.loader.load_data()

        logger.info(
            f"Computing SOAP (rcut={self.r_cut}, nmax={self.n_max}, lmax={self.l_max})..."
        )

        def _get_soap(smiles):
            if not smiles:
                return None

            try:
                # SMILES -> RDKit molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None

                mol = Chem.AddHs(mol)

                # Generate 3D geometry
                params = AllChem.ETKDG()
                params.randomSeed = 42

                if AllChem.EmbedMolecule(mol, params) == -1:
                    return None

                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass

                # RDKit -> ASE
                atoms = Atoms(
                    symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                    positions=mol.GetConformer().GetPositions(),
                )

                # Compute per-atom SOAP: shape (n_atoms, D)
                atomic_soap = self.soap.create(atoms)

                # Mean pooling → global structure vector
                global_soap = np.mean(atomic_soap, axis=0)

                return global_soap.tolist()

            except Exception as e:
                logger.warning(f"SOAP failed for {smiles}: {e}")
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

    - Computes per-atom ACSF
    - Mean pooling → global vector
    - Adds geometry optimization for consistency with SOAP
    """

    def __init__(self, loader, r_cut=6.0, species=None):
        self.loader = loader
        self.r_cut = r_cut
        self.species = species if species else ["C", "H", "O", "N", "F"]

        self.acsf = ACSF(
            species=self.species,
            r_cut=self.r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]],
            periodic=False,
        )

    def compute(self):
        """
        Computes ACSF features, averages atomic vectors,
        and updates loader DataFrame.
        """
        if self.loader.df.is_empty():
            logger.info("Loader is empty. Triggering load_data()...")
            self.loader.load_data()

        logger.info(f"Computing ACSF (rcut={self.r_cut})...")

        def _get_acsf(smiles):
            if not smiles:
                return None

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None

                mol = Chem.AddHs(mol)

                params = AllChem.ETKDG()
                params.randomSeed = 42

                if AllChem.EmbedMolecule(mol, params) == -1:
                    return None

                # Match SOAP pipeline → optimize geometry
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass

                atoms = Atoms(
                    symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                    positions=mol.GetConformer().GetPositions(),
                )

                # (n_atoms, D)
                atomic_features = self.acsf.create(atoms)

                # Mean pooling
                global_feature = np.mean(atomic_features, axis=0)

                return global_feature.tolist()

            except Exception as e:
                logger.warning(f"ACSF failed for {smiles}: {e}")
                return None

        self.loader.df = self.loader.df.with_columns(
            pl.col("canonical_smiles")
            .map_elements(_get_acsf, return_dtype=pl.List(pl.Float64))
            .alias("acsf_embedding")
        )

        logger.success("ACSF computed. Loader now contains 'acsf_embedding' column.")
