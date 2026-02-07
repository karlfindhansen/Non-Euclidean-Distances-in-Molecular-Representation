import os
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mp_api.client import MPRester
from loguru import logger
from ase import Atoms
from ase.io import write, read


class DataLoaderBase:
    """Base class for data loaders with common utility methods."""
    
    @staticmethod
    def _ensure_directory(path: str) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            path: Directory path to create
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def _validate_dataframe_columns(df: pl.DataFrame, required_columns: set) -> None:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: Set of required column names
            
        Raises:
            ValueError: If required columns are missing
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty")
        
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class QM9Loader(DataLoaderBase):
    """
    Loader for QM9 molecular dataset.
    
    Handles downloading, processing, and caching of QM9 molecular data,
    as well as generating perturbed molecular geometries for stress testing.
    """
    
    # Standard QM9 target names in order
    QM9_TARGETS = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", 
        "u", "h", "g", "cv", "u0_atom", "u_atom", "h_atom", "g_atom", 
        "A", "B", "C"
    ]
    
    # Required columns for validation
    REQUIRED_COLUMNS = {"mol_id", "canonical_smiles", "num_atoms"}

    def __init__(
        self, 
        root: str = "data/QM9", 
        filename: str = "dataset_cleaned.csv", 
        subset_size: int = 2000
    ) -> None:
        """
        Initialize QM9 data loader.
        
        Args:
            root: Root directory for QM9 data storage
            filename: Name of the processed CSV file
            subset_size: Number of molecules to process from QM9
        """
        self.root = root
        self.filename = filename
        self.file_path = os.path.join(self.root, self.filename)
        self.stress_test_path = os.path.join(self.root, "stress_test_perturbations.xyz")
        self.subset_size = subset_size
        self.df = pl.DataFrame()

        self._ensure_directory(self.root)

    def load_data(self, force_process: bool = False) -> pl.DataFrame:
        """
        Load QM9 data and ensure the Grassmann Stress Test exists.
        
        Args:
            force_process: If True, re-download and re-process the main QM9 CSV
                          (Does not overwrite stress test unless it is missing)
        
        Returns:
            Polars DataFrame containing QM9 molecular data
            
        Raises:
            ValueError: If no valid molecules are found or data is corrupted
        """
        # 1. Load Main Dataset
        if os.path.exists(self.file_path) and not force_process:
            logger.info(f"Found existing QM9 dataset at {self.file_path}. Loading with Polars...")
            try:
                self.df = pl.read_csv(self.file_path)
                self._validate_dataframe_columns(self.df, self.REQUIRED_COLUMNS)
            except (pl.exceptions.ComputeError, ValueError) as e:
                logger.error(f"Failed to load or validate existing dataset: {e}")
                logger.info("Attempting to reprocess QM9 data...")
                self._process_qm9()
        else:
            self._process_qm9()

        # 2. Handle Stress Test (Only generate if missing)
        if os.path.exists(self.stress_test_path):
            logger.info(f"Found existing Grassmann Stress Test at {self.stress_test_path}. Skipping generation.")
        else:
            self._generate_grassmann_stress_test()
        
        return self.df

    def _process_qm9(self) -> None:
        """
        Download and process raw QM9 data.
        
        Raises:
            RuntimeError: If QM9 dataset cannot be loaded
            ValueError: If no valid molecules are found
        """
        logger.info(f"Processing QM9 data (Target: {self.subset_size} molecules)...")
        
        try:
            dataset = QM9(root=self.root)
        except Exception as e:
            logger.error(f"Failed to load QM9 dataset: {e}")
            raise RuntimeError(f"Could not download/load QM9 dataset: {e}") from e

        data_list = []
        skipped_count = 0

        for i, data in enumerate(dataset):
            if len(data_list) >= self.subset_size:
                break
            
            smiles = getattr(data, 'smiles', None)
            if not smiles:
                skipped_count += 1
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    skipped_count += 1
                    continue
                
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                formula = CalcMolFormula(mol)
                
                mol_dict = {
                    "mol_id": f"qm9_{i}",
                    "name": formula, 
                    "original_smiles": smiles,
                    "canonical_smiles": canonical_smiles,
                    "num_atoms": int(data.num_nodes), 
                }
                
                # Add QM9 target properties
                mol_dict.update(dict(zip(self.QM9_TARGETS, data.y.tolist()[0])))
                data_list.append(mol_dict)
                
            except (ValueError, AttributeError, IndexError) as e:
                logger.debug(f"Skipping molecule {i} due to error: {e}")
                skipped_count += 1
                continue

        if not data_list:
            raise ValueError("No valid molecules found in QM9 dataset")
        
        logger.info(f"Successfully processed {len(data_list)} molecules ({skipped_count} skipped)")
        
        self.df = pl.DataFrame(data_list)
        
        try:
            self.df.write_csv(self.file_path)
            logger.success(f"QM9 dataset saved to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to save QM9 dataset: {e}")
            raise

    def _generate_grassmann_stress_test(
        self, 
        num_molecules: int = 10, 
        perturbations: int = 20, 
        stdev: float = 0.1, 
        seed: int = 40
    ) -> None:
        """
        Generate perturbed molecular geometries for stress testing.
        
        Args:
            num_molecules: Number of molecules to perturb
            perturbations: Number of perturbations per molecule
            stdev: Standard deviation of Gaussian noise (in Angstroms)
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If DataFrame is empty or insufficient molecules available
        """
        logger.info(f"Generating Grassmann Stress Test (Seed={seed})...")
        
        if self.df.is_empty():
            raise ValueError("Cannot generate stress test: QM9 DataFrame is empty")
        
        # Adjust num_molecules if dataset is too small
        available_molecules = len(self.df)
        if available_molecules < num_molecules:
            logger.warning(
                f"Only {available_molecules} molecules available, "
                f"requested {num_molecules}. Adjusting to available count."
            )
            num_molecules = available_molecules

        sample_df = self.df.sample(n=num_molecules, seed=seed)
        np.random.seed(seed)
        all_frames = []
        failed_molecules = 0

        for row in sample_df.iter_rows(named=True):
            mol_id = row['mol_id']
            smiles = row['canonical_smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    failed_molecules += 1
                    continue
                    
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                params = AllChem.ETKDG()
                params.randomSeed = seed
                
                if AllChem.EmbedMolecule(mol, params) == -1:
                    logger.debug(f"Failed to embed molecule {mol_id}")
                    failed_molecules += 1
                    continue
                
                base_positions = mol.GetConformer().GetPositions()
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

                # Generate perturbations
                for i in range(perturbations):
                    pert_atoms = Atoms(symbols=symbols, positions=base_positions.copy())
                    noise = np.random.normal(0.0, stdev, base_positions.shape)
                    pert_atoms.positions += noise
                    
                    pert_atoms.info.update({
                        'mol_id': mol_id, 
                        'perturbation_idx': i, 
                        'smiles': smiles
                    })
                    all_frames.append(pert_atoms)
                    
            except Exception as e:
                logger.debug(f"Error processing molecule {mol_id}: {e}")
                failed_molecules += 1
                continue

        if not all_frames:
            raise ValueError("Failed to generate any perturbed structures")
        
        logger.info(
            f"Generated {len(all_frames)} perturbed structures from "
            f"{num_molecules - failed_molecules} molecules ({failed_molecules} failed)"
        )
        
        try:
            write(self.stress_test_path, all_frames)
            logger.success(f"Stress test saved to {self.stress_test_path}")
        except Exception as e:
            logger.error(f"Failed to save stress test: {e}")
            raise

    def get_stress_test_data(self) -> List[Atoms]:
        """
        Load the stress test data.
        
        Returns:
            List of ASE Atoms objects containing perturbed molecular geometries
        """
        if not os.path.exists(self.stress_test_path):
            logger.warning("No stress test file found.")
            return []
        
        try:
            return read(self.stress_test_path, index=":")
        except Exception as e:
            logger.error(f"Failed to read stress test file: {e}")
            return []


class MaterialsProjectLoader(DataLoaderBase):
    """
    Loader for Materials Project crystallographic data.
    
    Handles fetching and caching of stable oxide materials from the Materials Project API.
    """
    
    # Required columns for validation
    REQUIRED_COLUMNS = {"material_id", "formula_pretty", "energy_per_atom"}

    def __init__(
        self, 
        base_path: str = "data/Materials Project/", 
        file_name: str = "stable_oxides.csv",
        config_path: str = "config/api_key.json"
    ) -> None:
        """
        Initialize Materials Project data loader.
        
        Args:
            base_path: Base directory for Materials Project data storage
            file_name: Name of the processed CSV file
            config_path: Path to API key configuration file
        """
        self.api_key = self._load_api_key(config_path)
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.df = pl.DataFrame()

        self._ensure_directory(self.base_path)

    @staticmethod
    def _load_api_key(config_path: str) -> Optional[str]:
        """
        Load API key from configuration file.
        
        Args:
            config_path: Path to JSON config file containing API key
            
        Returns:
            API key string or None if loading fails
        """
        try:
            config = pl.read_json(config_path)
            api_key = config['key'][0]
            logger.success("Successfully loaded Materials Project API key")
            return api_key
        except (FileNotFoundError, KeyError, IndexError, pl.exceptions.ComputeError) as e:
            logger.warning(f"Could not load API key from {config_path}: {e}")
            return None

    def load_data(self, force_fetch: bool = False, limit: int = 1000) -> pl.DataFrame:
        """
        Load Materials Project data, fetching from API if necessary.
        
        Args:
            force_fetch: If True, fetch fresh data from API regardless of cached file
            limit: Maximum number of materials to fetch
            
        Returns:
            Polars DataFrame containing materials data
            
        Raises:
            ValueError: If API key is missing or no materials are found
        """
        if os.path.exists(self.file_path) and not force_fetch:
            logger.info(f"Found existing cleaned dataset at {self.file_path}. Loading...")
            try:
                self.df = pl.read_csv(self.file_path)
                self._validate_dataframe_columns(self.df, self.REQUIRED_COLUMNS)
                return self.df
            except (pl.exceptions.ComputeError, ValueError) as e:
                logger.error(f"Failed to load or validate existing dataset: {e}")
                logger.info("Attempting to fetch fresh data from API...")
        
        return self._fetch_from_api(limit)
    
    def _fetch_from_api(self, limit: int) -> pl.DataFrame:
        """
        Fetch materials data from Materials Project API.
        
        Args:
            limit: Maximum number of materials to fetch
            
        Returns:
            Polars DataFrame containing materials data
            
        Raises:
            ValueError: If API key is missing
            RuntimeError: If API request fails
        """
        logger.info(f"Fetching up to {limit} stable oxides from Materials Project...")
        
        if not self.api_key:
            raise ValueError(
                "Materials Project API key is missing. "
                "Please provide valid API key in config file."
            )

        try:
            with MPRester(self.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    is_stable=True,
                    elements=["O"],
                    fields=[
                        "material_id", 
                        "formula_pretty", 
                        "structure", 
                        "symmetry", 
                        "energy_per_atom", 
                        "formation_energy_per_atom"
                    ]
                )
                
                if not docs:
                    logger.warning("No materials found matching criteria.")
                    return pl.DataFrame()

                # Limit results
                docs = docs[:limit]
                logger.info(f"Retrieved {len(docs)} materials from API")

                # Convert to dictionaries
                raw_data = [doc.dict() for doc in docs]
                raw_df = pl.DataFrame(raw_data)

                logger.info("Flattening nested crystal structures...")
                
                # Extract nested fields
                self.df = raw_df.with_columns([
                    pl.col("symmetry").struct.field("crystal_system").alias("crystal_system"),
                    pl.col("symmetry").struct.field("symbol").alias("space_group"),
                    
                    pl.col("structure").struct.field("lattice").struct.field("a").alias("a"),
                    pl.col("structure").struct.field("lattice").struct.field("b").alias("b"),
                    pl.col("structure").struct.field("lattice").struct.field("c").alias("c"),
                    pl.col("structure").struct.field("lattice").struct.field("alpha").alias("alpha"),
                    pl.col("structure").struct.field("lattice").struct.field("beta").alias("beta"),
                    pl.col("structure").struct.field("lattice").struct.field("gamma").alias("gamma"),
                    pl.col("structure").struct.field("lattice").struct.field("volume").alias("volume"),
                ]).drop(["symmetry", "structure", "fields_not_requested"], strict=False) 

                # Save to file
                self.df.write_csv(self.file_path)
                logger.success(f"Cleaned materials dataset saved to {self.file_path}")
                
        except ValueError as e:
            # Re-raise ValueError (e.g., from missing API key)
            raise
        except Exception as e:
            logger.error(f"Materials Project API request failed: {e}")
            raise RuntimeError(f"Failed to fetch data from Materials Project API: {e}") from e

        return self.df