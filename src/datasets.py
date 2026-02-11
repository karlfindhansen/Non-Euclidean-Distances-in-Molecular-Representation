import os
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import AllChem,  Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mp_api.client import MPRester
from loguru import logger
from ase import Atoms
from ase.io import write, read
import selfies as sf
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler

import os
import polars as pl
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import selfies as sf
from loguru import logger

#Import the refactored modules
from utils.file_ops import ensure_directory, validate_columns, validate_size
from src.features import MolecularFeaturizer
from src.geometry import GeometryPerturber
from src.distance import DistanceCalculator

class QM9Dataset:
    """
    Orchestrator for the QM9 dataset. 
    Manages loading raw data and delegating complex tasks to specialized classes.
    """
    
    QM9_TARGETS = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", 
        "u", "h", "g", "cv", "u0_atom", "u_atom", "h_atom", "g_atom", 
        "A", "B", "C"
    ]
    REQUIRED_COLUMNS = {"mol_id", "canonical_smiles", "num_atoms", "selfies"}

    def __init__(self, root: str = "data/QM9", filename: str = "dataset_cleaned.csv", subset_size: int = 2000):
        self.root = root
        self.filename = filename
        self.file_path = os.path.join(root, filename)
        self.subset_size = subset_size
        self.df = pl.DataFrame()
        self.scaler = StandardScaler()
        self.is_scaled = False
        
        ensure_directory(self.root)
        
        # Initialize Sub-Components
        self.geometry_engine = GeometryPerturber(save_path=os.path.join(root, "stress_test.xyz"))
        self.distance_engine = DistanceCalculator(cache_dir=root)

    def load(self, force_process: bool = False) -> pl.DataFrame:
        """Loads the main dataset, processing if necessary."""
        if os.path.exists(self.file_path) and not force_process:
            logger.info(f"Loading QM9 from {self.file_path}...")
            try:
                self.df = pl.read_csv(self.file_path)
                validate_columns(self.df, self.REQUIRED_COLUMNS)
                validate_size(self.df, self.subset_size)
                return self.df
            except Exception as e:
                logger.error(f"Load failed ({e}). Reprocessing...")

        self._process_raw_qm9()
        return self.df

    def _process_raw_qm9(self) -> None:
        """Downloads and cleans raw QM9 data from Torch Geometric."""
        logger.info(f"Processing raw QM9 data (Limit: {self.subset_size})...")
        try:
            dataset = QM9(root=self.root)
        except Exception as e:
            raise RuntimeError(f"QM9 download failed: {e}")

        data_list = []
        for i, data in enumerate(dataset):
            if len(data_list) >= self.subset_size: break
            
            smiles = getattr(data, 'smiles', None)
            if not smiles: continue

            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue

            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)

            # Determine Structure Class
            if n_rings == 0:
                struct_class = "Acyclic"
            elif n_arom > 0:
                struct_class = "Aromatic"
            else:
                struct_class = "Aliphatic Ring"
            
            # Basic Featurization
            canonical = Chem.MolToSmiles(mol, canonical=True)
            mol_dict = {
                "mol_id": f"qm9_{i}",
                "canonical_smiles": canonical,
                "selfies": sf.encoder(canonical),
                "num_atoms": int(data.num_nodes),
                "structure_class": struct_class,

                # Physical Properties
                "mol_weight": Descriptors.MolWt(mol),        # Molecular Weight
                "logp": Descriptors.MolLogP(mol),            # Lipophilicity
                "tpsa": Descriptors.TPSA(mol),               # Polar Surface Area
                
                # Structural/Complexity Descriptors
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                
                # Flexibility Descriptors (Crucial for your Stress Test context)
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol), # 3D complexity
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
            }

            mol_dict.update(dict(zip(self.QM9_TARGETS, data.y.tolist()[0])))
            data_list.append(mol_dict)

        self.df = pl.DataFrame(data_list)
        self.df.write_csv(self.file_path)
        logger.success(f"Saved processed dataset to {self.file_path}")

    def add_morgan_fingerprints(self, radius: int = 3, fp_size: int = 2048) -> None:
        """Adds 'morgan_fingerprint' column to the dataframe in-place."""
        if "morgan_fingerprint" in self.df.columns: return

        fp_series = MolecularFeaturizer.compute_morgan_fingerprints(
            self.df["canonical_smiles"], radius, fp_size
        )
        self.df = self.df.with_columns(fp_series.alias("morgan_fingerprint"))

    def add_selfies_transformer(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1") -> None:
        """Adds transformer embeddings to the dataframe in-place."""
        if "selfies_transformer" in self.df.columns: return
        
        emb_series = MolecularFeaturizer.compute_selfies_transformer(
            self.df["selfies"], model_name
        )
        self.df = self.df.with_columns(emb_series)

    def add_selfies_onehot(self) -> None:
        """Adds 'selfies_onehot' column to the dataframe."""
        if "selfies_onehot" in self.df.columns: return
            
        onehot_series = MolecularFeaturizer.compute_selfies_onehot(self.df["selfies"])
        self.df = self.df.with_columns(onehot_series)
        logger.success("Added One-Hot embeddings to DataFrame.")

    def get_distance_matrix(self, metric: str = 'morgan', dist_type: str = 'jaccard') -> 'np.ndarray':
        """Calculates distance matrix using the distance engine."""
        if metric == 'morgan':
            self.add_morgan_fingerprints()
            return self.distance_engine.get_matrix(
                self.df["morgan_fingerprint"], 
                metric=dist_type, 
                filename=f"dist_morgan_{dist_type}.npy"
            )
        elif metric == 'selfies_transformer':
            self.add_selfies_transformer()
            return self.distance_engine.get_matrix(
                self.df["selfies_transformer"], 
                metric=dist_type, 
                filename=f"dist_selfies_transformer_{dist_type}.npy"
            )
        
        elif metric == 'selfies_onehot':
            self.add_selfies_onehot()

            flattened_onehot = self.df["selfies_onehot"].map_elements(
                lambda x: np.array(x).flatten().tolist() if x is not None else None,
                return_dtype=pl.Object
            )
            
            return self.distance_engine.get_matrix(
                flattened_onehot, 
                metric=dist_type, 
                filename=f"dist_selfies_onehot_{dist_type}.npy"
            )

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def run_stress_test(self, num_molecules: int = 10) -> list:
        """Runs the geometry stress test using the geometry engine."""
        if os.path.exists(self.geometry_engine.save_path):
            return self.geometry_engine.load_stress_test()
        
        return self.geometry_engine.generate_stress_test(
            self.df, num_molecules=num_molecules
        )
    
    def apply_scaling(self, columns, mode="fit_transform"):
        """
        Standardizes specific numeric columns.
        Modes: 
          'fit_transform' -> Use for Training data
          'transform'     -> Use for Test/Validation data
        """
        if self.df.is_empty():
            logger.error("No data to scale!")
            return

        if mode == "fit_transform":
            logger.info(f"Fitting and transforming columns: {columns}")
            scaled_values = self.scaler.fit_transform(self.df.select(columns).to_numpy())
            self.is_scaled = True
        else:
            logger.info(f"Transforming columns using existing parameters: {columns}")
            scaled_values = self.scaler.transform(self.df.select(columns).to_numpy())

        scaled_df = pl.DataFrame(scaled_values, schema=columns)
        self.df = self.df.with_columns([scaled_df[col] for col in columns])
    

class MaterialsProjectLoader():
    """
    Loader for Materials Project crystallographic data.
    
    Handles fetching and caching of stable oxide materials from the Materials Project API.
    """
    
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