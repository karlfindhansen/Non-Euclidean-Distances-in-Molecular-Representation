from utils.file_ops import ensure_directory, validate_columns, validate_size
from src.features import MolecularFeaturizer
from src.geometry import GeometryPerturber
from src.distance import DistanceCalculator

import os
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import AllChem,  Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mp_api.client import MPRester
from loguru import logger
from ase import Atoms
from ase.io import write, read
import selfies as sf
import torch
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler

from transformers import logging as tf_log
tf_log.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["REPORT_TO"] = "none"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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
    REQUIRED_COLUMNS = {"mol_id", "smiles", "canonical_smiles", "num_atoms", "selfies"}

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
        buffer_size = int(self.subset_size * 1.1)
        for i, data in enumerate(dataset):
            if len(data_list) >= buffer_size: break
            smiles = getattr(data, 'smiles', None)
            if not smiles: continue

            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue

            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res != 0: continue

            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)

            if n_rings == 0:
                struct_class = "Acyclic"
            elif n_arom > 0:
                struct_class = "Aromatic"
            else:
                struct_class = "Aliphatic Ring"
            
            canonical = Chem.MolToSmiles(mol, canonical=True)
            selfies_str = sf.encoder(canonical)

            dist_matrix = Chem.GetDistanceMatrix(mol)

            mol_dict = {
                "mol_id": f"qm9_{i}",
                "smiles": smiles, 
                "canonical_smiles": canonical,
                "selfies": selfies_str,
                "num_atoms": int(data.num_nodes),
                "structure_class": struct_class,

                # Physical Properties
                "mol_weight": Descriptors.MolWt(mol),        
                "logp": Descriptors.MolLogP(mol),            
                "tpsa": Descriptors.TPSA(mol),               
                
                # Structural/Complexity Descriptors
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                
                # Flexibility/Complexity & newly added string/graph complexity metrics
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol), # Already captures sp3 fraction
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                
                # Syntactic and Complexity Descriptors
                "branching_index": sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2),
                "num_sp_carbons": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP),
                "main_chain_length":  int(dist_matrix.max()) if len(dist_matrix) > 0 else 0,
                "raw_token_count": selfies_str.count('['),

                # These count specific chemical motifs
                "fr_benzene": Fragments.fr_benzene(mol),           # Benzene rings
                "fr_alcohol": Fragments.fr_Al_OH(mol),             # Aliphatic alcohols
                "fr_phenol": Fragments.fr_Ar_OH(mol),              # Aromatic alcohols
                "fr_amine": Fragments.fr_NH2(mol),                 # Primary amines
                "fr_amide": Fragments.fr_amide(mol),               # Amide groups
                "fr_carboxylic_acid": Fragments.fr_COO(mol),       # Carboxylic acids
                "fr_ester": Fragments.fr_ester(mol),               # Ester groups
                "fr_ketone": Fragments.fr_ketone(mol),             # Ketones
                "fr_ether": Fragments.fr_ether(mol),               # Ether linkages
                "fr_nitro": Fragments.fr_nitro(mol),               # Nitro groups
                "fr_halogen": rdMolDescriptors.CalcNumHeteroatoms(mol), # Simple heteroatom count
            }

            mol_dict.update(dict(zip(self.QM9_TARGETS, data.y.tolist()[0])))
            data_list.append(mol_dict)

        self.df = pl.DataFrame(data_list)

        self.add_soap()
        self.add_acsf()

        valid_mask = (
            pl.col("soap_embedding").is_not_null() & 
            pl.col("acsf_embedding").is_not_null()
        )
        
        valid_count = self.df.filter(valid_mask).height
        logger.info(f"Valid molecules (SOAP+ACSF success): {valid_count}")

        self.df = (
            self.df
            .filter(valid_mask)         
            .unique(subset=["canonical_smiles"], keep="first")     
            .head(self.subset_size)
            .drop(["soap_embedding", "acsf_embedding"])
        )

        self.df.write_csv(self.file_path)
        logger.success(f"Saved processed dataset to {self.file_path}")

    def add_morgan_fingerprints(self, radius: int = 3, fp_size: int = 2048) -> None:
        if "morgan_fingerprint" in self.df.columns: return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_morgan_fingerprints(
                self.df["canonical_smiles"], radius, fp_size
            ).alias("morgan_fingerprint")
        )

    def add_selfies_transformer(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1") -> None:
        if "selfies_transformer" in self.df.columns: return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_selfies_transformer(
                self.df["selfies"], model_name
            )
        )

    def add_selfies_onehot(self) -> None:
        if "selfies_onehot" in self.df.columns: return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_selfies_onehot(self.df["selfies"])
        )

    def add_soap(self, r_cut=6.0, n_max=8, l_max=6, sigma=0.5) -> None:
        """Adds SOAP descriptors to the dataframe."""
        if "soap_embedding" in self.df.columns: return
        
        soap_series = MolecularFeaturizer.compute_soap(
            self.df["canonical_smiles"], 
            r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma
        )
        self.df = self.df.with_columns(soap_series.alias("soap_embedding"))
        logger.success("Added SOAP embeddings.")

    def add_acsf(self, r_cut=6.0) -> None:
        """Adds ACSF descriptors to the dataframe."""
        if "acsf_embedding" in self.df.columns: return
        
        acsf_series = MolecularFeaturizer.compute_acsf(
            self.df["canonical_smiles"], r_cut=r_cut
        )
        self.df = self.df.with_columns(acsf_series.alias("acsf_embedding"))
        logger.success("Added ACSF embeddings.")

    def add_chemprop(
        self,
        model_path: str | None = None,
        batch_size: int = 64
    ) -> None:

        if "chemprop_embedding" in self.df.columns:
            return

        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_chemprop_embeddings(
                self.df["canonical_smiles"],
                model_path=model_path,
                batch_size=batch_size
            ).alias("chemprop_embedding")
        )
    

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
    

# TODO: Refractor and Rewrite...
import json
from pymatgen.core import Structure
from tqdm import tqdm
import polars as pl
import numpy as np
from mp_api.client import MPRester
from loguru import logger
from pymatgen.io.ase import AseAtomsAdaptor
from dscribe.descriptors import SOAP, ACSF

class MaterialsProject:
    def __init__(
        self, 
        base_path: str = "data/Materials Project/", 
        file_name: str = "stable_oxides.parquet",
        config_path: str = "config/api_key.json"
    ) -> None:
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.api_key = self._load_api_key(config_path)
        self.df = pl.DataFrame()
        os.makedirs(self.base_path, exist_ok=True)

    def _load_api_key(self, path):
        try:
            config = pl.read_json(path)
            # Handle both list and single value formats
            return config['key'][0] if isinstance(config['key'], (list, pl.Series)) else config['key']
        except Exception:
            return None

    def load(self, force_fetch: bool = False, limit: int = 1000) -> pl.DataFrame:
        if os.path.exists(self.file_path) and not force_fetch:
            logger.info(f"Loading cached Parquet data from {self.file_path}...")
            self.df = pl.read_parquet(self.file_path)
            return self.df
        return self._fetch_from_api(limit)
    
    def _fetch_from_api(self, limit: int) -> pl.DataFrame:
        logger.info(f"Fetching {limit} stable oxides from API...")
        if not self.api_key:
            raise ValueError("API Key not found. Check your config path.")

        with MPRester(self.api_key) as mpr:
            # Added formation_energy and density for better physical context
            docs = mpr.materials.summary.search(
                is_stable=True, 
                elements=["O"],
                fields=[
                    "material_id", "formula_pretty", "structure", 
                    "symmetry", "energy_per_atom", "formation_energy_per_atom",
                    "density"
                ]
            )
            docs = docs[:limit]
            
            data_list = []
            for d in docs:
                struct = d.structure
                lat = struct.lattice
                
                struct_json = json.dumps(struct.as_dict())
                
                data_list.append({
                    "material_id": str(d.material_id),
                    "formula_pretty": str(d.formula_pretty),
                    "energy_per_atom": float(d.energy_per_atom),
                    "formation_energy_per_atom": float(d.formation_energy_per_atom),
                    "raw_structure": struct_json,
                    "crystal_system": str(d.symmetry.crystal_system),
                    "space_group": str(d.symmetry.symbol),
                    "density": float(d.density),
                    # --- GEOMETRIC SUFFICIENCY ---
                    "a": float(lat.a),
                    "b": float(lat.b),
                    "c": float(lat.c),
                    "alpha": float(lat.alpha),
                    "beta": float(lat.beta),
                    "gamma": float(lat.gamma),
                    "volume": float(struct.volume),
                    "num_sites": int(len(struct))
                })

            self.df = pl.DataFrame(data_list)
            
            # Cast to ensure consistency
            self.df = self.df.with_columns([
                pl.col("raw_structure").cast(pl.String)
            ])
            
            self.df.write_parquet(self.file_path)
            logger.success(f"Dataset saved with {len(self.df)} entries.")
            
        return self.df

    def _get_structures(self):
        """Reconstructs Pymatgen structures from JSON strings."""
        logger.info("Reconstructing Pymatgen structures from JSON...")
        struct_strings = self.df["raw_structure"].to_list()
        return [Structure.from_dict(json.loads(s)) for s in struct_strings]

    def add_soap(self, r_cut=6.0, n_max=8, l_max=6, sigma=0.5) -> None:
        """Computes SOAP embeddings and updates the Parquet file."""
        if "soap_embedding" in self.df.columns:
            logger.info("SOAP embeddings already exist. Skipping.")
            return
        
        structures = self._get_structures()
        
        # Get unique species for the descriptor environment
        unique_elements = set()
        for s in structures:
            unique_elements.update([e.symbol for e in s.composition.elements])
        species_list = sorted(list(unique_elements))
        
        logger.info(f"Computing Periodic SOAP for {len(structures)} structures...")
        soap_engine = SOAP(
            species=species_list, 
            periodic=True,
            r_cut=r_cut, 
            n_max=n_max, 
            l_max=l_max, 
            sigma=sigma
        )

        features = []
        for struct in tqdm(structures, desc="SOAP progress"):
            try:
                atoms = AseAtomsAdaptor.get_atoms(struct)
                # create() returns (n_atoms, n_features), we mean-pool to (n_features,)
                vec = np.mean(soap_engine.create(atoms), axis=0).tolist()
                features.append(vec)
            except Exception as e:
                logger.warning(f"SOAP failed: {e}")
                features.append(None)

        self.df = self.df.with_columns(pl.Series("soap_embedding", features))
        self.df.write_parquet(self.file_path)
        logger.success("SOAP embeddings added and Parquet updated.")

    def add_acsf(self, r_cut=6.0) -> None:
        """Computes ACSF embeddings and updates the Parquet file."""
        if "acsf_embedding" in self.df.columns:
            logger.info("ACSF embeddings already exist. Skipping.")
            return
            
        structures = self._get_structures()
        unique_elements = set()
        for s in structures:
            unique_elements.update([e.symbol for e in s.composition.elements])
        species_list = sorted(list(unique_elements))

        logger.info(f"Computing Periodic ACSF for {len(structures)} structures...")
        acsf_engine = ACSF(
            species=species_list, 
            periodic=True, 
            r_cut=r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]]
        )

        features = []
        for struct in tqdm(structures, desc="ACSF progress"):
            try:
                atoms = AseAtomsAdaptor.get_atoms(struct)
                vec = np.mean(acsf_engine.create(atoms), axis=0).tolist()
                features.append(vec)
            except Exception as e:
                logger.warning(f"ACSF failed: {e}")
                features.append(None)

        self.df = self.df.with_columns(pl.Series("acsf_embedding", features))
        self.df.write_parquet(self.file_path)
        logger.success("ACSF embeddings added and Parquet updated.")