from utils.file_ops import ensure_directory, validate_columns, validate_size
from src.features import MolecularFeaturizer
from src.geometry import GeometryPerturber
from src.distance import DistanceCalculator

import os
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Chem import AllChem,  Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mp_api.client import MPRester
from loguru import logger
import json
from tqdm import tqdm 
import selfies as sf
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
from dscribe.descriptors import SOAP, ACSF
from pymatgen.io.ase import AseAtomsAdaptor

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
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                
                # Syntactic and Complexity Descriptors
                "branching_index": sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2),
                "num_sp_carbons": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP),
                "num_sp2_carbons": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2),
                "num_sp3_carbons": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3),
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

    def add_selfies_transformer(self, model_name: str = "HUBioDataLab/SELFormer") -> None:
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

    def run_stress_test(
        self,
        num_molecules: int = 10,
        perturbations: int = 20,
        include_base: bool = True,
        max_rattle : float = 0.5,
        mol_ids: Optional[List[str]] = None,
        rotated: bool = False
    ) -> list:
        """Runs the geometry stress test using the geometry engine."""
        default_path = self.geometry_engine.save_path
        target_path = (
            os.path.join(self.root, "stress_test_rotated.xyz")
            if rotated
            else default_path
        )

        if os.path.exists(target_path) and mol_ids is None and not include_base and not max_rattle > 0:
            return self.geometry_engine.load_stress_test(
                save_path=target_path,
                mol_ids=mol_ids
            )
        
        return self.geometry_engine.generate_stress_test(
            self.df,
            num_molecules=num_molecules,
            mol_ids=mol_ids,
            perturbations=perturbations,
            include_base=include_base,
            rotated=rotated,
            save_path=target_path,
            max_rattle=max_rattle
        )

    def get_grassmann_distance_matrix(self, frames: List) -> np.ndarray:
        """
        Computes a Grassmann distance matrix using only ASE frames as input.
        """
        return self.geometry_engine.get_grassmann_distance_matrix(frames)
    
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
    

class MaterialsProject:
    def __init__(
        self,
        base_path: str = "data/Materials Project/",
        file_name: str = "stable_oxides.parquet",
        config_path: str = "config/api_key.json",
    ) -> None:
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.api_key = self._load_api_key(config_path)
        self.df = pl.DataFrame()
        os.makedirs(self.base_path, exist_ok=True)

    def _load_api_key(self, path: str) -> Optional[str]:
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            return config.get("key")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load API key from {path}: {e}")
            return None

    def load(self, force_fetch: bool = False, limit: int = 1000) -> pl.DataFrame:
        if os.path.exists(self.file_path) and not force_fetch:
            logger.info(f"Loading cached Parquet data from {self.file_path}...")
            self.df = pl.read_parquet(self.file_path)
            if "soap_embedding" not in self.df.columns or "acsf_embedding" not in self.df.columns:
                logger.info("Descriptors not found in cached data, computing them now.")
                self._add_descriptors()
                self.df.write_parquet(self.file_path)
                logger.success("Descriptors added and Parquet updated.")
            return self.df
        return self._fetch_from_api(limit)

    def _fetch_from_api(self, limit: int) -> pl.DataFrame:
        logger.info(f"Fetching {limit} stable oxides from API...")
        if not self.api_key:
            raise ValueError("API Key not found. Check your config path.")

        with MPRester(self.api_key) as mpr:
            query_kwargs = dict(
                is_stable=True,
                elements=["O"],
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "symmetry", "energy_per_atom", "formation_energy_per_atom",
                    "density"
                ],
            )

            # mp-api versions no longer expose mpr.listen; paginate via search instead.
            chunk_size = min(max(limit, 1), 1000)
            num_chunks = max((limit + chunk_size - 1) // chunk_size, 1)
            try:
                docs = mpr.materials.summary.search(
                    **query_kwargs,
                    chunk_size=chunk_size,
                    num_chunks=num_chunks,
                )
            except TypeError:
                try:
                    docs = mpr.materials.summary.search(**query_kwargs, limit=limit)
                except TypeError:
                    docs = mpr.materials.summary.search(**query_kwargs)

            docs = list(docs)[:limit]

            data_list = [
                self._process_doc(d)
                for d in tqdm(docs, desc="Processing materials")
            ]

            self.df = pl.DataFrame(data_list)
            self._add_descriptors()

            self.df.write_parquet(self.file_path)
            logger.success(f"Dataset saved with {len(self.df)} entries.")

        return self.df

    def _process_doc(self, d) -> Dict[str, Any]:
        struct = d.structure
        lat = struct.lattice
        return {
            "material_id": str(d.material_id),
            "formula_pretty": str(d.formula_pretty),
            "energy_per_atom": float(d.energy_per_atom),
            "formation_energy_per_atom": float(d.formation_energy_per_atom),
            "raw_structure": json.dumps(struct.as_dict()),
            "crystal_system": str(d.symmetry.crystal_system),
            "space_group": str(d.symmetry.symbol),
            "density": float(d.density),
            "a": float(lat.a),
            "b": float(lat.b),
            "c": float(lat.c),
            "alpha": float(lat.alpha),
            "beta": float(lat.beta),
            "gamma": float(lat.gamma),
            "volume": float(struct.volume),
            "num_sites": int(len(struct))
        }

    def _get_structures(self) -> List[Structure]:
        """Reconstructs Pymatgen structures from JSON strings."""
        logger.info("Reconstructing Pymatgen structures from JSON...")
        struct_strings = self.df["raw_structure"].to_list()
        return [Structure.from_dict(json.loads(s)) for s in tqdm(struct_strings, desc="Reconstructing structures")]

    def _add_descriptors(self, r_cut=6.0, n_max=8, l_max=6, sigma=0.5) -> None:
        """Computes and adds SOAP and ACSF descriptors to the DataFrame."""
        structures = self._get_structures()
        
        unique_elements = sorted(list(set(el.symbol for s in structures for el in s.composition.elements)))
        
        # Compute SOAP
        if "soap_embedding" not in self.df.columns:
            logger.info(f"Computing Periodic SOAP for {len(structures)} structures...")
            soap_engine = SOAP(species=unique_elements, periodic=True, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, sparse=True)
            soap_features = self._compute_feature(structures, soap_engine, "SOAP")
            # print the size of the soap features
            print(f"SOAP features shape: {len(soap_features)} x {len(soap_features[0]) if soap_features else 0}")
            self.df = self.df.with_columns(pl.Series("soap_embedding", soap_features)) # the problem is here...
            logger.success("SOAP embeddings added.")

        # Compute ACSF
        if "acsf_embedding" not in self.df.columns:
            logger.info(f"Computing Periodic ACSF for {len(structures)} structures...")
            acsf_engine = ACSF(
                species=unique_elements, periodic=True, r_cut=r_cut,
                g2_params=[[1, 1], [1, 2], [1, 3]],
                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]]
            )
            acsf_features = self._compute_feature(structures, acsf_engine, "ACSF")
            self.df = self.df.with_columns(pl.Series("acsf_embedding", acsf_features))
            logger.success("ACSF embeddings added.")

    def _compute_feature(
        self,
        structures: List[Structure],
        engine,
        desc_name: str,
        batch_size: int = 32,
    ) -> List[Optional[List[float]]]:
        """Compute features in batches for better performance and fewer overhead calls."""

        features = []

        for i in tqdm(
            range(0, len(structures), batch_size),
            desc=f"{desc_name} progress",
        ):
            batch_structs = structures[i : i + batch_size]

            try:
                batch_atoms = [
                    AseAtomsAdaptor.get_atoms(s) for s in batch_structs
                ]

                batch_out = engine.create(batch_atoms, n_jobs=1)

                for vec in batch_out:
                    features.append(np.mean(vec, axis=0).tolist())

            except Exception as e:
                logger.warning(f"{desc_name} batch failed: {e}")

                for s in batch_structs:
                    try:
                        atoms = AseAtomsAdaptor.get_atoms(s)
                        vec = np.mean(engine.create(atoms, n_jobs=1), axis=0).tolist()
                        features.append(vec)
                    except Exception as e2:
                        logger.warning(f"{desc_name} failed for a structure: {e2}")
                        features.append(None)

        return features
