from utils.file_ops import ensure_directory, validate_columns, validate_size
from src.features import MolecularFeaturizer
from src.geometry import GeometryPerturber
from src.distance import DistanceCalculator

import os
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Any, Set
from torch_geometric.datasets import QM9
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit import RDLogger
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
    FUNCTIONAL_GROUP_DETECTORS = {
        "benzene": Fragments.fr_benzene,
        "alcohol": Fragments.fr_Al_OH,
        "phenol": Fragments.fr_Ar_OH,
        "amine": Fragments.fr_NH2,
        "amide": Fragments.fr_amide,
        "carboxylic_acid": Fragments.fr_COO,
        "ester": Fragments.fr_ester,
        "ketone": Fragments.fr_ketone,
        "ether": Fragments.fr_ether,
        "nitro": Fragments.fr_nitro,
    }
    REQUIRED_COLUMNS = {"mol_id", "smiles", "canonical_smiles", "num_atoms", "selfies", "formula", "functional_groups", "avg_bond_length"}

    def __init__(
        self,
        root: str = "data/QM9",
        filename: str = "dataset_cleaned.csv",
        subset_size: int = 2000,
        required_mol_ids: Optional[List[str]] = None,
        embed_seed: int = 40,
        sampling_strategy: str = "stratified",
        stratify_by: Optional[List[str]] = None,
        stratify_bins: int = 10,
        sampling_seed: int = 40,
        min_per_stratum: int = 1,
        sampling_buffer: float = 1.1,
    ):
        self.root = root
        self.filename = filename
        self.file_path = os.path.join(root, filename)
        self.subset_size = subset_size
        self.required_mol_ids = required_mol_ids or []
        self.embed_seed = embed_seed
        self.sampling_strategy = sampling_strategy
        self.stratify_by = stratify_by or ["num_atoms", "gap"]
        self.stratify_bins = stratify_bins
        self.sampling_seed = sampling_seed
        self.min_per_stratum = min_per_stratum
        self.sampling_buffer = sampling_buffer
        self.df = pl.DataFrame()
        self.scaler = StandardScaler()
        self.is_scaled = False
        
        ensure_directory(self.root)
        
        # Initialize Sub-Components
        self.geometry_engine = GeometryPerturber(save_path=os.path.join(root, "stress_test.xyz"))
        self.distance_engine = DistanceCalculator(cache_dir=root)

        RDLogger.DisableLog("rdApp.error")

    def _select_qm9_indices(self, dataset: QM9) -> List[int]:
        """Selects QM9 indices using a stratified sampling scheme."""
        if self.sampling_strategy not in {"stratified", "head"}:
            raise ValueError(
                f"Unsupported sampling_strategy='{self.sampling_strategy}'. "
                "Use 'stratified' or 'head'."
            )

        if self.sampling_strategy == "head":
            buffer_size = int(self.subset_size * 1.1)
            required_indices = self._parse_required_indices()
            max_required_index = max(required_indices) if required_indices else -1
            limit = max(buffer_size, max_required_index + 1)
            return list(range(min(limit, len(dataset))))

        required_indices = set(self._parse_required_indices())
        out_of_range = [i for i in required_indices if i < 0 or i >= len(dataset)]
        if out_of_range:
            logger.warning(
                "Some required mol_id indices are outside QM9 range and will be ignored: "
                f"{sorted(out_of_range)}"
            )
            required_indices = {i for i in required_indices if 0 <= i < len(dataset)}
        valid_stratify_keys = set(self.QM9_TARGETS) | {"num_atoms"}
        invalid = [k for k in self.stratify_by if k not in valid_stratify_keys]
        if invalid:
            raise ValueError(
                f"Invalid stratify_by key(s): {invalid}. "
                f"Valid keys: {sorted(valid_stratify_keys)}"
            )

        n = len(dataset)
        if n == 0:
            return []

        values: Dict[str, np.ndarray] = {}
        for key in self.stratify_by:
            values[key] = np.zeros(n, dtype=float)

        for i, data in enumerate(dataset):
            for key in self.stratify_by:
                if key == "num_atoms":
                    values[key][i] = float(data.num_nodes)
                else:
                    target_idx = self.QM9_TARGETS.index(key)
                    values[key][i] = float(data.y[0, target_idx].item())

        binned: Dict[str, np.ndarray] = {}
        for key in self.stratify_by:
            if key == "num_atoms":
                binned[key] = values[key].astype(int)
                continue
            series = values[key]
            if self.stratify_bins <= 1 or np.all(series == series[0]):
                binned[key] = np.zeros_like(series, dtype=int)
                continue
            edges = np.quantile(series, np.linspace(0, 1, self.stratify_bins + 1))
            edges = np.unique(edges)
            if edges.size <= 2:
                binned[key] = np.zeros_like(series, dtype=int)
                continue
            binned[key] = np.digitize(series, edges[1:-1], right=True)

        strata: Dict[tuple, List[int]] = {}
        for i in range(n):
            if i in required_indices:
                continue
            key = tuple(int(binned[k][i]) for k in self.stratify_by)
            strata.setdefault(key, []).append(i)

        target_size = int(np.ceil(self.subset_size * self.sampling_buffer))
        slots = max(target_size - len(required_indices), 0)
        if slots <= 0:
            return sorted(required_indices)

        total = sum(len(v) for v in strata.values())
        if total == 0:
            return sorted(required_indices)

        counts = {k: len(v) for k, v in strata.items()}
        num_strata = len(counts)
        min_take = 0
        if self.min_per_stratum > 0 and slots >= num_strata:
            min_take = self.min_per_stratum

        base_take = {k: min(min_take, counts[k]) for k in counts}
        remaining_slots = slots - sum(base_take.values())
        remaining_counts = {k: counts[k] - base_take[k] for k in counts}
        remaining_total = sum(remaining_counts.values())

        if remaining_slots > 0 and remaining_total > 0:
            raw = {k: remaining_counts[k] / remaining_total * remaining_slots for k in counts}
            take = {k: int(np.floor(raw[k])) for k in counts}
            remainder = remaining_slots - sum(take.values())
            if remainder > 0:
                frac = sorted(
                    ((raw[k] - take[k], k) for k in counts),
                    reverse=True
                )
                for _, k in frac[:remainder]:
                    take[k] += 1
        else:
            take = {k: 0 for k in counts}

        rng = np.random.default_rng(self.sampling_seed)
        selected = set(required_indices)
        for k, indices in strata.items():
            k_take = base_take[k] + take[k]
            if k_take <= 0:
                continue
            if k_take >= len(indices):
                selected.update(indices)
            else:
                chosen = rng.choice(indices, size=k_take, replace=False)
                selected.update(chosen.tolist())

        return sorted(selected)

    def _parse_required_indices(self) -> List[int]:
        required_set: Set[str] = set(self.required_mol_ids)
        required_indices = []
        for mol_id in required_set:
            if mol_id.startswith("qm9_"):
                suffix = mol_id.split("qm9_", 1)[1]
                if suffix.isdigit():
                    required_indices.append(int(suffix))
        return required_indices

    @staticmethod
    def _classify_structure_type(mol: Chem.Mol) -> str:
        """Classify molecule topology as aromatic, acyclic, or cyclic."""
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        if n_rings == 0:
            return "acyclic"
        if n_arom > 0:
            return "aromatic"
        return "cyclic"

    @classmethod
    def _detect_functional_groups(cls, mol: Chem.Mol) -> List[str]:
        """Return a compact list of detected functional-group labels."""
        groups = [
            name for name, detector in cls.FUNCTIONAL_GROUP_DETECTORS.items()
            if int(detector(mol)) > 0
        ]
        has_halogen = any(
            atom.GetAtomicNum() in {9, 17, 35, 53}
            for atom in mol.GetAtoms()
        )
        if has_halogen:
            groups.append("halogen")
        return groups

    @staticmethod
    def _compute_average_bond_length(mol: Chem.Mol) -> float:
        """Compute average bond length (Angstrom) from the molecule's conformer."""
        if mol.GetNumBonds() == 0:
            return 0.0
        conf = mol.GetConformer()
        total = 0.0
        count = 0
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            dist = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2) ** 0.5
            total += dist
            count += 1
        return float(total / count) if count > 0 else 0.0

    def _build_qm9_row(self, i: int, data) -> Optional[Dict[str, Any]]:
        smiles = getattr(data, "smiles", None)
        if not smiles:
            return None

        mol = self._embed_molecule(
            smiles=smiles,
            seed=self.embed_seed,
            invariant=True,
        )
        if mol is None:
            return None

        formula = CalcMolFormula(mol)

        structure_type = self._classify_structure_type(mol)
        if structure_type == "acyclic":
            struct_class = "Acyclic"
        elif structure_type == "aromatic":
            struct_class = "Aromatic"
        else:
            struct_class = "Aliphatic Ring"

        canonical = Chem.MolToSmiles(mol, canonical=True)
        selfies_str = sf.encoder(canonical)
        functional_groups = self._detect_functional_groups(mol)
        functional_groups_str = ",".join(functional_groups)

        dist_matrix = Chem.GetDistanceMatrix(mol)
        avg_bond_length = self._compute_average_bond_length(mol)

        num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        num_sp_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.HybridizationType.SP
        )
        num_sp2_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.HybridizationType.SP2
        )
        num_sp3_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.HybridizationType.SP3
        )
        denom_c = float(num_carbons) if num_carbons > 0 else 1.0
        fraction_csp1 = float(num_sp_carbons / denom_c)
        fraction_csp2 = float(num_sp2_carbons / denom_c)
        fraction_csp3 = float(num_sp3_carbons / denom_c)

        mol_dict = {
            "mol_id": f"qm9_{i}",
            "formula": formula,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "selfies": selfies_str,
            "functional_groups": functional_groups_str,
            "num_atoms": int(data.num_nodes),
            "structure_class": struct_class,

            # Physical Properties
            "mol_weight": int(Descriptors.MolWt(mol)),
            "logp": int(Descriptors.MolLogP(mol)),
            "tpsa": int(Descriptors.TPSA(mol)),

            # Structural/Complexity Descriptors
            "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),

            # Flexibility/Complexity & newly added string/graph complexity metrics
            "num_rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
            "fraction_csp1": fraction_csp1,
            "fraction_csp2": fraction_csp2,
            "fraction_csp3": fraction_csp3,
            "h_bond_donors": int(Descriptors.NumHDonors(mol)),
            "h_bond_acceptors": int(Descriptors.NumHAcceptors(mol)),

            # Syntactic and Complexity Descriptors
            "branching_index": sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2),
            "num_sp_carbons": int(num_sp_carbons),
            "num_sp2_carbons": int(num_sp2_carbons),
            "num_sp3_carbons": int(num_sp3_carbons),
            "main_chain_length": int(dist_matrix.max()) if len(dist_matrix) > 0 else 0,
            "raw_token_count": int(selfies_str.count("[")),
            "avg_bond_length": avg_bond_length,

            # These count specific chemical motifs
            "fr_benzene": int(Fragments.fr_benzene(mol)),
            "fr_alcohol": int(Fragments.fr_Al_OH(mol)),
            "fr_phenol": int(Fragments.fr_Ar_OH(mol)),
            "fr_amine": int(Fragments.fr_NH2(mol)),
            "fr_amide": int(Fragments.fr_amide(mol)),
            "fr_carboxylic_acid": int(Fragments.fr_COO(mol)),
            "fr_ester": int(Fragments.fr_ester(mol)),
            "fr_ketone": int(Fragments.fr_ketone(mol)),
            "fr_ether": int(Fragments.fr_ether(mol)),
            "fr_nitro": int(Fragments.fr_nitro(mol)),
            "fr_halogen": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        }

        mol_dict.update(dict(zip(self.QM9_TARGETS, data.y.tolist()[0])))
        return mol_dict

    def load(self, force_process: bool = False) -> pl.DataFrame:
        """Loads the main dataset, processing if necessary."""
        if os.path.exists(self.file_path) and not force_process:
            logger.info(f"Loading QM9 from {self.file_path}...")
            try:
                self.df = pl.read_csv(self.file_path)
                validate_columns(self.df, self.REQUIRED_COLUMNS)
                validate_size(self.df, self.subset_size)
                if self.required_mol_ids:
                    existing_ids = set(self.df["mol_id"].to_list())
                    missing_cached = sorted(set(self.required_mol_ids) - existing_ids)
                    if missing_cached:
                        raise ValueError(
                            f"Cached dataset is missing required mol_id(s): {missing_cached}"
                        )
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

        required_set: Set[str] = set(self.required_mol_ids)
        required_indices = self._parse_required_indices()
        max_required_index = max(required_indices) if required_indices else -1
        data_list = []
        if self.sampling_strategy == "head":
            buffer_size = int(self.subset_size * 1.1)
            for i, data in enumerate(dataset):
                # Keep iterating until we both have enough candidates and have passed required indices.
                if len(data_list) >= buffer_size and i >= max_required_index:
                    break
                mol_dict = self._build_qm9_row(i, data)
                if mol_dict is not None:
                    data_list.append(mol_dict)
        else:
            selected_indices = self._select_qm9_indices(dataset)
            selected_set = set(selected_indices)
            max_selected = max(selected_indices) if selected_indices else -1
            processed = 0
            logger.info(
                "Stratified sampling enabled. "
                f"Selected {len(selected_indices)} indices from QM9."
            )
            for i, data in enumerate(dataset):
                if processed >= len(selected_set) and i > max_selected:
                    break
                if i not in selected_set:
                    continue
                mol_dict = self._build_qm9_row(i, data)
                if mol_dict is None:
                    continue
                data_list.append(mol_dict)
                processed += 1

        self.df = pl.DataFrame(data_list)

        self.add_soap()
        self.add_acsf()
        self.add_coulomb_matrix()

        valid_mask = (
            pl.col("soap_embedding").is_not_null() & 
            pl.col("acsf_embedding").is_not_null() &
            pl.col("coulomb_matrix").is_not_null()
        )

        invalid_mask = (
            pl.col("soap_embedding").is_null() | 
            pl.col("acsf_embedding").is_null() |
            pl.col("coulomb_matrix").is_null()
        )

        failed_molecules = self.df.filter(invalid_mask)
        logger.warning(
            "Invalid molecules (SOAP+ACSF+Coulomb failure): "
            f"{failed_molecules.select('mol_id').to_series().to_list()}"
        )
        
        valid_count = self.df.filter(valid_mask).height
        logger.info(f"Valid molecules (SOAP+ACSF+Coulomb success): {valid_count}")
        required_df = (
            self.df.filter(pl.col("mol_id").is_in(list(required_set)))
            if required_set
            else pl.DataFrame(schema=self.df.schema)
        )
        if required_set:
            present_required = set(required_df["mol_id"].to_list())
            missing_required = sorted(required_set - present_required)
            if missing_required:
                logger.warning(f"Requested mol_id(s) were not found in processed candidates: {missing_required}")

        non_required_valid = (
            self.df
            .filter(valid_mask & ~pl.col("mol_id").is_in(list(required_set)))
        )
        if required_df.height > 0:
            required_canon = set(required_df["canonical_smiles"].to_list())
            non_required_valid = non_required_valid.filter(
                ~pl.col("canonical_smiles").is_in(list(required_canon))
            )

        deduped_non_required = non_required_valid.unique(subset=["canonical_smiles"], keep="first")
        slots_for_non_required = max(self.subset_size - required_df.height, 0)
        if slots_for_non_required <= 0:
            selected_non_required = pl.DataFrame(schema=self.df.schema)
        else:
            available = deduped_non_required.height
            if available < slots_for_non_required:
                logger.warning(
                    "Not enough non-required molecules after filtering to reach "
                    f"subset_size={self.subset_size}. Available={available}."
                )
            take_n = min(slots_for_non_required, available)
            if self.sampling_strategy == "head":
                selected_non_required = deduped_non_required.head(take_n)
            else:
                selected_non_required = deduped_non_required.sample(
                    n=take_n,
                    seed=self.sampling_seed,
                    shuffle=True
                )

        if required_df.height > 0:
            self.df = pl.concat([required_df, selected_non_required], how="vertical_relaxed")
        else:
            self.df = selected_non_required

        # Keep deterministic order by QM9 numeric index.
        self.df = (
            self.df
            .with_columns(
                pl.col("mol_id")
                .str.replace("qm9_", "")
                .cast(pl.Int64, strict=False)
                .alias("_qm9_idx")
            )
            .sort("_qm9_idx")
            .drop("_qm9_idx")
            .drop(["soap_embedding", "acsf_embedding", "coulomb_matrix"])
        )

        self.df.write_csv(self.file_path)
        logger.success(f"Saved processed dataset with {self.df.height} rows to {self.file_path}")

    @staticmethod
    def _embed_molecule(smiles: str, seed: int = 42, invariant: bool = True) -> Optional[Chem.Mol]:
        """
        Takes a SMILES string, generates a 3D conformer using RDKit, 
        assigns Gasteiger charges, and optionally ensures permutational invariance.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogens before embedding
            mol = Chem.AddHs(mol)
            
            # Compute partial charges
            AllChem.ComputeGasteigerCharges(mol)

            # Reorder atoms to ensure deterministic, invariant atom indexing
            if invariant:
                order = Chem.CanonicalRankAtoms(mol)
                mol = Chem.RenumberAtoms(mol, list(order))

            # Embed the molecule in 3D space using ETKDG
            params = AllChem.ETKDG()
            params.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params) == -1:
                return None

            return mol
            
        except Exception as e:
            logger.debug(f"Molecule embedding failed for SMILES '{smiles}': {e}")
            return None

    def add_morgan_fingerprints(self, radius: int = 3, fp_size: int = 2048) -> None:
        if "morgan_fingerprint" in self.df.columns: 
            return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_morgan_fingerprints(
                self.df["canonical_smiles"], radius, fp_size
            ).alias("morgan_fingerprint")
        )

    def add_selfies_transformer(self, model_name: str = "HUBioDataLab/SELFormer") -> None:
        if "selfies_transformer" in self.df.columns: 
            return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_selfies_transformer(
                self.df["selfies"], model_name
            )
        )

    def add_selfies_onehot(self, flatten: bool = False) -> None:
        if "selfies_onehot" in self.df.columns: 
            return
        self.df = self.df.with_columns(
            MolecularFeaturizer.compute_selfies_onehot(
                self.df["selfies"],
                flatten=flatten
            )
        )

    def add_soap(self, r_cut=6.0, n_max=8, l_max=6, sigma=0.5) -> None:
        """Adds SOAP descriptors to the dataframe."""
        if "soap_embedding" in self.df.columns: 
            return
        
        soap_series = MolecularFeaturizer.compute_soap(
            self.df["canonical_smiles"], 
            r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma
        )
        self.df = self.df.with_columns(soap_series.alias("soap_embedding"))
        logger.success("Added SOAP embeddings.")

    def add_acsf(self, r_cut=6.0) -> None:
        """Adds ACSF descriptors to the dataframe."""
        if "acsf_embedding" in self.df.columns: 
            return
        
        acsf_series = MolecularFeaturizer.compute_acsf(
            self.df["canonical_smiles"], r_cut=r_cut
        )
        self.df = self.df.with_columns(acsf_series.alias("acsf_embedding"))
        logger.success("Added ACSF embeddings.")

    def add_coulomb_matrix(
        self,
        n_atoms_max: int | None = None,
        permutation: str = "sorted_l2"
    ) -> None:
        """Adds Coulomb matrix descriptors to the dataframe."""
        if "coulomb_matrix" in self.df.columns:
            return

        coulomb_series = MolecularFeaturizer.compute_coulomb_matrix(
            self.df["canonical_smiles"],
            n_atoms_max=n_atoms_max,
            permutation=permutation
        )
        self.df = self.df.with_columns(coulomb_series.alias("coulomb_matrix"))
        logger.success("Added Coulomb matrix descriptors.")

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

    def add_all_descriptors(
        self,
        radius: int = 3,
        fp_size: int = 2048,
        model_name: str = "HUBioDataLab/SELFormer",
        r_cut: float = 6.0,
        n_max: int = 8,
        l_max: int = 6,
        sigma: float = 0.5,
        coulomb_n_atoms_max: int | None = None,
        coulomb_permutation: str = "sorted_l2",
        include_chemprop: bool = True,
        chemprop_model_path: str | None = None,
        chemprop_batch_size: int = 64,
    ) -> None:
        """
        Adds all available QM9 descriptor columns in one call.
        Existing columns are skipped by each add_* method.
        """
        if self.df.is_empty():
            raise ValueError("Dataset is empty. Call `load()` before adding descriptors.")

        logger.info("Adding all descriptors to QM9 dataframe...")
        self.add_morgan_fingerprints(radius=radius, fp_size=fp_size)
        self.add_selfies_transformer(model_name=model_name)
        self.add_selfies_onehot()
        self.add_soap(r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma)
        self.add_acsf(r_cut=r_cut)
        self.add_coulomb_matrix(
            n_atoms_max=coulomb_n_atoms_max,
            permutation=coulomb_permutation
        )

        if include_chemprop:
            self.add_chemprop(
                model_path=chemprop_model_path,
                batch_size=chemprop_batch_size
            )

        logger.success("Finished adding all requested descriptors.")
    

    def get_distance_matrix(self, descriptor: str = "morgan", dist_type: str = "jaccard") -> 'np.ndarray':
        """
        Computes a distance matrix for a chosen descriptor.

        Descriptors:
            - morgan
            - selfies_transformer (alias: transformer)
            - selfies_onehot (alias: onehot)
            - soap
            - acsf
            - coulomb_matrix
            - chemprop

        Distance types:
            - jaccard (Tanimoto)
            - euclidean
            - cosine
            - soap_kernel (1 - normalized SOAP dot product)
            - hamming
        """

        descriptor = descriptor.lower()
        aliases = {
            "transformer": "selfies_transformer",
            "onehot": "selfies_onehot",
        }
        descriptor = aliases.get(descriptor, descriptor)

        def _series_for(desc: str) -> pl.Series:
            if desc == "morgan":
                self.add_morgan_fingerprints()
                return self.df["morgan_fingerprint"]
            if desc == "selfies_transformer":
                self.add_selfies_transformer()
                return self.df["selfies_transformer"]
            if desc == "selfies_onehot":
                self.add_selfies_onehot(flatten=True)
                return self.df["selfies_onehot"]
            if desc == "soap":
                self.add_soap()
                return self.df["soap_embedding"]
            if desc == "acsf":
                self.add_acsf()
                return self.df["acsf_embedding"]
            if desc == "coulomb_matrix":
                self.add_coulomb_matrix()
                return self.df["coulomb_matrix"]
            if desc == "chemprop":
                self.add_chemprop()
                return self.df["chemprop_embedding"]
            raise ValueError(
                f"Unknown descriptor: {desc}. "
                "Expected one of: morgan, selfies_transformer, selfies_onehot, "
                "soap, acsf, coulomb_matrix, chemprop."
            )

        if dist_type == "jaccard" and descriptor not in {"morgan", "selfies_onehot"}:
            logger.warning(
                "Jaccard distance is usually used for binary fingerprints. "
                f"Descriptor='{descriptor}' may not be binary."
            )
        if dist_type == "soap_kernel" and descriptor != "soap":
            logger.warning(
                "SOAP kernel is designed for SOAP descriptors. "
                f"Descriptor='{descriptor}' may not be compatible."
            )
        if dist_type == "hamming" and descriptor not in {"morgan", "selfies_onehot"}:
            logger.warning(
                "Hamming distance is usually used for binary fingerprints. "
                f"Descriptor='{descriptor}' may not be binary."
            )

        series = _series_for(descriptor)
        return self.distance_engine.get_matrix(
            series,
            metric=dist_type,
            filename=f"dist_{descriptor}_{dist_type}.npy"
        )

    def run_stress_test(
        self,
        num_molecules: int = 10,
        perturbations: int = 20,
        include_base: bool = True,
        max_bond_rattle : float = 0.05,
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

        if os.path.exists(target_path) and mol_ids is None and not include_base and not max_bond_rattle > 0:
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
            max_bond_rattle=max_bond_rattle
        )

    def get_positions(
        self,
        invariant: bool = True,
        output_filename: str = "qm9_subset.xyz",
        subset_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Atoms]:
        """
        Export QM9 molecules to a single .xyz file.
        Uses extxyz format so atom-level arrays (mass, partial charge) are preserved.
        """
        if self.df.is_empty():
            self.load()

        target_size = subset_size if subset_size is not None else self.subset_size
        if target_size <= 0:
            raise ValueError("subset_size must be a positive integer.")

        sample_df = self.df.head(min(target_size, self.df.height))
        if sample_df.is_empty():
            raise ValueError("No molecules available to export.")

        if seed is None:
            seed = self.embed_seed
        elif seed != self.embed_seed:
            logger.warning(
                f"get_positions(seed={seed}) differs from dataset embed_seed={self.embed_seed}; "
                "this may change which molecules can be embedded."
            )

        frames: List[Atoms] = []
        failed_count = 0
        failed_ids: List[str] = []

        for row in sample_df.iter_rows(named=True):
            mol_id = row["mol_id"]
            smiles = row["smiles"]
            canonical_smiles = row["canonical_smiles"]
            formula = row["formula"]
            try:
                mol = self._embed_molecule(
                    smiles=smiles,
                    seed=seed,
                    invariant=invariant,
                )
                if mol is None:
                    raise ValueError("Embedding failed.")

                conf = mol.GetConformer()

                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                positions = conf.GetPositions()
                charges = np.array(
                    [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
                    dtype=np.float64,
                )
                heavy_atom_count = int(mol.GetNumHeavyAtoms())
                element_counts = {"C": 0, "N": 0, "O": 0, "F": 0}
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in element_counts:
                        element_counts[symbol] += 1
                heavy_atom_denom = float(heavy_atom_count) if heavy_atom_count > 0 else 1.0

                atoms = Atoms(symbols=symbols, positions=positions)
                masses = atoms.get_masses()
                atoms.set_initial_charges(charges)
                # Keep explicit arrays for tools expecting named per-atom properties.
                atoms.arrays["partial_charge"] = charges
                atoms.arrays["mass"] = masses
                atoms.info.update(
                    {
                        "mol_id": mol_id,
                        "smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        "formula": formula,
                        "structure_type": self._classify_structure_type(mol),
                        "functional_groups": ",".join(self._detect_functional_groups(mol)),
                        "heavy_atom_count": heavy_atom_count,
                        "element_count_C": int(element_counts["C"]),
                        "element_count_N": int(element_counts["N"]),
                        "element_count_O": int(element_counts["O"]),
                        "element_count_F": int(element_counts["F"]),
                        "element_ratio_C": float(element_counts["C"] / heavy_atom_denom),
                        "element_ratio_N": float(element_counts["N"] / heavy_atom_denom),
                        "element_ratio_O": float(element_counts["O"] / heavy_atom_denom),
                        "element_ratio_F": float(element_counts["F"] / heavy_atom_denom),
                        "mu": float(row["mu"]) if row.get("mu") is not None else None,
                        "gap": float(row["gap"]) if row.get("gap") is not None else None,
                        "cv": float(row["cv"]) if row.get("cv") is not None else None,
                        "u0": float(row["u0"]) if row.get("u0") is not None else None,
                        "homo": float(row["homo"]) if row.get("homo") is not None else None,
                        "lumo": float(row["lumo"]) if row.get("lumo") is not None else None,
                        "num_atoms": int(len(atoms)),
                        "total_mass": float(np.sum(masses)),
                        "mean_partial_charge": float(np.mean(charges)),
                        "branching_index": int(row["branching_index"]),
                        "num_sp_carbons": int(row["num_sp_carbons"]),
                        "num_sp2_carbons": int(row["num_sp2_carbons"]),
                        "num_sp3_carbons": int(row["num_sp3_carbons"]),
                        "main_chain_length": int(row["main_chain_length"]),
                        "raw_token_count": int(row["raw_token_count"]),
                        "avg_bond_length": float(row["avg_bond_length"]),
                    }
                )
                frames.append(atoms)
                
            except Exception as e:
                logger.debug(f"Skipping {mol_id}: {e}")
                failed_count += 1
                failed_ids.append(mol_id)

        if not frames:
            raise ValueError("Failed to generate geometries for all selected molecules.")
        if failed_count > 0 or len(frames) != sample_df.height:
            raise ValueError(
                "Failed to generate geometries for all selected molecules: "
                f"requested={sample_df.height}, generated={len(frames)}, failed={failed_count}. "
                f"failed_ids={failed_ids}"
            )

        output_path = os.path.join(self.root, output_filename)
        write(output_path, frames, format="extxyz")
        logger.success(
            f"Saved {len(frames)} molecules to {output_path} "
            f"(failed: {failed_count}, requested: {target_size})."
        )
        
        return frames

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
            self.df = self.df.with_columns(pl.Series("soap_embedding", soap_features, dtype=pl.List(pl.Float64)))
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
