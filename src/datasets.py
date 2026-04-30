import json
from mendeleev import element
import os
import polars as pl
import numpy as np
import pyarrow as pa
import selfies as sf

from ase.io import write
from typing import Optional, List, Dict, Any, Set, Sequence, Union
from torch_geometric.datasets import QM9
from ase import Atoms
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem,  Descriptors, rdMolDescriptors, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mp_api.client import MPRester
from loguru import logger
from tqdm import tqdm 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
from dscribe.descriptors import SOAP, ACSF
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from transformers import logging as tf_log

from utils.file_ops import ensure_directory, validate_columns
from src.features import MolecularFeaturizer
from src.geometry import GeometryPerturber
from src.distance import DistanceCalculator

tf_log.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["REPORT_TO"] = "none"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

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
    REQUIRED_COLUMNS = {
        "mol_id",
        "smiles",
        "canonical_smiles",
        "num_atoms",
        "selfies",
        "formula",
        "functional_groups",
        "avg_bond_length",
        "scaffold_smiles",
        "coordinates",
        "atomic_numbers",
    }
    DESCRIPTOR_ALIASES = {
        "morgan": "morgan",
        "morgan_fingerprint": "morgan",
        "fingerprint": "morgan",
        "transformer": "transformer",
        "selfies_transformer": "transformer",
        "selformer": "transformer",
        "onehot": "onehot",
        "one_hot": "onehot",
        "selfies_onehot": "onehot",
        "soap": "soap",
        "soap_embedding": "soap",
        "soap_matrix": "soap",
        "acsf": "acsf",
        "acsf_embedding": "acsf",
        "acsf_matrix": "acsf",
        "mace": "mace",
        "mace_embedding": "mace",
        "mace_matrix": "mace",
        "coulomb": "coulomb",
        "coulomb_matrix": "coulomb",
        "chemprop": "chemprop",
        "chemprop_embedding": "chemprop",
    }

    def __init__(
        self,
        root: str = "data/QM9",
        filename: str = "dataset_cleaned.parquet",
        limit: int = None,
        required_mol_ids: Optional[List[str]] = None,
        embed_seed: int = 40,
        sampling_strategy: str = "stratified",
        stratify_by: Optional[List[str]] = None,
        stratify_bins: int = 10,
        sampling_seed: int = 40,
        min_per_stratum: int = 1,
        sampling_buffer: float = 1.1,
        injected_molecules: Optional[Union[pl.DataFrame, Sequence[str]]] = None,
        descriptors: Optional[Sequence[str]] = None,
    ):
        self.root = root
        self.filename = filename
        self.file_path = os.path.join(root, filename)
        self.subset_size = limit
        self.required_mol_ids = required_mol_ids or []
        self.embed_seed = embed_seed
        self.sampling_strategy = sampling_strategy
        self.stratify_by = stratify_by or ["num_atoms", "gap"]
        self.stratify_bins = stratify_bins
        self.sampling_seed = sampling_seed
        self.min_per_stratum = min_per_stratum
        self.sampling_buffer = sampling_buffer
        self.injected_molecules = injected_molecules
        requested_descriptors = self._normalize_descriptors(descriptors)
        self.add_morgan_fingerprint_flag = False
        self.add_selfies_transformer_flag = False
        self.add_selfies_onehot_flag = False
        self.add_soap_embedding_flag = False
        self.add_acsf_embedding_flag = False
        self.add_mace_embedding_flag = False
        self.add_coulomb_matrix_flag = False
        self.add_chemprop_embedding_flag = False
        self._set_descriptor_flags(requested_descriptors)
        self.df = pl.DataFrame()
        self.scaler = StandardScaler()
        self.is_scaled = False
        self._electron_affinity_cache: Dict[int, float] = {}
        self._ionization_energy_cache: Dict[int, float] = {}
        
        ensure_directory(self.root)
        
        # Initialize Sub-Components
        self.geometry_engine = GeometryPerturber(save_path=os.path.join(root, "stress_test.xyz"))
        self.distance_engine = DistanceCalculator(cache_dir=root)

        RDLogger.DisableLog("rdApp.error")

    @classmethod
    def _normalize_descriptors(cls, descriptors: Optional[Sequence[str]]) -> List[str]:
        if descriptors is None:
            return []

        normalized: List[str] = []
        unknown: List[str] = []
        for descriptor in descriptors:
            key = str(descriptor).strip().lower()
            canonical = cls.DESCRIPTOR_ALIASES.get(key)
            if canonical is None:
                unknown.append(str(descriptor))
                continue
            if canonical not in normalized:
                normalized.append(canonical)

        if unknown:
            valid = ", ".join(sorted(set(cls.DESCRIPTOR_ALIASES.values())))
            raise ValueError(
                f"Unknown QM9 descriptor(s): {unknown}. "
                f"Supported descriptors: {valid}."
            )

        return normalized

    def _set_descriptor_flags(self, descriptors: Sequence[str]) -> None:
        requested = set(descriptors)
        self.add_morgan_fingerprint_flag = "morgan" in requested
        self.add_selfies_transformer_flag = "transformer" in requested
        self.add_selfies_onehot_flag = "onehot" in requested
        self.add_soap_embedding_flag = "soap" in requested
        self.add_acsf_embedding_flag = "acsf" in requested
        self.add_mace_embedding_flag = "mace" in requested
        self.add_coulomb_matrix_flag = "coulomb" in requested
        self.add_chemprop_embedding_flag = "chemprop" in requested

    def _add_requested_descriptors(self) -> bool:
        """Adds descriptor columns requested at init-time; returns True if dataframe schema changed."""
        if self.df.is_empty():
            return False

        before_cols = set(self.df.columns)
        logger.info(
            "Applying requested QM9 descriptors to sampled dataframe "
            f"(rows={self.df.height})."
        )

        if self.add_morgan_fingerprint_flag:
            self.add_morgan_fingerprints()
        if self.add_selfies_transformer_flag:
            self.add_selfies_transformer()
        if self.add_selfies_onehot_flag:
            self.add_selfies_onehot()
        if self.add_soap_embedding_flag:
            self.add_soap()
        if self.add_acsf_embedding_flag:
            self.add_acsf()
        if self.add_mace_embedding_flag:
            self.add_mace()
        if self.add_coulomb_matrix_flag:
            self.add_coulomb_matrix()
        if self.add_chemprop_embedding_flag:
            self.add_chemprop()

        after_cols = set(self.df.columns)
        added_cols = sorted(list(after_cols - before_cols))
        if added_cols:
            logger.info(f"Added descriptor column(s): {added_cols}")
        else:
            logger.info("No new descriptor columns added (already present or none requested).")
        return after_cols != before_cols

    def _upsert_descriptor_column(
        self,
        column_name: str,
        compute_series: callable,
    ) -> bool:
        """
        Adds a descriptor column if missing, or fills null/empty values for rows that
        were appended later (for example injected molecules).
        Returns True when the dataframe was updated.
        """
        if self.df.is_empty():
            return False

        if column_name not in self.df.columns:
            self.df = self.df.with_columns(compute_series(self.df).alias(column_name))
            return True

        indexed_df = self.df.with_row_index("_row_idx")
        
        if self.df.schema[column_name] == pl.List:
            missing_mask = pl.col(column_name).is_null() | (pl.col(column_name).list.len() == 0)
        else:
            missing_mask = pl.col(column_name).is_null()

        missing_rows = indexed_df.filter(missing_mask)
        if missing_rows.is_empty():
            return False

        updates = missing_rows.select("_row_idx").with_columns(
            compute_series(missing_rows).alias(column_name)
        )

        self.df = (
            indexed_df
            .join(updates, on="_row_idx", how="left", suffix="_new")
            .with_columns(
                pl.coalesce(
                    pl.col(f"{column_name}_new"),
                    pl.col(column_name),
                ).alias(column_name)
            )
            .drop("_row_idx", f"{column_name}_new")
        )
        return True

    def _upsert_descriptor_columns(
        self,
        column_names: Sequence[str],
        compute_columns: callable,
    ) -> bool:
        """
        Adds multiple descriptor columns together, or fills rows where any of the
        requested columns are missing/null/empty.
        """
        if self.df.is_empty():
            return False

        target_cols = list(column_names)
        missing_cols = [col for col in target_cols if col not in self.df.columns]

        if missing_cols:
            computed = compute_columns(self.df)
            self.df = self.df.with_columns([computed[col] for col in target_cols])
            return True

        indexed_df = self.df.with_row_count("_row_idx")
        missing_expr = pl.lit(False)
        for col in target_cols:
            missing_expr = missing_expr | pl.col(col).is_null() | (pl.col(col).list.len() == 0)

        missing_rows = indexed_df.filter(missing_expr)
        if missing_rows.is_empty():
            return False

        computed = compute_columns(missing_rows)
        updates = missing_rows.select("_row_idx").with_columns(
            [computed[col] for col in target_cols]
        )

        merged = indexed_df.join(updates, on="_row_idx", how="left", suffix="_new")
        coalesced = [
            pl.coalesce(pl.col(f"{col}_new"), pl.col(col)).alias(col)
            for col in target_cols
        ]
        drop_cols = ["_row_idx"] + [f"{col}_new" for col in target_cols]
        self.df = merged.with_columns(coalesced).drop(*drop_cols)
        return True

    def _select_qm9_indices(self, dataset: QM9) -> List[int]:
        """Selects QM9 indices using the configured sampling strategy."""
        if self.sampling_strategy not in {"stratified", "head", "random"}:
            raise ValueError(
                f"Unsupported sampling_strategy='{self.sampling_strategy}'. "
                "Use 'stratified', 'random', or 'head'."
            )

        if self.sampling_strategy == "head":
            buffer_size = int(self.subset_size * self.sampling_buffer)
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

        if self.sampling_strategy == "random":
            n = len(dataset)
            if n == 0:
                return []
            target_size = int(np.ceil(self.subset_size * self.sampling_buffer))
            slots = max(target_size - len(required_indices), 0)
            if slots <= 0:
                return sorted(required_indices)

            available = [i for i in range(n) if i not in required_indices]
            rng = np.random.default_rng(self.sampling_seed)
            if not available:
                return sorted(required_indices)
            take = min(slots, len(available))
            chosen = rng.choice(available, size=take, replace=False)
            selected = set(required_indices)
            selected.update(chosen.tolist())
            return sorted(selected)
            
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

        # 1. Extract values
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

        # 2. Assign values to bins
        binned: Dict[str, np.ndarray] = {}
        for key in self.stratify_by:
            if key == "num_atoms":
                # Discrete variable: Use exact integer as the bin
                binned[key] = values[key].astype(int)
            else:
                # Continuous variable: Classical equal-width binning
                series = values[key]
                val_min, val_max = series.min(), series.max()
                if self.stratify_bins <= 1 or val_min == val_max:
                    binned[key] = np.zeros_like(series, dtype=int)
                else:
                    edges = np.linspace(val_min, val_max, self.stratify_bins + 1)
                    edges[-1] += 1e-9  # Catch the absolute max value
                    binned[key] = np.digitize(series, edges, right=False)

        # 3. Form Strata intersections (e.g., bin 4 for num_atoms AND bin 2 for gap)
        strata: Dict[tuple, List[int]] = {}
        for i in range(n):
            if i in required_indices:
                continue
            key = tuple(int(binned[k][i]) for k in self.stratify_by)
            strata.setdefault(key, []).append(i)

        target_size = int(np.ceil(self.subset_size * self.sampling_buffer))
        slots = max(target_size - len(required_indices), 0)
        
        total_available = sum(len(v) for v in strata.values())
        if slots <= 0 or total_available == 0:
            return sorted(required_indices)

        # 4. Proportional Allocation
        chosen_indices = []
        rng = np.random.default_rng(self.sampling_seed)

        for _, indices in strata.items():
            # Proportion of this specific strata in the full dataset
            proportion = len(indices) / total_available
            take = int(np.round(slots * proportion))
            take = min(take, len(indices))
            
            if take > 0:
                chosen = rng.choice(indices, size=take, replace=False)
                chosen_indices.extend(chosen.tolist())

        # 5. Fill remaining slots due to rounding down
        remaining_slots = slots - len(chosen_indices)
        if remaining_slots > 0:
            chosen_set = set(chosen_indices)
            all_unchosen = [idx for indices in strata.values() for idx in indices if idx not in chosen_set]
            if all_unchosen:
                extra = rng.choice(
                    all_unchosen,
                    size=min(remaining_slots, len(all_unchosen)),
                    replace=False
                )
                chosen_indices.extend(extra.tolist())

        selected = set(required_indices)
        selected.update(chosen_indices)
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

    def _build_smiles_row(
        self,
        smiles: str,
        mol_id: str,
        *,
        is_injected: int,
        outlier_category: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not smiles:
            return None

        raw_mol = Chem.MolFromSmiles(smiles)
        if raw_mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(raw_mol, canonical=True)
        mol = self._embed_molecule(
            smiles=canonical_smiles,
            seed=self.embed_seed,
            invariant=True,
        )
        if mol is None:
            return None

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)

        if not scaffold_smiles:
            scaffold_smiles, generic_scaffold = "Acyclic", "Acyclic"
        else:
            generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            generic_scaffold = Chem.MolToSmiles(generic_mol, canonical=True)

        brics_fragments = list(BRICS.BRICSDecompose(mol))
        scaffold_tree_nodes = []
        root_scaffold = "Acyclic"

        if scaffold_smiles != "Acyclic":
            params = rdScaffoldNetwork.ScaffoldNetworkParams()
            network = rdScaffoldNetwork.CreateScaffoldNetwork([mol], params)
            scaffold_tree_nodes = list(network.nodes)
            if scaffold_tree_nodes:
                root_scaffold = scaffold_tree_nodes[-1]

        brics_str = ",".join(brics_fragments)
        scaffold_tree_str = ",".join(scaffold_tree_nodes)

        formula = CalcMolFormula(mol)

        structure_type = self._classify_structure_type(mol)
        if structure_type == "acyclic":
            struct_class = "Acyclic"
        elif structure_type == "aromatic":
            struct_class = "Aromatic"
        else:
            struct_class = "Aliphatic Ring"

        selfies_str = sf.encoder(canonical_smiles)
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
        coordination = float(np.mean([atom.GetDegree() for atom in mol.GetAtoms()])) if mol.GetNumAtoms() > 0 else 0.0
        electron_affinities = []
        ionization_energies = []
        for atom in mol.GetAtoms():
            atomic_num = int(atom.GetAtomicNum())
            if atomic_num not in self._electron_affinity_cache:
                try:
                    ea_val = getattr(element(atomic_num), "electron_affinity", 0.0) or 0.0
                    self._electron_affinity_cache[atomic_num] = float(ea_val)
                except Exception:
                    self._electron_affinity_cache[atomic_num] = 0.0
            if atomic_num not in self._ionization_energy_cache:
                try:
                    ion_list = getattr(element(atomic_num), "ionenergies", None)
                    first_ion = ion_list.get(1, 0.0) if isinstance(ion_list, dict) else 0.0
                    self._ionization_energy_cache[atomic_num] = float(first_ion or 0.0)
                except Exception:
                    self._ionization_energy_cache[atomic_num] = 0.0
            electron_affinities.append(self._electron_affinity_cache[atomic_num])
            ionization_energies.append(self._ionization_energy_cache[atomic_num])
        election_affinity = float(np.mean(electron_affinities)) if electron_affinities else 0.0
        ionization_energies_value = float(np.mean(ionization_energies)) if ionization_energies else 0.0

        row = {
            "mol_id": mol_id,
            "formula": formula,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "scaffold_smiles": scaffold_smiles,
            "generic_scaffold": generic_scaffold,
            "root_scaffold": root_scaffold,
            "brics_fragments": brics_str,
            "scaffold_tree_nodes": scaffold_tree_str,
            "selfies": selfies_str,
            "functional_groups": functional_groups_str,
            "structure_class": struct_class,
            "is_injected": int(is_injected),
            "outlier_category": outlier_category,

            # Physical Properties
            "mol_weight": int(Descriptors.MolWt(mol)),
            "logp": int(Descriptors.MolLogP(mol)),
            "tpsa": int(Descriptors.TPSA(mol)),
            "election_affinity": election_affinity,
            "ionization_energies": ionization_energies_value,

            # Structural/Complexity Descriptors
            "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "num_fluorine": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9})),
            "num_heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "num_atoms": mol.GetNumAtoms(),
            "coordination": coordination,

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
        }

        return row

    def _build_qm9_row(self, i: int, data) -> Optional[Dict[str, Any]]:
        smiles = getattr(data, "smiles", None)
        
        mol_dict = self._build_smiles_row(
            smiles=smiles,
            mol_id=f"qm9_{i}",
            is_injected=0,
            outlier_category=None,
        )
        if mol_dict is None:
            return None

        coordinates = getattr(data, "pos", None)
        atomic_numbers = getattr(data, "z", None)
        if coordinates is not None:
            mol_dict["coordinates"] = np.asarray(coordinates, dtype=np.float64).tolist()
        else:
            mol_dict["coordinates"] = None
        if atomic_numbers is not None:
            mol_dict["atomic_numbers"] = np.asarray(atomic_numbers, dtype=np.int64).tolist()
        else:
            mol_dict["atomic_numbers"] = None

        mol_dict.update(dict(zip(self.QM9_TARGETS, data.y.tolist()[0])))
        return mol_dict

    @staticmethod
    def _ensure_injection_columns(df: pl.DataFrame) -> pl.DataFrame:
        if "is_injected" in df.columns:
            df = df.with_columns(
                pl.col("is_injected").fill_null(0).cast(pl.Int64, strict=False)
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("is_injected"))

        if "outlier_category" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("outlier_category"))
        else:
            df = df.with_columns(pl.col("outlier_category").cast(pl.Utf8, strict=False))

        return df

    def _next_injected_mol_id(self) -> int:
        if self.df.is_empty() or "mol_id" not in self.df.columns:
            return 0

        injected_ids = (
            self.df
            .filter(pl.col("mol_id").cast(pl.Utf8).str.starts_with("injected_"))
            .select(
                pl.col("mol_id")
                .cast(pl.Utf8)
                .str.replace("injected_", "")
                .cast(pl.Int64, strict=False)
                .alias("idx")
            )
            .drop_nulls()
        )
        if injected_ids.is_empty():
            return 0
        return int(injected_ids["idx"].max()) + 1

    @staticmethod
    def _align_to_schema(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        aligned = df
        missing = [col for col in columns if col not in aligned.columns]
        if missing:
            aligned = aligned.with_columns(
                [pl.lit(None).alias(col) for col in missing]
            )
        return aligned.select(columns)

    def _apply_configured_injections(self) -> None:
        """Inject molecules provided at init-time into the current sampled dataframe."""
        if self.injected_molecules is None:
            return
        self.inject_outliers(self.injected_molecules)

    def _base_row_count(self) -> int:
        if self.df.is_empty() or "is_injected" not in self.df.columns:
            return self.df.height
        return self.df.filter(pl.col("is_injected") != 1).height

    def _finalize_loaded_dataframe(self) -> pl.DataFrame:
        """
        Keeps the requested number of base QM9 molecules and preserves all injected rows.
        """
        if self.df.is_empty() or "is_injected" not in self.df.columns:
            self.df = self.df.head(self.subset_size)
            return self.df

        base_df = self.df.filter(pl.col("is_injected") != 1).head(self.subset_size)
        injected_df = self.df.filter(pl.col("is_injected") == 1)
        self.df = pl.concat([base_df, injected_df], how="vertical_relaxed")
        return self.df

    def inject_outliers(
        self,
        outliers: Union[pl.DataFrame, Sequence[str]],
    ) -> pl.DataFrame:
        """
        Appends user-supplied outliers and tags them with `is_injected=1`.
        Accepts either:
        - a Polars dataframe with at least a `smiles` column and optionally `outlier_category`
        - a sequence of SMILES strings
        Call `load()` before injecting so there is a base dataframe to extend.
        """
        if self.df.is_empty():
            raise ValueError("Dataset is empty. Call `load()` before injecting outliers.")

        if isinstance(outliers, pl.DataFrame):
            if outliers.is_empty():
                self.df = self._ensure_injection_columns(self.df)
                return self.df
            if "smiles" not in outliers.columns:
                raise ValueError("Outlier dataframe must contain a `smiles` column.")
            normalized_outliers = outliers.with_columns(
                pl.col("smiles").cast(pl.Utf8, strict=False),
                (
                    pl.col("outlier_category").cast(pl.Utf8, strict=False)
                    if "outlier_category" in outliers.columns
                    else pl.lit(None).cast(pl.Utf8).alias("outlier_category")
                ),
            )
            outlier_records = normalized_outliers.to_dicts()
        else:
            if not outliers:
                self.df = self._ensure_injection_columns(self.df)
                return self.df
            outlier_records = [
                {"smiles": str(smiles), "outlier_category": None}
                for smiles in outliers
            ]

        self.df = self._ensure_injection_columns(self.df)
        if not outlier_records:
            return self.df

        next_idx = self._next_injected_mol_id()
        injected_rows: List[Dict[str, Any]] = []
        failed_smiles: List[str] = []

        for record in outlier_records:
            smiles = record["smiles"]
            mol_dict = self._build_smiles_row(
                smiles=str(smiles),
                mol_id=f"injected_{next_idx}",
                is_injected=1,
                outlier_category=record.get("outlier_category"),
                extra_fields=record,
            )
            if mol_dict is None:
                failed_smiles.append(str(smiles))
                continue
            injected_rows.append(mol_dict)
            next_idx += 1

        if not injected_rows:
            raise ValueError(
                "None of the provided SMILES could be parsed and embedded for injection."
            )

        injected_df = pl.DataFrame(injected_rows)
        injected_df = self._align_to_schema(injected_df, self.df.columns)
        self.df = pl.concat([self.df, injected_df], how="vertical_relaxed")

        # Ensure any descriptors that were already requested are populated for new rows.
        self._add_requested_descriptors()

        if failed_smiles:
            logger.warning(
                "Some injected SMILES were skipped because parsing or 3D embedding failed: "
                f"{failed_smiles}"
            )

        logger.info(
            "Injected custom outliers into QM9 dataframe: "
            f"requested={len(outlier_records)}, injected={len(injected_rows)}, total_rows={self.df.height}."
        )
        return self.df

    def load(self, force_process: bool = False) -> pl.DataFrame:
        """
        Loads full QM9 cache (or processes it), then applies sampling.
        Requested descriptors are added after sampling.
        """
        if os.path.exists(self.file_path) and not force_process:
            logger.info(f"Loading cached full QM9 dataset from: {self.file_path}")
            try:
                full_df = pl.read_parquet(self.file_path)
                validate_columns(full_df, self.REQUIRED_COLUMNS)
            except Exception as e:
                logger.error(f"Could not read/validate cached full QM9 parquet ({e}). Rebuilding cache.")
                full_df = self._process_raw_qm9()
        else:
            full_df = self._process_raw_qm9()

        needs_non_null_descriptor_filter = self._requires_non_null_descriptor_rows()
        if not needs_non_null_descriptor_filter:
            self.df = self._sample_qm9_df(full_df, self.subset_size)
            self.df = self._ensure_injection_columns(self.df)
            self._apply_configured_injections()
            self._add_requested_descriptors()
            return self.df

        candidate_target = int(
            min(
                full_df.height,
                max(
                    self.subset_size,
                    np.ceil(self.subset_size * self.sampling_buffer),
                ),
            )
        )
        attempt = 0
        while True:
            attempt += 1
            self.df = self._sample_qm9_df(full_df, candidate_target)
            self.df = self._ensure_injection_columns(self.df)
            self._apply_configured_injections()
            self._add_requested_descriptors()
            self._drop_rows_with_null_required_descriptors()

            base_count = self._base_row_count()
            if base_count >= self.subset_size:
                self._finalize_loaded_dataframe()
                logger.info(
                    "QM9 descriptor null-filtering complete: "
                    f"attempts={attempt}, requested_limit={self.subset_size}, "
                    f"returned_rows={self.df.height}, base_rows={base_count}."
                )
                return self.df

            if candidate_target >= full_df.height:
                self._finalize_loaded_dataframe()
                logger.warning(
                    "Unable to reach requested QM9 limit after filtering null descriptor rows. "
                    f"requested_limit={self.subset_size}, returned_rows={self.df.height}, "
                    f"base_rows={base_count}, "
                    f"descriptor_cols={self._required_descriptor_columns()}."
                )
                return self.df

            candidate_target = int(
                min(
                    full_df.height,
                    max(candidate_target + 1, np.ceil(candidate_target * 1.5)),
                )
            )
            logger.info(
                "QM9 descriptor filtering needs more candidates; resampling with larger pool "
                f"(attempt={attempt}, next_candidate_target={candidate_target})."
            )

    def _required_descriptor_columns(self) -> List[str]:
        cols: List[str] = []
        if self.add_soap_embedding_flag:
            cols.append("soap_embedding")
        if self.add_acsf_embedding_flag:
            cols.append("acsf_embedding")
        if self.add_mace_embedding_flag:
            cols.append("mace_embedding")
        return cols

    def _requires_non_null_descriptor_rows(self) -> bool:
        return len(self._required_descriptor_columns()) > 0

    def _drop_rows_with_null_required_descriptors(self) -> None:
        required_cols = [c for c in self._required_descriptor_columns() if c in self.df.columns]
        if not required_cols:
            return

        before = self.df.height
        mask = pl.lit(True)
        for col_name in required_cols:
            mask = mask & pl.col(col_name).is_not_null() & (pl.col(col_name).list.len() > 0)

        self.df = self.df.filter(mask)
        dropped = before - self.df.height
        if dropped > 0:
            logger.info(
                "Dropped QM9 rows with null/empty descriptor vectors: "
                f"dropped={dropped}, remaining={self.df.height}, descriptor_cols={required_cols}."
            )

    def _process_raw_qm9(self) -> pl.DataFrame:
        """Downloads and cleans ALL raw QM9 data from Torch Geometric, then caches to parquet."""
        logger.info("Building full QM9 master parquet from raw Torch Geometric data.")
        try:
            dataset = QM9(root=self.root)
        except Exception as e:
            raise RuntimeError(f"QM9 download failed: {e}")

        data_list = []
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing QM9"):
            mol_dict = self._build_qm9_row(i, data)
            if mol_dict is not None:
                data_list.append(mol_dict)

        full_df = pl.DataFrame(data_list)
        dropped = len(dataset) - full_df.height
        if dropped > 0:
            logger.warning(
                "Filtered out QM9 molecules during load because 3D embedding failed "
                f"(dropped={dropped}, kept={full_df.height})."
            )
        full_df = (
            full_df
            .with_columns(
                pl.col("mol_id")
                .str.replace("qm9_", "")
                .cast(pl.Int64, strict=False)
                .alias("_qm9_idx")
            )
            .sort("_qm9_idx")
            .drop("_qm9_idx")
        )
        full_df.write_parquet(self.file_path)
        logger.success(
            f"Saved full QM9 master parquet: rows={full_df.height}, path={self.file_path}"
        )
        return full_df

    def _sample_qm9_df(self, full_df: pl.DataFrame, target_size: Optional[int]) -> pl.DataFrame:
        """
        Samples from already processed full QM9 dataframe.
        Required mol_ids are always included if available.
        """
        if full_df.is_empty():
            return full_df

        if target_size is None or target_size >= full_df.height:
            sampled_df = full_df
        elif target_size <= 0:
            sampled_df = full_df.clear()
        else:
            required_set: Set[str] = set(self.required_mol_ids)
            required_df = (
                full_df.filter(pl.col("mol_id").is_in(list(required_set)))
                if required_set else pl.DataFrame(schema=full_df.schema)
            )

            if required_set:
                present_required = set(required_df["mol_id"].to_list())
                missing_required = sorted(required_set - present_required)
                if missing_required:
                    logger.warning(
                        f"Requested required mol_id(s) not found in full QM9 dataset: {missing_required}"
                    )

            slots = max(target_size - required_df.height, 0)
            non_required = full_df.filter(~pl.col("mol_id").is_in(list(required_set)))

            if slots <= 0:
                sampled_non_required = pl.DataFrame(schema=full_df.schema)
            elif self.sampling_strategy == "head":
                sampled_non_required = non_required.head(slots)
            elif self.sampling_strategy == "random":
                sampled_non_required = non_required.sample(
                    n=min(slots, non_required.height),
                    seed=self.sampling_seed,
                    shuffle=True,
                )
            elif self.sampling_strategy == "stratified":
                sampled_non_required = self._stratified_sample_qm9_df(
                    non_required,
                    target_size=slots,
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            if required_df.height > 0:
                sampled_df = pl.concat([required_df, sampled_non_required], how="vertical_relaxed")
            else:
                sampled_df = sampled_non_required

        sampled_df = (
            sampled_df
            .with_columns(
                pl.col("mol_id")
                .str.replace("qm9_", "")
                .cast(pl.Int64, strict=False)
                .alias("_qm9_idx")
            )
            .sort("_qm9_idx")
            .drop("_qm9_idx")
        )
        logger.info(
            "QM9 sampling complete: "
            f"strategy={self.sampling_strategy}, requested_limit={target_size}, returned_rows={sampled_df.height}."
        )
        return sampled_df

    def _stratified_sample_qm9_df(self, df: pl.DataFrame, target_size: int) -> pl.DataFrame:
        """Classical equal-width stratified sampling directly on a QM9 Polars DataFrame."""
        if target_size <= 0:
            return df.clear()
        if target_size >= len(df):
            return df

        valid_stratify_keys = set(self.QM9_TARGETS) | {"num_atoms"}
        invalid = [k for k in self.stratify_by if k not in valid_stratify_keys]
        if invalid:
            raise ValueError(
                f"Invalid stratify_by key(s): {invalid}. "
                f"Valid keys: {sorted(valid_stratify_keys)}"
            )

        rng = np.random.default_rng(self.sampling_seed)
        values_by_key: Dict[str, np.ndarray] = {}
        for key in self.stratify_by:
            if key not in df.columns:
                logger.warning(
                    f"Stratification key '{key}' missing in QM9 dataframe. Falling back to random sampling."
                )
                return df.sample(n=target_size, seed=self.sampling_seed)
            values_by_key[key] = df[key].cast(pl.Float64).fill_null(float("nan")).to_numpy()

        valid_mask = np.ones(len(df), dtype=bool)
        for values in values_by_key.values():
            valid_mask &= ~np.isnan(values)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            logger.warning("All stratification values are NaN for sampled candidates. Falling back to random sampling.")
            return df.sample(n=target_size, seed=self.sampling_seed)

        binned: Dict[str, np.ndarray] = {}
        for key, values in values_by_key.items():
            valid_values = values[valid_indices]
            if key == "num_atoms":
                binned[key] = valid_values.astype(int)
                continue

            val_min, val_max = valid_values.min(), valid_values.max()
            if self.stratify_bins <= 1 or val_min == val_max:
                binned[key] = np.zeros_like(valid_values, dtype=int)
            else:
                edges = np.linspace(val_min, val_max, self.stratify_bins + 1)
                edges[-1] += 1e-9
                binned[key] = np.digitize(valid_values, edges, right=False)

        strata: Dict[tuple, List[int]] = {}
        for i, idx in enumerate(valid_indices):
            key = tuple(int(binned[k][i]) for k in self.stratify_by)
            strata.setdefault(key, []).append(idx)

        chosen_indices: List[int] = []
        total_valid = len(valid_indices)
        for indices in strata.values():
            proportion = len(indices) / total_valid
            take = int(np.round(target_size * proportion))
            take = min(take, len(indices))
            if take > 0:
                chosen = rng.choice(indices, size=take, replace=False)
                chosen_indices.extend(chosen.tolist())

        remaining_slots = target_size - len(chosen_indices)
        if remaining_slots > 0:
            remaining = list(set(valid_indices) - set(chosen_indices))
            if remaining:
                extra = rng.choice(
                    remaining,
                    size=min(remaining_slots, len(remaining)),
                    replace=False,
                )
                chosen_indices.extend(extra.tolist())

        return df[chosen_indices]

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

            # Primary Attempt: Embed the molecule in 3D space using ETKDG
            params = AllChem.ETKDG()
            params.randomSeed = seed
            
            if AllChem.EmbedMolecule(mol, params) == -1:
                # Fallback 1: Use random coordinates to jumpstart the algorithm
                params.useRandomCoords = True
                if AllChem.EmbedMolecule(mol, params) == -1:
                    # Fallback 2: Disable chemical rules (crucial for highly strained QM9 rings)
                    params.useBasicKnowledge = False
                    params.useExpTorsionAnglePrefs = False
                    if AllChem.EmbedMolecule(mol, params) == -1:
                        # If it STILL fails, it's truly un-embeddable
                        logger.warning(f"Failed to embed molecule {smiles}")
                        return None

            return mol
            
        except Exception as e:
            logger.debug(f"Molecule embedding failed for SMILES '{smiles}': {e}")
            return None
    def add_morgan_fingerprints(self, radius: int = 3, fp_size: int = 2048) -> None:
        self._upsert_descriptor_column(
            "morgan_fingerprint",
            lambda frame: MolecularFeaturizer.compute_morgan_fingerprints(
                frame["canonical_smiles"], radius, fp_size
            ),
        )

    def add_selfies_transformer(self, model_name: str = "HUBioDataLab/SELFormer") -> None:
        self._upsert_descriptor_column(
            "selfies_transformer",
            lambda frame: MolecularFeaturizer.compute_selfies_transformer(
                frame["selfies"], model_name
            ),
        )

    def add_selfies_onehot(self, flatten: bool = True) -> None:
        self._upsert_descriptor_column(
            "selfies_onehot",
            lambda frame: MolecularFeaturizer.compute_selfies_onehot(
                frame["selfies"],
                flatten=flatten
            ),
        )

    def add_soap(self, r_cut=6.0, n_max=8, l_max=6, sigma=0.5) -> None:
        """Adds pooled SOAP embeddings and atom-wise SOAP matrices to the dataframe."""

        def _compute(frame: pl.DataFrame) -> Dict[str, pl.Series]:
            pooled, matrix = MolecularFeaturizer.compute_soap_outputs(
                frame["canonical_smiles"],
                coordinates_series=frame["coordinates"] if "coordinates" in frame.columns else None,
                atomic_numbers_series=frame["atomic_numbers"] if "atomic_numbers" in frame.columns else None,
                r_cut=r_cut,
                n_max=n_max,
                l_max=l_max,
                sigma=sigma,
            )
            return {"soap_embedding": pooled, "soap_matrix": matrix}

        updated = self._upsert_descriptor_columns(
            ["soap_embedding", "soap_matrix"],
            _compute,
        )
        if updated:
            logger.success("Added SOAP embeddings and matrices.")

    def add_acsf(self, r_cut=6.0) -> None:
        """Adds pooled ACSF embeddings and atom-wise ACSF matrices to the dataframe."""

        def _compute(frame: pl.DataFrame) -> Dict[str, pl.Series]:
            pooled, matrix = MolecularFeaturizer.compute_acsf_outputs(
                frame["canonical_smiles"],
                coordinates_series=frame["coordinates"] if "coordinates" in frame.columns else None,
                atomic_numbers_series=frame["atomic_numbers"] if "atomic_numbers" in frame.columns else None,
                r_cut=r_cut,
            )
            return {"acsf_embedding": pooled, "acsf_matrix": matrix}

        updated = self._upsert_descriptor_columns(
            ["acsf_embedding", "acsf_matrix"],
            _compute,
        )
        if updated:
            logger.success("Added ACSF embeddings and matrices.")

    def add_mace(
        self,
        model: str = "medium",
        batch_size: int = 32,
    ) -> None:
        """Adds pooled MACE embeddings and atom-wise MACE matrices to the dataframe."""

        def _compute(frame: pl.DataFrame) -> Dict[str, pl.Series]:
            pooled, matrix = MolecularFeaturizer.compute_mace_outputs(
                frame["canonical_smiles"],
                coordinates_series=frame["coordinates"] if "coordinates" in frame.columns else None,
                atomic_numbers_series=frame["atomic_numbers"] if "atomic_numbers" in frame.columns else None,
                model=model,
                batch_size=batch_size,
            )
            return {"mace_embedding": pooled, "mace_matrix": matrix}

        updated = self._upsert_descriptor_columns(
            ["mace_embedding", "mace_matrix"],
            _compute,
        )
        if updated:
            logger.success("Added MACE embeddings and matrices.")

    def add_coulomb_matrix(
        self,
        n_atoms_max: int | None = None,
        permutation: str = "sorted_l2"
    ) -> None:
        """Adds Coulomb matrix descriptors to the dataframe."""
        updated = self._upsert_descriptor_column(
            "coulomb_matrix",
            lambda frame: MolecularFeaturizer.compute_coulomb_matrix(
                frame["canonical_smiles"],
                coordinates_series=frame["coordinates"] if "coordinates" in frame.columns else None,
                atomic_numbers_series=frame["atomic_numbers"] if "atomic_numbers" in frame.columns else None,
                n_atoms_max=n_atoms_max,
                permutation=permutation
            ),
        )
        if updated:
            logger.success("Added Coulomb matrix descriptors.")

    def add_chemprop(
        self,
    ) -> None:
        self._upsert_descriptor_column(
            "chemprop_embedding",
            lambda frame: MolecularFeaturizer.compute_chemprop_embeddings(
                frame["canonical_smiles"],
            ),
        )

    @staticmethod
    def _series_to_feature_matrices(series: pl.Series) -> List[np.ndarray]:
        matrices: List[np.ndarray] = []
        for value in series.to_list():
            if value is None:
                matrices.append(np.empty((0, 0), dtype=np.float64))
                continue

            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            matrices.append(arr)
        return matrices

    def get_descriptor_matrices(
        self,
        descriptor: str,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Returns per-molecule descriptor matrices with shape (n_atoms, d) when supported.
        If the cached matrix column is missing, the corresponding descriptor pair is
        computed and attached to the dataframe first.
        """
        desc = self.DESCRIPTOR_ALIASES.get(descriptor.strip().lower())
        if desc is None:
            raise ValueError(f"Unknown descriptor '{descriptor}'.")

        if self.df.is_empty():
            raise ValueError("Dataset is empty. Call `load()` before requesting descriptor matrices.")

        if desc == "soap":
            if "soap_matrix" not in self.df.columns:
                self.add_soap(**kwargs)
            series = self.df["soap_matrix"]
        elif desc == "acsf":
            if "acsf_matrix" not in self.df.columns:
                self.add_acsf(**kwargs)
            series = self.df["acsf_matrix"]
        elif desc == "mace":
            if "mace_matrix" not in self.df.columns:
                self.add_mace(**kwargs)
            series = self.df["mace_matrix"]
        else:
            raise ValueError(
                f"Descriptor '{descriptor}' does not have a matrix-form API. "
                "Supported matrix descriptors: soap, acsf, mace."
            )

        return self._series_to_feature_matrices(series)

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
        self.add_mace()
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
    

    def get_distance_matrix(self, descriptor: str = "morgan", dist_type: str = "jaccard", pca_components=None, force_calculate=False) -> np.ndarray:
        """
        Computes a distance matrix for a chosen descriptor.
        
        Args:
            pca_components: If provided, applies PCA to reduce the descriptor
                to the requested number of dimensions before calculating
                distances.
        """

        descriptor = descriptor.lower()
        aliases = {
            "transformer": "selfies_transformer",
            "onehot": "selfies_onehot",
        }
        if dist_type == 'tanimoto':
            dist_type = 'jaccard'
            
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
            if desc == "mace":
                self.add_mace()
                return self.df["mace_embedding"]
            if desc == "coulomb_matrix":
                self.add_coulomb_matrix()
                return self.df["coulomb_matrix"]
            if desc == "chemprop":
                self.add_chemprop()
                return self.df["chemprop_embedding"]
            raise ValueError(
                f"Unknown descriptor: {desc}. "
                "Expected one of: morgan, selfies_transformer, selfies_onehot, "
                "soap, acsf, mace, coulomb_matrix, chemprop."
            )

        # Distance type warnings
        if dist_type in {"jaccard", "hamming"} and descriptor not in {"morgan", "selfies_onehot"}:
            logger.warning(f"{dist_type.capitalize()} distance is usually for binary fingerprints. '{descriptor}' may not be binary.")
        if dist_type == "soap_kernel" and descriptor != "soap":
            logger.warning(f"SOAP kernel is designed for SOAP. '{descriptor}' may not be compatible.")

        series = _series_for(descriptor)
        if pca_components is not None:
            if not isinstance(pca_components, int) or pca_components <= 0:
                raise ValueError("pca_components must be a positive integer.")

            # Convert series to 2D NumPy array
            X = np.array(series.to_list())
            
            # Only apply if we actually have enough dimensions to reduce
            if X.shape[1] > pca_components:
                logger.info(f"Applying PCA to reduce '{descriptor}' to {pca_components} dimensions.")
                pca = PCA(n_components=pca_components)
                X_reduced = pca.fit_transform(X)
                explained_variance = float(np.sum(pca.explained_variance_ratio_))
                logger.info(f"PCA kept {explained_variance * 100:.2f}% of the variance for '{descriptor}'.")
                
                logger.info(f"PCA reduced '{descriptor}' dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
                
                # Convert back to Polars Series of lists
                series = pl.Series(series.name, X_reduced.tolist())
                filename = f"dist_{descriptor}_{dist_type}_pca{pca_components}.npy"
            else:
                logger.warning(
                    f"Descriptor '{descriptor}' has {X.shape[1]} dimensions, "
                    f"which is not greater than pca_components={pca_components}. "
                    "Skipping reduction."
                )
                filename = f"dist_{descriptor}_{dist_type}.npy"
        else:
            filename = f"dist_{descriptor}_{dist_type}.npy"

        logger.info(f"Calculating distance matrix for {descriptor} using {dist_type} distance.")
        return self.distance_engine.get_matrix(
            series,
            metric=dist_type,
            filename=filename, 
            force_calculate=force_calculate,
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

    def get_molecules(
            self,
            invariant: bool = True,
            output_filename: str = "qm9_subset.xyz",
            subset_size: Optional[int] = None,
            seed: Optional[int] = None,
        ) -> List[Atoms]:
            """
            Export QM9 molecules to a single .xyz file.
            Uses native coordinates and atomic numbers extracted directly from the QM9 dataset
            instead of generating them from SMILES. Uses extxyz format so atom-level arrays
            (mass, partial charge) are preserved.
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
                    f"get_molecules(seed={seed}) differs from dataset embed_seed={self.embed_seed}; "
                    "this may change which molecules can be embedded if falling back to SMILES."
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
                    positions = row.get("coordinates")
                    atomic_numbers = row.get("atomic_numbers")

                    # Core update: Extract directly from QM9 if available
                    if positions is not None and atomic_numbers is not None:
                        positions = np.array(positions, dtype=np.float64)
                        atomic_numbers = np.array(atomic_numbers, dtype=np.int64)
                        
                        # Since we bypass RDKit embedding here, default charges to 0 
                        # (QM9 uses mulliken, but it's not strictly mapped here)
                        charges = np.zeros(len(atomic_numbers), dtype=np.float64)
                    
                    else:
                        # Fallback to SMILES embedding if QM9 coords are missing (e.g., injected outliers)
                        mol = self._embed_molecule(
                            smiles=smiles,
                            seed=seed,
                            invariant=invariant,
                        )
                        if mol is None:
                            raise ValueError("No QM9 coordinates available and 3D embedding failed.")

                        conf = mol.GetConformer()
                        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
                        positions = conf.GetPositions()
                        charges = np.array(
                            [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
                            dtype=np.float64,
                        )

                    # Initialize the ASE Atoms object
                    atoms = Atoms(numbers=atomic_numbers, positions=positions)
                    masses = atoms.get_masses()
                    symbols = atoms.get_chemical_symbols()
                    
                    atoms.set_initial_charges(charges)
                    # Keep explicit arrays for tools expecting named per-atom properties.
                    atoms.arrays["partial_charge"] = charges
                    atoms.arrays["mass"] = masses
                    
                    # Derive element counts and heavy atoms from the native array
                    heavy_atom_count = sum(1 for z in atomic_numbers if z > 1)
                    element_counts = {
                        "C": symbols.count("C"),
                        "N": symbols.count("N"),
                        "O": symbols.count("O"),
                        "F": symbols.count("F")
                    }
                    heavy_atom_denom = float(heavy_atom_count) if heavy_atom_count > 0 else 1.0

                    atom_info = {
                        "mol_id": mol_id,
                        "smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        
                        "scaffold": row.get("scaffold_smiles", ""),
                        "formula": formula,
                        
                        # Read structural topology features dynamically mapped in _build_smiles_row()
                        "structure_type": row.get("structure_class", "Unknown"),
                        "functional_groups": row.get("functional_groups", ""),
                        
                        "heavy_atom_count": heavy_atom_count,
                        "element_count_C": element_counts["C"],
                        "element_count_N": element_counts["N"],
                        "element_count_O": element_counts["O"],
                        "element_count_F": element_counts["F"],
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
                        "branching_index": int(row.get("branching_index", 0)),
                        "num_sp_carbons": int(row.get("num_sp_carbons", 0)),
                        "num_sp2_carbons": int(row.get("num_sp2_carbons", 0)),
                        "num_sp3_carbons": int(row.get("num_sp3_carbons", 0)),
                        "main_chain_length": int(row.get("main_chain_length", 0)),
                        "raw_token_count": int(row.get("raw_token_count", 0)),
                        "avg_bond_length": float(row.get("avg_bond_length", 0.0)),
                    }
                    
                    if "soap_embedding" in self.df.columns:
                        atom_info["soap"] = row.get("soap_embedding")
                    atoms.info.update(atom_info)
                    frames.append(atoms)
                    
                except Exception as e:
                    logger.debug(f"Skipping {mol_id}: {e}")
                    failed_count += 1
                    failed_ids.append(mol_id)

            if not frames:
                raise ValueError("Failed to generate geometries for all selected molecules.")

            output_path = os.path.join(self.root, output_filename)
            write(output_path, frames, format="extxyz")
            logger.success(
                f"Saved {len(frames)} molecules to {output_path} "
                f"(failed: {failed_count}, requested: {target_size})."
            )
            
            return frames
    
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
    DESCRIPTOR_ALIASES = {
        "soap": "soap",
        "soap_embedding": "soap",
        "acsf": "acsf",
        "acsf_embedding": "acsf",
        "mace": "mace",
        "mace_embedding": "mace",
        "coulomb": "coulomb",
        "coulomb_matrix": "coulomb",
    }

    def __init__(
        self,
        base_path: str = "data/Materials Project/",
        file_name: str = "materials.parquet",
        config_path: str = "config/api_key.json",
        limit: Optional[int] = 2000,
        sampling_strategy: str = "head",
        stratify_on: Optional[Union[str, Sequence[str]]] = None,
        stratify_bins: int = 10,
        sampling_seed: int = 40,
        min_per_bin: int = 1,
        injected_materials: Optional[Union[pl.DataFrame, Sequence[Dict[str, Any]]]] = None,
        descriptors: Optional[Sequence[str]] = None,
    ) -> None:
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.api_key = self._load_api_key(config_path)
        self.subset_size = limit
        self.sampling_strategy = sampling_strategy
        if stratify_on is None:
            self.stratify_on = ["band_gap", "energy_above_hull"]
        elif isinstance(stratify_on, str):
            self.stratify_on = [stratify_on]
        else:
            self.stratify_on = list(stratify_on)
        self.stratify_bins = stratify_bins
        self.sampling_seed = sampling_seed
        self.min_per_bin = min_per_bin
        self.sampling_buffer = 1.2
        self.injected_materials = injected_materials
        requested_descriptors = self._normalize_descriptors(descriptors)
        self.add_soap = False
        self.add_acsf = False
        self.add_mace = False
        self.add_coulomb = False
        self._set_descriptor_flags(requested_descriptors)
        self.df = pl.DataFrame()
        os.makedirs(self.base_path, exist_ok=True)

    @classmethod
    def _normalize_descriptors(cls, descriptors: Optional[Sequence[str]]) -> List[str]:
        if descriptors is None:
            return []

        normalized: List[str] = []
        unknown: List[str] = []
        for descriptor in descriptors:
            key = str(descriptor).strip().lower()
            canonical = cls.DESCRIPTOR_ALIASES.get(key)
            if canonical is None:
                unknown.append(str(descriptor))
                continue
            if canonical not in normalized:
                normalized.append(canonical)

        if unknown:
            valid = ", ".join(sorted(set(cls.DESCRIPTOR_ALIASES.values())))
            raise ValueError(
                f"Unknown Materials Project descriptor(s): {unknown}. "
                f"Supported descriptors: {valid}."
            )

        return normalized

    def _set_descriptor_flags(self, descriptors: Sequence[str]) -> None:
        requested = set(descriptors)
        self.add_soap = "soap" in requested
        self.add_acsf = "acsf" in requested
        self.add_mace = "mace" in requested
        self.add_coulomb = "coulomb" in requested

    def _load_api_key(self, path: str) -> Optional[str]:
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            return config.get("key")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load API key from {path}: {e}")
            return None

    def _required_descriptor_columns(self) -> List[str]:
            """Returns a list of descriptor columns that should be present and valid."""
            cols = []
            if getattr(self, "add_soap", False):
                cols.extend(["soap_embedding", "soap_matrix"])
            if getattr(self, "add_acsf", False): 
                cols.extend(["acsf_embedding", "acsf_matrix"])
            if getattr(self, "add_coulomb", False): 
                cols.append("coulomb_matrix")
            if getattr(self, "add_mace", False): 
                cols.extend(["mace_embedding", "mace_matrix"])
            return cols

    def _drop_rows_with_null_required_descriptors(self) -> None:
        """Filters out rows where any of the requested 3D descriptors failed (are null or empty)."""
        required_cols = [c for c in self._required_descriptor_columns() if c in self.df.columns]
        if not required_cols:
            return

        before = self.df.height
        mask = pl.lit(True)
        for col_name in required_cols:
            col_expr = pl.col(col_name)
            mask = mask & col_expr.is_not_null()
            
            # If the column is a Polars List type, ensure it's not an empty list
            if self.df[col_name].dtype == pl.List:
                mask = mask & (col_expr.list.len() > 0)
            
        self.df = self.df.filter(mask)
        dropped = before - self.df.height
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} materials due to failed descriptor calculations. "
                f"Remaining valid candidates: {self.df.height}"
            )

    def _sample_materials_df(self, full_df: pl.DataFrame, target_size: int) -> pl.DataFrame:
        """Helper to route the dataframe sampling."""
        if target_size <= 0:
            return full_df.clear()
        if target_size >= full_df.height:
            return full_df

        if self.sampling_strategy == "stratified":
            return self._stratified_sample_df(full_df, target_size=target_size)
        elif self.sampling_strategy == "head":
            return full_df.head(target_size)
        elif self.sampling_strategy == "random":
            return full_df.sample(n=target_size, seed=self.sampling_seed)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    @staticmethod
    def _ensure_injection_columns(df: pl.DataFrame) -> pl.DataFrame:
        if "is_injected" in df.columns:
            df = df.with_columns(
                pl.col("is_injected").fill_null(0).cast(pl.Int64, strict=False)
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("is_injected"))

        if "outlier_category" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("outlier_category"))
        else:
            df = df.with_columns(pl.col("outlier_category").cast(pl.Utf8, strict=False))

        return df

    @staticmethod
    def _align_to_schema(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        aligned = df
        missing = [col for col in columns if col not in aligned.columns]
        if missing:
            aligned = aligned.with_columns([pl.lit(None).alias(col) for col in missing])
        return aligned.select(columns)

    def _next_injected_material_id(self) -> int:
        if self.df.is_empty() or "material_id" not in self.df.columns:
            return 0

        injected_ids = (
            self.df
            .filter(pl.col("material_id").cast(pl.Utf8).str.starts_with("injected_"))
            .select(
                pl.col("material_id")
                .cast(pl.Utf8)
                .str.replace("injected_", "")
                .cast(pl.Int64, strict=False)
                .alias("idx")
            )
            .drop_nulls()
        )
        if injected_ids.is_empty():
            return 0
        return int(injected_ids["idx"].max()) + 1

    def _apply_configured_injections(self) -> None:
        if self.injected_materials is None:
            return
        self.inject_outliers(self.injected_materials)

    def _base_row_count(self) -> int:
        if self.df.is_empty() or "is_injected" not in self.df.columns:
            return self.df.height
        return self.df.filter(pl.col("is_injected") != 1).height

    def _finalize_loaded_dataframe(self, limit: Optional[int]) -> pl.DataFrame:
        if limit is None:
            return self.df

        if self.df.is_empty() or "is_injected" not in self.df.columns:
            self.df = self.df.head(limit)
            return self.df

        base_df = self.df.filter(pl.col("is_injected") != 1).head(limit)
        injected_df = self.df.filter(pl.col("is_injected") == 1)
        self.df = pl.concat([base_df, injected_df], how="vertical_relaxed")
        return self.df

    def inject_outliers(
        self,
        outliers: Union[pl.DataFrame, Sequence[Dict[str, Any]]],
    ) -> pl.DataFrame:
        """
        Appends user-supplied synthetic materials and tags them with `is_injected=1`.
        Accepts either a Polars dataframe matching the Materials Project schema
        or a sequence of dictionaries (for example from `process_synthetic_structure`).
        Call `load()` before injecting so there is a base dataframe to extend.
        """
        if self.df.is_empty():
            raise ValueError("Dataset is empty. Call `load()` before injecting outliers.")

        if isinstance(outliers, pl.DataFrame):
            if outliers.is_empty():
                self.df = self._ensure_injection_columns(self.df)
                return self.df
            normalized_outliers = outliers
        else:
            outlier_records = list(outliers)
            if not outlier_records:
                self.df = self._ensure_injection_columns(self.df)
                return self.df
            normalized_outliers = pl.DataFrame(outlier_records)

        self.df = self._ensure_injection_columns(self.df)
        normalized_outliers = self._ensure_injection_columns(normalized_outliers)

        if "material_id" not in normalized_outliers.columns:
            next_idx = self._next_injected_material_id()
            normalized_outliers = normalized_outliers.with_row_count("_inject_row").with_columns(
                (pl.col("_inject_row") + next_idx).cast(pl.Utf8).radd("injected_").alias("material_id")
            ).drop("_inject_row")

        normalized_outliers = normalized_outliers.with_columns(
            pl.col("material_id").cast(pl.Utf8, strict=False),
            pl.lit(1).cast(pl.Int64).alias("is_injected"),
            pl.col("outlier_category").cast(pl.Utf8, strict=False),
        )

        injected_df = self._align_to_schema(normalized_outliers, self.df.columns)
        self.df = pl.concat([self.df, injected_df], how="vertical_relaxed")

        if any(col in self.df.columns for col in self._required_descriptor_columns()):
            self._add_descriptors(attach_to_df=True)

        logger.info(
            "Injected custom outliers into Materials Project dataframe: "
            f"requested={injected_df.height}, total_rows={self.df.height}."
        )
        return self.df

    def load(self, force_fetch: bool = False, limit: Optional[int] = None) -> pl.DataFrame:
        """
        Loads the dataset, applies sampling, computes descriptors, and ensures 
        exactly 'limit' rows are returned by filtering out failed structures.
        """
        effective_limit = self.subset_size if limit is None else limit
        
        # 1. Load or fetch the raw full dataset
        if os.path.exists(self.file_path) and not force_fetch:
            logger.info(f"Loading full cached Parquet data from {self.file_path}...")
            full_df = pl.read_parquet(self.file_path)
        else:
            self._fetch_from_api(compute_descriptors=False)
            full_df = self.df
        full_df = self._ensure_injection_columns(full_df)
            
        needs_descriptors = len(self._required_descriptor_columns()) > 0
        
        # 2. Simple Path: No limit requested, or no descriptors to compute
        if effective_limit is None or not needs_descriptors:
            target = effective_limit if effective_limit is not None else full_df.height
            self.df = self._sample_materials_df(full_df, target)
            self.df = self._ensure_injection_columns(self.df)
            self._apply_configured_injections()
            if needs_descriptors:
                self._add_descriptors(attach_to_df=True)
                self._drop_rows_with_null_required_descriptors()
            self._finalize_loaded_dataframe(effective_limit)
            return self.df
            
        # 3. Robust Path: We need exactly `effective_limit` valid rows
        sampling_buffer = getattr(self, "sampling_buffer", 1.2)
        candidate_target = int(min(full_df.height, max(effective_limit, np.ceil(effective_limit * sampling_buffer))))
        
        attempt = 0
        while True:
            attempt += 1
            self.df = self._sample_materials_df(full_df, candidate_target)
            self.df = self._ensure_injection_columns(self.df)
            self._apply_configured_injections()
            
            # Compute descriptors on the candidate pool
            tag_parts = [f"sample_n{candidate_target}", f"seed{self.sampling_seed}"]
            if self.sampling_strategy: 
                tag_parts.append(self.sampling_strategy)
            output_tag = "_".join(tag_parts)
            
            self._add_descriptors(output_tag=output_tag, attach_to_df=True)
            
            # Dynamically drop rows that failed (e.g. Coulomb matrix returned None)
            self._drop_rows_with_null_required_descriptors()
            
            # Did we end up with enough valid rows?
            if self._base_row_count() >= effective_limit:
                self._finalize_loaded_dataframe(effective_limit)
                logger.success(f"Successfully reached requested limit of {effective_limit} valid rows (Attempt {attempt}).")
                return self.df
                
            # If we've processed everything and still don't have enough, return what we have
            if candidate_target >= full_df.height:
                self._finalize_loaded_dataframe(effective_limit)
                logger.warning(
                    f"Exhausted the full dataset. Could only find {self._base_row_count()} valid base rows "
                    f"(requested {effective_limit})."
                )
                return self.df
                
            # Scale up the pool and try again
            candidate_target = int(min(full_df.height, max(candidate_target + 1, np.ceil(candidate_target * 1.5))))
            logger.info(f"Attempt {attempt} fell short due to failed structures. Resampling with larger pool: target={candidate_target}.")

    def _fetch_from_api(self, compute_descriptors: bool = True) -> None:
        """
        Fetches ALL matching materials, optionally computes descriptors, and caches to disk.
        Does not perform any sampling here.
        """
        logger.info("Fetching all materials from API to build master dataset.")

        if not self.api_key:
            raise ValueError("API Key not found.")

        with MPRester(self.api_key) as mpr:
            query_kwargs = dict(
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "symmetry", "energy_per_atom", "formation_energy_per_atom", 
                    "density", "band_gap", "is_metal",
                    "energy_above_hull",
                ],
            )

            # Fetch all docs
            docs = list(
                mpr.materials.summary.search(
                    **query_kwargs,
                    chunk_size=1000,
                )
            )
            logger.info(f"Fetched {len(docs)} total materials.")

        # Process everything
        data_list = [
            self._process_doc(d)
            for d in tqdm(docs, desc="Processing materials")
        ]

        self.df = pl.DataFrame(data_list)
        
        # Compute descriptors for the whole dataset (optional)
        if compute_descriptors and (self.add_soap or self.add_acsf):
            self._add_descriptors()

        # Save the full master dataset
        self.df.write_parquet(self.file_path)
        logger.success(f"Full master dataset saved with {len(self.df)} entries.")

    def _stratified_sample_df(self, df: pl.DataFrame, target_size: int) -> pl.DataFrame:
        """
        Pure classical stratified sampling based on equal-width binning,
        now operating directly on a Polars DataFrame.
        """
        if target_size <= 0:
            return df.clear()
        if target_size >= len(df):
            return df

        rng = np.random.default_rng(self.sampling_seed)
        values_by_key = {}

        # Safely extract target features dynamically from DataFrame
        for key in self.stratify_on:
            if key not in df.columns:
                logger.warning(f"Attribute '{key}' not found in DataFrame. Falling back to random sampling.")
                return df.sample(n=target_size, seed=self.sampling_seed)
            
            # Extract to numpy, treating nulls as nan
            values_by_key[key] = df[key].cast(pl.Float64).fill_null(float('nan')).to_numpy()

        # Filter out NaNs for stratification purposes
        valid_mask = np.ones(len(df), dtype=bool)
        for values in values_by_key.values():
            valid_mask &= ~np.isnan(values)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logger.warning(f"All values for {self.stratify_on} are NaN. Falling back to random sampling.")
            return df.sample(n=target_size, seed=self.sampling_seed)

        # Classical equal-width binning per feature
        binned = {}
        for key, values in values_by_key.items():
            valid_values = values[valid_indices]
            val_min, val_max = valid_values.min(), valid_values.max()

            if self.stratify_bins <= 1 or val_min == val_max:
                binned[key] = np.zeros_like(valid_values, dtype=int)
                continue

            edges = np.linspace(val_min, val_max, self.stratify_bins + 1)
            edges[-1] += 1e-9  
            binned[key] = np.digitize(valid_values, edges, right=False)

        strata = {}
        for i, idx in enumerate(valid_indices):
            key = tuple(int(binned[k][i]) for k in self.stratify_on)
            strata.setdefault(key, []).append(idx)

        chosen_indices = []
        total_valid = len(valid_indices)

        # Sample proportionally based on the bin's size in the original population
        for indices in strata.values():
            proportion = len(indices) / total_valid
            take = int(np.round(target_size * proportion))
            take = min(take, len(indices))
            
            if take > 0:
                chosen = rng.choice(indices, size=take, replace=False)
                chosen_indices.extend(chosen.tolist())

        # Fill any remaining slots due to rounding or small bins
        remaining_slots = target_size - len(chosen_indices)
        if remaining_slots > 0:
            remaining = list(set(valid_indices) - set(chosen_indices))
            if remaining:
                extra = rng.choice(
                    remaining,
                    size=min(remaining_slots, len(remaining)),
                    replace=False
                )
                chosen_indices.extend(extra.tolist())

        # Select rows from polars DataFrame using integer indexing
        return df[chosen_indices]

    @staticmethod
    def _compute_bond_length_stats(
        struct: Structure,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Approximate average and maximum bond length (Angstrom) from a periodic structure.
        Uses the nearest-neighbor distance per site with an adaptive cutoff.
        """
        if struct is None or len(struct) == 0:
            return None, None

        cutoffs = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        neighbors = None
        for r in cutoffs:
            try:
                all_neighbors = struct.get_all_neighbors(r)
            except Exception:
                continue
            if any(len(nlist) > 0 for nlist in all_neighbors):
                neighbors = all_neighbors
                break

        if neighbors is None:
            return None, None

        nearest_distances = []
        for nlist in neighbors:
            if not nlist:
                continue
            distances = []
            for nn in nlist:
                dist = getattr(nn, "nn_distance", None)
                if dist is None:
                    dist = getattr(nn, "distance", None)

                if callable(dist):
                    try:
                        dist = dist()
                    except TypeError:
                        # Some objects (e.g., PeriodicSite) require an argument.
                        dist = None

                if dist is not None:
                    distances.append(float(dist))
            if distances:
                nearest_distances.append(min(distances))

        if not nearest_distances:
            return None, None

        return float(np.mean(nearest_distances)), float(np.max(nearest_distances))

    def _process_doc(self, d) -> Dict[str, Any]:
        """
        Processes a single MP document into a dictionary, adding chemical
        baseline features like anonymized formula and EN difference.
        """
        # Helper to safely extract data whether 'd' is a dict or an MP-API object
        def get_val(key):
            try:
                return d[key]
            except (KeyError, TypeError):
                return getattr(d, key, None)
                
        # Safe casting helpers
        def safe_float(val):
            return float(val) if val is not None else None
            
        def safe_str(val):
            return str(val) if val is not None else None
            
        def safe_bool(val):
            return bool(val) if val is not None else None

        # 1. Structure and Lattice Handling
        raw_struct = get_val("structure")
        if isinstance(raw_struct, Structure):
            struct = raw_struct
        else:
            struct = Structure.from_dict(raw_struct)
            
        lat = struct.lattice
        avg_bond_length, max_bond_length = self._compute_bond_length_stats(struct)

        # 2. Symmetry Handling
        sym = get_val("symmetry")
        if isinstance(sym, dict):
            c_sys = sym.get("crystal_system")
            sg = sym.get("symbol")
        elif sym is not None:
            c_sys = getattr(sym, "crystal_system", None)
            sg = getattr(sym, "symbol", None)
        else:
            c_sys, sg = None, None

        # ---------------------------------------------------------
        # 3. CHEMICAL BASELINE FEATURES (New)
        # ---------------------------------------------------------
        formula = safe_str(get_val("formula_pretty"))
        anon_formula = None
        max_en_diff = None

        # Get rigorous symmetry data
        sga = SpacegroupAnalyzer(struct)
        sym_dataset = sga.get_symmetry_dataset()

        # Extract true baseline labels
        pearson = sga.get_pearson_symbol()
        wyckoff_seq = sym_dataset["wyckoffs"] if sym_dataset else []
        # Sort and join unique wyckoff letters to match standard notation (e.g., 'a_b')
        wyckoff_str = "_".join(sorted(list(set(wyckoff_seq))))

        # The true AFLOW-style prototype
        true_prototype = f"{safe_str(anon_formula)}_{pearson}_{safe_str(sg)}_{wyckoff_str}"

        if formula:
            try:
                comp = Composition(formula)
                
                # Prototype: Maps specific formula to generic (e.g., SrTiO3 -> ABC3)
                anon_formula = comp.anonymized_formula
                
                # Bonding: Calculate max Pauling electronegativity difference (Δχ)
                elements = comp.elements
                # Filter elements that have defined Pauling electronegativity
                ens = [el.X for el in elements if getattr(el, 'X', None) and el.X > 0]
                
                if len(ens) > 1:
                    max_en_diff = max(ens) - min(ens)
                elif len(ens) == 1:
                    max_en_diff = 0.0 # Pure elemental solids
            except Exception as e:
                logger.debug(f"Could not compute chemistry features for {formula}: {e}")

        # 4. Return flattened dictionary
        return {
            "material_id": safe_str(get_val("material_id")),
            "formula_pretty": formula,
            "anonymized_formula": safe_str(anon_formula),
            "structural_prototype": true_prototype,
            "max_en_diff": safe_float(max_en_diff), 
            "energy_per_atom": safe_float(get_val("energy_per_atom")),
            "formation_energy_per_atom": safe_float(get_val("formation_energy_per_atom")),
            "band_gap": safe_float(get_val("band_gap")),
            "is_metal": safe_bool(get_val("is_metal")),
            "raw_structure": json.dumps(struct.as_dict()),
            "crystal_system": safe_str(c_sys),
            "space_group": safe_str(sg),
            "pearson_symbol": safe_str(pearson),
            "density": safe_float(get_val("density")),
            "a": safe_float(lat.a),
            "b": safe_float(lat.b),
            "c": safe_float(lat.c),
            "alpha": safe_float(lat.alpha),
            "beta": safe_float(lat.beta),
            "gamma": safe_float(lat.gamma),
            "volume": safe_float(struct.volume),
            "num_sites": int(len(struct)),
            "energy_above_hull": safe_float(get_val("energy_above_hull")),
            "avg_bond_length": safe_float(avg_bond_length),
            "max_bond_length": safe_float(max_bond_length),
            "is_injected": 0,
            "outlier_category": None,
        }
    def _get_structures(self) -> List[Structure]:
        logger.info("Reconstructing Pymatgen structures from JSON...")
        struct_strings = self.df["raw_structure"].to_list()
        return [Structure.from_dict(json.loads(s)) for s in tqdm(struct_strings, desc="Reconstructing structures")]


    def _add_descriptors(
        self,
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.5,
        chunk_size=1000, 
        output_tag: Optional[str] = None,
        attach_to_df: bool = True,
    ) -> None:

        # 1. Check if any descriptors need to be computed
        if not (self.add_soap or self.add_acsf or getattr(self, "add_mace", False) or getattr(self, "add_coulomb", False)):
            logger.info("Skipping descriptor computation (all descriptor flags are False).")
            return
        
        if output_tag:
            logger.info(f"Ignoring output_tag={output_tag} since descriptors are attached directly to dataframe.")

        # 2. Extract unique elements (Needed for SOAP/ACSF)
        if self.add_soap or self.add_acsf:
            logger.info("Extracting unique elements from formulas...")
            formulas = self.df["formula_pretty"].to_list()
            unique_elements_set = set()
            for f in formulas:
                comp = Composition(f)
                unique_elements_set.update([el.symbol for el in comp.elements])
            
            unique_elements = sorted(list(unique_elements_set))
            logger.info(f"Found {len(unique_elements)} unique elements.")

        # 3. Initialize Engines
        soap_engine, soap_matrix_engine, acsf_engine, coulomb_engine, mace_engine = None, None, None, None, None
        
        if self.add_soap:
            soap_engine = SOAP(
                species=unique_elements, periodic=True, r_cut=r_cut, n_max=n_max, 
                l_max=l_max, sigma=sigma, sparse=False, average="inner",
                compression={"mode": "mu2"}
            )
            soap_matrix_engine = SOAP(
                species=unique_elements, periodic=True, r_cut=r_cut, n_max=n_max, 
                l_max=l_max, sigma=sigma, sparse=False, average="off",
                compression={"mode": "mu2"}
            )
        
        if self.add_acsf:
            acsf_engine = ACSF(
                species=unique_elements, periodic=True, r_cut=r_cut,
                g2_params=[[1, 1], [1, 2], [1, 3]],
                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]],
            )

        if getattr(self, "add_coulomb", False):
            from dscribe.descriptors import CoulombMatrix
            logger.info("Calculating max atoms for Coulomb Matrix padding...")
            max_atoms = max([json.loads(s)["num_sites"] for s in self.df["raw_structure"] if "num_sites" in json.loads(s)] or [100])
            coulomb_engine = CoulombMatrix(n_atoms_max=max_atoms)

        if getattr(self, "add_mace", False):
            from mace.calculators import mace_mp
            logger.info("Loading MACE-MP model...")
            mace_engine = mace_mp(model="medium", device='cpu', default_dtype="float32")

        # 4. Process in Chunks
        total_rows = len(self.df)
        soap_all: List[Optional[List[float]]] = []
        soap_matrix_all: List[Optional[List[List[float]]]] = []
        
        acsf_all: List[Optional[List[float]]] = []
        acsf_matrix_all: List[Optional[List[List[float]]]] = []
        
        coulomb_all: List[Optional[List[float]]] = []
        
        mace_all: List[Optional[List[float]]] = []
        mace_matrix_all: List[Optional[List[List[float]]]] = []
        
        for chunk_idx, start_row in enumerate(range(0, total_rows, chunk_size)):
            end_row = min(start_row + chunk_size, total_rows)
            chunk_df = self.df[start_row:end_row]
            struct_strings = chunk_df["raw_structure"].to_list()

            # --- PROCESS SOAP ---
            if self.add_soap:
                logger.info(f"Computing SOAP chunk {chunk_idx} ({start_row} to {end_row})...")
                soap_all.extend(self._compute_feature(struct_strings, soap_engine, output_mode="pooled"))
                soap_matrix_all.extend(self._compute_feature(struct_strings, soap_matrix_engine, output_mode="matrix"))

            # --- PROCESS ACSF ---
            if self.add_acsf:
                logger.info(f"Computing ACSF chunk {chunk_idx} ({start_row} to {end_row})...")
                acsf_all.extend(self._compute_feature(struct_strings, acsf_engine, output_mode="pooled"))
                acsf_matrix_all.extend(self._compute_feature(struct_strings, acsf_engine, output_mode="matrix"))

            # --- PROCESS COULOMB ---
            if getattr(self, "add_coulomb", False):
                logger.info(f"Computing Coulomb Matrix chunk {chunk_idx} ({start_row} to {end_row})...")
                coulomb_all.extend(self._compute_feature(struct_strings, coulomb_engine, output_mode="pooled"))

            # --- PROCESS MACE ---
            if getattr(self, "add_mace", False):
                logger.info(f"Computing MACE chunk {chunk_idx} ({start_row} to {end_row})...")
                mace_all.extend(self._compute_mace_features(struct_strings, mace_engine, output_mode="pooled"))
                mace_matrix_all.extend(self._compute_mace_features(struct_strings, mace_engine, output_mode="matrix"))

        # 5. Attach features to DataFrame
        def _attach_or_fill_column(column_name: str, values: List[Any]) -> None:
            update_df = pl.DataFrame({
                "_row_idx": np.arange(len(values), dtype=np.int64),
                column_name: pa.array(values),
            })
            indexed_df = self.df.with_row_count("_row_idx")

            if column_name not in self.df.columns:
                self.df = indexed_df.join(update_df, on="_row_idx", how="left").drop("_row_idx")
                return

            self.df = (
                indexed_df
                .join(update_df, on="_row_idx", how="left", suffix="_new")
                .with_columns(
                    pl.coalesce(
                        pl.col(f"{column_name}_new"),
                        pl.col(column_name),
                    ).alias(column_name)
                )
                .drop("_row_idx", f"{column_name}_new")
            )

        if attach_to_df:
            if self.add_soap:
                _attach_or_fill_column("soap_embedding", soap_all)
                _attach_or_fill_column("soap_matrix", soap_matrix_all)
                
            if self.add_acsf:
                _attach_or_fill_column("acsf_embedding", acsf_all)
                _attach_or_fill_column("acsf_matrix", acsf_matrix_all)
                
            if getattr(self, "add_coulomb", False):
                _attach_or_fill_column("coulomb_matrix", coulomb_all)
                
            if getattr(self, "add_mace", False):
                _attach_or_fill_column("mace_embedding", mace_all)
                _attach_or_fill_column("mace_matrix", mace_matrix_all)

        logger.success("All requested descriptors successfully added to dataframe.")

    @staticmethod
    def _format_material_descriptor_output(
        vec,
        output_mode: str = "pooled",
    ) -> Optional[List[float] | List[List[float]]]:
        if vec is None:
            return None

        mode = output_mode.strip().lower()
        if mode not in {"pooled", "matrix"}:
            raise ValueError(f"Unsupported output_mode '{output_mode}'. Expected 'pooled' or 'matrix'.")

        arr = np.asarray(vec, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if mode == "matrix":
            return np.atleast_2d(arr).tolist()

        if arr.ndim == 1:
            return arr.ravel().tolist()
        return np.mean(arr, axis=0).ravel().tolist()

    def _compute_mace_features(
            self, 
            struct_strings: List[str], 
            mace_calc, 
            batch_size: int = 32,
            output_mode: str = "pooled",
        ) -> List[Optional[List[float] | List[List[float]]]]:
            """
            Helper method for extracting node embeddings from MACE and returning
            either pooled vectors or per-site matrices.
            """
            
            features = []
            for i in range(0, len(struct_strings), batch_size):
                batch_jsons = struct_strings[i : i + batch_size]
                batch_structs = [Structure.from_dict(json.loads(s)) for s in batch_jsons]
                
                for struct in batch_structs:
                    try:
                        atoms = AseAtomsAdaptor.get_atoms(struct)
                        
                        desc = mace_calc.get_descriptors(atoms)
                        
                        if isinstance(desc, (list, tuple)):
                            node_embeddings = np.array(desc[0])
                        else:
                            node_embeddings = np.array(desc)

                        features.append(
                            self._format_material_descriptor_output(
                                node_embeddings,
                                output_mode=output_mode,
                            )
                        )
                        
                    except Exception as e:
                        features.append(None)
                        
            return features


    def _compute_feature(
        self,
        struct_strings: List[str],
        engine,
        batch_size: int = 32,
        output_mode: str = "pooled",
    ) -> List[Optional[List[float] | List[List[float]]]]:
        features = []
        
        for i in range(0, len(struct_strings), batch_size):
            batch_jsons = struct_strings[i : i + batch_size]
            batch_structs = [Structure.from_dict(json.loads(s)) for s in batch_jsons]

            try:
                batch_atoms = [AseAtomsAdaptor.get_atoms(s) for s in batch_structs]
                batch_out = engine.create(batch_atoms, n_jobs=1)

                for vec in batch_out:
                    dense_vec = vec.toarray() if hasattr(vec, "toarray") else np.array(vec)
                    features.append(
                        self._format_material_descriptor_output(
                            dense_vec,
                            output_mode=output_mode,
                        )
                    )

            except Exception as e:
                for s in batch_structs:
                    try:
                        atoms = AseAtomsAdaptor.get_atoms(s)
                        vec = engine.create(atoms, n_jobs=1)
                        dense_vec = vec.toarray() if hasattr(vec, "toarray") else np.array(vec)
                        features.append(
                            self._format_material_descriptor_output(
                                dense_vec,
                                output_mode=output_mode,
                            )
                        )
                    except Exception as e2:
                        features.append(None)

        return features

    @staticmethod
    def _values_to_feature_matrices(
        values: Sequence[Optional[List[float] | List[List[float]]]]
    ) -> List[np.ndarray]:
        matrices: List[np.ndarray] = []
        for value in values:
            if value is None:
                matrices.append(np.empty((0, 0), dtype=np.float64))
                continue

            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            matrices.append(arr)
        return matrices
