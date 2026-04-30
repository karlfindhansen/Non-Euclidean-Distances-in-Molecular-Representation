import numpy as np
from scipy.spatial.distance import pdist
import polars as pl
import torch

import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from chemprop import data, featurizers, models, nn
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF, CoulombMatrix

from utils.file_ops import get_device
from scripts.qm9.chemprop import CheMeleonFingerprint

class MolecularFeaturizer:
    """
    Responsible for converting SMILES/SELFIES into vector representations.
    Now includes 3D physics-based descriptors (SOAP, ACSF, Coulomb Matrix).
    """
    
    @staticmethod
    def _generate_3d_mol(smiles: str):
        """Helper to generate a 3D RDKit molecule from SMILES."""
        if not smiles: return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            mol = Chem.AddHs(mol)
            
            # Embed 3D coordinates
            params = AllChem.ETKDG()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                return None
            
            # Optimize geometry (MMFF94)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass # Accept unoptimized if MMFF fails
                
            return mol
        except Exception:
            return None

    @staticmethod
    def _rdkit_to_ase(mol):
        """Helper to convert RDKit molecule to ASE Atoms object."""
        return Atoms(
            symbols=[a.GetSymbol() for a in mol.GetAtoms()],
            positions=mol.GetConformer().GetPositions(),
        )

    @staticmethod
    def _series_values(series: pl.Series | None, length: int) -> list:
        if series is None:
            return [None] * length
        return series.to_list()

    @staticmethod
    def _build_ase_atoms(
        smiles: str | None,
        coordinates: list[list[float]] | np.ndarray | None = None,
        atomic_numbers: list[int] | np.ndarray | None = None,
    ) -> Atoms | None:
        """Build ASE atoms from explicit coordinates when available, else from RDKit."""
        if coordinates is not None and atomic_numbers is not None:
            try:
                positions = np.asarray(coordinates, dtype=np.float64)
                numbers = np.asarray(atomic_numbers, dtype=np.int64)
                if (
                    positions.ndim == 2
                    and positions.shape[1] == 3
                    and numbers.ndim == 1
                    and len(numbers) == len(positions)
                    and len(numbers) > 0
                ):
                    return Atoms(numbers=numbers, positions=positions)
            except Exception:
                pass
            
        logger.warning("Generating molecule from smiles")
        mol = MolecularFeaturizer._generate_3d_mol(smiles or "")
        if mol is None:
            return None
        return MolecularFeaturizer._rdkit_to_ase(mol)

    @staticmethod
    def _collect_species(
        smiles_series: pl.Series,
        atomic_numbers_series: pl.Series | None = None,
    ) -> list[int]:
        species_set: set[int] = set()
        atomic_number_rows = MolecularFeaturizer._series_values(
            atomic_numbers_series,
            len(smiles_series),
        )

        for smi, atomic_numbers in zip(smiles_series.to_list(), atomic_number_rows):
            if atomic_numbers:
                try:
                    species_set.update(int(z) for z in atomic_numbers)
                    continue
                except Exception:
                    pass

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            mol = Chem.AddHs(mol)
            for atom in mol.GetAtoms():
                species_set.add(int(atom.GetAtomicNum()))

        return sorted(species_set)

    @staticmethod
    def _format_descriptor_output(
        vec,
        output_mode: str = "pooled",
        reduce: str | None = "mean",
    ) -> list[float] | list[list[float]] | None:
        """
        Convert a descriptor output into either a pooled 1D list or a matrix-shaped
        nested list.
        """
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
        if reduce == "mean":
            return np.mean(arr, axis=0).ravel().tolist()
        return arr.ravel().tolist()

    @staticmethod
    def _compute_atomwise_descriptor_pair(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None,
        atomic_numbers_series: pl.Series | None,
        descriptor_fn,
        pooled_name: str,
        matrix_name: str,
        reduce: str = "mean",
    ) -> tuple[pl.Series, pl.Series]:
        """
        Compute atom-wise descriptors once per molecule and return both
        pooled and matrix-shaped outputs.
        """
        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))

        pooled_values: list[list[float] | None] = []
        matrix_values: list[list[list[float]] | None] = []

        for smiles, coordinates, atomic_numbers in zip(
            smiles_series.to_list(),
            coordinate_rows,
            atomic_number_rows,
        ):
            try:
                atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
                if atoms is None:
                    pooled_values.append(None)
                    matrix_values.append(None)
                    continue

                atomwise_values = descriptor_fn(atoms)
                pooled_values.append(
                    MolecularFeaturizer._format_descriptor_output(
                        atomwise_values,
                        output_mode="pooled",
                        reduce=reduce,
                    )
                )
                matrix_values.append(
                    MolecularFeaturizer._format_descriptor_output(
                        atomwise_values,
                        output_mode="matrix",
                        reduce=reduce,
                    )
                )
            except Exception:
                pooled_values.append(None)
                matrix_values.append(None)

        return pl.Series(pooled_name, pooled_values), pl.Series(matrix_name, matrix_values)

    @staticmethod
    def compute_morgan_fingerprints(smiles_series: pl.Series, radius: int = 3, fp_size: int = 2048) -> pl.Series:
        logger.info(f"Computing Morgan Fingerprints (Radius={radius}, Size={fp_size})...")
        gen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        
        def _compute(s):
            if not s: return None
            mol = Chem.MolFromSmiles(s)
            return list(gen.GetFingerprint(mol)) if mol else None

        return smiles_series.map_elements(_compute, return_dtype=pl.List(pl.Int8))

    @staticmethod
    def compute_selfies_transformer(selfies_series: pl.Series, 
                                    model_name: str = "HUBioDataLab/SELFormer", 
                                    batch_size: int = 32) -> pl.Series:
        """
        Computes molecular embeddings using the SELFormer encoder-only architecture.
        """
        logger.info(f"Computing SELFormer Embeddings using {model_name}...")

        device = get_device()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            model.eval()
        except Exception as e:
            logger.error(f"SELFormer load failed: {e}")
            raise

        clean_selfies = [s.replace("][", "] [") if s else "[nop]" for s in selfies_series.to_list()]
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(clean_selfies), batch_size):
                batch = clean_selfies[i : i + batch_size]
                
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                
                hidden_states = outputs.last_hidden_state 
                
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                
                embeddings.extend(mean_pooled.cpu().tolist())

        return pl.Series("selfies_transformer", embeddings)

    @staticmethod
    def compute_selfies_onehot(selfies_series: pl.Series, flatten: bool = False) -> pl.Series:
        logger.info("Computing One-Hot Encodings...")
        data = [s for s in selfies_series.to_list() if s]
        if not data: return pl.Series("selfies_onehot", [None]*len(selfies_series))

        alphabet = sf.get_alphabet_from_selfies(data)
        alphabet.add("[nop]")
        vocab = {s: i for i, s in enumerate(sorted(list(alphabet)))}
        max_len = max(sf.len_selfies(s) for s in data)

        def _encode(s):
            if not s: return None
            encoded = sf.selfies_to_encoding(
                s, vocab_stoi=vocab, pad_to_len=max_len, enc_type="one_hot"
            )
            if not flatten:
                return encoded
            return np.asarray(encoded).reshape(-1).tolist()

        return pl.Series("selfies_onehot", [_encode(s) for s in selfies_series.to_list()])

    @staticmethod
    def compute_soap(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.5,
        output_mode: str = "pooled",
    ) -> pl.Series:
        """
        Computes SOAP descriptors (Smooth Overlap of Atomic Positions).
        Returns either pooled vectors or atom-level descriptor matrices.
        """
        mode = output_mode.strip().lower()

        species = MolecularFeaturizer._collect_species(
            smiles_series,
            atomic_numbers_series=atomic_numbers_series,
        )
        if not species:
            return pl.Series("soap_embedding", [None] * len(smiles_series))

        logger.info(f"Computing SOAP (rcut={r_cut}, nmax={n_max}, lmax={l_max})...")
        soap_engine = SOAP(
            species=species, periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average="inner" if output_mode == 'pooled' else "off",
            compression={"mode": "mu2"},
        )

        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))

        def _compute_single_soap(smiles, coordinates, atomic_numbers):
            try:
                atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
                if atoms is None:
                    return None
                soap_values = soap_engine.create(atoms)
                return MolecularFeaturizer._format_descriptor_output(
                    soap_values,
                    output_mode=mode,
                    reduce="mean",
                )
            except Exception:
                return None

        return pl.Series(
            "soap_embedding",
            [
                _compute_single_soap(smiles, coordinates, atomic_numbers)
                for smiles, coordinates, atomic_numbers in zip(
                    smiles_series.to_list(),
                    coordinate_rows,
                    atomic_number_rows,
                )
            ],
        )

    @staticmethod
    def compute_soap_outputs(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.5,
    ) -> tuple[pl.Series, pl.Series]:
        """
        Computes both pooled SOAP embeddings and atom-level SOAP matrices.
        The pooled output preserves the existing `average="inner"` behavior.
        """
        species = MolecularFeaturizer._collect_species(
            smiles_series,
            atomic_numbers_series=atomic_numbers_series,
        )
        if not species:
            n_rows = len(smiles_series)
            return (
                pl.Series("soap_embedding", [None] * n_rows),
                pl.Series("soap_matrix", [None] * n_rows),
            )

        logger.info(f"Computing SOAP (rcut={r_cut}, nmax={n_max}, lmax={l_max})...")
        pooled_engine = SOAP(
            species=species,
            periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average="inner",
            compression={"mode": "mu2"},
        )
        matrix_engine = SOAP(
            species=species,
            periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average="off",
            compression={"mode": "mu2"},
        )

        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))
        pooled_values: list[list[float] | None] = []
        matrix_values: list[list[list[float]] | None] = []

        for smiles, coordinates, atomic_numbers in zip(
            smiles_series.to_list(),
            coordinate_rows,
            atomic_number_rows,
        ):
            try:
                atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
                if atoms is None:
                    pooled_values.append(None)
                    matrix_values.append(None)
                    continue

                pooled_values.append(
                    MolecularFeaturizer._format_descriptor_output(
                        pooled_engine.create(atoms),
                        output_mode="pooled",
                        reduce="mean",
                    )
                )
                matrix_values.append(
                    MolecularFeaturizer._format_descriptor_output(
                        matrix_engine.create(atoms),
                        output_mode="matrix",
                        reduce="mean",
                    )
                )
            except Exception:
                pooled_values.append(None)
                matrix_values.append(None)

        return (
            pl.Series("soap_embedding", pooled_values),
            pl.Series("soap_matrix", matrix_values),
        )

    @staticmethod
    def compute_acsf(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        r_cut=6.0,
        output_mode: str = "pooled",
    ) -> pl.Series:
        """
        Computes ACSF (Atom-Centered Symmetry Functions).
        Returns either pooled vectors or atom-level descriptor matrices.
        """
        mode = output_mode.strip().lower()
        species = MolecularFeaturizer._collect_species(
            smiles_series,
            atomic_numbers_series=atomic_numbers_series,
        )
        if not species:
            return pl.Series("acsf_embedding", [None] * len(smiles_series))
        
        logger.info(f"Computing ACSF (rcut={r_cut})...")
        
        acsf_engine = ACSF(
            species=species, periodic=False, r_cut=r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]]
        )

        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))

        def _compute_single_acsf(smiles, coordinates, atomic_numbers):
            try:
                atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
                if atoms is None:
                    return None
                atomic_acsf = acsf_engine.create(atoms)
                return MolecularFeaturizer._format_descriptor_output(
                    atomic_acsf,
                    output_mode=mode,
                    reduce="mean",
                )
            except Exception:
                return None

        return pl.Series(
            "acsf_embedding",
            [
                _compute_single_acsf(smiles, coordinates, atomic_numbers)
                for smiles, coordinates, atomic_numbers in zip(
                    smiles_series.to_list(),
                    coordinate_rows,
                    atomic_number_rows,
                )
            ],
        )

    @staticmethod
    def compute_acsf_outputs(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        r_cut=6.0,
    ) -> tuple[pl.Series, pl.Series]:
        """
        Computes both pooled ACSF embeddings and atom-level ACSF matrices.
        """
        species = MolecularFeaturizer._collect_species(
            smiles_series,
            atomic_numbers_series=atomic_numbers_series,
        )
        if not species:
            n_rows = len(smiles_series)
            return (
                pl.Series("acsf_embedding", [None] * n_rows),
                pl.Series("acsf_matrix", [None] * n_rows),
            )

        logger.info(f"Computing ACSF (rcut={r_cut})...")
        acsf_engine = ACSF(
            species=species,
            periodic=False,
            r_cut=r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]],
        )

        return MolecularFeaturizer._compute_atomwise_descriptor_pair(
            smiles_series,
            coordinates_series,
            atomic_numbers_series,
            descriptor_fn=acsf_engine.create,
            pooled_name="acsf_embedding",
            matrix_name="acsf_matrix",
            reduce="mean",
        )
    
    @staticmethod
    def compute_coulomb_matrix(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        n_atoms_max: int | None = None,
        permutation: str = "sorted_l2"
    ) -> pl.Series:
        """
        Computes Coulomb Matrix descriptors from 3D geometries.
        Returns flattened vectors (length n_atoms_max * n_atoms_max).
        """
        logger.info(
            f"Computing Coulomb matrices (n_atoms_max={n_atoms_max}, permutation={permutation})..."
        )

        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))

        atoms_objects = []
        max_atoms = 0
        for smiles, coordinates, atomic_numbers in zip(
            smiles_series.to_list(),
            coordinate_rows,
            atomic_number_rows,
        ):
            atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
            atoms_objects.append(atoms)
            if atoms is not None:
                max_atoms = max(max_atoms, len(atoms))

        if max_atoms == 0:
            return pl.Series("coulomb_matrix", [None] * len(smiles_series))

        n_atoms = n_atoms_max if n_atoms_max is not None else max_atoms
        cm_engine = CoulombMatrix(
            n_atoms_max=n_atoms,
            permutation=permutation,
            sparse=False
        )

        features = []
        for atoms in atoms_objects:
            if atoms is None:
                features.append(None)
                continue

            try:
                vec = cm_engine.create(atoms)
                features.append(np.asarray(vec).ravel().tolist())
            except Exception:
                features.append(None)

        return pl.Series("coulomb_matrix", features)

    @staticmethod
    def compute_mace_embeddings(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        model: str = "medium",
        batch_size: int = 32,
        output_mode: str = "pooled",
    ) -> pl.Series:
        """
        Computes pooled or atom-level MACE embeddings from 3D molecular geometries.
        Each molecule is embedded with RDKit, converted to ASE Atoms, then
        optionally mean-pooled over atom-level MACE descriptors to form a
        single vector.
        """
        logger.info(f"Computing MACE embeddings (model={model}, batch_size={batch_size})...")
        mode = output_mode.strip().lower()

        from mace.calculators import mace_off

        mace_calc = mace_off(model=model, device="cpu", default_dtype="float32")
        smiles_list = smiles_series.to_list()
        coordinate_rows = MolecularFeaturizer._series_values(coordinates_series, len(smiles_series))
        atomic_number_rows = MolecularFeaturizer._series_values(atomic_numbers_series, len(smiles_series))
        embeddings: list[list[float] | None] = []

        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start:start + batch_size]
            batch_coordinates = coordinate_rows[start:start + batch_size]
            batch_atomic_numbers = atomic_number_rows[start:start + batch_size]
            for smiles, coordinates, atomic_numbers in zip(batch, batch_coordinates, batch_atomic_numbers):
                atoms = MolecularFeaturizer._build_ase_atoms(smiles, coordinates, atomic_numbers)
                if atoms is None:
                    embeddings.append(None)
                    continue

                try:
                    desc = mace_calc.get_descriptors(atoms)
                    if isinstance(desc, (list, tuple)):
                        node_embeddings = np.asarray(desc[0])
                    else:
                        node_embeddings = np.asarray(desc)

                    embeddings.append(
                        MolecularFeaturizer._format_descriptor_output(
                            node_embeddings,
                            output_mode=mode,
                            reduce="mean",
                        )
                    )
                except Exception:
                    embeddings.append(None)

        return pl.Series("mace_embedding", embeddings)

    @staticmethod
    def compute_mace_outputs(
        smiles_series: pl.Series,
        coordinates_series: pl.Series | None = None,
        atomic_numbers_series: pl.Series | None = None,
        model: str = "medium",
        batch_size: int = 32,
    ) -> tuple[pl.Series, pl.Series]:
        """
        Computes both pooled and atom-level MACE descriptors.
        """
        logger.info(f"Computing MACE embeddings (model={model}, batch_size={batch_size})...")
        from mace.calculators import mace_off

        mace_calc = mace_off(model=model, device="cpu", default_dtype="float32")

        def _descriptor_fn(atoms):
            desc = mace_calc.get_descriptors(atoms)
            if isinstance(desc, (list, tuple)):
                return np.asarray(desc[0])
            return np.asarray(desc)

        return MolecularFeaturizer._compute_atomwise_descriptor_pair(
            smiles_series,
            coordinates_series,
            atomic_numbers_series,
            descriptor_fn=_descriptor_fn,
            pooled_name="mace_embedding",
            matrix_name="mace_matrix",
            reduce="mean",
        )


    @staticmethod
    def compute_chemprop_embeddings(
        smiles_series: pl.Series,
        device: str = get_device()
    ) -> pl.Series:
        """
        Compute Chemprop learned molecular embeddings for v2.2.2+.
        """

        logger.info(f"Computing Chemprop embeddings on {device}...")
        chemeleon_fingerprint = CheMeleonFingerprint()
        smiles_list = smiles_series.to_list()
        descriptor = chemeleon_fingerprint(smiles_list)
        return pl.Series("chemprop_embedding", descriptor)

    
def get_features_xyz(frames):
    """
    Converts a list of ASE Atoms objects into fixed-length 
    feature vectors based on sorted pairwise atomic distances.
    """
    feature_vectors = []
    
    raw_distances = []
    max_len = 0
    
    for frame in frames:
        dists = pdist(frame.get_positions())
        dists.sort()
        
        raw_distances.append(dists)
        max_len = max(max_len, len(dists))
        
    for dists in raw_distances:
        vec = np.zeros(max_len)
        vec[:len(dists)] = dists
        feature_vectors.append(vec)
        
    return np.array(feature_vectors)

def get_raw_xyz_features(frames):
    """
    Flattens XYZ coordinates and pads them to a fixed length
    to handle molecules with different numbers of atoms.
    """
    flat_coords_list = [f.get_positions().flatten() for f in frames]
    
    max_len = max(len(c) for c in flat_coords_list)
    
    padded_features = []
    for coords in flat_coords_list:
        vec = np.zeros(max_len)
        vec[:len(coords)] = coords
        padded_features.append(vec)
        
    return np.array(padded_features)

def get_weighted_point_clouds(frames):
    masses = [molecule.get_masses() for molecule in frames]
    flat_coords_list = [f.get_positions().flatten() for f in frames]
    max_len = max(len(c) for c in flat_coords_list)

    weighted_point_clouds = []
    for coords, mass in zip(flat_coords_list, masses):
        weighted_coords = coords + np.repeat(mass, 3)  
        vec = np.zeros(max_len)
        vec[:len(weighted_coords)] = weighted_coords
        weighted_point_clouds.append(vec)
    
    return np.array(weighted_point_clouds)
    
