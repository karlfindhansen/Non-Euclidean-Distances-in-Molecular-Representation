from utils.file_ops import get_device

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
from dscribe.descriptors import SOAP, ACSF
from typing import Sequence
from scipy.linalg import subspace_angles
import geomstats.geometry.spd_matrices as spd
from pyriemann.utils.distance import distance_riemann, distance_logeuclid

class MolecularFeaturizer:
    """
    Responsible for converting SMILES/SELFIES into vector representations.
    Now includes 3D physics-based descriptors (SOAP, ACSF).
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
    def compute_selfies_onehot(selfies_series: pl.Series) -> pl.Series:
        logger.info("Computing One-Hot Encodings...")
        data = [s for s in selfies_series.to_list() if s]
        if not data: return pl.Series("selfies_onehot", [None]*len(selfies_series))

        alphabet = sf.get_alphabet_from_selfies(data)
        alphabet.add("[nop]")
        vocab = {s: i for i, s in enumerate(sorted(list(alphabet)))}
        max_len = max(sf.len_selfies(s) for s in data)

        def _encode(s):
            if not s: return None
            return sf.selfies_to_encoding(s, vocab_stoi=vocab, pad_to_len=max_len, enc_type="one_hot")

        return pl.Series("selfies_onehot", [_encode(s) for s in selfies_series.to_list()])

    @staticmethod
    def compute_soap(smiles_series: pl.Series, r_cut=6.0, n_max=8, l_max=6, sigma=0.5, species=None) -> pl.Series:
        """
        Computes SOAP descriptors (Smooth Overlap of Atomic Positions).
        Returns the mean vector of atomic SOAP features for each molecule.
        """
        if species is None: species = ["C", "H", "O", "N", "F"]
        
        logger.info(f"Computing SOAP (rcut={r_cut}, nmax={n_max}, lmax={l_max})...")
        
        soap_engine = SOAP(
            species=species, periodic=False, 
            r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma
        )

        def _compute_single_soap(s):
            mol = MolecularFeaturizer._generate_3d_mol(s)
            if mol is None: return None
            
            try:
                atoms = MolecularFeaturizer._rdkit_to_ase(mol)
                # atomic_soap shape: (n_atoms, n_features)
                atomic_soap = soap_engine.create(atoms)
                # Mean pooling -> (n_features,)
                return np.mean(atomic_soap, axis=0).tolist()
            except Exception as e:
                return None

        return smiles_series.map_elements(_compute_single_soap, return_dtype=pl.List(pl.Float64))

    @staticmethod
    def compute_acsf(smiles_series: pl.Series, r_cut=6.0, species=None) -> pl.Series:
        """
        Computes ACSF (Atom-Centered Symmetry Functions).
        Returns the mean vector of atomic ACSF features.
        """
        if species is None: species = ["C", "H", "O", "N", "F"]
        
        logger.info(f"Computing ACSF (rcut={r_cut})...")
        
        acsf_engine = ACSF(
            species=species, periodic=False, r_cut=r_cut,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1]]
        )

        def _compute_single_acsf(s):
            mol = MolecularFeaturizer._generate_3d_mol(s)
            if mol is None: return None
            
            try:
                atoms = MolecularFeaturizer._rdkit_to_ase(mol)
                atomic_acsf = acsf_engine.create(atoms)
                return np.mean(atomic_acsf, axis=0).tolist()
            except Exception:
                return None

        return smiles_series.map_elements(_compute_single_acsf, return_dtype=pl.List(pl.Float64))
    

    @staticmethod
    def compute_chemprop_embeddings(
        smiles_series: pl.Series,
        model_path: str | None = None,
        batch_size: int = 64,
        device: str = get_device()
    ) -> pl.Series:
        """
        Compute Chemprop learned molecular embeddings for v2.2.2+.
        """

        logger.info(f"Computing Chemprop embeddings on {device}...")

        # 1. LOAD OR INITIALIZE MODEL
        if model_path is not None:
            logger.info(f"Loading trained model from {model_path}...")
            predictor = models.load_model(model_path)
            model = predictor.encoder
        else:
            logger.warning("No model_path provided. Using RANDOM (untrained) MPNN weights.")
            
            d_h = 300
            message_passing = nn.BondMessagePassing(d_h=d_h, depth=3)
            aggregator = nn.MeanAggregation()
            predictor = nn.RegressionFFN()
            
            model = models.MPNN(message_passing, aggregator, predictor)

        model = model.to(device)
        model.eval()

        valid_indices = []
        valid_smiles = []
        
        for idx, s in enumerate(smiles_series):
            if s and Chem.MolFromSmiles(s):
                valid_smiles.append(s)
                valid_indices.append(idx)

        if not valid_smiles:
            return pl.Series([None] * len(smiles_series))

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        
        datapoints = [data.MoleculeDatapoint.from_smi(s) for s in valid_smiles]
        
        dset = data.MoleculeDataset(datapoints, featurizer=featurizer)
        loader = data.build_dataloader(dset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 4. INFERENCE
        embeddings = []
        
        with torch.no_grad():
            for batch in loader:
                batch_graph = batch.bmg
                batch_graph.to(device)

                features = batch.X_d
                if features is not None:
                    features = features.to(device)
                    
                atom_descriptors = batch.V_d
                if atom_descriptors is not None:
                    atom_descriptors = atom_descriptors.to(device)

                if hasattr(model, "fingerprint"):
                    batch_vecs = model.fingerprint(batch_graph, V_d=atom_descriptors, X_d=features)
                else:
                    H_v = model.message_passing(batch_graph, V_d=atom_descriptors)
                    batch_vecs = model.aggregator(H_v, batch_graph)

                embeddings.extend(batch_vecs.cpu().numpy().tolist())

        # 5. RECONSTRUCT RESULT
        final_result = [None] * len(smiles_series)
        for idx, emb in zip(valid_indices, embeddings):
            final_result[idx] = emb

        return pl.Series("chemprop_embedding", final_result)
    
    
def get_features_xyz(frames):
    """
    Converts a list of ASE Atoms objects into fixed-length 
    feature vectors based on sorted pairwise atomic distances.
    """
    feature_vectors = []
    
    # 1. Calculate pairwise distances for every frame
    raw_distances = []
    max_len = 0
    
    for frame in frames:
        dists = pdist(frame.get_positions())
        # Sort distances to be invariant to atom indexing (permutation invariant)
        dists.sort()
        
        raw_distances.append(dists)
        max_len = max(max_len, len(dists))
        
    # 2. Pad vectors with zeros so they are all the same length
    for dists in raw_distances:
        # Create a zero vector of max length
        vec = np.zeros(max_len)
        # Fill it with the sorted distances
        vec[:len(dists)] = dists
        feature_vectors.append(vec)
        
    return np.array(feature_vectors)

def get_raw_xyz_features(frames):
    """
    Flattens XYZ coordinates and pads them to a fixed length
    to handle molecules with different numbers of atoms.
    """
    # 1. Get flattened coordinates for all frames
    flat_coords_list = [f.get_positions().flatten() for f in frames]
    
    # 2. Find the maximum length (3 * max_num_atoms)
    max_len = max(len(c) for c in flat_coords_list)
    
    # 3. Pad smaller vectors with zeros
    padded_features = []
    for coords in flat_coords_list:
        # Create a zero vector of the maximum length
        vec = np.zeros(max_len)
        # Fill the beginning with the actual coordinates
        vec[:len(coords)] = coords
        padded_features.append(vec)
        
    return np.array(padded_features)

class Grassmann:
    """
    Task 7.1: Grassmann Manifold (Subspaces)
    Represents molecules as k-dimensional planes.
    """
    @staticmethod
    def get_uk_bases(frames: Sequence[Atoms], k: int = 3) -> np.ndarray:
        """
        Step A & B: Center coordinates and perform SVD to get 
        the top-k Left Singular Vectors (Uk).
        """
        bases = []
        for frame in frames:
            coords = frame.get_positions()
            centered = coords - np.mean(coords, axis=0)
            
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            bases.append(u[:, :k])
        return np.array(bases)

    @staticmethod
    def distance(U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Step C: Compute Grassmann Distance using Principal Angles.
        Note: distance = sqrt(sum(theta_i^2))
        """
        angles = subspace_angles(U1, U2)
        return float(np.linalg.norm(angles))

    @classmethod
    def distance_matrix(cls, frames: Sequence[Atoms], k: int = 3) -> np.ndarray:
        bases = cls.get_uk_bases(frames, k=k)
        num_frames = len(bases)
        dist_matrix = np.zeros((num_frames, num_frames))

        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                dist = cls.distance(bases[i], bases[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

class Riemann:

    _METRICS = {
        "log-euclidean":    distance_logeuclid,
        "affine-invariant": distance_riemann,
    }

    @staticmethod
    def compute_covariance_matrices(frames: Sequence[Atoms]) -> np.ndarray:

        covs = []
        for frame in frames:
            positions = frame.get_positions()       
            cov = np.cov(positions, rowvar=False)   
            cov += np.eye(cov.shape[0]) * 1e-6     
            covs.append(cov)
        return np.array(covs)

    @classmethod
    def distance_matrix(
        cls,
        frames: Sequence[Atoms],
        metric_type: str = "log-euclidean",
    ) -> np.ndarray:

        key = metric_type.strip().lower().replace("_", "-").replace(" ", "-")
        metric_fn = cls._METRICS.get(key)
        if metric_fn is None:
            raise ValueError(
                f"Unknown metric_type '{metric_type}'. "
                f"Choose from: {list(cls._METRICS)}."
            )

        covs = cls.compute_covariance_matrices(frames)
        n = len(covs)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = metric_fn(covs[i], covs[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d

        return dist_matrix
