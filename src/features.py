import numpy as np
from scipy.spatial.distance import pdist

import polars as pl
import torch
import selfies as sf
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
from loguru import logger

import polars as pl
import numpy as np
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF
from transformers import AutoTokenizer, AutoModel
import selfies as sf

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
    def compute_selfies_transformer(selfies_series: pl.Series, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", batch_size: int = 32) -> pl.Series:
        logger.info(f"Computing Transformer Embeddings ({model_name})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

        # Pre-process inputs
        clean_selfies = [s if s else "[nop]" for s in selfies_series.to_list()]
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(clean_selfies), batch_size):
                batch = clean_selfies[i : i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                output = model(**inputs)
                # Mean pooling over tokens
                embeddings.extend(output.last_hidden_state.mean(dim=1).tolist())

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