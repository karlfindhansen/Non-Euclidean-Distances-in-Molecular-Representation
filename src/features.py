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

class MolecularFeaturizer:
    """
    Responsible for converting SMILES/SELFIES into vector representations.
    """
    
    @staticmethod
    def compute_morgan_fingerprints(
        smiles_series: pl.Series, 
        radius: int = 3, 
        fp_size: int = 2048
    ) -> pl.Series:
        """Computes Morgan Fingerprints."""
        logger.info(f"Computing Morgan Fingerprints (Radius={radius}, Size={fp_size})...")
        
        morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)

        def _smiles_to_fp(s: str):
            if not s: return None
            mol = Chem.MolFromSmiles(s)
            return list(morgan_gen.GetFingerprint(mol)) if mol else None

        return smiles_series.map_elements(
            _smiles_to_fp, return_dtype=pl.List(pl.Int8)
        )

    @staticmethod
    def compute_selfies_transformer(
        selfies_series: pl.Series,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        batch_size: int = 32
    ) -> pl.Series:
        """Computes Transformer embeddings from SELFIES."""
        logger.info(f"Loading Transformer model: {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise

        selfies_list = selfies_series.to_list()
        clean_list = [s if s else "[nop]" for s in selfies_list]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(clean_list), batch_size):
                batch = clean_list[i : i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.extend(batch_emb.tolist())

        return pl.Series("selfies_transformer", all_embeddings)
    
    @staticmethod
    def compute_selfies_onehot(selfies_series: pl.Series) -> pl.Series:
        """
        Builds a vocabulary from the provided SELFIES and returns 
        one-hot encoded matrices.
        """
        logger.info("Computing One-Hot Encodings...")
        selfies_list = [s for s in selfies_series.to_list() if s is not None]
        
        if not selfies_list:
            return pl.Series("selfies_onehot", [None] * len(selfies_series))

        # Build Alphabet
        alphabet = sf.get_alphabet_from_selfies(selfies_list)
        alphabet.add("[nop]")  # Padding token
        vocab = {s: i for i, s in enumerate(sorted(list(alphabet)))}
        max_len = max(sf.len_selfies(s) for s in selfies_list)
        
        def _encode(s):
            if s is None: return None
            return sf.selfies_to_encoding(
                s, vocab_stoi=vocab, pad_to_len=max_len, enc_type="one_hot"
            )

        onehot_list = [(_encode(s) if s else None) for s in selfies_series.to_list()]
        return pl.Series("selfies_onehot", onehot_list)


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