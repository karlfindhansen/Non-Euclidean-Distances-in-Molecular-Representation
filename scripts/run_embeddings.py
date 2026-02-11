import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import QM9Dataset
from utils.file_ops import get_device
from loguru import logger

def main():
    device = get_device()
    logger.info(f"Running on device: {device}")

    # 1. Initialize
    loader = QM9Dataset(root="data/QM9", subset_size=100)
    loader.load()

    # 2. Define Model
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    
    logger.info(f"Generating Embeddings using {model_name}...")
    
    # 3. Run Generation
    loader.add_selfies_transformer(model_name=model_name)
    
    # 4. Verification
    if "selfies_transformer" in loader.df.columns:
        emb_shape = len(loader.df["selfies_transformer"][0])
        logger.success("Embeddings generated successfully!")
        logger.info(f"Embedding Vector Dimension: {emb_shape}")
        
    else:
        logger.error("Failed to generate embeddings.")

if __name__ == "__main__":
    main()