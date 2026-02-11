import numpy as np
from loguru import logger

from src.datasets import QM9Dataset

def main():
    # 1. Initialize and Load
    loader = QM9Dataset()
    loader.load()

    # --- Scenario A: Morgan Fingerprints (Jaccard) ---
    logger.info("--- Computing Distance: Morgan Fingerprints (Jaccard) ---")
    dist_morgan = loader.get_distance_matrix(metric='morgan')
    
    logger.info(f"Shape: {dist_morgan.shape} | Mean Dist: {np.mean(dist_morgan):.4f}")

    # --- Scenario B: Transformer Embeddings (Euclidean) ---
    logger.info("\n--- Computing Distance: SELFIES Transformers (Euclidean) ---")
    dist_transformer = loader.get_distance_matrix(metric='selfies_transformer')
    
    logger.info(f"Shape: {dist_transformer.shape} | Mean Dist: {np.mean(dist_transformer):.4f}")

    # --- Scenario C: One-Hot Encodings (Euclidean) ---
    logger.info("\n--- Computing Distance: SELFIES One-Hot (Euclidean) ---")
    dist_onehot = loader.get_distance_matrix(metric='selfies_onehot')
    
    logger.info(f"Shape: {dist_onehot.shape} | Mean Dist: {np.mean(dist_onehot):.4f}")

    logger.info("\n" + "="*30)
    logger.info("PREVIEW: FIRST 3x3 BLOCK COMPARISON")
    logger.info("="*30)
    
    print(f"\nMorgan (Jaccard):\n{dist_morgan[:3, :3]}")
    print(f"\nTransformer (Euclidean):\n{dist_transformer[:3, :3]}")
    print(f"\nOne-Hot (Euclidean):\n{dist_onehot[:3, :3]}")

    # Logic Check: Diagonal should always be 0.0
    for name, mat in [("Morgan", dist_morgan), ("Transformer", dist_transformer), ("OneHot", dist_onehot)]:
        diag_sum = np.trace(mat)
        if not np.isclose(diag_sum, 0):
            logger.warning(f"{name} matrix diagonal is not zero! Check your distance logic.")
        else:
            logger.success(f"{name} matrix passed diagonal integrity check.")

if __name__ == "__main__":
    main()