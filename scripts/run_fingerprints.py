from src.datasets import QM9Dataset
from loguru import logger

def main():
    # 1. Initialize Loader
    loader = QM9Dataset(root="data/QM9")
    loader.load()
    
    # 2. Load Base Data
    logger.info("Loading QM9 Data...")
    
    # 3. Generate Morgan Fingerprints
    radius = 2
    fp_size = 1024
    
    logger.info(f"Generating Morgan Fingerprints (r={radius}, n={fp_size})...")
    
    loader.add_morgan_fingerprints(radius=radius, fp_size=fp_size)
    
    # 4. Verification
    if "morgan_fingerprint" in loader.df.columns:
        logger.success("Fingerprints generated successfully!")
        logger.info(f"DataFrame Shape: {loader.df.shape}")
    else:
        logger.error("Failed to generate fingerprints.")

if __name__ == "__main__":
    main()