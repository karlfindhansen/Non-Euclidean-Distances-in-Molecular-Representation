import sys
import os

from src.datasets import QM9Dataset
from utils.file_ops import ensure_directory

def main():
    # Initialize
    loader = QM9Dataset(root="data/QM9")
    
    # Load Data
    df = loader.load()
    print(f"Loaded {len(df)} molecules.")

    # Run Stress Test
    print("Running stress test...")
    perturbations = loader.run_stress_test()
    
    # Calculate Distances
    print("Calculating distances...")
    dist_matrix = loader.get_distance_matrix(descriptor='morgan')
    print(f"Distance matrix shape: {dist_matrix.shape}")

if __name__ == "__main__":
    main()