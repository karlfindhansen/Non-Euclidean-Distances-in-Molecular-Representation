import os

import polars as pl

qm9_outliers = {
    # ---------------------------------------------------------
    # 1. Size Outliers (10-15 heavy atoms)
    # These share the QM9 elements (C, N, O, F) but violate the 
    # 9-atom maximum limit. Your model should catch these easily.
    # ---------------------------------------------------------
    "size_outliers": [
        "c1ccccc1-c2ccccc2",         # Biphenyl (12 atoms)
        "CC(=O)Oc1ccccc1C(=O)O",     # Aspirin (13 atoms)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",# Ibuprofen (15 atoms)
        "C1C2CC3CC1CC(C2)C3",        # Adamantane (10 atoms)
        "O=C1CCC(=O)N1C2CCC(=O)NC2=O"# Thalidomide core (14 atoms)
    ],

    # ---------------------------------------------------------
    # 2. Element Outliers (<= 9 heavy atoms)
    # These fit the size constraint but contain elements completely 
    # absent from QM9 (S, P, Cl, Br, Si). 
    # ---------------------------------------------------------
    "element_outliers": [
        "c1ccsc1",                   # Thiophene (Sulfur, 5 atoms)
        "CS(=O)(=O)C",               # Dimethyl sulfone (Sulfur, 5 atoms)
        "Clc1ccccc1",                # Chlorobenzene (Chlorine, 7 atoms)
        "CP(C)C",                    # Trimethylphosphine (Phosphorus, 4 atoms)
        "C[Si](C)(C)C",              # Tetramethylsilane (Silicon, 5 atoms)
        "BrC(Br)Br"                  # Bromoform (Bromine, 4 atoms)
    ],

    # ---------------------------------------------------------
    # 3. Topological & Chemical Oddities (<= 9 heavy atoms)
    # These only contain C, N, O, F and fit the size limit, making 
    # them the hardest to detect. They represent highly strained, 
    # reactive, or unusual topologies that are statistically rare.
    # ---------------------------------------------------------
    "topology_outliers": [
        "C1#CC#CC#C1",               # Cyclohexa-1,3,5-triyne (Extremely strained/unstable)
        "O=C=C=C=O",                 # Carbon suboxide (Highly linear, unusual bonding)
        "N1N=NN=N1",                 # Pentazole (All-nitrogen ring)
        "C1=C=C=C1",                 # Cyclobutadiene (Anti-aromatic)
        "ON(O)N(O)O"                 # Tetrahydroxydiazene (Highly unusual N-O clustering)
    ],

    # ---------------------------------------------------------
    # 4. "Real World" Complex Molecules
    # Massive outliers that violate both size and sometimes element 
    # constraints. Good for testing if your pipeline breaks or 
    # handles extreme extremes.
    # ---------------------------------------------------------
    "extreme_outliers": [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine (14 atoms)
        "C1=CC=C(C=C1)S(=O)(=O)N",      # Benzenesulfonamide (S, 11 atoms)
        "NC(C)C(=O)O",                  # Alanine (Amino acid check)
        "C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=CC=CC=C3" # Triphenylphosphine (20 atoms, P)
    ]
}

if __name__ == '__main__':
    data = []
    for category, smiles_list in qm9_outliers.items():
        for smiles in smiles_list:
            data.append({
                "smiles": smiles, 
                "outlier_category": category,
                "is_injected": 1
            })

    df = pl.DataFrame(data)

    # 3. Define the exact path you requested
    save_dir = "data/QM9/outliers"
    file_path = os.path.join(save_dir, "synthetic_outliers.parquet")

    # 4. Ensure the directory exists and write the file
    os.makedirs(save_dir, exist_ok=True)
    df.write_parquet(file_path)

    print(f"Successfully saved {df.height} outliers to: {file_path}")
