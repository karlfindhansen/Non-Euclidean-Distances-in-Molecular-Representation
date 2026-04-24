import os

import polars as pl

qm9_outliers = {
    # ---------------------------------------------------------
    # 1. Size Outliers (10-15 heavy atoms)
    # These share the QM9 elements (C, N, O, F) but violate the 
    # 9-atom maximum limit. Your model should catch these easily.
    # ---------------------------------------------------------
    "size_outliers": [
        "c1ccccc1CCC(=O)O",          # 3-Phenylpropionic acid (11 atoms)
        "c1ccc(cc1)C(=O)NCC",        # N-Ethylbenzamide (10 atoms)
        "c1cc(ccc1Cc2ccccc2)O",      # 4-Benzylphenol (14 atoms)
        "CC(C)c1ccc(C)cc1",          # p-Cymene (10 atoms)
        "CC1(C)C2CCC1(C)C(=O)C2",    # Camphor (11 atoms)
        "C1CC2CCC1CC2",              # Bicyclo[3.3.1]nonane (9 atoms in QM9, 10+ if branched) -> "CC1CC2CCC1CC2" (11 atoms)
        "O=C(O)c1ccccc1C(=O)O",      # Phthalic acid (12 atoms)
        "c1ccc2c(c1)cccc2",          # Naphthalene (10 atoms)
        "CCN(CC)C(=O)c1ccccc1",      # N,N-Diethylbenzamide (12 atoms)
        "c1ccc(cc1)N=Nc2ccccc2"      # Azobenzene (14 atoms)
    ],

    # ---------------------------------------------------------
    # 2. Element Outliers (<= 9 heavy atoms)
    # These fit the size constraint but contain elements completely 
    # absent from QM9 (S, P, Cl, Br, Si). 
    # ---------------------------------------------------------
    "element_outliers": [
        "CCS",                       # Ethanethiol (Sulfur, 3 atoms)
        "ClC(Cl)Cl",                 # Chloroform (Chlorine, 4 atoms)
        "C[Si](C)(C)Cl",             # Trimethylsilyl chloride (Si/Cl, 5 atoms)
        "COP(=O)(OC)OC",             # Trimethyl phosphate (Phosphorus, 8 atoms)
        "Ic1ccccc1",                 # Iodobenzene (Iodine, 7 atoms)
        "CC(C)Br",                   # 2-Bromopropane (Bromine, 4 atoms)
        "O=S(=O)(Cl)c1ccccc1",       # Benzenesulfonyl chloride (S/Cl, 9 atoms)
        "S=C=S",                     # Carbon disulfide (Sulfur, 3 atoms)
        "P#CC",                      # Methylidynephosphane (Phosphorus, 3 atoms)
        "F[Si](F)(F)F"               # Silicon tetrafluoride (Silicon, 5 atoms)
    ],

    # ---------------------------------------------------------
    # 3. Topological & Chemical Oddities (<= 9 heavy atoms)
    # These only contain C, N, O, F and fit the size limit, making 
    # them the hardest to detect. They represent highly strained, 
    # reactive, or unusual topologies that are statistically rare.
    # ---------------------------------------------------------
    "topology_outliers": [
        "C12C3C1C4C2C34",            # Prismane (High strain, 6 atoms)
        "C12C3C4C1C5C2C3C45",        # Cubane (Extreme strain, 8 atoms)
        "C12C3C1C23",                # [1.1.1]Propellane (Bridgehead carbons with inverted geometry)
        "C=C=C=C=C=C",               # Hexahexaene (Cumulene chain)
        "N#N=O",                     # Nitrous oxide (Small but distinct electronic structure)
        "C=C1C=C1",                   # Methylenecyclopropene (Highly strained exocyclic double bond)
        "F[N+](F)(F)F",              # Tetrafluorofluorammonium (Charged species)
        "O=C1C(=O)C1=O",             # Cyclopropanetrione (Highly reactive cyclic polyketone)
        "C12C(C1)C2",                # Bicyclo[1.1.0]butane (Highly strained)
        "N1(N)N(N)N1"                # Tetraazacyclobutane (Extremely unstable)
    ],

    # ---------------------------------------------------------
    # 4. "Real World" Complex Molecules
    # Massive outliers that violate both size and sometimes element 
    # constraints. Good for testing if your pipeline breaks or 
    # handles extreme extremes.
    # ---------------------------------------------------------
    "extreme_outliers": [
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # Testosterone (21 atoms)
        "CN1CCC[C@H]1c2cccnc2",               # Nicotine (12 atoms, chiral)
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", # Imatinib (37 atoms)
        "CS(=O)c1ccc(cc1)C2=C(C(=O)OC2)c3ccccc3",  # Sulindac analog/Vioxx-like (21 atoms)
        "CC(=O)NO",                           # Acetohydroxamic acid (Small but complex O-N-O)
        "c1ccc(cc1)P(c2ccccc2)c3ccccc3",      # Triphenylphosphine (19 atoms)
        "C1=CC=C2C(=C1)C=CC3=CC=CC=C32",      # Benz[a]anthracene (18 atoms)
        "COP(=S)(OC)OC=C(Cl)Cl",              # Dichlorvos (P/S/Cl mix)
        "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O", # Penicillin G (23 atoms)
        "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4" # Terfenadine (35 atoms)
    ],
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
