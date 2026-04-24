import os
import json
import numpy as np
import polars as pl
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_diamond_coords():
    return [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0], 
            [0.25,0.25,0.25], [0.25,0.75,0.75], [0.75,0.25,0.75], [0.75,0.75,0.25]]

def get_fcc_coords():
    return [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]

def get_bcc_coords():
    return [[0,0,0], [0.5,0.5,0.5]]

# =====================================================================
# 1. Define the Outlier Generating Functions
# =====================================================================
def generate_materials_outliers():
    outliers = []

    # ---------------------------------------------------------
    # 1. Size Outliers (Massive Supercells)
    # Stresses the O(N) or O(N^2) scaling of SOAP and MACE.
    # Standard MP cells are usually <50 atoms. These are 200+.
    # ---------------------------------------------------------
    si_base = Structure(Lattice.cubic(5.43), ["Si"]*8, get_diamond_coords())
    massive_supercell = si_base.copy()
    massive_supercell.make_supercell([3, 3, 3]) # 216 atoms
    outliers.append(("size_outliers", massive_supercell, {"band_gap": 1.1, "is_metal": False}))

    # ---------------------------------------------------------
    # 2. Element Outliers (Actinides, Noble Gases, Rare Earths)
    # Often completely missing from pre-trained generic potentials
    # or poorly parameterized in ACSF/SOAP weighting.
    # ---------------------------------------------------------
    solid_xe = Structure(Lattice.cubic(6.2), ["Xe"]*4, get_fcc_coords())
    outliers.append(("element_outliers", solid_xe, {"band_gap": 9.3, "is_metal": False}))

    uranium_bcc = Structure(Lattice.cubic(3.48), ["U"]*2, get_bcc_coords())
    outliers.append(("element_outliers", uranium_bcc, {"band_gap": 0.0, "is_metal": True}))
    
    # Purely organic crystal proxy (extreme C/H/O/N ratio for a material)
    # A simple mock of solid methane
    solid_ch4 = Structure(
        Lattice.cubic(5.8), 
        ["C", "H", "H", "H", "H"], 
        [[0,0,0], [0.1,0.1,0.1], [-0.1,-0.1,0.1], [0.1,-0.1,-0.1], [-0.1,0.1,-0.1]]
    )
    outliers.append(("element_outliers", solid_ch4, {"band_gap": 6.0, "is_metal": False}))

    # ---------------------------------------------------------
    # 3. Topological & Geometric Oddities
    # Breaks neighborhood radii (r_cut).
    # ---------------------------------------------------------
    # A. 2D Material with massive vacuum. Will cause isolated graphs in MACE.
    graphene_large_c = Structure(
        Lattice.from_parameters(2.46, 2.46, 50.0, 90, 90, 120),
        ["C", "C"], [[0,0,0], [1/3, 2/3, 0]]
    )
    outliers.append(("topology_outliers", graphene_large_c, {"band_gap": 0.0, "is_metal": True}))

    # B. Ultra-low density "Ghost" Lattice. Distances > r_cut (e.g. 10 Angstroms).
    # Causes empty SOAP vectors or NaN Coulomb matrices.
    ghost_lattice = Structure(Lattice.cubic(20.0), ["Fe"], [[0,0,0]])
    outliers.append(("topology_outliers", ghost_lattice, {"band_gap": 0.0, "is_metal": True}))

    # C. Unphysically compressed lattice. Atoms are too close (e.g., 1.0 Angstrom).
    # Causes explosive gradients in MACE and extreme Coulomb repulsions.
    compressed_fe = Structure(Lattice.cubic(1.2), ["Fe"]*2, get_bcc_coords())
    outliers.append(("topology_outliers", compressed_fe, {"energy_above_hull": 50.0})) # Highly unstable

    # ---------------------------------------------------------
    # 4. Compositional Complexity (High Entropy / Many Elements)
    # Stresses the species embedding layers.
    # ---------------------------------------------------------
    # 5-element complex mock structure
    hea_mock = Structure(
        Lattice.cubic(4.0),
        ["Fe", "Ni", "Co", "Cr", "Mn"],
        [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.5,0.5]]
    )
    outliers.append(("extreme_outliers", hea_mock, {"is_metal": True}))

    return outliers

# =====================================================================
# 2. Processor (Mimics your MaterialsProject._process_doc)
# =====================================================================
def process_synthetic_structure(category: str, struct: Structure, mock_props: dict) -> dict:
    """Processes a pymatgen Structure into your exact DataFrame schema."""
    
    # Calculate baseline chemistry features
    formula = struct.composition.reduced_formula
    comp = Composition(formula)
    anon_formula = comp.anonymized_formula
    
    # Calculate EN difference
    ens = [el.X for el in comp.elements if getattr(el, 'X', None) and el.X > 0]
    max_en_diff = (max(ens) - min(ens)) if len(ens) > 1 else 0.0

    # Calculate Symmetry
    try:
        sga = SpacegroupAnalyzer(struct)
        sym_dataset = sga.get_symmetry_dataset()
        c_sys = sga.get_crystal_system()
        sg = sga.get_space_group_symbol()
        pearson = sga.get_pearson_symbol()
        wyckoff_seq = sym_dataset["wyckoffs"] if sym_dataset else []
        wyckoff_str = "_".join(sorted(list(set(wyckoff_seq))))
    except Exception:
        c_sys, sg, pearson, wyckoff_str = "Unknown", "Unknown", "Unknown", "Unknown"

    true_prototype = f"{anon_formula}_{pearson}_{sg}_{wyckoff_str}"
    
    # Calculate simple bond length approximations
    try:
        neighbors = struct.get_all_neighbors(r=5.0)
        min_dists = [min([nn.nn_distance for nn in nlist]) for nlist in neighbors if nlist]
        avg_bond = float(np.mean(min_dists)) if min_dists else None
        max_bond = float(np.max(min_dists)) if min_dists else None
    except Exception:
        avg_bond, max_bond = None, None

    lat = struct.lattice

    # Build the dictionary
    data = {
        "material_id": f"synthetic-{category}-{formula}",
        "formula_pretty": formula,
        "anonymized_formula": anon_formula,
        "structural_prototype": true_prototype,
        "max_en_diff": float(max_en_diff),
        "energy_per_atom": mock_props.get("energy_per_atom", 0.0),
        "formation_energy_per_atom": mock_props.get("formation_energy_per_atom", 0.0),
        "band_gap": mock_props.get("band_gap", 0.0),
        "is_metal": mock_props.get("is_metal", False),
        "raw_structure": json.dumps(struct.as_dict()),
        "crystal_system": c_sys,
        "space_group": sg,
        "pearson_symbol": pearson,
        "density": float(struct.density),
        "a": float(lat.a),
        "b": float(lat.b),
        "c": float(lat.c),
        "alpha": float(lat.alpha),
        "beta": float(lat.beta),
        "gamma": float(lat.gamma),
        "volume": float(struct.volume),
        "num_sites": int(len(struct)),
        "energy_above_hull": mock_props.get("energy_above_hull", 0.0),
        "avg_bond_length": avg_bond,
        "max_bond_length": max_bond,
        
        # Meta flags for your dataframe
        "outlier_category": category,
        "is_injected": 1
    }
    return data

if __name__ == '__main__':
    raw_outliers = generate_materials_outliers()
    
    processed_data = []
    for category, struct, mock_props in raw_outliers:
        processed_row = process_synthetic_structure(category, struct, mock_props)
        processed_data.append(processed_row)

    df = pl.DataFrame(processed_data)

    save_dir = "data/Materials Project/outliers"
    file_path = os.path.join(save_dir, "synthetic_materials_outliers.parquet")

    os.makedirs(save_dir, exist_ok=True)
    df.write_parquet(file_path)

    print(f"Successfully generated and saved {df.height} material outliers to: {file_path}")