import os
import chemiscope
import json
import webbrowser
import kmedoids
import polars as pl
import numpy as np

from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from sklearn.manifold import TSNE, Isomap, MDS
from geomstats.learning.pca import TangentPCA
from sklearn.decomposition import PCA
from typing import Sequence, Optional, Iterable
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from umap import UMAP
from ase import Atoms

from src.non_euclidean import Grassmann, Riemann, Wasserstein, PersistentHomology
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def get_structures(df, mol_id_list = None):
    # Extract both the IDs and the SMILES strings
    if mol_id_list is None:
        mol_ids = df["mol_id"].to_list()
    else:
        mol_ids = mol_id_list
        df = df.filter(pl.col("mol_id").is_in(mol_id_list))
    
    smiles_list = df["canonical_smiles"].to_list()
    
    structures = []  
    valid_indices = [] # We need to track this (see warning below)
    
    for i, (mol_id, s) in enumerate(zip(mol_ids, smiles_list)):
        if s is None:
            print(f"Skipping {mol_id}: SMILES is missing.")
            continue
            
        mol = Chem.MolFromSmiles(s)
        if mol is None: 
            print(f"Skipping {mol_id}: RDKit could not parse SMILES.")
            continue
            
        mol = Chem.AddHs(mol)

        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0: 
            print(f"Skipping {mol_id}: Failed to generate 3D conformer.")
            continue

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
        
        atoms = Atoms(numbers=numbers, positions=positions)
        structures.append(atoms)
        valid_indices.append(i)
        
    return structures, valid_indices


def align_frames_to_dist_matrix(
    frames: Sequence[Atoms],
    dist_matrix: Optional[np.ndarray] = None,
    return_matrix: bool = False,
) -> list[Atoms]:
    """
    Ensures frames match a precomputed pairwise distance matrix size.
    If more frames are provided than the matrix size, truncate deterministically.
    """
    aligned_frames = list(frames)
    if dist_matrix is None:
        return aligned_frames if not return_matrix else (aligned_frames, dist_matrix)

    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n_frames = len(aligned_frames)
    n_matrix = int(dist_matrix.shape[0])
    if n_frames != n_matrix:
        min_n = min(n_frames, n_matrix)
        logger.warning(
            "Aligning frames and dist_matrix sizes: "
            f"frames={n_frames}, dist_matrix={n_matrix}. Using first {min_n} entries."
        )
        aligned_frames = aligned_frames[:min_n]
        dist_matrix = dist_matrix[:min_n, :min_n]

    return aligned_frames if not return_matrix else (aligned_frames, dist_matrix)

def get_distances(frames, frames_ph=None, dataset = 'QM9', include_ph=True):
    expected_n = len(frames)
    data_dir = f'data/{dataset}/distance_matrices_n{expected_n}'
    os.makedirs(data_dir, exist_ok=True)
    print(os.listdir(data_dir))

    matrix_tasks = {
        'grassmann': {
            'path': f'{data_dir}/dist_matrix_grassmann.npy',
            'compute': lambda: Grassmann.distance_matrix(frames)
        },
        'euclidean_riemann': {
            'path': f'{data_dir}/dist_matrix_euclidean_riemann.npy',
            'compute': lambda: Riemann.distance_matrix(frames, metric='log-euclidean')
        },
        'affine_riemann': {
            'path': f'{data_dir}/dist_matrix_affine_riemann.npy',
            'compute': lambda: Riemann.distance_matrix(frames, metric='affine-invariant')
        },
        'wasserstein': {
            'path': f'{data_dir}/dist_matrix_wasserstein.npy',
            'compute': lambda: Wasserstein.distance_matrix(frames)
        },
        'ph_bottleneck': {
            'path': f'{data_dir}/persistent_dist_matrix_bottleneck.npy',
            'compute': lambda: PersistentHomology.distance_matrix(frames, metric="bottleneck")
        },
        'ph_sliced_wasserstein': {
            'path': f'{data_dir}/persistent_dist_matrix_sw.npy',
            'compute': lambda: PersistentHomology.distance_matrix(frames, metric="sliced_wasserstein")
        }
    }

    matrices = {}

    for name, task in matrix_tasks.items():
        file_path = task['path']
        
        if os.path.exists(file_path):
            logger.info(f"Loading {name} distance matrix...")
            matrices[name] = np.load(file_path)
        else:
            if not include_ph and name.startswith('ph_'):
                logger.warning("Skipping PH task")
                continue
            logger.info(f"Computing {name} distance matrix...")
            matrices[name] = task['compute']()
            np.save(file_path, matrices[name])

        mat = np.asarray(matrices[name])
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{name} distance matrix must be square. Got shape {mat.shape}."
            )
        if mat.shape[0] != expected_n:
            raise ValueError(
                f"{name} distance matrix size mismatch: expected ({expected_n}, {expected_n}), got {mat.shape}."
            )

    logger.success("✓ All distance matrices are ready!")
    
    return matrices


def find_best_kmedoids_k(
    dist_matrix: np.ndarray,
    k_range: Iterable[int] = range(2, 15),
    random_state: int = 42,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate K-Medoids clustering over a range of k using:
    - inertia (sum of distances to assigned medoids)
    - silhouette score (precomputed distances)
    - Calinski-Harabasz score (on feature_matrix if provided, else on dist_matrix)
    """
    dist_matrix = np.asarray(dist_matrix)
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square matrix.")

    n = dist_matrix.shape[0]
    k_list = [int(k) for k in k_range if 2 <= int(k) <= n - 1]
    if not k_list:
        raise ValueError("k_range must contain values in [2, n-1].")

    results = {"k": [], "inertia": [], "silhouette": [], "ch": []}

    use_features_for_ch = feature_matrix is not None
    if feature_matrix is None:
        logger.warning(
            "feature_matrix is None; Calinski-Harabasz will be computed on dist_matrix "
            "(treated as features). Provide feature_matrix for a more meaningful CH score."
        )

    for k in k_list:
        model = kmedoids.KMedoids(n_clusters=k, metric="precomputed", random_state=random_state)
        labels = model.fit_predict(dist_matrix)
        medoid_indices = model.medoid_indices_

        inertia = float(
            sum(dist_matrix[i, medoid_indices[labels[i]]] for i in range(n))
        )

        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

        if use_features_for_ch:
            if feature_matrix.shape[0] != n:
                raise ValueError(
                    "feature_matrix must have the same number of rows as dist_matrix."
                )
            ch = float(calinski_harabasz_score(feature_matrix, labels))
        else:
            ch = float(calinski_harabasz_score(dist_matrix, labels))

        results["k"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil)
        results["ch"].append(ch)

    best_k = {
        "inertia": results["k"][int(np.argmin(results["inertia"]))],
        "silhouette": results["k"][int(np.argmax(results["silhouette"]))],
        "ch": results["k"][int(np.argmax(results["ch"]))],
    }

    return {"results": results, "best_k": best_k}

def _open_in_browser(path_or_url):
    try:
        if not path_or_url:
            return
        if isinstance(path_or_url, str) and (
            path_or_url.startswith("http://") or path_or_url.startswith("https://")
        ):
            webbrowser.open(path_or_url)
            return
        webbrowser.open(Path(path_or_url).resolve().as_uri())
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")


def _build_chemiscope_frames(df: pl.DataFrame, qm9_seed: int = 40, qm9_invariant: bool = True):
    """Build ASE frames for either materials (raw_structure) or QM9 (smiles)."""
    if "raw_structure" in df.columns:
        frames = []
        adaptor = AseAtomsAdaptor()
        for struct_json in df["raw_structure"]:
            struct = Structure.from_dict(json.loads(struct_json))
            frames.append(adaptor.get_atoms(struct))
        return frames, list(range(df.height)), "materials"

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles" if "smiles" in df.columns else None
    if smiles_col is None:
        raise ValueError("DataFrame must contain either 'raw_structure' or a SMILES column.")

    # Local import avoids circular imports at module load time.
    from src.datasets import QM9Dataset

    frames = []
    valid_indices = []
    for i, row in enumerate(df.iter_rows(named=True)):
        smiles = row.get(smiles_col)
        mol_id = row.get("mol_id")
        if not smiles:
            continue

        mol = QM9Dataset._embed_molecule(smiles=smiles, seed=qm9_seed, invariant=qm9_invariant)
        if mol is None:
            logger.warning(f"QM9 embedding failed for mol_id={mol_id} smiles={smiles}")
            continue

        conf = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = conf.GetPositions()
        charges = np.array(
            [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
            dtype=np.float64,
        )

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_initial_charges(charges)
        atoms.arrays["partial_charge"] = charges
        atoms.arrays["mass"] = atoms.get_masses()
        if mol_id is not None:
            atoms.info["mol_id"] = mol_id
        atoms.info["smiles"] = smiles
        frames.append(atoms)
        valid_indices.append(i)

    return frames, valid_indices, "qm9"


def create_chemiscope_viewer(df, dist_matrix, labels, reduction_method='t-SNE'):
    print("Running " + reduction_method + " dimensionality reduction...")

    dist_matrix = np.asarray(dist_matrix)
    labels = np.asarray(labels)
    if labels.shape[0] != df.height:
        raise ValueError(
            f"labels length ({labels.shape[0]}) must match dataframe rows ({df.height})."
        )

    print("Converting structures/molecules to ASE Atoms for Chemiscope...")
    frames, valid_indices, dataset_kind = _build_chemiscope_frames(df)
    if not frames:
        raise ValueError("No valid structures/molecules could be converted for Chemiscope.")

    # Keep dataframe/labels/distances aligned with successfully built frames.
    if len(valid_indices) != df.height:
        df = df[valid_indices]
        labels = labels[valid_indices]
        if dist_matrix.ndim == 2:
            if dist_matrix.shape[0] == dist_matrix.shape[1]:
                dist_matrix = dist_matrix[np.ix_(valid_indices, valid_indices)]
            else:
                dist_matrix = dist_matrix[valid_indices]

    if dist_matrix.shape[0] != len(frames):
        min_n = min(dist_matrix.shape[0], len(frames), labels.shape[0], df.height)
        logger.warning(
            "Chemiscope alignment mismatch; truncating to first "
            f"{min_n} entries (frames={len(frames)}, dist={dist_matrix.shape}, labels={labels.shape[0]}, df={df.height})."
        )
        frames = frames[:min_n]
        labels = labels[:min_n]
        df = df.head(min_n)
        if dist_matrix.ndim == 2 and dist_matrix.shape[0] == dist_matrix.shape[1]:
            dist_matrix = dist_matrix[:min_n, :min_n]
        else:
            dist_matrix = dist_matrix[:min_n]

    if reduction_method == 't-SNE':
        if dist_matrix.shape[1] != dist_matrix.shape[0]:
            tsne = TSNE(n_components=2, random_state=42)
        else:
            tsne = TSNE(n_components=2, metric='precomputed',random_state=42)
        coords = tsne.fit_transform(dist_matrix)
    elif reduction_method in ['UMAP', 'umap']:
        if dist_matrix.shape[1] != dist_matrix.shape[0]:
            reducer = UMAP(metric='euclidean', random_state=42)
        else:
            reducer = UMAP(metric='precomputed', random_state=42)
        coords = reducer.fit_transform(dist_matrix)
    elif reduction_method == 'PCA':
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(dist_matrix)
    elif reduction_method == 'ISOMAP':
        isomap = Isomap(n_components=2, metric='precomputed')
        coords = isomap.fit_transform(dist_matrix)
    elif reduction_method == 'MDS':
        mds = MDS(n_components=2, metric='precomputed', n_init=4)
        coords = mds.fit_transform(dist_matrix)
    elif reduction_method == 'PGA':
        pga = TangentPCA(n_components=2)
        coords = pga.fit_transform(dist_matrix)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")

    print("Assembling properties for Chemiscope...")

    properties = {
        f"{reduction_method}_1": coords[:, 0],
        f"{reduction_method}_2": coords[:, 1],
        "Cluster": labels.astype(int),
    }

    if dataset_kind == "materials":
        materials_cols = {
            "Formula": "formula_pretty",
            "Band_Gap": "band_gap",
            "Energy_per_Atom": "energy_per_atom",
            "Is_Metal": "is_metal",
            "crystal_system": "crystal_system",
            "Density": "density",
            "Space_Group": "space_group",
            "energy_above_hull": "energy_above_hull",
            "formation_energy_per_atom": "formation_energy_per_atom",
            "volume": "volume",
            "num_sites": "num_sites",
            "max_en_diff": "max_en_diff",
            "avg_bond_length": "avg_bond_length",
            "max_bond_length": "max_bond_length",
            "material_id": "material_id",
        }
        for prop_name, col_name in materials_cols.items():
            if col_name in df.columns:
                properties[prop_name] = df[col_name].to_list()
    else:
        qm9_cols = {
            "mol_id": "mol_id",
            "Formula": "formula",
            "smiles": "canonical_smiles" if "canonical_smiles" in df.columns else "smiles",
            "num_atoms": "num_atoms",
            "coordination": "coordination",
            "structure_class": "structure_class",
            "functional_groups": "functional_groups",
            "gap": "gap",
            "homo": "homo",
            "lumo": "lumo",
            "mu": "mu",
            "alpha": "alpha",
            "avg_bond_length": "avg_bond_length",
        }
        for prop_name, col_name in qm9_cols.items():
            if col_name in df.columns:
                properties[prop_name] = df[col_name].to_list()

    settings = {
        "map": {
            "x": {"property": f"{reduction_method}_1"},
            "y": {"property": f"{reduction_method}_2"},
            "color": {"property": "Cluster"},
            #"symbol": "circle",
            "size": {"factor": 35}
        },
        "structure": [{"keepOrientation": True}],
    }
    if dataset_kind == "materials":
        settings["structure"][0]["supercell"] = [2, 2, 2]

    print("Generating Chemiscope widget...")
    title_prefix = "Materials Project" if dataset_kind == "materials" else "QM9"
    output_prefix = "materials" if dataset_kind == "materials" else "qm9"

    if hasattr(chemiscope, "write_html"):
        output_html = f"{output_prefix}_{reduction_method}_clustering.html"
        chemiscope.write_html(
            output_html,
            frames=frames,
            properties=properties,
            settings=settings,
            title=f"{title_prefix} - {reduction_method} Clustering",
        )
        print(f"Saved interactive viewer to: {output_html}")
        _open_in_browser(output_html)
        return chemiscope.show(frames=frames, properties=properties, settings=settings)

    output_json = f"{output_prefix}_{reduction_method}_clustering.json"
    if not hasattr(chemiscope, "write_input"):
        raise AttributeError(
            "chemiscope does not provide write_html or write_input; "
            "please upgrade/downgrade chemiscope to a supported version."
        )

    chemiscope.write_input(
        output_json,
        structures=frames,
        properties=properties,
        settings=settings,
        metadata={"name": f"{title_prefix} - {reduction_method} Clustering"},
    )
    print(f"Saved Chemiscope input to: {output_json}")
    viewer = chemiscope.show_input(output_json)
    viewer_url = getattr(viewer, "url", None)
    if viewer_url:
        _open_in_browser(viewer_url)
    else:
        print(
            "If the viewer does not open automatically, run "
            f"`chemiscope show {output_prefix}_{reduction_method}_clustering.json`."
        )
    return viewer


if __name__ == '__main__':
    from src.datasets import QM9Dataset
    dataset = QM9Dataset()
    df = dataset.load()
    frames = dataset.get_positions()
    matrices = get_distances(frames)
    
