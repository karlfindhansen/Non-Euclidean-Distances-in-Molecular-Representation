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


def create_chemiscope_viewer(df, dist_matrix, labels, reduction_method='t-SNE'):
    print("Running " + reduction_method + " dimensionality reduction...")

    if reduction_method == 't-SNE':
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42, perplexity=30)
        coords = tsne.fit_transform(dist_matrix)
    elif reduction_method == 'UMAP':
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
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")

    print("Converting Pymatgen structures to ASE Atoms for Chemiscope...")
    frames = []
    adaptor = AseAtomsAdaptor()
    
    for struct_json in df["raw_structure"]:
        struct = Structure.from_dict(json.loads(struct_json))
        atoms = adaptor.get_atoms(struct)
        frames.append(atoms)

    print("Assembling properties for Chemiscope...")
  
    properties = {
        f"{reduction_method}_1": coords[:, 0],
        f"{reduction_method}_2": coords[:, 1],
        "Cluster": labels.astype(int),
        "Formula": df["formula_pretty"].to_list(),
        "Band_Gap": df["band_gap"].to_list(),
        "Energy_per_Atom": df["energy_per_atom"].to_list(),
        "Is_Metal": df["is_metal"].to_list(),
        "crystal_system": df["crystal_system"].to_list(),
        "Density": df["density"].to_list(),
        "Space_Group": df["space_group"].to_list(),
        "energy_above_hull": df["energy_above_hull"].to_list(),
        "formation_energy_per_atom": df["formation_energy_per_atom"].to_list(),
        "volume": df["volume"].to_list(),
        "num_sites": df["num_sites"].to_list(),
        "max_en_diff":df["max_en_diff"].to_list(),
    }

    settings = {
        "map": {
            "x": {"property": f"{reduction_method}_1"},
            "y": {"property": f"{reduction_method}_2"},
            "color": {"property": "Cluster"},
            #"symbol": "circle",
            "size": {"factor": 35}
        },
        "structure": [
            {"keepOrientation": True, "supercell": [2, 2, 2]}
        ]
    }

    print("Generating Chemiscope widget...")
    if hasattr(chemiscope, "write_html"):
        output_html = f"materials_{reduction_method}_clustering.html"
        chemiscope.write_html(
            output_html,
            frames=frames,
            properties=properties,
            settings=settings,
            title=f"Materials Project - SOAP {reduction_method} Clustering",
        )
        print(f"Saved interactive viewer to: {output_html}")
        _open_in_browser(output_html)
        return chemiscope.show(frames=frames, properties=properties, settings=settings)

    output_json = f"materials_{reduction_method}_clustering.json"
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
        metadata={"name": f"Materials Project - SOAP {reduction_method} Clustering"},
    )
    print(f"Saved Chemiscope input to: {output_json}")
    viewer = chemiscope.show_input(output_json)
    viewer_url = getattr(viewer, "url", None)
    if viewer_url:
        _open_in_browser(viewer_url)
    else:
        print(
            "If the viewer does not open automatically, run "
            f"`chemiscope show materials_{reduction_method}_clustering.json`."
        )
    return viewer


if __name__ == '__main__':
    from src.datasets import QM9Dataset
    dataset = QM9Dataset()
    df = dataset.load()
    frames = dataset.get_positions()
    matrices = get_distances(frames)
    
