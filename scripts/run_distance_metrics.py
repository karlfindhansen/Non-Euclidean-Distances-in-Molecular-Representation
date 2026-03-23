import os
import json

import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from rdkit import Chem
from rdkit.Chem import Draw

from sklearn.metrics import pairwise_distances

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms

from src.datasets import QM9Dataset, MaterialsProject
from src.helper_functions import get_distances

def plot_distance_matrix(dist_matrix, title="Distance Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    
    plt.imshow(dist_matrix, interpolation='nearest')
    plt.colorbar(label='Distance')
    
    plt.title(title)
    plt.xlabel("Molecule index")
    plt.ylabel("Molecule index")
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    #plt.show()

def extract_extreme_pairs(dist_matrix, df, top_k=5):
    n = dist_matrix.shape[0]
    if n == 0:
        raise ValueError("Empty distance matrix.")

    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = dist_matrix[i_idx, j_idx]

    order = np.argsort(dists)
    low_idx = order[:top_k]
    high_idx = order[-top_k:][::-1]

    def build_pairs(sel_idx):
        pairs = []
        for k in sel_idx:
            i = int(i_idx[k])
            j = int(j_idx[k])
            pairs.append({
                "i": i,
                "j": j,
                "mol_id_i": df["mol_id"][i],
                "mol_id_j": df["mol_id"][j],
                "smiles_i": df["canonical_smiles"][i],
                "smiles_j": df["canonical_smiles"][j],
                "distance": float(dists[k]),
            })
        return pairs

    return build_pairs(low_idx), build_pairs(high_idx)

def extract_extreme_pairs_materials(dist_matrix, df, top_k=5):
    n = dist_matrix.shape[0]
    if n == 0:
        raise ValueError("Empty distance matrix.")

    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = dist_matrix[i_idx, j_idx]

    order = np.argsort(dists)
    low_idx = order[:top_k]
    high_idx = order[-top_k:][::-1]

    def build_pairs(sel_idx):
        pairs = []
        for k in sel_idx:
            i = int(i_idx[k])
            j = int(j_idx[k])
            pairs.append({
                "i": i,
                "j": j,
                "material_id_i": df["material_id"][i],
                "material_id_j": df["material_id"][j],
                "formula_i": df["formula_pretty"][i],
                "formula_j": df["formula_pretty"][j],
                "distance": float(dists[k]),
            })
        return pairs

    return build_pairs(low_idx), build_pairs(high_idx)

def plot_pair_grid(pairs, title, save_path=None, max_pairs=6):
    if not pairs:
        logger.warning(f"No pairs to plot for {title}")
        return

    pairs = pairs[:max_pairs]
    mols = []
    legends = []
    for p in pairs:
        mol_i = Chem.MolFromSmiles(p["smiles_i"])
        mol_j = Chem.MolFromSmiles(p["smiles_j"])
        mols.extend([mol_i, mol_j])
        legends.extend([
            f"{p['mol_id_i']}",
            f"{p['mol_id_j']}\nd={p['distance']:.3f}"
        ])

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(250, 200),
        legends=legends
    )

    plt.figure(figsize=(6, 3 * len(pairs)))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        logger.info(f"Saved pair plot to {save_path}")

    plt.show()

def save_pairs_csv(pairs, save_path):
    if not pairs:
        logger.warning(f"No pairs to save: {save_path}")
        return
    import polars as pl
    pl.DataFrame(pairs).write_csv(save_path)
    logger.info(f"Saved pairs to {save_path}")

def plot_pairs_table(pairs, title, save_path, max_pairs=10):
    if not pairs:
        logger.warning(f"No pairs to plot for {title}")
        return
    rows = pairs[:max_pairs]
    columns = list(rows[0].keys())
    cell_text = [[str(r.get(c, "")) for c in columns] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved table plot to {save_path}")

def plot_material_pairs(pairs, df, title, save_path, max_pairs=6):
    if not pairs:
        logger.warning(f"No pairs to plot for {title}")
        return

    pairs = pairs[:max_pairs]
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])

    adaptor = AseAtomsAdaptor()

    for row_idx, p in enumerate(pairs):
        left_idx = int(p["i"])
        right_idx = int(p["j"])

        left_struct = Structure.from_dict(json.loads(df["raw_structure"][left_idx]))
        right_struct = Structure.from_dict(json.loads(df["raw_structure"][right_idx]))

        left_atoms = adaptor.get_atoms(left_struct)
        right_atoms = adaptor.get_atoms(right_struct)

        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        plot_atoms(left_atoms, ax_left)
        plot_atoms(right_atoms, ax_right)

        ax_left.set_axis_off()
        ax_right.set_axis_off()

        ax_left.set_title(f"{p['material_id_i']} ({p['formula_i']})")
        ax_right.set_title(f"{p['material_id_j']} ({p['formula_j']})")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved materials plot to {save_path}")

def distance(
    qm9,
    descriptor="morgan",
    dist_type="cosine",
    top_k=6,
    save_dir="figures/qm9/distances/cosine"
):

    dist_matrix = qm9.get_distance_matrix(descriptor=descriptor, dist_type=dist_type)

    save_dir = os.path.join(save_dir, f"n{dist_matrix.shape[0]}")
    os.makedirs(save_dir, exist_ok=True)

    plot_distance_matrix(
        dist_matrix,
        title=f"{descriptor} {dist_type.capitalize()} Distance Matrix"
    )

    low_pairs, high_pairs = extract_extreme_pairs(dist_matrix, qm9.df, top_k=top_k)

    plot_pair_grid(low_pairs, title=f"Most Similar Pairs ({descriptor}/{dist_type})", save_path=os.path.join(save_dir, f"qm9_{descriptor}_most_similar_{dist_type}.png"), max_pairs=top_k)
    plot_pair_grid(high_pairs, title=f"Least Similar Pairs ({descriptor}/{dist_type})", save_path=os.path.join(save_dir, f"qm9_{descriptor}_least_similar_{dist_type}.png"), max_pairs=top_k)

def distance_materials(
    mp,
    descriptor="soap",
    dist_type="euclidean",
    top_k=6,
    save_dir="figures/materials/distances/euclidean"
):
    descriptor = descriptor.lower()
    aliases = {
        "soap": "soap_embedding",
        "acsf": "acsf_embedding",
    }
    col = aliases.get(descriptor, descriptor)
    if col not in mp.df.columns:
        raise ValueError(f"Unknown descriptor '{descriptor}' for MaterialsProject.")

    df = mp.df.filter(mp.df[col].is_not_null())
    if df.is_empty():
        raise ValueError(f"No valid rows for descriptor '{descriptor}'.")

    X = np.stack(df[col].to_list())
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    dist_matrix = pairwise_distances(X, metric=dist_type)

    save_dir = os.path.join(save_dir, f"n{dist_matrix.shape[0]}")
    os.makedirs(save_dir, exist_ok=True)
    plot_distance_matrix(
        dist_matrix,
        title=f"Materials {descriptor} {dist_type.capitalize()} Distance Matrix",
        save_path=os.path.join(save_dir, f"materials_{descriptor}_{dist_type}_matrix.png")
    )

    low_pairs, high_pairs = extract_extreme_pairs_materials(dist_matrix, df, top_k=top_k)
    plot_material_pairs(
        low_pairs,
        df,
        title=f"Most Similar Materials ({descriptor}/{dist_type})",
        save_path=os.path.join(save_dir, f"materials_{descriptor}_most_similar_{dist_type}.png"),
        max_pairs=top_k,
    )

def non_euclidean_distance(
    qm9,
    subset_size=200,
    top_k=6,
    save_dir="figures/qm9/distances/non_euclidean",
    include_ph=True,
):
    """
    Computes non-Euclidean distances for a subset of molecules using get_distances.
    """
    frames = qm9.get_positions(subset_size=subset_size)
    df_subset = qm9.df.head(len(frames))

    dist_matrices = get_distances(
        frames,
        dataset=f"QM9/distance_matrices_n{len(frames)}",
        include_ph=include_ph,
    )

    save_dir = os.path.join(save_dir, f"n{len(frames)}")
    os.makedirs(save_dir, exist_ok=True)

    name_to_title = {
        "grassmann": "Grassmann",
        "euclidean_riemann": "Riemann (log-euclidean)",
        "affine_riemann": "Riemann (affine-invariant)",
        "wasserstein": "Wasserstein",
        "ph_bottleneck": "Persistent Homology (bottleneck)",
        "ph_sliced_wasserstein": "Persistent Homology (sliced-wasserstein)",
    }

    for name, dist_matrix in dist_matrices.items():
        title = name_to_title.get(name, name)
        tag = name
        plot_distance_matrix(
            dist_matrix,
            title=f"{title} Distance Matrix",
            save_path=os.path.join(save_dir, f"qm9_{tag}_matrix.png")
        )

        low_pairs, high_pairs = extract_extreme_pairs(dist_matrix, df_subset, top_k=top_k)
        plot_pair_grid(
            low_pairs,
            title=f"Most Similar Pairs ({title})",
            save_path=os.path.join(save_dir, f"qm9_{tag}_most_similar.png"),
            max_pairs=top_k
        )
        plot_pair_grid(
            high_pairs,
            title=f"Least Similar Pairs ({title})",
            save_path=os.path.join(save_dir, f"qm9_{tag}_least_similar.png"),
            max_pairs=top_k
        )

def _build_mp_frames(mp_df):
    adaptor = AseAtomsAdaptor()
    frames = []
    for struct_json in mp_df["raw_structure"]:
        struct = Structure.from_dict(json.loads(struct_json))
        frames.append(adaptor.get_atoms(struct))
    return frames

def non_euclidean_distance_materials(
    mp,
    subset_size=200,
    top_k=6,
    save_dir="figures/materials/distances/non_euclidean",
    include_ph=True,
):
    df_subset = mp.df.head(min(subset_size, mp.df.height))
    frames = _build_mp_frames(df_subset)

    dist_matrices = get_distances(
        frames,
        dataset=f"Materials Project/distance_matrices_n{len(frames)}",
        include_ph=include_ph,
    )

    save_dir = os.path.join(save_dir, f"n{len(frames)}")
    os.makedirs(save_dir, exist_ok=True)

    name_to_title = {
        "grassmann": "Grassmann",
        "euclidean_riemann": "Riemann (log-euclidean)",
        "affine_riemann": "Riemann (affine-invariant)",
        "wasserstein": "Wasserstein",
        "ph_bottleneck": "Persistent Homology (bottleneck)",
        "ph_sliced_wasserstein": "Persistent Homology (sliced-wasserstein)",
    }

    for name, dist_matrix in dist_matrices.items():
        title = name_to_title.get(name, name)
        tag = name
        plot_distance_matrix(
            dist_matrix,
            title=f"{title} Distance Matrix",
            save_path=os.path.join(save_dir, f"materials_{tag}_matrix.png")
        )

        low_pairs, high_pairs = extract_extreme_pairs_materials(dist_matrix, df_subset, top_k=top_k)
        plot_material_pairs(
            low_pairs,
            df_subset,
            title=f"Most Similar Materials ({title})",
            save_path=os.path.join(save_dir, f"materials_{tag}_most_similar.png"),
            max_pairs=top_k,
        )

        plot_material_pairs(
            high_pairs,
            df_subset,
            title=f"Least Similar Pairs ({title})",
            save_path=os.path.join(save_dir, f"materials_{tag}_least_similar.png"),
            max_pairs=top_k
        )

if __name__ == "__main__":

    qm9 = QM9Dataset()
    qm9.load()
    
    descriptor = "soap"
    dist_type = "euclidean"
    top_k = 6
    save_dir = f"figures/qm9/distances/{dist_type}"

    #distance(qm9, descriptor=descriptor, dist_type=dist_type, top_k=6)

    subset_size = 2000

    # Generates all non-euclidean distances via get_distances(...)
    #non_euclidean_distance(qm9, subset_size=2000, include_ph=True)

    mp = MaterialsProject()
    mp.load()
    distance_materials(mp, descriptor="soap", dist_type="euclidean", top_k=6)
    non_euclidean_distance_materials(mp, subset_size=1000, include_ph=False)
