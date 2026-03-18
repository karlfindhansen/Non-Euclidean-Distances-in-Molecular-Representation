import numpy as np
import matplotlib.pyplot as plt

from loguru import logger

from src.datasets import QM9Dataset
from src.non_euclidean import Wasserstein, PersistentHomology, Grassmann, Riemann
from rdkit import Chem
from rdkit.Chem import Draw
import os

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

    plt.show()

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

def distance(
    qm9,
    descriptor="morgan",
    dist_type="cosine",
    top_k=6,
    save_dir="figures/qm9/distances/cosine"
):

    dist_matrix = qm9.get_distance_matrix(descriptor=descriptor, dist_type=dist_type)

    plot_distance_matrix(
        dist_matrix,
        title=f"{descriptor} {dist_type.capitalize()} Distance Matrix"
    )

    low_pairs, high_pairs = extract_extreme_pairs(dist_matrix, qm9.df, top_k=top_k)

    os.makedirs(save_dir, exist_ok=True)
    plot_pair_grid(low_pairs, title=f"Most Similar Pairs ({descriptor}/{dist_type})", save_path=os.path.join(save_dir, f"qm9_{descriptor}_most_similar_{dist_type}.png"), max_pairs=top_k)
    plot_pair_grid(high_pairs, title=f"Least Similar Pairs ({descriptor}/{dist_type})", save_path=os.path.join(save_dir, f"qm9_{descriptor}_least_similar_{dist_type}.png"), max_pairs=top_k)

def non_euclidean_distance(
    qm9,
    kind="wasserstein",
    subset_size=200,
    top_k=6,
    save_dir="figures/qm9/distances/non_euclidean",
    **kwargs
):
    """
    Computes non-Euclidean distances for a subset of molecules.
    kinds: wasserstein, persistent_homology, grassmann, riemann
    """
    frames = qm9.get_positions(subset_size=subset_size)
    df_subset = qm9.df.head(len(frames))

    kind = kind.lower()
    if kind == "wasserstein":
        metric = kwargs.get("ground_metric", "euclidean")
        dist_matrix = Wasserstein.distance_matrix(frames, metric=metric)
        title = f"Wasserstein ({metric})"
        tag = f"wasserstein_{metric}"
    elif kind in {"persistent_homology", "persistence"}:
        metric = kwargs.get("metric", "bottleneck")
        max_dim = kwargs.get("max_homology_dim", 2)
        dims = kwargs.get("homology_dims", (0, 1, 2))
        dist_matrix = PersistentHomology.distance_matrix(
            frames,
            metric=metric,
            max_homology_dim=max_dim,
            homology_dims=dims,
        )
        title = f"Persistent Homology ({metric})"
        tag = f"persistent_homology_{metric}"
    elif kind == "grassmann":
        k = kwargs.get("k", 3)
        method = kwargs.get("method", "svd")
        normalized = kwargs.get("normalized", False)
        dist_matrix = Grassmann.distance_matrix(frames, k=k, method=method, normalized=normalized)
        title = f"Grassmann (k={k}, {method})"
        tag = f"grassmann_k{k}_{method}"
    elif kind == "riemann":
        metric = kwargs.get("metric", "log-euclidean")
        normalized = kwargs.get("normalized", False)
        dist_matrix = Riemann.distance_matrix(frames, metric=metric, normalized=normalized)
        title = f"Riemann ({metric})"
        tag = f"riemann_{metric}"
    else:
        raise ValueError("kind must be one of: wasserstein, persistent_homology, grassmann, riemann")

    os.makedirs(save_dir, exist_ok=True)
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

if __name__ == "__main__":

    qm9 = QM9Dataset()
    qm9.load()
    
    descriptor = "soap"
    dist_type = "euclidean"
    top_k = 6
    save_dir = f"figures/qm9/distances/{dist_type}"

    #distance(qm9, descriptor=descriptor, dist_type=dist_type, top_k=6)

    kind = "wasserstein"
    subset_size = 2000
    
    #non_euclidean_distance(qm9, kind="wasserstein", subset_size=subset_size)
    non_euclidean_distance(qm9, kind="persistent_homology", subset_size=500, metric="bottleneck")
    # non_euclidean_distance(qm9, kind="grassmann", subset_size=200, k=3, method="svd")
    # non_euclidean_distance(qm9, kind="riemann", subset_size=200, metric="log-euclidean")

