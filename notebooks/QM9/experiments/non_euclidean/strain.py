import math
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from umap import UMAP

# Assuming rdkit, REMatch, Wasserstein, Grassmann, and Riemann are imported globally
from rdkit import Chem
from src.non_euclidean import REMatch, Wasserstein, Grassmann, Riemann


def categorize_pure_carbocycle(smiles: str) -> str:
    """
    Parses a SMILES string. 
    Returns '3-ring', '6-ring', or 'Acyclic' ONLY if the structure is entirely Carbon/Hydrogen.
    Rejects heterocycles (rings with O, N, F, etc.) and mixed structures.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or "O" in smiles or "N" in smiles or "F" in smiles:
        return "Other"
    
    ring_info = mol.GetRingInfo().AtomRings()
    if not ring_info:
        return "Acyclic"
        
    for ring in ring_info:
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() != 6:
                return "Heterocycle"
                
    sizes = set(len(ring) for ring in ring_info)
    if sizes == {3}:
        return "3-ring"
    elif sizes == {6}:
        return "6-ring"
    else:
        return "Mixed/Other"


def optimize_rematch_alpha(df: pl.DataFrame, d_w1: np.ndarray) -> float:
    """
    Scans a logarithmic grid of alpha values to maximize topological correlation 
    with the exact W1 distance matrix baseline.
    """
    print("\n[+] Initiating REMatch alpha parameter optimization...")
    rematch = REMatch()
    
    n_samples = d_w1.shape[0]
    triu_idx = np.triu_indices(n_samples, k=1)
    flat_w1 = d_w1[triu_idx]
    
    alpha_grid = np.logspace(-3.5, 1, num=50)
    
    best_alpha = 0.1
    best_corr = -1.0
    
    for alpha in alpha_grid:
        d_test = rematch.distance_matrix(df, alpha=alpha)
        
        if d_test is None:
            continue
            
        flat_test = d_test[triu_idx]
        corr, _ = pearsonr(flat_test, flat_w1)
        
        print(f" -> Testing alpha = {alpha:.6f} | Pearson Alignment with W1: {corr:.4f}")
        
        if corr > best_corr:
            best_corr = corr
            best_alpha = alpha
            
    print(f"[+] Optimization Complete. Best Alpha: {best_alpha:.6f} (Pearson r = {best_corr:.4f})")
    return float(best_alpha)


def calculate_spatial_metrics(dist_matrix: np.ndarray, legend_labels: np.ndarray, unique_classes: np.ndarray) -> Dict[str, float]:
    """Helper to compute clustering metrics given a precomputed distance matrix across N classes."""
    sil = silhouette_score(dist_matrix, legend_labels, metric="precomputed")
    
    # Calculate generalized Mean Intra-Class Distance
    intra_dists = []
    for cls in unique_classes:
        idx = np.where(legend_labels == cls)[0]
        if len(idx) > 1:
            sub_matrix = dist_matrix[np.ix_(idx, idx)]
            triu_idx = np.triu_indices(len(idx), k=1)
            intra_dists.append(np.mean(sub_matrix[triu_idx]))
    mean_intra = np.mean(intra_dists) if intra_dists else 0.0

    # Calculate generalized Mean Inter-Class Distance
    inter_dists = []
    for i in range(len(unique_classes)):
        for j in range(i + 1, len(unique_classes)):
            idx_A = np.where(legend_labels == unique_classes[i])[0]
            idx_B = np.where(legend_labels == unique_classes[j])[0]
            inter_matrix = dist_matrix[np.ix_(idx_A, idx_B)]
            inter_dists.append(np.mean(inter_matrix))
    mean_inter = np.mean(inter_dists) if inter_dists else 1.0

    separation_ratio = mean_intra / mean_inter
    
    return {
        "Silhouette": sil, 
        "Intra-Class": mean_intra, 
        "Inter-Class": mean_inter, 
        "Sep-Ratio": separation_ratio
    }


def run_strain_topology_comparison(
    df: pl.DataFrame,
    matrices: Dict[str, np.ndarray],
    label_col: str = "ring_category"
) -> None:
    """
    Computes topological summary properties (pre- and post-UMAP) and generates a dynamic side-by-side UMAP plot.
    """
    raw_labels = df.get_column(label_col).to_numpy()
    
    # Map raw RDKit categories to clean legend labels
    conditions = [
        raw_labels == "3-ring", 
        raw_labels == "6-ring", 
        raw_labels == "Acyclic"
    ]
    choices = [
        "Strained (3-Membered)", 
        "Relaxed (6-Membered)", 
        "Acyclic (No Rings)"
    ]
    legend_labels = np.select(conditions, choices, default="Other")
    unique_classes = np.unique(legend_labels)

    pre_umap_metrics = {}
    post_umap_metrics = {}

    # Dynamic Grid Layout (max 4 columns per row)
    n_plots = len(matrices)
    cols = 4
    rows = 2
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), dpi=300)
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
        
    # Updated Palette to include Acyclic
    palette = {
        "Strained (3-Membered)": "#cc5e53", 
        "Relaxed (6-Membered)": "#568bbd",
        "Acyclic (No Rings)": "#8ebd77"
    }

    # Process each matrix: compute pre-metrics, project UMAP, compute post-metrics, and plot
    for ax, (name, dist_matrix) in zip(axes, matrices.items()):
        # 1. Pre-UMAP (Ambient Space) Metrics
        pre_umap_metrics[name] = calculate_spatial_metrics(dist_matrix, legend_labels, unique_classes)
        
        # 2. Compute UMAP projection
        print(f"Computing UMAP projection from precomputed {name} matrix...")
        reducer = UMAP(n_neighbors=5, metric="precomputed", random_state=42)
        umap_coords = reducer.fit_transform(dist_matrix)
        
        # 3. Post-UMAP (Embedded Space) Metrics
        d_umap = squareform(pdist(umap_coords, metric="euclidean"))
        post_umap_metrics[name] = calculate_spatial_metrics(d_umap, legend_labels, unique_classes)

        # 4. Plotting
        sns.scatterplot(
            x=umap_coords[:, 0], y=umap_coords[:, 1],
            hue=legend_labels, palette=palette, s=90, alpha=0.85,
            edgecolors="#2d3436", linewidths=0.8, ax=ax, zorder=10,
        )

        ax.set_title(name, fontsize=11, fontweight="bold", pad=15)
        ax.set_xlabel("UMAP Dimension 1", fontsize=10, fontweight="medium")
        ax.set_ylabel("UMAP Dimension 2", fontsize=10, fontweight="medium")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)

    # Output Evaluation Summaries
    print("\n" + "="*115)
    print(" PRE-UMAP METRICS (True Ambient Space Distances)")
    print("="*115)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<15} | {'Inter-Class (↑)':<15} | {'Sep Ratio (↓)':<13} | {'Silhouette (↑)':<12}")
    print("-" * 115)
    for name, m in pre_umap_metrics.items():
        print(f"{name:<35} | {m['Intra-Class']:<15.4f} | {m['Inter-Class']:<15.4f} | {m['Sep-Ratio']:<13.4f} | {m['Silhouette']:<12.4f}")
    
    print("\n" + "="*115)
    print(" POST-UMAP METRICS (Distorted 2D Embedding Distances - USE WITH CAUTION)")
    print("="*115)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<15} | {'Inter-Class (↑)':<15} | {'Sep Ratio (↓)':<13} | {'Silhouette (↑)':<12}")
    print("-" * 115)
    for name, m in post_umap_metrics.items():
        print(f"{name:<35} | {m['Intra-Class']:<15.4f} | {m['Inter-Class']:<15.4f} | {m['Sep-Ratio']:<13.4f} | {m['Silhouette']:<12.4f}")
    print("="*115 + "\n")

    # Hide any unused subplots in the grid
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Topology Mapping", title_fontsize=11, fontsize=10,
        loc="lower center", bbox_to_anchor=(0.5, -0.05 if rows > 1 else -0.15), ncol=3,
        frameon=True, facecolor="white", edgecolor="#e2e8f0",
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15 if rows > 1 else 0.2) 
    output_filename = "transport_strain_comparison.png"
    print(f"Saved high-res comparison plot to {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()


def run_pipeline(df_base: pl.DataFrame, grassmann_k: int = 3):
    """
    Executes structural cleaning, unifies balance allocations across matching
    chemical formulas, optimizes the entropic framework, and generates benchmarks.
    """
    # Step A: Perform RDKit classification
    print("Categorizing structures and enforcing pure carbocycles using RDKit...")
    df_classified = df_base.with_columns(
        pl.col("smiles")
        .map_elements(categorize_pure_carbocycle, return_dtype=pl.Utf8)
        .alias("ring_category")
    )

    df_3_rings_all = df_classified.filter(pl.col("ring_category") == "3-ring")
    df_6_rings_all = df_classified.filter(pl.col("ring_category") == "6-ring")
    df_acyclic_all = df_classified.filter(pl.col("ring_category") == "Acyclic")

    # Step B: Identify overlapping formula matches across ALL THREE topologies
    counts_3 = df_3_rings_all.group_by("formula").len().rename({"len": "count_3"})
    counts_6 = df_6_rings_all.group_by("formula").len().rename({"len": "count_6"})
    counts_0 = df_acyclic_all.group_by("formula").len().rename({"len": "count_0"})
    
    formula_overlap = (
        counts_3
        .join(counts_6, on="formula", how="inner")
        .join(counts_0, on="formula", how="inner")
    )

    formula_overlap = formula_overlap.with_columns(
        pl.min_horizontal("count_3", "count_6", "count_0").alias("max_balanced_size")
    ).sort("max_balanced_size", descending=True)

    if formula_overlap.height == 0:
        raise ValueError("No overlapping pure carbocycle formulas found across all three topologies. Increase QM9 limit.")

    # Step C: Extract best formula matches and balance allocations
    best_row = formula_overlap.row(0, named=True)
    best_formula = best_row["formula"]
    max_size = min(best_row["max_balanced_size"], 100)

    print(f"\nOptimal configuration located: {best_formula}")
    print(f"Sampling balanced sets of size: {max_size} per topology")

    df_3_sampled = df_3_rings_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    df_6_sampled = df_6_rings_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    df_0_sampled = df_acyclic_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    
    df_experiment = pl.concat([df_3_sampled, df_6_sampled, df_0_sampled])

    # Step D: Compute Base Reference Spaces
    print("Computing Euclidean and Wasserstein distance spaces...")
    X_averaged = np.vstack(df_experiment["soap_embedding"].to_list())
    d_euclidean = squareform(pdist(X_averaged, metric="euclidean"))

    wasserstein = Wasserstein()
    d_w1 = wasserstein.distance_matrix(df_experiment, metric='euclidean')
    d_w2 = wasserstein.distance_matrix(df_experiment, metric='sqeuclidean')

    # Step E: Optimize REMatch alpha
    optimized_alpha = optimize_rematch_alpha(df=df_experiment, d_w1=d_w1)

    # Step F: Generate Non-Euclidean Distance Matrices
    print("\nComputing REMatch, Riemann, and Grassmann distance spaces...")
    rematch = REMatch()
    d_rematch_high = rematch.distance_matrix(df_experiment, alpha=10.0)
    d_rematch_opt = rematch.distance_matrix(df_experiment, alpha=optimized_alpha)
    
    if d_rematch_high is None or d_rematch_opt is None:
        print("Error: REMatch returned non-finite matrices during final evaluation blocks.")
        return

    riemann = Riemann()

    d_riemann_log = riemann.distance_matrix(
        df_experiment,
        "soap",
        distance_type="log-euclidean"
    )

    d_riemann_airm = riemann.distance_matrix(
        df_experiment,
        "soap",
        distance_type="affine-invariant"
    )
    d_grassmann = Grassmann().distance_matrix(df_experiment, "soap", distance_type="geodesic", k=grassmann_k)

    # Step G: Consolidate matrices into a dictionary for clean passing
    matrices = {
        "Averaged SOAP (Euclidean)": d_euclidean,
        "Wasserstein ($W_1$)": d_w1,
        "Wasserstein ($W_2$)": d_w2,
        "REMatch (High $\\alpha = 10.0$)": d_rematch_high,

        f"REMatch (Optimized $\\alpha = {optimized_alpha:.4f}$)": d_rematch_opt,
        "Riemann (Log-Euclidean)": d_riemann_log,
        "Riemann (Affine-Invariant)": d_riemann_airm,
        f"Grassmann (k={grassmann_k})": d_grassmann,
    }

    # Step H: Trigger evaluation summary and plotting
    run_strain_topology_comparison(
        df=df_experiment, 
        matrices=matrices, 
        label_col="ring_category"
    )

if __name__ == "__main__":
    from src.datasets import QM9Dataset

    qm9 = QM9Dataset(
        limit=80_000,
        descriptors=["soap"],
    )
    df = qm9.load()
    run_pipeline(df_base=df, grassmann_k=3)