import math
import os
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict, List

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from umap import UMAP

# Assuming rdkit, REMatch, Wasserstein, Grassmann, and Riemann are imported globally
from rdkit import Chem
from src.non_euclidean import Grassmann, Riemann, PersistentHomology
from src.optimal_transport import REMatch, Wasserstein


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


def optimize_rematch_alpha(df: pl.DataFrame, label_col: str = "ring_category") -> float:
    """
    Scans a logarithmic grid of alpha values to find the hyperparameter that
    maximizes the silhouette separation score of the classes in the ambient distance space.
    """
    print("\n[+] Initiating REMatch alpha optimization for maximum class separation...")
    rematch = REMatch()
    
    raw_labels = df.get_column(label_col).to_numpy()
    conditions = [raw_labels == "3-ring", raw_labels == "6-ring", raw_labels == "Acyclic"]
    choices = ["Strained", "Relaxed", "Acyclic"]
    legend_labels = np.select(conditions, choices, default="Other")
    
    alpha_grid = np.logspace(-3.5, 1, num=50)
    
    best_alpha = 0.1
    best_silhouette = -1.0
    
    for alpha in alpha_grid:
        d_test = rematch.distance_matrix(df, alpha=alpha)
        if d_test is None:
            continue
            
        try:
            current_sil = silhouette_score(d_test, legend_labels, metric="precomputed")
            print(f" -> Testing alpha = {alpha:.6f} | Ambient Silhouette Score: {current_sil:.4f}")
            
            if current_sil > best_silhouette:
                best_silhouette = current_sil
                best_alpha = alpha
        except ValueError:
            continue
            
    print("[+] Optimization Complete.")
    print(f"    Best Alpha: {best_alpha:.6f} (Max Ambient Silhouette: {best_silhouette:.4f})")
    return float(best_alpha)


def calculate_spatial_metrics(dist_matrix: np.ndarray, legend_labels: np.ndarray, unique_classes: np.ndarray) -> Dict[str, float]:
    """Helper to compute clustering metrics given a precomputed distance matrix across N classes."""
    sil = silhouette_score(dist_matrix, legend_labels, metric="precomputed")
    
    intra_dists = []
    for cls in unique_classes:
        idx = np.where(legend_labels == cls)[0]
        if len(idx) > 1:
            sub_matrix = dist_matrix[np.ix_(idx, idx)]
            triu_idx = np.triu_indices(len(idx), k=1)
            intra_dists.append(np.mean(sub_matrix[triu_idx]))
    mean_intra = np.mean(intra_dists) if intra_dists else 0.0

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
    Computes topological summaries, saves standalone figures, and groups core 
    visual benchmarks into three distinct narrative-driven 1x3 composite figures.
    Runs UMAP across 5 distinct random states to generate statistical error bars.
    All outputs are saved to results/qm9/strain.
    """
    output_dir = "results/qm9/strain"
    os.makedirs(output_dir, exist_ok=True)

    raw_labels = df.get_column(label_col).to_numpy()
    conditions = [raw_labels == "3-ring", raw_labels == "6-ring", raw_labels == "Acyclic"]
    choices = ["Strained (3-Membered)", "Relaxed (6-Membered)", "Acyclic (No Rings)"]
    legend_labels = np.select(conditions, choices, default="Other")
    unique_classes = np.unique(legend_labels)

    pre_umap_metrics = {}
    # Track runs across multiple seeds to compute mean and std
    post_umap_runs = {name: [] for name in matrices.keys()}
    umap_coordinates_cache = {}

    # Define 5 random states for statistical validation
    umap_seeds = [42, 13, 108, 2026, 888]

    plt.style.use("seaborn-v0_8-whitegrid")
    palette = {
        "Strained (3-Membered)": "#cc5e53", 
        "Relaxed (6-Membered)": "#568bbd",
        "Acyclic (No Rings)": "#8ebd77"
    }

    # 1. Pipeline execution: Compute ambient metrics, loop seeds, run UMAP, and export panels
    for name, dist_matrix in matrices.items():
        # Ambient space is deterministic, calculate once
        pre_umap_metrics[name] = calculate_spatial_metrics(dist_matrix, legend_labels, unique_classes)
        
        print(f"\n[+] Processing framework: {name}")
        for idx, seed in enumerate(umap_seeds):
            print(f"    -> Running UMAP with random_state={seed}... ", end="", flush=True)
            reducer = UMAP(n_neighbors=5, metric="precomputed", random_state=seed)
            umap_coords = reducer.fit_transform(dist_matrix)
            
            # Compute 2D Euclidean spatial metrics for this run
            d_umap = squareform(pdist(umap_coords, metric="euclidean"))
            run_metrics = calculate_spatial_metrics(d_umap, legend_labels, unique_classes)
            post_umap_runs[name].append(run_metrics)
            print("Done.")

            # Use the first seed as the baseline for visualization output
            if seed == umap_seeds[0]:
                umap_coordinates_cache[name] = umap_coords
                
                # Generate and Save Individual Subplot Figure
                fig_ind, ax_ind = plt.subplots(figsize=(6, 5), dpi=300)
                sns.scatterplot(
                    x=umap_coords[:, 0], y=umap_coords[:, 1],
                    hue=legend_labels, palette=palette, s=85, alpha=0.85,
                    edgecolors="#2d3436", linewidths=0.6, ax=ax_ind, zorder=10,
                )
                ax_ind.set_title(name, fontsize=12, fontweight="bold", pad=12)
                ax_ind.set_xlabel("UMAP 1", fontsize=10, fontweight="medium")
                ax_ind.set_ylabel("UMAP 2", fontsize=10, fontweight="medium")
                ax_ind.grid(True, linestyle=":", alpha=0.6)
                ax_ind.legend(title="Topology Mapping", frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="best")
                sns.despine(ax=ax_ind)
                plt.tight_layout()
                
                safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("$", "").replace("\\", "")
                fig_ind.savefig(os.path.join(output_dir, f"sub_{safe_name}.png"), dpi=300, bbox_inches="tight")
                plt.close(fig_ind)

    # Helper function to plot a structured thematic group row using baseline coordinates
    def plot_thematic_group(group_keys: List[str], filename: str, figure_title: str):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
        
        for idx, key in enumerate(group_keys):
            ax = axes[idx]
            actual_key = next((k for k in matrices.keys() if k.startswith(key)), None)
            
            if actual_key is None:
                ax.axis('off')
                continue
                
            coords = umap_coordinates_cache[actual_key]
            sns.scatterplot(
                x=coords[:, 0], y=coords[:, 1],
                hue=legend_labels, palette=palette, s=75, alpha=0.85,
                edgecolors="#2d3436", linewidths=0.6, ax=ax, zorder=10,
            )
            ax.set_title(actual_key, fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel("UMAP 1", fontsize=9, fontweight="medium")
            ax.set_ylabel("UMAP 2", fontsize=9, fontweight="medium")
            ax.grid(True, linestyle=":", alpha=0.6)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            sns.despine(ax=ax)
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Topology Mapping", title_fontsize=10, fontsize=9,
            loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3,
            frameon=True, facecolor="white", edgecolor="#e2e8f0",
        )
        fig.suptitle(figure_title, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. Render 1x3 composite figures using the cached baseline mappings
    group_a = ["Averaged SOAP (Euclidean)", "Wasserstein ($W_2$)", "REMatch (Optimized"]
    plot_thematic_group(group_a, "fig18a_transport_topologies.png", "Figure 18a: Limits of Flattening vs. Distribution Spaces")

    group_b = ["Riemann (Log-Euclidean)", "Riemann (Affine-Invariant)", "Grassmann (Geodesic"]
    plot_thematic_group(group_b, "fig18b_curved_manifolds.png", "Figure 18b: Statistical Covariance Spaces and Curved Subspaces")

    group_c = ["PH (Coords, Bottleneck)", "PH (Coords, Sliced", "PH (SOAP, Sliced"]
    plot_thematic_group(group_c, "fig18c_persistent_homology.png", "Figure 18c: Topological Persistence Over Atomic Complexes vs. Smoothed Field Space")

    # 3. Output Evaluation Summaries to Terminal
    print("\n" + "="*125)
    print(" PRE-UMAP METRICS (True Ambient Space Distances)")
    print("="*125)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<15} | {'Inter-Class (↑)':<15} | {'Sep Ratio (↓)':<13} | {'Silhouette (↑)':<12}")
    print("-" * 125)
    for name, m in pre_umap_metrics.items():
        print(f"{name:<35} | {m['Intra-Class']:<15.4f} | {m['Inter-Class']:<15.4f} | {m['Sep-Ratio']:<13.4f} | {m['Silhouette']:<12.4f}")
    
    print("\n" + "="*145)
    print(" POST-UMAP METRICS (Statistical Aggregation across 5 Random States: mean ± std)")
    print("="*145)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<22} | {'Inter-Class (↑)':<22} | {'Sep Ratio (↓)':<20} | {'Silhouette (↑)':<20}")
    print("-" * 145)
    
    for name in matrices.keys():
        runs = post_umap_runs[name]
        
        # Unpack accumulated metrics lists
        sil_vals = [r["Silhouette"] for r in runs]
        intra_vals = [r["Intra-Class"] for r in runs]
        inter_vals = [r["Inter-Class"] for r in runs]
        sep_vals = [r["Sep-Ratio"] for r in runs]
        
        # Format strings as mean ± standard deviation
        intra_str = f"{np.mean(intra_vals):.4f} ± {np.std(intra_vals):.4f}"
        inter_str = f"{np.mean(inter_vals):.4f} ± {np.std(inter_vals):.4f}"
        sep_str = f"{np.mean(sep_vals):.4f} ± {np.std(sep_vals):.4f}"
        sil_str = f"{np.mean(sil_vals):.4f} ± {np.std(sil_vals):.4f}"
        
        print(f"{name:<35} | {intra_str:<22} | {inter_str:<22} | {sep_str:<20} | {sil_str:<20}")
        
    print("="*145 + "\n")
    print(f"[+] Successfully exported grouped composite grids and standalone panels to: {output_dir}")


def run_pipeline(df_base: pl.DataFrame, grassmann_k: int = 3):
    """
    Executes structural cleaning, unifies balance allocations across matching
    chemical formulas, optimizes the entropic framework, and generates benchmarks.
    """
    print("Categorizing structures and enforcing pure carbocycles using RDKit...")
    df_classified = df_base.with_columns(
        pl.col("smiles")
        .map_elements(categorize_pure_carbocycle, return_dtype=pl.Utf8)
        .alias("ring_category")
    )

    df_3_rings_all = df_classified.filter(pl.col("ring_category") == "3-ring")
    df_6_rings_all = df_classified.filter(pl.col("ring_category") == "6-ring")
    df_acyclic_all = df_classified.filter(pl.col("ring_category") == "Acyclic")

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

    best_row = formula_overlap.row(0, named=True)
    best_formula = best_row["formula"]
    max_size = min(best_row["max_balanced_size"], 100)

    print(f"\nOptimal configuration located: {best_formula}")
    print(f"Sampling balanced sets of size: {max_size} per topology")

    df_3_sampled = df_3_rings_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    df_6_sampled = df_6_rings_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    df_0_sampled = df_acyclic_all.filter(pl.col("formula") == best_formula).sample(n=max_size, seed=42)
    
    df_experiment = pl.concat([df_3_sampled, df_6_sampled, df_0_sampled])

    print("Computing Euclidean and Wasserstein distance spaces...")
    X_averaged = np.vstack(df_experiment["soap_embedding"].to_list())
    d_euclidean = squareform(pdist(X_averaged, metric="euclidean"))

    wasserstein = Wasserstein()
    d_w1 = wasserstein.distance_matrix(df_experiment, metric='euclidean')
    d_w2 = wasserstein.distance_matrix(df_experiment, metric='sqeuclidean')

    optimized_alpha = optimize_rematch_alpha(df=df_experiment, label_col="ring_category")

    print("\nComputing REMatch, Riemann, and Grassmann distance spaces...")
    rematch = REMatch()
    d_rematch_high = rematch.distance_matrix(df_experiment, alpha=10.0)
    d_rematch_opt = rematch.distance_matrix(df_experiment, alpha=optimized_alpha)
    
    if d_rematch_high is None or d_rematch_opt is None:
        print("Error: REMatch returned non-finite matrices during final evaluation blocks.")
        return

    riemann = Riemann()
    d_riemann_log = riemann.distance_matrix(df_experiment, "soap", distance_type="log-euclidean")
    d_riemann_airm = riemann.distance_matrix(df_experiment, "soap", distance_type="affine-invariant")
    
    d_grassmann_geodesic = Grassmann().distance_matrix(df_experiment, "soap", distance_type="geodesic", k=grassmann_k)
    d_grassmann_chordal = Grassmann().distance_matrix(df_experiment, "soap", distance_type="chordal", k=grassmann_k)

    print("Computing Persistent Homology distance spaces (Bottleneck and Sliced Wasserstein)...")
    d_ph_coords_bn = PersistentHomology.distance_matrix(df_experiment, descriptor='coordinates', metric='bottleneck')
    d_ph_coords_sw = PersistentHomology.distance_matrix(df_experiment, descriptor='coordinates', metric='sliced_wasserstein')
    d_ph_soap_bn = PersistentHomology.distance_matrix(df_experiment, descriptor='soap', metric='bottleneck')
    d_ph_soap_sw = PersistentHomology.distance_matrix(df_experiment, descriptor='soap', metric='sliced_wasserstein')

    matrices = {
        "Averaged SOAP (Euclidean)": d_euclidean,
        "Wasserstein ($W_1$)": d_w1,
        "Wasserstein ($W_2$)": d_w2,
        "REMatch (High $\\alpha = 10.0$)": d_rematch_high,
        f"REMatch (Optimized $\\alpha = {optimized_alpha:.4f}$)": d_rematch_opt,
        "Riemann (Log-Euclidean)": d_riemann_log,
        "Riemann (Affine-Invariant)": d_riemann_airm,
        f"Grassmann (Geodesic, k={grassmann_k})": d_grassmann_geodesic,
        f"Grassmann (Chordal, k={grassmann_k})": d_grassmann_chordal,
        "PH (Coords, Bottleneck)": d_ph_coords_bn,
        "PH (Coords, Sliced Wasserstein)": d_ph_coords_sw,
        "PH (SOAP, Bottleneck)": d_ph_soap_bn,
        "PH (SOAP, Sliced Wasserstein)": d_ph_soap_sw
    }

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