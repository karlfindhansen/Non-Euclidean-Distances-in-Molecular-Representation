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
from loguru import logger

from rdkit import Chem
from src.non_euclidean import Grassmann, Riemann, PersistentHomology
from src.optimal_transport import REMatch, Wasserstein


def optimize_rematch_alpha(df: pl.DataFrame, label_col: str = "dipole_segment") -> float:
    """
    Scans a logarithmic grid of alpha values to find the hyperparameter that
    maximizes the silhouette separation score of the classes in the ambient distance space.
    """
    print("\n[+] Initiating REMatch alpha optimization for maximum class separation...")
    rematch = REMatch()
    
    raw_labels = df.get_column(label_col).to_numpy()
    vals = label_col.split('_')[0]

    conditions = [raw_labels == "Low", raw_labels == "Medium", raw_labels == "High"]
    choices = [f"Low {vals}", f"Medium {vals}", f"High {vals}"]
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


def run_quantum_topology_comparison(
    df: pl.DataFrame,
    matrices: Dict[str, np.ndarray],
    label_col: str = "dipole_segment"
) -> None:
    """
    Computes topological summaries, saves standalone figures, and groups core 
    visual benchmarks into three distinct narrative-driven 1x3 composite figures.
    All outputs are saved to results/qm9/quantum_dipole.
    """
    vals = label_col.split('_')[0]
    print(vals)
    output_dir = f"results/qm9/quantum_{vals}"
    os.makedirs(output_dir, exist_ok=True)

    raw_labels = df.get_column(label_col).to_numpy()
    conditions = [raw_labels == "Low", raw_labels == "Medium", raw_labels == "High"]
    choices = [f"Low {vals}", f"Medium {vals}", f"High {vals}"]
    legend_labels = np.select(conditions, choices, default="Other")
    unique_classes = np.unique(legend_labels)

    pre_umap_metrics = {}
    post_umap_metrics = {}
    umap_coordinates_cache = {}

    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Dipole property palette
    palette = {
        f"Low {vals}": "#4a7c59", 
        f"Medium {vals}": "#dda15e",
        f"High {vals}": "#bc6c25"
    }
    print(palette)

    # 1. Pipeline execution: Compute metrics, run UMAP, and export separate panels
    for name, dist_matrix in matrices.items():
        pre_umap_metrics[name] = calculate_spatial_metrics(dist_matrix, legend_labels, unique_classes)
        
        print(f"Computing UMAP projection from precomputed {name} matrix...")
        reducer = UMAP(n_neighbors=5, metric="precomputed", random_state=42)
        umap_coords = reducer.fit_transform(dist_matrix)
        umap_coordinates_cache[name] = umap_coords
        
        d_umap = squareform(pdist(umap_coords, metric="euclidean"))
        post_umap_metrics[name] = calculate_spatial_metrics(d_umap, legend_labels, unique_classes)

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
        ax_ind.legend(title=f"{vals} Mapping", frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="best")
        sns.despine(ax=ax_ind)
        plt.tight_layout()
        
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("$", "").replace("\\", "")
        fig_ind.savefig(os.path.join(output_dir, f"sub_{safe_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_ind)

    # Helper function to plot a structured thematic group row
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
            handles, labels, title=f"{vals} Target", title_fontsize=10, fontsize=9,
            loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3,
            frameon=True, facecolor="white", edgecolor="#e2e8f0",
        )
        fig.suptitle(figure_title, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. Render Figure 19a: Flattening vs. Transport Topologies
    group_a = ["Averaged SOAP (Euclidean)", "Wasserstein ($W_2$)", "REMatch (Optimized"]
    plot_thematic_group(group_a, "fig19a_transport_topologies.png", f"Figure 19a: Resolution of {vals} Spaces Across Transport Topologies")

    # 3. Render Figure 19b: Covariance Mapping on Curved Manifolds
    group_b = ["Riemann (Log-Euclidean)", "Riemann (Affine-Invariant)", "Grassmann (Geodesic"]
    plot_thematic_group(group_b, "fig19b_curved_manifolds.png", f"Figure 19b: Covariance Frameworks and Curved Manifold Scaling on {vals} Moment")

    # 4. Render Figure 19c: Topological Persistence vs. Feature Smoothing
    group_c = ["PH (Coords, Bottleneck)", "PH (Coords, Sliced)", "PH (SOAP, Sliced)"]
    plot_thematic_group(group_c, "fig19c_persistent_homology.png", "Figure 19c: Persistence Over Coordinate Boundaries vs. Smoothed Field Space")

    # 5. Output Evaluation Summaries to Terminal
    print("\n" + "="*115)
    print(" PRE-UMAP METRICS (True Ambient Space Distances)")
    print("="*115)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<15} | {'Inter-Class (↑)':<15} | {'Sep Ratio (↓)':<13} | {'Silhouette (↑)':<12}")
    print("-" * 115)
    for name, m in pre_umap_metrics.items():
        print(f"{name:<35} | {m['Intra-Class']:<15.4f} | {m['Inter-Class']:<15.4f} | {m['Sep-Ratio']:<13.4f} | {m['Silhouette']:<12.4f}")
    
    print("\n" + "="*115)
    print(" POST-UMAP METRICS (Distorted 2D Embedding Distances)")
    print("="*115)
    print(f"{'Framework / Metric':<35} | {'Intra-Class (↓)':<15} | {'Inter-Class (↑)':<15} | {'Sep Ratio (↓)':<13} | {'Silhouette (↑)':<12}")
    print("-" * 115)
    for name, m in post_umap_metrics.items():
        print(f"{name:<35} | {m['Intra-Class']:<15.4f} | {m['Inter-Class']:<15.4f} | {m['Sep-Ratio']:<13.4f} | {m['Silhouette']:<12.4f}")
    print("="*115 + "\n")
    print(f"[+] Successfully exported grouped composite grids and standalone panels to: {output_dir}")


def run_pipeline(df_base: pl.DataFrame, grassmann_k: int = 3, property = "mu"):
    """
    Executes baseline filtering, isolates the most abundant polar isometric stoichiometry,
    segments the continuous dipole moment target into balanced tertiles, and runs benchmarks.
    """
    print("Identifying the most abundant polar chemical formula to enforce isometric constraint...")
    
    # Ensure the isolated formula contains heteroatoms to guarantee dipole variance
    formula_counts = df_base.filter(
        pl.col("formula").str.contains("O") | pl.col("formula").str.contains("N")
    ).group_by("formula").len().sort("len", descending=True)
    
    best_formula = formula_counts.row(0, named=True)["formula"]
    print(f"Isolating subspace for polar formula: {best_formula}")
    df_base = df_base.filter(pl.col("structure_class") == "Acyclic")
    df_subspace = df_base.filter(pl.col("formula") == best_formula)
    
    # Compute tertiles (33.3% and 66.6% quantiles) for the dipole moment (mu) within this formula
    mu_values = df_subspace.get_column(property).to_numpy()
    q1, q2 = np.percentile(mu_values, [33.33, 66.67])
    
    print(f"Slicing dipole distribution into tertiles: Low (<={q1:.4f}), Medium ({q1:.4f}-{q2:.4f}), High (>{q2:.4f})")
    df_segmented = df_subspace.with_columns(
        pl.when(pl.col(property) <= q1).then(pl.lit("Low"))
        .when(pl.col(property) <= q2).then(pl.lit("Medium"))
        .otherwise(pl.lit("High"))
        .alias(f"{property}_segment")
    )
    
    # Ensure balanced subsets across the tertiles
    min_size = df_segmented.group_by(f"{property}_segment").len().get_column("len").min()
    if min_size < 30:
        logger.warning("There are less than 50 molecules in each class in the selected set")

    max_balanced_sample = min(min_size, 50)
    
    print(f"Sampling balanced configurations: {max_balanced_sample} entries per electronic segment")
    df_low = df_segmented.filter(pl.col(f"{property}_segment") == "Low").sample(n=max_balanced_sample, seed=42)
    df_med = df_segmented.filter(pl.col(f"{property}_segment") == "Medium").sample(n=max_balanced_sample, seed=42)
    df_high = df_segmented.filter(pl.col(f"{property}_segment") == "High").sample(n=max_balanced_sample, seed=42)
    
    df_experiment = pl.concat([df_low, df_med, df_high])

    print("Computing Euclidean and Wasserstein distance spaces...")
    X_averaged = np.vstack(df_experiment["soap_embedding"].to_list())
    d_euclidean = squareform(pdist(X_averaged, metric="euclidean"))

    wasserstein = Wasserstein()
    d_w1 = wasserstein.distance_matrix(df_experiment, metric='euclidean')
    d_w2 = wasserstein.distance_matrix(df_experiment, metric='sqeuclidean')

    optimized_alpha = optimize_rematch_alpha(df=df_experiment, label_col=f"{property}_segment")

    print("\nComputing REMatch, Riemann, and Grassmann distance spaces...")
    rematch = REMatch()
    d_rematch_high = rematch.distance_matrix(df_experiment, alpha=10.0)
    d_rematch_opt = rematch.distance_matrix(df_experiment, alpha=optimized_alpha)
    
    if d_rematch_high is None or d_rematch_opt is None:
        print("Error: REMatch returned non-finite matrices during evaluation blocks.")
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

    run_quantum_topology_comparison(
        df=df_experiment, 
        matrices=matrices, 
        label_col=f"{property}_segment"
    )


if __name__ == "__main__":
    from src.datasets import QM9Dataset

    qm9 = QM9Dataset(
        limit=50_000,
        descriptors=["soap"],
    )
    df = qm9.load()
    run_pipeline(df_base=df, grassmann_k=3, property="r2")