import polars as pl

from src.datasets import QM9Dataset
from scripts.materials_project.euclidean_evaluation_pipeline import (
    evaluate_dbscan_combinations,
    evaluate_hierarchical_combinations,
    evaluate_kmeans_combinations,
    evaluate_spectral_combinations,
)


if __name__ == "__main__":
    qm9 = QM9Dataset(limit=50_000, stratify_by=["num_atoms", "gap"], sampling_strategy="stratified")
    df = qm9.load()
    # find the isomers and filter these
    limit = 100
    formula_counts = df.group_by("formula").len().sort("len", descending=True)

    eligible = formula_counts.filter(pl.col("len") >= limit)
    if eligible.is_empty():
        raise ValueError(f"No formulas found with at least {limit} isomers")

    target_formula = eligible.row(0)[0]

    isomers_df = df.filter(pl.col("formula") == target_formula)
    df = isomers_df.head(limit)
    df = df.unique(subset="canonical_smiles")

    output_base_dir = "figures/qm9/clustering/isomers_euclidean"

    # Invariant-only evaluations
    # evaluate_hierarchical_combinations(
    #     df,
    #     linkage="average",
    #     k_min=2,
    #     k_max=20,
    #     mode="invariant",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_hierarchical_combinations(
    #     df,
    #     linkage="complete",
    #     k_min=2,
    #     k_max=20,
    #     mode="invariant",
    #     output_base_dir=output_base_dir,
    # )
    evaluate_kmeans_combinations(
        df,
        k_min=2,
        k_max=20,
        mode="invariant",
        output_base_dir=output_base_dir,
    )
    evaluate_spectral_combinations(
        df,
        k_min=2,
        k_max=20,
        mode="invariant",
        output_base_dir=output_base_dir,
    )
    evaluate_dbscan_combinations(
        df,
        min_samples=3,
        mode="invariant",
        output_base_dir=output_base_dir,
    )

    # SOAP evaluations using the same methods
    evaluate_hierarchical_combinations(
        df,
        linkage="average",
        k_min=2,
        k_max=20,
        mode="soap",
        output_base_dir=output_base_dir,
    )
    evaluate_hierarchical_combinations(
        df,
        linkage="complete",
        k_min=2,
        k_max=20,
        mode="soap",
        output_base_dir=output_base_dir,
    )
    evaluate_kmeans_combinations(
        df,
        k_min=2,
        k_max=20,
        mode="soap",
        output_base_dir=output_base_dir,
    )
    evaluate_spectral_combinations(
        df,
        k_min=2,
        k_max=20,
        mode="soap",
        output_base_dir=output_base_dir,
    )
    evaluate_dbscan_combinations(
        df,
        min_samples=3,
        mode="soap",
        output_base_dir=output_base_dir,
    )
