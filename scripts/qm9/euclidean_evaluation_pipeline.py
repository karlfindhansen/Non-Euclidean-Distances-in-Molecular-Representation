from src.datasets import QM9Dataset
from scripts.materials_project.euclidean_evaluation_pipeline import (
    evaluate_dbscan_combinations,
    evaluate_hierarchical_combinations,
    evaluate_kmeans_combinations,
    evaluate_spectral_combinations,
)


if __name__ == "__main__":
    qm9 = QM9Dataset(limit=400, stratify_by=["num_atoms", "gap"], sampling_strategy="stratified", add_soap=True)
    df = qm9.load()

    output_base_dir = "figures/qm9/clustering"

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
    # evaluate_kmeans_combinations(
    #     df,
    #     k_min=2,
    #     k_max=20,
    #     mode="invariant",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_spectral_combinations(
    #     df,
    #     k_min=2,
    #     k_max=20,
    #     mode="invariant",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_dbscan_combinations(
    #     df,
    #     min_samples=3,
    #     min_samples_values=[2, 3, 5, 8, 10],
    #     mode="invariant",
    #     output_base_dir=output_base_dir,
    # )

    # SOAP evaluations using the same methods
    # evaluate_hierarchical_combinations(
    #     df,
    #     linkage="average",
    #     k_min=2,
    #     k_max=20,
    #     mode="soap",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_hierarchical_combinations(
    #     df,
    #     linkage="complete",
    #     k_min=2,
    #     k_max=20,
    #     mode="soap",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_kmeans_combinations(
    #     df,
    #     k_min=2,
    #     k_max=20,
    #     mode="soap",
    #     output_base_dir=output_base_dir,
    # )
    # evaluate_spectral_combinations(
    #     df,
    #     k_min=2,
    #     k_max=20,
    #     mode="soap",
    #     output_base_dir=output_base_dir,
    # )
    evaluate_dbscan_combinations(
        df,
        min_samples_values=[2, 3, 5, 8, 10],
        mode="soap",
        output_base_dir=output_base_dir,
    )
