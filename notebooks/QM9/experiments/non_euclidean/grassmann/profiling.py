import time
import tracemalloc
import numpy as np
import polars as pl
from typing import Dict, Any

from src.non_euclidean import Grassmann, Riemann, PersistentHomology
from src.optimal_transport import Wasserstein, REMatch


def generate_dummy_molecular_dataset(
    num_molecules: int = 15,
    num_atoms: int = 20,
    dimension: int = 128,
) -> pl.DataFrame:
    """
    Generates a synthetic Polars DataFrame simulating a QM9-like dataset.
    Populates local descriptor matrices, Cartesian coordinates, and atomic properties.
    """
    data = []

    for i in range(num_molecules):
        soap_mat = np.random.uniform(
            0.1, 1.0, size=(num_atoms, dimension)
        ).tolist()

        coords = np.random.uniform(
            -5.0, 5.0, size=(num_atoms, 3)
        ).tolist()

        atomic_numbers = [6] * num_atoms

        data.append(
            {
                "molecule_id": f"mol_{i}",
                "num_atoms": num_atoms,
                "atomic_numbers": atomic_numbers,
                "coordinates": coords,
                "soap_matrix": soap_mat,
            }
        )

    return pl.DataFrame(data)


def compute_average_soap_euclidean_matrix(
    df: pl.DataFrame,
    descriptor: str = "soap",
) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between molecules after
    averaging local descriptors into a global molecular representation.
    """
    col_name = f"{descriptor}_matrix"
    matrices = df[col_name].to_list()

    molecular_vectors = np.array(
        [np.mean(mat, axis=0) for mat in matrices]
    )

    diffs = (
        molecular_vectors[:, np.newaxis, :]
        - molecular_vectors[np.newaxis, :, :]
    )

    distance_matrix = np.sqrt(
        np.sum(diffs**2, axis=-1)
    )

    return distance_matrix


def profile_framework_execution(
    framework_label: str,
    runner_function: Any,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """
    Measures wall-clock runtime and peak memory consumption.
    """
    import gc

    gc.collect()

    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        _ = runner_function(*args, **kwargs)
        failed = False

    except Exception as e:
        print(
            f"\n[!] Critical failure during "
            f"{framework_label} execution: {e}"
        )
        failed = True

    end_time = time.perf_counter()

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end_time - start_time
    peak_mem_mb = peak_memory / (1024 * 1024)

    return {
        "runtime": runtime if not failed else np.nan,
        "memory": peak_mem_mb if not failed else np.nan,
    }


def execute_comprehensive_benchmark():
    """
    Benchmarks scaling with respect to the number of molecules
    while keeping the molecular size fixed.
    """

    # Fixed molecular size
    num_atoms = 24

    # Variable dataset size
    molecule_counts = [10, 25, 50, 100]

    # Descriptor dimensions to benchmark
    dimensions = [256, 512, 1024, 2048]

    benchmark_records = []

    print("=" * 95)
    print(
        "STARTING TOPOLOGY PROFILE FRONTIER "
        "(Scaling with Number of Molecules)"
    )
    print("=" * 95)

    for n_molecules in molecule_counts:
        for d_dim in dimensions:

            print(
                f"\n[+] Processing Dataset: "
                f"N_molecules = {n_molecules} | "
                f"N_atoms = {num_atoms} | "
                f"D = {d_dim}"
            )

            df_dummy = generate_dummy_molecular_dataset(
                num_molecules=n_molecules,
                num_atoms=num_atoms,
                dimension=d_dim,
            )

            target_tasks = [
                {
                    "name": "Euclidean (Average SOAP)",
                    "func": compute_average_soap_euclidean_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                    },
                },
                {
                    "name": "Wasserstein ($W_2$ EMD)",
                    "func": Wasserstein.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                        "metric": "sqeuclidean",
                    },
                },
                {
                    "name": "REMatch ($\\alpha=0.3393$)",
                    "func": REMatch.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                        "alpha": 0.3393,
                    },
                },
                {
                    "name": "Riemann (Log-Euclidean)",
                    "func": Riemann.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                        "distance_type": "log-euclidean",
                        "pca": False,
                    },
                },
                {
                    "name": "Riemann (Affine-Invariant)",
                    "func": Riemann.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                        "distance_type": "affine-invariant",
                        "pca": False,
                    },
                },
                {
                    "name": "Grassmann (Geodesic, k=3)",
                    "func": Grassmann.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "soap",
                        "distance_type": "geodesic",
                        "k": 3,
                    },
                },
                {
                    "name": "PH (Coords, Dim 1)",
                    "func": PersistentHomology.distance_matrix,
                    "kwargs": {
                        "df": df_dummy,
                        "descriptor": "coordinates",
                        "metric": "sliced_wasserstein",
                        "max_homology_dim": 1,
                        "homology_dims": (0, 1),
                    },
                },
            ]

            for task in target_tasks:

                metrics = profile_framework_execution(
                    framework_label=task["name"],
                    runner_function=task["func"],
                    **task["kwargs"],
                )

                benchmark_records.append(
                    {
                        "Framework": task["name"],
                        "N_molecules": n_molecules,
                        "N_atoms": num_atoms,
                        "D_dim": d_dim,
                        "Runtime": metrics["runtime"],
                        "Memory": metrics["memory"],
                    }
                )

    print("\n\n" + "=" * 120)
    print("FINAL ARCHITECTURAL METRIC BENCHMARK PROFILE")
    print("=" * 120)

    print(
        f"{'Framework Target':<30} | "
        f"{'N_molecules':<12} | "
        f"{'N_atoms':<8} | "
        f"{'D_dim':<8} | "
        f"{'Runtime (s)':<15} | "
        f"{'Peak Memory (MB)':<18}"
    )

    print("-" * 120)

    for r in benchmark_records:

        run_str = (
            f"{r['Runtime']:.6f}"
            if not np.isnan(r["Runtime"])
            else "FAILED"
        )

        mem_str = (
            f"{r['Memory']:.4f}"
            if not np.isnan(r["Memory"])
            else "FAILED"
        )

        print(
            f"{r['Framework']:<30} | "
            f"{r['N_molecules']:<12} | "
            f"{r['N_atoms']:<8} | "
            f"{r['D_dim']:<8} | "
            f"{run_str:<15} | "
            f"{mem_str:<18}"
        )

    print("=" * 120 + "\n")


if __name__ == "__main__":
    execute_comprehensive_benchmark()