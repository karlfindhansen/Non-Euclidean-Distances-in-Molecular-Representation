import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

import chemiscope
import kmedoids
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.clusters import MolecularClusterScore, calculate_congruence
from src.datasets import MaterialsProject
from src.helper_functions import find_best_kmedoids_k, get_distances


def _materials_cluster_score(
    df_scored: pl.DataFrame,
    embedding: np.ndarray,
    labels: np.ndarray,
):
    prop_cols = [
        "energy_per_atom",
        "formation_energy_per_atom",
        "density",
        "volume",
        "num_sites",
    ]

    if not all(c in df_scored.columns for c in prop_cols):
        return {"error": "Missing material property columns."}

    property_matrix = df_scored.select(prop_cols).to_numpy()

    if "space_group" in df_scored.columns:
        categories = df_scored["space_group"].to_list()
    elif "crystal_system" in df_scored.columns:
        categories = df_scored["crystal_system"].to_list()
    else:
        return {"error": "Missing categorical columns (space_group/crystal_system)."}

    mcs = MolecularClusterScore()
    mcs_result = mcs.compute_total_score(
        embedding,
        labels,
        property_matrix,
        categories,
    )

    if "crystal_system" in df_scored.columns:
        mcs_result["structure_class_score"] = mcs.compute_category_score(
            df_scored["crystal_system"].to_list(),
            labels,
        )
    return mcs_result


def evaluate_materials_descriptor_kmeans(
    k_range=range(2, 15),
    limit=1000,
    sample_size=None,
    output_dir="results/cluster_reports/materials_descriptors_kmeans",
):
    mp = MaterialsProject()
    mp.load(limit=limit)

    embeddings = ["soap_embedding", "acsf_embedding"]
    os.makedirs(output_dir, exist_ok=True)

    for desc in embeddings:
        if desc not in mp.df.columns:
            logger.warning(f"Skipping {desc}: column not found.")
            continue

        df = mp.df.filter(pl.col(desc).is_not_null())
        if df.is_empty():
            logger.warning(f"Skipping {desc}: all values are null.")
            continue

        if sample_size and df.height > sample_size:
            df = df.sample(n=sample_size, seed=42)

        X = np.stack(df[desc].to_list())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        results = {"k": [], "inertia": [], "silhouette": [], "ch": []}

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)

            results["k"].append(int(k))
            results["inertia"].append(float(kmeans.inertia_))
            results["silhouette"].append(float(silhouette_score(X, labels)))
            results["ch"].append(float(calinski_harabasz_score(X, labels)))

        best_k = {
            "inertia": results["k"][int(np.argmin(results["inertia"]))],
            "silhouette": results["k"][int(np.argmax(results["silhouette"]))],
            "ch": results["k"][int(np.argmax(results["ch"]))],
        }

        out_base = os.path.join(output_dir, desc)
        pl.DataFrame(results).write_csv(f"{out_base}_kmeans_eval.csv")
        with open(f"{out_base}_kmeans_best.json", "w") as f:
            json.dump(best_k, f, indent=2)

        try:
            best_k_used = (
                best_k.get("silhouette") or best_k.get("ch") or best_k.get("inertia")
            )
            model = KMeans(n_clusters=int(best_k_used), random_state=42, n_init="auto")
            labels_best = model.fit_predict(X)

            embedding_list = [row.tolist() for row in X]
            df_scored = df.with_columns(
                [
                    pl.Series("cluster_eval", labels_best),
                    pl.Series("embedding_eval", embedding_list),
                ]
            )

            scores_payload = {
                "embedding": desc,
                "best_k_used": int(best_k_used),
                "valid_count": int(len(embedding_list)),
                "total_count": int(len(embedding_list)),
            }

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            scores_payload["molecular_cluster_score"] = _materials_cluster_score(
                df_scored, X, labels_best
            )

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute materials cluster scores for {desc}: {e}")

        logger.success(f"Materials KMeans evaluation complete for {desc}")


def evaluate_materials_euclidean_agglomerative(
    k_range=range(2, 15),
    limit=1000,
    sample_size=None,
    output_dir="results/cluster_reports/materials_euclidean_metrics",
):
    mp = MaterialsProject()
    mp.load(limit=limit)

    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "name": "acsf_embedding",
            "prep": lambda X: StandardScaler().fit_transform(X),
            "metric": "euclidean",
            "linkage": "ward",
        },
        {
            "name": "soap_embedding",
            "prep": lambda X: normalize(X, norm="l2", axis=1),
            "metric": "euclidean",
            "linkage": "ward",
        },
    ]

    for cfg in configs:
        name = cfg["name"]
        if name not in mp.df.columns:
            logger.warning(f"Skipping {name}: column not found.")
            continue

        df = mp.df.filter(pl.col(name).is_not_null())
        if df.is_empty():
            logger.warning(f"Skipping {name}: all values are null.")
            continue

        if sample_size and df.height > sample_size:
            df = df.sample(n=sample_size, seed=42)

        X = np.stack(df[name].to_list())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = cfg["prep"](X)

        scores = {"k": [], "silhouette": []}

        for k in k_range:
            model = AgglomerativeClustering(
                n_clusters=int(k),
                metric=cfg["metric"],
                linkage=cfg["linkage"],
            )
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels, metric=cfg["metric"])
            scores["k"].append(int(k))
            scores["silhouette"].append(float(score))

        best_k = scores["k"][int(np.argmax(scores["silhouette"]))] if scores["k"] else None

        out_base = os.path.join(output_dir, name)
        pl.DataFrame(scores).write_csv(f"{out_base}_agglo_eval.csv")
        with open(f"{out_base}_agglo_best.json", "w") as f:
            json.dump(
                {
                    "best_k": int(best_k) if best_k is not None else None,
                    "metric": cfg["metric"],
                    "linkage": cfg["linkage"],
                },
                f,
                indent=2,
            )

        if best_k is None:
            logger.warning(f"Skipping cluster scores for {name}: no valid k.")
            continue

        try:
            model = AgglomerativeClustering(
                n_clusters=int(best_k),
                metric=cfg["metric"],
                linkage=cfg["linkage"],
            )
            labels_best = model.fit_predict(X)

            embedding_list = [row.tolist() for row in X]
            df_scored = df.with_columns(
                [
                    pl.Series("cluster_eval", labels_best),
                    pl.Series("embedding_eval", embedding_list),
                ]
            )

            scores_payload = {
                "embedding": name,
                "best_k_used": int(best_k),
                "valid_count": int(len(embedding_list)),
                "total_count": int(len(embedding_list)),
            }

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            scores_payload["molecular_cluster_score"] = _materials_cluster_score(
                df_scored, X, labels_best
            )

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute materials agglomerative scores for {name}: {e}")

        logger.success(f"Materials agglomerative evaluation complete for {name}")


def plot_chemiscope(mp_df, embedding_col="soap_embedding", n_clusters=5):
    df = mp_df.to_pandas()

    X = np.vstack(df[embedding_col].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    X_tsne = tsne.fit_transform(X_scaled)

    df["tsne_1"] = X_tsne[:, 0]
    df["tsne_2"] = X_tsne[:, 1]
    df["cluster_id"] = clusters

    frames = []

    for s_str in df["raw_structure"]:
        pmg = Structure.from_dict(json.loads(s_str))
        atoms = AseAtomsAdaptor.get_atoms(pmg)
        frames.append(atoms)

    properties = {
        "Cluster": df["cluster_id"].astype(int).values,
        "Energy per Atom": df["energy_per_atom"].astype(float).values,
        "Formula": df["formula_pretty"].astype(str).tolist(),
        "t-SNE 1": df["tsne_1"].values,
        "t-SNE 2": df["tsne_2"].values,
    }

    settings = {
        "map": {
            "x": {"property": "t-SNE 1"},
            "y": {"property": "t-SNE 2"},
            "color": {"property": "Cluster"},
        }
    }

    chemiscope.write_input(
        "soap_tsne_clusters.json",
        structures=frames,
        properties=properties,
        settings=settings,
        metadata={
            "name": "SOAP clustering visualization",
            "description": "KMeans clustering with t-SNE projection",
        },
    )

    chemiscope.show_input("soap_tsne_clusters.json")


def _build_mp_frames(mp_df: pl.DataFrame):
    adaptor = AseAtomsAdaptor()
    frames_unit = []

    for struct_json in mp_df["raw_structure"]:
        struct = Structure.from_dict(json.loads(struct_json))
        prim = struct.get_primitive_structure()

        if len(prim) < 3:
            prim.make_supercell([2, 2, 2])
            while len(prim) < 3:
                prim.make_supercell([2, 2, 2])

        frames_unit.append(adaptor.get_atoms(prim))

    return frames_unit


def load_precomputed_distances(
    dataset_dir: str,
    expected_n: int,
    include_ph: bool = True,
):
    data_dir = Path("data") / dataset_dir
    matrix_files = {
        "grassmann": "dist_matrix_grassmann.npy",
        "euclidean_riemann": "dist_matrix_euclidean_riemann.npy",
        "affine_riemann": "dist_matrix_affine_riemann.npy",
        "wasserstein": "dist_matrix_wasserstein.npy",
        "ph_bottleneck": "persistent_dist_matrix_bottleneck.npy",
        "ph_sliced_wasserstein": "persistent_dist_matrix_sw.npy",
    }

    matrices = {}
    for name, fname in matrix_files.items():
        if not include_ph and name.startswith("ph_"):
            continue

        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing precomputed matrix for {name}: {path}. "
                "Run get_distances(...) once or disable use_precomputed_only."
            )

        mat = np.load(path)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{name} distance matrix must be square. Got shape {mat.shape}."
            )
        if mat.shape[0] != expected_n:
            raise ValueError(
                f"{name} distance matrix size mismatch: expected ({expected_n}, {expected_n}), got {mat.shape}."
            )

        matrices[name] = mat

    logger.success("✓ All precomputed distance matrices loaded.")
    return matrices


def consolidate_cluster_reports(
    output_path="results/cluster_reports/consolidated_report.json",
    root_dir="results/cluster_reports",
):
    report = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not (fname.endswith(".json") or fname.endswith(".csv")):
                continue

            rel_dir = os.path.relpath(dirpath, root_dir)
            key_prefix = rel_dir.replace(os.sep, "/")
            key = f"{key_prefix}/{fname}".lstrip("./")

            full_path = os.path.join(dirpath, fname)

            if fname.endswith(".json"):
                try:
                    with open(full_path, "r") as f:
                        report[key] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read {full_path}: {e}")
            else:
                try:
                    df = pl.read_csv(full_path)
                    report[key] = df.to_dict(as_series=False)
                except Exception as e:
                    logger.warning(f"Failed to read {full_path}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.success(f"Consolidated report written to {output_path}")


def evaluate_non_euclidean_kmedoids(
    limit=500,
    k_range=range(2, 15),
    include_ph=True,
    use_precomputed_only=False,
    compute_missing=True,
    output_dir="results/cluster_reports/materials_non_euclidean",
):
    os.makedirs(output_dir, exist_ok=True)

    mp = MaterialsProject()
    mp.load(limit=limit)
    frames = _build_mp_frames(mp.df)
    dataset_dir = f"Materials Project/distance_matrices_n{len(frames)}"
    if use_precomputed_only:
        try:
            dist_matrices = load_precomputed_distances(
                dataset_dir=dataset_dir,
                expected_n=len(frames),
                include_ph=include_ph,
            )
        except FileNotFoundError as e:
            if not compute_missing:
                raise
            logger.warning(f"{e} Computing missing matrices now...")
            dist_matrices = get_distances(
                frames,
                dataset=dataset_dir,
                include_ph=include_ph,
            )
    else:
        dist_matrices = get_distances(
            frames,
            dataset=dataset_dir,
            include_ph=include_ph,
        )

    for name, dist_matrix in dist_matrices.items():
        logger.info(f"Evaluating kmedoids for {name}")
        mds = MDS(
            n_components=5,
            metric="precomputed",
            random_state=42,
            n_init=2,
            init="classical_mds",
        )
        X_embedded = mds.fit_transform(dist_matrix)

        eval_out = find_best_kmedoids_k(
            dist_matrix,
            k_range=k_range,
            random_state=42,
            feature_matrix=X_embedded,
        )

        out_base = os.path.join(output_dir, name)
        pl.DataFrame(eval_out["results"]).write_csv(f"{out_base}_kmedoids_eval.csv")
        with open(f"{out_base}_kmedoids_best.json", "w") as f:
            json.dump(eval_out["best_k"], f, indent=2)

        try:
            best_k = eval_out["best_k"].get("silhouette")
            if best_k is None:
                best_k = eval_out["best_k"].get("ch")
            if best_k is None:
                best_k = eval_out["best_k"].get("inertia")

            model = kmedoids.KMedoids(
                n_clusters=int(best_k),
                metric="precomputed",
                random_state=42,
            )
            labels = model.fit_predict(dist_matrix)

            df_subset = mp.df.head(len(frames))
            scores_payload = {
                "embedding": f"{name}_mds",
                "best_k_used": int(best_k),
                "valid_count": int(len(X_embedded)),
                "total_count": int(len(X_embedded)),
            }

            embedding_list = [row.tolist() for row in X_embedded]
            df_scored = df_subset.with_columns(
                [
                    pl.Series("cluster_eval", labels),
                    pl.Series("embedding_eval", embedding_list),
                ]
            )

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            scores_payload["molecular_cluster_score"] = _materials_cluster_score(
                df_scored, X_embedded, labels
            )

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to compute materials cluster scores for {name}: {e}")

        logger.success(f"KMedoids evaluation complete for {name}")


def main():
    print("Running Materials Project cluster report...")
    evaluate_non_euclidean_kmedoids(
        limit=1000,
        k_range=range(2, 15),
        use_precomputed_only=True,
        include_ph=False,
        output_dir="results/cluster_reports/materials_non_euclidean",
    )
    consolidate_cluster_reports()


if __name__ == "__main__":
    main()
