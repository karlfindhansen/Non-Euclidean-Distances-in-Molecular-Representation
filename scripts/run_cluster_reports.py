import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
import json
import os
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import MDS

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import chemiscope
import kmedoids

from src.clusters import ClusterAnalysis, MolecularClusterScore, calculate_congruence
from src.datasets import QM9Dataset, MaterialsProject
from src.helper_functions import get_distances, find_best_kmedoids_k, get_structures


def descriptors():
    loader = QM9Dataset()
    loader.load()
    #soap = SOAPDescriptor(loader, r_cut=6.0, n_max=8)
    #soap.compute()

    #acsf = ACSFDescriptor(loader, r_cut=6.0)
    #acsf.compute()

    cols = [
        "mol_weight", "logp", "tpsa", "num_heavy_atoms", "num_rings", 
        "num_aromatic_rings", "num_rotatable_bonds", "fraction_csp3", 
        "h_bond_donors", "h_bond_acceptors", "mu", "alpha", "homo", 
        "lumo", "gap", "r2", "zpve", "u0", "u", "h", "g", "cv"
    ]

    loader.apply_scaling(cols, mode="fit_transform")

    embeddings = ['soap_embedding','acsf_embedding']
    methods = ['kmeans', 'dbscan', 'hierarchical']

    for embedding in embeddings:
        for method in methods:

            df_clean = loader.df.filter(pl.col("soap_embedding").is_not_null())
            X_soap = np.array(df_clean["soap_embedding"].to_list())
            analyzer = ClusterAnalysis(X_soap, 
                                    true_labels=df_clean["structure_class"], 
                                    meta_df=df_clean)
            
            analyzer.run(method=method)
            analyzer.evaluate()
            analyzer.plot_pca(show=False)
            misclassification_report = analyzer.get_misclassification_report()
            save_path = f'results/cluster_reports/{embedding}/cluster_outlier_report_{method}.csv'
            misclassification_report.write_csv(save_path)

    logger.success("Generated descriptor cluster reports")

def finger_prints():
    loader = QM9Dataset()
    loader.load()
    loader.add_morgan_fingerprints()
    
    loader.add_selfies_transformer()
    loader.add_selfies_onehot()


    X_morgan = np.array(loader.df["morgan_fingerprint"].to_list())
    X_transformer = np.array(loader.df["selfies_transformer"].to_list())
    
    onehot_raw = np.array(loader.df["selfies_onehot"].to_list())
    X_onehot = onehot_raw.reshape(onehot_raw.shape[0], -1)

    true_labels = loader.df["structure_class"].to_list()

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    methods = ['kmeans', 'dbscan', 'hierarchical']
    datasets = {"morgan": X_morgan, "one-hot": X_onehot, "transformer": X_transformer}

    for method in methods:
        for i, (name, X) in enumerate(datasets.items()):
            print(f"\n{'='*10} ANALYZING: {name} {'='*10}")
            
            # 2. Initialize Analyzer
            analyzer = ClusterAnalysis(X, true_labels=true_labels, meta_df=loader.df)
            
            # 3. Run Clustering (e.g., KMeans)
            if method == 'kmeans':
                labels = analyzer.run(method='kmeans', n_clusters=4)
            elif method == 'dbscan':
                labels = analyzer.run(method='dbscan', eps=0.5, min_samples=3)
            elif method == 'hierarchical':
                labels = analyzer.run(method='hierarchical', n_clusters=4, linkage='ward')
            
            if max(labels) == -1:
                print("skipping ", method)
                continue

            # 4. Evaluate
            analyzer.evaluate()

            analyzer.analyze_mismatches()
            
            bad_clusters_df = analyzer.get_misclassification_report(n_neighbors=3)
            save_path = f'results/cluster_reports/{name}/cluster_outlier_report_{method}.csv'
            bad_clusters_df.write_csv(save_path)
    
    logger.success("Generated fingerprint cluster reports")

def interactive_clustering(clustering_method, embedding_type):
    qm9 = QM9Dataset()
    qm9.load()
    if embedding_type == 'soap_embedding':
        qm9.add_soap()
    elif embedding_type == 'acsf_embedding':
        qm9.add_acsf()
    elif embedding_type == 'chemprop_embedding':
        qm9.add_chemprop()
    elif embedding_type == 'morgan_fingerprint':
        qm9.add_morgan_fingerprints()
    elif embedding_type == 'selfies_transformer':
        qm9.add_selfies_transformer()
    elif embedding_type == 'selfies_onehot':
        qm9.add_selfies_onehot()
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    true_labels = qm9.df['structure_class']
    num_clusters = len(set(true_labels))

    X = np.array(qm9.df[embedding_type].to_list(), dtype=np.float32)
    analyzer = ClusterAnalysis(X, true_labels=true_labels, meta_df=qm9.df) 
    _ = analyzer.run(method=clustering_method, n_clusters=num_clusters)
    analyzer.plot_interactive(method='tsne', perplexity=30)


def plot_chemiscope(mp_df, embedding_col="soap_embedding", n_clusters=5):

    df = mp_df.to_pandas()

    # ---- embeddings ----
    X = np.vstack(df[embedding_col].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- clustering ----
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_scaled)

    df["tsne_1"] = X_tsne[:, 0]
    df["tsne_2"] = X_tsne[:, 1]
    df["cluster_id"] = clusters

    # ---- structures ----
    frames = []

    for s_str in df["raw_structure"]:
        pmg = Structure.from_dict(json.loads(s_str))
        atoms = AseAtomsAdaptor.get_atoms(pmg)
        frames.append(atoms)

    # ---- properties ----
    properties = {
        "Cluster": df["cluster_id"].astype(int).values,
        "Energy per Atom": df["energy_per_atom"].astype(float).values,
        "Formula": df["formula_pretty"].astype(str).tolist(),
        "t-SNE 1": df["tsne_1"].values,
        "t-SNE 2": df["tsne_2"].values,
    }

    # ---- chemiscope settings for 2D plot ----
    settings = {
        "map": {
            "x": {"property": "t-SNE 1"},
            "y": {"property": "t-SNE 2"},
            "color": {"property": "Cluster"}
        }
    }
    
    chemiscope.write_input(
        "soap_tsne_clusters.json",
        structures=frames,
        properties=properties,
        settings=settings,
        metadata={
            "name": "SOAP clustering visualization",
            "description": "KMeans clustering with t-SNE projection"
        }
    )

    chemiscope.show_input("soap_tsne_clusters.json")

def evaluate_descriptor_kmeans(
    k_range=range(2, 15),
    sample_size=10000,
    output_dir="results/cluster_reports/descriptors_kmeans",
):
    qm9 = QM9Dataset()
    qm9.load()
    qm9.add_all_descriptors()

    descriptors = [
        "morgan_fingerprint",
        "selfies_onehot",
        "soap_embedding",
        "acsf_embedding",
        "coulomb_matrix",
        "chemprop_embedding",
        "selfies_transformer",
    ]

    os.makedirs(output_dir, exist_ok=True)

    for desc in descriptors:
        if desc not in qm9.df.columns:
            logger.warning(f"Skipping {desc}: column not found.")
            continue

        df = qm9.df.filter(pl.col(desc).is_not_null())
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

        # Cluster quality scores (congruence + MolecularClusterScore) on THIS embedding
        try:
            best_k_used = best_k.get("silhouette") or best_k.get("ch") or best_k.get("inertia")
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

            prop_cols = ["logp", "tpsa", "mol_weight", "homo", "lumo", "num_sp_carbons", "num_sp2_carbons", "num_sp3_carbons", "num_rings"]

            if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                property_matrix = df_scored.select(prop_cols).to_numpy()
                categories = df_scored["functional_groups"].to_list()
                mcs = MolecularClusterScore()
                mcs_result = mcs.compute_total_score(
                    X,
                    labels_best,
                    property_matrix,
                    categories,
                )
                if "structure_class" in df_scored.columns:
                    mcs_result["structure_class_score"] = mcs.compute_category_score(
                        df_scored["structure_class"].to_list(),
                        labels_best,
                    )
                scores_payload["molecular_cluster_score"] = mcs_result
            else:
                scores_payload["molecular_cluster_score"] = {
                    "error": "Missing property or category columns."
                }

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute cluster scores for {desc}: {e}")

        logger.success(f"KMeans evaluation complete for {desc}")


def evaluate_euclidean_agglomerative(
    k_range=range(2, 15),
    output_dir="results/cluster_reports/euclidean_metrics",
):
    qm9 = QM9Dataset()
    qm9.load()

    qm9.add_morgan_fingerprints()
    qm9.add_selfies_transformer()
    qm9.add_chemprop()
    qm9.add_acsf()
    qm9.add_soap()

    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "name": "morgan_fingerprint",
            "prep": lambda X: (X == 1).astype(int),
            "metric": "jaccard",
            "linkage": "average",
        },
        {
            "name": "chemprop_embedding",
            "prep": lambda X: normalize(X, norm="l2", axis=1),
            "metric": "cosine",
            "linkage": "average",
        },
        {
            "name": "selfies_transformer",
            "prep": lambda X: normalize(X, norm="l2", axis=1),
            "metric": "cosine",
            "linkage": "average",
        },
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
        if name not in qm9.df.columns:
            logger.warning(f"Skipping {name}: column not found.")
            continue

        df = qm9.df.filter(pl.col(name).is_not_null())
        if df.is_empty():
            logger.warning(f"Skipping {name}: all values are null.")
            continue

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

        best_k = scores["k"][int(np.argmax(scores["silhouette"]))]

        out_base = os.path.join(output_dir, name)
        pl.DataFrame(scores).write_csv(f"{out_base}_agglo_eval.csv")
        with open(f"{out_base}_agglo_best.json", "w") as f:
            json.dump(
                {
                    "best_k": int(best_k),
                    "metric": cfg["metric"],
                    "linkage": cfg["linkage"],
                },
                f,
                indent=2,
            )

        # Cluster quality scores (congruence + MolecularClusterScore) on THIS embedding
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

            prop_cols = ["logp", "tpsa", "mol_weight", "homo", "lumo", "num_sp_carbons", "num_sp2_carbons", "num_sp3_carbons", "num_rings"]

            if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                property_matrix = df_scored.select(prop_cols).to_numpy()
                categories = df_scored["functional_groups"].to_list()
                mcs = MolecularClusterScore()
                mcs_result = mcs.compute_total_score(
                    X,
                    labels_best,
                    property_matrix,
                    categories,
                )
                if "structure_class" in df_scored.columns:
                    mcs_result["structure_class_score"] = mcs.compute_category_score(
                        df_scored["structure_class"].to_list(),
                        labels_best,
                    )
                scores_payload["molecular_cluster_score"] = mcs_result
            else:
                scores_payload["molecular_cluster_score"] = {
                    "error": "Missing property or category columns."
                }

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute cluster scores for {name}: {e}")

        logger.success(f"Agglomerative evaluation complete for {name}")


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


def evaluate_non_euclidean_kmedoids(
    dataset="materials_project",
    limit=500,
    k_range=range(2, 15),
    include_ph=True,
    use_precomputed_only=False,
    compute_missing=True,
    output_dir="results/cluster_reports/non_euclidean",
):
    os.makedirs(output_dir, exist_ok=True)

    qm9 = None
    if dataset == "materials_project":
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
    else:
        qm9 = QM9Dataset()
        qm9.load()
        frames = qm9.get_positions(subset_size=limit)
        dataset_dir = f"QM9/distance_matrices_n{len(frames)}"
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

        if qm9 is not None:
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

                df_subset = qm9.df.head(len(frames))
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

                prop_cols = ["logp", "tpsa", "mol_weight", "homo", "lumo", "num_sp_carbons", "num_sp2_carbons", "num_sp3_carbons", "num_rings"]
                if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                    property_matrix = df_scored.select(prop_cols).to_numpy()
                    categories = df_scored["functional_groups"].to_list()
                    mcs = MolecularClusterScore()
                    mcs_result = mcs.compute_total_score(
                        X_embedded,
                        labels,
                        property_matrix,
                        categories,
                    )
                    if "structure_class" in df_scored.columns:
                        mcs_result["structure_class_score"] = mcs.compute_category_score(
                            df_scored["structure_class"].to_list(),
                            labels,
                        )
                    scores_payload["molecular_cluster_score"] = mcs_result
                else:
                    scores_payload["molecular_cluster_score"] = {
                        "error": "Missing property or category columns."
                    }

                with open(f"{out_base}_cluster_scores.json", "w") as f:
                    json.dump(scores_payload, f, indent=2)

            except Exception as e:
                logger.warning(f"Failed to compute cluster scores for {name}: {e}")

        logger.success(f"KMedoids evaluation complete for {name}")

def evaluate_isomer_non_euclidean_kmedoids(
    k_range=range(2, 15),
    include_ph=True,
    use_precomputed_only=False,
    compute_missing=True,
    output_dir="results/cluster_reports/isomers_non_euclidean",
    target_formula=None,
):
    qm9 = QM9Dataset()
    qm9.load()
    qm9.add_all_descriptors()

    if "formula" not in qm9.df.columns:
        raise ValueError("QM9 dataframe missing 'formula' column.")

    formula_counts = qm9.df.group_by("formula").len().sort("len", descending=True)
    if formula_counts.is_empty():
        raise ValueError("No formulas found in QM9 dataframe.")

    if target_formula is None:
        target_formula = formula_counts.row(0)[0]

    isomers_df = qm9.df.filter(pl.col("formula") == target_formula)
    if isomers_df.is_empty():
        raise ValueError(f"No isomers found for formula {target_formula}.")

    frames, valid_indices = get_structures(isomers_df)
    if len(frames) < 2:
        logger.warning(
            f"Not enough valid isomer structures for {target_formula} to compute distances."
        )
        return

    if valid_indices and len(valid_indices) != isomers_df.height:
        isomers_df = isomers_df.take(valid_indices)

    dataset_dir = f"QM9/isomers_{target_formula}/distance_matrices_n{len(frames)}"
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

    os.makedirs(output_dir, exist_ok=True)

    for name, dist_matrix in dist_matrices.items():
        logger.info(f"Evaluating isomer kmedoids for {name}")
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

        out_base = os.path.join(output_dir, f"{name}_formula_{target_formula}")
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

            scores_payload = {
                "embedding": f"{name}_mds",
                "formula": target_formula,
                "best_k_used": int(best_k),
                "valid_count": int(len(X_embedded)),
                "total_count": int(len(X_embedded)),
            }

            embedding_list = [row.tolist() for row in X_embedded]
            df_scored = isomers_df.with_columns(
                [
                    pl.Series("cluster_eval", labels),
                    pl.Series("embedding_eval", embedding_list),
                ]
            )

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            prop_cols = [
                "logp",
                "tpsa",
                "mol_weight",
                "homo",
                "lumo",
                "num_sp_carbons",
                "num_sp2_carbons",
                "num_sp3_carbons",
                "num_rings",
            ]
            if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                property_matrix = df_scored.select(prop_cols).to_numpy()
                categories = df_scored["functional_groups"].to_list()
                mcs = MolecularClusterScore()
                mcs_result = mcs.compute_total_score(
                    X_embedded,
                    labels,
                    property_matrix,
                    categories,
                )
                if "structure_class" in df_scored.columns:
                    mcs_result["structure_class_score"] = mcs.compute_category_score(
                        df_scored["structure_class"].to_list(),
                        labels,
                    )
                scores_payload["molecular_cluster_score"] = mcs_result
            else:
                scores_payload["molecular_cluster_score"] = {
                    "error": "Missing property or category columns."
                }

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to compute isomer cluster scores for {name}: {e}")

        logger.success(f"Isomer non-euclidean evaluation complete for {name}")

def evaluate_isomer_euclidean_agglomerative(
    k_range=range(2, 15),
    output_dir="results/cluster_reports/isomers_euclidean",
    target_formula=None,
):
    qm9 = QM9Dataset()
    qm9.load()

    qm9.add_morgan_fingerprints()
    qm9.add_selfies_transformer()
    qm9.add_chemprop()
    qm9.add_acsf()
    qm9.add_soap()

    if "formula" not in qm9.df.columns:
        raise ValueError("QM9 dataframe missing 'formula' column.")

    formula_counts = qm9.df.group_by("formula").len().sort("len", descending=True)
    if formula_counts.is_empty():
        raise ValueError("No formulas found in QM9 dataframe.")

    if target_formula is None:
        target_formula = formula_counts.row(0)[0]

    isomers_df = qm9.df.filter(pl.col("formula") == target_formula)
    if isomers_df.is_empty():
        raise ValueError(f"No isomers found for formula {target_formula}.")

    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "name": "morgan_fingerprint",
            "prep": lambda X: (X == 1).astype(int),
            "metric": "jaccard",
            "linkage": "average",
        },
        {
            "name": "chemprop_embedding",
            "prep": lambda X: normalize(X, norm="l2", axis=1),
            "metric": "cosine",
            "linkage": "average",
        },
        {
            "name": "selfies_transformer",
            "prep": lambda X: normalize(X, norm="l2", axis=1),
            "metric": "cosine",
            "linkage": "average",
        },
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
        if name not in isomers_df.columns:
            logger.warning(f"Skipping {name}: column not found.")
            continue

        df = isomers_df.filter(pl.col(name).is_not_null())
        if df.is_empty():
            logger.warning(f"Skipping {name}: all values are null.")
            continue

        X = np.stack(df[name].to_list())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = cfg["prep"](X)

        if X.shape[0] < 3:
            logger.warning(f"Skipping {name}: not enough samples for silhouette.")
            continue

        k_list = [int(k) for k in k_range if 2 <= int(k) < X.shape[0]]
        if not k_list:
            logger.warning(f"Skipping {name}: no valid k in k_range for n={X.shape[0]}.")
            continue

        scores = {"k": [], "silhouette": []}

        for k in k_list:
            model = AgglomerativeClustering(
                n_clusters=int(k),
                metric=cfg["metric"],
                linkage=cfg["linkage"],
            )
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels, metric=cfg["metric"])
            scores["k"].append(int(k))
            scores["silhouette"].append(float(score))

        best_k = scores["k"][int(np.argmax(scores["silhouette"]))]

        out_base = os.path.join(output_dir, f"{name}_formula_{target_formula}")
        pl.DataFrame(scores).write_csv(f"{out_base}_agglo_eval.csv")
        with open(f"{out_base}_agglo_best.json", "w") as f:
            json.dump(
                {
                    "best_k": int(best_k),
                    "metric": cfg["metric"],
                    "linkage": cfg["linkage"],
                    "formula": target_formula,
                },
                f,
                indent=2,
            )

        # Cluster quality scores (congruence + MolecularClusterScore) on THIS embedding
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
                "formula": target_formula,
                "best_k_used": int(best_k),
                "valid_count": int(len(embedding_list)),
                "total_count": int(len(embedding_list)),
            }

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            prop_cols = ["logp", "tpsa", "mol_weight", "homo", "lumo", "num_sp_carbons", "num_sp2_carbons", "num_sp3_carbons", "num_rings"]

            if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                property_matrix = df_scored.select(prop_cols).to_numpy()
                categories = df_scored["functional_groups"].to_list()
                mcs = MolecularClusterScore()
                mcs_result = mcs.compute_total_score(
                    X,
                    labels_best,
                    property_matrix,
                    categories,
                )
                if "structure_class" in df_scored.columns:
                    mcs_result["structure_class_score"] = mcs.compute_category_score(
                        df_scored["structure_class"].to_list(),
                        labels_best,
                    )
                scores_payload["molecular_cluster_score"] = mcs_result
            else:
                scores_payload["molecular_cluster_score"] = {
                    "error": "Missing property or category columns."
                }

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute isomer euclidean scores for {name}: {e}")

        logger.success(f"Isomer euclidean evaluation complete for {name}")

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


def plot_cluster_evaluations(
    root_dir="results/cluster_reports",
    output_dir="results/cluster_reports/plots",
):
    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith("_eval.csv"):
                continue

            full_path = os.path.join(dirpath, fname)
            try:
                df = pl.read_csv(full_path)
            except Exception as e:
                logger.warning(f"Failed to read {full_path}: {e}")
                continue

            cols = set(df.columns)
            if "k" not in cols:
                logger.warning(f"Skipping {full_path}: missing 'k' column.")
                continue

            k = df["k"].to_list()
            title = os.path.splitext(fname)[0].replace("_", " ")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(title, fontsize=14, fontweight="bold")

            if "inertia" in cols:
                axes[0].plot(k, df["inertia"].to_list(), marker="o", color="tab:blue")
                axes[0].set_title("Inertia")
                axes[0].set_xlabel("k")
            else:
                axes[0].axis("off")

            if "silhouette" in cols:
                axes[1].plot(k, df["silhouette"].to_list(), marker="o", color="tab:green")
                axes[1].set_title("Silhouette")
                axes[1].set_xlabel("k")
            else:
                axes[1].axis("off")

            if "ch" in cols:
                axes[2].plot(k, df["ch"].to_list(), marker="o", color="tab:red")
                axes[2].set_title("Calinski-Harabasz")
                axes[2].set_xlabel("k")
            else:
                axes[2].axis("off")

            plt.tight_layout()

            rel_dir = os.path.relpath(dirpath, root_dir)
            rel_dir = rel_dir.replace(os.sep, "_").strip(".")
            out_name = f"{rel_dir}__{fname.replace('.csv', '')}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.savefig(out_path, dpi=200)
            plt.close(fig)

            logger.success(f"Saved plot: {out_path}")

def evaluate_isomer_embeddings(
    k_range=range(2, 15),
    output_dir="results/cluster_reports/isomers",
):
    qm9 = QM9Dataset()
    qm9.load()
    qm9.add_all_descriptors()

    if "formula" not in qm9.df.columns:
        raise ValueError("QM9 dataframe missing 'formula' column.")

    formula_counts = qm9.df.group_by("formula").len().sort("len", descending=True)
    if formula_counts.is_empty():
        raise ValueError("No formulas found in QM9 dataframe.")

    target_formula = formula_counts.row(0)[0]
    isomers_df = qm9.df.filter(pl.col("formula") == target_formula)

    os.makedirs(output_dir, exist_ok=True)

    descriptors = [
        "morgan_fingerprint",
        "selfies_onehot",
        "soap_embedding",
        "acsf_embedding",
        "coulomb_matrix",
        "chemprop_embedding",
        "selfies_transformer",
    ]

    for desc in descriptors:
        if desc not in isomers_df.columns:
            logger.warning(f"Skipping {desc}: column not found.")
            continue

        df = isomers_df.filter(pl.col(desc).is_not_null())
        if df.is_empty():
            logger.warning(f"Skipping {desc}: all values are null.")
            continue

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

        out_base = os.path.join(output_dir, f"{desc}_formula_{target_formula}")
        pl.DataFrame(results).write_csv(f"{out_base}_kmeans_eval.csv")
        with open(f"{out_base}_kmeans_best.json", "w") as f:
            json.dump(best_k, f, indent=2)

        # Cluster quality scores on this embedding
        try:
            best_k_used = best_k.get("silhouette") or best_k.get("ch") or best_k.get("inertia")
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
                "formula": target_formula,
                "best_k_used": int(best_k_used),
                "valid_count": int(len(embedding_list)),
                "total_count": int(len(embedding_list)),
            }

            scores_payload["congruence"] = calculate_congruence(
                df_scored, "cluster_eval", embedding_col="embedding_eval"
            )

            prop_cols = ["logp", "tpsa", "mol_weight", "homo", "lumo", "num_sp_carbons", "num_sp2_carbons", "num_sp3_carbons", "num_rings"]
            if all(c in df_scored.columns for c in prop_cols) and "functional_groups" in df_scored.columns:
                property_matrix = df_scored.select(prop_cols).to_numpy()
                categories = df_scored["functional_groups"].to_list()
                mcs = MolecularClusterScore()
                mcs_result = mcs.compute_total_score(
                    X,
                    labels_best,
                    property_matrix,
                    categories,
                )
                if "structure_class" in df_scored.columns:
                    mcs_result["structure_class_score"] = mcs.compute_category_score(
                        df_scored["structure_class"].to_list(),
                        labels_best,
                    )
                scores_payload["molecular_cluster_score"] = mcs_result
            else:
                scores_payload["molecular_cluster_score"] = {
                    "error": "Missing property or category columns."
                }

            with open(f"{out_base}_cluster_scores.json", "w") as f:
                json.dump(scores_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to compute cluster scores for {desc}: {e}")

        logger.success(f"Isomer KMeans evaluation complete for {desc}")

if __name__ == "__main__":
    #descriptors()
    #finger_prints()
    evaluate_isomer_embeddings()
    evaluate_isomer_non_euclidean_kmedoids(
        k_range=range(2, 15),
        use_precomputed_only=False,
        include_ph=False,
    )
    evaluate_isomer_euclidean_agglomerative(
        k_range=range(2, 15),
    )

    evaluate_descriptor_kmeans()
    evaluate_euclidean_agglomerative()
    evaluate_non_euclidean_kmedoids(
        dataset="materials_project",
        limit=1000,
        k_range=range(2, 15),
        use_precomputed_only=True,
        include_ph=False,
    )
    evaluate_non_euclidean_kmedoids(
        dataset="qm9",
        limit=2000,
        k_range=range(2, 15),
        use_precomputed_only=True,
        include_ph=False,
    )
    consolidate_cluster_reports()
    plot_cluster_evaluations()

    clustering_method = 'kmeans'
    embedding_type = 'morgan_fingerprint'
    #interactive_clustering(clustering_method, embedding_type)

    mp = MaterialsProject()
    mp.load()
    #plot_chemiscope(mp.df, "acsf_embedding")
        
