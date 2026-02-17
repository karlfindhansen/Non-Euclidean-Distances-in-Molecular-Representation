import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from src.descriptors import SOAPDescriptor, ACSFDescriptor
from src.clusters import ClusterAnalysis
from src.datasets import QM9Dataset


def descriptors():
    loader = QM9Dataset()
    loader.load()
    soap = SOAPDescriptor(loader, r_cut=6.0, n_max=8)
    soap.compute()

    acsf = ACSFDescriptor(loader, r_cut=6.0)
    acsf.compute()

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
    qm9.add_morgan_fingerprints()

    true_labels = qm9.df['structure_class']
    num_clusters = len(set(true_labels))

    X = np.array(qm9.df["morgan_fingerprint"].to_list(), dtype=np.float32)
    analyzer = ClusterAnalysis(X, true_labels=true_labels, meta_df=qm9.df) 
    _ = analyzer.run(method='kmeans', n_clusters=num_clusters)
    analyzer.plot_interactive(method='tsne', perplexity=30)

    

if __name__ == "__main__":
    #descriptors()
    #finger_prints()

    clustering_method = 'kmeans'
    embedding_type = 'morgan_fingerprint'
    interactive_clustering(clustering_method, embedding_type)
    