import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collections import Counter

class ClusterAnalysis:
    def __init__(self, X, true_labels=None, meta_df=None):
        """
        Initialize the analysis with a feature matrix.
        
        Args:
            X (np.array): Feature matrix (n_samples, n_features).
            true_labels (list/array, optional): Ground truth labels for external evaluation.
            meta_df (pl.DataFrame, optional): Metadata (smiles, ids) for reporting.
        """
        self.X = X
        self.true_labels = true_labels
        self.meta_df = meta_df
        self.labels_ = None
        self.model_ = None
        self.method_name_ = ""

    def run(self, method='kmeans', **kwargs):
        """
        Run a specific clustering algorithm.
        """
        self.method_name_ = method.lower()
        print(f"--- Running {self.method_name_.upper()} ---")

        if self.method_name_ == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 5)
            self.model_ = KMeans(n_clusters=n_clusters, 
                                 random_state=kwargs.get('random_state', 42),
                                 n_init=kwargs.get('n_init', 10))
            self.labels_ = self.model_.fit_predict(self.X)
            
        elif self.method_name_ == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            self.model_ = DBSCAN(eps=eps, min_samples=min_samples)
            self.labels_ = self.model_.fit_predict(self.X)
            
        elif self.method_name_ == 'hierarchical':
            n_clusters = kwargs.get('n_clusters', 5)
            linkage = kwargs.get('linkage', 'ward')
            self.model_ = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            self.labels_ = self.model_.fit_predict(self.X)
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'kmeans', 'dbscan', or 'hierarchical'.")
        
        return self.labels_

    def evaluate(self):
        """
        Calculates and prints internal and external clustering metrics.
        """
        if self.labels_ is None:
            print("Run clustering first.")
            return

        unique_labels = set(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in self.labels_ else 0)
        print(f"Found {n_clusters} clusters (excluding noise).")

        metrics = {}
        
        # 1. External Metrics (Requires Ground Truth)
        if self.true_labels is not None:
            ari = adjusted_rand_score(self.true_labels, self.labels_)
            metrics['ARI'] = ari
            print(f"Adjusted Rand Index (Ground Truth): {ari:.4f}")

        # 2. Internal Metrics
        if n_clusters > 1:
            sil = silhouette_score(self.X, self.labels_)
            ch = calinski_harabasz_score(self.X, self.labels_)
            metrics['Silhouette'] = sil
            print(f"Silhouette Score: {sil:.4f}")
            print(f"Calinski-Harabasz Score: {ch:.4f}")
        else:
            print("Not enough clusters for internal metrics.")

        return metrics
    
    def calculate_overlap_detailed(self, k=20, use_pca=False):
        """
        Calculates overlap score and identifies the specific interfering cluster.
        
        Args:
            k (int): Number of neighbors to check.
            use_pca (bool): 
                - True: Checks overlap in 2D PCA space (evaluates the PLOT).
                - False: Checks overlap in original High-D space (evaluates the EMBEDDING).
        
        Returns:
            overlap_scores (np.array): 0.0 to 1.0 score of how "misplaced" the point is.
            dominant_neighbors (list): The class label of the cluster this point is overlapping with.
        """
        if self.true_labels is None:
            return None, None

        # 1. Choose Space
        if use_pca:
            # Evaluates the visualization artifacts
            pca = PCA(n_components=2)
            X_space = pca.fit_transform(self.X)
        else:
            # Evaluates the actual chemical descriptor quality
            X_space = self.X

        # 2. Find Neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_space)
        _, indices = nbrs.kneighbors(X_space)

        overlap_scores = []
        dominant_neighbors = []
        
        # Handle Polars vs Numpy input
        true_labels_np = self.true_labels.to_numpy() if isinstance(self.true_labels, pl.Series) else np.array(self.true_labels)

        for i, neighbor_indices in enumerate(indices):
            # neighbor_indices[0] is the point itself; skip it
            others = neighbor_indices[1:]
            
            own_class = true_labels_np[i]
            neighbor_classes = true_labels_np[others]
            
            # A. Calculate Score (% mismatch)
            # 0.0 = Perfect cluster, 1.0 = Completely surrounded by enemies
            score = np.mean(neighbor_classes != own_class)
            overlap_scores.append(score)
            
            # B. Identify Dominant Neighbor (The "Who")
            if score > 0: 
                # Find which class is invading this neighborhood most often
                counts = Counter(neighbor_classes)
                # Remove own class from counts to find the *interfering* class
                if own_class in counts:
                    del counts[own_class]
                
                if counts:
                    most_common_invader = counts.most_common(1)[0][0]
                    dominant_neighbors.append(most_common_invader)
                else:
                    dominant_neighbors.append(None) # Only neighbors were own class (score was 0)
            else:
                dominant_neighbors.append(None)

        return np.array(overlap_scores), dominant_neighbors

    def analyze_mismatches(self):
        """
        Identifies molecules that appear to be in the wrong cluster 
        (Visual, Mathematical, or Label mismatch).
        
        Returns:
            visual_mismatch, math_mismatch, label_mismatch (Polars DataFrames)
        """
        if self.labels_ is None:
            print("Run clustering first.")
            return None, None, None

        # 1. Prepare Data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        # Start with meta_df if exists, else create empty
        if self.meta_df is not None:
            results = self.meta_df.clone()
        else:
            results = pl.DataFrame()

        # 2. Calculate Metrics
        # A. Visual Neighbors (KNN)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_pca, self.labels_)
        visual_pred = knn.predict(X_pca)
        
        # B. Silhouette Scores per sample
        sil_scores = silhouette_samples(self.X, self.labels_)
        
        # Combine into DataFrame
        results = results.with_columns([
            pl.Series("cluster", self.labels_),
            pl.Series("visual_neighbor_cluster", visual_pred),
            pl.Series("silhouette_score", sil_scores),
            pl.Series("pca_x", X_pca[:,0]),
            pl.Series("pca_y", X_pca[:,1])
        ])
        
        if self.true_labels is not None:
             results = results.with_columns(pl.Series("true_label", self.true_labels))

        print("\n--- Mismatch Analysis ---")
        
        # 3. Filter Results
        vis_err = results.filter(pl.col("cluster") != pl.col("visual_neighbor_cluster"))
        print(f"Visual Intruders: {len(vis_err)} (Look like they belong elsewhere)")

        math_err = results.filter(pl.col("silhouette_score") < 0)
        print(f"Silhouette Outliers: {len(math_err)} (Ambiguous assignment)")

        chem_err = None
        if self.true_labels is not None:
            dom_classes = (
                results.group_by("cluster")
                .agg(pl.col("true_label").mode().first().alias("dominant_class"))
            )
            chem_err = (
                results.join(dom_classes, on="cluster")
                .filter(pl.col("true_label") != pl.col("dominant_class"))
                .sort("silhouette_score")
            )
            print(f"Label Mismatches: {len(chem_err)} (Don't match cluster's dominant class)")
            
        return vis_err, math_err, chem_err
    
    def get_misclassification_report(self, n_neighbors=3, id_col='mol_id', smiles_col='canonical_smiles'):
        """
        Generates a detailed report of misplaced molecules, including SMILES
        for visual comparison with neighbors.
        """
        if self.labels_ is None or self.meta_df is None:
            print("Error: Run clustering first and ensure meta_df was provided.")
            return None

        # 1. Setup Data
        report = self.meta_df.clone()
        
        # Add Clustering Info
        report = report.with_columns([
            pl.Series("Assigned_Cluster", self.labels_),
        ])
        
        if self.true_labels is not None:
            report = report.with_columns(pl.Series("True_Class", self.true_labels))

        # 2. Find Neighbors
        print(f"Finding top {n_neighbors} neighbors for every molecule...")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(self.X)
        distances, indices = nbrs.kneighbors(self.X)
        
        # 3. Retrieve IDs and SMILES for lookup
        try:
            ids = self.meta_df[id_col].to_list()
            smiles_list = self.meta_df[smiles_col].to_list()
        except Exception as e:
            print(f"Error accessing columns: {e}. Check if '{id_col}' and '{smiles_col}' exist in your DF.")
            return None

        # 4. format Neighbor Strings
        neighbor_info = []
        
        for row_idx in range(len(indices)):
            neighbor_idxs = indices[row_idx, 1:]
            
            info_parts = []
            for i in neighbor_idxs:
                n_id = str(ids[i])
                n_smiles = str(smiles_list[i])
                info_parts.append(f"{n_id} ({n_smiles})")
                
            neighbor_info.append(" || ".join(info_parts))
            
        report = report.with_columns(pl.Series("Closest_Neighbors_Info", neighbor_info))

        # 5. Filter for Mismatches
        if self.true_labels is not None:
            dom_classes = (
                report.group_by("Assigned_Cluster")
                .agg(pl.col("True_Class").mode().first().alias("Cluster_Dominant_Class"))
            )
            
            mismatches = (
                report.join(dom_classes, on="Assigned_Cluster")
                .filter(pl.col("True_Class") != pl.col("Cluster_Dominant_Class"))
                .select([
                    id_col,
                    smiles_col,         
                    "True_Class", 
                    "Assigned_Cluster", 
                    "Cluster_Dominant_Class", 
                    "Closest_Neighbors_Info"
                ])
                .sort("Assigned_Cluster")
            )
            
            print(f"Found {len(mismatches)} mismatches.")
            return mismatches
        
        return report
    
    def plot_pca(self, show=False, title_suffix="", highlight_top_overlaps=5, use_pca=False):
        """
        Visualizes the clustering using PCA (2D) with overlap highlighting.
        Labels clusters by their dominant true class and purity percentage.
        """
        if self.labels_ is None:
            print("Run clustering first.")
            return

        # 1. Generate PCA coordinates
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        plt.figure(figsize=(12, 8))
        
        # 2. Prepare True Labels for fast indexing
        # Convert to numpy array once to avoid Polars indexing issues inside the loop
        if self.true_labels is not None:
            if hasattr(self.true_labels, 'to_numpy'):
                true_labels_np = self.true_labels.to_numpy()
            else:
                true_labels_np = np.array(self.true_labels)
        else:
            true_labels_np = None

        # 3. Plot Each Cluster
        unique_labels = np.unique(self.labels_)
        
        # Create a colormap
        cmap = plt.get_cmap('tab10') if len(unique_labels) <= 10 else plt.get_cmap('viridis')
        
        for i, k in enumerate(unique_labels):
            # Create a mask for points in this cluster
            mask = (self.labels_ == k)
            
            # --- NEW: Calculate Dominant Class Logic ---
            if k == -1:
                col = 'k'; marker = 'x'; label = 'Noise'; alpha = 0.3
            else:
                col = cmap(i % 10); marker = 'o'; alpha = 0.6
                
                if true_labels_np is not None:
                    # Get the true labels for points in THIS cluster
                    cluster_true_labels = true_labels_np[mask]
                    
                    # Find dominant class
                    counts = Counter(cluster_true_labels)
                    dominant_class, count = counts.most_common(1)[0]
                    total = len(cluster_true_labels)
                    percentage = (count / total) * 100
                    
                    # Set the label
                    label = f"{dominant_class} ({percentage:.1f}%)"
                else:
                    label = f"Cluster {k}"
            # -------------------------------------------

            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=[col], label=label, marker=marker, alpha=alpha, s=60)

        # 4. Highlight Top Overlaps
        if true_labels_np is not None and highlight_top_overlaps > 0:
            scores, _ = self.calculate_overlap_detailed(k=20, use_pca=use_pca)
            top_indices = np.argsort(-scores)[:highlight_top_overlaps]
            
            print(f"\n--- Highlighting Top {highlight_top_overlaps} Overlapping Molecules ---")
            
            for idx in top_indices:
                score = scores[idx]
                if score == 0: continue 
                
                # FIX: Cast numpy int64 to python int
                py_idx = int(idx)
                
                x_coord, y_coord = X_pca[py_idx, 0], X_pca[py_idx, 1]
                
                # Retrieve Label/ID for annotation
                mol_id = "Unknown"
                if self.meta_df is not None and "mol_id" in self.meta_df.columns:
                    mol_id = self.meta_df["canonical_smiles"][py_idx]
                
                true_lbl = true_labels_np[py_idx]
                
                print(f"ID: {mol_id} | True: {true_lbl} | Overlap Score: {score:.2f}")

                plt.scatter(x_coord, y_coord, facecolors='none', edgecolors='red', s=200, linewidth=2, zorder=10)
                plt.text(x_coord + 0.05, y_coord + 0.05, f"{mol_id}\n({score:.2f})", fontsize=9, color='darkred', weight='bold', zorder=11)

        plt.title(f"{self.method_name_.upper()} Clustering (PCA)\n{title_suffix}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster (Dominant Class)")
        plt.tight_layout()
        
        if show:
            plt.show()

    def get_summary_df(self):
        """Returns summary dataframe."""
        if self.meta_df is None:
            return pl.DataFrame({"cluster": self.labels_})