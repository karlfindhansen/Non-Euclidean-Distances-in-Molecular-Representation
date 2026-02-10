import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    confusion_matrix,
    silhouette_samples
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm

class ClusterEvaluator:
    """
    A class to evaluate and visualize clustering results.
    
    Attributes:
        X (array-like): The feature matrix used for clustering.
        labels_pred (array-like): The cluster labels predicted by the algorithm.
        labels_true (array-like, optional): The ground truth labels (if available).
    """
    
    def __init__(self, X, labels_pred, labels_true=None):
        self.X = X
        self.labels_pred = labels_pred
        self.labels_true = labels_true
        self.n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
        
        # Set style for plots
        sns.set_theme(style="whitegrid")

    def print_metrics(self):
        """Prints standard clustering metrics to the console."""
        print(f"--- Clustering Performance (n_clusters={self.n_clusters}) ---")
        
        # 1. Internal Validation (No Ground Truth needed)
        # Note: These fail if only 1 cluster exists
        if self.n_clusters > 1:
            sil = silhouette_score(self.X, self.labels_pred)
            ch = calinski_harabasz_score(self.X, self.labels_pred)
            db = davies_bouldin_score(self.X, self.labels_pred)
            
            print(f"Silhouette Score:       {sil:.4f} (Higher is better, -1 to 1)")
            print(f"Calinski-Harabasz:    {ch:.1f}  (Higher is better)")
            print(f"Davies-Bouldin:       {db:.4f} (Lower is better)")
        else:
            print("Skipping internal metrics (need > 1 cluster).")

        # 2. External Validation (Requires Ground Truth)
        if self.labels_true is not None:
            ari = adjusted_rand_score(self.labels_true, self.labels_pred)
            nmi = normalized_mutual_info_score(self.labels_true, self.labels_pred)
            
            print(f"Adjusted Rand Index:    {ari:.4f} (1.0 is perfect match)")
            print(f"Normalized Mutual Info: {nmi:.4f} (1.0 is perfect match)")
        else:
            print("External metrics skipped (no ground truth provided).")
        print("-" * 50)

    def plot_dimensionality_reduction(self, method='pca', title=None):
        """
        Visualizes clusters using PCA or t-SNE.
        
        Args:
            method (str): 'pca' or 'tsne'.
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            title = title or "Cluster Visualization (PCA)"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title = title or "Cluster Visualization (t-SNE)"
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        # Reduce dimensions
        X_embedded = reducer.fit_transform(self.X)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=X_embedded[:, 0], 
            y=X_embedded[:, 1], 
            hue=self.labels_pred, 
            palette='viridis', 
            s=60, 
            legend='full'
        )
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title='Cluster')
        plt.show()

    def plot_silhouette_analysis(self):
        """
        Plots the silhouette coefficient for each sample to visualize cluster tightness.
        """
        if self.n_clusters < 2:
            print("Silhouette plot requires at least 2 clusters.")
            return

        silhouette_avg = silhouette_score(self.X, self.labels_pred)
        sample_silhouette_values = silhouette_samples(self.X, self.labels_pred)

        plt.figure(figsize=(10, 6))
        y_lower = 10
        
        # Iterate over clusters to draw the silhouette "knife shapes"
        unique_labels = sorted(set(self.labels_pred))
        for i in unique_labels:
            if i == -1: continue # Skip noise in DBSCAN
            
            ith_cluster_values = sample_silhouette_values[self.labels_pred == i]
            ith_cluster_values.sort()

            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / self.n_clusters)
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the cluster numbers
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # Add space between plots

        plt.title("Silhouette Plot for Various Clusters")
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")
        
        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.show()

    def plot_confusion_matrix(self, title="Cluster vs. Structure Class"):
        """
        Shows exactly which chemical classes (Aromatic, Acyclic) 
        are being grouped into which cluster ID.
        """
        if self.labels_true is None:
            print("Error: Ground truth labels (structure_class) required.")
            return

        # Create cross-tabulation
        data = pd.DataFrame({'Predicted': self.labels_pred, 'True': self.labels_true})
        cm = pd.crosstab(data['True'], data['Predicted'])

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(title, fontweight='bold')
        plt.ylabel('Actual Structure Class')
        plt.xlabel('Cluster ID')
        plt.show()

    def plot_cluster_purity(self):
        """
        Visualizes the 'Concentration' of structure classes per cluster.
        Perfect for seeing if Cluster 2 is 'The Aromatic Cluster'.
        """
        data = pd.DataFrame({'Cluster': self.labels_pred, 'Class': self.labels_true})
        counts = data.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)
        purity = counts.div(counts.sum(axis=1), axis=0) * 100

        purity.plot(kind='bar', stacked=True, colormap='Set3', figsize=(10, 6))
        plt.title("Cluster Composition Purity (%)", fontweight='bold')
        plt.ylabel("Percentage of Molecules")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()