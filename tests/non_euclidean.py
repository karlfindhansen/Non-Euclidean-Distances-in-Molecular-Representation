import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
import persim
import math

from src.non_euclidean import Wasserstein, PersistentHomology, Grassmann, Riemann
from src.datasets import QM9Dataset
from src.features import get_raw_xyz_features, get_weighted_point_clouds

def plot_distance_matrix(dist_matrix: np.ndarray, title: str = "Distance Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    plt.show()

def plot_stress_test_comparison(clean_dgm, noisy_dgm):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    persim.plot_diagrams(list(clean_dgm.values()), ax=ax[0])
    ax[0].set_title("Clean Molecule")
    
    persim.plot_diagrams(list(noisy_dgm.values()), ax=ax[1])
    ax[1].set_title("Noisy Molecule (Rattle 0.5Å)")
    
    plt.tight_layout()
    plt.show()

def plot_all_distance_matrices(**matrices):
    """
    Plots an arbitrary number of distance matrices in a grid.
    Usage: plot_all_distance_matrices(Wasserstein=wasserstein_matrix, Grassmann=g_matrix)
    """
    num_plots = len(matrices)
    if num_plots == 0:
        print("No matrices provided to plot.")
        return
    
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    
    if num_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, (name, matrix) in enumerate(matrices.items()):
        im = axes_flat[i].imshow(matrix, cmap='viridis')
        axes_flat[i].set_title(f"{name} Distance Matrix")
        axes_flat[i].set_xlabel('Frame Index')
        axes_flat[i].set_ylabel('Frame Index')
        plt.colorbar(im, ax=axes_flat[i], label='Distance')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    qm9 = QM9Dataset()
    qm9.load()

    frames = qm9.run_stress_test()
    
    xyz_features = get_raw_xyz_features(frames)
    point_clouds = get_weighted_point_clouds(frames)

    dist_matrix_xyz = pairwise_distances(xyz_features, metric='euclidean')
    dist_matrix_point_cloud = pairwise_distances(point_clouds, metric='euclidean')



    wasserstein_dist_matrix = Wasserstein.distance_matrix(frames)

    persistence_diagrams = PersistentHomology.compute_persistence_diagrams(frames, max_homology_dim=2)
    persistance_dist_matrix = PersistentHomology.distance_matrix(frames)

    grassmann_dist_matrix = Grassmann.distance_matrix(frames, method='qr')
    # plot_distance_matrix(grassmann_dist_matrix, title="Grassmann Distance Matrix")
   
    riemann_dist_matrix = Riemann.distance_matrix(frames)

    #plot_distance_matrix(riemann_dist_matrix, title="Riemannian Distance Matrix")
    plot_all_distance_matrices(
                                Wasserstein=wasserstein_dist_matrix,
                                PersistentHomology=persistance_dist_matrix,
                                Grassmann=grassmann_dist_matrix, 
                                Riemannian=riemann_dist_matrix,
                                XYZ=dist_matrix_xyz,
                                PointCloud=dist_matrix_point_cloud
                               )