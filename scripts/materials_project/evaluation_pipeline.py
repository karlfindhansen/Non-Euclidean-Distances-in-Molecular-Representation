import json
from kmedoids import KMedoids
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd


from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Element

from src.datasets import MaterialsProject
from src.non_euclidean import _compute_invariant_feature_matrix as invariant_matrix
from src.non_euclidean import _compute_soap_feature_matrices as soap_matrix
from src.non_euclidean import Grassmann
from src.helper_functions import create_chemiscope_viewer

def plot_evaluation(res):
    n = 25
    x_range = np.arange(2, n + 2)
    
    # Define metrics and whether we want the max or min
    metrics = [
        ('sil', 'Silhouette Score', 'max'),
        ('ch', 'Calinski-Harabasz Index', 'max'),
        ('db', 'Davies-Bouldin Index', 'min'),
        ('chemical_score', 'Overall Chemical Coherence', 'max')
    ]

    plt.figure(figsize=(20, 5))

    for i, (key, title, goal) in enumerate(metrics, 1):
        plt.subplot(1, 4, i)
        data = np.array(res[key][:n])
        plt.plot(x_range, data, label=title)
        
        # Find the best index based on the goal (max or min)
        if goal == 'max':
            best_idx = np.argmax(data)
        else:
            best_idx = np.argmin(data)
            
        best_x = x_range[best_idx]
        best_y = data[best_idx]

        # Add the point and the text
        plt.scatter(best_x, best_y, color='red', zorder=5)
        plt.annotate(
            f'Best: {best_x}', 
            xy=(best_x, best_y), 
            xytext=(best_x + 1, best_y),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10,
            fontweight='bold'
        )

        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig('figures/materials/clustering/evaluation_soap_matrix.png', dpi=300)
    plt.show()

def run_evaluation(dist_matrix, df):
    res = {'sil': [], 'ch': [], 'db': [], 'chemical_score': []}
    for i in tqdm(range(2, 100), desc='Clustering'):
        labels = hierachial_clustering(dist_matrix, i)
        sil, ch, db = evaluation(distance_soap_matrix, labels)
        chemical_score = get_overall_chemical_coherence(df, labels)
        res['sil'].append(sil)
        res['ch'].append(ch)
        res['db'].append(db)
        res['chemical_score'].append(chemical_score)
        
    plot_evaluation(res)
    return res

def evaluation(dist_matrix, labels):
    sil = silhouette_score(dist_matrix, labels, metric='precomputed')
    ch = calinski_harabasz_score(dist_matrix, labels)
    db = davies_bouldin_score(dist_matrix, labels)
    return sil, ch, db

def get_overall_chemical_coherence(df, labels):
    """
    Returns a single score between 0 and 1 indicating how well the 
    structural clusters group materials with similar chemical/physical properties.
    """
    features = ['band_gap', 'density', 'energy_per_atom', 'formation_energy_per_atom']
    
    # Create a DataFrame for easy grouping
    eval_df = pd.DataFrame({f: df[f].to_list() for f in features})
    eval_df['cluster'] = labels
    
    coherence_scores = []
    
    for feature in features:
        global_std = eval_df[feature].std()
        
        if global_std > 0:
            # Size-weighted average of the standard deviation within each cluster
            cluster_stds = eval_df.groupby('cluster')[feature].std().fillna(0)
            cluster_counts = eval_df.groupby('cluster')[feature].count()
            
            weighted_intra_std = np.sum(cluster_stds * cluster_counts) / eval_df.shape[0]
            
            # Coherence: 1.0 means perfect grouping (0 variance within clusters)
            coherence = 1.0 - (weighted_intra_std / global_std)
            coherence_scores.append(max(0.0, coherence)) # Floor at 0 just in case of weird distributions
        else:
            coherence_scores.append(0.0)
            
    # The single number representing overall chemical coherence
    overall_score = np.mean(coherence_scores)
    
    return overall_score

def _compute_invariant_feature_matrix_materials(frame: Atoms, cutoff: float = 3.0) -> np.ndarray:
    """
    Maps a periodic material to a D x N matrix of invariant physical features.
    D is the fixed ambient dimension. N is the number of atoms in the unit/super cell.
    """
    # 1. Periodic neighbor list calculation
    # "ijd" returns: center atom index, neighbor atom index, and distance
    # ASE automatically respects the periodic boundary conditions (pbc=True) of the frame
    i_list, j_list, d_list = neighbor_list("ijd", frame, cutoff)

    neighbors = {i: [] for i in range(len(frame))}
    distances = {i: [] for i in range(len(frame))}

    for i, j, d in zip(i_list, j_list, d_list):
        neighbors[i].append(frame[j].number)
        distances[i].append(d)

    features = []
    
    # 2. Global invariant: Volume per atom
    # Replaces the Center of Mass distance, providing a scale-invariant packing metric
    vol_per_atom = frame.get_volume() / len(frame)

    # 3. Iterate over atoms to build the invariant matrix
    for i, atom in enumerate(frame):
        z = atom.number
        el = Element.from_Z(z)
        
        # Safely extract elemental properties (some noble gases lack Pauling electronegativity)
        en = el.X if getattr(el, 'X', None) else 0.0
        rad = el.atomic_radius if getattr(el, 'atomic_radius', None) else 0.0
        mass = atom.mass

        # 4. Local geometric invariants
        coord = len(neighbors[i])

        if coord > 0:
            avg_neighbor_z = float(np.mean(neighbors[i]))
            avg_neighbor_dist = float(np.mean(distances[i]))
        else:
            avg_neighbor_z = 0.0
            avg_neighbor_dist = 0.0

        # Assemble D-dimensional feature vector (D=8)
        feat_vector = [
            z,                  # 1. Atomic number
            rad,                # 2. Atomic radius
            en,                 # 3. Electronegativity
            mass,               # 4. Atomic mass
            vol_per_atom,       # 5. Packing density (Global)
            coord,              # 6. Local coordination number
            avg_neighbor_z,     # 7. Chemical environment (Avg neighbor Z)
            avg_neighbor_dist   # 8. Structural environment (Avg bond length)
        ]

        features.append(feat_vector)

    # Return transposed to match D x N shape requirements
    return np.array(features).T

def build_invariant_matrix(df, cutoff: float = 3.0):
    """
    Iterates through the materials dataframe, converts JSON structures to ASE Atoms, 
    and computes the D x N invariant feature matrix for each material.
    """
    adaptor = AseAtomsAdaptor()
    invariant_matrices = []
    
    for struct_json in df["raw_structure"]:
        # 1. Reconstruct the Pymatgen Structure
        struct = Structure.from_dict(json.loads(struct_json))
        
        # 2. Convert to ASE Atoms (this automatically preserves periodic boundary conditions)
        atoms = adaptor.get_atoms(struct)
        
        # 3. Compute the D x N invariant matrix
        matrix = _compute_invariant_feature_matrix_materials(atoms, cutoff=cutoff)
        
        invariant_matrices.append(matrix)
        
    return invariant_matrices


def hierachial_clustering(dist_matrix, n_clusters):

    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='precomputed')
    labels = hierarchical_cluster.fit_predict(dist_matrix)

    return labels

def kmedoids_clustering(dist_matrix, n_clusters):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed')
    labels = kmedoids.fit_predict(dist_matrix)
    return labels

if __name__ == '__main__':

    mp = MaterialsProject(add_soap_acsf=True)
    df = mp.load(limit=1000)

    print(df['acsf_embedding', 'soap_embedding'].head(3))
    print(df.columns)
    df = df.drop('acsf_embedding')
    
    soap_matrix = np.array(df['soap_embedding'].to_list())
    distance_soap_matrix = squareform(pdist(soap_matrix, metric='euclidean'))
    run_evaluation(distance_soap_matrix, df) # best number of clusters : 4
    hier_labels = hierachial_clustering(distance_soap_matrix, 4) 

    materials_invariant_matrix = build_invariant_matrix(df, cutoff=3.0)
    grassmann_dist_matrix = Grassmann.distance_matrix(precomputed_matrices=materials_invariant_matrix)
    kmedoids_labels = kmedoids_clustering(grassmann_dist_matrix, 4)

    