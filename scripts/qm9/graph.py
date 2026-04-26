from src.datasets import QM9Dataset
from src.graph import ChemicalSpaceNetwork
from src.non_euclidean import Wasserstein

def main(distance_matrix, molecule_labels):

    csn = ChemicalSpaceNetwork(distance_matrix, molecule_labels)

    csn.build_knn_graph(
        k=10,
        mutual=False,
        gamma=0.001
    )
    csn.detect_clusters()
    csn.plot_interactive("csn.html")

if __name__ == '__main__':
    descriptor = "soap"

    qm9 = QM9Dataset(limit=250, 
                    sampling_strategy="stratified",
                    stratify_by=["num_atoms", "gap"], 
                    descriptors=["soap"],
                    )
    df = qm9.load()
    soap_matrix = qm9.get_descriptor_matrices("soap")

    wasserstein = Wasserstein()
    dist_matrix = wasserstein.distance_matrix(feature_matrices=soap_matrix)

    main(dist_matrix, df['formula'].to_list())