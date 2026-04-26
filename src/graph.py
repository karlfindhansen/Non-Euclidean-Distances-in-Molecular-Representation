import os
import numpy as np
import webbrowser
import networkx as nx
from typing import Optional, List, Dict, Union
from loguru import logger

# Community detection
try:
    import community as community_louvain  # pip install python-louvain
except ImportError:
    community_louvain = None

# Interactive visualization
try:
    from pyvis.network import Network
except ImportError:
    Network = None


class ChemicalSpaceNetwork:
    """
    Constructs and analyzes a Chemical Space Network (CSN)
    with clustering + interactive visualization.
    """

    def __init__(self, distance_matrix: np.ndarray, molecule_labels: Optional[List[str]] = None):
        self.distance_matrix = np.asarray(distance_matrix)

        if self.distance_matrix.ndim != 2 or self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be a square 2D array.")

        # 🔴 Safety: check NaNs early
        if np.isnan(self.distance_matrix).any():
            raise ValueError("distance_matrix contains NaNs.")

        self.n_nodes = self.distance_matrix.shape[0]
        self.graph = nx.Graph()

        if molecule_labels is not None:
            if len(molecule_labels) != self.n_nodes:
                raise ValueError("Length of molecule_labels must match distance matrix.")
            self.labels = molecule_labels
        else:
            self.labels = [f"Mol_{i}" for i in range(self.n_nodes)]

        self.partition = None  # cluster assignment

    def _connected_component_partition(self) -> Dict[int, int]:
        """
        Build a deterministic fallback partition from graph connectivity.
        If the graph has no edges, each node becomes its own cluster.
        """
        partition: Dict[int, int] = {}

        for cluster_id, component in enumerate(nx.connected_components(self.graph)):
            for node in component:
                partition[node] = cluster_id

        return partition

    def _convert_to_similarity(self, distance: float, gamma: float = 1.0) -> float:
        return float(np.exp(-gamma * (distance ** 2)))

    # =========================
    # GRAPH BUILDING
    # =========================

    def build_knn_graph(
        self,
        k: int,
        mutual: bool = True,
        use_similarity_weights: bool = True,
        gamma: float = 2.0
    ) -> nx.Graph:

        logger.info(f"Building k-NN graph (k={k}, mutual={mutual})")

        self.graph = nx.Graph()

        for i in range(self.n_nodes):
            self.graph.add_node(i, label=self.labels[i])

        np.fill_diagonal(self.distance_matrix, np.inf)

        adj_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        for i in range(self.n_nodes):
            k_nearest = np.argsort(self.distance_matrix[i])[:k]
            adj_matrix[i, k_nearest] = True

        np.fill_diagonal(self.distance_matrix, 0.0)

        if mutual:
            adj_matrix = adj_matrix & adj_matrix.T
        else:
            adj_matrix = adj_matrix | adj_matrix.T

        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if adj_matrix[i, j]:
                    dist = self.distance_matrix[i, j]

                    if np.isnan(dist):
                        continue

                    weight = (
                        self._convert_to_similarity(dist, gamma)
                        if use_similarity_weights else dist
                    )

                    edges.append((i, j, weight))

        self.graph.add_weighted_edges_from(edges)

        logger.success(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    # =========================
    # CLUSTERING
    # =========================

    def detect_clusters(self):
        if self.graph.number_of_edges() == 0:
            logger.warning("Graph has no edges → using connected components as fallback clusters.")
            self.partition = self._connected_component_partition()
            return self.partition

        # 🔴 Check total weight
        total_weight = sum(data.get("weight", 1.0) for _, _, data in self.graph.edges(data=True))

        if total_weight == 0:
            logger.warning(
                "All edge weights are zero. This usually means exp(-gamma * distance^2) "
                "underflowed because distances are large for the chosen gamma. "
                "Using connected components as fallback clusters."
            )
            self.partition = self._connected_component_partition()
            return self.partition

        if community_louvain is None:
            logger.warning("python-louvain is not installed → using connected components as fallback clusters.")
            self.partition = self._connected_component_partition()
            return self.partition

        logger.info("Running Louvain clustering...")

        self.partition = community_louvain.best_partition(
            self.graph,
            weight='weight'
        )

        logger.success(f"Detected {len(set(self.partition.values()))} clusters")
        return self.partition
    # =========================
    # INTERACTIVE VISUALIZATION
    # =========================

    # =========================
    # INTERACTIVE VISUALIZATION
    # =========================

    def plot_interactive(self, filename="csn.html"):
        """
        Interactive visualization with:
        - cluster coloring
        - hover labels
        """

        if Network is None:
            raise ImportError(
                "pyvis is required for interactive plotting. Install it with `pip install pyvis`."
            )

        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty.")
            return

        if self.partition is None:
            logger.info("No clusters found yet → running detection")
            self.detect_clusters()

        if self.partition is None:
            logger.warning("Cluster detection did not produce a partition; using a single fallback cluster.")
            self.partition = {node: 0 for node in self.graph.nodes()}

        # Note: notebook=False since we are running in the terminal
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#111111", 
            font_color="white",
            notebook=False 
        )

        net.barnes_hut()

        # Generate colors
        import random
        random.seed(42)

        cluster_ids = list(set(self.partition.values()))
        colors = {
            cid: "#{:06x}".format(random.randint(0, 0xFFFFFF))
            for cid in cluster_ids
        }

        # Add nodes
        for node, data in self.graph.nodes(data=True):
            label = data.get("label", str(node))
            cluster_id = self.partition[node]

            net.add_node(
                node,
                label=label,
                title=f"{label}<br>Cluster: {cluster_id}",
                color=colors[cluster_id]
            )

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(u, v, value=data.get("weight", 1.0))

        # Physics tuning for better clusters
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -10000,
              "springLength": 120,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          }
        }
        """)

        # 1. Generate the HTML file silently
        net.write_html(filename)
        
        # 2. Get the absolute path to the file so the browser can definitely find it
        file_path = f"file://{os.path.abspath(filename)}"
        
        # 3. Force the OS to open it
        logger.info(f"Opening graph in default web browser: {file_path}")
        webbrowser.open(file_path)

    # =========================
    # STATS
    # =========================

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        if self.graph.number_of_nodes() == 0:
            return {}

        components = list(nx.connected_components(self.graph))

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "components": len(components),
        }
