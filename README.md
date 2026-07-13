# Geometric Distance Frameworks in Cheminformatics: Evaluating Flat vs. Curved Geometries for Molecular Comparison

## Advanced Geometries Implemented

This repository replaces standard mean-pooling operations with frameworks that preserve the higher-order statistical structure of local atomic environments:

* **Optimal Transport (The Full Distribution):** Preserves individual atomic environments without structural loss. Computes exact Wasserstein distances and implements a Regularized Entropy Match (REMatch) kernel via the Sinkhorn-Knopp algorithm to evaluate the cost of matching local neighborhoods.
* **The Symmetric Positive Definite (SPD) Manifold (The Second Moment):** Captures multi-body correlations through the empirical covariance of the environment cloud. Operates on the curved Riemannian manifold $\mathcal{S}_{++}^D$. Implements true Affine-Invariant Riemannian Metrics (AIRM) and Log-Euclidean metric projections to the tangent space $T_I\mathcal{S}_{++}^D$ for downstream kernel ridge regression.
* **The Grassmann Manifold (The Dominant Subspace):** Compresses high-dimensional descriptors by isolating the principal modes of structural variance. Describes molecules as points on $\mathbb{G}(k,D)$. Evaluates distances using exact geodesic paths (principal angles) and flat chordal symmetric projections.
* **Persistent Homology (Topological Space):** Tracks topological features (connected components, loops) across structural scales using Vietoris-Rips complexes, comparing persistence diagrams via Sliced-Wasserstein and Bottleneck distances.

## Supported Descriptors

The geometries are evaluated across continuous, 3D molecular representations that natively output a per-atom feature matrix $X \in \mathbb{R}^{N \times D}$:

* **SOAP** (Smooth Overlap of Atomic Positions)
* **ACSF** (Atom-Centered Symmetry Functions)
* **MACE** (Message Passing Atomic Cluster Expansion) embeddings