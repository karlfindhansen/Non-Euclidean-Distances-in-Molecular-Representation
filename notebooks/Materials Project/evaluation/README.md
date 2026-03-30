# Evaluation of clusters using Hieracial Clustering

1. Determining SOAP reduction method
    - PCA: plots are not good. even though a lot of variance is perserved.
    - T-SNE: non-linear reduction technique. keeps density.
    - UMAP: plots looks better

2. Evaluating clusters based on meaningful coherence
    - Silhouette: Measures how closely a material matches its own cluster compared to the next most similar cluster, with higher scores indicating well-separated groups.
    - Calanski-Harabazs: Evaluates overall cluster validity using the ratio of between-cluster variance to within-cluster variance, rewarding dense, well-spaced clusters with higher scores.
        - This sometimes keeps increasing - indicating that 
    - Davies-Bouldin: Calculates the average similarity between each cluster and its closest neighboring cluster, meaning a lower score represents a tighter, better-separated clustering result.

3. Visualizing
    - Using chemiscope.

4. Determining best invariant features. Doing this to avoid curse of dimensionality.
    - z: atomic number. The identity of the atom.
    - en: electronegativity. describes charge transfer, polarizability, and bonding types.
    - coord: coordination number. This defines the local geometric environment (e.g., tetrahedral vs. octahedral). 
    - avg_neighbor_dist: the local structural environment of the material.
    - vol_per_atom: captures the global material packing.

## Results Euclidean
1. Reduced SOAP vector using t-SNE. 
    - Using complete clustering. When using average materials "chain", meaning that materials that are similar to each other are being swallowed by a large cluster one at a time. 
    - Best using UMAP for reduction and visualization (most coherent and dense clusters). Also no small not well defined islands. 
    - Clearly clusters well on density. In tightly packed materials the gaussian blobs will overlap. The resulting power spectrum (the SOAP vector) will have very high-intensity peaks at short radial distances. Because SOAP is measuring the spatial distribution of local density, materials with similar packing densities will naturally have very similar SOAP vectors. And because materials are periodic they will cluster. 
2. Invariant feature matrices.
    - Average clustering isolates outliers, while complete 5 distinct distinct clusters.
    - Outliers are materials with very high volume and low density. This is a metric in the feature vector and it may be quite high, and could dominate the vector. 

## Results Non-Euclidean
1. Riemannian
    - Is projected using UMAP and clustered using complete linking.
    - Isolates Symmorphic materials. Covariance matrix describes the internal correlation of features of atoms across the unit cell. Symmorphic space groups are those that do not contain glide planes or screw axes—their symmetry operations are strictly point-group operations (rotations, inversions) around a fixed point. 
    - In symmorphic crystals, the atomic environments tend to be highly uniform and follow strict geometric alignments. This creates very "clean," high-variance covariance matrices that look radically different from non-symmorphic (distorted or complex) structures. The Riemannian manifold is exceptionally good at sensing the "shape" of this internal symmetry.
    
2. Grassmanian:
    - Is projected using UMAP and clustered using complete linking.
    - Clusters by linear subspace spanned by atoms. Put materials with symmetrical space groups in same clusters. These often have electronegativites that are close to each other, and high coordination number. 
    - Clusters by differences between features. If some atoms are very heavy but has low electronegativity, they will tilt the subspace that will be isolated. 

3. Wasserstein:
    - Projection and linking are not important.
    - Seperates orthocombic Symmorphic materials very well. 
    - In feature space a symmorphic material doesn't look like a blurry cloud; it looks like a few highly concentrated, "heavy" spikes of mass.
    - EMD is exceptionally good at comparing these "sparse" distributions. Moving mass from one high-concentration spike to another is mathematically "expensive." EMD detects that these symmorphic materials have a "signature" distribution that doesn't exist in lower-symmetry or non-symmorphic crystals where atoms are more "smearing" across the feature space.
    - Comparing an orthorhombic crystal to a monoclinic or triclinic one, the "cost" to move the mass in the avg_neighbor_dist dimension is high because the orthorhombic structure has very specific, fixed distance intervals. EMD feels the "friction" of trying to map a rectangular coordinate system onto a tilted one.

## Notes:
1. Using SOAP as inputs to Riemannian and Grassmannian:
    - SOAP is a 1d vector in flat euclidean space. 
    - Riemann needs a $D \times N$ matrix to calculate the variance between those points to construct a Symmetric Positive Definite (SPD) covariance matrix.
    - Grassmann manifold is the space of all linear subspaces. To define a subspace, one needs a set of basis vectors in $D \times N$.

2. Using complete vs. average linking for Riemannian and Grassmanian.
    - Average: Measures the distance between every single material in Cluster A to every single material in Cluster B, and takes the average of all those distances. It is resilient to noise and forms clusters that are "generally" close to each other. However, if data has a continuous chain of points, it will merge them up one by one because the "average" distance remains low enough.
        - Gives one very large cluster, and multiple small ones (1-5 materials).  
    - Complete: It looks at the two materials that are the farthest apart from each other (one in Cluster A, one in Cluster B). That single, maximum distance becomes the official distance between the two clusters.
        - Gives more well defined clustering. More large and coherent clusters. 

3. Wasserstein
    - Must not average the atomic descriptors into a single vector. This gives just one point in space. Optimal transport is designed to compare distributions or clouds of points. It calculates the optimal "cost" of moving the cloud of atoms in Material A to match the cloud of atoms in Material B in the 5-dimensional feature space.