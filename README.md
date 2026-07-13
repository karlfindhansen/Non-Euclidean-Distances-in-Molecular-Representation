# Geometric Distance Frameworks in Cheminformatics: Evaluating Flat vs. Curved Geometries for Molecular Comparison

This repository evaluates advanced geometric frameworks for comparing molecular structures by replacing standard Euclidean distance metrics with sophisticated manifold-based and topological approaches. Rather than reducing atomic environments to single vectors, these methods preserve the statistical and structural properties of local atomic neighborhoods.

## Table of Contents
- [Advanced Geometries](#advanced-geometries)
- [Supported Descriptors](#supported-descriptors)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Advanced Geometries

This repository implements four distinct geometric frameworks for molecular comparison:

### 1. **Optimal Transport (Wasserstein Distance)**
- **Mathematical Space:** The space of all probability distributions
- **What it captures:** The full distribution of atomic environments
- **Key feature:** Computes exact Wasserstein distances via optimal transport
- **Use case:** Best for comparing crystalline structures with distinct density patterns
- **Implementation:** Regularized Entropy Matching (REM) for computational efficiency
- **Location:** See notebooks under `notebooks/Materials Project/` and `notebooks/QM9/`

### 2. **Symmetric Positive Definite (SPD) Manifold (Riemannian Geometry)**
- **Mathematical Space:** Manifold of positive definite matrices equipped with Riemannian metric
- **What it captures:** Multi-body correlations through empirical covariance matrices
- **Key feature:** Operates on curved Riemannian geometry using geodesic distances
- **Use case:** Identifies materials with uniform atomic environments (symmorphic crystals)
- **Mathematical foundation:** Covariance between atomic descriptors across the local environment
- **Best for:** Detecting structural symmetry and uniform atomic arrangements
- **Location:** Advanced geometry implementations in geometry modules

### 3. **Grassmann Manifold (Linear Subspace Geometry)**
- **Mathematical Space:** Space of all linear subspaces $\mathbb{G}(k, d)$ with dimension $k$ in $\mathbb{R}^d$
- **What it captures:** Dominant principal modes of structural variance
- **Key feature:** Compresses high-dimensional descriptors by isolating principal subspaces
- **Use case:** Clusters materials by characteristic linear features
- **Best for:** Identifying materials with similar feature patterns regardless of scale
- **Location:** Advanced geometry implementations in geometry modules

### 4. **Persistent Homology (Topological Space)**
- **Mathematical Space:** Space of persistence diagrams
- **What it captures:** Topological features (connected components, loops, voids) across structural scales
- **Key feature:** Uses Vietoris-Rips complexes and persistence barcodes
- **Distance metric:** Bottleneck distance or Wasserstein distance on persistence diagrams
- **Use case:** Multi-scale structural comparison
- **Location:** Advanced geometry implementations in geometry modules

## Supported Descriptors

The advanced geometries operate on continuous, 3D molecular representations that natively output per-atom feature matrices $X \in \mathbb{R}^{N \times D}$:

- **SOAP** (Smooth Overlap of Atomic Positions) - Local atomic environment descriptor
- **ACSF** (Atom-Centered Symmetry Functions) - Symmetry-adapted features for atomic environments
- **MACE** (Message Passing Atomic Cluster Expansion) - Graph neural network embeddings

## Datasets

### QM9 Dataset

The QM9 dataset contains quantum mechanical properties of ~134k organic molecules.

#### Getting Started with QM9
1. Navigate to `notebooks/QM9/`
2. The dataset is automatically downloaded and processed by DScale (OpenQDC wrapper) on first use
3. Molecular structures are stored as ASE atoms objects
4. Features calculated from quantum mechanical computations

#### QM9 Notebooks Include:
- Data loading and preprocessing
- SOAP/ACSF descriptor generation
- Geometric distance computation
- Clustering and visualization

#### Key QM9 Properties Tracked:
- Molecular energy levels
- Dipole moment
- HOMO-LUMO gap
- Heat capacity at 298K
- Atomic charge distributions

### Materials Project Dataset

The Materials Project contains ~140k experimentally-validated crystal structures with computed properties.

#### Getting Started with Materials Project
1. Navigate to `notebooks/Materials Project/`
2. **API Setup Required:**
   ```python
   from mp_api.client import MPRestClient
   client = MPRestClient(api_key="YOUR_MP_API_KEY")
   ```
   Get your free API key at: https://materialsproject.org/

3. Data is queried directly from Materials Project REST API
4. Crystal structures provided in CIF format, converted to ASE atoms objects
5. Computed properties available: energy, band gap, formation energy, etc.

#### Materials Project Notebooks Include:
- Crystal structure retrieval and parsing
- Unit cell and atomic environment analysis
- SOAP descriptor generation for materials
- Multi-scale clustering (by space group, density, composition)
- Evaluation using hierarchical clustering with multiple linkage methods

#### Key Materials Project Features Tracked:
- Space group symmetry
- Atomic number (z)
- Electronegativity (en)
- Coordination number (coord)
- Average neighbor distance (avg_neighbor_dist)
- Volume per atom (vol_per_atom)

#### Evaluation Metrics (Materials Project)
- **Silhouette Score:** Measures how well each structure matches its cluster vs. neighboring clusters
- **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index:** Average similarity between each cluster and its nearest neighbor (lower is better)
- **Visualization:** chemiscope for interactive 3D structure exploration

## Project Structure

```
.
├── README.md                          # This file
├── pyproject.toml                     # Project dependencies and configuration
├── notebooks/                         # Main analysis notebooks
│   ├── QM9/                           # QM9 organic molecules analysis
│   │   ├── data_processing/           # QM9 preprocessing pipelines
│   │   ├── descriptors/               # Descriptor generation (SOAP, ACSF)
│   │   ├── clustering/                # Clustering analysis
│   │   └── evaluation/                # Metrics and visualization
│   │
│   └── Materials Project/             # Materials Project crystals analysis
│       ├── data_processing/           # Crystal structure loading and conversion
│       ├── descriptors/               # Descriptor generation for materials
│       ├── clustering/                # Hierarchical clustering workflows
│       └── evaluation/                # Cluster evaluation and visualization
│           └── README.md              # Detailed evaluation methodology
│
├── src/                               # Source code modules
│   ├── descriptors/                   # Descriptor calculation utilities
│   ├── geometries/                    # Advanced geometry implementations
│   │   ├── optimal_transport.py       # Wasserstein distance computation
│   │   ├── riemannian.py              # SPD manifold operations
│   │   ├── grassmann.py               # Grassmann manifold operations
│   │   └── persistent_homology.py     # Topological feature extraction
│   ├── clustering/                    # Clustering algorithms
│   └── metrics/                       # Evaluation metrics
│
├── config/                            # Configuration files for different experiments
├── data/                              # Data directory (gitignored for large files)
├── results/                           # Results and outputs (gitignored)
├── figures/                           # Generated figures (gitignored)
└── tests/                             # Unit tests
```

## Installation

### Requirements
- Python ≥ 3.12
- PyTorch ≥ 2.6.0
- Optional: Materials Project API key

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/karlfindhansen/Non-Euclidean-Distances-in-Molecular-Representation.git
   cd Non-Euclidean-Distances-in-Molecular-Representation
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   uv sync
   ```
   
   Or with pip:
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python -c "import ase; import dscribe; import geomstats; print('Installation successful!')"
   ```

## Usage

### Working with QM9

```python
import sys
sys.path.insert(0, './notebooks/QM9')

# Load QM9 dataset
from data_processing import QM9Dataset
dataset = QM9Dataset()
structures = dataset.get_structures()

# Generate SOAP descriptors
from descriptors import SOAPDescriptor
soap = SOAPDescriptor()
soap_features = soap.compute_descriptors(structures)

# Compute Wasserstein distances
from geometries import WassersteinDistance
wasserstein = WassersteinDistance()
distances = wasserstein.compute_distance_matrix(soap_features)

# Perform clustering
from clustering import HierarchicalClustering
clustering = HierarchicalClustering(method='complete')
labels = clustering.fit_predict(distances)
```

### Working with Materials Project

```python
import sys
sys.path.insert(0, './notebooks/Materials Project')

from mp_api.client import MPRestClient

# Initialize API client
client = MPRestClient(api_key="YOUR_API_KEY")

# Query crystal structures
materials = client.materials.search(is_stable=True, limit=1000)

# Load and convert to ASE atoms objects
from data_processing import MaterialsProjectConverter
converter = MaterialsProjectConverter()
ase_structures = converter.to_ase_atoms(materials)

# Generate SOAP descriptors
from descriptors import SOAPDescriptor
soap = SOAPDescriptor()
soap_features = soap.compute_descriptors(ase_structures)

# Evaluate with Riemannian geometry
from geometries import RiemannianManifold
riemann = RiemannianManifold()
covariance_matrices = riemann.compute_covariance_matrices(soap_features)
distances = riemann.compute_geodesic_distances(covariance_matrices)

# Evaluate clustering quality
from evaluation import ClusteringMetrics
metrics = ClusteringMetrics(distances, labels)
print(f"Silhouette Score: {metrics.silhouette()}")
print(f"Davies-Bouldin Index: {metrics.davies_bouldin()}")
print(f"Calinski-Harabasz Index: {metrics.calinski_harabasz()}")
```

### Key Notebooks to Explore

#### QM9 Analysis
- `notebooks/QM9/descriptors/` - Understand SOAP/ACSF generation for organic molecules
- `notebooks/QM9/clustering/` - See how different geometries cluster similar molecules

#### Materials Project Analysis
- `notebooks/Materials Project/data_processing/` - Learn to query and load crystal structures
- `notebooks/Materials Project/evaluation/README.md` - Detailed explanation of clustering evaluation methodology
- `notebooks/Materials Project/evaluation/` - Explore how different geometries reveal material symmetry and structural properties

## Important Notes

### About SOAP and Manifold Methods
- **SOAP input:** 1D vector in flat Euclidean space
- **Riemannian/Grassmann input:** Need $D \times N$ matrix to compute variance and define subspaces
- **Wasserstein input:** Point cloud; must preserve atomic descriptor diversity (do NOT average into single vector)
- **Persistent Homology input:** Point cloud of atomic features with persistence tracking

### Linkage Strategies for Hierarchical Clustering

- **Average Linkage:** Average distance between all pairs across clusters
  - Pros: Noise resistant, balanced clustering
  - Cons: Can result in one large cluster with outliers
  
- **Complete Linkage:** Maximum distance between any two points across clusters
  - Pros: Produces well-separated, coherent clusters
  - Cons: Can fragment into many small clusters
  
- **Recommended:** Complete linkage for manifold geometries (Riemannian, Grassmann, Wasserstein)

## References & Key Dependencies

- **DScale** - Molecular descriptor computation
- **ASE (Atomic Simulation Environment)** - Atomic structure handling
- **DScribe** - SOAP and ACSF descriptors
- **Geomstats** - Riemannian geometry computations
- **GUDHI** - Persistent homology computation
- **POT (Python Optimal Transport)** - Wasserstein distance calculations
- **PyRiemann** - SPD manifold operations
- **Chemiscope** - Interactive 3D structure visualization

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code structure.

```bash
pytest tests/
```

## License

[Add your license here]

## Contact

For questions or discussions, please open an issue on this repository.
