import numpy as np
from scipy.spatial.distance import pdist

def get_features_xyz(frames):
    """
    Converts a list of ASE Atoms objects into fixed-length 
    feature vectors based on sorted pairwise atomic distances.
    """
    feature_vectors = []
    
    # 1. Calculate pairwise distances for every frame
    raw_distances = []
    max_len = 0
    
    for frame in frames:
        dists = pdist(frame.get_positions())
        # Sort distances to be invariant to atom indexing (permutation invariant)
        dists.sort()
        
        raw_distances.append(dists)
        max_len = max(max_len, len(dists))
        
    # 2. Pad vectors with zeros so they are all the same length
    for dists in raw_distances:
        # Create a zero vector of max length
        vec = np.zeros(max_len)
        # Fill it with the sorted distances
        vec[:len(dists)] = dists
        feature_vectors.append(vec)
        
    return np.array(feature_vectors)

def get_raw_xyz_features(frames):
    """
    Flattens XYZ coordinates and pads them to a fixed length
    to handle molecules with different numbers of atoms.
    """
    # 1. Get flattened coordinates for all frames
    flat_coords_list = [f.get_positions().flatten() for f in frames]
    
    # 2. Find the maximum length (3 * max_num_atoms)
    max_len = max(len(c) for c in flat_coords_list)
    
    # 3. Pad smaller vectors with zeros
    padded_features = []
    for coords in flat_coords_list:
        # Create a zero vector of the maximum length
        vec = np.zeros(max_len)
        # Fill the beginning with the actual coordinates
        vec[:len(coords)] = coords
        padded_features.append(vec)
        
    return np.array(padded_features)