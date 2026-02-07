import os
import numpy as np
import pytest
from ase.io import read
from loguru import logger

# Path to your generated file
STRESS_TEST_FILE = "data/QM9/stress_test_perturbations.xyz"

def test_grassmann_perturbations_exist():
    """
    Verifies that the stress test file exists and contains frames.
    """
    assert os.path.exists(STRESS_TEST_FILE), f"File not found: {STRESS_TEST_FILE}"
    
    frames = read(STRESS_TEST_FILE, index=":")
    assert len(frames) > 0, "The XYZ file is empty!"
    logger.info(f"Loaded {len(frames)} frames for testing.")

def test_perturbations_are_distinct():
    """
    Verifies that the atoms have actually moved (coordinates are not identical).
    """
    frames = read(STRESS_TEST_FILE, index=":")
    
    # We grab the first molecule's first two perturbations
    # Assuming the first 20 frames belong to the first molecule
    mol_0_pert_0 = frames[0]
    mol_0_pert_1 = frames[1]
    
    # Check that they are the same molecule ID
    assert mol_0_pert_0.info['mol_id'] == mol_0_pert_1.info['mol_id']
    
    # Check positions
    pos_0 = mol_0_pert_0.get_positions()
    pos_1 = mol_0_pert_1.get_positions()
    
    # Calculate the difference matrix
    diff = np.abs(pos_0 - pos_1)
    
    # Assert that the maximum difference is NOT zero (i.e., they moved)
    max_diff = np.max(diff)
    logger.info(f"Max atomic displacement between frames: {max_diff:.5f} Å")
    
    assert max_diff > 1e-5, "Coordinates are identical! The molecules were NOT rattled."

def test_perturbation_magnitude():
    """
    Verifies that the rattling is within the expected range (approx 0.1 Å).
    """
    frames = read(STRESS_TEST_FILE, index=":")
    
    # Get all positions from the first frame (as a reference "anchor" is tricky without the original, 
    # so we compare frame 0 to frame 1 just to check the scale of movement)
    pos_0 = frames[0].get_positions()
    pos_1 = frames[1].get_positions()
    
    # Calculate Euclidean distance for each atom
    # dist = sqrt((x1-x2)^2 + ...)
    distances = np.linalg.norm(pos_0 - pos_1, axis=1)
    mean_dist = np.mean(distances)
    
    logger.info(f"Mean displacement observed: {mean_dist:.5f} Å")
    
    # We expect movement roughly around scale * sqrt(2) or similar magnitude, 
    # but definitely between 0.01 and 0.5 for a 0.1 rattle.
    assert 0.01 < mean_dist < 0.5, f"Displacement {mean_dist} is out of expected bounds (0.01 - 0.5 Å)"

if __name__ == "__main__":
    try:
        test_grassmann_perturbations_exist()
        test_perturbations_are_distinct()
        test_perturbation_magnitude()
        logger.success("All Grassmann tests passed!")
    except AssertionError as e:
        logger.error(f"Test Failed: {e}")