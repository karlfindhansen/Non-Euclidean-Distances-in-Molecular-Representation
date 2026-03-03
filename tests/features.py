import numpy as np

from src.features import get_weighted_point_clouds, get_raw_xyz_features
from src.datasets import QM9Dataset

if __name__ == "__main__":

    mol_ids = ["qm9_1237", "qm9_1244", "qm9_1246", "qm9_1248", "qm9_1474", "qm9_1476", "qm9_1478", "qm9_1486", "qm9_1447", "qm9_1449"]
    qm9_loader = QM9Dataset(required_mol_ids=mol_ids)
    qm9_loader.load()

    frames = qm9_loader.run_stress_test(mol_ids=mol_ids, max_rattle=1.5)
    pcs = get_weighted_point_clouds(frames)
    raw_frames = get_raw_xyz_features(frames)
