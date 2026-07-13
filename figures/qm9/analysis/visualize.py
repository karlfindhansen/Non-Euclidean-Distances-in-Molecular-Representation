"""3D side-by-side visualization of three C9H16 topologies: acyclic, 3-membered ring, 6-membered ring."""

import os

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

MOLECULES = [
    {"label": "Acyclic", "name": "non-1-yne", "smiles": "C#CCCCCCCC"},
    {"label": "3-membered ring", "name": "hex-5-en-1-ylcyclopropane", "smiles": "C1CC1CCCCC=C"},
    {"label": "6-membered ring", "name": "allylcyclohexane", "smiles": "C1CCCCC1CC=C"},
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "c9h16_ring_comparison.png")

ELEMENT_STYLE = {
    "H": {"color": "#FAFAFA", "edgecolor": "#999999", "size": 220, "alpha": 0.55},
    "C": {"color": "#4d4d4d", "edgecolor": "#1a1a1a", "size": 620, "alpha": 1.0},
}
BOND_LINEWIDTH = {1.0: 4.0, 1.5: 5.5, 2.0: 7.5, 3.0: 10.0}
RING_COLOR_BY_SIZE = {
    3: "#E76F51",
    6: "#2A9D8F",
}
EMBED_SEED = 40
VIEW_TILT_DEG = 16


def _embed_3d(smiles: str, seed: int = EMBED_SEED) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"3D embedding failed for SMILES: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _smallest_ring_atoms(mol: Chem.Mol):
    """Return the atom indices of the smallest ring (in traversal order), or None if acyclic."""
    atom_rings = mol.GetRingInfo().AtomRings()
    if not atom_rings:
        return None
    return list(min(atom_rings, key=len))


def _plane_normal(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return normal


def _auto_view(coords: np.ndarray, ring_idx) -> tuple:
    """Aim the camera roughly down the ring's normal (or the molecule's flattest axis) for a clear face-on view."""
    reference = coords[ring_idx] if ring_idx else coords
    normal = _plane_normal(reference)
    elev = np.degrees(np.arcsin(np.clip(normal[2], -1.0, 1.0))) + VIEW_TILT_DEG
    azim = np.degrees(np.arctan2(normal[1], normal[0]))
    return elev, azim


def _draw_ring_face(ax, coords: np.ndarray, ring_idx) -> None:
    ring_size = len(ring_idx)
    color = RING_COLOR_BY_SIZE.get(ring_size, "#457B9D")
    verts = [coords[i] for i in ring_idx]

    face = Poly3DCollection([verts], alpha=0.35, zorder=1)
    face.set_facecolor(color)
    face.set_edgecolor("none")
    ax.add_collection3d(face)


def _plot_molecule_3d(ax, mol: Chem.Mol) -> None:
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    coords = coords - coords.mean(axis=0)

    ring_idx = _smallest_ring_atoms(mol)
    ring_set = set(ring_idx) if ring_idx else set()
    ring_bond_color = RING_COLOR_BY_SIZE.get(len(ring_idx), "#457B9D") if ring_idx else None

    if ring_idx:
        _draw_ring_face(ax, coords, ring_idx)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        is_ring_bond = i in ring_set and j in ring_set
        linewidth = BOND_LINEWIDTH.get(bond.GetBondTypeAsDouble(), 4.0)
        color = ring_bond_color if is_ring_bond else "#6e6e6e"
        if is_ring_bond:
            linewidth += 1.5
        xs, ys, zs = zip(coords[i], coords[j])
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, solid_capstyle="round", zorder=2)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        style = ELEMENT_STYLE.get(atom.GetSymbol(), {"color": "#FF6699", "edgecolor": "#333333", "size": 400, "alpha": 1.0})
        face_color = ring_bond_color if idx in ring_set and atom.GetSymbol() == "C" else style["color"]
        x, y, z = coords[idx]
        ax.scatter(
            x, y, z,
            s=style["size"],
            c=face_color,
            edgecolors=style["edgecolor"],
            linewidths=1.4,
            alpha=style["alpha"],
            depthshade=True,
            zorder=3,
        )

    pad = 0.6
    mins = coords.min(axis=0) - pad
    maxs = coords.max(axis=0) + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)

    ax.set_axis_off()
    elev, azim = _auto_view(coords, ring_idx)
    ax.view_init(elev=elev, azim=azim)


def plot_c9h16_variants(molecules=MOLECULES, save_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, len(molecules), figsize=(16, 6.5), subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")

    for ax, entry in zip(axes, molecules):
        ax.set_facecolor("white")
        mol_3d = _embed_3d(entry["smiles"])
        formula = CalcMolFormula(Chem.RemoveHs(mol_3d))
        _plot_molecule_3d(ax, mol_3d)
        ax.set_title(f"{entry['label']}\n{entry['name']}\n{formula}", fontsize=13, pad=-4)

    fig.suptitle(
        "QM9-style C9H16 isomers in 3D: acyclic vs. 3-membered vs. 6-membered ring",
        fontsize=15, y=0.94,
    )
    plt.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.02, wspace=0.05)
    plt.savefig(save_path, dpi=250, facecolor="white")
    logger.info(f"Saved figure to {save_path}")


if __name__ == "__main__":
    plot_c9h16_variants()
