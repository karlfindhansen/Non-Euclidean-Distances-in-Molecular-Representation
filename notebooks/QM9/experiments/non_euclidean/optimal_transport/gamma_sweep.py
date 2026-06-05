"""
Experiment 2: The gamma-continuum sweep.

This script selects two structurally distinct QM9 isomers, sweeps the REMatch
regularization strength, extracts the corresponding Sinkhorn transport plans,
and plots the transition from sharp OT-like matching to global averaging.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import ot
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist

from src.datasets import QM9Dataset
from src.non_euclidean import REMatch, Wasserstein 

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAMMA_GRID = np.logspace(start=-4, stop=2, num=100)

@dataclass(frozen=True)
class IsomerPair:
    df: pl.DataFrame
    formula: str
    tanimoto_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80_000, help="QM9 rows to scan for the isomer pair.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed.")
    parser.add_argument("--min-formula-count", type=int, default=8, help="Minimum isomers per formula to consider.")
    parser.add_argument("--top-formula-scan", type=int, default=25, help="Number of formula groups to inspect.")
    parser.add_argument("--max-candidates-per-formula", type=int, default=80, help="Candidate cap per formula.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/qm9/gamma_sweep")
    return parser.parse_args()


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def choose_isomer_pair(
    df: pl.DataFrame,
    *,
    seed: int,
    min_formula_count: int,
    top_formula_scan: int,
    max_candidates_per_formula: int,
) -> IsomerPair:
    required = {"formula", "canonical_smiles", "scaffold_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QM9 dataframe is missing required columns: {sorted(missing)}")

    formula_summary = (
        df.group_by("formula")
        .agg(
            pl.len().alias("count"),
            pl.col("scaffold_smiles").n_unique().alias("n_scaffolds"),
        )
        .filter((pl.col("count") >= min_formula_count) & (pl.col("n_scaffolds") >= 2))
        .sort(["n_scaffolds", "count", "formula"], descending=[True, True, False])
        .head(top_formula_scan)
    )
    if formula_summary.is_empty():
        raise ValueError("No formula group with enough distinct scaffold isomers was found.")

    best_pair = None
    best_similarity = np.inf
    best_formula = None

    for row in formula_summary.iter_rows(named=True):
        formula = row["formula"]
        candidates = df.filter(pl.col("formula") == formula)
        if candidates.height > max_candidates_per_formula:
            candidates = candidates.sample(n=max_candidates_per_formula, seed=seed)

        fps = [fingerprint(smiles) for smiles in candidates["canonical_smiles"].to_list()]
        scaffolds = candidates["scaffold_smiles"].to_list()

        for i in range(candidates.height):
            if fps[i] is None:
                continue
            for j in range(i + 1, candidates.height):
                if fps[j] is None or scaffolds[i] == scaffolds[j]:
                    continue
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if similarity < best_similarity:
                    best_similarity = float(similarity)
                    best_pair = pl.concat([candidates.slice(i, 1), candidates.slice(j, 1)])
                    best_formula = formula

    if best_pair is None or best_formula is None:
        raise ValueError("Could not select a structurally distinct isomer pair.")

    return IsomerPair(df=best_pair, formula=best_formula, tanimoto_similarity=best_similarity)


def clean_descriptor_matrix(matrix: Iterable[Iterable[float]]) -> np.ndarray:
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def average_soap_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def sinkhorn_plan(X: np.ndarray, Y: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
    b = np.ones(Y.shape[0], dtype=np.float64) / Y.shape[0]
    cost = cdist(X, Y, metric="euclidean")
    plan = ot.sinkhorn(
        a,
        b,
        cost,
        reg=gamma,
        method="sinkhorn_log",
        numItermax=50_000,
        stopThr=1e-12,
    )
    return np.asarray(plan, dtype=np.float64), cost


def pct_entries_carrying_mass(plan: np.ndarray, mass_fraction: float = 0.95) -> float:
    flat = np.sort(plan.ravel())[::-1]
    cutoff_index = int(np.searchsorted(np.cumsum(flat), mass_fraction * flat.sum(), side="left") + 1)
    return 100.0 * cutoff_index / flat.size


def compute_gamma_sweep(df_pair: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    matrices = [clean_descriptor_matrix(matrix) for matrix in df_pair["soap_matrix"].to_list()]
    X, Y = matrices

    wasserstein_distance = float(Wasserstein.distance_matrix(df_pair, descriptor="soap", metric="euclidean")[0, 1])
    average_distance = average_soap_distance(X, Y)

    rows = []
    plans: dict[str, np.ndarray] = {}
    costs: dict[str, np.ndarray] = {}

    for gamma in GAMMA_GRID:
        rematch_distance = float(REMatch.distance_matrix(df_pair, descriptor="soap", alpha=float(gamma))[0, 1])
        plan, cost = sinkhorn_plan(X, Y, float(gamma))

        key = f"gamma_{gamma:g}"
        plans[key] = plan
        costs[key] = cost

        rows.append(
            {
                "gamma": float(gamma),
                "log10_gamma": float(np.log10(gamma)),
                "rematch_distance": rematch_distance,
                "wasserstein_w1_distance": wasserstein_distance,
                "average_soap_distance": average_distance,
                "abs_rematch_minus_w1": abs(rematch_distance - wasserstein_distance),
                "abs_rematch_minus_average_soap": abs(rematch_distance - average_distance),
                "pct_entries_for_95pct_mass": pct_entries_carrying_mass(plan),
                "transport_cost_from_plan": float(np.sum(plan * cost)),
                "plan_entropy": float(-np.sum(plan[plan > 0] * np.log(plan[plan > 0]))),
            }
        )

    return pl.DataFrame(rows), {**plans, **{f"cost_{k}": v for k, v in costs.items()}}



def plot_results(results: pl.DataFrame, output_path: Path) -> None:
    """Generates a publication-quality 1x2 subplot mapping the REMatch gamma-continuum

    sweep profiles and dumps a high-res chart to disk.
    """
    # 1. Establish strict manuscript/thesis typography and layout defaults
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.titlesize": 13,
        }
    )

    gamma = results["gamma"].to_numpy()

    # Create the figure with a modern, constrained layout engine
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), layout="constrained", dpi=300)

    # Cohesive, high-contrast editorial color palette
    colors = ["#344e41", "#a3b18a", "#bc6c25"]
    grid_color = "#e2e8f0"

    # =========================================================================
    # SUBPLOT 1: TRANSPORT PLAN FLATTENING
    # =========================================================================
    ax0 = axes[0]
    y0 = results["pct_entries_for_95pct_mass"].to_numpy()

    # Utilizing a professional white-faced thick border marker style
    ax0.plot(
        gamma,
        y0,
        color=colors[0],
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.0,
        zorder=3,
    )

    ax0.set_xscale("log")
    ax0.set_xlabel("REMatch Regularization Parameter (gamma)", labelpad=10)
    ax0.set_ylabel("Entries Needed for 95% Mass (%)", labelpad=10)
    ax0.set_title("Transport Plan Flattening", fontweight="bold", pad=12)

    # Style grid lines subtly so they do not compete with data lines
    ax0.grid(True, which="both", linestyle=":", color=grid_color, alpha=0.7, zorder=0)

    # Clean border spines
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    ax0.spines["left"].set_color("#475569")
    ax0.spines["bottom"].set_color("#475569")

    # =========================================================================
    # SUBPLOT 2: ENDPOINT DIVERGENCE
    # =========================================================================
    ax1 = axes[1]
    y1_w1 = results["abs_rematch_minus_w1"].to_numpy()
    y1_soap = results["abs_rematch_minus_average_soap"].to_numpy()

    ax1.plot(
        gamma,
        y1_w1,
        color=colors[1],
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.0,
        label="|REMatch - exact W1|",
        zorder=3,
    )
    ax1.plot(
        gamma,
        y1_soap,
        color=colors[2],
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.0,
        label="|REMatch - average SOAP|",
        zorder=3,
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("REMatch Regularization Parameter (gamma)", labelpad=10)
    ax1.set_ylabel("Absolute Distance Difference", labelpad=10)
    ax1.set_title("Endpoint Divergence Bounds", fontweight="bold", pad=12)

    ax1.grid(True, which="both", linestyle=":", color=grid_color, alpha=0.7, zorder=0)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#475569")
    ax1.spines["bottom"].set_color("#475569")

    # Place descriptive legend with a clean background patch
    ax1.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#f1f5f9",
        framealpha=0.9,
    )

    # Global structural layout configurations
    fig.suptitle(
        "REMatch Regularization Gamma Sweep",
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qm9 = QM9Dataset(root=str(REPO_ROOT / "data/QM9"), limit=args.limit)
    df = qm9.load()

    pair = choose_isomer_pair(
        df,
        seed=args.seed,
        min_formula_count=args.min_formula_count,
        top_formula_scan=args.top_formula_scan,
        max_candidates_per_formula=args.max_candidates_per_formula,
    )

    qm9.df = pair.df
    qm9.add_soap()
    df_pair = qm9.df

    results, matrices = compute_gamma_sweep(df_pair)

    figure_path = args.output_dir / "gamma_continuum_sweep.png"
    csv_path = args.output_dir / "gamma_continuum_sweep.csv"
    pair_path = args.output_dir / "selected_isomer_pair.csv"
    matrix_path = args.output_dir / "gamma_transport_plans.npz"

    plot_results(results, figure_path)
    results.write_csv(csv_path)
    df_pair.select(["mol_id", "formula", "canonical_smiles", "scaffold_smiles", "num_atoms"]).write_csv(pair_path)
    np.savez_compressed(matrix_path, **matrices)

    print("Selected isomer pair")
    print(df_pair.select(["mol_id", "formula", "canonical_smiles", "scaffold_smiles", "num_atoms"]))
    print(f"Formula: {pair.formula}")
    print(f"Morgan Tanimoto similarity: {pair.tanimoto_similarity:.4f}")
    print("\nGamma sweep")
    print(results)
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved pair metadata: {pair_path}")
    print(f"Saved transport plans: {matrix_path}")


if __name__ == "__main__":
    main()
