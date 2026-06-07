from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp

from torch_geometric.datasets import QM9
from src.datasets import QM9Dataset

from loguru import logger
import sys

logger.remove()  # removes default handler
logger.add(sys.stderr, level="WARNING")  # only warnings and above

def load_qm9_samples(limit=1000, stratify_by=None):
    df_head = QM9Dataset(sampling_strategy="head", limit=limit).load()
    df_random = QM9Dataset(sampling_strategy="random", limit=limit).load()

    df_strat = QM9Dataset(
        sampling_strategy="stratified",
        stratify_by=stratify_by,
        stratify_bins=203,
        limit=limit
    ).load()

    df_full = QM9Dataset().load()

    return df_head, df_random, df_strat, df_full


def optimize_joint_stratification(
    limit=5000,
    targets=None,
    weights=None
):
    """
    Optimizes stratification using multiple KS aggregation strategies:
    - mean KS
    - weighted KS
    - L2 norm KS

    Returns best params per metric.
    """

    if targets is None:
        raise ValueError("targets must be provided as a list")

    if isinstance(targets, str):
        targets = [targets]

    if weights is None:
        weights = {t: 1.0 for t in targets}
    else:
        # ensure all targets have weights
        weights = {t: weights.get(t, 1.0) for t in targets}

    print("Loading full QM9 baseline...")
    df_full = QM9Dataset().load()
    pop_data = {t: df_full[t].to_numpy() for t in targets}

    bin_counts = np.linspace(2, 300, num=100)
    strategies = ["quantile", "width"]

    best = {
        "mean": {"score": float("inf"), "params": None},
        "weighted": {"score": float("inf"), "params": None},
        "l2": {"score": float("inf"), "params": None},
    }

    print(f"\nOptimizing stratification for: {targets}")
    print("-" * 80)

    for strategy in tqdm(strategies, desc="Strategies"):
        for bins in bin_counts:

            dataset = QM9Dataset(
                sampling_strategy="stratified",
                stratify_by=targets,
                limit=limit,
                stratify_bins=int(np.floor(bins)),
                binning_strategy=strategy
            )
            df_strat = dataset.load()

            ks_scores = {
                t: ks_2samp(pop_data[t], df_strat[t].to_numpy())[0]
                for t in targets
            }

            ks_vec = np.array(list(ks_scores.values()))

            # -----------------------
            # 1. Mean KS
            # -----------------------
            mean_ks = ks_vec.mean()

            # -----------------------
            # 2. Weighted KS
            # -----------------------
            w = np.array([weights[t] for t in targets])
            weighted_ks = np.sum(w * ks_vec) / np.sum(w)

            # -----------------------
            # 3. L2 norm KS
            # -----------------------
            l2_ks = np.sqrt(np.sum(ks_vec ** 2))

            params = {
                "strategy": strategy,
                "bins": bins,
                "ks_scores": ks_scores
            }

            # update best mean
            if mean_ks < best["mean"]["score"]:
                best["mean"] = {"score": mean_ks, "params": params}

            # update best weighted
            if weighted_ks < best["weighted"]["score"]:
                best["weighted"] = {"score": weighted_ks, "params": params}

            # update best L2
            if l2_ks < best["l2"]["score"]:
                best["l2"] = {"score": l2_ks, "params": params}

    print("\n" + "-" * 80)
    print("Optimization complete.\n")

    for k in best:
        p = best[k]["params"]
        print(f"\n[{k.upper()} BEST]")
        print(f"Strategy: {p['strategy']}, Bins: {p['bins']}")
        print(f"Score: {best[k]['score']:.6f}")
        for t in targets:
            print(f"  KS {t}: {p['ks_scores'][t]:.5f}")

    return best

def evaluate_ks_tests(pop_data, strat_data, head_data, rand_data, target_name="gap"):
    """Calculates and prints the Kolmogorov-Smirnov statistics."""
    stat_strat, p_strat = ks_2samp(pop_data, strat_data)
    stat_rand, p_rand = ks_2samp(pop_data, rand_data)
    stat_head, p_head = ks_2samp(pop_data, head_data)

    print(f"\n--- KS Test Results: {target_name.upper()} ---")
    print(f"Stratified vs Pop : Stat = {stat_strat:.4f} | p-value = {p_strat:.4e}")
    print(f"Random vs Pop     : Stat = {stat_rand:.4f} | p-value = {p_rand:.4e}")
    print(f"Head vs Pop       : Stat = {stat_head:.4f} | p-value = {p_head:.4e}")

def plot_distributions(pop_data, strat_data, head_data, rand_data, target_name="gap"):
    """Generates a clean distribution plot using Histplot for discrete data."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    is_discrete = (target_name == "num_atoms")

    if is_discrete:
        # Define common settings for discrete bar-style histograms
        common_params = {
            "discrete": True, 
            "element": "step", 
            "stat": "density", 
            "linewidth": 2
        }
        
        # Plot population (filled)
        sns.histplot(pop_data, label="Original QM9 Population", color="gray", fill=True, alpha=0.3, **common_params)
        
        # Plot samples (unfilled)
        sns.histplot(strat_data, label="Stratified Sample", color="blue", fill=False, **common_params)
        sns.histplot(rand_data, label="Random Sample", color="green", fill=False, linestyle="-.", **common_params)
        sns.histplot(head_data, label="Head Sample (Sequential)", color="red", fill=False, linestyle="--", **common_params)
    else:
        # Use KDE for continuous data (e.g., gap)
        sns.kdeplot(pop_data, label="Original QM9 Population", color="gray", fill=True, bw_adjust=0.5, alpha=0.3)
        sns.kdeplot(strat_data, label="Stratified Sample", color="blue", linewidth=2)
        sns.kdeplot(rand_data, label="Random Sample", color="green", linestyle="-.", linewidth=2)
        sns.kdeplot(head_data, label="Head Sample (Sequential)", color="red", linestyle="--", linewidth=2)

    plt.title(f"QM9 Sampling Distribution Comparison: {target_name}", fontsize=14, pad=15)
    unit = "(eV/Hartree)" if target_name == "gap" else "(Count)"
    plt.xlabel(f"{target_name} {unit}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    targets_to_evaluate = ["gap"]

    limit = 1_000
    df_head, df_random, df_strat, df_full = load_qm9_samples(
        limit=limit,
        stratify_by=targets_to_evaluate
    )
    # 203 and 2

    # optimize_joint_stratification(
    #    limit=limit,
    #    targets=targets_to_evaluate
    # )

    for target in targets_to_evaluate:

        full_array = df_full[target].to_numpy()
        head_array = df_head[target].to_numpy()
        rand_array = df_random[target].to_numpy()
        strat_array = df_strat[target].to_numpy()

        evaluate_ks_tests(
            full_array,
            strat_array,
            head_array,
            rand_array,
            target_name=target
        )

        plot_distributions(
            full_array,
            strat_array,
            head_array,
            rand_array,
            target_name=target
        )