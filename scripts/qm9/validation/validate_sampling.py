from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp

from torch_geometric.datasets import QM9
from src.datasets import QM9Dataset

def load_qm9_samples(limit=1000):
    df_head = QM9Dataset(sampling_strategy="head", limit=limit).load()

    df_random = QM9Dataset(sampling_strategy="random", limit=limit).load()

    df_strat = QM9Dataset(sampling_strategy="stratified", stratify_by=['num_atoms', 'gap'], limit=limit).load()

    df_full = QM9Dataset().load()

    return df_head, df_random, df_strat, df_full

def optimize_joint_stratification(limit=5000, targets=["gap", "num_atoms"]):
    """
    Sweeps through binning strategies and counts to find the optimal 
    configuration that minimizes the worst-case KS statistic across all targets.
    """
    # Load the full population baseline once
    print("Loading full QM9 baseline...")
    df_full = QM9Dataset().load()
    pop_data = {t: df_full[t].to_numpy() for t in targets}

    # Define hyperparameter grid
    bin_counts = np.linspace(2, 100, num=100)
    strategies = ["quantile", "width"]
    
    best_combined_ks = float('inf')
    best_params = {}

    print(f"\nOptimizing joint stratification for: {targets}")
    print("-" * 65)
    print(f"{'Strategy':<12} | {'Bins':<5} | {'KS Gap':<8} | {'KS Atoms':<8} | {'Combined (Max)':<14}")
    print("-" * 65)

    for strategy in tqdm(strategies, desc="Binning Strategies"):
        for bins in bin_counts:
            # Instantiate dataset with current grid parameters
            dataset = QM9Dataset(
                sampling_strategy="stratified", 
                stratify_by=targets, 
                limit=limit,
                stratify_bins=int(np.floor(bins)),
                binning_strategy=strategy
            )
            df_strat = dataset.load()
            
            # Calculate KS Score for all targets
            ks_scores = {}
            for t in targets:
                strat_data = df_strat[t].to_numpy()
                stat, _ = ks_2samp(pop_data[t], strat_data)
                ks_scores[t] = stat
            
            # Objective: Minimize the worst-case KS statistic
            combined_ks = max(ks_scores.values()) 
            
            #print(f"{strategy:<12} | {bins:<5} | {ks_scores['gap']:.5f}  | {ks_scores['num_atoms']:.5f}  | {combined_ks:.5f}")

            # Track the best configuration
            if combined_ks < best_combined_ks:
                best_combined_ks = combined_ks
                best_params = {
                    'strategy': strategy, 
                    'bins': bins, 
                    'ks_scores': ks_scores
                }

    print("-" * 65)
    print("Joint Optimization Complete.")
    print(f"Best Configuration -> Strategy: {best_params['strategy']}, Bins: {best_params['bins']}")
    print(f"Resulting KS Gap: {best_params['ks_scores']['gap']:.5f} | Resulting KS Atoms: {best_params['ks_scores']['num_atoms']:.5f}")


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
    """Generates and saves a clean KDE plot comparing the distributions."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(pop_data, label="Original QM9 Population", color="gray", fill=True, bw_adjust=0.5, alpha=0.3)
    sns.kdeplot(strat_data, label="Stratified Sample", color="blue", linewidth=2)
    sns.kdeplot(rand_data, label="Random Sample", color="green", linestyle="-.", linewidth=2)
    sns.kdeplot(head_data, label="Head Sample (Sequential)", color="red", linestyle="--", linewidth=2)

    plt.title(f"QM9 Sampling Distribution Comparison: {target_name}", fontsize=14, pad=15)
    
    # Dynamic axis labeling
    unit = "(eV/Hartree)" if target_name == "gap" else "(Count)"
    plt.xlabel(f"{target_name} {unit}", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=True, shadow=True)

    out_path = Path("figures/qm9/sampling")
    out_path.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    #plt.savefig(out_path / f"{target_name}_sampling_validation.png", dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    #optimize_joint_stratification()
    df_head, df_random, df_strat, df_full = load_qm9_samples(limit=5000)
    
    targets_to_evaluate = ["gap", "num_atoms"]
    
    for target in targets_to_evaluate:
        full_array_target = df_full[target].to_numpy()
        head_array_target = df_head[target].to_numpy()
        rand_array_target = df_random[target].to_numpy()
        strat_array_target = df_strat[target].to_numpy()

        evaluate_ks_tests(full_array_target, strat_array_target, head_array_target, rand_array_target, target_name=target)
        plot_distributions(full_array_target, strat_array_target, head_array_target, rand_array_target, target_name=target)