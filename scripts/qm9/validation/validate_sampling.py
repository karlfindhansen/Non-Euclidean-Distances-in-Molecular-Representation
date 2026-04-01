from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# Assuming these are your local imports
from torch_geometric.datasets import QM9
from src.datasets import QM9Dataset

def load_qm9_samples(limit=1000):
    print("Loading sequential head sample...")
    df_head = QM9Dataset(sampling_strategy="head", limit=limit).load(force_process=True)

    print("Loading random sample...")
    df_random = QM9Dataset(sampling_strategy="random", limit=limit).load(force_process=True)

    print("Loading stratified sample...")
    df_strat = QM9Dataset(
        sampling_strategy="stratified", 
        stratify_by=['num_atoms', 'gap'], 
        limit=limit
    ).load(force_process=True)

    return df_head, df_random, df_strat

def extract_population_target(target_name="gap", data_root="data/QM9"):
    """Bypasses processing to quickly extract the full population target array."""
    print(f"\nExtracting full QM9 population targets for '{target_name}' validation...")
    raw_dataset = QM9(root=data_root)
    
    # Handle num_atoms dynamically since it isn't in the data.y matrix
    if target_name == "num_atoms":
        return np.array([data.z.size(0) for data in raw_dataset])
    
    target_idx = QM9Dataset.QM9_TARGETS.index(target_name)
    return np.array([data.y[0, target_idx].item() for data in raw_dataset])

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
    plt.savefig(out_path / f"{target_name}_sampling_validation.png", dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    df_head, df_random, df_strat = load_qm9_samples(limit=1000)
    
    targets_to_evaluate = ["gap", "num_atoms"]
    
    for target in targets_to_evaluate:
        pop_array = extract_population_target(target_name=target)

        # Clean nulls and convert to flat numpy arrays for Scipy/Seaborn
        head_array = df_head[target].drop_nulls().to_numpy()
        rand_array = df_random[target].drop_nulls().to_numpy()
        strat_array = df_strat[target].drop_nulls().to_numpy()

        evaluate_ks_tests(pop_array, strat_array, head_array, rand_array, target_name=target)
        plot_distributions(pop_array, strat_array, head_array, rand_array, target_name=target)