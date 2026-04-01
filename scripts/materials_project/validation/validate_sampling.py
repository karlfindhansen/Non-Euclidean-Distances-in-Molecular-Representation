from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

from src.datasets import MaterialsProject

def validate_distributions(df_pop, df_strat, df_head, df_random, column="band_gap"):
    """Plots the KDE distributions of the different sampling strategies."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Format the column name for titles and labels (e.g., "energy_above_hull" -> "Energy Above Hull")
    col_name_clean = column.replace("_", " ").title()

    sns.kdeplot(data=df_pop, x=column, label="Original Population", color="gray", fill=True, bw_adjust=0.5, alpha=0.3)
    sns.kdeplot(data=df_strat, x=column, label="Stratified Sample", color="blue", linewidth=2)
    sns.kdeplot(data=df_random, x=column, label="Random Sample", color="green", linestyle=":", linewidth=2)
    sns.kdeplot(data=df_head, x=column, label="Head Sample (Sequential)", color="red", linestyle="--", linewidth=2)

    plt.title(f"Sampling Distribution Comparison: {col_name_clean}", fontsize=14, pad=15)
    plt.xlabel(col_name_clean, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=True, shadow=True)

    out_path = Path("figures/materials/sampling")
    out_path.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path / f"{column}_sampling_validation.png", dpi=300)
    plt.show()
    plt.close()

def evaluate_ks_tests(df_pop, df_strat, df_head, df_random, column="band_gap"):
    """Calculates and prints the Kolmogorov-Smirnov statistics for the samples."""
    
    pop_data = df_pop[column].drop_nulls()
    strat_data = df_strat[column].drop_nulls()
    random_data = df_random[column].drop_nulls()
    head_data = df_head[column].drop_nulls()

    stat_strat, p_strat = ks_2samp(pop_data, strat_data)
    stat_random, p_random = ks_2samp(pop_data, random_data)
    stat_head, p_head = ks_2samp(pop_data, head_data)

    col_name_clean = column.replace("_", " ").title()
    print(f"\n--- KS Test Results: {col_name_clean} ---")
    print(f"Stratified vs Pop : Stat = {stat_strat:.4f} | p-value = {p_strat:.4e}")
    print(f"Random vs Pop     : Stat = {stat_random:.4f} | p-value = {p_random:.4e}")
    print(f"Head vs Pop       : Stat = {stat_head:.4f} | p-value = {p_head:.4e}")

def load_dfs():
    mp_full = MaterialsProject()
    df_full = mp_full.load(limit=None)

    mp_stratified = MaterialsProject(sampling_strategy="stratified", stratify_on=["band_gap", "energy_above_hull"])
    df_stratified = mp_stratified.load(limit=1000)

    mp_random = MaterialsProject(sampling_strategy="random")
    df_random = mp_random.load(limit=1000)

    mp_head = MaterialsProject(sampling_strategy="head")
    df_head = mp_head.load(limit=1000)

    return df_full, df_stratified, df_head, df_random

if __name__ == "__main__":
    df_full, df_stratified, df_head, df_random = load_dfs()
    
    features_to_evaluate = ["band_gap", "energy_above_hull"]
    
    for feature in features_to_evaluate:
        evaluate_ks_tests(df_full, df_stratified, df_head, df_random, column=feature)
        validate_distributions(df_full, df_stratified, df_head, df_random, column=feature)