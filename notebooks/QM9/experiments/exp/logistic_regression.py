import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from loguru import logger
import pathlib

from src.datasets import QM9Dataset

def evaluate_and_track_errors(
    df: pl.DataFrame, 
    target_col: str, 
    n_splits: int = 5, 
    min_class_samples: int = 15, 
    seeds: list[int] = [42, 123, 2026]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Evaluates descriptors and tracks explicit misclassifications.
    Returns:
        - metrics_df: Mean ± Std performance metrics.
        - errors_df: Top misclassified (Actual -> Predicted) pairs per descriptor.
    """
    logger.info(f"\nEvaluating target: '{target_col}' | {n_splits}-Fold CV")
    
    # 1. Base filter
    df_filtered = df.filter(
        (pl.col(target_col) != "") & 
        (pl.col(target_col).str.count_matches(",") == 0)
    ).drop_nulls(subset=[target_col])

    # 2. Minority class filter
    class_counts = df_filtered.group_by(target_col).agg(pl.len().alias("count"))
    valid_classes = class_counts.filter(pl.col("count") >= min_class_samples)[target_col]
    df_filtered = df_filtered.filter(pl.col(target_col).is_in(valid_classes))
    
    y = df_filtered[target_col].to_numpy()

    descriptor_cols = [
        "soap_embedding", "mace_embedding", "chemprop_embedding",
        "selfies_onehot", "morgan_fingerprint", "selfies_transformer"
    ]

    results = []
    error_logs = []

    for desc in descriptor_cols:
        if desc not in df_filtered.columns:
            continue
            
        logger.info(f"Processing {desc}...")
        X = np.array(df_filtered[desc].to_list())
        if X.shape[1] == 0:
            continue

        desc_metrics = {"acc": [], "f1": [], "prec": [], "rec": []}
        
        # We track errors ONLY for the first seed to prevent duplicate counting
        # across multiple random splits of the same dataset.
        primary_seed_errors = Counter() 

        for seed_idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)

                # Track metrics
                desc_metrics["acc"].append(accuracy_score(y_test, y_pred))
                desc_metrics["f1"].append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                desc_metrics["prec"].append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                desc_metrics["rec"].append(recall_score(y_test, y_pred, average="weighted", zero_division=0))

                if seed_idx == 0:
                    for actual, predicted in zip(y_test, y_pred):
                        if actual != predicted:
                            primary_seed_errors[(actual, predicted)] += 1

        # Format metrics
        def format_metric(metric_list):
            return f"{np.mean(metric_list):.3f} ± {np.std(metric_list):.3f}"

        desc_name = desc.replace("_embedding", "").replace("_fingerprint", "")
        
        results.append({
            "Descriptor": desc_name,
            "Accuracy": format_metric(desc_metrics["acc"]),
            "F1_Score": format_metric(desc_metrics["f1"]),
            "Precision": format_metric(desc_metrics["prec"]),
            "Recall": format_metric(desc_metrics["rec"]),
            "Raw_F1_Mean": np.mean(desc_metrics["f1"]) 
        })

        # Get top 5 confusions for this descriptor
        for (actual, predicted), count in primary_seed_errors.most_common(5):
            error_logs.append({
                "Descriptor": desc_name,
                "Actual_Class": actual,
                "Predicted_As": predicted,
                "Error_Count": count
            })

    metrics_df = pl.DataFrame(results).sort("Raw_F1_Mean", descending=True).drop("Raw_F1_Mean")
    errors_df = pl.DataFrame(error_logs)

    return metrics_df, errors_df


if __name__ == "__main__":

    save_dir = pathlib.Path("notebooks/QM9/experiments/exp/logs")
    save_dir.mkdir(parents=True, exist_ok=True)

    qm9 = QM9Dataset(limit=10_000, descriptors=["mace", "soap", "chemprop", "onehot", "morgan", "transformer"])
    df = qm9.load()
    
    table_fg_metrics, table_fg_errors = evaluate_and_track_errors(
        df, 
        target_col="functional_groups", 
        n_splits=5, 
        min_class_samples=15, 
        seeds=[42, 123, 2026, 777, 2024]
    )
    
    logger.info("\n--- Functional Groups Metrics ---")
    print(table_fg_metrics)
    
    logger.info("\n--- Top Misclassifications (Functional Groups) ---")
    print(table_fg_errors)

    metrics_path = save_dir / "functional_groups_metrics.csv"
    errors_path = save_dir / "functional_groups_errors.csv"
    
    table_fg_metrics.write_csv(metrics_path)
    table_fg_errors.write_csv(errors_path)
    
    logger.info(f"\n--- Metrics saved to {metrics_path} ---")
    print(table_fg_metrics)
    
    logger.info(f"\n--- Errors saved to {errors_path} ---")
    print(table_fg_errors)

    table_sc_metrics, table_sc_errors = evaluate_and_track_errors(
        df, 
        target_col="structure_class", 
        n_splits=5, 
        min_class_samples=15, 
        seeds=[42, 123, 2026, 777, 2024]
    )
    
    logger.info("\n--- Structure Class Metrics ---")
    print(table_sc_metrics)
    
    logger.info("\n--- Top Misclassifications (Structure Class) ---")
    print(table_sc_errors)

    metrics_path = save_dir / "structure_class_metrics.csv"
    errors_path = save_dir / "structure_class_errors.csv"

    table_sc_metrics.write_csv(metrics_path)
    table_sc_errors.write_csv(errors_path)