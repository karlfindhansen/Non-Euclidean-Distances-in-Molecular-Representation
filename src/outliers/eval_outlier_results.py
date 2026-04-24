import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluate_outlier_methods(df: pl.DataFrame, methods: list[str] = ["hdbscan", "lof", "knn"]) -> pl.DataFrame:
    """
    Evaluates outlier detection methods globally and per outlier category.
    Assumes `is_injected` is 1 for outliers and 0 for normal QM9 molecules.
    Assumes method labels use -1 to denote an outlier.
    """
    # 1. Prepare True Labels (Positive class = 1 = Outlier)
    y_true = df["is_injected"].fill_null(0).to_numpy()
    
    # 2. Get the unique categories of injected outliers
    # Filter out nulls/Nones which represent the normal QM9 data
    categories = df.filter(pl.col("is_injected") == 1)["outlier_category"].drop_nulls().unique().to_list()
    
    results = []
    
    for method in methods:
        label_col = f"{method}_label"

        if label_col not in df.columns:
            print(f"Skipping {method}: {label_col} not found in DataFrame.")
            continue
            
        # Convert model labels (-1 = outlier, otherwise inlier) to binary (1 = outlier, 0 = inlier)
        y_pred_labels = np.where(df[label_col].to_numpy() == -1, 1, 0)
        
        # Calculate Global Hard-Label Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
        global_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        global_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate (Flagged QM9s)
        
        row = {
            "Method": method.upper(),
            "Global_Recall": round(global_recall, 3),
            "False_Positive_Rate": round(global_fpr, 4),
            "Flagged_QM9_Count": fp,
            "Total_Flagged": tp,   # True Positives (Correctly identified outliers)
            "Total_Missed": fn     # False Negatives (Missed outliers)
        }
        
        # Calculate Global Continuous Metric (ROC-AUC) if scores exist
        score_col = f"{method}_score"
        if score_col in df.columns:
            y_scores = df[score_col].to_numpy()
            # Handle NaNs just in case
            y_scores = np.nan_to_num(y_scores, nan=0.0) 
            roc_auc = roc_auc_score(y_true, y_scores)
            row["ROC_AUC"] = round(roc_auc, 3)
        else:
            row["ROC_AUC"] = None
            
        # 3. Calculate Category-Specific Recall
        for cat in categories:
            # Mask for just this category
            cat_mask = (df["outlier_category"] == cat).fill_null(False).to_numpy()
            
            # True positives for this specific category
            cat_tp = np.sum((y_pred_labels == 1) & cat_mask)
            cat_total = np.sum(cat_mask)
            
            cat_recall = cat_tp / cat_total if cat_total > 0 else 0
            row[f"Recall: {cat}"] = round(cat_recall, 3)
            
        results.append(row)
        
    return pl.DataFrame(results)


def plot_score_distributions(df: pl.DataFrame):
    # Convert to Pandas for Seaborn
    pd_df = df.select(["outlier_category", "lof_score", "knn_score", "hdbscan_score"]).to_pandas()
    pd_df["outlier_category"] = pd_df["outlier_category"].fillna("Normal QM9")
    
    # Melt the dataframe so we can plot both scores side-by-side
    melted_df = pd_df.melt(
        id_vars=["outlier_category"], 
        value_vars=["lof_score", "knn_score", "hdbscan_score"], 
        var_name="Method", 
        value_name="Score"
    )
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x="outlier_category", y="Score", hue="Method")
    
    plt.title("Distribution of Outlier Scores by Category")
    plt.xticks(rotation=45)
    plt.ylabel("Outlier Score (Higher = More Anomalous)")
    plt.tight_layout()
    plt.show()
