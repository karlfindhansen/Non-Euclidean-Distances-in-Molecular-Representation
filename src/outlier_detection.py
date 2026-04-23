import polars as pl
import numpy as np

from sklearn.cluster import HDBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def dist_to_soap_kernel(dist_matrix: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - (dist_matrix ** 2) / 2.0, 0.0, 1.0)


def hdbscan_outliers(
    df: pl.DataFrame,
    embedding: np.ndarray,
) -> pl.DataFrame:
    clusterer = HDBSCAN(min_cluster_size=150, min_samples=1, metric="precomputed")
    labels = clusterer.fit_predict(embedding)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    print(f"HDBSCAN — {len(unique_labels)} distinct: {label_summary}")

    return df.with_columns(pl.Series("hdbscan_labels", labels))

def ocsvm_outliers(df: pl.DataFrame, dist_matrix: np.ndarray, nu: float = 0.1) -> pl.DataFrame:
    kernel_matrix = dist_to_soap_kernel(dist_matrix)
    ocsvm = OneClassSVM(kernel="precomputed", nu=nu)
    labels = ocsvm.fit_predict(kernel_matrix)
    scores = ocsvm.decision_function(kernel_matrix)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    print(f"OCSVM — {len(unique_labels)} distinct: {label_summary}")

    return df.with_columns([
        pl.Series("ocsvm_label", labels),
        pl.Series("ocsvm_score", scores),
    ])

def lof_outliers(df: pl.DataFrame, dist_matrix: np.ndarray) -> pl.DataFrame:
    lof = LocalOutlierFactor(n_neighbors=20, metric="precomputed")
    labels = lof.fit_predict(dist_matrix)
    scores = -lof.negative_outlier_factor_ 

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_summary = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in zip(unique_labels, counts))
    print(f"LOF — {len(unique_labels)} distinct: {label_summary}")

    return df.with_columns([
        pl.Series("lof_label", labels),
        pl.Series("lof_score", scores),
    ])