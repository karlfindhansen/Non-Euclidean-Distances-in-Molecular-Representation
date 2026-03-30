from src.datasets import QM9Dataset

if __name__ == "__main__":
    qm9 = QM9Dataset(limit=5000, sampling_strategy="stratified", stratify_by=["num_atoms", "gap"])
    df = qm9.load()

    df.head(5)

