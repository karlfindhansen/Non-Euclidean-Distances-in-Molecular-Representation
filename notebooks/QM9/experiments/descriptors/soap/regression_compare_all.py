from src.datasets import QM9Dataset

if __name__ == "__main__":

    qm9 = QM9Dataset(limit=80_000, descriptors=["soap", "transformer", ""])
    df = qm9.load()