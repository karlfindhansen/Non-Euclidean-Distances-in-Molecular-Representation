from src.datasets import QM9Dataset

if __name__ == "__main__":
    dataset = QM9Dataset()
    dataset.load(force_process=True)