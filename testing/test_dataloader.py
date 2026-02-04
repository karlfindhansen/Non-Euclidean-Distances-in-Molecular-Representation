from utils.load_datasets import OMat24Loader

loader = OMat24Loader(split="train")

train_df, train_counts = loader.get_omat24(sample_size=100)

print(train_df.head())