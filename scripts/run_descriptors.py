import numpy as np
import polars as pl
from loguru import logger

from src.datasets import QM9Dataset
from src.descriptors import SOAPDescriptor, ACSFDescriptor

def descriptors_deliverable():

    loader = QM9Dataset()
    loader.load()

    soap = SOAPDescriptor(loader, r_cut=6.0, n_max=8)
    soap.compute()


    acsf = ACSFDescriptor(loader, r_cut=6.0)
    acsf.compute()

    loader.df = loader.df.filter(
        pl.col("soap_embedding").is_not_null() &
        pl.col("acsf_embedding").is_not_null()
    )

    soap_matrix = np.array(loader.df["soap_embedding"].to_list())
    acsf_matrix = np.array(loader.df["acsf_embedding"].to_list())

    np.save("results/descriptors/features_soap.npy", soap_matrix)
    np.save("results/descriptors/features_acsf.npy", acsf_matrix)

    logger.success(f"Generated descriptor features, with dimensions: SOAP: {soap_matrix.shape}, ACSF: {acsf_matrix.shape}")

if __name__ == "__main__":
    descriptors_deliverable()