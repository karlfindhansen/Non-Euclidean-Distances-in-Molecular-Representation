import os
import polars as pl
from loguru import logger

def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def validate_columns(df: pl.DataFrame, required_columns: set) -> None:
    """Validate that DataFrame contains required columns."""
    if df.is_empty():
        raise ValueError("DataFrame is empty")
    
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
def validate_size(df: pl.DataFrame, size: int) -> None:
    """Validate that DataFrame has correct amount of rows"""
    if len(df) != size:
        raise ValueError(f"DataFrame size does not match expected value. Expected: {size}, Actual: {len(df)}")

def get_device():
    """
    Returns the most powerful available torch device.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")
        
    return device
