"""
Data loading and sampling utilities for Creative Fatigue Analysis
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional
import json
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_stratified_sample(
    df: pd.DataFrame,
    sample_rate: float,
    random_seed: int = 42,
    click_col: str = 'click'
) -> pd.DataFrame:
    """
    Create stratified sample preserving click/non-click distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    sample_rate : float
        Proportion of data to sample (e.g., 0.01 for 1%)
    random_seed : int
        Random seed for reproducibility
    click_col : str
        Name of the click label column
        
    Returns:
    --------
    pd.DataFrame
        Stratified sample
    """
    np.random.seed(random_seed)
    
    # Stratify by click label
    sample = df.groupby(click_col, group_keys=False).apply(
        lambda x: x.sample(frac=sample_rate, random_state=random_seed)
    ).reset_index(drop=True)
    
    return sample


def save_sample_metadata(
    sample_df: pd.DataFrame,
    output_path: str,
    sample_rate: float,
    random_seed: int,
    original_size: int
):
    """Save metadata about the sample."""
    metadata = {
        'sample_date': datetime.now().isoformat(),
        'sample_rate': sample_rate,
        'random_seed': random_seed,
        'original_size': original_size,
        'sample_size': len(sample_df),
        'click_rate': sample_df['click'].mean() if 'click' in sample_df.columns else None
    }
    
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sample metadata saved to {metadata_path}")


def load_criteo_data(
    data_path: Optional[str] = None,
    use_huggingface: bool = True,
    dataset_name: str = "criteo/criteo-uplift",
    hf_file_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load Criteo Display Advertising Challenge dataset.
    
    Parameters:
    -----------
    data_path : str, optional
        Local path to dataset. If None, will try HuggingFace.
    use_huggingface : bool
        Whether to load from HuggingFace Hub
    dataset_name : str
        HuggingFace dataset name (default: "criteo/criteo-uplift")
    hf_file_path : str, optional
        Specific file path within HuggingFace dataset (e.g., "criteo-research-uplift-v2.1.csv.gz")
        If provided, will load this specific file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if use_huggingface and data_path is None:
        try:
            from datasets import load_dataset
            import os
            
            print("Loading Criteo dataset from HuggingFace...")
            
            if hf_file_path:
                # Load specific file from dataset
                print(f"Loading file: {hf_file_path}")
                # For CSV files in HuggingFace, we need to use data_files parameter
                dataset = load_dataset(
                    dataset_name,
                    data_files=hf_file_path,
                    split="train"
                )
            else:
                # Load default split
                dataset = load_dataset(dataset_name, split="train")
            
            # Convert to pandas
            if hasattr(dataset, 'to_pandas'):
                df = dataset.to_pandas()
            else:
                # If it's a DatasetDict, get the train split
                if isinstance(dataset, dict):
                    df = dataset['train'].to_pandas()
                else:
                    # Convert iterable to list then to DataFrame
                    df = pd.DataFrame(list(dataset))
            
            print(f"Loaded {len(df):,} records from HuggingFace")
            print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            return df
            
        except ImportError:
            print("HuggingFace datasets library not installed.")
            print("Install it with: pip install datasets")
            print("Or provide a local data_path")
            raise
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Please provide local data_path or install datasets library")
            print("Install: pip install datasets")
            raise
    
    if data_path:
        print(f"Loading data from {data_path}...")
        # Handle HuggingFace URL format
        if data_path.startswith("hf://"):
            # Convert hf:// URL to HuggingFace dataset loading
            path_parts = data_path.replace("hf://", "").split("/")
            if len(path_parts) >= 2:
                dataset_name = "/".join(path_parts[:-1])
                file_name = path_parts[-1]
                return load_criteo_data(
                    use_huggingface=True,
                    dataset_name=dataset_name,
                    hf_file_path=file_name
                )
            else:
                raise ValueError(f"Invalid HuggingFace URL format: {data_path}")
        
        # Handle different file formats
        # Check compressed formats first (before uncompressed)
        if data_path.endswith('.tsv.gz'):
            # Tab-separated values, gzipped
            print("Detected TSV.GZ format, loading with tab separator...")
            df = pd.read_csv(data_path, compression='gzip', sep='\t', low_memory=False)
        elif data_path.endswith('.csv.gz'):
            df = pd.read_csv(data_path, compression='gzip', low_memory=False)
        elif data_path.endswith('.tsv'):
            df = pd.read_csv(data_path, sep='\t', low_memory=False)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, low_memory=False)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Supported: .csv, .csv.gz, .tsv, .tsv.gz, .parquet")
        
        print(f"Loaded {len(df):,} records")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        return df
    
    raise ValueError("Either data_path must be provided or use_huggingface must be True")


def inspect_data_structure(df: pd.DataFrame) -> dict:
    """
    Inspect and document data structure.
    
    Returns:
    --------
    dict
        Dictionary with data structure information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None
    }
    
    return info


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally to avoid data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    train_ratio : float
        Proportion for training
    val_ratio : float
        Proportion for validation
    test_ratio : float
        Proportion for testing
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    Tuple of (train_df, val_df, test_df)
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    print(f"Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    print(f"Val: {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    print(f"Test: {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df

