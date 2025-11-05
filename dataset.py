#!/usr/bin/env python3
"""
Simple Dataset Loader for Experiments

Minimal implementation for loading datasets in experimental workflows.
Supports CSV, JSON, and Excel formats with basic error handling.
"""

import sys
from pathlib import Path
import pandas as pd


def load_dataset(data_path):
    """
    Load dataset from file path.
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format not supported or file corrupted
    """
    file_path = Path(data_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Failed to load {data_path}: {e}")

def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python dataset.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        # Load dataset
        df = load_dataset(data_path)
        
        # Basic confirmation output
        print(f"\nDataset loaded: {Path(data_path).name}")
        print(f"Samples: {len(df)}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if len(df) > 0:
            print(f"Preview:\n{df.head(n=5)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()