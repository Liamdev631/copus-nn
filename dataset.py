#!/usr/bin/env python3
"""
Custom PyTorch Dataset for COPUS data with eager tensor loading.
Optimized for fast batch building on GPU.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path


class COPUSDataset(Dataset):
    """
    Custom PyTorch Dataset for COPUS observation data.
    
    Eagerly loads specified columns into tensors for fast batch building.
    All data is loaded into GPU memory for optimal performance.
    """
    
    # Target columns for tensor conversion
    TARGET_COLUMNS = [
        'L', 'Ind', 'CG', 'WG', 'OG', 'S_Anq', 'SQ', 'WC', 'PRD', 'SP', 
        'TQ', 'S_W', 'S_O', 'Lec', 'RTW', 'FUP', 'T_PQ', 'CQ', 'T_ANQ', 
        'MG', 'ONE_O_ONE', 'DV', 'ADMIN', 'T_W'
    ]
    
    def __init__(self, data_path, device='cuda', dtype=torch.float32):
        """
        Initialize dataset with eager tensor loading.
        
        Args:
            data_path: Path to CSV file
            device: Device to load tensors on ('cuda', 'cpu', etc.)
            dtype: Data type for tensors
        """
        self.data_path = Path(data_path)
        self.device = device
        self.dtype = dtype
        
        # Load data
        self.df = self._load_data()
        
        # Convert target columns to tensors
        self.tensors = self._create_tensors()
        
        # Store metadata
        self.n_samples = len(self.df)
        self.n_features = len(self.TARGET_COLUMNS)
        
    def _load_data(self):
        """Load CSV data with error handling."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            
            # Validate required columns exist
            missing_cols = set(self.TARGET_COLUMNS) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load {self.data_path}: {e}")
    
    def _create_tensors(self):
        """Convert target columns to tensors with GPU optimization."""
        tensors = {}
        
        for col in self.TARGET_COLUMNS:
            # Handle missing values and convert to numeric
            col_data = pd.to_numeric(self.df[col], errors='coerce')
            
            # Fill NaN values with 0 (or could use mean/median)
            col_data = col_data.fillna(0.0)
            
            # Convert to numpy array
            np_array = col_data.values.astype(np.float32)
            
            # Create tensor and move to device
            tensor = torch.tensor(np_array, dtype=self.dtype, device=self.device)
            
            tensors[col] = tensor
        
        return tensors
    
    def __len__(self):
        """Return dataset size."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get item by index - returns stacked tensor of all features.
        
        Args:
            idx: Index or slice
            
        Returns:
            torch.Tensor: Stacked feature tensor of shape (n_features,)
        """
        if isinstance(idx, slice):
            # Handle slicing for batch operations
            indices = range(*idx.indices(self.n_samples))
            return torch.stack([self._get_single_item(i) for i in indices])
        else:
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx):
        """Get single item as stacked tensor."""
        features = []
        for col in self.TARGET_COLUMNS:
            features.append(self.tensors[col][idx])
        
        return torch.stack(features)
    
    def get_batch(self, indices):
        """
        Get batch of items - optimized for fast batch building.
        
        Args:
            indices: List of indices
            
        Returns:
            torch.Tensor: Batch tensor of shape (batch_size, n_features)
        """
        batch_features = []
        for col in self.TARGET_COLUMNS:
            batch_features.append(self.tensors[col][indices])
        
        return torch.stack(batch_features, dim=1)
    
    def get_column(self, column_name):
        """
        Get specific column tensor.
        
        Args:
            column_name: Name of column to retrieve
            
        Returns:
            torch.Tensor: Column tensor
        """
        if column_name not in self.tensors:
            raise ValueError(f"Column '{column_name}' not found. Available: {list(self.tensors.keys())}")
        
        return self.tensors[column_name]
    
    def get_info(self):
        """Get dataset information."""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'device': self.device,
            'dtype': str(self.dtype),
            'columns': self.TARGET_COLUMNS,
            'memory_usage_mb': sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values()) / (1024 * 1024)
        }


def create_copus_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create optimized DataLoader for COPUS dataset.
    
    Note: num_workers should be 0 since data is already on GPU.
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 0 for GPU-loaded data
        pin_memory=False,  # False since data is already on GPU
        drop_last=False
    )

def main():
    """Test the dataset implementation."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python dataset.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        # Create dataset
        print(f"Loading dataset from: {data_path}")
        dataset = COPUSDataset(data_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Display info
        info = dataset.get_info()
        print(f"\nDataset loaded successfully!")
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        print(f"Device: {info['device']}")
        print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
        
        # Test single item access
        print(f"\nSample tensor shape: {dataset[0].shape}")
        print(f"Sample values: {dataset[0][:5]}")  # First 5 features
        
        # Test batch access
        batch_indices = [0, 1, 2, 3, 4]
        batch = dataset.get_batch(batch_indices)
        print(f"\nBatch shape: {batch.shape}")
        print(f"Batch device: {batch.device}")
        
        # Test DataLoader
        dataloader = create_copus_dataloader(dataset, batch_size=8)
        print(f"\nDataLoader created with batch_size=8")
        
        # Get one batch from DataLoader
        for batch_data in dataloader:
            print(f"DataLoader batch shape: {batch_data.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()