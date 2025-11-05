#!/usr/bin/env python3
"""
Dynamic Autoencoder Model for Hyperparameter Optimization

Configurable autoencoder with dynamic hidden layer architecture for COPUS data.
Supports variable number of hidden layers and dimensions for hyperparameter search.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional


class DynamicAutoencoder(nn.Module):
    """
    Dynamic autoencoder with configurable hidden layer architecture.
    
    Architecture:
    - Input layer normalization
    - Encoder: input → [hidden layers] → latent
    - Decoder: latent → [hidden layers] → output
    - Layer normalization and ReLU activations between layers
    """
    
    def __init__(self, input_dim: int = 24, latent_dim: int = 3, 
                 hidden_dims: List[int] = None, dropout: float = 0.1):
        """
        Initialize dynamic autoencoder with configurable architecture.
        
        Args:
            input_dim: Input feature dimensions (default: 24 for COPUS)
            latent_dim: Bottleneck dimension (default: 3)
            hidden_dims: List of hidden layer dimensions (default: [64, 32])
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super(DynamicAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout_rate = dropout
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-6, elementwise_affine=True)
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder with dynamic architecture."""
        layers = []
        
        # Input to first hidden layer
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder with dynamic architecture."""
        layers = []
        
        # Latent to first hidden layer (reverse of encoder)
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer to output
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (reconstructed_output, latent_representation)
        """
        x = self.input_norm(x)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (encoder only)."""
        x = self.input_norm(x)
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent representation (decoder only)."""
        return self.decoder(latent)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_rate,
            'total_params': sum(p.numel() for p in self.parameters()),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters())
        }


def create_dynamic_autoencoder(input_dim: int = 24, latent_dim: int = 3, 
                              hidden_dims: List[int] = None, dropout: float = 0.1) -> DynamicAutoencoder:
    """
    Factory function to create dynamic autoencoder.
    
    Args:
        input_dim: Input dimensions (default: 24 for COPUS)
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        
    Returns:
        DynamicAutoencoder: Configured dynamic autoencoder
    """
    return DynamicAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )