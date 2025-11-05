#!/usr/bin/env python3
"""
Simple Autoencoder Model for COPUS Data

Configurable autoencoder with encoder-decoder architecture for dimensionality reduction.
Optimized for experimental use with clear, readable implementation.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class SimpleAutoencoder(nn.Module):
    """
    Simple autoencoder for COPUS observation data.
    
    Architecture:
    - Input layer (24 features)
    - Encoder: hidden layer → ReLU → latent bottleneck
    - Decoder: hidden layer → ReLU → output layer
    - Output: sigmoid activation for reconstruction
    """
    
    def __init__(self, input_dim=24, latent_dim=3, hidden_dim=16, dropout=0.1):
        """
        Initialize autoencoder with configurable dimensions.
        
        Args:
            input_dim: Input feature dimensions (default: 24 for COPUS)
            latent_dim: Bottleneck dimension (default: 3)
            hidden_dim: Hidden layer size (default: 16)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: input → hidden → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: latent → hidden → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid for output reconstruction
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (reconstructed_output, latent_representation)
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x):
        """
        Get latent representation (encoder only).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, latent):
        """
        Reconstruct from latent representation (decoder only).
        
        Args:
            latent: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim)
        """
        return self.decoder(latent)
    
    def get_model_info(self):
        """Get model architecture information."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'total_params': sum(p.numel() for p in self.parameters()),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters())
        }


def create_autoencoder_for_copus(latent_dim=3, **kwargs):
    """
    Factory function to create autoencoder optimized for COPUS data.
    
    Args:
        latent_dim: Dimension of latent space (default: 3)
        **kwargs: Additional arguments for SimpleAutoencoder
        
    Returns:
        SimpleAutoencoder: Configured autoencoder for COPUS data
    """
    return SimpleAutoencoder(input_dim=24, latent_dim=latent_dim, **kwargs)


def main():
    """Test autoencoder implementation."""
    import sys
    
    print("Testing SimpleAutoencoder...")
    
    # Create model with default parameters
    model = create_autoencoder_for_copus(latent_dim=3)
    
    # Display model info
    info = model.get_model_info()
    print(f"\nModel Configuration:")
    print(f"  Input dimensions: {info['input_dim']}")
    print(f"  Latent dimensions: {info['latent_dim']}")
    print(f"  Hidden dimensions: {info['hidden_dim']}")
    print(f"  Total parameters: {info['total_params']}")
    
    # Test forward pass
    print(f"\nTesting forward pass:")
    batch_size = 8
    test_input = torch.randn(batch_size, 24)
    
    reconstructed, latent = model(test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {reconstructed.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Test encoder-only access
    print(f"\nTesting encoder access:")
    encoder_output = model.encode(test_input)
    print(f"  Encoder output shape: {encoder_output.shape}")
    
    # Test decoder-only access
    print(f"\nTesting decoder access:")
    test_latent = torch.randn(batch_size, 3)
    decoder_output = model.decode(test_latent)
    print(f"  Decoder output shape: {decoder_output.shape}")
    
    # Test different latent dimensions
    print(f"\nTesting different latent dimensions:")
    for latent_dim in [2, 5, 10]:
        test_model = create_autoencoder_for_copus(latent_dim=latent_dim)
        _, test_latent = test_model(test_input)
        print(f"  Latent dim {latent_dim}: shape {test_latent.shape}")
    
    print(f"\n✓ Autoencoder implementation successful!")


if __name__ == "__main__":
    main()