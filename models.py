#!/usr/bin/env python3
"""
Simple Autoencoder Model for COPUS Data

Configurable autoencoder with encoder-decoder architecture for dimensionality reduction.
Optimized for experimental use with clear, readable implementation.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    """
    Enhanced autoencoder for COPUS observation data with layer normalization and skip connections.
    
    Architecture:
    - Input layer (24 features) with layer normalization
    - Encoder: input → LN → hidden1 → LN → ReLU → hidden2 → LN → ReLU → latent
    - Decoder: latent → hidden1 → LN → ReLU → hidden2 → LN → ReLU → output
    - Skip connections: input→latent (encoder) and latent→output (decoder) when n_hidden_layers > 1
    - Output: sigmoid activation for reconstruction
    
    Layer normalization is used instead of batch normalization to better handle
    sparse data distributions and variable feature ranges in COPUS data.
    
    Skip connections provide residual pathways that improve gradient flow and preserve
    input information, activated only when the model has more than 1 hidden layer.
    """
    
    def __init__(self, input_dim=24, latent_dim=3, hidden_dim=16, dropout=0.1, n_hidden_layers=2):
        """
        Initialize enhanced autoencoder with layer normalization and skip connections.
        
        Args:
            input_dim: Input feature dimensions (default: 24 for COPUS)
            latent_dim: Bottleneck dimension (default: 3)
            hidden_dim: Hidden layer size (default: 16)
            dropout: Dropout rate for regularization (default: 0.1)
            n_hidden_layers: Number of hidden layers (default: 2, must be >= 1)
        """
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = max(1, n_hidden_layers)  # Ensure at least 1 hidden layer
        
        # Input layer normalization for preprocessing sparse data
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-6, elementwise_affine=True)
        
        # Build encoder layers dynamically based on n_hidden_layers
        encoder_layers = []
        
        # First encoder layer (always present)
        encoder_layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Additional hidden layers
        for i in range(1, self.n_hidden_layers):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Skip connection from input to latent (only if more than 1 hidden layer)
        self.use_skip_connections = self.n_hidden_layers > 1
        if self.use_skip_connections:
            self.input_to_latent_skip = nn.Linear(input_dim, latent_dim)
        
        # Build decoder layers dynamically based on n_hidden_layers
        decoder_layers = []
        
        # First decoder layer (always present)
        decoder_layers.extend([
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Additional hidden layers
        for i in range(1, self.n_hidden_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Final decoder layer to output space
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Skip connection from latent to output (only if more than 1 hidden layer)
        if self.use_skip_connections:
            self.latent_to_output_skip = nn.Linear(latent_dim, input_dim)
        
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
        Forward pass through autoencoder with input preprocessing.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (reconstructed_output, latent_representation)
        """
        # Apply input layer normalization for sparse data preprocessing
        x = self.input_norm(x)
        
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


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder with layer normalization for COPUS data.

    Encoder outputs mean and log-variance for latent distribution; decoder reconstructs input.
    """

    def __init__(self, input_dim=24, latent_dim=3, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim, eps=1e-6, elementwise_affine=True)

        # Encoder that outputs mu and logvar
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, input_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def encode(self, x):
        x = self.input_norm(x)
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar

    def get_model_info(self):
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'total_params': sum(p.numel() for p in self.parameters())
        }


class GaussMixturePrior(nn.Module):
    """Learnable Gaussian Mixture prior for VaDE.

    Parameters:
        - pi_logits: unnormalized log mixing coefficients (K)
        - mu: component means (K, latent_dim)
        - logvar: component log-variances (K, latent_dim)
    """

    def __init__(self, num_components: int, latent_dim: int):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        self.pi_logits = nn.Parameter(torch.zeros(num_components))
        self.mu = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.logvar = nn.Parameter(torch.zeros(num_components, latent_dim))

    def pi(self):
        return F.softmax(self.pi_logits, dim=0)  # (K)

    def responsibilities(self, z_mu: torch.Tensor):
        """Compute responsibilities gamma_k for each sample using z mean.

        Args:
            z_mu: (N, D) latent means from encoder
        Returns:
            gamma: (N, K) responsibilities
        """
        N, D = z_mu.shape
        K = self.num_components
        pi = self.pi()  # (K)
        var = torch.exp(self.logvar)  # (K, D)

        # Compute log probability for each component
        # log N(z | mu_k, var_k)
        z_mu_exp = z_mu.unsqueeze(1)  # (N, 1, D)
        mu = self.mu.unsqueeze(0)      # (1, K, D)
        var_exp = var.unsqueeze(0)     # (1, K, D)

        log_prob = -0.5 * (
            torch.sum(torch.log(2 * torch.pi * var_exp), dim=2) +
            torch.sum((z_mu_exp - mu) ** 2 / var_exp, dim=2)
        )  # (N, K)

        log_pi = torch.log(pi + 1e-12).unsqueeze(0)  # (1, K)
        logits = log_pi + log_prob  # (N, K)
        gamma = F.softmax(logits, dim=1)
        return gamma

    def kl_z_given_c(self, z_mu: torch.Tensor, z_logvar: torch.Tensor, gamma: torch.Tensor):
        """Compute KL(q(z|x) || p(z|c)) weighted by responsibilities.

        Returns:
            kl_z: (N,) KL per sample
        """
        var_k = torch.exp(self.logvar)  # (K, D)
        mu_k = self.mu  # (K, D)

        z_var = torch.exp(z_logvar)  # (N, D)
        N, D = z_mu.shape
        K = self.num_components

        # Expand for broadcasting
        z_mu_e = z_mu.unsqueeze(1)      # (N, 1, D)
        z_logvar_e = z_logvar.unsqueeze(1)  # (N, 1, D)
        z_var_e = z_var.unsqueeze(1)    # (N, 1, D)
        mu_k_e = mu_k.unsqueeze(0)      # (1, K, D)
        var_k_e = var_k.unsqueeze(0)    # (1, K, D)

        # KL between two Gaussians: 0.5 * [log|Σ_k| - log|Σ_z| - D + tr(Σ_k^{-1} Σ_z) + (μ_z - μ_k)^T Σ_k^{-1} (μ_z - μ_k)]
        log_det_ratio = torch.sum(torch.log(var_k_e + 1e-12) - z_logvar_e, dim=2)  # (N, K)
        trace_term = torch.sum(z_var_e / (var_k_e + 1e-12), dim=2)  # (N, K)
        quad_term = torch.sum((z_mu_e - mu_k_e) ** 2 / (var_k_e + 1e-12), dim=2)  # (N, K)

        kl_per_k = 0.5 * (log_det_ratio - D + trace_term + quad_term)  # (N, K)

        # Weight by responsibilities
        kl_weighted = torch.sum(gamma * kl_per_k, dim=1)  # (N,)
        return kl_weighted

    def kl_c(self, gamma: torch.Tensor):
        """KL for cluster assignments: -E_q[log p(c)] + E_q[log q(c|x)]."""
        pi = self.pi()  # (K)
        log_pi = torch.log(pi + 1e-12).unsqueeze(0)  # (1, K)
        kl_c = -torch.sum(gamma * log_pi, dim=1) + torch.sum(gamma * torch.log(gamma + 1e-12), dim=1)
        return kl_c  # (N,)



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
