#!/usr/bin/env python3
"""
Autoencoder training pipeline for COPUS dataset.
Simple, configurable training script with real-time visualization.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Import custom modules
from dataset import COPUSDataset, create_copus_dataloader
from models import SimpleAutoencoder, VariationalAutoencoder, GaussMixturePrior
from utils import create_combined_plot

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt/Wayland issues
import matplotlib.pyplot as plt
import seaborn as sns


def setup_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train autoencoder on COPUS dataset')
    
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=3, 
                       help='Latent dimension size (default: 3)')
    parser.add_argument('--hidden_dims', type=str, default='[64,64,64]',
                       help='Hidden layer dimensions as JSON list (default: [64,64,64])')
    parser.add_argument('--mode', type=str, default='ae', choices=['ae', 'vae', 'vade'],
                       help='Training mode: ae (autoencoder), vae (variational autoencoder), vade (VaDE with GMM prior)')
    parser.add_argument('--num_clusters', type=int, default=8,
                       help='Number of mixture components for VaDE (default: 8)')
    
    # Training configuration
    parser.add_argument('--initial_lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    
    # Data and device
    parser.add_argument('--data_path', type=str, default='data/stains.csv',
                       help='Path to dataset (default: data/stains.csv)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu (default: auto)')
    
    # Output
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model weights')
    parser.add_argument('--plot_freq', type=int, default=1,
                       help='Plot update frequency in epochs (default: 1)')
    parser.add_argument('--no_log_scale', action='store_true',
                       help='Disable log scale for loss plotting (default: False)')
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    if args.latent_dim <= 0:
        raise ValueError(f"Latent dimension must be positive, got {args.latent_dim}")
    
    # Parse hidden dimensions from JSON string
    try:
        hidden_dims = json.loads(args.hidden_dims)
        if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
            raise ValueError("Hidden dimensions must be a non-empty list")
        if not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers")
        args.hidden_dims_list = hidden_dims
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format for hidden dimensions: {args.hidden_dims}")
    
    if args.initial_lr <= 0 or args.eta_min <= 0:
        raise ValueError("Learning rates must be positive")
    
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {args.epochs}")
    
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if args.mode == 'vade' and args.num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive for VaDE, got {args.num_clusters}")


def setup_device(device_arg):
    """Setup computing device with proper GPU utilization."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    
    # Additional GPU setup if CUDA is available
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable memory optimization
        torch.cuda.empty_cache()
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
    
    return device


def create_model(input_dim, latent_dim, hidden_dims, mode, num_clusters=None):
    """Create model based on training mode with dynamic hidden dimensions."""
    if mode == 'ae':
        model = SimpleAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dims[0],  # Use first dimension for compatibility
            dropout=0.1
        )
        return model, None
    else:
        vae = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dims[0],  # Use first dimension for compatibility
            dropout=0.1
        )
        if mode == 'vae':
            return vae, None
        elif mode == 'vade':
            prior = GaussMixturePrior(num_components=num_clusters, latent_dim=latent_dim)
            return vae, prior
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def sparse_aware_loss(reconstructed, original, sparsity_weight=0.1):
    """
    Custom loss function that handles sparse data better.
    
    Combines MSE loss with L1 loss for sparse reconstruction,
    giving more weight to non-zero elements.
    """
    # Standard MSE loss
    mse_loss = nn.MSELoss()(reconstructed, original)
    
    # L1 loss for sparsity awareness (better for sparse data)
    l1_loss = nn.L1Loss()(reconstructed, original)
    
    # Weight the losses: emphasize reconstruction of non-zero elements
    non_zero_mask = (original != 0).float()
    if non_zero_mask.sum() > 0:
        weighted_mse = (non_zero_mask * (reconstructed - original) ** 2).sum() / non_zero_mask.sum()
        weighted_l1 = (non_zero_mask * torch.abs(reconstructed - original)).sum() / non_zero_mask.sum()
        
        # Combine losses with emphasis on non-zero reconstruction
        total_loss = 0.7 * weighted_mse + 0.2 * weighted_l1 + 0.1 * mse_loss
    else:
        total_loss = mse_loss
    
    return total_loss

def kl_standard_normal(mu, logvar):
    """KL divergence between q(z|x)=N(mu,diag(var)) and p(z)=N(0,I). Returns (batch,)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def train_epoch_ae(model, dataloader, criterion, optimizer, device, use_sparse_loss=True):
    """Train AE for one epoch with proper device handling."""
    model.train()
    model = model.to(device)
    total_loss = 0
    num_batches = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        reconstructed, _ = model(batch_data)
        loss = sparse_aware_loss(reconstructed, batch_data) if use_sparse_loss else criterion(reconstructed, batch_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def train_epoch_vae(model, dataloader, optimizer, device):
    """Train VAE for one epoch with sparse-aware reconstruction + KL to N(0,I)."""
    model.train()
    model = model.to(device)
    total_loss = 0
    num_batches = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        recon, z, mu, logvar = model(batch_data)
        recon_loss = sparse_aware_loss(recon, batch_data)
        kl = kl_standard_normal(mu, logvar).mean()
        loss = recon_loss + kl
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def train_epoch_vade(model, prior, dataloader, optimizer, device):
    """Train VaDE (VAE with GMM prior) for one epoch with proper device handling.

    Total loss = reconstruction + KL(q(z|x) || p(z|c)) + KL(q(c|x) || p(c)).
    """
    model.train()
    model = model.to(device)
    if prior is not None:
        prior = prior.to(device)
    total_loss = 0
    num_batches = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        recon, z, mu, logvar = model(batch_data)
        recon_loss = sparse_aware_loss(recon, batch_data)
        gamma = prior.responsibilities(mu)  # (N, K)
        kl_z = prior.kl_z_given_c(mu, logvar, gamma).mean()
        kl_c = prior.kl_c(gamma).mean()
        loss = recon_loss + kl_z + kl_c
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def update_training_plots(epochs, losses, current_epoch, output_dir, model=None, 
                         dataloader=None, device='cpu', latent_dim=3, 
                         use_log_scale=True, title_suffix=""):
    """Update and save both loss plot and latent space visualization."""
    
    # Use the new combined plot function from utils
    combined_plot_path, latent_plot_path = create_combined_plot(
        epochs=epochs,
        losses=losses,
        current_epoch=current_epoch,
        output_dir=output_dir,
        latent_dim=latent_dim,
        dataloader=dataloader,
        model=model,
        device=device,
        use_log_scale=use_log_scale,
        title_suffix=title_suffix
    )
    
    return combined_plot_path, latent_plot_path


def main():
    """Main training function."""
    print("Autoencoder Training Pipeline")
    print("=" * 50)
    
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Setup
        output_dir = setup_output_dir()
        device = setup_device(args.device)
        
        print(f"Configuration:")
        print(f"  Latent dim: {args.latent_dim}")
        print(f"  Hidden dims: {args.hidden_dims_list}")
        print(f"  Mode: {args.mode}")
        if args.mode == 'vade':
            print(f"  VaDE clusters: {args.num_clusters}")
        print(f"  Initial LR: {args.initial_lr}")
        print(f"  Min LR: {args.eta_min}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print()
        
        # Load dataset
        print("Loading dataset...")
        dataset = COPUSDataset(args.data_path, device=device)
        print(f"Dataset loaded: {dataset.get_info()}")
        
        # Create dataloader
        dataloader = create_copus_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"DataLoader created with batch size {args.batch_size}")
        
        # Create model
        print("Creating model...")
        model, prior = create_model(
            input_dim=dataset.n_features,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims_list,
            mode=args.mode,
            num_clusters=args.num_clusters
        )
        model = model.to(device)
        if prior is not None:
            prior = prior.to(device)
        print(f"Model created: {model.get_model_info()}")
        
        # Setup training with improved settings for large networks
        criterion = nn.MSELoss()
        # Include prior parameters if VaDE so they are optimized
        params = list(model.parameters()) + (list(prior.parameters()) if prior is not None else [])
        optimizer = optim.AdamW(params, lr=args.initial_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        
        # Training tracking
        losses = np.zeros(args.epochs)
        epochs = np.arange(args.epochs)
        
        print(f"\nStarting training for {args.epochs} epochs...")
        print("-" * 50)
        
        # Training loop
        for epoch in range(args.epochs):
            # Train for one epoch according to mode
            if args.mode == 'ae':
                avg_loss = train_epoch_ae(model, dataloader, criterion, optimizer, device, use_sparse_loss=True)
            elif args.mode == 'vae':
                avg_loss = train_epoch_vae(model, dataloader, optimizer, device)
            elif args.mode == 'vade':
                avg_loss = train_epoch_vade(model, prior, dataloader, optimizer, device)
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
            losses[epoch] = avg_loss
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # Update plot
            if (epoch + 1) % args.plot_freq == 0:
                combined_plot_path, latent_plot_path = update_training_plots(
                    epochs, losses, epoch, output_dir, 
                    model=model, dataloader=dataloader, device=device,
                    latent_dim=args.latent_dim,
                    use_log_scale=not args.no_log_scale, 
                    title_suffix=args.mode.upper()
                )
                print(f"  Combined plot updated: {combined_plot_path}")
                if latent_plot_path:
                    print(f"  Latent plot updated: {latent_plot_path}")
        
        print("-" * 50)
        print(f"Training completed!")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Best loss: {losses.min():.4f} (epoch {losses.argmin()+1})")
        
        # Save model if requested
        if args.save_model:
            hidden_dims_str = '_'.join(map(str, args.hidden_dims_list))
            model_path = output_dir / f'autoencoder_latent{args.latent_dim}_hidden{hidden_dims_str}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
        
        # Final combined plot
        final_combined_plot_path, final_latent_plot_path = update_training_plots(
            epochs, losses, args.epochs-1, output_dir,
            model=model, dataloader=dataloader, device=device,
            latent_dim=args.latent_dim,
            use_log_scale=not args.no_log_scale, 
            title_suffix=args.mode.upper()
        )
        print(f"Final combined plot saved: {final_combined_plot_path}")
        if final_latent_plot_path:
            print(f"Final latent plot saved: {final_latent_plot_path}")
        
        print(f"\nAll outputs saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
