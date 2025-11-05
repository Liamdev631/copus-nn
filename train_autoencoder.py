#!/usr/bin/env python3
"""
Autoencoder training pipeline for COPUS dataset.
Simple, configurable training script with real-time visualization.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Import custom modules
from dataset import COPUSDataset, create_copus_dataloader
from models import SimpleAutoencoder

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
    parser.add_argument('--hidden_dim', type=int, default=16,
                       help='Hidden layer dimensions (default: 16)')
    
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
    
    if args.hidden_dim <= 0:
        raise ValueError(f"Hidden dimensions must be positive, got {args.hidden_dim}")
    
    if args.initial_lr <= 0 or args.eta_min <= 0:
        raise ValueError("Learning rates must be positive")
    
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {args.epochs}")
    
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")


def setup_device(device_arg):
    """Setup computing device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return device


def create_model(input_dim, latent_dim, hidden_dim):
    """Create autoencoder model."""
    model = SimpleAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=0.1
    )
    return model


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

def train_epoch(model, dataloader, criterion, optimizer, device, use_sparse_loss=True):
    """Train for one epoch with improved loss function."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_data in dataloader:
        # Move data to device if needed (already on device for GPU dataset)
        if device == 'cpu' and batch_data.device != device:
            batch_data = batch_data.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, latent = model(batch_data)
        
        # Calculate loss with sparse awareness
        if use_sparse_loss:
            loss = sparse_aware_loss(reconstructed, batch_data)
        else:
            loss = criterion(reconstructed, batch_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def update_loss_plot(epochs, losses, current_epoch, output_dir, use_log_scale=True):
    """Update and save loss plot."""
    plt.figure(figsize=(10, 6))
    
    # Create plot
    sns.set_style("whitegrid")
    
    # Apply log transformation if requested
    if use_log_scale:
        # Add small epsilon to avoid log(0)
        log_losses = np.log10(losses[:current_epoch+1] + 1e-10)
        plt.plot(epochs[:current_epoch+1], log_losses, 
                marker='o', markersize=4, linewidth=2, color='blue')
        ylabel = 'Log₁₀(MSE Loss + ε)'
        current_loss_display = np.log10(losses[current_epoch] + 1e-10)
    else:
        plt.plot(epochs[:current_epoch+1], losses[:current_epoch+1], 
                marker='o', markersize=4, linewidth=2, color='blue')
        ylabel = 'MSE Loss'
        current_loss_display = losses[current_epoch]
    
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add current epoch info
    if current_epoch >= 0:
        plt.axvline(x=current_epoch, color='red', linestyle='--', alpha=0.5)
        plt.text(current_epoch, current_loss_display, 
                f'Epoch {current_epoch+1}\nLoss: {losses[current_epoch]:.4f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10, ha='center')
    
    # Save plot
    plot_path = output_dir / 'training_loss.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


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
        print(f"  Hidden dim: {args.hidden_dim}")
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
        model = create_model(
            input_dim=dataset.n_features,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        print(f"Model created: {model.get_model_info()}")
        
        # Setup training with improved settings for large networks
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        
        # Training tracking
        losses = np.zeros(args.epochs)
        epochs = np.arange(args.epochs)
        
        print(f"\nStarting training for {args.epochs} epochs...")
        print("-" * 50)
        
        # Training loop
        for epoch in range(args.epochs):
            # Train for one epoch with sparse-aware loss
            avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, use_sparse_loss=True)
            losses[epoch] = avg_loss
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # Update plot
            if (epoch + 1) % args.plot_freq == 0:
                plot_path = update_loss_plot(epochs, losses, epoch, output_dir, use_log_scale=not args.no_log_scale)
                print(f"  Plot updated: {plot_path}")
        
        print("-" * 50)
        print(f"Training completed!")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Best loss: {losses.min():.4f} (epoch {losses.argmin()+1})")
        
        # Save model if requested
        if args.save_model:
            model_path = output_dir / f'autoencoder_latent{args.latent_dim}_hidden{args.hidden_dim}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
        
        # Final plot
        final_plot_path = update_loss_plot(epochs, losses, args.epochs-1, output_dir, use_log_scale=not args.no_log_scale)
        print(f"Final plot saved: {final_plot_path}")
        
        print(f"\nAll outputs saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()