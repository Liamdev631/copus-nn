#!/usr/bin/env python3
"""
Multi-latent dimension training pipeline for COPUS dataset.
Trains autoencoders with different latent dimensions and compares their performance.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import itertools

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
    parser = argparse.ArgumentParser(description='Train autoencoders with multiple latent dimensions on COPUS dataset')
    
    # Multiple latent dimensions
    parser.add_argument('--latent_dims', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                       help='List of latent dimensions to test (default: [2, 4, 8, 16, 32])')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimensions (default: 64)')
    
    # Training configuration
    parser.add_argument('--initial_lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    
    # Data and device
    parser.add_argument('--data_path', type=str, default='data/stains.csv',
                       help='Path to dataset (default: data/stains.csv)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu (default: auto)')
    
    # Output
    parser.add_argument('--save_models', action='store_true',
                       help='Save all trained model weights')
    parser.add_argument('--plot_freq', type=int, default=1,
                       help='Plot update frequency in epochs (default: 1)')
    parser.add_argument('--no_log_scale', action='store_true',
                       help='Disable log scale for loss plotting (default: False)')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if not args.latent_dims or any(d <= 0 for d in args.latent_dims):
        raise ValueError("All latent dimensions must be positive")
    
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


def evaluate_model(model, dataloader, device):
    """Evaluate model on test data and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            if device == 'cpu' and batch_data.device != device:
                batch_data = batch_data.to(device)
            
            reconstructed, latent = model(batch_data)
            loss = sparse_aware_loss(reconstructed, batch_data)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model_with_latent_dim(latent_dim, train_dataset, test_dataset, full_dataset, train_loader, test_loader, args, device, output_dir):
    """Train a single autoencoder model with specified latent dimension and track test loss."""
    print(f"\n{'='*60}")
    print(f"Training model with latent dimension: {latent_dim}")
    print(f"{'='*60}")
    
    # Create model
    model = SimpleAutoencoder(
        input_dim=full_dataset.n_features,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Model created: {{'input_dim': {full_dataset.n_features}, 'latent_dim': {latent_dim}, 'hidden_dim': {args.hidden_dim}, 'total_params': {total_params}, 'encoder_params': {encoder_params}, 'decoder_params': {decoder_params}}}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    
    # Training tracking - now track both train and test losses
    train_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    epochs = np.arange(args.epochs)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch with sparse-aware loss
        avg_train_loss = 0
        model.train()
        num_batches = 0
        
        for batch_data in train_loader:
            # Move data to device if needed (already on device for GPU dataset)
            if device == 'cpu' and batch_data.device != device:
                batch_data = batch_data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent = model(batch_data)
            
            # Calculate loss with sparse awareness
            loss = sparse_aware_loss(reconstructed, batch_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            avg_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = avg_train_loss / num_batches
        train_losses[epoch] = avg_train_loss
        
        # Evaluate on test set
        avg_test_loss = evaluate_model(model, test_loader, device)
        test_losses[epoch] = avg_test_loss
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print progress with both train and test losses
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {current_lr:.6f}")
    
    # Save model if requested
    if args.save_models:
        model_path = output_dir / f'autoencoder_latent{latent_dim}_hidden{args.hidden_dim}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
    
    return test_losses, model  # Return test losses instead of train losses


def plot_multiple_losses(all_losses, latent_dims, current_epoch, output_dir, use_log_scale=True):
    """Plot losses from multiple latent dimensions on the same plot."""
    plt.figure(figsize=(12, 8))
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(latent_dims)))
    
    # Plot each latent dimension (only up to current epoch)
    epochs = np.arange(current_epoch + 1)
    for i, (latent_dim, losses) in enumerate(zip(latent_dims, all_losses)):
        if use_log_scale:
            # Add small epsilon to avoid log(0)
            log_losses = np.log10(losses[:current_epoch + 1] + 1e-10)
            plt.plot(epochs, log_losses, 
                    marker='o', markersize=3, linewidth=2, 
                    color=colors[i], label=f'Latent dim: {latent_dim}')
        else:
            plt.plot(epochs, losses[:current_epoch + 1], 
                    marker='o', markersize=3, linewidth=2, 
                    color=colors[i], label=f'Latent dim: {latent_dim}')
    
    # Customize plot
    sns.set_style("whitegrid")
    plt.title(f'Training Loss Comparison Across Different Latent Dimensions (Epoch {current_epoch + 1})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Log₁₀(MSE Loss + ε)' if use_log_scale else 'MSE Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add summary statistics
    final_losses = [losses[current_epoch] for losses in all_losses]
    min_loss_idx = np.argmin(final_losses)
    
    summary_text = f"""Current Losses (Epoch {current_epoch + 1}):
"""
    for i, (latent_dim, final_loss) in enumerate(zip(latent_dims, final_losses)):
        if i == min_loss_idx:
            summary_text += f"• Latent dim {latent_dim}: {final_loss:.4f} ⭐\n"
        else:
            summary_text += f"• Latent dim {latent_dim}: {final_loss:.4f}\n"
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # Save plot
    plot_path = output_dir / 'multiple_latent_dims_loss.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    """Main function to train multiple autoencoders with different latent dimensions using train/test split."""
    print("Multi-Latent Dimension Autoencoder Training Pipeline")
    print("=" * 60)
    
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Setup
        output_dir = setup_output_dir()
        device = setup_device(args.device)
        
        print(f"Configuration:")
        print(f"  Latent dimensions: {args.latent_dims}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Initial LR: {args.initial_lr}")
        print(f"  Min LR: {args.eta_min}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Log scale: {not args.no_log_scale}")
        print()
        
        # Load full dataset
        print("Loading dataset...")
        full_dataset = COPUSDataset(args.data_path, device=device)
        print(f"Dataset loaded: {full_dataset.get_info()}")
        
        # Create train/test split (80/20 split)
        dataset_size = len(full_dataset)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train/Test split: {train_size}/{test_size} samples")
        
        # Create dataloaders
        train_loader = create_copus_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = create_copus_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"DataLoaders created with batch size {args.batch_size}")
        print()
        
        # Initialize tracking
        all_test_losses = []
        
        # Train each latent dimension one at a time
        for latent_dim_idx, latent_dim in enumerate(args.latent_dims):
            print(f"\n{'='*60}")
            print(f"Training model {latent_dim_idx + 1}/{len(args.latent_dims)}: latent dimension {latent_dim}")
            print(f"{'='*60}")
            
            # Train this model
            model_test_losses, _ = train_model_with_latent_dim(
                latent_dim=latent_dim,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                full_dataset=full_dataset,
                train_loader=train_loader,
                test_loader=test_loader,
                args=args,
                device=device,
                output_dir=output_dir
            )
            
            all_test_losses.append(model_test_losses)
            
            # Update the comparison plot after each model
            plot_multiple_losses(
                all_losses=all_test_losses,
                latent_dims=args.latent_dims[:len(all_test_losses)],  # Only plot completed models
                current_epoch=args.epochs - 1,  # Show full training history
                output_dir=output_dir,
                use_log_scale=not args.no_log_scale
            )
            
            print(f"✓ Completed training for latent dimension {latent_dim}")
        
        print(f"\n{'='*60}")
        print("Training completed for all latent dimensions!")
        print(f"{'='*60}")
        
        # Final summary
        print(f"\nFinal Summary:")
        final_losses = [losses[-1] for losses in all_test_losses]
        min_loss_idx = np.argmin(final_losses)
        
        for i, (latent_dim, final_loss) in enumerate(zip(args.latent_dims, final_losses)):
            if i == min_loss_idx:
                print(f"  Latent dim {latent_dim:2d}: {final_loss:.4f} ⭐ (Best)")
            else:
                print(f"  Latent dim {latent_dim:2d}: {final_loss:.4f}")
        
        print(f"\nPlot saved: {output_dir}/multiple_latent_dims_loss.png")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()