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
from utils import (
    setup_output_dir,
    setup_device,
    sparse_aware_loss,
    kl_standard_normal,
    evaluate_model,
    train_epoch_ae,
    train_epoch_vae,
    train_epoch_vade,
    create_model,
    create_combined_plot,
)

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# setup_output_dir is now imported from utils


def setup_matplotlib_backend(interactive: bool):
    """Configure matplotlib backend.

    If interactive is True, try to use a GUI backend (Qt5Agg or TkAgg). Otherwise, keep default.
    Falls back gracefully if GUI backends are unavailable.
    """
    try:
        if interactive:
            # Prefer Qt5Agg, then TkAgg
            for backend in ["Qt5Agg", "TkAgg"]:
                try:
                    matplotlib.use(backend, force=True)
                    return
                except Exception:
                    continue
        # If not interactive or no GUI backends available, keep current backend
    except Exception as e:
        print(f"Warning: Failed to set interactive backend: {e}")


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
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for plots and models (default: output)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model weights')
    parser.add_argument('--plot_freq', type=int, default=1,
                       help='Plot update frequency in epochs (default: 1)')
    parser.add_argument('--no_log_scale', action='store_true',
                       help='Disable log scale for loss plotting (default: False)')
    parser.add_argument('--interactive_3d', action='store_true',
                       help='Enable interactive 3D plot popup window for latent space visualization (default: False)')
    
    # Loss function configuration
    parser.add_argument('--loss_weights', type=str, default='[0.7, 0.2, 0.1]',
                       help='JSON list of loss weights [mse_weight, l1_weight, sparse_weight] (default: [0.7, 0.2, 0.1])')
    parser.add_argument('--mse_weight', type=float, default=0.7,
                       help='Weight for MSE loss component (default: 0.7)')
    parser.add_argument('--l1_weight', type=float, default=0.2,
                       help='Weight for L1 loss component (default: 0.2)')
    parser.add_argument('--sparse_weight', type=float, default=0.0,
                       help='Weight for sparse reconstruction component (default: 0.0, disabled)')
    
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
    
    # Validate loss function parameters
    if args.mse_weight < 0 or args.l1_weight < 0 or args.sparse_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    
    # Automatically enable sparse loss if sparse_weight is positive
    args.use_sparse_loss = args.sparse_weight > 0
    
    if abs(args.mse_weight + args.l1_weight + args.sparse_weight - 1.0) > 1e-6:
        # Normalize weights if they don't sum to 1
        total_weight = args.mse_weight + args.l1_weight + args.sparse_weight
        args.mse_weight /= total_weight
        args.l1_weight /= total_weight
        args.sparse_weight /= total_weight
        print(f"Normalized loss weights: MSE={args.mse_weight:.3f}, L1={args.l1_weight:.3f}, Sparse={args.sparse_weight:.3f}")
    
    # Print sparse loss status
    if args.use_sparse_loss:
        print(f"Sparse loss enabled (sparse_weight={args.sparse_weight:.3f})")
    else:
        print("Sparse loss disabled (sparse_weight=0)")


# setup_device is now imported from utils


# create_model is provided in utils for consolidated usage (if needed elsewhere)


# sparse_aware_loss now imported from utils

# kl_standard_normal now imported from utils

# train_epoch_ae now imported from utils

# train_epoch_vae now imported from utils

# train_epoch_vade now imported from utils


# evaluate_model now imported from utils


def plot_realtime_losses(train_losses, test_losses, current_epoch, output_dir, use_log_scale=True):
    """Plot both training and testing loss up to current epoch and save to file.

    Intended to be called during training to show both curves in real-time.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    epochs = np.arange(current_epoch + 1)
    train_series = train_losses[:current_epoch + 1]
    test_series = test_losses[:current_epoch + 1]

    if use_log_scale:
        train_series = np.log10(train_series + 1e-10)
        test_series = np.log10(test_series + 1e-10)
        ylabel = 'Log₁₀(MSE + ε)'
    else:
        ylabel = 'MSE'

    plt.plot(epochs, train_series, marker='o', markersize=4, linewidth=2, color='blue', label='Train loss')
    plt.plot(epochs, test_series, marker='o', markersize=4, linewidth=2, color='orange', label='Test loss')
    plt.title(f'Training vs Testing Loss Progress (Epoch {current_epoch + 1})', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = output_dir / 'testing_loss_progress.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path





def plot_latent_space(model, dataloader, output_dir, latent_dim=3, device='cpu', title_suffix='', interactive=False, epoch=None):
    """Create and save latent space visualization; optionally show interactive popup at the end.

    Saves a latent visualization image.
    If interactive=True, shows only the latent visualization in a popup window (no loss curves).
    """
    model.eval()
    latents = []

    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]
            if getattr(batch_data, 'device', device) != device:
                batch_data = batch_data.to(device)

            outputs = model(batch_data)
            if hasattr(model, 'encode'):
                enc = model.encode(batch_data)
                z = enc[0] if isinstance(enc, tuple) else enc
            elif isinstance(outputs, tuple) and len(outputs) >= 2:
                z = outputs[1]
            else:
                z = outputs

            latents.append(z.detach().cpu().numpy())

    all_latents = np.vstack(latents) if len(latents) > 0 else np.zeros((0, latent_dim))

    if latent_dim == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        sns.set_style("whitegrid")
        ax.scatter(all_latents[:, 0], all_latents[:, 1], c='blue', alpha=0.6, s=30, label=f'{len(all_latents)} Samples')
        ax.set_xlabel('Latent Dimension 1', fontsize=12)
        ax.set_ylabel('Latent Dimension 2', fontsize=12)
        title = '2D Latent Space'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_latents[:, 0], all_latents[:, 1], all_latents[:, 2], c='blue', alpha=0.6, s=30, label=f'{len(all_latents)} Samples')
        ax.set_xlabel('Latent Dimension 1', fontsize=12)
        ax.set_ylabel('Latent Dimension 2', fontsize=12)
        ax.set_zlabel('Latent Dimension 3', fontsize=12)
        title = '3D Latent Space'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

    # Generate filename based on whether this is a real-time or final plot
    latent_plot_path = output_dir / 'latent_encodings.png'
    if epoch is not None:
        title_suffix = f"Epoch {epoch+1}"
    
    plt.tight_layout()
    plt.savefig(latent_plot_path, dpi=150, bbox_inches='tight')

    # Show interactive popup only for latent visualization if requested
    if interactive:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Interactive display failed: {e}")

    plt.close()
    return latent_plot_path


def update_training_plots(epochs, losses, current_epoch, output_dir, model=None, 
                         dataloader=None, device='cpu', latent_dim=3, 
                         use_log_scale=True, title_suffix="", interactive_3d=False):
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
        title_suffix=title_suffix,
        interactive_3d=interactive_3d
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
        
        # Set matplotlib backend based on interactive_3d flag
        if args.interactive_3d:
            # Try to use interactive backend, fallback to TkAgg if default fails
            try:
                matplotlib.use('Qt5Agg')  # Preferred interactive backend
            except ImportError:
                try:
                    matplotlib.use('TkAgg')  # Fallback interactive backend
                except ImportError:
                    print("Warning: No interactive backend available. Interactive 3D plots will not display.")
                    matplotlib.use('Agg')  # Fallback to non-interactive
        else:
            matplotlib.use('Agg')  # Use non-interactive backend for batch processing
        
        # Setup
        output_dir = setup_output_dir(args.output_dir)
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
        print(f"  Loss weights: MSE={args.mse_weight:.3f}, L1={args.l1_weight:.3f}, Sparse={args.sparse_weight:.3f}")
        print(f"  Use sparse loss: {args.use_sparse_loss}")
        print()
        
        # Load dataset
        print("Loading dataset...")
        dataset = COPUSDataset(args.data_path, device=device)
        info = dataset.get_info()
        print(f"Dataset loaded: {info}")
        
        # Train/test split
        from torch.utils.data import random_split
        n = len(dataset)
        train_size = int(0.8 * n)
        test_size = n - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create dataloaders
        train_loader = create_copus_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = create_copus_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
        full_loader = create_copus_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"DataLoaders created: train={train_size} samples, test={test_size} samples, batch_size={args.batch_size}")
        
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
        train_losses = np.zeros(args.epochs)
        test_losses = np.zeros(args.epochs)
        epochs = np.arange(args.epochs)
        
        print(f"\nStarting training for {args.epochs} epochs...")
        print("-" * 50)
        
        # Training loop
        for epoch in range(args.epochs):
            # Train for one epoch according to mode
            if args.mode == 'ae':
                avg_train_loss = train_epoch_ae(model, train_loader, criterion, optimizer, device, 
                                                use_sparse_loss=args.use_sparse_loss,
                                                mse_weight=args.mse_weight, l1_weight=args.l1_weight, 
                                                sparse_weight=args.sparse_weight)
            elif args.mode == 'vae':
                avg_train_loss = train_epoch_vae(model, train_loader, optimizer, device,
                                                 mse_weight=args.mse_weight, l1_weight=args.l1_weight,
                                                 sparse_weight=args.sparse_weight)
            elif args.mode == 'vade':
                avg_train_loss = train_epoch_vade(model, prior, train_loader, optimizer, device,
                                                  mse_weight=args.mse_weight, l1_weight=args.l1_weight,
                                                  sparse_weight=args.sparse_weight)
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
            train_losses[epoch] = avg_train_loss

            # Evaluate on test set
            avg_test_loss = evaluate_model(model, test_loader, device,
                                           use_sparse_loss=args.use_sparse_loss,
                                           mse_weight=args.mse_weight, l1_weight=args.l1_weight,
                                           sparse_weight=args.sparse_weight)
            test_losses[epoch] = avg_test_loss
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {current_lr:.6f}")
            
            # Real-time loss plot update (both train and test losses)
            if (epoch + 1) % args.plot_freq == 0:
                plot_realtime_losses(train_losses, test_losses, epoch, output_dir, use_log_scale=not args.no_log_scale)
                
                # Real-time latent space visualization
                plot_latent_space(model, full_loader, output_dir, latent_dim=args.latent_dim, 
                                device=device, title_suffix=args.mode.upper(), epoch=epoch)
        
        print("-" * 50)
        print(f"Training completed!")
        print(f"Final Train loss: {train_losses[-1]:.4f}")
        print(f"Best Train loss: {train_losses.min():.4f} (epoch {train_losses.argmin()+1})")
        print(f"Final Test loss: {test_losses[-1]:.4f}")
        print(f"Best Test loss: {test_losses.min():.4f} (epoch {test_losses.argmin()+1})")
        
        # Save model if requested
        if args.save_model:
            hidden_dims_str = '_'.join(map(str, args.hidden_dims_list))
            model_path = output_dir / f'autoencoder_latent{args.latent_dim}_hidden{hidden_dims_str}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
        
        # Real-time loss plot already shows both curves; no separate final curves file needed

        # Final latent space visualization saved separately; interactive popup shows only latent
        final_latent_plot_path = plot_latent_space(model, full_loader, output_dir, latent_dim=args.latent_dim, device=device, title_suffix=args.mode.upper(), interactive=args.interactive_3d)
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
