#!/usr/bin/env python3
"""
Hyperparameter Optimization for Autoencoder Hidden Dimensions

Systematic search for optimal hidden layer configurations given a fixed latent dimension.
Follows experimental coding rules for simplicity and rapid prototyping.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# Import custom modules
from dataset import COPUSDataset, create_copus_dataloader
from models import SimpleAutoencoder, VariationalAutoencoder, GaussMixturePrior, create_dynamic_autoencoder

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    setup_device,
    evaluate_model,
    train_epoch_ae,
    train_epoch_vae,
    train_epoch_vade,
    sparse_aware_loss,
    kl_standard_normal,
    create_model,
)

# create_model is now imported from utils

# evaluate_model now imported from utils

# train_epoch_ae now imported from utils

# train_epoch_vae now imported from utils

# train_epoch_vade now imported from utils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for autoencoder hidden dimensions')
    
    # Fixed latent dimension
    parser.add_argument('--latent_dim', type=int, default=3,
                       help='Fixed latent dimension (default: 3)')
    
    # Hidden dimension search space
    parser.add_argument('--hidden_dims_search', type=str, default='16,32,64,128',
                       help='Comma-separated list of hidden dimensions to test (default: 16,32,64,128)')
    parser.add_argument('--max_layers', type=int, default=3,
                       help='Maximum number of hidden layers (default: 3)')
    
    # Model architecture
    parser.add_argument('--mode', type=str, default='ae', choices=['ae', 'vae', 'vade'],
                       help='Training mode: ae (autoencoder), vae (variational autoencoder), vade (VaDE with GMM prior)')
    parser.add_argument('--num_clusters', type=int, default=8,
                       help='Number of mixture components for VaDE (default: 8)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs per configuration (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    
    # Data and device
    parser.add_argument('--data_path', type=str, default='data/stains.csv',
                       help='Path to dataset (default: data/stains.csv)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu (default: auto)')
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='output/hyperparameters',
                       help='Output directory (default: output/hyperparameters)')
    parser.add_argument('--save_models', action='store_true',
                       help='Save best model weights')
    parser.add_argument('--plot_freq', type=int, default=1,
                       help='Plot update frequency in epochs (default: 1)')
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


# setup_device now imported from utils


def generate_hidden_configs(search_dims: List[int], max_layers: int) -> List[List[int]]:
    """
    Generate all possible hidden layer configurations.
    
    Args:
        search_dims: List of dimensions to test
        max_layers: Maximum number of layers
        
    Returns:
        List of configurations, each is a list of hidden dimensions
    """
    configs = []
    
    # Single layer configurations
    for dim in search_dims:
        configs.append([dim])
    
    # Two layer configurations
    for dim1 in search_dims:
        for dim2 in search_dims:
            if dim2 <= dim1:  # Decreasing dimensions
                configs.append([dim1, dim2])
    
    # Three layer configurations
    if max_layers >= 3:
        for dim1 in search_dims:
            for dim2 in search_dims:
                for dim3 in search_dims:
                    if dim3 <= dim2 <= dim1:  # Decreasing dimensions
                        configs.append([dim1, dim2, dim3])
    
    return configs


def sparse_aware_loss(reconstructed, original, mse_weight=0.7, l1_weight=0.2, sparse_weight=0.1):
    """
    Configurable sparse-aware loss function.
    
    Combines MSE loss with L1 loss for sparse reconstruction,
    with configurable weights for each component.
    
    Args:
        reconstructed: Reconstructed data
        original: Original data
        mse_weight: Weight for MSE loss component
        l1_weight: Weight for L1 loss component  
        sparse_weight: Weight for sparse reconstruction component
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
        
        # Combine losses with configurable weights
        total_loss = mse_weight * weighted_mse + l1_weight * weighted_l1 + sparse_weight * mse_loss
    else:
        total_loss = mse_loss
    
    return total_loss


def train_configuration(config: List[int], latent_dim: int, train_loader: DataLoader, 
                       test_loader: DataLoader, args, device: str) -> Dict[str, Any]:
    """
    Train a single configuration and return results.
    
    Args:
        config: Hidden layer dimensions configuration
        latent_dim: Fixed latent dimension
        train_loader: Training data loader
        test_loader: Test data loader
        args: Training arguments
        device: Computing device
        
    Returns:
        Dictionary with training results
    """
    print(f"\nTraining configuration: {config}")
    
    # Create model based on mode
    model, prior = create_model(
        input_dim=24,
        latent_dim=latent_dim,
        hidden_dims=config,
        mode=args.mode,
        num_clusters=args.num_clusters
    )
    model = model.to(device)
    if prior is not None:
        prior = prior.to(device)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_params']}")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    
    # Training tracking
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        # Training phase based on mode
        if args.mode == 'ae':
            train_loss = train_epoch_ae(
                model, train_loader, nn.MSELoss(), optimizer, device,
                args.use_sparse_loss, args.mse_weight, args.l1_weight, args.sparse_weight
            )
        elif args.mode == 'vae':
            train_loss = train_epoch_vae(
                model, train_loader, optimizer, device,
                args.mse_weight, args.l1_weight, args.sparse_weight
            )
        elif args.mode == 'vade':
            train_loss = train_epoch_vade(
                model, prior, train_loader, optimizer, device,
                args.mse_weight, args.l1_weight, args.sparse_weight
            )
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        
        train_losses.append(train_loss)
        
        # Test phase
        test_loss = evaluate_model(
            model, test_loader, device, args.use_sparse_loss,
            args.mse_weight, args.l1_weight, args.sparse_weight
        )
        test_losses.append(test_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Return results
    return {
        'config': config,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': min(test_losses),
        'model_info': model_info,
        'model_state': model.state_dict() if args.save_models else None
    }


def visualize_results(results: List[Dict], latent_dim: int, output_dir: Path):
    """Create visualizations of hyperparameter search results."""
    
    # Create performance comparison plot
    plt.figure(figsize=(12, 8))
    
    # Sort by final test loss
    results_sorted = sorted(results, key=lambda x: x['final_test_loss'])
    
    # Plot test losses for top configurations
    top_n = min(5, len(results_sorted))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    
    for i, result in enumerate(results_sorted[:top_n]):
        config_str = '-'.join(map(str, result['config']))
        plt.plot(result['test_losses'], label=f'{config_str}', color=colors[i], linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'Hyperparameter Search Results - Latent Dim {latent_dim}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / f'hyperparam_search_latent{latent_dim}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create configuration performance scatter plot
    plt.figure(figsize=(10, 6))
    
    # Extract configuration metrics
    configs = [len(r['config']) for r in results]
    final_losses = [r['final_test_loss'] for r in results]
    param_counts = [r['model_info']['total_params'] for r in results]
    
    scatter = plt.scatter(param_counts, final_losses, c=configs, 
                         cmap='viridis', s=60, alpha=0.7)
    plt.colorbar(scatter, label='Number of Hidden Layers')
    
    plt.xlabel('Total Parameters')
    plt.ylabel('Final Test Loss')
    plt.title(f'Configuration Performance - Latent Dim {latent_dim}')
    plt.grid(True, alpha=0.3)
    
    # Annotate best configuration
    best_idx = np.argmin(final_losses)
    best_config = results[best_idx]['config']
    plt.annotate(f'Best: {best_config}', 
                xy=(param_counts[best_idx], final_losses[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    scatter_path = output_dir / f'config_performance_latent{latent_dim}.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path, scatter_path


def visualize_latent_space(model, train_loader, test_loader, device, output_dir: Path, latent_dim: int):
    """Generate 2D scatter plot of latent space if latent_dim == 2 using ALL samples."""
    
    if latent_dim != 2:
        return None
    
    print("Generating 2D latent space visualization using ALL samples...")
    
    model.eval()
    latent_vectors = []
    sample_labels = []  # To distinguish train vs test samples
    
    with torch.no_grad():
        # Process training samples
        for batch_data in train_loader:
            if device == 'cpu' and batch_data.device != device:
                batch_data = batch_data.to(device)
            
            _, latent = model(batch_data)
            latent_vectors.append(latent.cpu().numpy())
            sample_labels.extend(['train'] * len(batch_data))  # Mark as training samples
        
        # Process test samples
        for batch_data in test_loader:
            if device == 'cpu' and batch_data.device != device:
                batch_data = batch_data.to(device)
            
            _, latent = model(batch_data)
            latent_vectors.append(latent.cpu().numpy())
            sample_labels.extend(['test'] * len(batch_data))  # Mark as test samples
    
    # Combine all latent vectors
    all_latents = np.vstack(latent_vectors)
    
    # Create scatter plot with different colors for train vs test
    plt.figure(figsize=(12, 8))
    
    # Convert labels to numpy array for easier filtering
    sample_labels = np.array(sample_labels)
    train_mask = sample_labels == 'train'
    test_mask = sample_labels == 'test'
    
    # Plot training samples
    plt.scatter(all_latents[train_mask, 0], all_latents[train_mask, 1], 
               c='blue', alpha=0.6, s=30, label=f'Training ({np.sum(train_mask)} samples)')
    
    # Plot test samples
    plt.scatter(all_latents[test_mask, 0], all_latents[test_mask, 1], 
               c='red', alpha=0.8, s=40, marker='^', label=f'Test ({np.sum(test_mask)} samples)')
    
    plt.legend()
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(f'2D Latent Space Visualization - Best Configuration (All Samples)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, f'Train: {np.sum(train_mask)}, Test: {np.sum(test_mask)}\nRange X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\nRange Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    latent_path = output_dir / f'latent_space_2d_best_config.png'
    plt.savefig(latent_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return latent_path


def save_results(results: List[Dict], best_config: Dict, args, output_dir: Path):
    """Save comprehensive results to JSON file."""
    
    # Convert best configuration to JSON-serializable format
    best_config_json = {
        'config': best_config['config'],
        'final_test_loss': float(best_config['final_test_loss']),
        'best_test_loss': float(best_config['best_test_loss']),
        'final_train_loss': float(best_config['final_train_loss']),
        'model_info': best_config['model_info']
    }
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'total_configurations': len(results),
        'best_configuration': best_config_json,
        'all_results': [
            {
                'config': r['config'],
                'final_test_loss': float(r['final_test_loss']),
                'best_test_loss': float(r['best_test_loss']),
                'final_train_loss': float(r['final_train_loss']),
                'model_params': r['model_info']['total_params'],
                'num_layers': len(r['config'])
            }
            for r in results
        ]
    }
    
    results_path = output_dir / f'hyperparam_results_latent{args.latent_dim}.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return results_path


def main():
    """Main hyperparameter optimization function."""
    print("Hyperparameter Optimization for Autoencoder Hidden Dimensions")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    full_dataset = COPUSDataset(args.data_path, device=device)
    print(f"Dataset: {len(full_dataset)} samples, {full_dataset.n_features} features")
    
    # Create train/test split
    dataset_size = len(full_dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train/Test split: {train_size}/{test_size} samples")
    
    # Create dataloaders
    train_loader = create_copus_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = create_copus_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Generate configurations
    search_dims = [int(x.strip()) for x in args.hidden_dims_search.split(',')]
    configs = generate_hidden_configs(search_dims, args.max_layers)
    
    print(f"\nGenerated {len(configs)} configurations to test:")
    print(f"Search dimensions: {search_dims}")
    print(f"Max layers: {args.max_layers}")
    print(f"Fixed latent dimension: {args.latent_dim}")
    
    # Test configurations
    results = []
    best_test_loss_so_far = float('inf')
    best_config_so_far = None
    best_model_state_so_far = None
    best_config_index = 0
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1:3d}/{len(configs):3d}] Testing configuration: {config}")
        
        try:
            result = train_configuration(
                config, args.latent_dim, train_loader, test_loader, args, device
            )
            results.append(result)
            
            print(f"  ✓ Final test loss: {result['final_test_loss']:.4f}")
            
            # Check if this is a new best configuration
            if result['final_test_loss'] < best_test_loss_so_far:
                print(f"  🌟 NEW BEST CONFIGURATION FOUND!")
                print(f"     Previous best: {best_test_loss_so_far:.4f}")
                print(f"     New best: {result['final_test_loss']:.4f}")
                
                best_test_loss_so_far = result['final_test_loss']
                best_config_so_far = result['config']
                best_model_state_so_far = result['model_state']
                best_config_index = i + 1
                
                # Generate and save latent space visualization if applicable
                if args.latent_dim == 2:
                    print(f"     Generating latent space visualization...")
                    
                    # Recreate best model for visualization
                    best_model = create_dynamic_autoencoder(
                        input_dim=24,
                        latent_dim=args.latent_dim,
                        hidden_dims=result['config'],
                        dropout=0.1
                    ).to(device)
                    
                    if result['model_state']:
                        best_model.load_state_dict(result['model_state'])
                    
                    # Create unique filename for this best configuration
                    config_str = '-'.join(map(str, result['config']))
                    loss_value = result['final_test_loss']
                    latent_path = output_dir / f'latent_space_2d_best_config_{config_str}_loss{loss_value:.4f}.png'
                    
                    # Generate the visualization
                    generated_path = visualize_latent_space(best_model, train_loader, test_loader, device, output_dir, args.latent_dim)
                    if generated_path:
                        # Rename to include configuration info
                        generated_path.rename(latent_path)
                        print(f"     Latent space saved: {latent_path}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Find best configuration for final summary
    if results:
        best_result = min(results, key=lambda x: x['final_test_loss'])
        
        print(f"\n{'='*70}")
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print(f"{'='*70}")
        print(f"Best configuration: {best_result['config']}")
        print(f"Best final test loss: {best_result['final_test_loss']:.4f}")
        print(f"Best model parameters: {best_result['model_info']['total_params']}")
        print(f"Best found at configuration #{best_config_index}")
        
        # Save final best model if requested
        if args.save_models and best_model_state_so_far:
            model_path = output_dir / f'best_model_latent{args.latent_dim}.pth'
            torch.save(best_model_state_so_far, model_path)
            print(f"Best model saved: {model_path}")
        
        # Create final visualizations
        print("\nGenerating final visualizations...")
        plot_path, scatter_path = visualize_results(results, args.latent_dim, output_dir)
        print(f"Performance plot: {plot_path}")
        print(f"Configuration scatter: {scatter_path}")
        
        # Generate final latent space visualization if applicable
        if args.latent_dim == 2 and best_config_so_far:
            print(f"Generating final latent space visualization...")
            
            # Recreate final best model for visualization
            final_best_model = create_dynamic_autoencoder(
                input_dim=24,
                latent_dim=args.latent_dim,
                hidden_dims=best_config_so_far,
                dropout=0.1
            ).to(device)
            
            if best_model_state_so_far:
                final_best_model.load_state_dict(best_model_state_so_far)
            
            final_latent_path = output_dir / f'latent_space_2d_final_best_config.png'
            final_generated_path = visualize_latent_space(final_best_model, train_loader, test_loader, device, output_dir, args.latent_dim)
            if final_generated_path:
                final_generated_path.rename(final_latent_path)
                print(f"Final latent space saved: {final_latent_path}")
        
        # Save comprehensive results
        results_path = save_results(results, best_result, args, output_dir)
        print(f"Results saved: {results_path}")
        
    else:
        print("No successful configurations found!")
    
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
