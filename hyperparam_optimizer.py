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
from dynamic_models import create_dynamic_autoencoder

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup computing device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return device


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


def sparse_aware_loss(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Custom loss function for sparse data."""
    mse_loss = nn.MSELoss()(reconstructed, original)
    non_zero_mask = (original != 0).float()
    
    if non_zero_mask.sum() > 0:
        weighted_mse = (non_zero_mask * (reconstructed - original) ** 2).sum() / non_zero_mask.sum()
        return 0.7 * weighted_mse + 0.3 * mse_loss
    else:
        return mse_loss


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
    
    # Create model
    model = create_dynamic_autoencoder(
        input_dim=24,
        latent_dim=latent_dim,
        hidden_dims=config,
        dropout=0.1
    ).to(device)
    
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
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_data in train_loader:
            if device == 'cpu' and batch_data.device != device:
                batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(batch_data)
            loss = sparse_aware_loss(reconstructed, batch_data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        train_losses.append(train_loss)
        
        # Test phase
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                if device == 'cpu' and batch_data.device != device:
                    batch_data = batch_data.to(device)
                
                reconstructed, _ = model(batch_data)
                loss = sparse_aware_loss(reconstructed, batch_data)
                
                test_loss += loss.item()
                test_batches += 1
        
        test_loss /= test_batches
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
    plt.text(0.02, 0.98, f'Total Samples: {len(all_latents)}\nTrain: {np.sum(train_mask)}, Test: {np.sum(test_mask)}\nRange X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\nRange Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]',
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