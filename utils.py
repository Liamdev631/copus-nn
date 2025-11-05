#!/usr/bin/env python3
"""
Utility functions for autoencoder training and visualization.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


def plot_latent_encodings(model, dataloader, device: str, latent_dim: int, 
                         current_epoch: int, output_dir: Path, 
                         title_suffix: str = "", interactive_3d: bool = False) -> Optional[Path]:
    """
    Generate 2D or 3D scatter plot of latent space encodings if latent_dim == 2 or 3.
    
    Args:
        model: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Computing device ('cuda' or 'cpu')
        latent_dim: Dimension of the latent space
        current_epoch: Current training epoch (for filename)
        output_dir: Directory to save the plot
        title_suffix: Suffix for plot title
        interactive_3d: If True and latent_dim == 3, shows interactive 3D plot in popup window
        
    Returns:
        Path to saved plot file, or None if latent_dim != 2 or 3
    """
    if latent_dim not in [2, 3]:
        return None
    
    try:
        model.eval()
        latent_vectors = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                if batch_data.device != device:
                    batch_data = batch_data.to(device)
                
                # Get latent encoding (handle both AE and VAE models)
                if hasattr(model, 'encode'):
                    # Standard autoencoder or VAE
                    encoder_output = model.encode(batch_data)
                    if isinstance(encoder_output, tuple):
                        # VAE encode returns (mu, logvar) - use mu as latent representation
                        latent = encoder_output[0]
                    else:
                        # Standard autoencoder
                        latent = encoder_output
                else:
                    # Fallback: use forward pass
                    outputs = model(batch_data)
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        # VAE forward returns (recon, z, mu, logvar) - use z
                        latent = outputs[1]
                    else:
                        latent = outputs
                
                latent_vectors.append(latent.cpu().numpy())
        
        # Combine all latent vectors
        all_latents = np.vstack(latent_vectors)
        
        # Create appropriate plot based on latent dimension
        if latent_dim == 2:
            # 2D scatter plot
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")
            
            plt.scatter(all_latents[:, 0], all_latents[:, 1], 
                       c='blue', alpha=0.6, s=30, 
                       label=f'{len(all_latents)} Samples')
            
            plt.xlabel('Latent Dimension 1', fontsize=12)
            plt.ylabel('Latent Dimension 2', fontsize=12)
            
            plot_title = f'2D Latent Space Encoding - Epoch {current_epoch + 1}'
            if title_suffix:
                plot_title += f' ({title_suffix})'
            plt.title(plot_title, fontsize=14, fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\\n'
                          f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]')
            
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            plt.legend()
            plt.tight_layout()
            
        elif latent_dim == 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(all_latents[:, 0], all_latents[:, 1], all_latents[:, 2],
                               c='blue', alpha=0.6, s=30, 
                               label=f'{len(all_latents)} Samples')
            
            ax.set_xlabel('Latent Dimension 1', fontsize=12)
            ax.set_ylabel('Latent Dimension 2', fontsize=12)
            ax.set_zlabel('Latent Dimension 3', fontsize=12)
            
            plot_title = f'3D Latent Space Encoding - Epoch {current_epoch + 1}'
            if title_suffix:
                plot_title += f' ({title_suffix})'
            ax.set_title(plot_title, fontsize=14, fontweight='bold')
            
            # Add statistics
            stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\n'
                         f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]\n'
                         f'Range Z: [{all_latents[:, 2].min():.3f}, {all_latents[:, 2].max():.3f}]')
            
            # Position text in 3D space (adjust position as needed)
            ax.text2D(0.02, 0.98, stats_text,
                     transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=10)
            
            ax.legend()
            plt.tight_layout()
            
            # Show interactive plot if requested
            if interactive_3d:
                plt.show()
        
        # Save plot (overwrite the same file each time)
        latent_plot_path = output_dir / 'latent_encodings.png'
        plt.savefig(latent_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {latent_dim}D latent space plot to {latent_plot_path}")
        return latent_plot_path
        
    except Exception as e:
        print(f"Warning: Failed to generate latent encoding plot: {e}")
        return None


def create_combined_plot(epochs: np.ndarray, losses: np.ndarray, current_epoch: int,
                        output_dir: Path, latent_dim: int, dataloader=None, 
                        model=None, device: str = 'cpu', use_log_scale: bool = True,
                        title_suffix: str = "", interactive_3d: bool = False) -> Tuple[Path, Optional[Path]]:
    """
    Create a combined figure with training loss and latent space visualization.
    
    Args:
        epochs: Array of epoch numbers
        losses: Array of loss values
        current_epoch: Current training epoch
        output_dir: Directory to save plots
        latent_dim: Dimension of latent space
        dataloader: DataLoader for latent visualization (optional)
        model: Model for latent visualization (optional)
        device: Computing device
        use_log_scale: Whether to use log scale for loss plot
        title_suffix: Suffix for plot titles
        interactive_3d: If True and latent_dim == 3, shows interactive 3D plot in popup window
        
    Returns:
        Tuple of (loss_plot_path, latent_plot_path)
    """
    # Initialize latent_plot_path
    latent_plot_path = None
    
    # Check if this is the final epoch for 3D latent dimensions
    is_final_epoch = (current_epoch == len(epochs) - 1)
    
    # Special handling for 3D latent dimensions: only show visualization at final stage
    if latent_dim == 3 and not is_final_epoch:
        # For 3D latent dimensions during training, only create training loss plot
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(1, 1, 1)
        sns.set_style("whitegrid")
        
        # Apply log transformation if requested
        if use_log_scale:
            # Add small epsilon to avoid log(0)
            log_losses = np.log10(losses[:current_epoch+1] + 1e-10)
            ax1.plot(epochs[:current_epoch+1], log_losses, 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'Log₁₀(MSE Loss + ε)'
            current_loss_display = np.log10(losses[current_epoch] + 1e-10)
        else:
            ax1.plot(epochs[:current_epoch+1], losses[:current_epoch+1], 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'MSE Loss'
            current_loss_display = losses[current_epoch]
        
        plot_title = 'Training Loss Over Time'
        if title_suffix:
            plot_title += f' ({title_suffix})'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add current epoch info to loss plot
        if current_epoch >= 0:
            ax1.axvline(x=current_epoch, color='red', linestyle='--', alpha=0.5)
            ax1.text(current_epoch, current_loss_display, 
                    f'Epoch {current_epoch+1}\\nLoss: {losses[current_epoch]:.4f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        
        # Save training progress plot only (no latent visualization during training for 3D)
        combined_plot_path = output_dir / 'training_progress.png'
        plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return combined_plot_path, None
    
    # Only create 2D/3D latent space subplot if latent_dim is 2 or 3 (and for 3D, only at final stage)
    if (latent_dim in [2, 3] and model is not None and dataloader is not None):
        # Create combined figure with subplots for 2D/3D latent space
        fig = plt.figure(figsize=(20, 8))  # Wider figure for side-by-side plots
        
        # Left subplot: Training loss
        ax1 = plt.subplot(1, 2, 1)
        sns.set_style("whitegrid")
        
        # Apply log transformation if requested
        if use_log_scale:
            # Add small epsilon to avoid log(0)
            log_losses = np.log10(losses[:current_epoch+1] + 1e-10)
            ax1.plot(epochs[:current_epoch+1], log_losses, 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'Log₁₀(MSE Loss + ε)'
            current_loss_display = np.log10(losses[current_epoch] + 1e-10)
        else:
            ax1.plot(epochs[:current_epoch+1], losses[:current_epoch+1], 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'MSE Loss'
            current_loss_display = losses[current_epoch]
        
        plot_title = 'Training Loss Over Time'
        if title_suffix:
            plot_title += f' ({title_suffix})'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add current epoch info to loss plot
        if current_epoch >= 0:
            ax1.axvline(x=current_epoch, color='red', linestyle='--', alpha=0.5)
            ax1.text(current_epoch, current_loss_display, 
                    f'Epoch {current_epoch+1}\\nLoss: {losses[current_epoch]:.4f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, ha='center')
        
        # Right subplot: 2D/3D Latent space visualization
        latent_plot_path = None
        
        try:
            model.eval()
            latent_vectors = []
            
            with torch.no_grad():
                for batch_data in dataloader:
                    # Handle both tuple (data, labels) and tensor formats
                    if isinstance(batch_data, (list, tuple)):
                        batch_data = batch_data[0]  # Extract data from (data, labels) tuple
                    
                    if batch_data.device != device:
                        batch_data = batch_data.to(device)
                    
                    # Get latent encoding
                    if hasattr(model, 'encode'):
                        # Standard autoencoder or VAE
                        encoder_output = model.encode(batch_data)
                        if isinstance(encoder_output, tuple):
                            # VAE encode returns (mu, logvar) - use mu as latent representation
                            latent = encoder_output[0]
                        else:
                            # Standard autoencoder
                            latent = encoder_output
                    else:
                        # Fallback: use forward pass
                        outputs = model(batch_data)
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            # VAE forward returns (recon, z, mu, logvar) - use z
                            latent = outputs[1]
                        else:
                            latent = outputs
                    
                    latent_vectors.append(latent.cpu().numpy())
            
            # Combine all latent vectors
            all_latents = np.vstack(latent_vectors)
            
            if latent_dim == 2:
                # 2D scatter plot
                ax2 = plt.subplot(1, 2, 2)
                ax2.scatter(all_latents[:, 0], all_latents[:, 1], 
                           c='blue', alpha=0.6, s=30, 
                           label=f'{len(all_latents)} Samples')
                
                ax2.set_xlabel('Latent Dimension 1', fontsize=12)
                ax2.set_ylabel('Latent Dimension 2', fontsize=12)
                
                latent_title = f'2D Latent Space Encoding - Epoch {current_epoch + 1}'
                if title_suffix:
                    latent_title += f' ({title_suffix})'
                ax2.set_title(latent_title, fontsize=14, fontweight='bold')
                
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Add statistics
                stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\\n'
                             f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]')
                
                ax2.text(0.02, 0.98, stats_text,
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
                
            elif latent_dim == 3:
                # 3D scatter plot
                ax2 = plt.subplot(1, 2, 2, projection='3d')
                ax2.scatter(all_latents[:, 0], all_latents[:, 1], all_latents[:, 2],
                           c='blue', alpha=0.6, s=30, 
                           label=f'{len(all_latents)} Samples')
                
                ax2.set_xlabel('Latent Dimension 1', fontsize=12)
                ax2.set_ylabel('Latent Dimension 2', fontsize=12)
                ax2.set_zlabel('Latent Dimension 3', fontsize=12)
                
                latent_title = f'3D Latent Space Encoding - Epoch {current_epoch + 1}'
                if title_suffix:
                    latent_title += f' ({title_suffix})'
                ax2.set_title(latent_title, fontsize=14, fontweight='bold')
                
                # Add statistics
                stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\n'
                             f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]\n'
                             f'Range Z: [{all_latents[:, 2].min():.3f}, {all_latents[:, 2].max():.3f}]')
                
                ax2.text2D(0.02, 0.98, stats_text,
                          transform=ax2.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=10)
                
                ax2.legend()
                
                # Show interactive plot if requested
                if interactive_3d:
                    plt.show()
            
            # Save individual latent plot as well (overwrite the same file each time)
            latent_plot_path = output_dir / 'latent_encodings.png'
            # Create and save the individual plot
            try:
                if latent_dim == 2:
                    plt.figure(figsize=(10, 8))
                    sns.set_style("whitegrid")
                    
                    plt.scatter(all_latents[:, 0], all_latents[:, 1], 
                               c='blue', alpha=0.6, s=30, 
                               label=f'{len(all_latents)} Samples')
                    
                    plt.xlabel('Latent Dimension 1', fontsize=12)
                    plt.ylabel('Latent Dimension 2', fontsize=12)
                    
                    individual_title = f'2D Latent Space Encoding - Epoch {current_epoch + 1}'
                    if title_suffix:
                        individual_title += f' ({title_suffix})'
                    plt.title(individual_title, fontsize=14, fontweight='bold')
                    
                    plt.grid(True, alpha=0.3)
                    
                    # Add statistics
                    stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\n'
                                 f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]')
                    
                    plt.text(0.02, 0.98, stats_text,
                            transform=plt.gca().transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=10)
                    
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(latent_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                elif latent_dim == 3:
                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(all_latents[:, 0], all_latents[:, 1], all_latents[:, 2],
                                       c='blue', alpha=0.6, s=30, 
                                       label=f'{len(all_latents)} Samples')
                    
                    ax.set_xlabel('Latent Dimension 1', fontsize=12)
                    ax.set_ylabel('Latent Dimension 2', fontsize=12)
                    ax.set_zlabel('Latent Dimension 3', fontsize=12)
                    
                    individual_title = f'3D Latent Space Encoding - Epoch {current_epoch + 1}'
                    if title_suffix:
                        individual_title += f' ({title_suffix})'
                    ax.set_title(individual_title, fontsize=14, fontweight='bold')
                    
                    # Add statistics
                    stats_text = (f'Range X: [{all_latents[:, 0].min():.3f}, {all_latents[:, 0].max():.3f}]\n'
                                 f'Range Y: [{all_latents[:, 1].min():.3f}, {all_latents[:, 1].max():.3f}]\n'
                                 f'Range Z: [{all_latents[:, 2].min():.3f}, {all_latents[:, 2].max():.3f}]')
                    
                    ax.text2D(0.02, 0.98, stats_text,
                              transform=ax.transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                              fontsize=10)
                    
                    ax.legend()
                    plt.tight_layout()
                    
                    # Show interactive plot if requested
                    if interactive_3d:
                        plt.show()
                    
                    plt.savefig(latent_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"Warning: Failed to save individual latent plot: {e}")
                latent_plot_path = None
            
        except Exception as e:
            print(f"Warning: Failed to generate latent encoding subplot: {e}")
            # Create a placeholder subplot if latent visualization fails
            ax2 = plt.subplot(1, 2, 2)
            ax2.text(0.5, 0.5, 'Latent visualization\nnot available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Latent Space Encoding', fontsize=14, fontweight='bold')
    else:
        # Create single plot for non-2D latent spaces
        fig = plt.figure(figsize=(10, 8))  # Standard single plot size
        
        # Single plot: Training loss only
        ax1 = plt.subplot(1, 1, 1)
        sns.set_style("whitegrid")
        
        # Apply log transformation if requested
        if use_log_scale:
            # Add small epsilon to avoid log(0)
            log_losses = np.log10(losses[:current_epoch+1] + 1e-10)
            ax1.plot(epochs[:current_epoch+1], log_losses, 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'Log₁₀(MSE Loss + ε)'
            current_loss_display = np.log10(losses[current_epoch] + 1e-10)
        else:
            ax1.plot(epochs[:current_epoch+1], losses[:current_epoch+1], 
                    marker='o', markersize=4, linewidth=2, color='blue')
            ylabel = 'MSE Loss'
            current_loss_display = losses[current_epoch]
        
        plot_title = 'Training Loss Over Time'
        if title_suffix:
            plot_title += f' ({title_suffix})'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add current epoch info to loss plot
        if current_epoch >= 0:
            ax1.axvline(x=current_epoch, color='red', linestyle='--', alpha=0.5)
            ax1.text(current_epoch, current_loss_display, 
                    f'Epoch {current_epoch+1}\\nLoss: {losses[current_epoch]:.4f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = output_dir / 'training_progress.png'
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return combined_plot_path, latent_plot_path