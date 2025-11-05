# Hyperparameter Optimization for Autoencoder Hidden Dimensions

A comprehensive system for systematically optimizing hidden layer configurations in autoencoders while maintaining a fixed latent dimension size.

## Overview

This hyperparameter optimization system systematically tests various hidden layer configurations to find the optimal architecture for autoencoders given a fixed latent dimension. The implementation follows experimental coding rules and provides comprehensive logging, visualization, and reproducibility features.

## Features

### 1. Systematic Hyperparameter Search
- **Fixed Latent Dimension**: Maintains consistent latent space size while optimizing hidden layers
- **Configurable Search Space**: Define hidden dimensions and maximum layers to test
- **Architecture Generation**: Automatically generates all valid configurations with decreasing dimensions
- **Performance Evaluation**: Tests each configuration with train/test split and comprehensive metrics

### 2. Dynamic Model Architecture
- **Flexible Configuration**: Accepts arrays of dimension sizes as input parameters
- **Dynamic Building**: Constructs model architecture based on provided dimensions
- **Existing Functionality**: Maintains all existing features (layer normalization, dropout, Xavier initialization)
- **Modular Design**: Clean separation between architecture definition and training logic

### 3. Advanced Visualization
- **Performance Curves**: Plot test loss evolution for top configurations
- **Configuration Analysis**: Scatter plot showing relationship between model complexity and performance
- **2D Latent Space**: Automatic generation of 2D scatter plots when latent dimension equals 2
- **Clustering Visualization**: Color-coded samples showing potential patterns in latent space

### 4. Comprehensive Logging
- **Configuration Tracking**: Records all tested configurations and their performance
- **Progress Monitoring**: Real-time updates during optimization process
- **Results Storage**: JSON format with complete experiment metadata
- **Model Persistence**: Optional saving of best model weights
- **Reproducibility**: Fixed random seeds and complete parameter logging

## Usage

### Basic Usage

```bash
# Optimize hidden dimensions for latent dimension 3
python hyperparam_optimizer.py --latent_dim 3 --hidden_dims_search "16,32,64" --max_layers 3 --epochs 20

# Quick test with fewer configurations
python hyperparam_optimizer.py --latent_dim 2 --hidden_dims_search "8,16,32" --max_layers 2 --epochs 5
```

### Advanced Configuration

```bash
# Comprehensive optimization with model saving
python hyperparam_optimizer.py \
    --latent_dim 4 \
    --hidden_dims_search "16,32,64,128" \
    --max_layers 3 \
    --epochs 30 \
    --batch_size 64 \
    --initial_lr 0.001 \
    --eta_min 1e-6 \
    --save_models \
    --output_dir results
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--latent_dim` | int | 3 | Fixed latent dimension size |
| `--hidden_dims_search` | str | "16,32,64,128" | Comma-separated hidden dimensions to test |
| `--max_layers` | int | 3 | Maximum number of hidden layers |
| `--epochs` | int | 20 | Training epochs per configuration |
| `--batch_size` | int | 32 | Batch size for training |
| `--initial_lr` | float | 0.001 | Initial learning rate |
| `--eta_min` | float | 1e-6 | Minimum learning rate for scheduler |
| `--data_path` | str | "data/stains.csv" | Path to dataset |
| `--device` | str | "auto" | Computing device (auto/cuda/cpu) |
| `--output_dir` | str | "output" | Output directory for results |
| `--save_models` | flag | False | Save best model weights |

## Architecture

### Configuration Generation
The system generates configurations following these rules:
- Single layer: `[dim]` for each dimension in search space
- Two layers: `[dim1, dim2]` where `dim2 <= dim1`
- Three layers: `[dim1, dim2, dim3]` where `dim3 <= dim2 <= dim1`

### Training Process
1. **Data Splitting**: 80/20 train/test split with fixed random seed
2. **Model Creation**: Dynamic architecture based on configuration
3. **Training Loop**: AdamW optimizer with cosine annealing scheduler
4. **Loss Function**: Sparse-aware loss for handling missing data
5. **Evaluation**: Test loss tracking throughout training

### Visualization Pipeline
1. **Performance Plot**: Top 5 configurations' test loss curves
2. **Configuration Scatter**: Model parameters vs. final test loss
3. **2D Latent Space**: Scatter plot with density coloring (when latent_dim=2)

## Output Files

### Generated Files
- `hyperparam_search_latent{N}.png`: Performance comparison plot
- `config_performance_latent{N}.png`: Configuration analysis scatter plot
- `latent_space_2d_best_config.png`: 2D latent space visualization (when applicable)
- `hyperparam_results_latent{N}.json`: Comprehensive results and metadata
- `best_model_latent{N}.pth`: Best model weights (if --save_models)

### JSON Results Structure
```json
{
  "timestamp": "2025-11-05T14:27:57.504953",
  "args": { /* command line arguments */ },
  "total_configurations": 9,
  "best_configuration": {
    "config": [32, 16],
    "final_test_loss": 0.0667,
    "model_info": { /* architecture details */ }
  },
  "all_results": [
    {
      "config": [32, 16],
      "final_test_loss": 0.0667,
      "model_params": 2986,
      "num_layers": 2
    }
    // ... more results
  ]
}
```

## Best Practices

### 1. Search Space Design
- Start with smaller search spaces for faster iteration
- Use powers of 2 for hidden dimensions (8, 16, 32, 64, 128)
- Consider computational constraints when setting max_layers

### 2. Training Configuration
- Use sufficient epochs (20-50) for meaningful comparison
- Adjust batch size based on available memory
- Use learning rate scheduling for better convergence

### 3. Reproducibility
- Fixed random seed (42) ensures reproducible train/test splits
- Complete parameter logging in JSON output
- Timestamp and environment tracking

### 4. Performance Optimization
- GPU acceleration when available
- Efficient batch processing with custom dataloader
- Memory optimization for large configurations

## Integration with Existing Codebase

The hyperparameter optimizer integrates seamlessly with existing components:

- **Dataset Module**: Uses `COPUSDataset` with eager tensor loading
- **Dynamic Models**: Leverages `DynamicAutoencoder` for flexible architectures
- **Loss Functions**: Employs sparse-aware loss for COPUS data characteristics
- **Output Directory**: Follows experimental coding rules for file organization

## Example Results

### Sample Optimization Output
```
======================================================================
HYPERPARAMETER OPTIMIZATION COMPLETED
======================================================================
Best configuration: [64, 64]
Best final test loss: 0.0597
Best model parameters: 12491
Best model saved: output/best_model_latent3.pth

Generating visualizations...
Performance plot: output/hyperparam_search_latent3.png
Configuration scatter: output/config_performance_latent3.png
Results saved: output/hyperparam_results_latent3.json
```

### Key Insights
- **Architecture Efficiency**: Two-layer configurations often outperform single layers
- **Parameter Efficiency**: Optimal balance between model complexity and performance
- **Convergence**: Consistent training behavior across different configurations
- **Generalization**: Test loss provides reliable model selection criterion

## Future Enhancements

### Planned Features
- **Multi-objective Optimization**: Balance performance and model complexity
- **Early Stopping**: Automatic termination based on validation metrics
- **Cross-validation**: K-fold validation for more robust evaluation
- **Bayesian Optimization**: Intelligent search space exploration
- **Distributed Training**: Parallel configuration evaluation

### Research Directions
- **Architecture Search**: Automated discovery of optimal layer arrangements
- **Regularization Study**: Systematic dropout and normalization analysis
- **Transfer Learning**: Pre-trained models for faster convergence
- **Ensemble Methods**: Combining multiple optimal configurations