# COPUS Autoencoder Clustering Project

This project applies autoencoder neural networks to improve clustering of Classroom Observation Protocol for Undergraduate STEM (COPUS) data. The goal is to reduce the dimensionality of COPUS observations while preserving meaningful patterns that can reveal different teaching and learning behaviors in STEM classrooms.

## What is COPUS?

COPUS (Classroom Observation Protocol for Undergraduate STEM) is a classroom observation protocol developed by the Carl Wieman Science Education Initiative at the University of British Columbia. It provides a systematic way to characterize how faculty and students spend their time in STEM classrooms.

### Key Features of COPUS:
- **Observer Training**: Requires only 1.5 hours of training for reliable characterization
- **Time-Based Coding**: Records what instructors and students are doing at 2-minute intervals
- **Comprehensive Coverage**: Captures both instructor and student behaviors
- **STEM Focus**: Specifically designed for undergraduate STEM education contexts

### COPUS Behavior Categories:
The protocol codes classroom activities into specific categories for both instructors and students, allowing researchers to identify different instructional approaches and their effectiveness.

## Project Overview

This repository implements an autoencoder-based approach to:
1. **Dimensionality Reduction**: Compress high-dimensional COPUS observation data into lower-dimensional latent representations
2. **Clustering Improvement**: Generate more meaningful clusters of classroom observations
3. **Pattern Discovery**: Identify underlying teaching and learning patterns
4. **Visualization**: Create interpretable 2D/3D representations of classroom dynamics

## Repository Structure

```
copus-nn/
├── dataset.py                    # COPUS dataset loading and preprocessing
├── models.py                     # Autoencoder model definitions
├── models.py                     # Autoencoder model definitions (includes dynamic architecture)
├── train_autoencoder.py          # Basic autoencoder training script
├── train_multiple_latent_dims.py # Multi-dimensional analysis pipeline
├── hyperparam_optimizer.py       # Hyperparameter optimization for hidden layers
├── output/                       # Training outputs and visualizations
│   ├── hyperparameters/         # Hyperparameter optimization results
│   └── *.png, *.json            # Generated plots and results
├── readme_files/                # Documentation visualizations
│   └── multiple_latent_dims_loss.png  # Latent dimension comparison plot
└── data/
    └── stains.csv                # COPUS dataset (24 behavior dimensions)
```

## Key Scripts and Their Purpose

### 1. `train_multiple_latent_dims.py`
This script trains autoencoders with different latent dimensions (2, 4, 8, 16, 32) to find the optimal dimensionality for representing COPUS data.

**Generated Visualization: `multiple_latent_dims_loss.png`**
- **Purpose**: Compares reconstruction loss across different latent dimensions
- **X-axis**: Training epochs
- **Y-axis**: Logarithmic scale of MSE loss (with small epsilon to avoid log(0))
- **Lines**: Each colored line represents a different latent dimension
- **Color Coding**: Uses viridis colormap from blue (low dimensions) to yellow (high dimensions)
- **Summary Box**: Shows final loss values for each dimension, with ⭐ marking the best performer
- **Interpretation**: Lower loss indicates better reconstruction quality. The plot helps identify the "elbow" point where additional dimensions provide diminishing returns.

### 2. `hyperparam_optimizer.py`
Systematically searches for optimal hidden layer configurations given a fixed latent dimension, tracking the best configurations and generating visualizations.

### 3. `train_autoencoder.py`
Basic autoencoder training script for single model training with customizable architecture.

## Refactor Summary (2025-11)

- Consolidated shared helpers into `utils.py` (losses, device/output setup, training loops, evaluation, plotting, and `create_model` factory).
- Merged the dynamic autoencoder into `models.py` and removed the legacy `dynamic_models.py` file.
- Updated `train_autoencoder.py` and `hyperparam_optimizer.py` to import helpers from `utils.py` and use unified `create_model`.
- Added unit tests under `./tests/test_models_and_utils.py` using `unittest` covering model forwards, losses, training loops, factory variants, and evaluation.

### Running Tests

Use Python’s standard library test runner:

```bash
python -m tests.test_models_and_utils
```

### Key APIs

- `utils.create_model(input_dim, latent_dim, hidden_dims, mode, num_clusters=None, use_dynamic=False)`
  - Returns `(model, prior)` where `prior` is `None` unless `mode == 'vade'`.
- Training loops:
  - `train_epoch_ae(model, dataloader, criterion, optimizer, device, use_sparse_loss=True, ...)`
  - `train_epoch_vae(model, dataloader, optimizer, device, ...)`
  - `train_epoch_vade(model, prior, dataloader, optimizer, device, ...)`
- Evaluation:
  - `evaluate_model(model, dataloader, device, use_sparse_loss=True, ...) -> float`

## Autoencoder Architecture

The project uses sparse-aware autoencoders specifically designed for COPUS data characteristics:

- **Input Dimension**: 24 (COPUS behavior codes)
- **Hidden Layers**: Configurable (optimized via hyperparameter search)
- **Latent Dimension**: Variable (2-32 dimensions, optimized per use case)
- **Activation**: ReLU with dropout for regularization
- **Loss Function**: Custom sparse-aware loss combining MSE and L1 loss
- **Optimizer**: AdamW with cosine annealing learning rate schedule

## Key Features

### Sparse-Aware Loss Function
Handles the sparse nature of COPUS data (many zero entries) by:
- Weighting reconstruction of non-zero elements more heavily (70% weight)
- Combining MSE and L1 loss for robust reconstruction
- Emphasizing behavioral patterns over zero-padding

### Dynamic Model Architecture
Supports flexible hidden layer configurations:
- Single to multiple hidden layers
- Decreasing dimension patterns (bottleneck architecture)
- Automatic parameter counting and model info generation

### Comprehensive Visualization
- **Training Progress**: Loss curves across epochs
- **Latent Space**: 2D scatter plots for dimensionality-reduced data
- **Configuration Performance**: Scatter plots of model parameters vs. loss
- **Hyperparameter Search**: Systematic comparison of architectures

## Usage Examples

### Multi-Dimensional Analysis
```bash
# Train autoencoders with different latent dimensions
python train_multiple_latent_dims.py --latent_dims 2 4 8 16 32 --epochs 50

# Custom dimensions and training
python train_multiple_latent_dims.py --latent_dims 3 6 12 24 --hidden_dim 128 --epochs 100
```

### Hyperparameter Optimization
```bash
# Search for optimal hidden layer configurations
python hyperparam_optimizer.py --latent_dim 2 --hidden_dims_search "8,16,32,64" --max_layers 3 --epochs 20

# Custom search space
python hyperparam_optimizer.py --latent_dim 8 --hidden_dims_search "16,32,64,128" --max_layers 4 --epochs 30
```

### Basic Training
```bash
# Train single autoencoder
python train_autoencoder.py --latent_dim 16 --hidden_dim 64 --epochs 100
```

## Output Interpretation

### Loss Curves
- **Training Loss**: Model's reconstruction error on training data
- **Test Loss**: Generalization performance on unseen data
- **Log Scale**: Helps visualize differences across orders of magnitude

### Latent Space Visualizations
For 2D latent dimensions, the autoencoder generates scatter plots showing:
- **Training Samples**: Blue circles representing classroom observations
- **Test Samples**: Red triangles for validation data
- **Clustering**: Natural groupings of similar teaching behaviors
- **Pattern Distribution**: Spread and density of different instructional approaches

### Hyperparameter Results
JSON files contain comprehensive results including:
- Best configuration found and its performance
- All tested configurations with their losses
- Model parameter counts and architecture details
- Timestamp and training parameters

## Educational Impact

This project supports STEM education research by:
- **Quantifying Teaching Practices**: Converting qualitative observations to analyzable data
- **Identifying Patterns**: Discovering common instructional approaches
- **Enabling Comparison**: Providing objective metrics for teaching method evaluation
- **Supporting Improvement**: Helping educators understand and refine their classroom practices

## Technical Requirements

- Python 3.11+
- PyTorch for neural network implementation
- NumPy for numerical computations
- Matplotlib/Seaborn for visualizations
- Pandas for data manipulation

## Dataset Information

The COPUS dataset (`data/stains.csv`) contains 24-dimensional feature vectors representing classroom behavior codes. Each dimension corresponds to specific instructor or student behaviors, enabling comprehensive characterization of classroom dynamics.

## References

Based on the COPUS protocol developed by:
- Smith, M. K., Jones, F. H. M., Gilbert, S. L., & Wieman, C. E. (2013). The Classroom Observation Protocol for Undergraduate STEM (COPUS): A New Instrument to Characterize University STEM Classroom Practices. *CBE—Life Sciences Education*, 12(4), 618-627. https://doi.org/10.1187/cbe.13-08-0154

For more information about COPUS: http://www.cwsei.ubc.ca/resources/tools/copus.html
