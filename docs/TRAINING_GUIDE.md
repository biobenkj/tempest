# Tempest Training Guide

Complete guide to training sequence annotation models with Tempest.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Running Training](#running-training)
4. [Understanding Output](#understanding-output)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Advanced Topics](#advanced-topics)

## Prerequisites

### Software Requirements
```bash
# Install dependencies
pip install tensorflow numpy pandas pyyaml biopython

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Data Requirements

You need one of the following:

**Option 1: Use simulation (recommended for getting started)**
- Configuration file (YAML)
- Optional: ACC PWM file
- Optional: ACC priors file
- Optional: Barcode file

**Option 2: Real data (for production)**
- Training sequences (FASTQ/FASTA)
- Labels for each sequence
- Configuration file

This guide focuses on **Option 1: Simulation**.

## Configuration

### Basic Configuration Structure

```yaml
# train_config.yaml

model:
  # Architecture parameters
  embedding_dim: 128
  cnn_filters: [64, 128]
  lstm_units: 128
  num_labels: 6
  max_seq_len: 256

simulation:
  # Sequence structure
  sequence_order: ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
  num_sequences: 5000

training:
  # Training parameters
  learning_rate: 0.001
  epochs: 20
  checkpoint_dir: './checkpoints'

pwm:
  # ACC PWM
  pwm_file: "acc_pwm.txt"
```

### Configuration Parameters Explained

#### Model Section

```yaml
model:
  vocab_size: 5              # A, C, G, T, N
  embedding_dim: 128         # Size of base embeddings (32-256)
  
  # CNN for local patterns
  use_cnn: true
  cnn_filters: [64, 128]     # Filter sizes (try [32,64] or [128,256])
  cnn_kernels: [3, 5]        # Kernel sizes (3-7 work well)
  
  # BiLSTM for context
  use_bilstm: true
  lstm_units: 128            # LSTM size (64-256)
  lstm_layers: 2             # Number of BiLSTM layers (1-3)
  dropout: 0.3               # Dropout rate (0.2-0.5)
  
  # Output
  num_labels: 6              # Number of unique labels
  max_seq_len: 256           # Maximum sequence length
  batch_size: 32             # Batch size (8-64)
```

**Guidelines:**
- Larger `embedding_dim` → better representations, more parameters
- More `cnn_filters` → capture more patterns, slower training
- Larger `lstm_units` → more capacity, more memory
- Higher `dropout` → less overfitting, might underfit

#### Simulation Section

```yaml
simulation:
  # Define your sequence structure
  sequence_order: ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
  
  # Fixed sequences
  sequences:
    ADAPTER5: 'AGATCGGAAGAGC'    # Known adapter
    ADAPTER3: 'AGATCGGAAGAGC'    # Known adapter
    INSERT: 'random'              # Random sequence
  
  # Optional: ACC priors from real data
  acc_priors_file: "acc_priors.tsv"
  
  # Optional: Known barcodes
  barcode_file: "barcodes.txt"
  
  # Parameters
  umi_length: 8                   # UMI length
  num_sequences: 5000             # Number to generate
  insert_min_length: 50           # Min insert size
  insert_max_length: 150          # Max insert size
  error_rate: 0.02                # 2% error rate
  random_seed: 42                 # For reproducibility
```

**Guidelines:**
- Match `sequence_order` to your actual read structure
- Use real adapter sequences if known
- Start with 5000 sequences, increase to 10000+ for production
- Error rate: 0.01-0.03 for good quality, 0.05-0.10 for nanopore-like

#### Training Section

```yaml
training:
  # Optimizer
  learning_rate: 0.001        # Start here (0.0001-0.01)
  optimizer: 'adam'           # 'adam' or 'sgd'
  epochs: 20                  # Maximum epochs (10-50)
  
  # Data split
  train_split: 0.8            # 80% training
  
  # Callbacks
  early_stopping_patience: 5  # Stop if no improvement
  reduce_lr_patience: 3       # Reduce LR if plateauing
  checkpoint_dir: './checkpoints'
```

**Guidelines:**
- `learning_rate`: Too high → unstable, too low → slow convergence
- `early_stopping_patience`: 3-7 works well
- Monitor validation loss to detect overfitting

#### PWM Section

```yaml
pwm:
  pwm_file: "acc_pwm.txt"     # Path to PWM file
  use_pwm: true               # Use PWM for ACC generation
  pwm_threshold: 0.7          # Detection threshold (0.6-0.8)
```

## Running Training

### Basic Training

```bash
# Train with default config
python main.py --config train_config.yaml
```

### With PWM File

```bash
# Explicitly specify PWM
python main.py --config train_config.yaml --pwm acc_pwm.txt
```

### Custom Output Directory

```bash
# Save to specific location
python main.py --config train_config.yaml --output-dir ./experiment_001
```

### What You'll See

```
================================================================================
                        TEMPEST TRAINING PIPELINE
================================================================================

2024-01-15 10:30:00 - INFO - Found 1 GPU(s)
2024-01-15 10:30:00 - INFO - ✓ Configured GPU memory growth
2024-01-15 10:30:01 - INFO - Loading configuration from: train_config.yaml
2024-01-15 10:30:01 - INFO - Using PWM file: acc_pwm.txt
2024-01-15 10:30:01 - INFO - ✓ Configuration loaded

================================================================================
STEP 1: DATA PREPARATION
================================================================================
2024-01-15 10:30:01 - INFO - Generating 5000 simulated reads...
2024-01-15 10:30:02 - INFO -   Generated 1000/5000 reads
...
2024-01-15 10:30:05 - INFO - ✓ Generated 5000 reads
2024-01-15 10:30:05 - INFO -   Average read length: 142.3 bp
2024-01-15 10:30:05 - INFO - Split: 4000 train, 1000 validation
2024-01-15 10:30:06 - INFO - ✓ Data preparation complete

================================================================================
STEP 2: MODEL BUILDING
================================================================================
2024-01-15 10:30:06 - INFO - Building CNN-BiLSTM-CRF model...
...
2024-01-15 10:30:07 - INFO - ✓ Model built with 1,234,567 parameters

================================================================================
STEP 3: MODEL TRAINING
================================================================================
Epoch 1/20
125/125 [==============================] - 15s - loss: 0.8234 - accuracy: 0.7123 - val_loss: 0.6543 - val_accuracy: 0.7834
...
Epoch 10/20
125/125 [==============================] - 12s - loss: 0.2341 - accuracy: 0.9234 - val_loss: 0.2876 - val_accuracy: 0.9056

2024-01-15 10:35:00 - INFO - ✓ Training complete

================================================================================
STEP 4: MODEL EVALUATION
================================================================================
2024-01-15 10:35:01 - INFO - ✓ Validation Loss: 0.2876
2024-01-15 10:35:01 - INFO - ✓ Validation Accuracy: 0.9056

================================================================================
STEP 5: SAVING MODEL
================================================================================
2024-01-15 10:35:02 - INFO - ✓ Saved final model to: checkpoints/model_final.h5
2024-01-15 10:35:02 - INFO - ✓ Saved label mapping to: checkpoints/label_mapping.json
2024-01-15 10:35:02 - INFO - ✓ Saved configuration to: checkpoints/config.yaml

================================================================================
                          TRAINING COMPLETE
================================================================================

Checkpoints saved to: ./checkpoints
  - model_best.h5: Best model based on validation loss
  - model_final.h5: Final model after training
  - training_history.csv: Training metrics
  - label_mapping.json: Label to index mapping
  - config.yaml: Configuration used for training
```

## Understanding Output

### Output Files

```
checkpoints/
├── model_best.h5           # Best model (use this for inference)
├── model_final.h5          # Final model
├── training_history.csv    # Metrics per epoch
├── label_mapping.json      # Label encodings
└── config.yaml             # Config snapshot
```

### Training History

View training progress:

```bash
# View CSV
cat checkpoints/training_history.csv

# Plot with Python
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/training_history.csv')
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df['loss'], label='train')
plt.plot(df['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['accuracy'], label='train')
plt.plot(df['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print('Saved to training_curves.png')
"
```

### Label Mapping

```bash
cat checkpoints/label_mapping.json

# Output:
{
  "ACC": 0,
  "ADAPTER3": 1,
  "ADAPTER5": 2,
  "BARCODE": 3,
  "INSERT": 4,
  "UMI": 5
}
```

## Hyperparameter Tuning

### Common Issues and Solutions

#### Issue: Low Accuracy (<70%)

**Possible causes:**
1. Too few training examples
2. Too complex model (overfitting)
3. Too simple model (underfitting)
4. High error rate in simulation

**Solutions:**
```yaml
# Increase data
simulation:
  num_sequences: 10000  # Was 5000

# Reduce overfitting
model:
  dropout: 0.4          # Was 0.3
  lstm_layers: 1        # Was 2

# Reduce errors
simulation:
  error_rate: 0.01      # Was 0.02
```

#### Issue: Training Loss Decreases but Validation Loss Increases

**Diagnosis:** Overfitting

**Solutions:**
```yaml
# Add regularization
model:
  dropout: 0.5          # Increase dropout
  lstm_layers: 1        # Reduce capacity

# Get more data
simulation:
  num_sequences: 10000

# Early stopping
training:
  early_stopping_patience: 3  # Stop sooner
```

#### Issue: Both Losses High and Not Decreasing

**Diagnosis:** Model too simple or learning rate issues

**Solutions:**
```yaml
# Increase capacity
model:
  embedding_dim: 256    # Was 128
  lstm_units: 256       # Was 128
  cnn_filters: [128, 256]  # Was [64, 128]

# Adjust learning rate
training:
  learning_rate: 0.0001  # Try lower
  # or
  learning_rate: 0.01    # Try higher
```

### Hyperparameter Grid Search

Try these combinations:

**Small/Fast model:**
```yaml
model:
  embedding_dim: 64
  cnn_filters: [32, 64]
  lstm_units: 64
  lstm_layers: 1
```

**Medium model (recommended):**
```yaml
model:
  embedding_dim: 128
  cnn_filters: [64, 128]
  lstm_units: 128
  lstm_layers: 2
```

**Large/Accurate model:**
```yaml
model:
  embedding_dim: 256
  cnn_filters: [128, 256, 512]
  lstm_units: 256
  lstm_layers: 3
```

## Advanced Topics

### Using Real Data

If you have labeled real data instead of simulation:

```python
# Load your data
sequences = load_fastq('reads.fastq')
labels = load_labels('labels.tsv')

# Convert to arrays
X, y, label_to_idx = prepare_arrays(sequences, labels)

# Train directly (skip simulation)
model = build_model_from_config(config)
model.compile(...)
model.fit(X, y, ...)
```

### Mixed Precision Training

For faster training on modern GPUs:

```python
# Add to main.py
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Distributed Training

For multiple GPUs:

```python
# Add to main.py
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model_from_config(config)
    model.compile(...)
```

### Custom Loss Functions

For specialized tasks:

```python
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # Weight different labels differently
        return K.categorical_crossentropy(y_true, y_pred) * weights
    return loss

model.compile(loss=weighted_categorical_crossentropy(class_weights))
```

## Tips and Best Practices

1. **Start simple**: Use default config first
2. **Monitor validation**: Watch for overfitting
3. **Use GPU**: 3-5× faster than CPU
4. **Save experiments**: Use meaningful `checkpoint_dir` names
5. **Version control**: Save configs alongside code
6. **Iterate quickly**: Start with 1000 sequences for fast experiments
7. **Scale up gradually**: Increase data size once hyperparameters tuned
8. **Check labels**: Verify label_mapping.json makes sense
9. **Visualize**: Plot training curves to diagnose issues
10. **Document**: Note what works in your experiment log

## Troubleshooting

### GPU Issues

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If GPU not detected
export CUDA_VISIBLE_DEVICES=0
```

### Memory Issues

```yaml
# Reduce memory usage
model:
  max_seq_len: 128      # Was 256
  batch_size: 16        # Was 32
  lstm_units: 64        # Was 128
```

### Slow Training

1. Use GPU
2. Reduce `num_sequences` for experiments
3. Reduce `max_seq_len`
4. Reduce model size
5. Enable XLA compilation (future feature)

## Next Steps

After successful training:

1. **Evaluate**: Check per-label accuracy
2. **Visualize**: Plot predictions on real data
3. **Iterate**: Tune hyperparameters based on results
4. **Deploy**: Use model for inference on new data
5. **Ensemble**: Train multiple models for robustness

Coming in Phase 2B:
- Length-constrained CRF integration
- PWM priors in model
- Better evaluation metrics
- Inference pipeline

## Example Workflows

### Quick Experiment
```bash
# Fast iteration (5 minutes)
python main.py --config quick_config.yaml
```
```yaml
# quick_config.yaml
simulation:
  num_sequences: 1000
training:
  epochs: 10
model:
  max_seq_len: 128
  batch_size: 64
```

### Production Training
```bash
# High quality (30-60 minutes)
python main.py --config production_config.yaml
```
```yaml
# production_config.yaml
simulation:
  num_sequences: 20000
training:
  epochs: 50
  early_stopping_patience: 10
model:
  embedding_dim: 256
  lstm_units: 256
```

### Hyperparameter Search
```bash
# Train multiple configurations
for lr in 0.001 0.0001 0.01; do
  python main.py --config config.yaml --output-dir checkpoints_lr_${lr}
done
```

Happy training! 🚀
