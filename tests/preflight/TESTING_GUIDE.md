# TEMPEST Testing Guide

## Overview

This guide provides test scripts to validate your TEMPEST installation and ensure all components are functioning correctly, especially GPU acceleration for training.

## Test Scripts

### 1. Quick Check (`quick_check.py`)

**Purpose:** Fast validation of essential components  
**Runtime:** ~30 seconds  
**Use when:** You want a quick sanity check

```bash
python quick_check.py
```

**What it tests:**
- TensorFlow, NumPy, and Pandas imports
- GPU detection and availability
- Basic data flow through a simple model
- Single training step execution

### 2. Comprehensive System Test (`test_tempest_system.py`)

**Purpose:** Thorough validation of all TEMPEST components  
**Runtime:** 2-5 minutes  
**Use when:** Setting up a new environment or debugging issues

```bash
python test_tempest_system.py
```

**What it tests:**

#### Core System Tests
1. **GPU Detection and CUDA Availability**
   - Detects NVIDIA GPUs
   - Verifies CUDA configuration
   - Sets memory growth to prevent allocation issues

2. **TensorFlow Core Imports**
   - Tests TensorFlow and Keras imports
   - Validates TensorFlow Addons availability
   - Verifies basic tensor operations

3. **CUDA GPU Operations**
   - Runs matrix multiplication on GPU
   - Tests gradient computation on GPU
   - Measures GPU performance

#### Data Pipeline Tests
4. **Data Generation and Simulation**
   - Creates synthetic sequence data
   - Validates data shapes and types
   - Tests one-hot encoding

#### Model Tests
5. **Model Creation and Architecture**
   - Builds CNN-BiLSTM model
   - Tests embedding layers
   - Validates model compilation

6. **Sample Training Loop**
   - Runs 3-epoch training on sample data
   - Tests forward and backward passes
   - Validates loss and accuracy metrics

7. **Inference Pipeline**
   - Tests prediction on new data
   - Validates output shapes and ranges
   - Tests batch processing

#### Advanced Features
8. **Length-Constrained CRF Operations**
   - Tests CRF potential computations
   - Validates length constraint masks
   - Tests gradient flow through CRF layer

9. **Position Weight Matrix (PWM) Operations**
   - Creates PWM for motif detection
   - Tests PWM scoring via convolution
   - Validates threshold-based matching

## Expected Output

### Successful Run
```
✓ GPU detection (Found 1 GPU(s))
✓ TensorFlow import
✓ Keras import
✓ Basic TensorFlow operations
✓ GPU matrix operations
...
Total tests: 20
Passed: 20
Failed: 0
Pass rate: 100.0%

✓ All tests passed!
Your system is ready for TEMPEST training.
```

### GPU Not Available
```
✗ GPU detection (No GPUs detected)
⚠ Training will run on CPU (will be slower)
```
- This is expected if you don't have an NVIDIA GPU
- Training will still work but will be slower
- Consider using a GPU for large-scale training

## Troubleshooting

### Common Issues

#### 1. No GPU Detected
**Symptoms:**
```
✗ GPU detection (No GPUs detected)
```

**Solutions:**
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA version matches TensorFlow requirements
- Ensure TensorFlow GPU version is installed
- Try: `pip install tensorflow[and-cuda]`

#### 2. CUDA Out of Memory
**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
- Enable memory growth (done automatically in tests)
- Reduce batch size in training
- Use gradient accumulation for larger effective batch sizes

#### 3. Import Errors
**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**
```bash
# Install requirements
pip install -r requirements.txt

# Or install TensorFlow separately
pip install tensorflow==2.15.1
```

#### 4. CRF Layer Issues
**Symptoms:**
```
✗ CRF gradient computation
```

**Solutions:**
- Ensure tensorflow-addons is installed: `pip install tensorflow-addons`
- Check TensorFlow version compatibility

## Running TEMPEST Training After Tests

Once tests pass, you can run TEMPEST training:

### Basic Training
```bash
# Using simulator data
python -m tempest.training.trainer \
    --config config/config.yaml \
    --use_simulator
```

### With GPU Specification
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python -m tempest.training.trainer \
    --config config/config.yaml
```

### With Custom Data
```bash
# Using your own data
python -m tempest.training.trainer \
    --config config/config.yaml \
    --sequences path/to/sequences.fasta \
    --labels path/to/labels.csv
```

## Performance Expectations

### GPU Training
- **1 epoch on 10K sequences:** ~2-5 minutes
- **Model inference (1K sequences):** ~5-10 seconds
- Expect 10-50x speedup vs CPU

### CPU Training
- **1 epoch on 10K sequences:** ~30-60 minutes  
- **Model inference (1K sequences):** ~30-60 seconds
- Suitable for testing and small datasets

## Test Data Generation

Both test scripts create synthetic data with these characteristics:

- **Vocabulary:** 5 tokens (A, C, G, T, N)
- **Sequence length:** 50-150 bases
- **Number of labels:** 5-10 classes
- **Batch size:** 8-32 sequences

This mimics real nanopore read annotation tasks.

## Continuous Testing

### During Development
```bash
# Quick check after code changes
python quick_check.py

# Full validation before commits
python test_tempest_system.py
```

### Before Production Runs
```bash
# Comprehensive test
python test_tempest_system.py

# Check GPU memory
nvidia-smi

# Monitor GPU during training
watch -n 1 nvidia-smi
```

## Advanced GPU Configuration

### Multi-GPU Setup
```python
import tensorflow as tf

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')

# Use specific GPUs
tf.config.set_visible_devices(gpus[0:2], 'GPU')

# Enable memory growth for all GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Mixed Precision Training
```python
# For faster training on modern GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## Getting Help

If tests fail consistently:

1. Check TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`
2. Verify GPU drivers: `nvidia-smi`
3. Review TensorFlow GPU guide: https://www.tensorflow.org/install/gpu
4. Check CUDA compatibility: https://www.tensorflow.org/install/source#gpu

## Automated Testing

### Add to CI/CD Pipeline
```yaml
# .github/workflows/test.yml
- name: Run TEMPEST tests
  run: |
    python quick_check.py
    python test_tempest_system.py
```

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python quick_check.py || exit 1
```

## Summary

These test scripts provide comprehensive validation of your TEMPEST installation:

- **Quick Check:** Essential validation in 30 seconds
- **Full System Test:** Complete validation in 2-5 minutes

Both scripts are designed to help you:
- Verify GPU setup
- Validate TensorFlow installation  
- Test data pipeline
- Confirm model training works
- Check inference functionality

Run these tests before any training run to ensure your system is configured correctly.
