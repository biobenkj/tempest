# TEMPEST Testing Suite

Complete testing toolkit for validating your TEMPEST installation and ensuring GPU-accelerated training works correctly.

## Quick Start

Run tests in this order:

```bash
# 1. Quick validation (30 seconds)
python quick_check.py

# 2. Comprehensive system test (2-5 minutes)
python test_tempest_system.py

# 3. End-to-end training test (5-10 minutes)
python test_end_to_end_training.py

# 4. Monitor GPU during training (optional)
python gpu_monitor.py
```

## Test Scripts

### 1. `quick_check.py` - Fast Essential Checks

**What it does:**
- Verifies TensorFlow, NumPy, and Pandas imports
- Detects GPU availability
- Tests basic model data flow
- Runs a single training step

**When to use:**
- First time setup
- After installing dependencies
- Quick sanity check before training

**Expected output:**
```
Checking imports...
✓ TensorFlow 2.15.1
✓ NumPy 1.26.4
✓ Pandas 2.3.3

Checking GPU availability...
✓ Found 1 GPU(s):
  - GPU 0: /physical_device:GPU:0
✓ GPU computation successful

Testing data flow...
✓ Model forward pass: input (10, 100) -> output (10, 100, 5)
✓ Training step completed

✓ System ready for TEMPEST training!
```

### 2. `test_tempest_system.py` - Comprehensive Validation

**What it does:**
Tests 9 critical components:

1. **GPU Detection** - Finds NVIDIA GPUs, checks CUDA
2. **TensorFlow Core** - Validates imports and basic operations
3. **GPU Operations** - Tests matrix multiplication and gradients on GPU
4. **Data Simulation** - Generates synthetic sequence data
5. **Model Creation** - Builds CNN-BiLSTM architecture
6. **Sample Training** - Runs mini training loop
7. **Inference Pipeline** - Tests prediction and batch processing
8. **Length Constraints** - Validates CRF operations
9. **PWM Functionality** - Tests position weight matrix scoring

**When to use:**
- New environment setup
- After system updates
- Troubleshooting issues
- Before important training runs

**Expected output:**
```
==================================================================
  TEST: GPU Detection and CUDA Availability
==================================================================

ℹ TensorFlow version: 2.15.1
ℹ Physical GPUs detected: 1
ℹ   GPU 0: /physical_device:GPU:0
✓ GPU detection
  Found 1 GPU(s)
✓ GPU memory growth configuration

...

==================================================================
  TEST SUMMARY
==================================================================

Total tests: 20
Passed: 20
Failed: 0
Pass rate: 100.0%

✓ All tests passed!
Your system is ready for TEMPEST training.
```

### 3. `test_end_to_end_training.py` - Full Training Pipeline

**What it does:**
- Generates realistic nanopore read data with segment structure
- Builds complete CNN-BiLSTM-CRF model
- Trains for 5 epochs
- Evaluates on test set
- Runs inference
- Reports per-segment accuracy

**When to use:**
- Validate complete workflow
- Test training on GPU
- Verify data-to-prediction pipeline
- Before using real data

**Expected output:**
```
==================================================================
  TEMPEST End-to-End Training Test
==================================================================

Checking GPU availability...
✓ Found 1 GPU(s)
  - /physical_device:GPU:0

------------------------------------------------------------------------
Step 1: Generating synthetic data
------------------------------------------------------------------------
Generating training data...
  Training: 500 sequences × 200 bp
Generating validation data...
  Validation: 100 sequences × 200 bp
Generating test data...
  Test: 100 sequences × 200 bp

------------------------------------------------------------------------
Step 2: Building model
------------------------------------------------------------------------
Building model architecture...
Compiling model...

Model parameters: 234,758

------------------------------------------------------------------------
Step 3: Training model
------------------------------------------------------------------------
Training on /GPU:0...
Running 5 epochs

Epoch 1/5
32/32 [==============================] - 5s 120ms/step - loss: 1.2345 - accuracy: 0.4567
...

------------------------------------------------------------------------
Step 4: Evaluating model
------------------------------------------------------------------------
Evaluating on test set...

Test Results:
  Loss:     0.8234
  Accuracy: 0.6789

------------------------------------------------------------------------
Step 5: Testing inference
------------------------------------------------------------------------
Running inference on sample sequences...

Inference Results:
  Input shape:  (5, 200)
  Output shape: (5, 200, 6)

Per-segment accuracy:
  Adapter1    : 78.23%
  UMI         : 82.45%
  CBC         : 80.12%
  Adapter2    : 79.56%
  PolyA       : 75.34%
  Insert      : 68.90%

==================================================================
  Summary
==================================================================

✓ End-to-end pipeline completed successfully!
  - Data generation: ✓
  - Model building: ✓
  - Training: ✓ (test accuracy: 67.89%)
  - Inference: ✓

✓ Model achieved reasonable accuracy (67.89%)
  System is ready for full training on real data!
```

### 4. `gpu_monitor.py` - Real-time GPU Monitoring

**What it does:**
- Shows GPU utilization in real-time
- Monitors memory usage
- Tracks temperature
- Displays power consumption
- Updates every 2 seconds

**When to use:**
- Run in separate terminal during training
- Monitor GPU performance
- Optimize batch sizes
- Debug memory issues

**Expected output:**
```
================================================================================
  GPU Monitor - 2025-11-07 10:30:45
================================================================================

GPU 0: NVIDIA RTX 4090
--------------------------------------------------------------------------------
  Utilization: ██████████████████████████████  95.2%
  Memory:      ████████████████░░░░░░░░░░░░░░  52.3% (12.5GB/24.0GB)
  Temperature: 72.0°C
  Power:       350.2W

Tips:
  - Utilization should be >80% during training for good GPU usage
  - If utilization is low, increase batch size
  - If memory is full, decrease batch size
  - Temperature should stay below 85°C

Press Ctrl+C to exit
```

## Usage Patterns

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick check
python quick_check.py

# 3. If quick check passes, run full validation
python test_tempest_system.py

# 4. If all tests pass, verify training pipeline
python test_end_to_end_training.py
```

### Before Important Training Runs

```bash
# Full validation
python test_tempest_system.py

# Terminal 1: Start training
python -m tempest.training.trainer --config config.yaml

# Terminal 2: Monitor GPU
python gpu_monitor.py
```

### Troubleshooting

```bash
# Check what's failing
python quick_check.py

# Get detailed diagnostics
python test_tempest_system.py

# Test GPU specifically
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Understanding Test Results

### GPU Detection

**✓ GPU detected:**
- Training will use GPU acceleration (10-50x faster)
- All features available

**✗ No GPU:**
- Training will run on CPU (slower)
- Still functional for small datasets
- Consider using GPU for production

### Training Performance

**Good indicators:**
- Test accuracy >60% after 5 epochs
- Loss decreasing consistently
- GPU utilization >80%
- Memory usage stable

**Warning signs:**
- Accuracy not improving
- Loss increasing (overfitting)
- GPU utilization <50% (increase batch size)
- Memory errors (decrease batch size)

## Common Issues and Solutions

### Issue: No GPU Detected

```
✗ GPU detection (No GPUs detected)
```

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation
3. Install TensorFlow GPU: `pip install tensorflow[and-cuda]`
4. Check compatibility: TensorFlow 2.15.1 needs CUDA 12.x

### Issue: CUDA Out of Memory

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. Reduce batch size in config
2. Enable memory growth (done automatically in tests)
3. Use gradient accumulation
4. Process sequences in smaller batches

### Issue: Slow Training

**Symptoms:** Training takes hours for small datasets

**Solutions:**
1. Verify GPU is being used: `python gpu_monitor.py`
2. Check GPU utilization is >80%
3. Increase batch size if utilization is low
4. Ensure data is loaded efficiently

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install -r requirements.txt
```

## Performance Benchmarks

### Expected Training Times

**With GPU (RTX 4090):**
- 1,000 sequences: ~30 seconds/epoch
- 10,000 sequences: ~2-5 minutes/epoch
- 100,000 sequences: ~15-30 minutes/epoch

**Without GPU (CPU):**
- 1,000 sequences: ~5-10 minutes/epoch
- 10,000 sequences: ~30-60 minutes/epoch
- 100,000 sequences: Not recommended

### Memory Requirements

**Sequence length 200bp:**
- Batch size 8: ~2GB GPU memory
- Batch size 16: ~4GB GPU memory
- Batch size 32: ~8GB GPU memory
- Batch size 64: ~16GB GPU memory

## Advanced Configuration

### Multi-GPU Training

```python
# In your training script
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)

model.fit(...)
```

### Mixed Precision Training

```python
# For faster training on modern GPUs
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=2

# Enable XLA compilation
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
```

## Integration with TEMPEST

After tests pass, use TEMPEST training:

```bash
# With simulator data
python -m tempest.training.trainer \
    --config config/config.yaml \
    --use_simulator \
    --num_samples 10000 \
    --epochs 50

# With real data
python -m tempest.training.trainer \
    --config config/config.yaml \
    --sequences data/sequences.fasta \
    --labels data/labels.csv \
    --epochs 100

# With length constraints
python -m tempest.training.trainer \
    --config config/config.yaml \
    --use_length_constraints \
    --min_lengths 10,5,10,10,8,30 \
    --max_lengths 30,15,20,30,40,150
```

## Getting Help

If tests consistently fail:

1. **Check Installation:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   nvidia-smi
   ```

2. **Review Documentation:**
   - See `TESTING_GUIDE.md` for detailed troubleshooting
   - Check TensorFlow GPU guide: https://tensorflow.org/install/gpu

3. **Collect Diagnostics:**
   ```bash
   python quick_check.py > diagnostics.txt 2>&1
   nvidia-smi >> diagnostics.txt
   ```

4. **Test Components:**
   - GPU: `nvidia-smi`
   - CUDA: `nvcc --version`
   - TensorFlow: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`

## Summary

This testing suite provides comprehensive validation:

- **Quick Check:** 30 seconds, essential validation
- **System Test:** 2-5 minutes, complete validation  
- **End-to-End:** 5-10 minutes, full pipeline test
- **GPU Monitor:** Real-time monitoring during training

All scripts are designed to help you verify GPU setup, validate TensorFlow installation, test the data pipeline, and confirm training functionality before running production jobs.

**Recommended workflow:**
1. Run `quick_check.py` after installation
2. Run `test_tempest_system.py` before first training
3. Run `test_end_to_end_training.py` to validate workflow
4. Use `gpu_monitor.py` during actual training

These tests will save you time by catching issues early and ensuring your system is properly configured for TEMPEST training.
