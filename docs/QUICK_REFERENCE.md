# Tempest Quick Reference Card

## 30-Second Start

```bash
# 1. Test
python test_simulator.py

# 2. Train
python main.py --config train_config.yaml --pwm acc_pwm.txt

# 3. Check results
ls checkpoints/
```

## Core Commands

### Testing
```bash
python test_simulator.py                    # Validate PWM and simulation
```

### Training
```bash
python main.py --config CONFIG              # Basic training
python main.py --config CONFIG --pwm PWM    # With explicit PWM
python main.py --config CONFIG --output-dir DIR  # Custom output
```

### Checking Results
```bash
cat checkpoints/training_history.csv        # Training metrics
cat checkpoints/label_mapping.json          # Label indices
ls -lh checkpoints/model_best.h5            # Best model
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Training pipeline |
| `train_config.yaml` | Configuration |
| `acc_pwm.txt` | ACC PWM data |
| `test_simulator.py` | Validation |
| `checkpoints/model_best.h5` | Trained model |

## Configuration Essentials

```yaml
model:
  num_labels: 6              # Number of labels
  max_seq_len: 256           # Sequence length
  batch_size: 32             # Batch size

simulation:
  sequence_order: [...]      # Your structure
  num_sequences: 5000        # Dataset size

training:
  epochs: 20                 # Max epochs
  learning_rate: 0.001       # LR
  checkpoint_dir: ./checkpoints

pwm:
  pwm_file: "acc_pwm.txt"
```

## Python API

### Load Configuration
```python
from tempest.utils import load_config
config = load_config('train_config.yaml')
```

### Simulate Data
```python
from tempest.data import SequenceSimulator
sim = SequenceSimulator(config.simulation, pwm_file='acc_pwm.txt')
reads = sim.generate(num_sequences=1000)
```

### Build Model
```python
from tempest.core.models import build_model_from_config
model = build_model_from_config(config)
```

### Use PWM
```python
from tempest.core import PWMScorer
from tempest.utils import load_pwm

pwm = load_pwm('acc_pwm.txt')
scorer = PWMScorer(pwm, threshold=0.7)
score = scorer.score_sequence('ACCGGG')  # → 1.0
```

## Common Tasks

### Quick Test (1000 sequences, 10 epochs)
```yaml
# quick_config.yaml
simulation:
  num_sequences: 1000
training:
  epochs: 10
model:
  max_seq_len: 128
```
```bash
python main.py --config quick_config.yaml
```

### Production (20000 sequences, 50 epochs)
```yaml
# prod_config.yaml
simulation:
  num_sequences: 20000
training:
  epochs: 50
  early_stopping_patience: 10
model:
  embedding_dim: 256
  lstm_units: 256
```
```bash
python main.py --config prod_config.yaml
```

### Hyperparameter Search
```bash
for lr in 0.001 0.0001 0.01; do
  python main.py --config config.yaml --output-dir exp_lr_${lr}
done
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Low accuracy | Increase `num_sequences` to 10000+ |
| Out of memory | Reduce `batch_size` or `max_seq_len` |
| Training too slow | Use GPU or reduce `num_sequences` |
| ACC sequences random | Verify PWM file path |
| Validation loss increases | Add dropout, reduce model size |

## Documentation

- `TRAINING_GUIDE.md` - Complete training guide
- `PWM_GUIDE.md` - PWM theory and usage

## Key Parameters

### Model Size
```yaml
# Small (fast)
embedding_dim: 64
lstm_units: 64
cnn_filters: [32, 64]

# Medium (recommended)
embedding_dim: 128
lstm_units: 128
cnn_filters: [64, 128]

# Large (accurate)
embedding_dim: 256
lstm_units: 256
cnn_filters: [128, 256]
```

### Data Size
```yaml
# Test
num_sequences: 1000

# Development
num_sequences: 5000

# Production
num_sequences: 20000
```

### Error Rates
```yaml
# High quality
error_rate: 0.01

# Typical
error_rate: 0.02

# Nanopore-like
error_rate: 0.05
```

## Expected Performance

| Dataset Size | Training Time (GPU) | Val Accuracy |
|--------------|---------------------|--------------|
| 1,000 | ~1 min | 80-85% |
| 5,000 | ~3 min | 85-90% |
| 10,000 | ~5 min | 90-95% |
| 20,000 | ~10 min | 92-96% |

## Typical Workflow

```bash
# 1. Quick test
python main.py --config quick_config.yaml

# 2. Review results
cat checkpoints/training_history.csv

# 3. Adjust hyperparameters
vim train_config.yaml

# 4. Production training
python main.py --config train_config.yaml

# 5. Use best model
# model_best.h5 is ready for inference
```

## Dependencies

```bash
pip install tensorflow numpy pandas pyyaml biopython
```

## GPU Setup

```bash
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If GPU not detected
export CUDA_VISIBLE_DEVICES=0
```

## File Sizes

| File | Size |
|------|------|
| PWM | ~1 KB |
| Config | ~1 KB |
| Model | 1-5 MB |
| Training data (5000) | ~5 MB |

## Support

See comprehensive docs:
- **Getting Started**: TRAINING_GUIDE.md
- **PWM Theory**: PWM_GUIDE.md
