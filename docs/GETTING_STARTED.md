# Getting Started with Tempest

A step-by-step guide to setting up your first experiment with Tempest.

## Prerequisites

- Python 3.8+
- Basic familiarity with deep learning
- FASTQ files (for annotation) or parameters for simulation

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python examples.py
```

## Your First Experiment

### Step 1: Understand Your Sequence Structure

Before configuring Tempest, map out your sequence structure. For example:

```
5' ADAPTER → UMI → ACC → BARCODE → INSERT → ADAPTER 3'
   ^^^^^^^^   ^^^   ^^^   ^^^^^^^^   ^^^^^^   ^^^^^^^^
   Fixed(13)   8     6    Fixed(16)  Random   Fixed(13)
```

Key questions:
- What elements are in your sequences?
- Which have fixed lengths? (for length constraints)
- Which have known sequences? (adapters, ACC)
- Which are variable? (UMI, barcodes, insert)

### Step 2: Prepare Your Data Files

Gather these files:

1. **ACC PWM** (`acc_pwm.txt`) - Position weight matrix for ACC detection
   ```
   pos  base  prob
   1    A     0.944
   1    C     0.026
   ...
   ```

2. **ACC Priors** (`acc_priors.tsv`) - Real ACC sequence distributions
   ```
   model_name  sequence  frequency
   my_model    ACCGGG    0.45
   my_model    ACCGGC    0.35
   ...
   ```

3. **Barcodes** (`barcodes.txt`) - One per line
   ```
   ACGTACGTACGTACGT
   TGCATGCATGCATGCA
   ...
   ```

### Step 3: Create Your Configuration

Copy and modify `example_config.yaml`:

```yaml
# my_experiment.yaml

model:
  # Architecture
  embedding_dim: 128
  cnn_filters: [64, 128]
  lstm_units: 128
  
  # Length constraints - CRITICAL for good performance
  length_constraints:
    constraints:
      UMI: [8, 8]        # Exactly 8 bases
      ACC: [6, 6]        # Exactly 6 bases  
      BARCODE: [16, 16]  # Exactly 16 bases
    constraint_weight: 5.0
    ramp_epochs: 5       # Gradually increase penalty
  
  # Labels (must match your sequence structure)
  num_labels: 10  # ADAPTER5, UMI, ACC, BARCODE, INSERT, ADAPTER3, etc.

pwm:
  pwm_file: "my_acc_pwm.txt"
  use_pwm: true
  pwm_threshold: 0.7

simulation:
  # Define sequence structure
  sequence_order: ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
  
  # Fixed sequences
  sequences:
    ADAPTER5: 'AGATCGGAAGAGC'
    ADAPTER3: 'AGATCGGAAGAGC'
    INSERT: 'random'
  
  # Data files
  acc_priors_file: "my_acc_priors.tsv"
  barcode_file: "my_barcodes.txt"
  
  # Simulation parameters
  num_sequences: 10000
  error_rate: 0.05

training:
  learning_rate: 0.001
  epochs: 20
  early_stopping_patience: 5

ensemble:
  method: 'bma'
  num_models: 5
```

### Step 4: Test Your Configuration

```bash
python examples.py
```

This will:
- Load your configuration
- Validate all parameters
- Check file paths
- Display configuration summary

### Step 5: (Coming in Phase 2) Run Your Experiment

Once Phase 2 is complete, you'll be able to:

```python
from tempest import Tempest

# Initialize
experiment = Tempest.from_config('my_experiment.yaml')

# 1. Simulate training data
experiment.simulate_data()

# 2. Train model
experiment.train()

# 3. Build ensemble
experiment.train_ensemble()

# 4. Annotate reads
experiment.annotate('my_reads.fastq', output='annotations.json')

# 5. Visualize
experiment.visualize(output='plots.pdf')
```

## Understanding Length Constraints

Length constraints are **crucial** for good performance. They implement a semi-Markov approximation:

```python
# Without constraints:
#   UMI could be 5-15 bases (bad!)
#   ACC could be 4-8 bases (bad!)

# With constraints:
length_constraints:
  constraints:
    UMI: [8, 8]    # Must be exactly 8 bases
    ACC: [6, 6]    # Must be exactly 6 bases
```

The penalty function:
```
Ω = [(L_min - L)₊² + (L - L_max)₊²]
```

This strongly discourages predictions outside the valid range.

**Constraint weight ramping**: Start with weight=0, gradually increase to full weight over N epochs. This lets the model learn basic patterns before enforcing strict constraints.

## Understanding ACC PWM Priors

The PWM (Position Weight Matrix) provides position-specific base probabilities:

```
Position 1: A=94.4%, C=2.6%, G=0.7%, T=2.2%
Position 2: A=1.1%, C=83.4%, G=0.7%, T=14.8%
...
```

This guides the model toward valid ACC sequences during training and inference.

## Tips for Best Results

1. **Start Small**: Use 1000-5000 sequences for initial testing
2. **Validate Constraints**: Make sure your length constraints match reality
3. **Check ACC Priors**: Ensure they sum to 1.0 and match IUPAC codes
4. **Use Ensembles**: 5-10 models with BMA gives robust predictions
5. **Monitor Training**: Watch for constraint weight ramping in logs
6. **Visualize Results**: Always check a sample of predictions visually

## Common Configuration Patterns

### Pattern 1: Dual Index (i7 + i5)
```yaml
sequence_order: ['ADAPTER', 'UMI', 'INDEX_i7', 'ACC', 'INDEX_i5', 'INSERT', 'ADAPTER']
length_constraints:
  constraints:
    UMI: [8, 8]
    INDEX_i7: [8, 8]
    INDEX_i5: [8, 8]
    ACC: [6, 6]
```

### Pattern 2: Simple UMI + Insert
```yaml
sequence_order: ['ADAPTER5', 'UMI', 'INSERT', 'ADAPTER3']
length_constraints:
  constraints:
    UMI: [10, 10]
```

### Pattern 3: Complex Multi-Element
```yaml
sequence_order: ['TSO', 'UMI', 'ACC', 'BARCODE', 'POLY_T', 'INSERT', 'ADAPTER']
length_constraints:
  constraints:
    UMI: [8, 8]
    ACC: [6, 6]
    BARCODE: [16, 16]
    POLY_T: [8, 20]  # Variable length
```

## Troubleshooting

### Issue: Configuration won't load
- Check YAML syntax (indentation, colons)
- Verify file paths exist
- Run `python examples.py` to debug

### Issue: Length constraints not enforcing
- Increase `constraint_weight` (try 10.0)
- Decrease `ramp_epochs` (try 3)
- Check that labels match exactly

### Issue: ACC detection poor
- Verify PWM file format
- Lower `pwm_threshold` (try 0.6)
- Check ACC priors sum to 1.0

## Next Steps

1. ✅ **Phase 1 Complete**: Configuration system ready
2. ⏳ **Phase 2 Next**: Model architecture implementation
3. ⏳ **Phase 3**: Data simulation
4. ⏳ **Phase 4**: Training pipeline
5. ⏳ **Phase 5**: Inference and visualization

Wait for Phase 2 to be completed before attempting model training!

## Questions?

Review:
- `README.md` - Overview and features
- `TEMPEST_ARCHITECTURE.md` - Detailed design
- `WORKFLOW_DIAGRAM.md` - Visual pipeline
- `IMPLEMENTATION_CHECKLIST.md` - Development progress

Or ask about specific configuration parameters or design decisions!
