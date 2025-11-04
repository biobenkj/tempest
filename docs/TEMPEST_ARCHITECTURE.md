# Tempest: Clean Architecture Design

## Overview
Tempest is inspired by Tranquillyzer, designed for sequence annotation using length-constrained CRFs with PWM-based priors and ensemble modeling.

## Project Structure

```
tempest/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── models.py           # CNN-BiLSTM-CRF architecture
│   ├── length_crf.py       # Length-constrained CRF layer
│   └── pwm.py              # PWM detection and scoring
├── data/
│   ├── __init__.py
│   ├── simulator.py        # Training data simulation
│   ├── preprocessor.py     # FASTQ/sequence preprocessing
│   └── generators.py       # TensorFlow data generators
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Single model training
│   └── ensemble.py         # Bayesian Model Averaging (BMA)
├── inference/
│   ├── __init__.py
│   ├── annotator.py        # Sequence annotation pipeline
│   └── postprocess.py      # Post-processing utilities
├── visualization/
│   ├── __init__.py
│   └── plots.py            # Visualization functions (retained from original)
└── utils/
    ├── __init__.py
    ├── config.py           # Configuration management
    └── io.py               # File I/O utilities
```

## Key Design Principles

### 1. **Splitting out modules**
- Each module has a single, well-defined responsibility
- Data simulation separate from training
- Training separate from inference
- Visualization decoupled from core logic

### 2. **Configuration-Driven**
- Use YAML/JSON configs for model architectures
- Separate configs for simulation, training, and inference
- Easy to version control and reproduce experiments

### 3. **Modular Components**
- PWM detection as sta
ndalone module
- Length-constrained CRF as reusable layer
- Ensemble logic separate from single-model training

### 4. **APIs**
Each major component exposes simple, intuitive APIs:

```python
# Data simulation
from tempest.data import SequenceSimulator
sim = SequenceSimulator(config)
train_data, val_data = sim.generate()

# Training
from tempest.training import ModelTrainer
trainer = ModelTrainer(config)
model = trainer.train(train_data, val_data)

# Ensemble
from tempest.training import EnsembleTrainer
ensemble = EnsembleTrainer(models=models, method='bma')
ensemble.fit(train_data)

# Inference
from tempest.inference import SequenceAnnotator
annotator = SequenceAnnotator(model=ensemble)
annotations = annotator.annotate_fastq('reads.fastq')

# Visualization
from tempest.visualization import plot_annotations
plot_annotations(annotations, output='results.pdf')
```

## Implementation Plan

### Phase 1: Core Infrastructure (Start Here)
1. ✓ Project structure and configuration system
2. Configuration schema and validation
3. File I/O utilities
4. Logging setup

### Phase 2: Model Architecture
1. Length-constrained CRF layer (clean version from uploaded file)
2. PWM scoring module
3. CNN-BiLSTM-CRF model builder
4. Model serialization/deserialization

### Phase 3: Data Pipeline
1. Sequence simulator with ACC priors
2. Data preprocessor for FASTQ files
3. TensorFlow data generators
4. Data augmentation utilities

### Phase 4: Training
1. Single model trainer with callbacks
2. Ensemble/BMA trainer
3. Training utilities (metrics, checkpointing)

### Phase 5: Inference & Visualization
1. Annotation pipeline
2. Post-processing (barcode correction, deduplication)
3. Visualization module (retain existing style)
4. Export utilities

### Phase 6: CLI & Documentation
1. Command-line interface
2. Usage examples
3. API documentation
