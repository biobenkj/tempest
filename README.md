# Tempest

**Modular sequence annotation using length-constrained CRFs with PWM priors and ensemble modeling.**

Tempest is inspired by Tranquillyzer, designed for annotating biological sequences (e.g., nanopore reads) with structured elements like adapters, UMIs, barcodes, and inserts using deep learning.

## Features

- **CNN-BiLSTM-semi-Markov CRF Architecture**: Captures local motifs, long-range dependencies, and leverages known segment lengths/boundaries
- **Length-Constrained CRF**: Semi-Markov approximation for segment-length priors
- **PWM-Based Priors**: Incorporates position weight matrices (e.g., for ACC detection)
- **Ensemble Modeling**: Bayesian Model Averaging for robust predictions
- **Visualization**: Colored sequence annotations (inspired by Tranquillyzer - https://github.com/huishenlab/tranquillyzer)
- **Configuration**: YAML-based configs for reproducible experiments

## Installation

```bash
# Clone or copy the tempest directory
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Your Experiment

Edit `example_config.yaml`:

```yaml
model:
  embedding_dim: 128
  lstm_units: 128
  length_constraints:
    constraints:
      UMI: [8, 8]
      ACC: [6, 6]
      BARCODE: [16, 16]

simulation:
  sequence_order: ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
  num_sequences: 10000

training:
  epochs: 20
  learning_rate: 0.001

ensemble:
  method: 'bma'
  num_models: 5
```

### 2. Simulate Training Data

```python
from tempest.data import SequenceSimulator
from tempest.utils import load_config

config = load_config('my_config.yaml')
simulator = SequenceSimulator(config.simulation)
train_data, val_data = simulator.generate()
```

### 3. Train a Model

```python
from tempest.training import ModelTrainer

trainer = ModelTrainer(config)
model = trainer.train(train_data, val_data)
```

### 4. Build an Ensemble

```python
from tempest.training import EnsembleTrainer

ensemble = EnsembleTrainer(config.ensemble)
ensemble.train(train_data, val_data)
```

### 5. Annotate Real Data

```python
from tempest.inference import SequenceAnnotator

annotator = SequenceAnnotator(model=ensemble)
annotations = annotator.annotate_fastq('reads.fastq')
```

### 6. Visualize Results

```python
from tempest.visualization import plot_annotations

plot_annotations(annotations, output='results.pdf')
```

## Architecture

Tempest follows a clean, modular architecture:

```
tempest/
├── core/          # Model architecture (CNN-BiLSTM-CRF)
├── data/          # Data simulation & preprocessing
├── training/      # Training & ensemble methods
├── inference/     # Annotation pipeline
├── visualization/ # Plotting functions
└── utils/         # Configuration & I/O
```

See `TEMPEST_ARCHITECTURE.md` for detailed design documentation.

## Model Architecture

```
Input → Embedding → CNN → BiLSTM → Length-CRF → Output
                                      ↑
                                   PWM Prior
```

The **Length-Constrained CRF** extends standard CRFs with segment-length regularization:

```
L = L_CRF + λ·Ω(y)

where Ω penalizes segments outside [L_min, L_max]
```

This approximates a semi-Markov CRF while maintaining O(T) inference complexity.

## Configuration System

All experiments are configured via YAML files with hierarchical structure:

- `model`: Architecture parameters (embedding, CNN, LSTM, CRF)
- `simulation`: Data generation (sequence structure, error rates)
- `training`: Optimization (learning rate, epochs, callbacks)
- `ensemble`: BMA parameters (number of models, diversity)
- `inference`: Post-processing (barcode correction, deduplication)
- `pwm`: Position weight matrix settings

Load with:
```python
from tempest.utils import load_config
config = load_config('config.yaml')
```

## Proposed Development Status

**Phase 1: Core Infrastructure**
- Configuration system
- I/O utilities
- Project structure

**Phase 2: Model Architecture**
- PWM module
- Length-constrained CRF
- Model builder

**Phase 3: Data Pipeline**
- Sequence simulator
- Preprocessor
- Data generators

**Phase 4: Training**
- Single model trainer
- Ensemble/BMA trainer

**Phase 5: Inference & Visualization**
- Annotation pipeline
- Post-processing
- Visualization

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- NumPy, Pandas, BioPython
- tf2crf, tensorflow-addons
- Matplotlib (for visualization)

See `requirements.txt` for complete list.

## Citation

If you use Tempest in your research, please cite:

```
[Citation to be added]
```

## Contributing

Tempest is under active development. Contributions welcome!

## License

MIT license

## Contact

Ben Johnson <ben.johnson@vai.org>
