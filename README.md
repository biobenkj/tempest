# <img src="tempest_logo.png" alt="TEMPEST" height="100">

## **TEMPEST: Modular Sequence Annotation with Length-Constrained CRFs**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)

TEMPEST is an advanced deep learning framework for annotating biological sequences (particularly nanopore RNA-seq reads) with structured elements using a CNN-BiLSTM-CRF architecture enhanced with semi-Markov approximation for length constraints.

## Key Features

- **CNN-BiLSTM-CRF Architecture**: Captures local motifs and long-range dependencies
- **Length-Constrained CRF**: Semi-Markov approximation for segment-length priors
- **Vectorized Implementation**: Fully XLA-compatible with O(T) inference complexity
- **Constraint Weight Ramping**: Gradual introduction of length penalties over epochs
- **PWM-Based Scoring**: Position weight matrix support for motif detection (e.g., ACC sites)
- **Ensemble Modeling**: Bayesian Model Averaging for robust predictions
- **Hybrid Training**: Combined soft and hard constraint approaches
- **Rich Visualization**: Publication-quality sequence annotation plots

## Package Structure

```
tempest/
├── tempest/                    # Main package directory
│   ├── core/                   # Core model architectures
│   │   ├── models.py          # CNN-BiLSTM-CRF models
│   │   ├── length_crf.py      # Length-constrained CRF implementation
│   │   ├── constrained_viterbi.py  # Hard constraint decoding
│   │   ├── hybrid_decoder.py  # Hybrid soft+hard constraints
│   │   ├── pwm.py             # PWM scoring utilities
│   │   └── pwm_probabilistic.py  # Probabilistic PWM extensions
│   ├── data/                   # Data handling and simulation
│   │   ├── simulator.py       # Sequence simulation
│   │   └── invalid_generator.py  # Invalid read generation
│   ├── training/               # Training utilities
│   │   ├── trainer.py         # Single model training
│   │   ├── ensemble.py        # Ensemble training with BMA
│   │   └── hybrid_trainer.py  # Hybrid training framework
│   ├── inference/              # Inference and prediction
│   │   ├── inference_utils.py # Core inference functions
│   │   ├── combine.py         # Model combination utilities
│   │   └── visualize_predictions.py  # Prediction visualization
│   ├── compare/                # Model comparison framework
│   │   ├── evaluation_framework.py  # Evaluation infrastructure
│   │   └── evaluator.py      # Model comparison tools
│   ├── visualization/          # Visualization tools
│   │   ├── annotated_reads.py # Sequence annotation plots
│   │   └── tempest_visualizer.py  # Main visualization class
│   ├── utils/                  # Utilities
│   │   ├── config.py          # Configuration management
│   │   └── io.py              # I/O utilities
│   ├── cli.py                 # Command-line interface
│   └── main.py                # Main entry point
├── bin/                        # Executable scripts
│   └── tempest                # Main CLI executable
├── config/                     # Configuration files
│   └── config.yaml            # Default configuration
├── docs/                       # Documentation
│   ├── GETTING_STARTED.md
│   ├── TRAINING_GUIDE.md
│   ├── PWM_GUIDE.md
│   └── VISUALIZATION_GUIDE.md
├── examples/                   # Example scripts
│   ├── example_usage_simulator.py
│   ├── train_with_length_constraints.py
│   └── compare_constraint_approaches.py
├── tests/                      # Test suite
│   ├── preflight/             # System tests
│   └── midflight/             # Unit tests
├── whitelist/                  # Whitelist sequences
│   ├── acc_pwm.txt
│   ├── udi_i5.txt
│   └── udi_i7.txt
├── setup.py                    # Package setup
├── pyproject.toml             # Modern Python packaging
├── requirements.txt           # Dependencies
└── environment.yml            # Conda environment
```

## Installation

### Via pip (recommended)
```bash
# Clone the repository
git clone https://github.com/biobenkj/tempest.git
cd tempest

# Install in development mode
pip install -e .
```

### Via conda
```bash
# Create conda environment
conda env create -f environment.yml
conda activate tempest

# Install package
pip install -e .
```

### Dependencies
- Python >= 3.8
- TensorFlow == 2.15.1
- Keras == 2.15.0
- TensorFlow addons == 0.23.0
- NumPy >= 1.21
- pandas >= 1.3
- scikit-learn >= 0.24
- matplotlib >= 3.3 (for visualization)
- pyyaml >= 5.4
- tqdm >= 4.62

## Quick Start

### 1. Import and Configuration
```python
from tempest.utils import load_config
from tempest.core import build_model_from_config
from tempest.data import SequenceSimulator
from tempest.training import ModelTrainer

# Load configuration
config = load_config('config/config.yaml')
```

### 2. Build Model with Length Constraints
```python
from tempest.core import build_model_with_length_constraints
import sklearn.preprocessing as sp

# Define labels
labels = ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3', 'PAD']
label_binarizer = sp.LabelBinarizer()
label_binarizer.fit(labels)

# Build base model
base_model = build_model_from_config(config)

# Add length constraints
model = build_model_with_length_constraints(
    base_model=base_model,
    length_constraints={
        'UMI': (8, 8),      # Exactly 8bp
        'ACC': (6, 6),      # Exactly 6bp
        'BARCODE': (16, 16) # Exactly 16bp
    },
    constraint_weight=5.0,
    label_binarizer=label_binarizer,
    max_seq_len=config.model.max_seq_len
)
```

### 3. Simulate Training Data
```python
from tempest.data import SequenceSimulator

simulator = SequenceSimulator(
    sequence_order=['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3'],
    whitelist_dir='whitelist/',
    use_pwm=True
)

# Generate training data
X_train, y_train = simulator.generate_batch(1000)
```

### 4. Train Model
```python
from tempest.training import ModelTrainer

trainer = ModelTrainer(
    model=model,
    config=config.training
)

history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=20
)
```

### 5. Make Predictions
```python
from tempest.inference import predict_sequences

predictions = predict_sequences(model, X_test)
```

### 6. Visualize Results
```python
from tempest.visualization import TempestVisualizer

visualizer = TempestVisualizer(label_colors={
    'ADAPTER5': 'blue',
    'UMI': 'green',
    'ACC': 'red',
    'BARCODE': 'orange',
    'INSERT': 'gray',
    'ADAPTER3': 'purple'
})

visualizer.visualize_batch_predictions(
    sequences=X_test[:10],
    predictions=predictions[:10],
    save_path='results/annotations.pdf'
)
```

## Model Architectures

### Standard CNN-BiLSTM-CRF
```python
from tempest.core import build_cnn_bilstm_crf

model = build_cnn_bilstm_crf(
    vocab_size=5,
    embedding_dim=128,
    lstm_units=128,
    num_labels=10
)
```

### With Soft Length Constraints (Training-time)
```python
from tempest.core import LengthConstrainedCRF

crf_layer = LengthConstrainedCRF(
    units=num_labels,
    length_constraints=length_constraints,
    constraint_weight=5.0
)
```

### With Hard Constraints (Inference-time)
```python
from tempest.core import ConstrainedViterbiDecoder

decoder = ConstrainedViterbiDecoder(
    length_constraints=length_constraints,
    label_map=label_map
)
```

### Hybrid Approach (Soft + Hard)
```python
from tempest.core import HybridConstraintDecoder

hybrid_model = HybridConstraintDecoder(
    base_model=model,
    soft_weight=0.7,
    hard_weight=0.3
)
```

## Testing

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test suite
```bash
# Unit tests
python -m pytest tests/midflight/

# System tests
python -m pytest tests/preflight/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [PWM Configuration](docs/PWM_GUIDE.md)
- [Visualization Guide](docs/VISUALIZATION_GUIDE.md)
- [API Reference](docs/DOCUMENTATION_INDEX.md)

## Command-Line Interface

TEMPEST provides a comprehensive CLI for common tasks:

```bash
# Train a model
tempest train --config config/train_config.yaml

# Run inference
tempest predict --model model.h5 --input sequences.fasta

# Compare models
tempest compare --models model1.h5 model2.h5 --test-data test.npz

# Visualize results
tempest visualize --predictions predictions.json --output plots.pdf
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under a modified MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TEMPEST in your research, please cite:

```bibtex
@software{tempest2024,
  title={TEMPEST: Modular Sequence Annotation with Length-Constrained CRFs},
  author={Johnson, Ben},
  year={2024},
  url={https://github.com/biobenkj/tempest}
}
```

## Acknowledgments

Tempest was inspired by Tranquillyzer (https://github.com/huishenlab/tranquillyzer) and builds upon the rich ecosystem of deep learning tools for biological sequence analysis.

## Contact

- **Author**: Ben Johnson
- **Email**: ben.johnson@vai.org
- **Institution**: Van Andel Institute

## Links

- [GitHub Repository](https://github.com/biobenkj/tempest)
- [Documentation](https://github.com/biobenkj/tempest/wiki)
- [Issue Tracker](https://github.com/biobenkj/tempest/issues)
