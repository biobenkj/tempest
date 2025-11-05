# Installation Guide for Tempest

## Quick Installation

### From PyPI (when published)

```bash
pip install tempest-bio
```

### From Source (Development)

1. Clone the repository:
```bash
git clone https://github.com/biobenkj/tempest.git
cd tempest
```

2. Install in development mode:
```bash
pip install -e .
```

3. Or install with all dependencies:
```bash
pip install -e .[dev,docs]
```

## Requirements

- Python >= 3.8
- TensorFlow >= 2.10.0, < 2.16.0
- NumPy >= 1.21.0, < 2.0.0
- Pandas >= 2.0.0
- BioPython >= 1.80
- tf2crf >= 0.1.33
- scikit-learn >= 1.2.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- PyYAML >= 6.0
- tqdm >= 4.60.0

## Optional Dependencies

### TensorFlow Addons (for Python < 3.11)

```bash
pip install tempest-bio[tf-addons]
```

Note: TensorFlow Addons is no longer actively maintained and is not compatible with Python 3.11+. The core functionality of Tempest does not strictly require it.

### Development Tools

```bash
pip install tempest-bio[dev]
```

This installs:
- pytest >= 7.0
- pytest-cov >= 4.0
- black >= 22.0
- flake8 >= 5.0
- mypy >= 0.990

### Documentation

```bash
pip install tempest-bio[docs]
```

This installs:
- sphinx >= 5.0
- sphinx-rtd-theme >= 1.0
- sphinx-autodoc-typehints >= 1.19

## Alternative: Using Conda/Mamba

If you prefer using conda:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate tempest

# Install tempest
pip install -e .
```

## Verification

After installation, verify it works:

```python
import tempest
print(tempest.__version__)
```

or

```bash
python -c "import tempest; print(tempest.__version__)"
```

## Troubleshooting

### GPU Support

For GPU support, ensure you have the appropriate CUDA drivers and cuDNN installed for your TensorFlow version. See the [TensorFlow GPU support guide](https://www.tensorflow.org/install/gpu) for details.

### TensorFlow Version Conflicts

If you encounter version conflicts with TensorFlow:

1. Create a fresh virtual environment
2. Install TensorFlow first: `pip install "tensorflow>=2.10.0,<2.16.0"`
3. Then install tempest: `pip install tempest-bio`

### ImportError for tf2crf

If you get an import error for tf2crf, install it directly:

```bash
pip install tf2crf
```

### Python Version Compatibility

- Python 3.8-3.10: Full support including optional tensorflow-addons
- Python 3.11+: Core functionality supported, but tensorflow-addons not available

## Uninstallation

```bash
pip uninstall tempest-bio
```
