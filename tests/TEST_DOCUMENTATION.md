# Tempest Test Suite Documentation

## Overview

This document describes the comprehensive pytest test suite for Tempest, including GPU-accelerated testing capabilities for all subcommands. The test suite is designed to ensure reliability, performance, and correctness across all Tempest functionality.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and GPU configuration
├── pytest.ini               # Pytest configuration
├── run_tests.py            # Test runner with multiple modes
├── commands/               # Tests for CLI subcommands
│   ├── test_simulate.py    # Simulate command tests
│   ├── test_train.py       # Train command tests
│   ├── test_evaluate.py    # Evaluate command tests
│   └── test_other_commands.py  # Visualize, compare, combine, demux tests
├── preflight/              # Pre-execution validation tests
├── midflight/              # Integration tests
└── test_results/           # Test outputs and reports
```

## Requirements

### Python Dependencies
- pytest >= 8.4.2
- tensorflow >= 2.15.1 (with GPU support)
- numpy >= 1.26.4
- pyyaml >= 6.0.3
- matplotlib >= 3.10.7

### GPU Requirements
- NVIDIA GPU with CUDA support (optional but recommended)
- CUDA 12.2+ and cuDNN 8.9+
- Minimum 4GB GPU memory for full test suite
- TensorFlow GPU drivers properly installed

## Running Tests

### Quick Start

```bash
# Run quick smoke tests
python tests/run_tests.py --quick

# Run all tests
python tests/run_tests.py --all

# Run with coverage
python tests/run_tests.py --coverage
```

### Test Categories

#### Unit Tests
Fast, isolated tests of individual components:
```bash
python tests/run_tests.py --unit
# or
pytest -m unit
```

#### Integration Tests
Tests of component interactions:
```bash
python tests/run_tests.py --integration
# or
pytest -m integration
```

#### GPU Tests
Tests requiring GPU hardware:
```bash
python tests/run_tests.py --gpu
# or
pytest -m gpu
```

#### Benchmark Tests
Performance and speed tests:
```bash
python tests/run_tests.py --benchmark
# or
pytest -m benchmark
```

### Subcommand-Specific Tests

Test individual Tempest subcommands:

```bash
# Test simulate command
python tests/run_tests.py --subcommand simulate

# Test train command
python tests/run_tests.py --subcommand train

# Test evaluate command
python tests/run_tests.py --subcommand evaluate

# Test other commands
python tests/run_tests.py --subcommand visualize
python tests/run_tests.py --subcommand compare
python tests/run_tests.py --subcommand combine
python tests/run_tests.py --subcommand demux
```

### Advanced Options

#### Parallel Execution
Run tests in parallel for faster execution:
```bash
python tests/run_tests.py --parallel -n 4
# or
pytest -n auto
```

#### Specific Test Selection
Run specific test files or functions:
```bash
# Run specific test file
pytest tests/commands/test_train.py

# Run specific test class
pytest tests/commands/test_train.py::TestTrainCommand

# Run specific test function
pytest tests/commands/test_train.py::TestTrainCommand::test_gpu_accelerated_training
```

#### Verbose Output
```bash
pytest -vv  # Very verbose
pytest -v   # Verbose
pytest -q   # Quiet
```

## Test Coverage by Subcommand

### Simulate Command Tests
- Basic sequence generation
- Train/validation splitting
- Reproducibility with seeds
- GPU-accelerated generation
- PWM constraint handling
- Performance benchmarks
- Memory efficiency tests

### Train Command Tests
- Standard model training
- Hybrid training approaches
- Ensemble training
- GPU acceleration
- Multi-GPU support
- Checkpoint resumption
- Different optimizers
- Length-constrained training
- Memory management

### Evaluate Command Tests
- Model evaluation metrics
- GPU-accelerated inference
- Confusion matrix generation
- Per-class metrics
- Batch size optimization
- Prediction saving
- Multi-model comparison

### Visualize Command Tests
- Training history plots
- Confusion matrix visualization
- Attention weight visualization
- Embedding space visualization
- Multiple plot formats

### Compare Command Tests
- Multi-model comparison
- Performance metrics
- GPU-accelerated comparison
- Result visualization

### Combine Command Tests
- Bayesian Model Averaging
- Weighted averaging
- Ensemble creation
- Model weight calculation

### Demux Command Tests
- Basic demultiplexing
- GPU-accelerated processing
- UMI correction
- Barcode whitelist handling
- Quality filtering

## GPU Testing Details

### GPU Detection
The test suite automatically detects available GPUs using both TensorFlow and PyTorch backends. GPU information is logged at the start of test runs.

### GPU Memory Management
Tests monitor GPU memory usage to ensure efficient resource utilization:
- Memory growth is enabled by default
- Tests verify memory usage stays within reasonable limits
- Memory monitoring fixtures track usage per test

### GPU Test Markers
Tests requiring GPU are marked with `@pytest.mark.gpu` and will be skipped if no GPU is available.

### Multi-GPU Testing
If multiple GPUs are available, specific tests will utilize multi-GPU strategies for distributed training and inference.

## Test Fixtures

### Core Fixtures (conftest.py)

- `gpu_config`: Session-wide GPU configuration
- `require_gpu`: Skip test if GPU unavailable
- `temp_dir`: Temporary directory for test outputs
- `sample_config`: Sample configuration dictionary
- `sample_config_file`: Temporary config YAML file
- `sample_sequences`: Sample sequence data
- `mock_model_path`: Mock model for testing
- `gpu_memory_monitor`: GPU memory usage monitoring

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-cov pytest-html
    
    - name: Run unit tests
      run: python tests/run_tests.py --unit
    
    - name: Run integration tests
      run: python tests/run_tests.py --integration
    
    - name: Generate coverage report
      run: python tests/run_tests.py --coverage
```

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

2. **Out of memory errors**
   - Reduce batch sizes in tests
   - Enable memory growth in conftest.py
   - Run GPU tests individually

3. **Import errors**
   - Ensure tempest is in PYTHONPATH
   - Install all requirements: `pip install -r requirements.txt`

4. **Slow test execution**
   - Use parallel execution: `pytest -n auto`
   - Run only specific test categories
   - Skip slow tests: `pytest -m "not slow"`

### Debug Mode

Run tests with detailed debugging:
```bash
pytest --pdb  # Drop into debugger on failure
pytest --capture=no  # Show all print statements
pytest --log-cli-level=DEBUG  # Show debug logs
```

## Performance Benchmarks

Expected performance metrics on standard hardware:

| Test Category | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| Simulate (10k seq) | 30s | 5s | 6x |
| Train (5 epochs) | 120s | 20s | 6x |
| Evaluate (1k samples) | 10s | 2s | 5x |
| Demux (10k reads) | 60s | 10s | 6x |

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Add appropriate markers (@pytest.mark.unit, @pytest.mark.gpu, etc.)
3. Include docstrings explaining test purpose
4. Ensure tests are independent and can run in any order
5. Clean up resources in teardown/finally blocks
6. Add GPU variants for performance-critical code

## Test Reports

Test results are saved in `tests/test_results/`:
- `junit.xml`: JUnit format results
- `report.html`: HTML test report
- `coverage_html/`: Coverage HTML report
- `benchmark.json`: Performance benchmark results
- `test_summary.json`: Summary of all test runs

## Contact

For questions or issues with the test suite, please refer to the main Tempest documentation or open an issue on the project repository.
