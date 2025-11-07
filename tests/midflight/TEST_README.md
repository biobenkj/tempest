# Tempest Comprehensive Test Suite

This directory contains extensive tests for the Tempest sequence annotation system, covering configuration ingestion, length-constrained CRF implementation, Bayesian Model Averaging (BMA), ensemble modeling, and hybrid training.

## Test Modules

### 1. test_config_ingestion.py
**Purpose**: Comprehensive tests for config.yaml ingestion and validation

**Test Coverage**:
- Basic configuration loading from YAML files
- Complete configuration with all sections (model, simulation, training, pwm, ensemble, hybrid)
- Default value handling and validation
- Nested configuration structures (length constraints)
- Type checking and conversion
- Configuration reproducibility
- Round-trip serialization (save and load)
- Random seed preservation for reproducible experiments

**Key Test Classes**:
- `TestConfigLoading`: YAML file loading and parsing
- `TestModelConfig`: Model architecture configuration
- `TestLengthConstraints`: Length constraint specifications
- `TestSimulationConfig`: Data simulation parameters
- `TestTrainingConfig`: Training hyperparameters
- `TestEnsembleConfig`: BMA configuration
- `TestHybridTrainingConfig`: Hybrid training settings
- `TestPWMConfig`: Position weight matrix settings
- `TestConfigReproducibility`: Reproducibility assertions

**Reproducibility Tests**:
- Random seed preservation across config load/save cycles
- Deterministic configuration parsing
- Consistent default values

### 2. test_length_constrained_crf.py
**Purpose**: Comprehensive tests for the length-constrained CRF implementation

**Test Coverage**:
- Constraint weight ramping (linear increase from 0 to maximum)
- Length penalty computation for segments
- Segment detection and boundary identification
- Vectorized operations for batch processing
- XLA compatibility for performance optimization
- Reproducibility with fixed seeds

**Key Test Classes**:
- `TestConstraintWeightRamping`: Weight scheduling over epochs
- `TestLengthPenaltyComputation`: Penalty calculation for correct/incorrect lengths
- `TestSegmentDetection`: Segment boundary identification
- `TestVectorizedOperations`: Batch processing and XLA compilation
- `TestReproducibility`: Deterministic penalty computation
- `TestConvenienceFunction`: Wrapper function testing
- `TestIntegration`: Full training scenario simulation

**Reproducibility Tests**:
- Deterministic penalty computation
- Consistent constraint array initialization
- Reproducible weight ramping schedules

**Key Assertions**:
- Initial constraint weight is zero at epoch 0
- Weight increases linearly during ramp period
- Weight caps at maximum after ramp period
- Correct segment lengths have zero/minimal penalty
- Too-short segments have positive penalty proportional to (min_len - actual_len)²
- Too-long segments have positive penalty proportional to (actual_len - max_len)²

### 3. test_ensemble_bma.py
**Purpose**: Comprehensive tests for Bayesian Model Averaging and ensemble modeling

**Test Coverage**:
- Ensemble creation and configuration
- Model diversity through architecture variation
- Model diversity through initialization variation
- Bayesian Model Averaging weight computation
- Uniform prior weights
- Performance-based posterior weights
- Ensemble prediction aggregation (weighted averaging)
- Majority voting aggregation
- Performance tracking and evaluation
- Reproducibility with fixed seeds

**Key Test Classes**:
- `TestEnsembleConfig`: Ensemble configuration
- `TestEnsembleCreation`: Ensemble initialization
- `TestBayesianModelAveraging`: BMA weight computation
- `TestEnsemblePrediction`: Prediction aggregation
- `TestEnsembleEvaluation`: Performance tracking
- `TestModelDiversity`: Diversity strategies
- `TestReproducibility`: Deterministic operations
- `TestIntegration`: Full BMA workflow
- `TestEdgeCases`: Edge case handling

**Reproducibility Tests**:
- Deterministic weight computation
- Reproducible predictions with fixed seeds
- Consistent ensemble aggregation

**Key Assertions**:
- Uniform prior weights sum to 1.0 and are equal
- Performance-based weights sum to 1.0
- Best model has highest weight in performance-based BMA
- Ensemble predictions match expected shape
- Weighted aggregation is numerically stable

### 4. test_hybrid_training.py
**Purpose**: Comprehensive tests for hybrid robustness training

**Test Coverage**:
- Hybrid training configuration
- Invalid read generation (segment loss, duplication, truncation, chimeric, scrambled)
- Training phase management (warmup, discriminator, pseudo-labeling)
- Discriminator functionality
- Pseudo-labeling with confidence thresholds
- Confidence threshold decay over epochs
- Loss weighting and scheduling
- Architecture validation
- Reproducibility with fixed seeds

**Key Test Classes**:
- `TestHybridTrainingConfig`: Hybrid configuration
- `TestInvalidReadGeneration`: Invalid read augmentation
- `TestTrainingPhaseManagement`: Phase transitions
- `TestDiscriminatorFunctionality`: Discriminator training
- `TestPseudoLabeling`: Pseudo-label generation
- `TestLossWeighting`: Multi-component loss
- `TestValidationChecks`: Architecture validation
- `TestReproducibility`: Deterministic operations
- `TestIntegration`: Full hybrid workflow

**Reproducibility Tests**:
- Deterministic invalid read generation
- Consistent phase detection
- Reproducible loss weighting

**Key Assertions**:
- Training phases execute in correct order (warmup → discriminator → pseudo-label)
- Invalid generation probabilities sum to 1.0
- Confidence threshold decreases with decay
- High-confidence predictions are selected for pseudo-labeling
- Architecture validation correctly identifies invalid structures
- Loss components combine correctly with specified weights

## Running the Tests

### Install Dependencies

```bash
pip install pytest numpy tensorflow pyyaml
```

### Run All Tests

Use the master test runner:

```bash
python run_all_tests.py
```

This will run all four test modules and provide a summary report.

### Run Individual Test Modules

```bash
# Config ingestion tests
python -m pytest test_config_ingestion.py -v

# Length-constrained CRF tests
python -m pytest test_length_constrained_crf.py -v

# BMA and ensemble tests
python -m pytest test_ensemble_bma.py -v

# Hybrid training tests
python -m pytest test_hybrid_training.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
python -m pytest test_config_ingestion.py::TestModelConfig -v

# Run a specific test method
python -m pytest test_length_constrained_crf.py::TestConstraintWeightRamping::test_weight_ramps_linearly -v
```

### Run with Coverage

```bash
pytest --cov=tempest --cov-report=html test_*.py
```

## Test Organization

Each test module follows a consistent structure:

1. **Configuration Tests**: Test parameter loading and validation
2. **Core Functionality Tests**: Test main algorithms and computations
3. **Integration Tests**: Test multiple components working together
4. **Reproducibility Tests**: Ensure deterministic behavior with fixed seeds
5. **Edge Case Tests**: Test boundary conditions and error handling

## Assertions for Reproducibility

All test modules include specific tests to ensure reproducible results:

1. **Fixed Random Seeds**: Tests use fixed seeds (e.g., 42) for numpy and TensorFlow operations
2. **Deterministic Computations**: Verify that same inputs always produce same outputs
3. **Configuration Preservation**: Ensure configs maintain values through save/load cycles
4. **Consistent Initialization**: Check that models initialize identically with same seeds

## Expected Test Results

When all tests pass, you should see output like:

```
✅ PASSED: Config.yaml Ingestion Tests
✅ PASSED: Length-Constrained CRF Tests
✅ PASSED: BMA and Ensemble Modeling Tests
✅ PASSED: Hybrid Training Tests

Total: 4 test modules
Passed: 4
Failed: 0

✅ All test modules passed!
```

## Key Features Tested

### Configuration System
- YAML/JSON loading
- Nested configuration structures
- Default value handling
- Type conversion and validation
- Reproducible serialization

### Length-Constrained CRF
- Segment length penalties: L_penalty = (L_min - L)₊² + (L - L_max)₊²
- Constraint weight ramping: λ(t) = λ_max × min(t / T_ramp, 1)
- Vectorized batch processing
- XLA-compatible operations

### Bayesian Model Averaging
- Uniform priors: P(M_i) = 1/N
- Performance-based posteriors: P(M_i|D) ∝ exp(accuracy_i)
- Weighted prediction: P(y|x) = Σ P(y|x,M_i) × P(M_i|D)

### Hybrid Training
- Invalid read augmentation
- Three-phase training: warmup → discriminator → pseudo-labeling
- Confidence-based pseudo-labeling
- Multi-component loss: L_total = L_CRF + λ_invalid × L_invalid + λ_adv × L_adv

## Troubleshooting

### Import Errors

If you see import errors, ensure the Tempest package is in your Python path:

```python
import sys
sys.path.insert(0, '/path/to/tempest')
```

### TensorFlow Errors

Some tests require TensorFlow. Install with:

```bash
pip install tensorflow>=2.10.0
```

### Mock Object Errors

Tests use unittest.mock extensively. Ensure you're using Python 3.8+:

```bash
python --version  # Should be 3.8 or higher
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: python run_all_tests.py
```

## Contributing

When adding new features to Tempest, please:

1. Add corresponding tests to the appropriate test module
2. Ensure all existing tests still pass
3. Add reproducibility tests with fixed seeds
4. Document any new test classes or assertions
5. Run the full test suite before submitting changes

## Test Statistics

Total test methods: 100+
Total test classes: 50+
Coverage target: >80%
Estimated runtime: 2-5 minutes

## Contact

For questions about the test suite, please contact:
Ben Johnson <ben.johnson@vai.org>

## License

MIT License - Same as Tempest project
