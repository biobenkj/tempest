# BMA and Ensemble Configuration Reconciliation Guide

## Overview

This document explains the reconciliation between the configuration files (YAML), configuration classes (`BMAConfig`, `EnsembleConfig`), and the `ModelCombiner` class in Tempest's inference module.

## Configuration Architecture

### Program Flow
```
main.py → cli.py → combine command → tempest.inference.combiner.ModelCombiner
```

### Configuration Classes Hierarchy

```
TempestConfig (master config)
    ├── ModelConfig
    ├── SimulationConfig
    ├── TrainingConfig
    ├── EnsembleConfig
    │   ├── BMAConfig (nested)
    │   ├── weighted_average_config (dict)
    │   ├── prediction_aggregation (dict)
    │   ├── calibration (dict)
    │   ├── diversity (dict)
    │   └── uncertainty (dict)
    ├── InferenceConfig
    ├── PWMConfig
    └── HybridTrainingConfig
```

## Key Configuration Classes

### BMAConfig (Bayesian Model Averaging)

The `BMAConfig` class provides comprehensive BMA functionality with multiple approximation methods:

```python
@dataclass
class BMAConfig:
    # Enable/disable BMA
    enabled: bool = True
    
    # Prior type: 'uniform', 'informative', 'adaptive'
    prior_type: str = 'uniform'
    prior_weights: Optional[Dict[str, float]] = None
    
    # Approximation method: 'bic', 'laplace', 'variational', 'cross_validation'
    approximation: str = 'bic'
    
    # Nested approximation parameters
    approximation_params: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Posterior computation
    temperature: float = 1.0
    compute_posterior_variance: bool = True
    normalize_posteriors: bool = True
    min_posterior_weight: float = 0.01
    
    # Model selection
    selection_criteria: Optional[Dict[str, Any]] = None
    use_model_averaging: bool = True
    evidence_threshold: float = 0.05
```

### EnsembleConfig

The `EnsembleConfig` class orchestrates ensemble creation and prediction:

```python
@dataclass
class EnsembleConfig:
    # General settings
    enabled: bool = True
    num_models: int = 3
    voting_method: str = 'bayesian_model_averaging'
    
    # Model specifications
    models: Optional[List[Dict[str, Any]]] = None
    
    # Nested BMA configuration
    bma_config: Optional[BMAConfig] = None
    
    # Alternative weighting
    weighted_average_config: Optional[Dict[str, Any]] = None
    
    # Prediction settings
    prediction_method: str = 'weighted_average'
    confidence_weighting: bool = True
    
    # Calibration
    calibration_enabled: bool = False
    calibration_method: str = 'isotonic'
    calibration_split: float = 0.2
    
    # Diversity enforcement
    enforce_diversity: bool = False
    diversity_metric: str = 'disagreement'
    min_diversity_threshold: float = 0.1
    
    # Uncertainty quantification
    compute_epistemic: bool = True
    compute_aleatoric: bool = True
    confidence_intervals: bool = True
    interval_alpha: float = 0.95
```

## YAML Configuration Structure

### Complete Ensemble Configuration Example

```yaml
ensemble:
  enabled: true
  num_models: 5
  voting_method: "bayesian_model_averaging"
  
  # BMA Configuration
  bma_config:
    enabled: true
    prior_type: "adaptive"
    approximation: "bic"  # or 'laplace', 'variational', 'cross_validation'
    
    approximation_params:
      bic:
        penalty_factor: 1.0
      laplace:
        num_samples: 1000
        damping: 0.01
      variational:
        num_iterations: 100
        learning_rate: 0.01
        convergence_threshold: 1e-4
      cross_validation:
        num_folds: 5
        stratified: true
    
    temperature: 1.0
    compute_posterior_variance: true
    normalize_posteriors: true
    min_posterior_weight: 0.01
    
    selection_criteria:
      metric: "validation_accuracy"
      threshold: 0.05
    
    use_model_averaging: true
    evidence_threshold: 0.05
  
  # Weighted Average (alternative to BMA)
  weighted_average_config:
    optimization: "performance"
    performance_metrics:
      val_accuracy: 0.7
      val_loss: 0.3
    type_bonus:
      hybrid: 1.1
      standard: 1.0
  
  # Prediction Aggregation
  prediction_aggregation:
    method: "weighted_average"
    confidence_weighting: true
    uncertainty_estimation: true
    apply_temperature_scaling: false
  
  # Calibration
  calibration:
    enabled: true
    method: "isotonic"
    use_separate_calibration_set: true
    calibration_split: 0.2
  
  # Diversity
  diversity:
    enforce_diversity: true
    diversity_metric: "disagreement"
    min_diversity_threshold: 0.1
  
  # Uncertainty
  uncertainty:
    compute_epistemic: true
    compute_aleatoric: true
    method: "ensemble_variance"
    confidence_intervals: true
    interval_alpha: 0.95
```

## ModelCombiner Usage

### Initialization

```python
from tempest.inference.combiner import ModelCombiner
from tempest.config import load_config

# Option 1: Load from YAML
config = load_config("config.yaml")
combiner = ModelCombiner(config.ensemble)

# Option 2: Direct dictionary
ensemble_config = {
    'voting_method': 'bayesian_model_averaging',
    'bma_config': {
        'approximation': 'bic',
        'temperature': 1.0
    }
}
combiner = ModelCombiner(ensemble_config)

# Option 3: Use EnsembleConfig object
from tempest.config import EnsembleConfig, BMAConfig

bma_config = BMAConfig(
    approximation='laplace',
    laplace_num_samples=1000
)
ensemble_config = EnsembleConfig(
    voting_method='bayesian_model_averaging',
    bma_config=bma_config
)
combiner = ModelCombiner(ensemble_config)
```

### Loading Models

```python
# Load from file paths
model_paths = [
    "/path/to/model1.h5",
    "/path/to/model2.h5",
    "/path/to/model3.h5"
]
combiner.load_models(model_paths)

# Or load with names
model_dict = {
    "standard_model_1": "/path/to/model1.h5",
    "hybrid_model_1": "/path/to/model2.h5",
    "standard_model_2": "/path/to/model3.h5"
}
combiner.load_models(model_dict)
```

### Computing Weights

```python
# Using BMA with BIC approximation
combiner.compute_weights(validation_data)

# The combiner will:
# 1. Compute evidence using selected approximation method
# 2. Apply priors (uniform, informative, or adaptive)
# 3. Compute posterior weights with temperature scaling
# 4. Apply minimum weight threshold
# 5. Normalize weights
```

### Making Predictions

```python
# Basic prediction
predictions = combiner.predict(X_test)

# With uncertainty quantification
result = combiner.predict(
    X_test,
    return_uncertainty=True,
    return_individual=True,
    apply_calibration=True
)

# Access results
ensemble_pred = result['predictions']
uncertainty = result['uncertainty']
individual_preds = result['individual_predictions']
```

## BMA Approximation Methods

### 1. BIC (Bayesian Information Criterion)
```yaml
bma_config:
  approximation: "bic"
  approximation_params:
    bic:
      penalty_factor: 1.0  # Standard BIC penalty
```

**When to use**: Fast approximation, works well with many models, good default choice.

### 2. Laplace Approximation
```yaml
bma_config:
  approximation: "laplace"
  approximation_params:
    laplace:
      num_samples: 1000
      damping: 0.01
```

**When to use**: More accurate than BIC, reasonable computational cost, good for smaller ensembles.

### 3. Variational Inference
```yaml
bma_config:
  approximation: "variational"
  approximation_params:
    variational:
      num_iterations: 100
      learning_rate: 0.01
      convergence_threshold: 1e-4
```

**When to use**: Most accurate, higher computational cost, best for critical applications.

### 4. Cross-Validation
```yaml
bma_config:
  approximation: "cross_validation"
  approximation_params:
    cross_validation:
      num_folds: 5
      stratified: true
```

**When to use**: Data-driven evidence estimation, good when you have sufficient validation data.

## Configuration Best Practices

### 1. Start Simple
```yaml
ensemble:
  enabled: true
  num_models: 3
  voting_method: "bayesian_model_averaging"
  bma_config:
    enabled: true
    approximation: "bic"  # Start with BIC
```

### 2. Enable Calibration for Production
```yaml
calibration:
  enabled: true
  method: "isotonic"
  calibration_split: 0.2
```

### 3. Use Diversity Enforcement
```yaml
diversity:
  enforce_diversity: true
  diversity_metric: "disagreement"
  min_diversity_threshold: 0.1
```

### 4. Enable Uncertainty Quantification
```yaml
uncertainty:
  compute_epistemic: true
  compute_aleatoric: true
  confidence_intervals: true
```

## Command-Line Usage

### Using the combine subcommand

```bash
# Basic usage
tempest combine \
    --config config.yaml \
    --models model1.h5 model2.h5 model3.h5 \
    --validation-data validation.pkl \
    --output-dir ./ensemble_results

# With test data evaluation
tempest combine \
    --config config.yaml \
    --models model1.h5 model2.h5 model3.h5 \
    --validation-data validation.pkl \
    --test-data test.pkl \
    --output-dir ./ensemble_results

# Load pre-trained ensemble
tempest combine \
    --config config.yaml \
    --models ./ensemble_dir \
    --validation-data validation.pkl
```

## Troubleshooting

### Issue: Configuration Not Loading Properly

**Problem**: BMA config not being recognized

**Solution**: Ensure YAML structure matches the dataclass fields:
```yaml
ensemble:
  bma_config:  # Must be 'bma_config', not 'bma' or 'bma_configuration'
    enabled: true
    approximation: "bic"
```

### Issue: Type Mismatch Errors

**Problem**: Config values have wrong types

**Solution**: Use proper YAML types:
```yaml
# Correct
num_models: 5  # integer
temperature: 1.0  # float
enabled: true  # boolean

# Incorrect
num_models: "5"  # string
temperature: 1  # integer instead of float
enabled: "true"  # string
```

### Issue: Missing Nested Config

**Problem**: Nested config dict is None

**Solution**: Always check for None before accessing nested configs:
```python
if self.config.bma_config is not None:
    approximation = self.config.bma_config.approximation
else:
    # Use default or raise error
    approximation = 'bic'
```

## Migration Guide

### From Old Config Format to New Format

**Old format** (simplified):
```yaml
ensemble:
  bma_config:
    method: "validation_accuracy"
    min_weight: 0.01
```

**New format** (comprehensive):
```yaml
ensemble:
  voting_method: "bayesian_model_averaging"
  bma_config:
    enabled: true
    approximation: "bic"
    min_posterior_weight: 0.01
```

## Example Workflow

```python
#!/usr/bin/env python3
"""Complete workflow for model combination."""

from tempest.config import load_config
from tempest.inference.combiner import ModelCombiner

# 1. Load configuration
config = load_config("config.yaml")

# 2. Initialize combiner
combiner = ModelCombiner(config.ensemble)

# 3. Load trained models
model_paths = [
    "outputs/model_1.h5",
    "outputs/model_2.h5",
    "outputs/model_3.h5"
]
combiner.load_models(model_paths)

# 4. Compute BMA weights
combiner.compute_weights("data/validation.pkl")

# 5. Calibrate (optional but recommended)
if config.ensemble.calibration_enabled:
    combiner.calibrate("data/validation.pkl")

# 6. Evaluate on test set
metrics = combiner.evaluate("data/test.pkl")
print(f"Ensemble accuracy: {metrics['ensemble_accuracy']:.4f}")
print(f"Mean epistemic uncertainty: {metrics['mean_epistemic']:.4f}")

# 7. Make predictions
result = combiner.predict(
    X_new,
    return_uncertainty=True,
    apply_calibration=True
)

# 8. Save results
combiner.save_results("./ensemble_results")
```

## Summary

The configuration system provides:
1. **Comprehensive BMA** with multiple approximation methods
2. **Flexible weighting** for different use cases
3. **Uncertainty quantification** for reliability
4. **Calibration** for production deployment
5. **Diversity enforcement** for robust ensembles

All configuration flows through the standardized classes (`BMAConfig`, `EnsembleConfig`) which are properly parsed from YAML and consumed by `ModelCombiner`.
