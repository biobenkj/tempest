"""
Training module for Tempest.

Contains model training utilities, including:
- Single model training (ModelTrainer)
- Ensemble training with BMA (EnsembleTrainer)
- Hybrid robustness training for handling invalid reads and architectural errors
"""

from .hybrid_trainer import (
    HybridTrainer,
    ArchitectureDiscriminator,
    PseudoLabelGenerator,
    pad_sequences,
    convert_labels_to_categorical,
    build_model_from_config,
    print_model_summary
)

from .trainer import (
    ModelTrainer,
    PerTokenAccuracy,
    PerLabelMetrics
)

from .ensemble import (
    EnsembleTrainer,
    BMAPredictor
)

__all__ = [
    # Single model trainer
    'ModelTrainer',
    
    # Ensemble trainer
    'EnsembleTrainer',
    'BMAPredictor',
    
    # Hybrid trainer
    'HybridTrainer',
    
    # Components
    'ArchitectureDiscriminator',
    'PseudoLabelGenerator',
    
    # Metrics and callbacks
    'PerTokenAccuracy',
    'PerLabelMetrics',
    
    # Utility functions
    'pad_sequences',
    'convert_labels_to_categorical',
    'build_model_from_config',
    'print_model_summary',
]
