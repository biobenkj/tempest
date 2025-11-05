"""
Training module for Tempest.

Contains model training utilities, including hybrid robustness training
for handling invalid reads and architectural errors.
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

__all__ = [
    # Main trainer
    'HybridTrainer',
    
    # Components
    'ArchitectureDiscriminator',
    'PseudoLabelGenerator',
    
    # Utility functions
    'pad_sequences',
    'convert_labels_to_categorical',
    'build_model_from_config',
    'print_model_summary',
]
