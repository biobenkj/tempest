"""
Tempest inference module for model predictions and analysis.

This module provides tools for:
- Running inference on sequences
- Visualizing model predictions
- Batch processing of sequence data
- Integration with trained Tempest models
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# Import core inference utilities
from .inference_utils import (
    predict_sequences,
    encode_sequences,
    decode_sequences,
    batch_process_fasta
)

# Import visualization components (with graceful fallback)
_HAS_VISUALIZATION = False
TempestInferenceVisualizer = None
visualize_main = None

try:
    from .visualize_predictions import (
        TempestInferenceVisualizer,
        main as visualize_main
    )
    _HAS_VISUALIZATION = True
    logger.debug("Visualization components loaded successfully")
except ImportError as e:
    logger.debug(f"Visualization components not available: {e}")
    # This is fine - visualization is optional and requires matplotlib


class SequencePredictor:
    """
    Base class for sequence prediction with Tempest models.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            config_path: Optional path to model configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        # This would integrate with Tempest's model loading
        # Implementation depends on your model saving/loading strategy
        pass
    
    def predict(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Run inference on sequences.
        
        Args:
            sequences: Input sequences
            
        Returns:
            Prediction array
        """
        logger.info(f"Running inference on {len(sequences)} sequences")
        return predict_sequences(self.model, sequences)
    
    def predict_batch(
        self,
        sequences: Union[List[str], np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Run batched inference on sequences.
        
        Args:
            sequences: Input sequences
            batch_size: Batch size for processing
            
        Returns:
            Prediction array
        """
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(sequences)} sequences in {num_batches} batches")
        return predict_sequences(self.model, sequences, batch_size=batch_size)


# Module exports
__all__ = [
    # Core inference functions
    'predict_sequences',
    'encode_sequences',
    'decode_sequences',
    'batch_process_fasta',
    
    # Classes
    'SequencePredictor',
]

# Add visualization exports if available
if _HAS_VISUALIZATION:
    __all__.extend([
        'TempestInferenceVisualizer',
        'visualize_main'
    ])
else:
    logger.info(
        "Visualization features not available. "
        "Install matplotlib and ensure TempestVisualizer is available for full functionality."
    )
