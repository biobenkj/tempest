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

# Import visualization components
try:
    from .visualize_predictions import (
        TempestInferenceVisualizer,
        main as visualize_main
    )
    _HAS_VISUALIZATION = True
except ImportError as e:
    logger.warning(f"Visualization components not available: {e}")
    _HAS_VISUALIZATION = False
    TempestInferenceVisualizer = None
    visualize_main = None


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
        # Placeholder for model loading logic
        logger.info(f"Loading model from {self.model_path}")
        # This would integrate with Tempest's model loading
        pass
    
    def predict(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Run inference on sequences.
        
        Args:
            sequences: Input sequences
            
        Returns:
            Prediction array
        """
        # Placeholder for prediction logic
        logger.info(f"Running inference on {len(sequences)} sequences")
        # This would call the actual model prediction
        pass
    
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
        # Placeholder for batched prediction
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(sequences)} sequences in {num_batches} batches")
        # This would implement batched processing
        pass


def predict_sequences(
    model,
    sequences: Union[List[str], np.ndarray],
    batch_size: int = 32,
    return_probabilities: bool = False
) -> np.ndarray:
    """
    Convenience function for sequence prediction.
    
    Args:
        model: Trained Tempest model
        sequences: Input sequences
        batch_size: Batch size for processing
        return_probabilities: Whether to return probabilities or labels
        
    Returns:
        Predictions as numpy array
    """
    logger.info(f"Predicting {len(sequences)} sequences")
    
    # Encode sequences if necessary
    if isinstance(sequences[0], str):
        # Convert to encoded format
        logger.debug("Encoding string sequences")
        encoded_sequences = encode_sequences(sequences)
    else:
        encoded_sequences = sequences
    
    # Run prediction
    predictions = model.predict(encoded_sequences, batch_size=batch_size)
    
    # Convert to labels if requested
    if not return_probabilities and len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    return predictions


def encode_sequences(
    sequences: List[str],
    encoding_map: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Encode sequence strings to numerical arrays.
    
    Args:
        sequences: List of sequence strings
        encoding_map: Optional custom encoding map
        
    Returns:
        Encoded sequences as numpy array
    """
    if encoding_map is None:
        encoding_map = {
            'A': 1, 'C': 2, 'G': 3, 'T': 4,
            'N': 0, 'a': 1, 'c': 2, 'g': 3, 't': 4, 'n': 0
        }
    
    max_len = max(len(seq) for seq in sequences)
    encoded = np.zeros((len(sequences), max_len), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            encoded[i, j] = encoding_map.get(base, 0)
    
    return encoded


def decode_sequences(
    encoded_sequences: np.ndarray,
    decoding_map: Optional[Dict[int, str]] = None
) -> List[str]:
    """
    Decode numerical arrays back to sequence strings.
    
    Args:
        encoded_sequences: Encoded sequence array
        decoding_map: Optional custom decoding map
        
    Returns:
        List of sequence strings
    """
    if decoding_map is None:
        decoding_map = {
            0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'
        }
    
    sequences = []
    for encoded_seq in encoded_sequences:
        # Skip padding (zeros)
        seq = ''.join([
            decoding_map.get(int(base), 'N') 
            for base in encoded_seq if base != 0
        ])
        sequences.append(seq)
    
    return sequences


def batch_process_fasta(
    fasta_path: str,
    model,
    batch_size: int = 32,
    max_sequences: Optional[int] = None,
    output_path: Optional[str] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Process sequences from a FASTA file in batches.
    
    Args:
        fasta_path: Path to FASTA file
        model: Trained model
        batch_size: Batch size for processing
        max_sequences: Maximum sequences to process
        output_path: Optional path to save results
        
    Returns:
        Tuple of (sequence_ids, predictions)
    """
    logger.info(f"Processing FASTA file: {fasta_path}")
    
    sequences = []
    sequence_ids = []
    
    # Simple FASTA reading (would integrate with Bio.SeqIO if available)
    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append(''.join(current_seq))
                    sequence_ids.append(current_id)
                    if max_sequences and len(sequences) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Add last sequence
        if current_id and current_seq:
            sequences.append(''.join(current_seq))
            sequence_ids.append(current_id)
    
    # Run predictions
    predictions = predict_sequences(model, sequences, batch_size=batch_size)
    
    # Save results if requested
    if output_path:
        logger.info(f"Saving results to {output_path}")
        # Implementation would save predictions
    
    return sequence_ids, predictions


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
    logger.info("Visualization features not available. Install matplotlib for full functionality.")
