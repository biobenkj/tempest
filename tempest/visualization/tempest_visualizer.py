"""
Tempest-specific visualization integration for model predictions.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
import tensorflow as tf

from .annotated_reads import (
    save_plots_to_pdf,
    get_default_colors,
    plot_annotation_statistics
)

logger = logging.getLogger(__name__)


class TempestVisualizer:
    """
    Visualizer for Tempest model predictions with support for
    length-constrained CRF outputs.
    """
    
    def __init__(
        self,
        label_names: List[str],
        colors: Optional[Dict[str, str]] = None,
        output_dir: str = "./visualizations"
    ):
        """
        Initialize the Tempest visualizer.
        
        Args:
            label_names: List of label names corresponding to model output indices
            colors: Optional custom color mapping for labels
            output_dir: Directory for saving visualizations
        """
        self.label_names = label_names
        self.colors = colors or get_default_colors(label_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TempestVisualizer with {len(label_names)} labels")
    
    def visualize_predictions(
        self,
        sequences: Union[List[str], np.ndarray],
        predictions: np.ndarray,
        read_names: Optional[List[str]] = None,
        output_filename: str = "predictions.pdf",
        include_statistics: bool = True,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Visualize model predictions for a batch of sequences.
        
        Args:
            sequences: Input sequences (strings or encoded arrays)
            predictions: Model predictions (logits or probabilities)
            read_names: Optional names for the sequences
            output_filename: Name for the output PDF file
            include_statistics: Whether to generate statistics plot
            metadata_list: Optional metadata for each sequence
            
        Returns:
            Path to the generated PDF file
        """
        # Convert encoded sequences back to strings if necessary
        if isinstance(sequences, np.ndarray):
            sequences = self._decode_sequences(sequences)
        
        # Generate read names if not provided
        if read_names is None:
            read_names = [f"Read_{i:04d}" for i in range(len(sequences))]
        
        # Ensure predictions are in the right format
        if len(predictions.shape) == 2:  # Missing class dimension
            # Assume these are integer labels
            predictions = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions = [predictions[i] for i in range(predictions.shape[0])]
        
        # Generate visualization PDF
        output_path = self.output_dir / output_filename
        save_plots_to_pdf(
            sequences=sequences,
            predictions=predictions,
            read_names=read_names,
            filename=str(output_path),
            colors=self.colors,
            label_names=self.label_names,
            metadata_list=metadata_list
        )
        
        # Generate statistics if requested
        if include_statistics:
            stats_filename = output_filename.replace('.pdf', '_stats.png')
            stats_path = self.output_dir / stats_filename
            plot_annotation_statistics(
                predictions=predictions,
                label_names=self.label_names,
                output_file=str(stats_path),
                title="Prediction Statistics"
            )
            logger.info(f"Saved statistics to {stats_path}")
        
        return str(output_path)
    
    def visualize_model_outputs(
        self,
        model: tf.keras.Model,
        sequences: np.ndarray,
        batch_size: int = 32,
        output_filename: str = "model_outputs.pdf",
        **kwargs
    ) -> str:
        """
        Visualize outputs directly from a Tempest model.
        
        Args:
            model: Trained Tempest model
            sequences: Encoded input sequences
            batch_size: Batch size for prediction
            output_filename: Output PDF filename
            **kwargs: Additional arguments for visualize_predictions
            
        Returns:
            Path to the generated PDF file
        """
        # Get predictions from model
        predictions = model.predict(sequences, batch_size=batch_size)
        
        # Handle different output formats (CRF vs regular)
        if hasattr(model, 'crf_layer'):
            # For CRF models, predictions might need special handling
            logger.info("Processing CRF model outputs")
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Take the decoded path
        
        return self.visualize_predictions(
            sequences=sequences,
            predictions=predictions,
            output_filename=output_filename,
            **kwargs
        )
    
    def _decode_sequences(self, encoded_sequences: np.ndarray) -> List[str]:
        """
        Decode encoded sequences back to strings.
        
        Args:
            encoded_sequences: Numpy array of encoded sequences
            
        Returns:
            List of decoded sequence strings
        """
        # Basic nucleotide mapping (extend as needed)
        decode_map = {
            0: 'N',  # Padding or unknown
            1: 'A',
            2: 'C',
            3: 'G',
            4: 'T',
        }
        
        decoded = []
        for seq in encoded_sequences:
            decoded_seq = ''.join([
                decode_map.get(int(base), 'N') for base in seq if base != 0
            ])
            decoded.append(decoded_seq)
        
        return decoded
    
    def compare_predictions(
        self,
        sequences: Union[List[str], np.ndarray],
        predictions_dict: Dict[str, np.ndarray],
        read_names: Optional[List[str]] = None,
        output_filename: str = "comparison.pdf"
    ) -> str:
        """
        Compare predictions from multiple models or methods.
        
        Args:
            sequences: Input sequences
            predictions_dict: Dictionary mapping method names to predictions
            read_names: Optional names for sequences
            output_filename: Output PDF filename
            
        Returns:
            Path to the generated comparison PDF
        """
        # Implementation for comparing multiple prediction sets
        # This would create side-by-side visualizations
        # Placeholder for now
        logger.info("Comparison visualization not yet implemented")
        return ""


def visualize_predictions(
    sequences: Union[List[str], np.ndarray],
    predictions: np.ndarray,
    label_names: List[str],
    output_file: str = "predictions.pdf",
    colors: Optional[Dict[str, str]] = None,
    **kwargs
) -> str:
    """
    Convenience function to visualize predictions without creating a visualizer object.
    
    Args:
        sequences: Input sequences
        predictions: Model predictions
        label_names: List of label names
        output_file: Output PDF filename
        colors: Optional color mapping
        **kwargs: Additional arguments for visualization
        
    Returns:
        Path to generated PDF file
    """
    visualizer = TempestVisualizer(
        label_names=label_names,
        colors=colors,
        output_dir=Path(output_file).parent
    )
    
    return visualizer.visualize_predictions(
        sequences=sequences,
        predictions=predictions,
        output_filename=Path(output_file).name,
        **kwargs
    )


def visualize_batch_predictions(
    model: tf.keras.Model,
    data_generator,
    label_names: List[str],
    num_batches: int = 1,
    output_dir: str = "./visualizations",
    colors: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Visualize predictions for multiple batches from a data generator.
    
    Args:
        model: Trained model
        data_generator: Data generator yielding (sequences, labels)
        label_names: List of label names
        num_batches: Number of batches to visualize
        output_dir: Directory for output files
        colors: Optional color mapping
        
    Returns:
        List of paths to generated PDF files
    """
    visualizer = TempestVisualizer(
        label_names=label_names,
        colors=colors,
        output_dir=output_dir
    )
    
    output_files = []
    
    for batch_idx in range(min(num_batches, len(data_generator))):
        # Get batch
        sequences, true_labels = data_generator[batch_idx]
        
        # Get predictions
        predictions = model.predict(sequences)
        
        # Generate filename
        output_filename = f"batch_{batch_idx:03d}_predictions.pdf"
        
        # Create visualization
        output_path = visualizer.visualize_predictions(
            sequences=sequences,
            predictions=predictions,
            output_filename=output_filename,
            include_statistics=True
        )
        
        output_files.append(output_path)
        logger.info(f"Visualized batch {batch_idx + 1}/{num_batches}")
    
    return output_files
