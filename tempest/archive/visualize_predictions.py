#!/usr/bin/env python3
"""
Integration script for visualizing Tempest model predictions on actual data.
This demonstrates how to connect the visualization module with Tempest's
inference pipeline.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

# Import from Tempest modules
from tempest.core import build_model_from_config
from tempest.inference.inference_utils import encode_sequences
from tempest.visualization import TempestVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TempestInferenceVisualizer:
    """
    Class for running inference and visualization on sequences using Tempest models.
    """
    
    def __init__(self, config_path: str, model_path: str, output_dir: str = "./results"):
        """
        Initialize the inference visualizer.
        
        Args:
            config_path: Path to model configuration file
            model_path: Path to trained model weights
            output_dir: Directory for saving results
        """
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Build model
        self.model = self._build_model()
        
        # Initialize visualizer if available
        self.visualizer = TempestVisualizer(
                label_names=self.config.get('label_names', self._default_labels()),
                output_dir=str(self.output_dir / "visualizations")
            )
        self._has_visualizer = True
        
        logger.info(f"Initialized TempestInferenceVisualizer")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Visualization: {'enabled' if self._has_visualizer else 'disabled'}")
    
    def _build_model(self):
        """Build and load the model."""
        try:
            # Build model from config
            model = build_model_from_config(self.config)
            
            # Load weights
            model.load_weights(self.model_path)
            logger.info("Successfully loaded model weights")
            
            return model
        except Exception as e:
            logger.error(f"Failed to build/load model: {e}")
            raise
    
    def _default_labels(self):
        """Return default label names if not in config."""
        num_labels = self.config.get('num_labels', 10)
        return [f"Label_{i}" for i in range(num_labels)]
    
    def process_fasta_file(
        self,
        fasta_path: str,
        batch_size: int = 32,
        max_sequences: int = None,
        visualize: bool = True,
        save_predictions: bool = True
    ):
        """
        Process sequences from a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            batch_size: Batch size for inference
            max_sequences: Maximum number of sequences to process
            visualize: Whether to create visualizations
            save_predictions: Whether to save predictions to file
        """
        logger.info(f"Processing FASTA file: {fasta_path}")
        
        # Load sequences
        sequences, read_names = self._load_fasta(fasta_path, max_sequences)
        logger.info(f"Loaded {len(sequences)} sequences")
        
        # Encode sequences
        encoded_sequences = self._encode_sequences(sequences)
        
        # Run inference
        predictions = self.model.predict(encoded_sequences, batch_size=batch_size)
        logger.info("Completed inference")
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(sequences, predictions, read_names)
        
        # Create visualizations if requested
        if visualize and self._has_visualizer:
            self._visualize_results(sequences, predictions, read_names)
        elif visualize and not self._has_visualizer:
            logger.warning("Visualization requested but TempestVisualizer not available")
        
        return predictions
    
    def process_sequences(
        self,
        sequences: list,
        read_names: list = None,
        visualize: bool = True,
        save_predictions: bool = True,
        batch_size: int = 32
    ):
        """
        Process a list of sequences directly.
        
        Args:
            sequences: List of sequence strings
            read_names: Optional list of sequence names
            visualize: Whether to create visualizations
            save_predictions: Whether to save predictions
            batch_size: Batch size for inference
        """
        if read_names is None:
            read_names = [f"Seq_{i:04d}" for i in range(len(sequences))]
        
        # Encode sequences
        encoded_sequences = self._encode_sequences(sequences)
        
        # Run inference
        predictions = self.model.predict(encoded_sequences, batch_size=batch_size)
        
        # Save and visualize as requested
        if save_predictions:
            self._save_predictions(sequences, predictions, read_names)
        
        if visualize and self._has_visualizer:
            self._visualize_results(sequences, predictions, read_names)
        elif visualize and not self._has_visualizer:
            logger.warning("Visualization requested but TempestVisualizer not available")
        
        return predictions
    
    def _load_fasta(self, fasta_path: str, max_sequences: int = None):
        """Load sequences from a FASTA file."""
        sequences = []
        read_names = []
        
        try:
            from Bio import SeqIO
            
            for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
                if max_sequences and i >= max_sequences:
                    break
                
                sequences.append(str(record.seq))
                read_names.append(record.id)
            
            logger.debug("Loaded sequences using BioPython")
        
        except ImportError:
            logger.debug("BioPython not available, using simple FASTA parser")
            
            # Simple FASTA parser
            current_seq = []
            current_name = None
            
            with open(fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_name and current_seq:
                            sequences.append(''.join(current_seq))
                            read_names.append(current_name)
                            if max_sequences and len(sequences) >= max_sequences:
                                break
                        current_name = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                # Add last sequence
                if current_name and current_seq:
                    sequences.append(''.join(current_seq))
                    read_names.append(current_name)
        
        return sequences, read_names
    
    def _encode_sequences(self, sequences: list):
        """Encode sequences for model input using inference_utils."""
        # Use the shared encoding function from inference_utils
        return encode_sequences(sequences)
    
    def _save_predictions(self, sequences: list, predictions: np.ndarray, read_names: list):
        """Save predictions to file."""
        output_file = self.output_dir / "predictions.json"
        
        # Convert predictions to labels
        if len(predictions.shape) > 2:
            predicted_labels = np.argmax(predictions, axis=-1)
        else:
            predicted_labels = predictions
        
        # Create results dictionary
        results = []
        for i, (seq, pred, name) in enumerate(zip(sequences, predicted_labels, read_names)):
            results.append({
                'read_name': name,
                'sequence_length': len(seq),
                'predicted_labels': pred.tolist()[:len(seq)]  # Trim padding
            })
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved predictions to {output_file}")
    
    def _visualize_results(self, sequences: list, predictions: np.ndarray, read_names: list):
        """Create visualizations of the results."""
        if not self._has_visualizer:
            logger.warning("Cannot create visualizations - TempestVisualizer not available")
            return
        
        # Create main visualization PDF
        try:
            pdf_path = self.visualizer.visualize_predictions(
                sequences=sequences,
                predictions=predictions,
                read_names=read_names,
                output_filename="inference_results.pdf",
                include_statistics=True
            )
            
            logger.info(f"Created visualization: {pdf_path}")
            
            # Also create individual visualizations for first few sequences
            if len(sequences) <= 5:
                for i, (seq, pred, name) in enumerate(zip(sequences, predictions, read_names)):
                    individual_path = self.visualizer.visualize_predictions(
                        sequences=[seq],
                        predictions=[pred],
                        read_names=[name],
                        output_filename=f"individual_{i:02d}_{name[:20]}.pdf",
                        include_statistics=False
                    )
                    logger.info(f"  Individual visualization: {individual_path}")
        
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run Tempest inference with visualization"
    )
    
    parser.add_argument(
        'input',
        help='Input FASTA file or directory'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--output-dir',
        default='./inference_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving predictions to file'
    )
    
    args = parser.parse_args()
    
    # Initialize the inference visualizer
    try:
        visualizer = TempestInferenceVisualizer(
            config_path=args.config,
            model_path=args.model,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize visualizer: {e}")
        sys.exit(1)
    
    # Process the input
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix in ['.fasta', '.fa', '.fna']:
        # Process single FASTA file
        visualizer.process_fasta_file(
            fasta_path=str(input_path),
            batch_size=args.batch_size,
            max_sequences=args.max_sequences,
            visualize=not args.no_visualization,
            save_predictions=not args.no_save
        )
    elif input_path.is_dir():
        # Process all FASTA files in directory
        fasta_files = list(input_path.glob("*.fasta")) + \
                     list(input_path.glob("*.fa")) + \
                     list(input_path.glob("*.fna"))
        
        if not fasta_files:
            logger.error(f"No FASTA files found in directory: {input_path}")
            sys.exit(1)
        
        for fasta_file in fasta_files:
            logger.info(f"Processing {fasta_file}")
            try:
                visualizer.process_fasta_file(
                    fasta_path=str(fasta_file),
                    batch_size=args.batch_size,
                    max_sequences=args.max_sequences,
                    visualize=not args.no_visualization,
                    save_predictions=not args.no_save
                )
            except Exception as e:
                logger.error(f"Failed to process {fasta_file}: {e}")
                continue
    else:
        logger.error(f"Invalid input: {input_path}. Must be a FASTA file or directory.")
        sys.exit(1)
    
    logger.info("Inference and visualization completed successfully")


if __name__ == "__main__":
    main()
