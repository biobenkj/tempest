#!/usr/bin/env python3
"""
Tempest CLI with modular subcommands.

This module provides the main command-line interface for Tempest,
organizing functionality into clear subcommands for simulation,
training, evaluation, and visualization.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Configure logging
def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


def simulate_command(args):
    """
    Simulate sequence reads for training and testing.
    
    This command generates synthetic sequence data based on the
    configuration parameters, optionally using PWM files for
    biologically-informed ACC sequence generation.
    """
    from tempest.utils import load_config
    from tempest.data import SequenceSimulator
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST SIMULATOR")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize simulator
    pwm_file = args.pwm or (
        config.pwm.pwm_file if hasattr(config, 'pwm') and hasattr(config.pwm, 'pwm_file') else None
    )
    
    if pwm_file:
        logger.info(f"Using PWM file: {pwm_file}")
    
    simulator = SequenceSimulator(config.simulation, pwm_file=pwm_file)
    
    # Generate sequences
    logger.info(f"Generating {args.num_sequences} sequences...")
    
    if args.split:
        # Generate training/validation split
        train_reads, val_reads = simulator.generate_train_val_split(
            total_reads=args.num_sequences,
            train_fraction=args.train_fraction
        )
        
        # Save to files
        output_dir = Path(args.output_dir or "./simulated_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = output_dir / "train_reads.txt"
        val_file = output_dir / "val_reads.txt"
        
        logger.info(f"Saving training reads to: {train_file}")
        with open(train_file, 'w') as f:
            for read in train_reads:
                f.write(f"{read.sequence}\t{','.join(read.labels)}\n")
        
        logger.info(f"Saving validation reads to: {val_file}")
        with open(val_file, 'w') as f:
            for read in val_reads:
                f.write(f"{read.sequence}\t{','.join(read.labels)}\n")
        
        logger.info(f"Generated {len(train_reads)} training and {len(val_reads)} validation reads")
        
    else:
        # Generate single dataset
        reads = simulator.generate_reads(args.num_sequences)
        
        # Save to file
        output_file = Path(args.output or "./simulated_reads.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving reads to: {output_file}")
        with open(output_file, 'w') as f:
            for read in reads:
                f.write(f"{read.sequence}\t{','.join(read.labels)}\n")
        
        logger.info(f"Generated {len(reads)} reads")
    
    logger.info("Simulation complete!")


def train_command(args):
    """
    Train a Tempest model on sequence data.
    
    Supports both standard and hybrid training modes.
    Hybrid mode provides improved robustness through
    invalid sequence handling and pseudo-labeling.
    """
    # Import here to avoid loading TF unless needed
    import warnings
    import numpy as np
    
    # Configure TensorFlow suppression
    if not args.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        warnings.filterwarnings("ignore")
    
    # Now import TF-dependent modules
    from tempest.main import prepare_data, train_standard, train_hybrid, setup_gpu
    from tempest.utils import load_config
    
    logger.info("="*80)
    logger.info(" " * 25 + "TEMPEST TRAINING PIPELINE")
    if args.hybrid:
        logger.info(" " * 25 + "(HYBRID ROBUSTNESS MODE)")
    logger.info("="*80)
    
    # Setup GPU
    setup_gpu()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override parameters if specified
    if args.output_dir:
        config.training.checkpoint_dir = args.output_dir
        logger.info(f"Output directory: {args.output_dir}")
    
    if args.epochs:
        config.training.epochs = args.epochs
        logger.info(f"Training epochs: {args.epochs}")
    
    if args.batch_size:
        config.model.batch_size = args.batch_size
        logger.info(f"Batch size: {args.batch_size}")
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        logger.info(f"Learning rate: {args.learning_rate}")
    
    # Determine PWM file
    pwm_file = args.pwm or (
        config.pwm.pwm_file if hasattr(config, 'pwm') and hasattr(config.pwm, 'pwm_file') else None
    )
    
    if pwm_file:
        logger.info(f"Using PWM file: {pwm_file}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads = prepare_data(config, pwm_file)
    
    # Train based on mode
    if args.hybrid:
        model = train_hybrid(config, train_reads, val_reads, args.unlabeled)
    else:
        model = train_standard(config, X_train, y_train, X_val, y_val, label_to_idx)
    
    # Summary
    print("\n" + "="*80)
    print(" " * 30 + "TRAINING COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
    print(f"  - model_best.h5: Best model based on validation loss")
    print(f"  - model_final.h5: Final trained model")
    print(f"  - training_history.csv: Training metrics")
    print("\n" + "="*80 + "\n")


def evaluate_command(args):
    """
    Evaluate a trained Tempest model on test data.
    
    This command loads a trained model and evaluates its
    performance on test sequences, providing detailed
    metrics and optionally saving predictions.
    """
    import warnings
    import numpy as np
    
    # Configure TensorFlow suppression
    if not args.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        warnings.filterwarnings("ignore")
    
    # Import TensorFlow-dependent modules
    import tensorflow as tf
    from tensorflow import keras
    from tempest.utils import load_config
    from tempest.data import reads_to_arrays
    from tempest.inference import predict_sequences, evaluate_predictions
    
    logger.info("="*80)
    logger.info(" " * 25 + "TEMPEST MODEL EVALUATION")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    model = keras.models.load_model(args.model, compile=False)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    
    # Parse test data format
    test_reads = []
    with open(args.test_data, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                seq, labels = line.split('\t')
                labels = labels.split(',')
            else:
                # Assume it's just sequences, will predict labels
                seq = line
                labels = None
            
            from tempest.data.simulator import AnnotatedRead
            test_reads.append(AnnotatedRead(
                sequence=seq,
                labels=labels if labels else [],
                acc_positions=[]
            ))
    
    logger.info(f"Loaded {len(test_reads)} test sequences")
    
    # Convert to arrays
    if test_reads[0].labels:
        X_test, y_test, label_to_idx = reads_to_arrays(test_reads)
        has_labels = True
    else:
        # Just sequences, no labels
        X_test = np.array([[ord(c) - ord('A') for c in read.sequence] for read in test_reads])
        y_test = None
        has_labels = False
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_test, batch_size=args.batch_size or 32, verbose=1)
    
    # Convert predictions to labels
    if predictions.ndim == 3:
        # Assuming shape (batch, seq_len, num_classes)
        predicted_labels = np.argmax(predictions, axis=-1)
    else:
        predicted_labels = predictions
    
    # Evaluate if we have ground truth
    if has_labels:
        logger.info("\nEvaluation Metrics:")
        logger.info("-" * 40)
        
        # Calculate accuracy
        accuracy = np.mean(predicted_labels == y_test)
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        
        # Per-class accuracy
        unique_labels = np.unique(y_test)
        for label in unique_labels:
            mask = y_test == label
            class_acc = np.mean(predicted_labels[mask] == y_test[mask])
            logger.info(f"  Class {label} Accuracy: {class_acc:.4f}")
    
    # Save predictions if requested
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving predictions to: {output_file}")
        with open(output_file, 'w') as f:
            for i, read in enumerate(test_reads):
                pred_labels = predicted_labels[i]
                # Convert numerical predictions back to label strings if possible
                pred_str = ','.join(map(str, pred_labels[:len(read.sequence)]))
                
                if has_labels:
                    true_str = ','.join(read.labels)
                    f.write(f"{read.sequence}\t{true_str}\t{pred_str}\n")
                else:
                    f.write(f"{read.sequence}\t{pred_str}\n")
        
        logger.info("Predictions saved successfully")
    
    # Generate confusion matrix if requested
    if args.confusion_matrix and has_labels:
        logger.info("\nGenerating confusion matrix...")
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        
        # Flatten for confusion matrix
        y_true_flat = y_test.flatten()
        y_pred_flat = predicted_labels.flatten()
        
        # Remove padding (zeros)
        mask = y_true_flat > 0
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(title='Confusion Matrix',
               xlabel='Predicted Label',
               ylabel='True Label')
        
        # Save plot
        cm_file = args.output.replace('.txt', '_confusion_matrix.png') if args.output else 'confusion_matrix.png'
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {cm_file}")
    
    logger.info("\nEvaluation complete!")


def visualize_command(args):
    """
    Visualize model predictions and training results.
    
    This command provides various visualization options including
    sequence annotations, training curves, attention weights,
    and prediction confidence plots.
    """
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Configure TensorFlow suppression
    if not args.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        warnings.filterwarnings("ignore")
    
    logger.info("="*80)
    logger.info(" " * 25 + "TEMPEST VISUALIZATION")
    logger.info("="*80)
    
    # Determine visualization type
    if args.type == 'training':
        # Visualize training history
        import pandas as pd
        
        logger.info(f"Loading training history from: {args.input}")
        history = pd.read_csv(args.input)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        axes[0].plot(history['epoch'], history['loss'], label='Training Loss', marker='o')
        if 'val_loss' in history.columns:
            axes[0].plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history.columns:
            axes[1].plot(history['epoch'], history['accuracy'], label='Training Accuracy', marker='o')
        if 'val_accuracy' in history.columns:
            axes[1].plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    elif args.type == 'predictions':
        # Visualize sequence predictions
        from tempest.visualization import plot_annotated_sequences
        from tempest.inference import visualize_predictions
        
        logger.info("Loading predictions for visualization...")
        
        # Load prediction file
        with open(args.input, 'r') as f:
            lines = f.readlines()
        
        # Parse predictions
        sequences = []
        true_labels = []
        pred_labels = []
        
        for line in lines:
            parts = line.strip().split('\t')
            sequences.append(parts[0])
            if len(parts) > 2:
                true_labels.append(parts[1].split(','))
                pred_labels.append(parts[2].split(','))
            else:
                pred_labels.append(parts[1].split(','))
                true_labels.append(None)
        
        # Create visualization
        if args.num_samples:
            sequences = sequences[:args.num_samples]
            pred_labels = pred_labels[:args.num_samples]
            if true_labels[0] is not None:
                true_labels = true_labels[:args.num_samples]
        
        logger.info(f"Visualizing {len(sequences)} sequences")
        
        # Create figure
        fig, axes = plt.subplots(len(sequences), 1, figsize=(15, 3 * len(sequences)))
        if len(sequences) == 1:
            axes = [axes]
        
        # Color map for labels
        label_colors = {
            '0': 'gray',
            '1': 'blue',
            '2': 'red',
            '3': 'green',
            '4': 'orange',
            '5': 'purple',
            'DONOR': 'blue',
            'ACC': 'red',
            'NON_ACC': 'green',
            'BACKGROUND': 'gray'
        }
        
        for i, (seq, pred) in enumerate(zip(sequences, pred_labels)):
            ax = axes[i]
            
            # Plot predictions as colored bars
            for j, label in enumerate(pred[:len(seq)]):
                color = label_colors.get(label, 'black')
                ax.barh(0, 1, left=j, height=0.3, color=color, alpha=0.7)
            
            # Add sequence text
            for j, char in enumerate(seq[:50]):  # Limit display to first 50 bases
                ax.text(j + 0.5, 0, char, ha='center', va='center', fontsize=8)
            
            # Add true labels if available
            if true_labels[0] is not None:
                true = true_labels[i]
                for j, label in enumerate(true[:len(seq)]):
                    color = label_colors.get(label, 'black')
                    ax.barh(-0.5, 1, left=j, height=0.3, color=color, alpha=0.7)
            
            ax.set_xlim(0, min(50, len(seq)))
            ax.set_ylim(-1, 0.5)
            ax.set_title(f'Sequence {i+1}')
            ax.set_xlabel('Position')
            ax.set_yticks([0, -0.5] if true_labels[0] is not None else [0])
            ax.set_yticklabels(['Predicted', 'True'] if true_labels[0] is not None else ['Predicted'])
        
        plt.tight_layout()
    
    elif args.type == 'attention':
        # Visualize attention weights (if model has attention layers)
        logger.info("Attention visualization requires model with attention layers")
        logger.info("This feature will be implemented based on specific model architecture")
        return
    
    elif args.type == 'embeddings':
        # Visualize learned embeddings
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.manifold import TSNE
        
        logger.info(f"Loading model from: {args.model}")
        model = keras.models.load_model(args.model, compile=False)
        
        # Extract embedding layer
        embedding_layer = None
        for layer in model.layers:
            if 'embedding' in layer.name.lower():
                embedding_layer = layer
                break
        
        if embedding_layer is None:
            logger.error("No embedding layer found in model")
            return
        
        # Get embedding weights
        embeddings = embedding_layer.get_weights()[0]
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # Reduce dimensions using t-SNE
        logger.info("Running t-SNE for dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(embeddings_2d)), cmap='viridis', s=50)
        ax.set_title('Learned Embeddings (t-SNE)')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Token ID')
        
        plt.tight_layout()
    
    else:
        logger.error(f"Unknown visualization type: {args.type}")
        return
    
    # Save figure
    if args.output:
        logger.info(f"Saving visualization to: {args.output}")
        plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    else:
        plt.show()
    
    logger.info("Visualization complete!")


def compare_command(args):
    """
    Compare multiple trained models on test data.
    
    This command evaluates and compares different model approaches
    (standard, hybrid, ensemble) to help select the best model
    for production use.
    """
    from tempest.compare import compare_models
    
    logger.info("="*80)
    logger.info(" " * 25 + "TEMPEST MODEL COMPARISON")
    logger.info("="*80)
    
    # Run comparison
    try:
        framework = compare_models(
            models_dir=args.models_dir,
            test_data_path=args.test_data,
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Print summary to console
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        comparison_df = framework.compare_models()
        print(comparison_df.to_string())
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print("\nKey outputs:")
        print(f"  - Model comparison table: {args.output_dir}/model_comparison.csv")
        print(f"  - Detailed report: {args.output_dir}/evaluation_report.json")
        print(f"  - Markdown summary: {args.output_dir}/evaluation_report.md")
        print(f"  - Visualizations: {args.output_dir}/comprehensive_evaluation.png")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise


def create_parser():
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='tempest',
        description='Tempest - Modular sequence annotation using length-constrained CRFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEMPEST OVERVIEW
================
Tempest is a deep learning framework for sequence annotation that combines:
  • Conditional Random Fields (CRFs) for structured prediction
  • Length constraints to enforce biologically meaningful segment sizes
  • Position Weight Matrix (PWM) priors for incorporating domain knowledge
  • Hybrid training modes for improved robustness

QUICK START
===========
1. Simulate training data:
   tempest simulate --config config.yaml --num-sequences 10000 --split

2. Train a model:
   tempest train --config config.yaml --epochs 50

3. Evaluate on test data:
   tempest evaluate --model model_final.h5 --test-data test_reads.txt

4. Visualize results:
   tempest visualize --type predictions --input predictions.txt --output viz.png

For detailed help on each command, use: tempest <command> --help
        """
    )
    
    # Add global arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (shows all warnings and TensorFlow output)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Use "tempest <command> --help" for command-specific help'
    )
    
    # ============ SIMULATE COMMAND ============
    parser_simulate = subparsers.add_parser(
        'simulate',
        help='Simulate sequence reads for training and testing',
        description='Generate synthetic sequence data with labels for model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Generate 10000 sequences with default parameters
  tempest simulate --config config.yaml --num-sequences 10000
  
  # Generate train/validation split with PWM
  tempest simulate --config config.yaml --num-sequences 20000 --split --pwm acc_pwm.txt
  
  # Custom output directory and train fraction
  tempest simulate --config config.yaml -n 15000 --split --train-fraction 0.9 -o ./data
        """
    )
    parser_simulate.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser_simulate.add_argument(
        '--num-sequences', '-n',
        type=int,
        default=10000,
        help='Number of sequences to generate (default: 10000)'
    )
    parser_simulate.add_argument(
        '--pwm',
        type=str,
        help='Path to PWM file for ACC generation'
    )
    parser_simulate.add_argument(
        '--output', '-o',
        type=str,
        help='Output file or directory for sequences'
    )
    parser_simulate.add_argument(
        '--split',
        action='store_true',
        help='Generate train/validation split'
    )
    parser_simulate.add_argument(
        '--train-fraction',
        type=float,
        default=0.8,
        help='Fraction of data for training when using --split (default: 0.8)'
    )
    parser_simulate.add_argument(
        '--output-dir',
        type=str,
        help='Output directory when using --split'
    )
    parser_simulate.set_defaults(func=simulate_command)
    
    # ============ TRAIN COMMAND ============
    parser_train = subparsers.add_parser(
        'train',
        help='Train a Tempest model on sequence data',
        description='Train models using standard or hybrid training modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TRAINING MODES:
  Standard: Basic supervised training with CRF layers
  Hybrid: Advanced training with invalid sequence handling and pseudo-labeling
  
EXAMPLES:
  # Standard training
  tempest train --config config.yaml --epochs 100
  
  # Hybrid training with PWM
  tempest train --config hybrid_config.yaml --hybrid --pwm acc_pwm.txt
  
  # Custom parameters
  tempest train --config config.yaml --epochs 50 --batch-size 64 --learning-rate 0.0001
        """
    )
    parser_train.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser_train.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid robustness training mode'
    )
    parser_train.add_argument(
        '--pwm',
        type=str,
        help='Path to PWM file for ACC generation'
    )
    parser_train.add_argument(
        '--unlabeled',
        type=str,
        help='Path to unlabeled FASTQ file for pseudo-labeling (hybrid mode only)'
    )
    parser_train.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for model checkpoints'
    )
    parser_train.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser_train.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Training batch size (overrides config)'
    )
    parser_train.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser_train.set_defaults(func=train_command)
    
    # ============ EVALUATE COMMAND ============
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained model on test data',
        description='Evaluate model performance and generate predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INPUT FORMAT:
  Test data should be tab-delimited with format:
  - With labels: SEQUENCE<TAB>LABEL1,LABEL2,...
  - Without labels: SEQUENCE
  
EXAMPLES:
  # Evaluate with ground truth labels
  tempest evaluate --model model_final.h5 --test-data test_reads.txt
  
  # Generate predictions and confusion matrix
  tempest evaluate --model model.h5 --test-data test.txt --output preds.txt --confusion-matrix
  
  # Predict on unlabeled sequences
  tempest evaluate --model model.h5 --test-data sequences.txt --output predictions.txt
        """
    )
    parser_evaluate.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    parser_evaluate.add_argument(
        '--test-data', '-t',
        type=str,
        required=True,
        help='Path to test data file'
    )
    parser_evaluate.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for predictions'
    )
    parser_evaluate.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for prediction (default: 32)'
    )
    parser_evaluate.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Generate and save confusion matrix'
    )
    parser_evaluate.set_defaults(func=evaluate_command)
    
    # ============ VISUALIZE COMMAND ============
    parser_visualize = subparsers.add_parser(
        'visualize',
        help='Visualize model predictions and training results',
        description='Generate various visualizations of model behavior',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VISUALIZATION TYPES:
  training: Plot training/validation loss and accuracy curves
  predictions: Visualize predicted vs true labels on sequences
  attention: Show attention weights (if model has attention)
  embeddings: Visualize learned embeddings using t-SNE
  
EXAMPLES:
  # Plot training history
  tempest visualize --type training --input training_history.csv --output curves.png
  
  # Visualize sequence predictions
  tempest visualize --type predictions --input predictions.txt --num-samples 5 -o viz.png
  
  # Plot learned embeddings
  tempest visualize --type embeddings --model model.h5 --output embeddings.png
        """
    )
    parser_visualize.add_argument(
        '--type', '-t',
        type=str,
        required=True,
        choices=['training', 'predictions', 'attention', 'embeddings'],
        help='Type of visualization to generate'
    )
    parser_visualize.add_argument(
        '--input', '-i',
        type=str,
        help='Input file (training history CSV or predictions file)'
    )
    parser_visualize.add_argument(
        '--model', '-m',
        type=str,
        help='Model file for embeddings or attention visualization'
    )
    parser_visualize.add_argument(
        '--output', '-o',
        type=str,
        help='Output image file (PNG, PDF, etc.)'
    )
    parser_visualize.add_argument(
        '--num-samples', '-n',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    parser_visualize.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for output image (default: 150)'
    )
    parser_visualize.set_defaults(func=visualize_command)
    
    # ============ COMPARE COMMAND ============
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare multiple trained models',
        description='Evaluate and compare different model approaches (standard, hybrid, ensemble)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMPARISON METRICS:
  - Basic metrics: accuracy, precision, recall, F1
  - Segment-level performance per label type
  - Length constraint satisfaction rates
  - Robustness to errors (missing/duplicated segments)
  - Computational efficiency (inference time)
  - Ensemble-specific metrics (uncertainty, agreement)

EXAMPLES:
  # Compare models in a directory
  tempest compare --models-dir ./trained_models --test-data test_data.pkl
  
  # Compare with custom configuration
  tempest compare --models-dir ./models --test-data test.pkl --config config.yaml
  
  # Save results to specific directory
  tempest compare --models-dir ./models --test-data test.pkl -o ./comparison_results
  
  # Compare specific model files
  tempest compare --models model1.h5,model2.h5,ensemble/ --test-data test.pkl
        """
    )
    parser_compare.add_argument(
        '--models-dir',
        type=str,
        default='./trained_models',
        help='Directory containing trained models to compare'
    )
    parser_compare.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of model files/directories (alternative to --models-dir)'
    )
    parser_compare.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data file (pickled X_test, y_test)'
    )
    parser_compare.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (uses model config if not specified)'
    )
    parser_compare.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory for comparison results'
    )
    parser_compare.add_argument(
        '--metrics',
        type=str,
        help='Comma-separated list of metrics to evaluate (default: all)'
    )
    parser_compare.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    parser_compare.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating markdown report'
    )
    parser_compare.set_defaults(func=compare_command)
    
    return parser


def main():
    """Main entry point for the CLI."""
    # Create parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check if a command was specified
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute the command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == '__main__':
    main()
