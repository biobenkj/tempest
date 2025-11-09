#!/usr/bin/env python3
"""
Tempest CLI with modular subcommands

This module provides the main command-line interface for Tempest,
organizing functionality into clear subcommands for simulation,
training, evaluation, visualization, demultiplexing, and model combination.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json

# Configure logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

logger = logging.getLogger(__name__)


def simulate_command(args):
    """
    Simulate sequence reads for training and testing.
    """
    from tempest.utils import load_config
    from tempest.data import SequenceSimulator
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST SIMULATOR")
    logger.info("="*80)
    
    # Load base configuration
    config = load_config(args.config)
    
    # Override simulation parameters from command line
    if args.num_sequences:
        config.simulation.num_sequences = args.num_sequences
    if args.seed:
        config.simulation.random_seed = args.seed
    
    # Create simulator
    simulator = SequenceSimulator(config)
    
    # Generate sequences
    if args.split:
        # Generate train/validation split
        train_fraction = args.train_fraction
        logger.info(f"Generating {config.simulation.num_sequences} sequences")
        logger.info(f"Split: {train_fraction:.0%} train, {1-train_fraction:.0%} validation")
        
        train_data, val_data = simulator.generate_split(
            num_sequences=config.simulation.num_sequences,
            train_fraction=train_fraction
        )
        
        # Save to output directory
        output_dir = Path(args.output_dir or "./data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = output_dir / "train.txt"
        val_file = output_dir / "val.txt"
        
        simulator.save_sequences(train_data, train_file)
        simulator.save_sequences(val_data, val_file)
        
        logger.info(f"Train data saved to: {train_file}")
        logger.info(f"Validation data saved to: {val_file}")
    else:
        # Generate single dataset
        logger.info(f"Generating {config.simulation.num_sequences} sequences")
        sequences = simulator.generate(config.simulation.num_sequences)
        
        output_file = Path(args.output or "sequences.txt")
        simulator.save_sequences(sequences, output_file)
        logger.info(f"Sequences saved to: {output_file}")
    
    logger.info("Simulation complete!")


def train_command(args):
    """
    Train Tempest models with standard, hybrid, or ensemble approaches.
    """
    from tempest.main import main as train_main
    from tempest.utils import load_config
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST TRAINER")
    logger.info("="*80)
    
    # Convert argparse namespace to dict for train_main
    train_args = vars(args).copy()
    
    # Call the training main function
    train_main(train_args)


def evaluate_command(args):
    """
    Evaluate trained models on test data.
    """
    from tempest.compare.evaluate import ModelEvaluator
    from tempest.utils import load_config
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST EVALUATOR")
    logger.info("="*80)
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config=config
    )
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = evaluator.load_test_data(args.test_data)
    
    # Evaluate
    results = evaluator.evaluate(
        test_data,
        batch_size=args.batch_size,
        per_segment_metrics=args.per_segment_metrics,
        confusion_matrix=args.confusion_matrix
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("\nEvaluation Results:")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    if 'segment_metrics' in results:
        logger.info("\nPer-segment F1 scores:")
        for segment, metrics in results['segment_metrics'].items():
            logger.info(f"  {segment}: {metrics['f1_score']:.4f}")


def visualize_command(args):
    """
    Create visualizations from training or evaluation results.
    """
    from tempest.visualization import create_visualizations
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST VISUALIZER")
    logger.info("="*80)
    
    # Create visualizations based on input type
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.history:
        # Visualize training history
        logger.info(f"Loading training history from: {args.history}")
        create_visualizations(
            history_file=args.history,
            output_dir=output_dir,
            types=['loss', 'accuracy']
        )
    
    if args.confusion_matrix:
        # Visualize confusion matrix
        logger.info(f"Loading confusion matrix from: {args.confusion_matrix}")
        create_visualizations(
            confusion_file=args.confusion_matrix,
            output_dir=output_dir,
            types=['confusion']
        )
    
    if args.predictions:
        # Visualize predictions
        logger.info(f"Loading predictions from: {args.predictions}")
        create_visualizations(
            predictions_file=args.predictions,
            output_dir=output_dir,
            types=['predictions'],
            num_examples=args.num_examples
        )
    
    logger.info(f"Visualizations saved to: {output_dir}")


def compare_command(args):
    """
    Compare multiple trained models.
    """
    from tempest.compare import ModelComparator
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST COMPARATOR")
    logger.info("="*80)
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load models
    model_paths = []
    if args.models:
        model_paths = args.models
    elif args.models_dir:
        models_dir = Path(args.models_dir)
        model_paths = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    
    logger.info(f"Comparing {len(model_paths)} models")
    
    # Compare models
    results = comparator.compare(
        model_paths=model_paths,
        test_data=args.test_data,
        metrics=args.metrics or ['accuracy', 'f1_score', 'precision', 'recall']
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Comparison saved to: {comparison_file}")
    
    # Print summary
    logger.info("\nModel Comparison Results:")
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


def combine_command(args):
    """
    Combine multiple models using ensemble methods.
    """
    from tempest.inference.combine import ModelCombiner
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST COMBINER")
    logger.info("="*80)
    
    # Get model paths
    model_paths = []
    if args.models:
        model_paths = args.models
    elif args.models_dir:
        models_dir = Path(args.models_dir)
        model_paths = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    
    logger.info(f"Combining {len(model_paths)} models using {args.method}")
    
    # Create combiner
    combiner = ModelCombiner(
        model_paths=model_paths,
        method=args.method,
        validation_data=args.validation_data
    )
    
    # Combine models
    if args.method == 'bma':
        ensemble_model = combiner.combine_bma(
            approximation=args.approximation,
            temperature=args.temperature
        )
    elif args.method == 'weighted_average':
        ensemble_model = combiner.combine_weighted()
    else:  # voting
        ensemble_model = combiner.combine_voting()
    
    # Save ensemble model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_file = output_dir / "ensemble_model.pkl"
    combiner.save_ensemble(ensemble_model, ensemble_file)
    
    logger.info(f"Ensemble model saved to: {ensemble_file}")


def demux_command(args):
    """
    Demultiplex FASTQ files using trained model.
    """
    from tempest.demux import ReadDemultiplexer, BarcodeWhitelist
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST DEMUX")
    logger.info("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load whitelists if provided
    whitelists = BarcodeWhitelist.from_files(
        cbc_file=args.whitelist_cbc,
        i5_file=args.whitelist_i5,
        i7_file=args.whitelist_i7
    )
    
    # Create demultiplexer
    demux = ReadDemultiplexer(
        model_path=args.model,
        whitelists=whitelists,
        max_edit_distance=args.max_edit_distance,
        batch_size=args.batch_size
    )
    
    # Process input
    if args.input_dir:
        # Process directory of FASTQs
        results = demux.process_directory(
            args.input_dir,
            str(output_dir),
            file_pattern=args.file_pattern
        )
    else:
        # Process single FASTQ file
        results = demux.process_fastq_file(
            args.input,
            str(output_dir)
        )
    
    # Generate visualizations if requested
    if args.plot_metrics and 'statistics' in results:
        try:
            from tempest.visualization import plot_demux_metrics
            plot_demux_metrics(results, output_dir)
            logger.info(f"Visualizations saved to: {output_dir}")
        except ImportError:
            logger.warning("Visualization module not available")
    
    logger.info("\nDemultiplexing Complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print key metrics
    if 'statistics' in results:
        stats = results['statistics']
        logger.info("\nKey Metrics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")


def create_parser():
    """Create the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog='tempest',
        description='Tempest - Advanced sequence annotation with length-constrained CRFs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log to file in addition to console'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose output'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # ============ SIMULATE COMMAND ============
    parser_simulate = subparsers.add_parser(
        'simulate',
        help='Generate synthetic sequence data',
        description='Generate synthetic sequence reads with configurable architecture'
    )
    parser_simulate.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser_simulate.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for generated sequences'
    )
    parser_simulate.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for split datasets'
    )
    parser_simulate.add_argument(
        '--num-sequences', '-n',
        type=int,
        help='Number of sequences to generate'
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
        help='Fraction of data for training (default: 0.8)'
    )
    parser_simulate.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser_simulate.set_defaults(func=simulate_command)
    
    # ============ TRAIN COMMAND ============
    parser_train = subparsers.add_parser(
        'train',
        help='Train Tempest models',
        description='Train models with standard, hybrid, or ensemble approaches'
    )
    parser_train.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser_train.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Output directory for trained models'
    )
    parser_train.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser_train.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    parser_train.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    parser_train.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid training with constraints'
    )
    parser_train.add_argument(
        '--ensemble',
        action='store_true',
        help='Train ensemble of models'
    )
    parser_train.add_argument(
        '--num-models',
        type=int,
        default=3,
        help='Number of models for ensemble'
    )
    parser_train.set_defaults(func=train_command)
    
    # ============ EVALUATE COMMAND ============
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained models',
        description='Comprehensive model evaluation with multiple metrics'
    )
    parser_evaluate.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser_evaluate.add_argument(
        '--test-data', '-t',
        type=str,
        required=True,
        help='Path to test data'
    )
    parser_evaluate.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (optional)'
    )
    parser_evaluate.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser_evaluate.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Evaluation batch size'
    )
    parser_evaluate.add_argument(
        '--per-segment-metrics',
        action='store_true',
        help='Compute per-segment metrics'
    )
    parser_evaluate.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Generate confusion matrix'
    )
    parser_evaluate.set_defaults(func=evaluate_command)
    
    # ============ VISUALIZE COMMAND ============
    parser_visualize = subparsers.add_parser(
        'visualize',
        help='Create visualizations',
        description='Generate plots and visualizations from results'
    )
    parser_visualize.add_argument(
        '--history',
        type=str,
        help='Training history file'
    )
    parser_visualize.add_argument(
        '--confusion-matrix',
        type=str,
        help='Confusion matrix file'
    )
    parser_visualize.add_argument(
        '--predictions',
        type=str,
        help='Predictions file'
    )
    parser_visualize.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./visualizations',
        help='Output directory for plots'
    )
    parser_visualize.add_argument(
        '--num-examples',
        type=int,
        default=10,
        help='Number of example predictions to plot'
    )
    parser_visualize.set_defaults(func=visualize_command)
    
    # ============ COMPARE COMMAND ============
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare multiple models',
        description='Compare performance of multiple trained models'
    )
    parser_compare.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='List of model paths to compare'
    )
    parser_compare.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing models to compare'
    )
    parser_compare.add_argument(
        '--test-data', '-t',
        type=str,
        required=True,
        help='Test data for comparison'
    )
    parser_compare.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./comparison_results',
        help='Output directory for results'
    )
    parser_compare.add_argument(
        '--metrics',
        nargs='+',
        type=str,
        help='Metrics to compare'
    )
    parser_compare.set_defaults(func=compare_command)
    
    # ============ COMBINE COMMAND ============
    parser_combine = subparsers.add_parser(
        'combine',
        help='Combine models using ensemble methods',
        description='Create ensemble models using various combination methods'
    )
    parser_combine.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='List of model paths to combine'
    )
    parser_combine.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing models to combine'
    )
    parser_combine.add_argument(
        '--method',
        type=str,
        choices=['bma', 'weighted_average', 'voting'],
        default='bma',
        help='Combination method'
    )
    parser_combine.add_argument(
        '--validation-data',
        type=str,
        help='Validation data for weighting'
    )
    parser_combine.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./ensemble_models',
        help='Output directory for ensemble'
    )
    parser_combine.add_argument(
        '--approximation',
        type=str,
        choices=['bic', 'laplace', 'variational'],
        default='laplace',
        help='BMA approximation method'
    )
    parser_combine.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for BMA'
    )
    parser_combine.set_defaults(func=combine_command)
    
    # ============ DEMUX COMMAND ============
    parser_demux = subparsers.add_parser(
        'demux',
        help='Demultiplex FASTQ files',
        description='Demultiplex reads using trained model and barcode whitelists'
    )
    parser_demux.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    # Input options (mutually exclusive)
    input_group = parser_demux.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input FASTQ file'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing FASTQ files'
    )
    
    parser_demux.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./demux_results',
        help='Output directory for results'
    )
    
    # Whitelist options
    parser_demux.add_argument(
        '--whitelist-cbc',
        type=str,
        help='CBC whitelist file'
    )
    parser_demux.add_argument(
        '--whitelist-i5',
        type=str,
        help='i5 whitelist file'
    )
    parser_demux.add_argument(
        '--whitelist-i7',
        type=str,
        help='i7 whitelist file'
    )
    
    # Processing options
    parser_demux.add_argument(
        '--max-edit-distance',
        type=int,
        default=2,
        help='Maximum edit distance for barcode correction'
    )
    parser_demux.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser_demux.add_argument(
        '--file-pattern',
        type=str,
        default='*.fastq*',
        help='File pattern for directory processing'
    )
    parser_demux.add_argument(
        '--plot-metrics',
        action='store_true',
        help='Generate visualization plots'
    )
    parser_demux.set_defaults(func=demux_command)
    
    return parser


def main():
    """Main entry point for the CLI."""
    # Create parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        args.log_level,
        args.log_file if hasattr(args, 'log_file') else None
    )
    
    # Enable debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
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
