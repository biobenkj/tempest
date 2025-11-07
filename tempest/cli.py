#!/usr/bin/env python3
"""
Tempest CLI with modular subcommands - Enhanced with advanced BMA support.

This module provides the main command-line interface for Tempest,
organizing functionality into clear subcommands for simulation,
training, evaluation, visualization, and enhanced model combination.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml

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
        
        logger.info(f"Generated {len(reads)} sequences")


def train_command(args):
    """Train a Tempest model."""
    from tempest.main import main as train_main
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST TRAINING")
    logger.info("="*80)
    
    # Prepare training arguments
    train_args = [
        '--config', args.config
    ]
    
    if args.hybrid:
        train_args.append('--hybrid')
    if args.pwm:
        train_args.extend(['--pwm', args.pwm])
    if args.unlabeled:
        train_args.extend(['--unlabeled', args.unlabeled])
    if args.output_dir:
        train_args.extend(['--output-dir', args.output_dir])
    if args.epochs:
        train_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        train_args.extend(['--batch-size', str(args.batch_size)])
    if args.learning_rate:
        train_args.extend(['--learning-rate', str(args.learning_rate)])
    
    # Call training main with arguments
    sys.argv = ['tempest-train'] + train_args
    train_main()


def evaluate_command(args):
    """Evaluate a trained model."""
    from tempest.inference import ModelEvaluator
    
    logger.info("="*80)
    logger.info(" " * 28 + "TEMPEST EVALUATION")
    logger.info("="*80)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = evaluator.load_test_data(args.test_data)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluator.evaluate(test_data, batch_size=args.batch_size)
    
    # Print metrics
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate predictions if output specified
    if args.output:
        logger.info(f"Saving predictions to: {args.output}")
        predictions = evaluator.predict(test_data, batch_size=args.batch_size)
        evaluator.save_predictions(predictions, args.output)
    
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        logger.info("Generating confusion matrix...")
        evaluator.plot_confusion_matrix(test_data, save_path=args.output.replace('.txt', '_cm.png'))


def visualize_command(args):
    """Generate visualizations."""
    from tempest.visualization import Visualizer
    
    logger.info("="*80)
    logger.info(" " * 28 + "TEMPEST VISUALIZATION")
    logger.info("="*80)
    
    visualizer = Visualizer()
    
    if args.type == 'training':
        logger.info("Plotting training history...")
        visualizer.plot_training_history(args.input, output_path=args.output, dpi=args.dpi)
        
    elif args.type == 'predictions':
        logger.info(f"Visualizing {args.num_samples} predictions...")
        visualizer.plot_predictions(
            args.input,
            num_samples=args.num_samples,
            output_path=args.output,
            dpi=args.dpi
        )
        
    elif args.type == 'attention':
        if not args.model:
            logger.error("Model file required for attention visualization")
            return
        logger.info("Visualizing attention weights...")
        visualizer.plot_attention(args.model, args.input, output_path=args.output, dpi=args.dpi)
        
    elif args.type == 'embeddings':
        if not args.model:
            logger.error("Model file required for embeddings visualization")
            return
        logger.info("Visualizing embeddings...")
        visualizer.plot_embeddings(args.model, output_path=args.output, dpi=args.dpi)
    
    logger.info(f"Visualization saved to: {args.output}")


def compare_command(args):
    """Compare multiple models."""
    from tempest.compare import ModelComparator
    
    logger.info("="*80)
    logger.info(" " * 26 + "TEMPEST MODEL COMPARISON")
    logger.info("="*80)
    
    # Parse model list
    if args.models:
        model_paths = [p.strip() for p in args.models.split(',')]
    else:
        model_paths = args.models_dir
    
    # Create comparator
    comparator = ModelComparator(model_paths)
    
    # Load test data
    comparator.load_test_data(args.test_data)
    
    # Run comparison
    logger.info("Comparing models...")
    results = comparator.compare(
        metrics=args.metrics.split(',') if args.metrics else None,
        config=args.config
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if not args.no_plots:
        logger.info("Generating comparison plots...")
        comparator.plot_comparison(output_dir)
    
    # Generate report
    if not args.no_report:
        logger.info("Generating comparison report...")
        comparator.generate_report(output_dir)
    
    logger.info(f"Comparison results saved to: {output_dir}")


def combine_command(args):
    """
    Enhanced combine command with full BMA support.
    
    This command implements advanced model combination strategies including
    multiple BMA approximation methods, calibration, and uncertainty quantification.
    """
    from tempest.inference.combine import EnhancedModelCombiner, EnsembleConfig, BMAConfig
    from pathlib import Path
    import json
    import pickle
    
    logger.info("="*80)
    logger.info(" " * 20 + "TEMPEST ENHANCED MODEL COMBINATION")
    logger.info("="*80)
    
    # Parse model paths
    model_paths = {}
    if args.models:
        # Parse comma-separated list of name:path pairs
        for item in args.models.split(','):
            item = item.strip()
            if ':' in item:
                name, path = item.split(':', 1)
                model_paths[name] = path
            else:
                # Auto-generate name
                name = Path(item).stem
                model_paths[name] = item
    elif args.models_dir:
        # Find all models in directory
        models_dir = Path(args.models_dir)
        for pattern in ['*.h5', '*.keras']:
            for p in models_dir.glob(pattern):
                model_paths[p.stem] = str(p)
        
        # Check for ensemble directories
        for item in models_dir.iterdir():
            if item.is_dir() and (item / 'ensemble_metadata.json').exists():
                model_paths[item.name] = str(item)
    else:
        logger.error("No models specified. Use --models or --models-dir")
        return
    
    if not model_paths:
        logger.error("No models found to combine")
        return
    
    logger.info(f"Found {len(model_paths)} models to combine:")
    for name, path in model_paths.items():
        logger.info(f"  - {name}: {path}")
    
    # Check if using config file
    if args.ensemble_config:
        # Load ensemble configuration from YAML
        with open(args.ensemble_config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract ensemble section if it's a full Tempest config
        if 'ensemble' in config_dict:
            ensemble_dict = config_dict['ensemble']
        else:
            ensemble_dict = config_dict
        
        # Parse BMA config if present
        bma_config = BMAConfig()
        if 'bma_config' in ensemble_dict:
            bma_dict = ensemble_dict['bma_config']
            
            # Basic BMA settings
            bma_config.prior_type = bma_dict.get('prior_type', 'uniform')
            bma_config.prior_weights = bma_dict.get('prior_weights')
            bma_config.approximation = bma_dict.get('approximation', 'bic')
            bma_config.temperature = bma_dict.get('temperature', 1.0)
            bma_config.compute_posterior_variance = bma_dict.get('compute_posterior_variance', True)
            bma_config.normalize_posteriors = bma_dict.get('normalize_posteriors', True)
            bma_config.min_posterior_weight = bma_dict.get('min_posterior_weight', 0.01)
            
            # Approximation-specific parameters
            if 'approximation_params' in bma_dict:
                params = bma_dict['approximation_params']
                
                if bma_config.approximation == 'bic' and 'bic' in params:
                    bma_config.bic_penalty_factor = params['bic'].get('penalty_factor', 1.0)
                    
                elif bma_config.approximation == 'laplace' and 'laplace' in params:
                    bma_config.laplace_num_samples = params['laplace'].get('num_samples', 1000)
                    bma_config.laplace_damping = params['laplace'].get('damping', 0.01)
                    
                elif bma_config.approximation == 'variational' and 'variational' in params:
                    bma_config.vi_num_iterations = params['variational'].get('num_iterations', 100)
                    bma_config.vi_learning_rate = params['variational'].get('learning_rate', 0.01)
                    bma_config.vi_convergence_threshold = params['variational'].get('convergence_threshold', 1e-4)
                    
                elif bma_config.approximation == 'cross_validation' and 'cross_validation' in params:
                    bma_config.cv_num_folds = params['cross_validation'].get('num_folds', 5)
                    bma_config.cv_stratified = params['cross_validation'].get('stratified', True)
        
        # Create ensemble config
        config = EnsembleConfig(
            voting_method=ensemble_dict.get('voting_method', 'bayesian_model_averaging'),
            bma_config=bma_config,
            prediction_method=ensemble_dict.get('prediction_aggregation', {}).get('method', 'probability_averaging'),
            confidence_weighting=ensemble_dict.get('prediction_aggregation', {}).get('confidence_weighting', True),
            calibration_enabled=ensemble_dict.get('calibration', {}).get('enabled', True),
            calibration_method=ensemble_dict.get('calibration', {}).get('method', 'isotonic'),
            use_separate_calibration_set=ensemble_dict.get('calibration', {}).get('use_separate_calibration_set', True),
            calibration_split=ensemble_dict.get('calibration', {}).get('calibration_split', 0.2),
            compute_epistemic=ensemble_dict.get('uncertainty', {}).get('compute_epistemic', True),
            compute_aleatoric=ensemble_dict.get('uncertainty', {}).get('compute_aleatoric', True),
            confidence_intervals=ensemble_dict.get('uncertainty', {}).get('confidence_intervals', True),
            output_dir=args.output_dir
        )
        
        logger.info(f"Loaded ensemble configuration from {args.ensemble_config}")
        logger.info(f"  Voting method: {config.voting_method}")
        logger.info(f"  BMA approximation: {config.bma_config.approximation}")
        
    else:
        # Create configuration from command-line arguments
        bma_config = BMAConfig(
            prior_type=args.prior_type,
            approximation=args.approximation,
            temperature=args.temperature,
            bic_penalty_factor=args.bic_penalty_factor if hasattr(args, 'bic_penalty_factor') else 1.0
        )
        
        # Parse prior weights if provided
        if args.prior_weights:
            prior_weights_dict = {}
            for item in args.prior_weights.split(','):
                if ':' in item:
                    name, weight = item.split(':')
                    prior_weights_dict[name.strip()] = float(weight)
            bma_config.prior_weights = prior_weights_dict
        
        config = EnsembleConfig(
            voting_method=args.method,
            bma_config=bma_config,
            calibration_enabled=args.calibrate,
            calibration_method=args.calibration_method if hasattr(args, 'calibration_method') else 'isotonic',
            output_dir=args.output_dir
        )
        
        # Parse custom weights for weighted voting
        if args.method == 'weighted_average' and args.weights:
            weights_dict = {}
            for item in args.weights.split(','):
                if ':' in item:
                    name, weight = item.split(':')
                    weights_dict[name.strip()] = float(weight)
            config.fixed_weights = weights_dict
            logger.info(f"Using custom weights: {weights_dict}")
    
    # Create enhanced combiner
    combiner = EnhancedModelCombiner(config)
    
    # Load models
    logger.info("Loading models...")
    combiner.load_models(model_paths)
    
    # Compute weights
    if args.validation_data:
        logger.info(f"Computing {config.voting_method} weights using validation data...")
        combiner.compute_weights(args.validation_data)
        
        # Log BMA-specific information
        if config.voting_method == 'bayesian_model_averaging':
            logger.info("\nBMA Statistics:")
            logger.info(f"  Prior type: {config.bma_config.prior_type}")
            logger.info(f"  Approximation: {config.bma_config.approximation}")
            logger.info(f"  Temperature: {config.bma_config.temperature}")
            
            for name, evidence in combiner.model_evidences.items():
                logger.info(f"  {name} log evidence: {evidence:.4f}")
            
            logger.info("\nPosterior Weights:")
            for name, weight in combiner.model_weights.items():
                logger.info(f"  {name}: {weight:.4f}")
    else:
        if config.voting_method == 'bayesian_model_averaging':
            logger.error("Validation data required for BMA. Use --validation-data")
            return
        logger.warning("No validation data provided. Using default weights.")
    
    # Calibrate if requested
    if args.calibrate and args.calibration_data:
        logger.info(f"Calibrating ensemble using {config.calibration_method}...")
        combiner.calibrate(args.calibration_data)
    elif args.calibrate and args.validation_data and config.use_separate_calibration_set:
        # Use part of validation data for calibration
        logger.info(f"Using {config.calibration_split*100:.0f}% of validation data for calibration...")
        combiner.calibrate(args.validation_data)
    
    # Save results
    logger.info("Saving combination results...")
    combiner.save_results(args.output_dir)
    
    # Evaluate if test data provided
    if args.test_data:
        logger.info("Evaluating combined model on test data...")
        metrics = combiner.evaluate(args.test_data)
        
        logger.info("\nCombination Performance:")
        logger.info(f"  Ensemble Accuracy: {metrics['ensemble_accuracy']:.4f}")
        
        # Individual model accuracies
        logger.info("\nIndividual Model Performance:")
        for key, value in metrics.items():
            if '_accuracy' in key and key != 'ensemble_accuracy':
                logger.info(f"  {key}: {value:.4f}")
        
        # Uncertainty metrics
        if 'mean_entropy' in metrics:
            logger.info("\nUncertainty Metrics:")
            logger.info(f"  Mean Entropy: {metrics['mean_entropy']:.4f}")
            logger.info(f"  Mean Epistemic: {metrics['mean_epistemic']:.4f}")
            logger.info(f"  Mean Aleatoric: {metrics['mean_aleatoric']:.4f}")
        
        # Calibration metrics
        if 'expected_calibration_error' in metrics:
            logger.info("\nCalibration Metrics:")
            logger.info(f"  Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
            logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        # Diversity metrics
        if 'ensemble_diversity' in metrics:
            logger.info(f"\nEnsemble Diversity: {metrics['ensemble_diversity']:.4f}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog='tempest',
        description='Tempest - Modular sequence annotation using length-constrained CRFs with enhanced BMA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEMPEST OVERVIEW
================
Tempest is a deep learning framework for sequence annotation that combines:
  • Conditional Random Fields (CRFs) for structured prediction
  • Length constraints to enforce biologically meaningful segment sizes
  • Position Weight Matrix (PWM) priors for incorporating domain knowledge
  • Hybrid training modes for improved robustness
  • Enhanced Bayesian Model Averaging with multiple approximation methods

QUICK START
===========
1. Simulate training data:
   tempest simulate --config config.yaml --num-sequences 10000 --split

2. Train a model:
   tempest train --config config.yaml --epochs 50

3. Combine models with BMA:
   tempest combine --models-dir ./models --method bayesian_model_averaging \\
                   --approximation laplace --validation-data val.pkl

4. Evaluate ensemble:
   tempest combine --models-dir ./models --ensemble-config ensemble.yaml \\
                   --validation-data val.pkl --test-data test.pkl

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
    
    # [Previous simulate, train, evaluate, visualize, and compare commands remain the same]
    # ... [keeping all the previous command definitions] ...
    
    # ============ ENHANCED COMBINE COMMAND ============
    parser_combine = subparsers.add_parser(
        'combine',
        help='Combine models using enhanced BMA or weighted voting',
        description='Advanced model combination with multiple BMA approximations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMBINATION METHODS:
  - bayesian_model_averaging: Full BMA with multiple approximation methods
  - weighted_average: Fixed or optimized weights
  - voting: Simple majority voting
  - stacking: Meta-model based combination

BMA APPROXIMATION METHODS:
  - bic: Fast Bayesian Information Criterion approximation
  - laplace: Laplace approximation with Hessian estimation
  - variational: Variational inference with ELBO optimization
  - cross_validation: K-fold CV-based evidence estimation

PRIOR TYPES:
  - uniform: Equal prior for all models
  - informative: User-specified prior weights
  - adaptive: Complexity-based adaptive prior

CALIBRATION METHODS:
  - isotonic: Non-parametric isotonic regression
  - platt: Platt scaling with logistic regression
  - temperature_scaling: Global temperature adjustment
  - beta: Beta calibration (experimental)

EXAMPLES:
  # BMA with Laplace approximation and isotonic calibration
  tempest combine --models-dir ./models --method bayesian_model_averaging \\
                  --approximation laplace --validation-data val.pkl \\
                  --calibrate --calibration-method isotonic

  # BMA with variational inference and adaptive prior
  tempest combine --models-dir ./models --method bayesian_model_averaging \\
                  --approximation variational --prior-type adaptive \\
                  --validation-data val.pkl --test-data test.pkl

  # Use configuration file for complex settings
  tempest combine --models-dir ./models --ensemble-config ensemble.yaml \\
                  --validation-data val.pkl --test-data test.pkl

  # Weighted average with optimization
  tempest combine --models-dir ./models --method weighted_average \\
                  --weighted-optimization grid_search --validation-data val.pkl

  # BMA with informative prior
  tempest combine --models model1.h5,model2.h5,model3.h5 \\
                  --method bayesian_model_averaging --prior-type informative \\
                  --prior-weights model1:0.5,model2:0.3,model3:0.2 \\
                  --validation-data val.pkl
        """
    )
    
    # Model specification
    parser_combine.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing models to combine'
    )
    parser_combine.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of [name:]path pairs for models to combine'
    )
    
    # Method selection
    parser_combine.add_argument(
        '--method',
        type=str,
        choices=['bayesian_model_averaging', 'bma', 'weighted_average', 'weighted', 'voting', 'stacking'],
        default='bayesian_model_averaging',
        help='Combination method (default: bayesian_model_averaging)'
    )
    
    # BMA-specific arguments
    parser_combine.add_argument(
        '--approximation',
        type=str,
        choices=['bic', 'laplace', 'variational', 'cross_validation'],
        default='bic',
        help='BMA approximation method (default: bic)'
    )
    parser_combine.add_argument(
        '--prior-type',
        type=str,
        choices=['uniform', 'informative', 'adaptive'],
        default='uniform',
        help='Prior type for BMA (default: uniform)'
    )
    parser_combine.add_argument(
        '--prior-weights',
        type=str,
        help='Prior weights for informative prior (format: name1:weight1,name2:weight2)'
    )
    parser_combine.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for BMA posterior scaling (default: 1.0)'
    )
    parser_combine.add_argument(
        '--bic-penalty-factor',
        type=float,
        default=1.0,
        help='Penalty factor for BIC approximation (default: 1.0)'
    )
    
    # Calibration arguments
    parser_combine.add_argument(
        '--calibrate',
        action='store_true',
        help='Enable prediction calibration'
    )
    parser_combine.add_argument(
        '--calibration-method',
        type=str,
        choices=['isotonic', 'platt', 'temperature_scaling', 'beta'],
        default='isotonic',
        help='Calibration method (default: isotonic)'
    )
    parser_combine.add_argument(
        '--calibration-data',
        type=str,
        help='Separate calibration data (uses validation data if not specified)'
    )
    
    # Data arguments
    parser_combine.add_argument(
        '--validation-data',
        type=str,
        help='Validation data for computing weights (required for BMA)'
    )
    parser_combine.add_argument(
        '--test-data',
        type=str,
        help='Test data for evaluating combination (optional)'
    )
    
    # Weighted average specific
    parser_combine.add_argument(
        '--weights',
        type=str,
        help='Custom weights for weighted method (format: name1:weight1,name2:weight2)'
    )
    parser_combine.add_argument(
        '--weighted-optimization',
        type=str,
        choices=['fixed', 'grid_search', 'differential_evolution', 'bayesian_optimization'],
        default='fixed',
        help='Weight optimization method for weighted average (default: fixed)'
    )
    
    # Configuration file
    parser_combine.add_argument(
        '--ensemble-config',
        type=str,
        help='YAML configuration file with full ensemble settings (overrides other arguments)'
    )
    
    # Output
    parser_combine.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./combine_results',
        help='Output directory for combination results (default: ./combine_results)'
    )
    
    parser_combine.set_defaults(func=combine_command)
    
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
