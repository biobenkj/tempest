#!/usr/bin/env python3
"""
Tempest CLI with modular subcommands - UPDATED VERSION

This module provides the main command-line interface for Tempest,
organizing functionality into clear subcommands for simulation,
training, evaluation, visualization, and enhanced model combination.

All configuration options from the YAML are now exposed as CLI arguments.
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
    
    Enhanced with all simulation configuration options from YAML.
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
    if args.full_rc_prob:
        config.simulation.full_read_reverse_complement_prob = args.full_rc_prob
        
    # PWM configuration
    if args.pwm:
        if not hasattr(config.simulation, 'pwm'):
            config.simulation.pwm = {}
        config.simulation.pwm['pwm_file'] = args.pwm
        if args.pwm_temperature:
            config.simulation.pwm['temperature'] = args.pwm_temperature
        if args.pwm_min_entropy:
            config.simulation.pwm['min_entropy'] = args.pwm_min_entropy
            
    # Whitelist files
    if args.whitelist_i7:
        if not hasattr(config.simulation, 'whitelist_files'):
            config.simulation.whitelist_files = {}
        config.simulation.whitelist_files['i7'] = args.whitelist_i7
    if args.whitelist_i5:
        config.simulation.whitelist_files['i5'] = args.whitelist_i5
    if args.whitelist_cbc:
        config.simulation.whitelist_files['CBC'] = args.whitelist_cbc
        
    # Transcript configuration
    if args.transcript_fasta:
        if not hasattr(config.simulation, 'transcript'):
            config.simulation.transcript = {}
        config.simulation.transcript['fasta_file'] = args.transcript_fasta
        if args.fragment_mode:
            config.simulation.transcript['fragment_mode'] = True
            if args.fragment_min:
                config.simulation.transcript['fragment_min'] = args.fragment_min
            if args.fragment_max:
                config.simulation.transcript['fragment_max'] = args.fragment_max
                
    # Error injection
    if args.enable_errors:
        if not hasattr(config.simulation, 'error_injection'):
            config.simulation.error_injection = {'enabled': True}
        if args.substitution_rate:
            config.simulation.error_injection['substitution_rate'] = args.substitution_rate
        if args.insertion_rate:
            config.simulation.error_injection['insertion_rate'] = args.insertion_rate
        if args.deletion_rate:
            config.simulation.error_injection['deletion_rate'] = args.deletion_rate
    
    # Initialize simulator
    simulator = SequenceSimulator(config.simulation, pwm_file=config.simulation.pwm.pwm_file if hasattr(config.simulation, 'pwm') else None)
    
    # Generate sequences
    logger.info(f"Generating {config.simulation.num_sequences} sequences...")
    
    if args.split:
        # Generate training/validation split
        train_reads, val_reads = simulator.generate_train_val_split(
            total_reads=config.simulation.num_sequences,
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
        reads = simulator.generate_reads(config.simulation.num_sequences)
        
        # Save to file
        output_file = Path(args.output or "./simulated_reads.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving reads to: {output_file}")
        with open(output_file, 'w') as f:
            for read in reads:
                f.write(f"{read.sequence}\t{','.join(read.labels)}\n")
        
        logger.info(f"Generated {len(reads)} sequences")
    
    # Save simulation statistics if requested
    if args.save_stats:
        stats_file = Path(args.output_dir or ".") / "simulation_stats.json"
        stats = simulator.get_statistics()
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved simulation statistics to: {stats_file}")


def train_command(args):
    """Train a Tempest model with all configuration options."""
    from tempest.main import main as train_main
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST TRAINING")
    logger.info("="*80)
    
    # Prepare training arguments
    train_args = [
        '--config', args.config
    ]
    
    # Model architecture
    if args.max_seq_len:
        train_args.extend(['--max-seq-len', str(args.max_seq_len)])
    if args.embedding_dim:
        train_args.extend(['--embedding-dim', str(args.embedding_dim)])
    if args.lstm_units:
        train_args.extend(['--lstm-units', str(args.lstm_units)])
    if args.lstm_layers:
        train_args.extend(['--lstm-layers', str(args.lstm_layers)])
    if args.dropout:
        train_args.extend(['--dropout', str(args.dropout)])
    if args.use_cnn:
        train_args.append('--use-cnn')
    if args.use_bilstm:
        train_args.append('--use-bilstm')
        
    # Training parameters
    if args.epochs:
        train_args.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        train_args.extend(['--batch-size', str(args.batch_size)])
    if args.learning_rate:
        train_args.extend(['--learning-rate', str(args.learning_rate)])
    if args.optimizer:
        train_args.extend(['--optimizer', args.optimizer])
    if args.use_class_weights:
        train_args.append('--use-class-weights')
        
    # Hybrid training
    if args.hybrid:
        train_args.append('--hybrid')
        
        # Constrained decoding
        if args.constrained_decoding:
            train_args.append('--constrained-decoding')
            if args.decoding_method:
                train_args.extend(['--decoding-method', args.decoding_method])
            if args.beam_width:
                train_args.extend(['--beam-width', str(args.beam_width)])
                
        # Length constraints
        if args.enforce_length_constraints:
            train_args.append('--enforce-length-constraints')
            
        # Whitelist constraints
        if args.enforce_whitelist_constraints:
            train_args.append('--enforce-whitelist-constraints')
            
        # PWM constraints
        if args.pwm:
            train_args.extend(['--pwm', args.pwm])
            if args.use_probabilistic_pwm:
                train_args.append('--use-probabilistic-pwm')
                if args.pwm_scoring_method:
                    train_args.extend(['--pwm-scoring-method', args.pwm_scoring_method])
                if args.pwm_min_score:
                    train_args.extend(['--pwm-min-score', str(args.pwm_min_score)])
                if args.pwm_score_weight:
                    train_args.extend(['--pwm-score-weight', str(args.pwm_score_weight)])
                    
    # Ensemble training
    if args.ensemble:
        train_args.append('--ensemble')
        if args.num_models:
            train_args.extend(['--num-models', str(args.num_models)])
        if args.ensemble_voting_method:
            train_args.extend(['--ensemble-voting-method', args.ensemble_voting_method])
            
        # BMA configuration
        if args.ensemble_voting_method == 'bayesian_model_averaging':
            if args.bma_prior_type:
                train_args.extend(['--bma-prior-type', args.bma_prior_type])
            if args.bma_approximation:
                train_args.extend(['--bma-approximation', args.bma_approximation])
            if args.bma_temperature:
                train_args.extend(['--bma-temperature', str(args.bma_temperature)])
                
        # Model diversity
        if args.enforce_diversity:
            train_args.append('--enforce-diversity')
            if args.diversity_metric:
                train_args.extend(['--diversity-metric', args.diversity_metric])
            if args.vary_architecture:
                train_args.append('--vary-architecture')
            if args.vary_initialization:
                train_args.append('--vary-initialization')
                
        # Calibration
        if args.enable_calibration:
            train_args.append('--enable-calibration')
            if args.calibration_method:
                train_args.extend(['--calibration-method', args.calibration_method])
                
    # Unlabeled data for pseudo-labeling
    if args.unlabeled:
        train_args.extend(['--unlabeled', args.unlabeled])
    if args.unlabeled_dir:
        train_args.extend(['--unlabeled-dir', args.unlabeled_dir])
    if args.max_pseudo_per_file:
        train_args.extend(['--max-pseudo-per-file', str(args.max_pseudo_per_file)])
    if args.max_pseudo_total:
        train_args.extend(['--max-pseudo-total', str(args.max_pseudo_total)])
        
    # Checkpointing and early stopping
    if args.checkpoint_dir:
        train_args.extend(['--checkpoint-dir', args.checkpoint_dir])
    if args.save_best_only:
        train_args.append('--save-best-only')
    if args.early_stopping:
        train_args.append('--early-stopping')
    if args.patience:
        train_args.extend(['--patience', str(args.patience)])
        
    # Output
    if args.output_dir:
        train_args.extend(['--output-dir', args.output_dir])
    if args.tensorboard:
        train_args.append('--tensorboard')
        if args.log_dir:
            train_args.extend(['--log-dir', args.log_dir])
    
    # Call training main with arguments
    sys.argv = ['tempest-train'] + train_args
    train_main()


def evaluate_command(args):
    """Evaluate a trained model with comprehensive metrics."""
    from tempest.inference import ModelEvaluator
    
    logger.info("="*80)
    logger.info(" " * 28 + "TEMPEST EVALUATION")
    logger.info("="*80)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, config_path=args.config)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = evaluator.load_test_data(args.test_data)
    
    # Configure evaluation metrics
    metrics_config = {
        'compute_per_segment': args.per_segment_metrics,
        'compute_confusion_matrix': args.confusion_matrix,
        'compute_boundary_accuracy': args.boundary_accuracy,
        'compute_edit_distance': args.edit_distance,
        'bootstrap_samples': args.bootstrap if args.bootstrap else 0
    }
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluator.evaluate(
        test_data, 
        batch_size=args.batch_size,
        **metrics_config
    )
    
    # Print metrics
    logger.info("\nEvaluation Results:")
    logger.info("-" * 40)
    
    # Basic metrics
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            logger.info(f"  {metric}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v:.4f}")
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_file}")
    
    # Generate predictions if requested
    if args.save_predictions:
        logger.info("Generating predictions...")
        predictions_file = output_dir / "predictions.txt"
        predictions = evaluator.predict(test_data, batch_size=args.batch_size)
        evaluator.save_predictions(predictions, predictions_file)
        logger.info(f"Saved predictions to: {predictions_file}")
    
    # Confusion matrix visualization
    if args.confusion_matrix and args.plot_confusion:
        logger.info("Generating confusion matrix plot...")
        cm_plot = output_dir / "confusion_matrix.png"
        evaluator.plot_confusion_matrix(test_data, save_path=cm_plot)
        logger.info(f"Saved confusion matrix to: {cm_plot}")
        
    # Error analysis
    if args.error_analysis:
        logger.info("Performing error analysis...")
        error_report = evaluator.analyze_errors(test_data, num_samples=args.num_error_samples)
        error_file = output_dir / "error_analysis.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        logger.info(f"Saved error analysis to: {error_file}")
        
    # ACC-specific evaluation (if applicable)
    if args.acc_evaluation:
        logger.info("Evaluating ACC generation...")
        acc_metrics = evaluator.evaluate_acc_generation(test_data)
        acc_file = output_dir / "acc_evaluation.json"
        with open(acc_file, 'w') as f:
            json.dump(acc_metrics, f, indent=2)
        logger.info(f"Saved ACC evaluation to: {acc_file}")


def visualize_command(args):
    """Generate enhanced visualizations."""
    from tempest.visualization import Visualizer
    
    logger.info("="*80)
    logger.info(" " * 28 + "TEMPEST VISUALIZATION")
    logger.info("="*80)
    
    visualizer = Visualizer(style=args.style)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == 'training':
        logger.info("Plotting training history...")
        output_file = output_dir / f"training_curves.{args.format}"
        visualizer.plot_training_history(
            args.input, 
            output_path=output_file, 
            dpi=args.dpi,
            smoothing=args.smoothing
        )
        
    elif args.type == 'predictions':
        logger.info(f"Visualizing {args.num_samples} predictions...")
        output_file = output_dir / f"predictions.{args.format}"
        visualizer.plot_predictions(
            args.input,
            num_samples=args.num_samples,
            output_path=output_file,
            dpi=args.dpi,
            show_confidence=args.show_confidence
        )
        
    elif args.type == 'attention':
        if not args.model:
            logger.error("Model file required for attention visualization")
            return
        logger.info("Visualizing attention weights...")
        output_file = output_dir / f"attention_weights.{args.format}"
        visualizer.plot_attention(
            args.model, 
            args.input, 
            output_path=output_file, 
            dpi=args.dpi
        )
        
    elif args.type == 'embeddings':
        if not args.model:
            logger.error("Model file required for embeddings visualization")
            return
        logger.info("Visualizing embeddings...")
        output_file = output_dir / f"embeddings.{args.format}"
        visualizer.plot_embeddings(
            args.model, 
            output_path=output_file, 
            dpi=args.dpi,
            method=args.embedding_method
        )
        
    elif args.type == 'segment_performance':
        logger.info("Plotting segment-wise performance...")
        output_file = output_dir / f"segment_performance.{args.format}"
        visualizer.plot_segment_performance(
            args.input,
            output_path=output_file,
            dpi=args.dpi
        )
        
    elif args.type == 'length_distribution':
        logger.info("Plotting length distributions...")
        output_file = output_dir / f"length_distribution.{args.format}"
        visualizer.plot_length_distributions(
            args.input,
            output_path=output_file,
            dpi=args.dpi
        )
        
    elif args.type == 'model_comparison':
        logger.info("Creating model comparison plots...")
        output_file = output_dir / f"model_comparison.{args.format}"
        visualizer.plot_model_comparison(
            args.input,
            output_path=output_file,
            dpi=args.dpi
        )
        
    elif args.type == 'acc_pwm':
        logger.info("Visualizing ACC PWM...")
        output_file = output_dir / f"acc_pwm_heatmap.{args.format}"
        visualizer.plot_acc_pwm_heatmap(
            args.pwm_file,
            output_path=output_file,
            dpi=args.dpi
        )
        
        # Also create sequence logo
        logo_file = output_dir / f"acc_sequence_logo.{args.format}"
        visualizer.plot_acc_sequence_logo(
            args.pwm_file,
            output_path=logo_file,
            dpi=args.dpi
        )
        
    logger.info(f"Visualization saved to: {output_dir}")


def compare_command(args):
    """Compare multiple models with comprehensive metrics."""
    from tempest.compare import ModelComparator
    
    logger.info("="*80)
    logger.info(" " * 26 + "TEMPEST MODEL COMPARISON")
    logger.info("="*80)
    
    # Parse model list
    if args.models:
        model_paths = [p.strip() for p in args.models.split(',')]
    else:
        model_paths = []
        models_dir = Path(args.models_dir)
        for pattern in ['*.h5', '*.keras', '*.pkl']:
            model_paths.extend([str(p) for p in models_dir.glob(pattern)])
        # Check for ensemble directories
        for item in models_dir.iterdir():
            if item.is_dir() and (item / 'ensemble_metadata.json').exists():
                model_paths.append(str(item))
    
    if not model_paths:
        logger.error("No models found to compare")
        return
        
    logger.info(f"Found {len(model_paths)} models to compare")
    
    # Create comparator
    comparator = ModelComparator(model_paths, config_path=args.config)
    
    # Load test data
    comparator.load_test_data(args.test_data)
    
    # Configure metrics
    if args.metrics:
        metrics = args.metrics.split(',')
    else:
        metrics = [
            'accuracy', 'precision', 'recall', 'f1',
            'segment_accuracy', 'edit_distance', 'boundary_accuracy'
        ]
        
    # Add ensemble-specific metrics if applicable
    if args.include_ensemble_metrics:
        metrics.extend(['diversity', 'uncertainty', 'agreement'])
    
    # Run comparison
    logger.info("Comparing models...")
    results = comparator.compare(
        metrics=metrics,
        compute_pairwise_agreement=args.pairwise_agreement,
        bootstrap_samples=args.bootstrap
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison data
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison results to: {results_file}")
    
    # Generate plots
    if not args.no_plots:
        logger.info("Generating comparison plots...")
        comparator.plot_comparison(output_dir, dpi=args.dpi)
    
    # Generate report
    if not args.no_report:
        logger.info("Generating comparison report...")
        report_file = output_dir / "comparison_report.md"
        comparator.generate_report(report_file)
        logger.info(f"Saved report to: {report_file}")
    
    # Performance profiling
    if args.profile_performance:
        logger.info("Profiling model performance...")
        perf_results = comparator.profile_performance()
        perf_file = output_dir / "performance_profile.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_results, f, indent=2)
        logger.info(f"Saved performance profile to: {perf_file}")
    
    logger.info(f"Comparison complete. Results saved to: {output_dir}")


def combine_command(args):
    """
    Enhanced combine command with full BMA support.
    
    Implements advanced model combination strategies including
    multiple BMA approximation methods, calibration, and uncertainty quantification.
    """
    from tempest.inference.combine import ModelCombiner, EnsembleConfig, BMAConfig
    from pathlib import Path
    import pickle
    
    logger.info("="*80)
    logger.info(" " * 20 + "TEMPEST MODEL COMBINATION")
    logger.info("="*80)
    
    # Load ensemble configuration if provided
    if args.ensemble_config:
        with open(args.ensemble_config, 'r') as f:
            ensemble_cfg = yaml.safe_load(f)
        logger.info(f"Loaded ensemble configuration from: {args.ensemble_config}")
    else:
        ensemble_cfg = {}
    
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
        for pattern in ['*.h5', '*.keras', '*.pkl']:
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
    
    # Create BMA configuration if using BMA
    bma_config = None
    if args.method in ['bayesian_model_averaging', 'bma']:
        bma_config = BMAConfig(
            approximation=args.approximation or ensemble_cfg.get('bma_config', {}).get('approximation', 'bic'),
            prior_type=args.prior_type or ensemble_cfg.get('bma_config', {}).get('prior_type', 'uniform'),
            temperature=args.temperature or ensemble_cfg.get('bma_config', {}).get('temperature', 1.0),
            compute_posterior_variance=ensemble_cfg.get('bma_config', {}).get('compute_posterior_variance', True),
            normalize_posteriors=ensemble_cfg.get('bma_config', {}).get('normalize_posteriors', True)
        )
        
        # Set prior weights if informative prior
        if args.prior_type == 'informative' and args.prior_weights:
            prior_weights = {}
            for item in args.prior_weights.split(','):
                name, weight = item.split(':')
                prior_weights[name.strip()] = float(weight)
            bma_config.prior_weights = prior_weights
        
        # Approximation-specific parameters
        if args.approximation == 'bic':
            bma_config.approximation_params = {
                'bic': {'penalty_factor': args.bic_penalty_factor or 1.0}
            }
        elif args.approximation == 'laplace':
            bma_config.approximation_params = {
                'laplace': {
                    'num_samples': ensemble_cfg.get('bma_config', {}).get('approximation_params', {}).get('laplace', {}).get('num_samples', 1000),
                    'damping': ensemble_cfg.get('bma_config', {}).get('approximation_params', {}).get('laplace', {}).get('damping', 0.01)
                }
            }
        elif args.approximation == 'variational':
            bma_config.approximation_params = {
                'variational': {
                    'num_iterations': ensemble_cfg.get('bma_config', {}).get('approximation_params', {}).get('variational', {}).get('num_iterations', 100),
                    'learning_rate': ensemble_cfg.get('bma_config', {}).get('approximation_params', {}).get('variational', {}).get('learning_rate', 0.01)
                }
            }
        elif args.approximation == 'cross_validation':
            bma_config.approximation_params = {
                'cross_validation': {
                    'num_folds': ensemble_cfg.get('bma_config', {}).get('approximation_params', {}).get('cross_validation', {}).get('num_folds', 5),
                    'stratified': True
                }
            }
        
        logger.info(f"Configured BMA with {args.approximation} approximation")
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        voting_method=args.method,
        bma_config=bma_config
    )
    
    # Weighted average configuration
    if args.method in ['weighted_average', 'weighted']:
        if args.weights:
            # Parse custom weights
            weights = {}
            for item in args.weights.split(','):
                name, weight = item.split(':')
                weights[name.strip()] = float(weight)
            ensemble_config.weighted_average_config = {
                'optimization': 'fixed',
                'fixed_weights': weights
            }
        else:
            # Use optimization
            ensemble_config.weighted_average_config = {
                'optimization': args.weighted_optimization or 'fixed'
            }
            if args.weighted_optimization == 'grid_search':
                ensemble_config.weighted_average_config['optimization_params'] = {
                    'grid_search': {'weight_resolution': 0.1}
                }
    
    # Calibration configuration
    if args.calibrate:
        ensemble_config.calibration = {
            'enabled': True,
            'method': args.calibration_method or 'isotonic',
            'use_separate_calibration_set': args.calibration_data is not None
        }
        logger.info(f"Enabled {args.calibration_method or 'isotonic'} calibration")
    
    # Create combiner
    combiner = ModelCombiner(model_paths, ensemble_config)
    
    # Load validation data (required for BMA and optimization)
    if args.validation_data:
        logger.info(f"Loading validation data from: {args.validation_data}")
        with open(args.validation_data, 'rb') as f:
            val_data = pickle.load(f)
        
        # Compute weights or posteriors
        if args.method in ['bayesian_model_averaging', 'bma']:
            logger.info("Computing BMA posterior weights...")
            combiner.compute_bma_weights(val_data)
        elif args.method in ['weighted_average', 'weighted'] and args.weighted_optimization != 'fixed':
            logger.info(f"Optimizing weights using {args.weighted_optimization}...")
            combiner.optimize_weights(val_data)
    
    # Load calibration data if separate
    if args.calibrate and args.calibration_data:
        logger.info(f"Loading calibration data from: {args.calibration_data}")
        with open(args.calibration_data, 'rb') as f:
            cal_data = pickle.load(f)
        combiner.calibrate(cal_data)
    elif args.calibrate and args.validation_data:
        logger.info("Using validation data for calibration")
        combiner.calibrate(val_data)
    
    # Save the ensemble
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_path = output_dir / "ensemble"
    combiner.save(ensemble_path)
    logger.info(f"Saved ensemble to: {ensemble_path}")
    
    # Save metadata and configuration
    metadata = combiner.get_metadata()
    metadata_file = output_dir / "ensemble_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_file}")
    
    # Evaluate on test data if provided
    if args.test_data:
        logger.info(f"Evaluating ensemble on test data: {args.test_data}")
        with open(args.test_data, 'rb') as f:
            test_data = pickle.load(f)
        
        # Get predictions with uncertainty
        predictions, uncertainty = combiner.predict_with_uncertainty(test_data['X'])
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = predictions.argmax(axis=-1).reshape(-1)
        y_true = test_data['y'].reshape(-1)
        
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save test results
        test_results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'mean_uncertainty': float(uncertainty.mean()),
            'std_uncertainty': float(uncertainty.std())
        }
        
        if args.method in ['bayesian_model_averaging', 'bma']:
            test_results['posterior_weights'] = combiner.get_posterior_weights()
            test_results['model_evidence'] = combiner.get_model_evidence()
        
        results_file = output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved test results to: {results_file}")
        
        # Plot uncertainty distribution
        if args.plot_uncertainty:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(uncertainty.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Predictive Uncertainty')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predictive Uncertainty')
            plt.grid(True, alpha=0.3)
            uncertainty_plot = output_dir / "uncertainty_distribution.png"
            plt.savefig(uncertainty_plot, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved uncertainty plot to: {uncertainty_plot}")
    
    logger.info(f"Model combination complete. Results saved to: {output_dir}")


def create_parser():
    """Create the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        prog='tempest',
        description='Tempest - Advanced sequence annotation with length-constrained CRFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEMPEST is a modular framework for sequence annotation using:
  • Length-constrained Conditional Random Fields (CRFs)
  • Position Weight Matrix (PWM) integration with probabilistic generation
  • Hybrid training modes with pseudo-labeling and constraints
  • Bayesian Model Averaging (BMA) for ensemble combination
  • Comprehensive evaluation and visualization tools

For detailed documentation, visit: https://github.com/yourusername/tempest
        """
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
    parser_simulate.add_argument(
        '--full-rc-prob',
        type=float,
        help='Probability of full read reverse complement'
    )
    
    # PWM options
    parser_simulate.add_argument(
        '--pwm',
        type=str,
        help='PWM file for ACC generation'
    )
    parser_simulate.add_argument(
        '--pwm-temperature',
        type=float,
        help='Temperature for probabilistic PWM sampling'
    )
    parser_simulate.add_argument(
        '--pwm-min-entropy',
        type=float,
        help='Minimum entropy at each PWM position'
    )
    
    # Whitelist options
    parser_simulate.add_argument(
        '--whitelist-i7',
        type=str,
        help='Whitelist file for i7 indices'
    )
    parser_simulate.add_argument(
        '--whitelist-i5',
        type=str,
        help='Whitelist file for i5 indices'
    )
    parser_simulate.add_argument(
        '--whitelist-cbc',
        type=str,
        help='Whitelist file for cell barcodes'
    )
    
    # Transcript options
    parser_simulate.add_argument(
        '--transcript-fasta',
        type=str,
        help='FASTA file for cDNA generation'
    )
    parser_simulate.add_argument(
        '--fragment-mode',
        action='store_true',
        help='Enable transcript fragmentation'
    )
    parser_simulate.add_argument(
        '--fragment-min',
        type=int,
        help='Minimum fragment length'
    )
    parser_simulate.add_argument(
        '--fragment-max',
        type=int,
        help='Maximum fragment length'
    )
    
    # Error injection
    parser_simulate.add_argument(
        '--enable-errors',
        action='store_true',
        help='Enable error injection'
    )
    parser_simulate.add_argument(
        '--substitution-rate',
        type=float,
        help='Substitution error rate'
    )
    parser_simulate.add_argument(
        '--insertion-rate',
        type=float,
        help='Insertion error rate'
    )
    parser_simulate.add_argument(
        '--deletion-rate',
        type=float,
        help='Deletion error rate'
    )
    parser_simulate.add_argument(
        '--save-stats',
        action='store_true',
        help='Save simulation statistics'
    )
    
    parser_simulate.set_defaults(func=simulate_command)
    
    # ============ TRAIN COMMAND ============
    parser_train = subparsers.add_parser(
        'train',
        help='Train a Tempest model',
        description='Train models with standard, hybrid, or ensemble approaches'
    )
    parser_train.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    # Model architecture
    parser_train.add_argument(
        '--max-seq-len',
        type=int,
        help='Maximum sequence length'
    )
    parser_train.add_argument(
        '--embedding-dim',
        type=int,
        help='Embedding dimension'
    )
    parser_train.add_argument(
        '--lstm-units',
        type=int,
        help='Number of LSTM units'
    )
    parser_train.add_argument(
        '--lstm-layers',
        type=int,
        help='Number of LSTM layers'
    )
    parser_train.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate'
    )
    parser_train.add_argument(
        '--use-cnn',
        action='store_true',
        help='Use CNN layers'
    )
    parser_train.add_argument(
        '--use-bilstm',
        action='store_true',
        help='Use bidirectional LSTM'
    )
    
    # Training parameters
    parser_train.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser_train.add_argument(
        '--batch-size',
        type=int,
        help='Batch size'
    )
    parser_train.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    parser_train.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'sgd', 'rmsprop'],
        help='Optimizer'
    )
    parser_train.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use class weights for imbalanced data'
    )
    
    # Hybrid training
    parser_train.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid training mode'
    )
    parser_train.add_argument(
        '--constrained-decoding',
        action='store_true',
        help='Enable constrained decoding'
    )
    parser_train.add_argument(
        '--decoding-method',
        type=str,
        choices=['beam_search', 'viterbi', 'greedy'],
        help='Decoding method'
    )
    parser_train.add_argument(
        '--beam-width',
        type=int,
        help='Beam width for beam search'
    )
    parser_train.add_argument(
        '--enforce-length-constraints',
        action='store_true',
        help='Enforce length constraints'
    )
    parser_train.add_argument(
        '--enforce-whitelist-constraints',
        action='store_true',
        help='Enforce whitelist constraints'
    )
    
    # PWM constraints
    parser_train.add_argument(
        '--pwm',
        type=str,
        help='PWM file for constraints'
    )
    parser_train.add_argument(
        '--use-probabilistic-pwm',
        action='store_true',
        help='Use probabilistic PWM scoring'
    )
    parser_train.add_argument(
        '--pwm-scoring-method',
        type=str,
        choices=['log_likelihood', 'geometric_mean', 'min_probability'],
        help='PWM scoring method'
    )
    parser_train.add_argument(
        '--pwm-min-score',
        type=float,
        help='Minimum PWM score threshold'
    )
    parser_train.add_argument(
        '--pwm-score-weight',
        type=float,
        help='Weight for PWM score in loss'
    )
    
    # Ensemble training
    parser_train.add_argument(
        '--ensemble',
        action='store_true',
        help='Train ensemble of models'
    )
    parser_train.add_argument(
        '--num-models',
        type=int,
        help='Number of models in ensemble'
    )
    parser_train.add_argument(
        '--ensemble-voting-method',
        type=str,
        choices=['bayesian_model_averaging', 'weighted_average', 'voting'],
        help='Ensemble voting method'
    )
    parser_train.add_argument(
        '--bma-prior-type',
        type=str,
        choices=['uniform', 'informative', 'adaptive'],
        help='BMA prior type'
    )
    parser_train.add_argument(
        '--bma-approximation',
        type=str,
        choices=['bic', 'laplace', 'variational', 'cross_validation'],
        help='BMA approximation method'
    )
    parser_train.add_argument(
        '--bma-temperature',
        type=float,
        help='BMA temperature scaling'
    )
    
    # Model diversity
    parser_train.add_argument(
        '--enforce-diversity',
        action='store_true',
        help='Enforce model diversity in ensemble'
    )
    parser_train.add_argument(
        '--diversity-metric',
        type=str,
        choices=['disagreement', 'correlation', 'kl_divergence'],
        help='Diversity metric'
    )
    parser_train.add_argument(
        '--vary-architecture',
        action='store_true',
        help='Vary architecture across ensemble'
    )
    parser_train.add_argument(
        '--vary-initialization',
        action='store_true',
        help='Vary initialization across ensemble'
    )
    
    # Calibration
    parser_train.add_argument(
        '--enable-calibration',
        action='store_true',
        help='Enable prediction calibration'
    )
    parser_train.add_argument(
        '--calibration-method',
        type=str,
        choices=['isotonic', 'platt', 'temperature_scaling', 'beta'],
        help='Calibration method'
    )
    
    # Pseudo-labeling
    parser_train.add_argument(
        '--unlabeled',
        type=str,
        help='Path to unlabeled FASTQ file'
    )
    parser_train.add_argument(
        '--unlabeled-dir',
        type=str,
        help='Directory with unlabeled FASTQ files'
    )
    parser_train.add_argument(
        '--max-pseudo-per-file',
        type=int,
        help='Max pseudo-labels per file'
    )
    parser_train.add_argument(
        '--max-pseudo-total',
        type=int,
        help='Max total pseudo-labels'
    )
    
    # Checkpointing
    parser_train.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory for checkpoints'
    )
    parser_train.add_argument(
        '--save-best-only',
        action='store_true',
        help='Save only best model'
    )
    parser_train.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser_train.add_argument(
        '--patience',
        type=int,
        help='Early stopping patience'
    )
    
    # Output
    parser_train.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for models and logs'
    )
    parser_train.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )
    parser_train.add_argument(
        '--log-dir',
        type=str,
        help='TensorBoard log directory'
    )
    
    parser_train.set_defaults(func=train_command)
    
    # ============ EVALUATE COMMAND ============
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained model',
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
        help='Batch size for evaluation'
    )
    
    # Metrics
    parser_evaluate.add_argument(
        '--per-segment-metrics',
        action='store_true',
        help='Compute per-segment metrics'
    )
    parser_evaluate.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Compute confusion matrix'
    )
    parser_evaluate.add_argument(
        '--boundary-accuracy',
        action='store_true',
        help='Compute boundary accuracy'
    )
    parser_evaluate.add_argument(
        '--edit-distance',
        action='store_true',
        help='Compute edit distance'
    )
    parser_evaluate.add_argument(
        '--bootstrap',
        type=int,
        help='Number of bootstrap samples for confidence intervals'
    )
    
    # Analysis
    parser_evaluate.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )
    parser_evaluate.add_argument(
        '--plot-confusion',
        action='store_true',
        help='Plot confusion matrix'
    )
    parser_evaluate.add_argument(
        '--error-analysis',
        action='store_true',
        help='Perform error analysis'
    )
    parser_evaluate.add_argument(
        '--num-error-samples',
        type=int,
        default=100,
        help='Number of samples for error analysis'
    )
    parser_evaluate.add_argument(
        '--acc-evaluation',
        action='store_true',
        help='Perform ACC-specific evaluation'
    )
    
    parser_evaluate.set_defaults(func=evaluate_command)
    
    # ============ VISUALIZE COMMAND ============
    parser_visualize = subparsers.add_parser(
        'visualize',
        help='Create visualizations',
        description='Generate various plots and visualizations'
    )
    parser_visualize.add_argument(
        '--type', '-t',
        type=str,
        required=True,
        choices=[
            'training', 'predictions', 'attention', 'embeddings',
            'segment_performance', 'length_distribution',
            'model_comparison', 'acc_pwm'
        ],
        help='Type of visualization'
    )
    parser_visualize.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file or directory'
    )
    parser_visualize.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./visualizations',
        help='Output directory'
    )
    parser_visualize.add_argument(
        '--model',
        type=str,
        help='Model file (for attention/embeddings)'
    )
    parser_visualize.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg', 'html'],
        default='png',
        help='Output format'
    )
    parser_visualize.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for raster formats'
    )
    parser_visualize.add_argument(
        '--style',
        type=str,
        choices=['default', 'paper', 'presentation'],
        default='default',
        help='Plot style'
    )
    
    # Specific options
    parser_visualize.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    parser_visualize.add_argument(
        '--show-confidence',
        action='store_true',
        help='Show prediction confidence'
    )
    parser_visualize.add_argument(
        '--smoothing',
        type=int,
        help='Smoothing window for training curves'
    )
    parser_visualize.add_argument(
        '--embedding-method',
        type=str,
        choices=['tsne', 'umap', 'pca'],
        default='tsne',
        help='Embedding visualization method'
    )
    parser_visualize.add_argument(
        '--pwm-file',
        type=str,
        help='PWM file for ACC visualization'
    )
    
    parser_visualize.set_defaults(func=visualize_command)
    
    # ============ COMPARE COMMAND ============
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare multiple models',
        description='Comprehensive model comparison and analysis'
    )
    parser_compare.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing models'
    )
    parser_compare.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of model paths'
    )
    parser_compare.add_argument(
        '--test-data', '-t',
        type=str,
        required=True,
        help='Path to test data'
    )
    parser_compare.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file'
    )
    parser_compare.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./comparison_results',
        help='Output directory'
    )
    parser_compare.add_argument(
        '--metrics',
        type=str,
        help='Comma-separated list of metrics'
    )
    parser_compare.add_argument(
        '--include-ensemble-metrics',
        action='store_true',
        help='Include ensemble-specific metrics'
    )
    parser_compare.add_argument(
        '--pairwise-agreement',
        action='store_true',
        help='Compute pairwise agreement'
    )
    parser_compare.add_argument(
        '--bootstrap',
        type=int,
        help='Bootstrap samples for confidence intervals'
    )
    parser_compare.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser_compare.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    parser_compare.add_argument(
        '--profile-performance',
        action='store_true',
        help='Profile model performance'
    )
    parser_compare.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for plots'
    )
    
    parser_compare.set_defaults(func=compare_command)
    
    # ============ COMBINE COMMAND ============
    parser_combine = subparsers.add_parser(
        'combine',
        help='Combine models with BMA or voting',
        description='Advanced model combination with multiple strategies'
    )
    
    # Model specification
    parser_combine.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing models'
    )
    parser_combine.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of [name:]path pairs'
    )
    
    # Method selection
    parser_combine.add_argument(
        '--method',
        type=str,
        choices=['bayesian_model_averaging', 'bma', 'weighted_average', 'weighted', 'voting', 'stacking'],
        default='bayesian_model_averaging',
        help='Combination method'
    )
    
    # BMA arguments
    parser_combine.add_argument(
        '--approximation',
        type=str,
        choices=['bic', 'laplace', 'variational', 'cross_validation'],
        default='bic',
        help='BMA approximation method'
    )
    parser_combine.add_argument(
        '--prior-type',
        type=str,
        choices=['uniform', 'informative', 'adaptive'],
        default='uniform',
        help='Prior type for BMA'
    )
    parser_combine.add_argument(
        '--prior-weights',
        type=str,
        help='Prior weights (format: name1:weight1,name2:weight2)'
    )
    parser_combine.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for BMA posterior'
    )
    parser_combine.add_argument(
        '--bic-penalty-factor',
        type=float,
        default=1.0,
        help='BIC penalty factor'
    )
    
    # Calibration
    parser_combine.add_argument(
        '--calibrate',
        action='store_true',
        help='Enable calibration'
    )
    parser_combine.add_argument(
        '--calibration-method',
        type=str,
        choices=['isotonic', 'platt', 'temperature_scaling', 'beta'],
        default='isotonic',
        help='Calibration method'
    )
    parser_combine.add_argument(
        '--calibration-data',
        type=str,
        help='Separate calibration data'
    )
    
    # Data
    parser_combine.add_argument(
        '--validation-data',
        type=str,
        help='Validation data for weight computation'
    )
    parser_combine.add_argument(
        '--test-data',
        type=str,
        help='Test data for evaluation'
    )
    
    # Weighted average
    parser_combine.add_argument(
        '--weights',
        type=str,
        help='Custom weights (format: name1:weight1,name2:weight2)'
    )
    parser_combine.add_argument(
        '--weighted-optimization',
        type=str,
        choices=['fixed', 'grid_search', 'differential_evolution', 'bayesian_optimization'],
        default='fixed',
        help='Weight optimization method'
    )
    
    # Configuration
    parser_combine.add_argument(
        '--ensemble-config',
        type=str,
        help='YAML configuration file for ensemble'
    )
    
    # Output
    parser_combine.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./combine_results',
        help='Output directory'
    )
    parser_combine.add_argument(
        '--plot-uncertainty',
        action='store_true',
        help='Plot uncertainty distribution'
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
