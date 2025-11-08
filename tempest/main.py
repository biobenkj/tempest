#!/usr/bin/env python3
"""
Tempest main execution - UPDATED VERSION

Comprehensive training module supporting:
- Standard CRF training
- Hybrid training with constraints
- Ensemble training with BMA
- Pseudo-labeling from unlabeled data
- PWM integration with probabilistic generation
"""

import argparse
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

# Import Tempest modules
from tempest.utils.config import load_config, TempestConfig
from tempest.utils.io import ensure_dir
from tempest.data.simulator import (
    SequenceSimulator, 
    SimulatedRead, 
    reads_to_arrays,
    TranscriptPool,
    PolyATailGenerator,
    WhitelistManager
)
from tempest.data.invalid_generator import InvalidReadGenerator
from tempest.training.hybrid_trainer import (
    HybridTrainer,
    build_model_from_config,
    print_model_summary,
    pad_sequences,
    convert_labels_to_categorical
)
from tempest.training.ensemble import EnsembleTrainer
from tempest.core.pwm_probabilistic import ProbabilisticPWM
from tempest.inference.combine import ModelCombiner, EnsembleConfig, BMAConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings for TensorFlow."""
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
        
        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled GPU memory growth")
        except RuntimeError as e:
            logger.warning(f"Could not set GPU memory growth: {e}")
    else:
        logger.info("No GPUs found - using CPU")


def parse_unlabeled_input(path_string: str) -> Union[str, Path]:
    """
    Parse unlabeled input path from command line.
    
    Args:
        path_string: Path string from command line
        
    Returns:
        Path object
    """
    path = Path(path_string)
    
    if not path.exists():
        logger.warning(f"Path does not exist: {path_string}")
    elif path.is_dir():
        logger.info(f"Unlabeled input is a directory: {path}")
        # Find FASTQ files
        fastq_files = list(path.glob("*.fastq")) + list(path.glob("*.fq")) + \
                     list(path.glob("*.fastq.gz")) + list(path.glob("*.fq.gz"))
        logger.info(f"Found {len(fastq_files)} FASTQ files")
    else:
        logger.info(f"Unlabeled input is a file: {path}")
        
    return path


def load_or_generate_data(config: TempestConfig, args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    """
    Load existing data or generate synthetic data based on configuration.
    
    Args:
        config: Tempest configuration
        args: Command line arguments
        
    Returns:
        Tuple of (X_train, y_train, label_encoder)
    """
    data_dir = Path(args.data_dir if args.data_dir else "./tempest_data")
    
    # Check if data already exists
    train_file = data_dir / "train_data.pkl"
    if train_file.exists() and not args.regenerate_data:
        logger.info(f"Loading existing training data from {train_file}")
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['y_train'], data['label_encoder']
    
    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    
    # Initialize simulator with enhanced configuration
    sim_config = config.simulation
    
    # Apply command-line overrides
    if args.num_sequences:
        sim_config.num_sequences = args.num_sequences
    
    # Initialize PWM if configured
    pwm = None
    if hasattr(sim_config, 'pwm') and sim_config.pwm.get('pwm_file'):
        pwm_file = sim_config.pwm['pwm_file']
        if args.pwm:
            pwm_file = args.pwm
            
        logger.info(f"Loading PWM from {pwm_file}")
        pwm = ProbabilisticPWM(
            pwm_file=pwm_file,
            temperature=sim_config.pwm.get('temperature', 1.2),
            min_entropy=sim_config.pwm.get('min_entropy', 0.1)
        )
    
    # Initialize simulator
    simulator = SequenceSimulator(sim_config, pwm_file=pwm_file if pwm else None)
    
    # Generate training data
    train_reads = simulator.generate_reads(sim_config.n_train)
    val_reads = simulator.generate_reads(sim_config.n_val)
    
    # Convert to arrays
    from tempest.data.simulator import LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit label encoder on all possible labels from config
    all_labels = set()
    for read in train_reads + val_reads:
        all_labels.update(read.labels)
    label_encoder.fit(list(all_labels))
    
    # Convert reads to arrays
    X_train, y_train = reads_to_arrays(train_reads, label_encoder)
    X_val, y_val = reads_to_arrays(val_reads, label_encoder)
    
    # Save generated data
    ensure_dir(data_dir)
    train_file = data_dir / "train_data.pkl"
    val_file = data_dir / "val_data.pkl"
    
    with open(train_file, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'label_encoder': label_encoder
        }, f)
        
    with open(val_file, 'wb') as f:
        pickle.dump({
            'X_val': X_val,
            'y_val': y_val,
            'label_encoder': label_encoder
        }, f)
    
    logger.info(f"Generated {len(X_train)} training and {len(X_val)} validation sequences")
    logger.info(f"Saved data to {data_dir}")
    
    return X_train, y_train, X_val, y_val, label_encoder


def train_standard_model(config: TempestConfig, args: argparse.Namespace,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        label_encoder: Any) -> keras.Model:
    """
    Train a standard CRF model.
    
    Args:
        config: Tempest configuration
        args: Command line arguments
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        label_encoder: Label encoder
        
    Returns:
        Trained model
    """
    logger.info("Training standard CRF model")
    
    # Apply command-line overrides to config
    model_config = config.model
    if args.max_seq_len:
        model_config.max_seq_len = args.max_seq_len
    if args.embedding_dim:
        model_config.embedding_dim = args.embedding_dim
    if args.lstm_units:
        model_config.lstm_units = args.lstm_units
    if args.lstm_layers:
        model_config.lstm_layers = args.lstm_layers
    if args.dropout:
        model_config.dropout = args.dropout
    if args.use_cnn:
        model_config.use_cnn = True
    if args.use_bilstm:
        model_config.use_bilstm = True
        
    training_config = config.training
    if args.epochs:
        training_config.epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.optimizer:
        training_config.optimizer = args.optimizer
        
    # Build model
    model = build_model_from_config(model_config, num_classes=len(label_encoder.classes_))
    
    # Compile model
    optimizer = keras.optimizers.get({
        'class_name': training_config.optimizer,
        'config': {'learning_rate': training_config.learning_rate}
    })
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print_model_summary(model)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        ensure_dir(checkpoint_dir)
        checkpoint_path = checkpoint_dir / "model_{epoch:02d}_{val_loss:.2f}.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=args.save_best_only if hasattr(args, 'save_best_only') else True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    if args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience if args.patience else training_config.early_stopping.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # TensorBoard
    if args.tensorboard:
        log_dir = Path(args.log_dir if args.log_dir else "./logs")
        ensure_dir(log_dir)
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
    
    # Prepare data
    X_train_pad = pad_sequences(X_train, maxlen=model_config.max_seq_len)
    X_val_pad = pad_sequences(X_val, maxlen=model_config.max_seq_len)
    y_train_cat = convert_labels_to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_val_cat = convert_labels_to_categorical(y_val, num_classes=len(label_encoder.classes_))
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights or training_config.get('use_class_weights'):
        from sklearn.utils.class_weight import compute_class_weight
        y_train_flat = y_train.reshape(-1)
        classes = np.unique(y_train_flat[y_train_flat != -1])  # Exclude padding
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train_flat[y_train_flat != -1]
        )
        class_weights = dict(zip(classes, weights))
        logger.info("Computed class weights for imbalanced data")
    
    # Train model
    history = model.fit(
        X_train_pad, y_train_cat,
        validation_data=(X_val_pad, y_val_cat),
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    output_dir = Path(args.output_dir if args.output_dir else "./tempest_output")
    ensure_dir(output_dir)
    
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history.history, f, indent=2)
    logger.info(f"Saved training history to {history_file}")
    
    return model, history


def train_hybrid_model(config: TempestConfig, args: argparse.Namespace,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      label_encoder: Any) -> keras.Model:
    """
    Train a hybrid model with constraints and pseudo-labeling.
    
    Args:
        config: Tempest configuration
        args: Command line arguments
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        label_encoder: Label encoder
        
    Returns:
        Trained model
    """
    logger.info("Training hybrid model with constraints")
    
    # Initialize hybrid trainer
    hybrid_config = config.hybrid if hasattr(config, 'hybrid') else {}
    
    # Apply command-line overrides
    if args.constrained_decoding:
        hybrid_config['constrained_decoding'] = {
            'enabled': True,
            'method': args.decoding_method if args.decoding_method else 'beam_search',
            'beam_width': args.beam_width if args.beam_width else 5
        }
        
    if args.enforce_length_constraints:
        hybrid_config['length_constraints'] = {
            'enabled': True,
            'enforce_during_training': True,
            'enforce_during_inference': True
        }
        
    if args.enforce_whitelist_constraints:
        hybrid_config['whitelist_constraints'] = {
            'enabled': True,
            'enforce_during_training': True,
            'enforce_during_inference': True
        }
        
    # PWM constraints
    pwm = None
    if args.pwm or (hasattr(config, 'hybrid') and config.hybrid.get('pwm_constraints', {}).get('enabled')):
        pwm_file = args.pwm if args.pwm else config.simulation.pwm.get('pwm_file')
        if pwm_file:
            logger.info(f"Loading PWM for constraints from {pwm_file}")
            if args.use_probabilistic_pwm:
                pwm = ProbabilisticPWM(
                    pwm_file=pwm_file,
                    temperature=config.simulation.pwm.get('temperature', 1.2),
                    min_entropy=config.simulation.pwm.get('min_entropy', 0.1)
                )
                hybrid_config['pwm_constraints'] = {
                    'enabled': True,
                    'use_probabilistic_scoring': True,
                    'scoring_method': args.pwm_scoring_method if args.pwm_scoring_method else 'log_likelihood',
                    'min_score': args.pwm_min_score if args.pwm_min_score else -10.0,
                    'score_weight': args.pwm_score_weight if args.pwm_score_weight else 0.5
                }
    
    # Initialize trainer
    trainer = HybridTrainer(
        config=config,
        hybrid_config=hybrid_config,
        pwm=pwm
    )
    
    # Load unlabeled data for pseudo-labeling if provided
    pseudo_labels = None
    if args.unlabeled or args.unlabeled_dir:
        logger.info("Generating pseudo-labels from unlabeled data...")
        
        if args.unlabeled:
            unlabeled_path = parse_unlabeled_input(args.unlabeled)
        else:
            unlabeled_path = parse_unlabeled_input(args.unlabeled_dir)
            
        pseudo_labels = trainer.generate_pseudo_labels(
            unlabeled_path=unlabeled_path,
            max_per_file=args.max_pseudo_per_file if args.max_pseudo_per_file else 1000,
            max_total=args.max_pseudo_total if args.max_pseudo_total else 10000,
            confidence_threshold=0.85
        )
        
        if pseudo_labels:
            logger.info(f"Generated {len(pseudo_labels)} pseudo-labels")
            # Add pseudo-labels to training data
            X_pseudo = np.array([pl['sequence'] for pl in pseudo_labels])
            y_pseudo = np.array([pl['labels'] for pl in pseudo_labels])
            X_train = np.concatenate([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])
    
    # Train model
    model = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_encoder=label_encoder,
        epochs=args.epochs if args.epochs else config.training.epochs,
        batch_size=args.batch_size if args.batch_size else config.training.batch_size
    )
    
    # Save hybrid training report
    output_dir = Path(args.output_dir if args.output_dir else "./tempest_output")
    ensure_dir(output_dir)
    
    report = trainer.generate_report()
    report_file = output_dir / "hybrid_training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved hybrid training report to {report_file}")
    
    return model


def train_ensemble_models(config: TempestConfig, args: argparse.Namespace,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         label_encoder: Any) -> List[keras.Model]:
    """
    Train an ensemble of models with BMA support.
    
    Args:
        config: Tempest configuration
        args: Command line arguments
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        label_encoder: Label encoder
        
    Returns:
        List of trained models
    """
    logger.info("Training ensemble of models")
    
    # Get ensemble configuration
    ensemble_config = config.ensemble if hasattr(config, 'ensemble') else {}
    
    # Apply command-line overrides
    if args.num_models:
        ensemble_config['num_models'] = args.num_models
    if args.ensemble_voting_method:
        ensemble_config['voting_method'] = args.ensemble_voting_method
        
    # Configure BMA if requested
    if args.ensemble_voting_method == 'bayesian_model_averaging':
        bma_config = ensemble_config.get('bma_config', {})
        if args.bma_prior_type:
            bma_config['prior_type'] = args.bma_prior_type
        if args.bma_approximation:
            bma_config['approximation'] = args.bma_approximation
        if args.bma_temperature:
            bma_config['temperature'] = args.bma_temperature
        ensemble_config['bma_config'] = bma_config
        
    # Configure diversity
    if args.enforce_diversity:
        diversity_config = ensemble_config.get('diversity', {})
        diversity_config['enforce_diversity'] = True
        if args.diversity_metric:
            diversity_config['diversity_metric'] = args.diversity_metric
        if args.vary_architecture:
            diversity_config['vary_architecture'] = True
        if args.vary_initialization:
            diversity_config['vary_initialization'] = True
        ensemble_config['diversity'] = diversity_config
        
    # Configure calibration
    if args.enable_calibration:
        calibration_config = ensemble_config.get('calibration', {})
        calibration_config['enabled'] = True
        if args.calibration_method:
            calibration_config['method'] = args.calibration_method
        ensemble_config['calibration'] = calibration_config
    
    # Initialize ensemble trainer
    trainer = EnsembleTrainer(
        config=config,
        ensemble_config=ensemble_config
    )
    
    # Train models
    models = trainer.train_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_encoder=label_encoder,
        num_models=ensemble_config.get('num_models', 3)
    )
    
    # Combine models if BMA is requested
    if ensemble_config.get('voting_method') == 'bayesian_model_averaging':
        logger.info("Combining models with Bayesian Model Averaging")
        
        # Create BMA configuration
        bma_config_obj = BMAConfig(
            approximation=ensemble_config['bma_config'].get('approximation', 'bic'),
            prior_type=ensemble_config['bma_config'].get('prior_type', 'uniform'),
            temperature=ensemble_config['bma_config'].get('temperature', 1.0)
        )
        
        # Create ensemble configuration
        ensemble_cfg = EnsembleConfig(
            voting_method='bayesian_model_averaging',
            bma_config=bma_config_obj
        )
        
        # Create combiner
        model_paths = {}
        for i, model in enumerate(models):
            model_name = f"model_{i}"
            model_paths[model_name] = model
            
        combiner = ModelCombiner(model_paths, ensemble_cfg)
        
        # Compute BMA weights
        logger.info("Computing BMA posterior weights...")
        combiner.compute_bma_weights({'X': X_val, 'y': y_val})
        
        # Calibrate if requested
        if ensemble_config.get('calibration', {}).get('enabled'):
            logger.info("Calibrating ensemble predictions...")
            combiner.calibrate({'X': X_val, 'y': y_val})
        
        # Save ensemble
        output_dir = Path(args.output_dir if args.output_dir else "./tempest_output")
        ensure_dir(output_dir)
        
        ensemble_path = output_dir / "ensemble"
        combiner.save(ensemble_path)
        logger.info(f"Saved BMA ensemble to {ensemble_path}")
        
        # Save ensemble report
        report = trainer.generate_ensemble_report()
        report['bma_weights'] = combiner.get_posterior_weights()
        report['model_evidence'] = combiner.get_model_evidence()
        
        report_file = output_dir / "ensemble_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved ensemble report to {report_file}")
        
        return combiner
    
    # Save individual models
    output_dir = Path(args.output_dir if args.output_dir else "./tempest_output")
    ensure_dir(output_dir)
    
    for i, model in enumerate(models):
        model_path = output_dir / f"model_{i}.h5"
        model.save(model_path)
        logger.info(f"Saved model {i} to {model_path}")
    
    # Save ensemble report
    report = trainer.generate_ensemble_report()
    report_file = output_dir / "ensemble_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved ensemble report to {report_file}")
    
    return models


def evaluate_model(model: Union[keras.Model, Any], 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  label_encoder: Any,
                  config: TempestConfig,
                  output_dir: Path) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model or ensemble
        X_test: Test sequences
        y_test: Test labels
        label_encoder: Label encoder
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    from tempest.inference import ModelEvaluator
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Check if model is an ensemble
    if hasattr(model, 'predict_with_uncertainty'):
        # Ensemble with uncertainty
        predictions, uncertainty = model.predict_with_uncertainty(X_test)
        y_pred = predictions.argmax(axis=-1)
        
        metrics = {
            'accuracy': accuracy_score(y_test.reshape(-1), y_pred.reshape(-1)),
            'mean_uncertainty': float(uncertainty.mean()),
            'std_uncertainty': float(uncertainty.std())
        }
        
    else:
        # Single model
        evaluator = ModelEvaluator(model)
        y_pred = evaluator.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test.reshape(-1), y_pred.reshape(-1))
        }
    
    # Classification report
    report = classification_report(
        y_test.reshape(-1),
        y_pred.reshape(-1),
        target_names=label_encoder.classes_,
        output_dict=True
    )
    metrics['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_test.reshape(-1), y_pred.reshape(-1))
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-segment metrics if configured
    if config.evaluation.get('per_segment_metrics'):
        segment_metrics = {}
        for i, label in enumerate(label_encoder.classes_):
            mask = y_test.reshape(-1) == i
            if mask.any():
                segment_metrics[label] = {
                    'accuracy': accuracy_score(
                        y_test.reshape(-1)[mask],
                        y_pred.reshape(-1)[mask]
                    ),
                    'support': int(mask.sum())
                }
        metrics['per_segment'] = segment_metrics
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_file}")
    
    # Log summary
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    if 'mean_uncertainty' in metrics:
        logger.info(f"Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
    
    return metrics


def main():
    """Main training execution."""
    parser = argparse.ArgumentParser(
        description='Tempest training with comprehensive configuration support'
    )
    
    # Required arguments
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration YAML file')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing or for saving data')
    parser.add_argument('--regenerate-data', action='store_true',
                       help='Regenerate synthetic data even if it exists')
    parser.add_argument('--num-sequences', type=int,
                       help='Override number of sequences to generate')
    
    # Model architecture overrides
    parser.add_argument('--max-seq-len', type=int,
                       help='Maximum sequence length')
    parser.add_argument('--embedding-dim', type=int,
                       help='Embedding dimension')
    parser.add_argument('--lstm-units', type=int,
                       help='Number of LSTM units')
    parser.add_argument('--lstm-layers', type=int,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float,
                       help='Dropout rate')
    parser.add_argument('--use-cnn', action='store_true',
                       help='Use CNN layers')
    parser.add_argument('--use-bilstm', action='store_true',
                       help='Use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str,
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # Hybrid training
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid training mode')
    parser.add_argument('--constrained-decoding', action='store_true',
                       help='Enable constrained decoding')
    parser.add_argument('--decoding-method', type=str,
                       choices=['beam_search', 'viterbi', 'greedy'],
                       help='Decoding method')
    parser.add_argument('--beam-width', type=int,
                       help='Beam width for beam search')
    parser.add_argument('--enforce-length-constraints', action='store_true',
                       help='Enforce length constraints')
    parser.add_argument('--enforce-whitelist-constraints', action='store_true',
                       help='Enforce whitelist constraints')
    
    # PWM constraints
    parser.add_argument('--pwm', type=str,
                       help='PWM file for constraints')
    parser.add_argument('--use-probabilistic-pwm', action='store_true',
                       help='Use probabilistic PWM scoring')
    parser.add_argument('--pwm-scoring-method', type=str,
                       choices=['log_likelihood', 'geometric_mean', 'min_probability'],
                       help='PWM scoring method')
    parser.add_argument('--pwm-min-score', type=float,
                       help='Minimum PWM score threshold')
    parser.add_argument('--pwm-score-weight', type=float,
                       help='Weight for PWM score in loss')
    
    # Ensemble training
    parser.add_argument('--ensemble', action='store_true',
                       help='Train ensemble of models')
    parser.add_argument('--num-models', type=int,
                       help='Number of models in ensemble')
    parser.add_argument('--ensemble-voting-method', type=str,
                       choices=['bayesian_model_averaging', 'weighted_average', 'voting'],
                       help='Ensemble voting method')
    parser.add_argument('--bma-prior-type', type=str,
                       choices=['uniform', 'informative', 'adaptive'],
                       help='BMA prior type')
    parser.add_argument('--bma-approximation', type=str,
                       choices=['bic', 'laplace', 'variational', 'cross_validation'],
                       help='BMA approximation method')
    parser.add_argument('--bma-temperature', type=float,
                       help='BMA temperature scaling')
    
    # Model diversity
    parser.add_argument('--enforce-diversity', action='store_true',
                       help='Enforce model diversity in ensemble')
    parser.add_argument('--diversity-metric', type=str,
                       choices=['disagreement', 'correlation', 'kl_divergence'],
                       help='Diversity metric')
    parser.add_argument('--vary-architecture', action='store_true',
                       help='Vary architecture across ensemble')
    parser.add_argument('--vary-initialization', action='store_true',
                       help='Vary initialization across ensemble')
    
    # Calibration
    parser.add_argument('--enable-calibration', action='store_true',
                       help='Enable prediction calibration')
    parser.add_argument('--calibration-method', type=str,
                       choices=['isotonic', 'platt', 'temperature_scaling', 'beta'],
                       help='Calibration method')
    
    # Pseudo-labeling
    parser.add_argument('--unlabeled', type=str,
                       help='Path to unlabeled FASTQ file')
    parser.add_argument('--unlabeled-dir', type=str,
                       help='Directory with unlabeled FASTQ files')
    parser.add_argument('--max-pseudo-per-file', type=int,
                       help='Max pseudo-labels per file')
    parser.add_argument('--max-pseudo-total', type=int,
                       help='Max total pseudo-labels')
    
    # Checkpointing and monitoring
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory for checkpoints')
    parser.add_argument('--save-best-only', action='store_true',
                       help='Save only best model')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int,
                       help='Early stopping patience')
    
    # Output
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for models and logs')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--log-dir', type=str,
                       help='TensorBoard log directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load or generate data
    data = load_or_generate_data(config, args)
    if len(data) == 5:
        X_train, y_train, X_val, y_val, label_encoder = data
    else:
        X_train, y_train, label_encoder = data
        # Generate validation data
        from tempest.data.simulator import SequenceSimulator
        simulator = SequenceSimulator(config.simulation)
        val_reads = simulator.generate_reads(config.simulation.n_val)
        X_val, y_val = reads_to_arrays(val_reads, label_encoder)
    
    # Generate test data for evaluation
    logger.info("Generating test data...")
    simulator = SequenceSimulator(config.simulation)
    test_reads = simulator.generate_reads(config.simulation.n_test)
    X_test, y_test = reads_to_arrays(test_reads, label_encoder)
    
    # Choose training mode
    if args.ensemble:
        # Train ensemble
        model = train_ensemble_models(
            config, args,
            X_train, y_train,
            X_val, y_val,
            label_encoder
        )
    elif args.hybrid:
        # Train hybrid model
        model = train_hybrid_model(
            config, args,
            X_train, y_train,
            X_val, y_val,
            label_encoder
        )
    else:
        # Train standard model
        model, history = train_standard_model(
            config, args,
            X_train, y_train,
            X_val, y_val,
            label_encoder
        )
    
    # Evaluate model
    output_dir = Path(args.output_dir if args.output_dir else "./tempest_output")
    ensure_dir(output_dir)
    
    metrics = evaluate_model(
        model, X_test, y_test,
        label_encoder, config, output_dir
    )
    
    # Save final model(s)
    if not args.ensemble or not isinstance(model, ModelCombiner):
        model_path = output_dir / "model_final.h5"
        if hasattr(model, 'save'):
            model.save(model_path)
            logger.info(f"Saved final model to {model_path}")
    
    # Save label encoder
    encoder_path = output_dir / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Saved label encoder to {encoder_path}")
    
    # Save configuration used
    config_path = output_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict() if hasattr(config, 'to_dict') else vars(config), f)
    logger.info(f"Saved configuration to {config_path}")
    
    logger.info("Training complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
