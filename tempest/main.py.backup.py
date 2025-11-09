#!/usr/bin/env python3
"""
Tempest main training module

This module provides the core training functionality for Tempest models,
including standard CRF training, hybrid training with constraints,
and ensemble training with Bayesian Model Averaging.
"""

import argparse
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings for TensorFlow."""
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


def load_data(data_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load training data from file.
    
    Args:
        data_path: Path to data file (.pkl, .txt, or .json)
        
    Returns:
        Dictionary containing data arrays
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    if data_path.suffix == '.pkl':
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    elif data_path.suffix == '.json':
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.suffix in ['.txt', '.tsv']:
        # Load text format data
        data = load_text_data(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    return data


def load_text_data(file_path: Path) -> Dict[str, Any]:
    """
    Load data from text format file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Dictionary containing sequences and labels
    """
    sequences = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sequences.append(parts[0])
                labels.append(parts[1].split(','))
    
    return {
        'sequences': sequences,
        'labels': labels
    }


def build_model(config: Dict[str, Any]) -> tf.keras.Model:
    """
    Build a Tempest model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    from tempest.core import build_cnn_bilstm_crf, build_model_with_length_constraints
    
    model_config = config.get('model', {})
    
    # Determine model type
    if model_config.get('use_length_constraints', False):
        logger.info("Building model with length constraints")
        model = build_model_with_length_constraints(config)
    else:
        logger.info("Building standard CRF model")
        model = build_cnn_bilstm_crf(
            max_seq_length=model_config.get('max_seq_length', 600),
            num_classes=model_config.get('num_classes', 14),
            embedding_dim=model_config.get('embedding_dim', 128),
            lstm_units=model_config.get('lstm_units', 256),
            lstm_layers=model_config.get('lstm_layers', 2),
            use_cnn=model_config.get('use_cnn', True),
            use_bidirectional=model_config.get('use_bidirectional', True),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
    
    return model


def train_standard(args: Dict[str, Any], config: Dict[str, Any]) -> tf.keras.Model:
    """
    Standard CRF model training.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    logger.info("Starting standard CRF training")
    
    # Load data
    train_data = load_data(config['data']['train_data'])
    val_data = load_data(config['data']['val_data']) if 'val_data' in config['data'] else None
    
    # Build model
    model = build_model(config)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.get('learning_rate', 0.001))
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare callbacks
    callbacks = []
    
    if args.get('checkpoint_dir'):
        checkpoint_dir = Path(args['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'model_{epoch:02d}_{val_loss:.4f}.h5'
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=args.get('save_best_only', True),
            monitor='val_loss' if val_data else 'loss',
            verbose=1
        ))
    
    if args.get('early_stopping'):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_data else 'loss',
            patience=args.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        ))
    
    if args.get('tensorboard'):
        log_dir = Path(args.get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(log_dir)))
    
    # Train model
    history = model.fit(
        x=train_data['X'],
        y=train_data['y'],
        validation_data=(val_data['X'], val_data['y']) if val_data else None,
        epochs=args.get('epochs', 50),
        batch_size=args.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    output_dir = Path(args.get('output_dir', './models'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'model_final.h5'
    model.save(str(model_path))
    logger.info(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    return model


def train_hybrid(args: Dict[str, Any], config: Dict[str, Any]) -> tf.keras.Model:
    """
    Hybrid training with constraints and pseudo-labeling.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    from tempest.training.hybrid_trainer import HybridTrainer
    
    logger.info("Starting hybrid training with constraints")
    
    # Create hybrid trainer
    trainer = HybridTrainer(config)
    
    # Load labeled data
    train_data = load_data(config['data']['train_data'])
    val_data = load_data(config['data']['val_data']) if 'val_data' in config['data'] else None
    
    # Load unlabeled data if provided
    unlabeled_data = None
    if args.get('unlabeled') or args.get('unlabeled_dir'):
        if args.get('unlabeled'):
            unlabeled_data = load_unlabeled_fastq(args['unlabeled'])
        else:
            unlabeled_data = load_unlabeled_directory(args['unlabeled_dir'])
        logger.info(f"Loaded {len(unlabeled_data)} unlabeled sequences")
    
    # Train with hybrid approach
    model = trainer.train(
        train_data=train_data,
        val_data=val_data,
        unlabeled_data=unlabeled_data,
        epochs=args.get('epochs', 50),
        batch_size=args.get('batch_size', 32),
        use_constraints=args.get('constrained_decoding', True),
        enforce_length=args.get('enforce_length_constraints', True),
        enforce_whitelist=args.get('enforce_whitelist_constraints', False),
        pwm_file=args.get('pwm'),
        use_probabilistic_pwm=args.get('use_probabilistic_pwm', True)
    )
    
    # Save model
    output_dir = Path(args.get('output_dir', './models'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'hybrid_model_final.h5'
    model.save(str(model_path))
    logger.info(f"Hybrid model saved to: {model_path}")
    
    return model


def train_ensemble(args: Dict[str, Any], config: Dict[str, Any]) -> List[tf.keras.Model]:
    """
    Train ensemble of models with optional BMA.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        List of trained models
    """
    from tempest.training.ensemble import EnsembleTrainer
    
    logger.info(f"Starting ensemble training with {args.get('num_models', 3)} models")
    
    # Create ensemble trainer
    trainer = EnsembleTrainer(
        config=config,
        num_models=args.get('num_models', 3),
        voting_method=args.get('ensemble_voting_method', 'bayesian_model_averaging')
    )
    
    # Load data
    train_data = load_data(config['data']['train_data'])
    val_data = load_data(config['data']['val_data']) if 'val_data' in config['data'] else None
    
    # Train ensemble
    models = trainer.train_ensemble(
        train_data=train_data,
        val_data=val_data,
        epochs=args.get('epochs', 50),
        batch_size=args.get('batch_size', 32),
        enforce_diversity=args.get('enforce_diversity', False),
        vary_architecture=args.get('vary_architecture', False),
        vary_initialization=args.get('vary_initialization', True)
    )
    
    # Save ensemble
    output_dir = Path(args.get('output_dir', './models'))
    ensemble_dir = output_dir / 'ensemble'
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    for i, model in enumerate(models):
        model_path = ensemble_dir / f'model_{i+1}.h5'
        model.save(str(model_path))
        logger.info(f"Ensemble model {i+1} saved to: {model_path}")
    
    # Save ensemble configuration
    ensemble_config = {
        'num_models': len(models),
        'voting_method': args.get('ensemble_voting_method', 'bayesian_model_averaging'),
        'model_paths': [f'model_{i+1}.h5' for i in range(len(models))]
    }
    
    config_path = ensemble_dir / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    logger.info(f"Ensemble configuration saved to: {config_path}")
    
    return models


def load_unlabeled_fastq(file_path: str) -> List[str]:
    """
    Load unlabeled sequences from FASTQ file.
    
    Args:
        file_path: Path to FASTQ file
        
    Returns:
        List of sequences
    """
    import gzip
    sequences = []
    
    opener = gzip.open if file_path.endswith('.gz') else open
    
    with opener(file_path, 'rt') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count % 4 == 2:  # Sequence line in FASTQ
                sequences.append(line.strip())
    
    return sequences


def load_unlabeled_directory(dir_path: str) -> List[str]:
    """
    Load unlabeled sequences from directory of FASTQ files.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        List of sequences
    """
    from pathlib import Path
    
    sequences = []
    dir_path = Path(dir_path)
    
    # Find all FASTQ files
    fastq_files = list(dir_path.glob("*.fastq")) + list(dir_path.glob("*.fq")) + \
                  list(dir_path.glob("*.fastq.gz")) + list(dir_path.glob("*.fq.gz"))
    
    for fastq_file in fastq_files:
        sequences.extend(load_unlabeled_fastq(str(fastq_file)))
    
    return sequences


def main(args: Union[Dict[str, Any], argparse.Namespace]):
    """
    Main training function.
    
    Args:
        args: Command line arguments (dict or namespace)
    """
    # Convert namespace to dict if needed
    if hasattr(args, '__dict__'):
        args = vars(args)
    
    # Setup GPU
    setup_gpu()
    
    # Load configuration
    from tempest.utils import load_config
    config = load_config(args['config'])
    
    # Override config with command line arguments
    if 'epochs' in args and args['epochs']:
        config.setdefault('training', {})['epochs'] = args['epochs']
    if 'batch_size' in args and args['batch_size']:
        config.setdefault('training', {})['batch_size'] = args['batch_size']
    if 'learning_rate' in args and args['learning_rate']:
        config.setdefault('training', {})['learning_rate'] = args['learning_rate']
    
    # Determine training mode
    if args.get('ensemble'):
        # Ensemble training
        models = train_ensemble(args, config)
        logger.info(f"Ensemble training complete! Trained {len(models)} models")
    elif args.get('hybrid'):
        # Hybrid training
        model = train_hybrid(args, config)
        logger.info("Hybrid training complete!")
    else:
        # Standard training
        model = train_standard(args, config)
        logger.info("Standard training complete!")
    
    logger.info("="*80)
    logger.info(" " * 25 + "TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    # This allows the module to be run directly if needed
    parser = argparse.ArgumentParser(description='Tempest model training')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--output-dir', default='./models', help='Output directory')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid training')
    parser.add_argument('--ensemble', action='store_true', help='Train ensemble')
    parser.add_argument('--num-models', type=int, default=3, help='Number of ensemble models')
    
    args = parser.parse_args()
    main(args)
