#!/usr/bin/env python3
"""
Tempest Training Pipeline with Directory Support for Pseudo-Labeling.

Enhanced to work seamlessly with the improved CLI and BMA functionality.

This script supports:
1. Passing a directory of FASTQ files for pseudo-label training
2. Flexible input handling (single file or directory)
3. Configurable batch processing parameters
4. Probabilistic PWM-based ACC generation
5. Integration with enhanced ensemble training

Usage:
    # Train with single FASTQ file for pseudo-labels
    python main.py --config config.yaml --hybrid --unlabeled /path/to/file.fastq
    
    # Train with directory of FASTQ files for pseudo-labels
    python main.py --config config.yaml --hybrid --unlabeled-dir /path/to/fastq_directory/
    
    # Train with custom limits for directory processing
    python main.py --config config.yaml --hybrid \
                   --unlabeled-dir /path/to/fastq_directory/ \
                   --max-pseudo-per-file 500 \
                   --max-pseudo-total 5000
    
    # Train with specific PWM file for ACC generation
    python main.py --config config.yaml --pwm /path/to/pwm_file.txt
    
    # Train for ensemble with model diversity
    python main.py --config config.yaml --ensemble --num-models 3 \
                   --vary-initialization --vary-architecture
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np
from tensorflow import keras
import json

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings for TensorFlow."""
    import tensorflow as tf
    
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
        # Check for FASTQ files
        fastq_files = list(path.glob("*.fastq")) + list(path.glob("*.fastq.gz")) + \
                      list(path.glob("*.fq")) + list(path.glob("*.fq.gz"))
        logger.info(f"Found {len(fastq_files)} FASTQ files in directory")
    elif path.is_file():
        logger.info(f"Unlabeled input is a single file: {path}")
    
    return path


def prepare_data(config: TempestConfig, pwm_file: Optional[str] = None) -> tuple:
    """
    Prepare training and validation data with PWM support.
    
    Args:
        config: Tempest configuration
        pwm_file: Optional PWM file path
        
    Returns:
        Tuple of training and validation data
    """
    logger.info("Preparing training data...")
    
    # Initialize simulator with PWM if provided
    simulator = SequenceSimulator(config.simulation, pwm_file=pwm_file)
    
    # Generate training and validation reads
    train_reads, val_reads = simulator.generate_train_val_split(
        total_reads=config.simulation.num_sequences,
        train_fraction=0.8
    )
    
    logger.info(f"Generated {len(train_reads)} training and {len(val_reads)} validation sequences")
    
    # Convert to arrays
    label_to_idx = simulator.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Convert reads to arrays
    X_train, y_train = reads_to_arrays(train_reads, label_to_idx)
    X_val, y_val = reads_to_arrays(val_reads, label_to_idx)
    
    # Pad sequences
    max_len = config.model.max_seq_len
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', value=0)
    y_train = pad_sequences(y_train, maxlen=max_len, padding='post', value=-1)
    X_val = pad_sequences(X_val, maxlen=max_len, padding='post', value=0)
    y_val = pad_sequences(y_val, maxlen=max_len, padding='post', value=-1)
    
    logger.info(f"Prepared data shapes:")
    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads


def train_standard(config, X_train, y_train, X_val, y_val, label_to_idx):
    """Standard training mode."""
    logger.info("\n" + "="*60)
    logger.info("Starting STANDARD training mode")
    logger.info("="*60 + "\n")
    
    # Build model
    model = build_model_from_config(config)
    print_model_summary(model)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(config.training.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    ensure_dir(config.training.checkpoint_dir)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f"{config.training.checkpoint_dir}/model_best.h5",
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.EarlyStopping(
            patience=config.training.early_stopping_patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=config.training.reduce_lr_patience,
            factor=0.5
        ),
        keras.callbacks.CSVLogger(
            f"{config.training.checkpoint_dir}/training_history.csv"
        )
    ]
    
    # Train model
    logger.info(f"Training for {config.training.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = f"{config.training.checkpoint_dir}/model_final.h5"
    model.save(final_path)
    logger.info(f"Saved final model to: {final_path}")
    
    # Save label mapping
    import pickle
    labels_path = f"{config.training.checkpoint_dir}/label_mapping.pkl"
    with open(labels_path, 'wb') as f:
        pickle.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {v: k for k, v in label_to_idx.items()}
        }, f)
    logger.info(f"Saved label mapping to: {labels_path}")
    
    # Save configuration
    config_path = f"{config.training.checkpoint_dir}/config.yaml"
    config.to_yaml(config_path)
    logger.info(f"Saved configuration to: {config_path}")
    
    return model


def train_hybrid(config, train_reads, val_reads, unlabeled_path=None, 
                max_pseudo_per_file=None, max_pseudo_total=None):
    """Hybrid training mode with pseudo-labeling support."""
    logger.info("\n" + "="*60)
    logger.info("Starting HYBRID training mode")
    logger.info("="*60 + "\n")
    
    # Initialize hybrid trainer
    trainer = HybridTrainer(config)
    
    # Process unlabeled data if provided
    if unlabeled_path:
        path_obj = Path(unlabeled_path)
        
        if path_obj.is_dir():
            # Process directory of FASTQ files
            logger.info(f"Processing FASTQ files from directory: {path_obj}")
            trainer.process_unlabeled_directory(
                path_obj,
                max_reads_per_file=max_pseudo_per_file,
                max_total_reads=max_pseudo_total
            )
        elif path_obj.is_file():
            # Process single FASTQ file
            logger.info(f"Loading unlabeled sequences from: {path_obj}")
            trainer.load_unlabeled_sequences(path_obj)
        else:
            logger.warning(f"Invalid unlabeled path: {unlabeled_path}")
    else:
        logger.info("No unlabeled data provided - skipping pseudo-labeling phase")
    
    # Train model
    model = trainer.train(train_reads, val_reads)
    
    # Save final model
    ensure_dir(config.training.checkpoint_dir)
    final_path = f"{config.training.checkpoint_dir}/model_hybrid_final.h5"
    model.save(final_path)
    logger.info(f"Saved hybrid model to: {final_path}")
    
    # Save training history
    if hasattr(trainer, 'history'):
        import pandas as pd
        history_df = pd.DataFrame(trainer.history)
        history_path = f"{config.training.checkpoint_dir}/hybrid_training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to: {history_path}")
    
    # Save configuration
    config_path = f"{config.training.checkpoint_dir}/hybrid_config.yaml"
    config.to_yaml(config_path)
    
    return model


def train_ensemble(config, X_train, y_train, X_val, y_val, label_to_idx, 
                  num_models=3, vary_architecture=True, vary_initialization=True):
    """
    Train ensemble of models with diversity.
    
    This function trains multiple models with variations to create
    a diverse ensemble suitable for BMA combination.
    
    Args:
        config: Training configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        label_to_idx: Label mapping
        num_models: Number of models to train
        vary_architecture: Whether to vary model architectures
        vary_initialization: Whether to vary random seeds
        
    Returns:
        List of trained models
    """
    logger.info("\n" + "="*60)
    logger.info(f"Training ENSEMBLE with {num_models} models")
    logger.info("="*60 + "\n")
    
    models = []
    ensemble_dir = Path(config.training.checkpoint_dir) / "ensemble"
    ensure_dir(str(ensemble_dir))
    
    # Architecture variations
    architecture_variations = []
    if vary_architecture:
        base_units = config.model.lstm_units
        base_dropout = config.model.dropout
        architecture_variations = [
            {'lstm_units': base_units, 'dropout': base_dropout},
            {'lstm_units': base_units // 2, 'dropout': base_dropout + 0.1},
            {'lstm_units': base_units * 2, 'dropout': max(0.2, base_dropout - 0.1)},
        ]
    else:
        architecture_variations = [
            {'lstm_units': config.model.lstm_units, 'dropout': config.model.dropout}
        ] * num_models
    
    # Random seeds for initialization diversity
    random_seeds = [42, 123, 456, 789, 1011] if vary_initialization else [42] * num_models
    
    # Train each model
    for i in range(num_models):
        logger.info(f"\nTraining model {i+1}/{num_models}...")
        
        # Set random seed
        np.random.seed(random_seeds[i % len(random_seeds)])
        tf.random.set_seed(random_seeds[i % len(random_seeds)])
        
        # Modify config for this model
        model_config = config
        if vary_architecture:
            arch_params = architecture_variations[i % len(architecture_variations)]
            model_config.model.lstm_units = arch_params['lstm_units']
            model_config.model.dropout = arch_params['dropout']
            logger.info(f"  Architecture: LSTM units={arch_params['lstm_units']}, dropout={arch_params['dropout']}")
        
        # Build and compile model
        model = build_model_from_config(model_config)
        model.compile(
            optimizer=keras.optimizers.Adam(config.training.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Setup callbacks for this model
        model_dir = ensemble_dir / f"model_{i}"
        ensure_dir(str(model_dir))
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(model_dir / "best.h5"),
                save_best_only=True,
                monitor='val_loss'
            ),
            keras.callbacks.EarlyStopping(
                patience=config.training.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.CSVLogger(
                str(model_dir / "history.csv")
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.training.epochs,
            batch_size=config.model.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model_path = ensemble_dir / f"ensemble_model_{i}.h5"
        model.save(str(model_path))
        logger.info(f"  Saved model to: {model_path}")
        
        # Save model-specific config
        model_config_path = model_dir / "config.yaml"
        model_config.to_yaml(str(model_config_path))
        
        models.append(model)
    
    # Save ensemble metadata
    metadata = {
        'num_models': num_models,
        'vary_architecture': vary_architecture,
        'vary_initialization': vary_initialization,
        'model_paths': [f"ensemble_model_{i}.h5" for i in range(num_models)],
        'architecture_variations': architecture_variations[:num_models] if vary_architecture else None,
        'random_seeds': random_seeds[:num_models]
    }
    
    metadata_path = ensemble_dir / "ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved ensemble metadata to: {metadata_path}")
    
    # Save label mapping
    import pickle
    labels_path = ensemble_dir / "label_mapping.pkl"
    with open(labels_path, 'wb') as f:
        pickle.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {v: k for k, v in label_to_idx.items()}
        }, f)
    
    logger.info(f"\nEnsemble training complete. Models saved to: {ensemble_dir}")
    logger.info("Use 'tempest combine' to create BMA ensemble from these models")
    
    return models


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Tempest Training Pipeline with Enhanced Ensemble Support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    # Training mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid training mode with invalid sequence handling'
    )
    mode_group.add_argument(
        '--ensemble',
        action='store_true',
        help='Train ensemble of models for BMA combination'
    )
    
    # Ensemble parameters
    parser.add_argument(
        '--num-models',
        type=int,
        default=3,
        help='Number of models to train for ensemble (default: 3)'
    )
    parser.add_argument(
        '--vary-architecture',
        action='store_true',
        help='Vary model architectures in ensemble'
    )
    parser.add_argument(
        '--vary-initialization',
        action='store_true',
        help='Vary random initialization in ensemble'
    )
    
    # PWM parameters
    parser.add_argument(
        '--pwm',
        type=str,
        help='Path to PWM file for probabilistic ACC generation'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Temperature for PWM sampling (higher = more random)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for model checkpoints (overrides config)'
    )
    
    # Unlabeled data for hybrid training
    unlabeled_group = parser.add_mutually_exclusive_group()
    unlabeled_group.add_argument(
        '--unlabeled',
        type=str,
        help='Path to unlabeled FASTQ file for pseudo-label training'
    )
    unlabeled_group.add_argument(
        '--unlabeled-dir',
        type=str,
        default=None,
        help='Path to directory containing FASTQ files for pseudo-label training'
    )
    
    # Directory processing parameters
    parser.add_argument(
        '--max-pseudo-per-file',
        type=int,
        default=None,
        help='Maximum pseudo-labels to generate per FASTQ file (for directory input)'
    )
    parser.add_argument(
        '--max-pseudo-total',
        type=int,
        default=None,
        help='Maximum total pseudo-labels to generate across all files (for directory input)'
    )
    
    # Training hyperparameters (override config)
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*25 + "TEMPEST TRAINING PIPELINE")
    print(" "*20 + "with Enhanced Ensemble Support")
    if args.hybrid:
        print(" "*15 + "(HYBRID ROBUSTNESS MODE WITH DIRECTORY SUPPORT)")
    elif args.ensemble:
        print(" "*20 + f"(ENSEMBLE MODE: {args.num_models} MODELS)")
    print("="*80 + "\n")
    
    # Setup GPU
    setup_gpu()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override configuration with command-line arguments
    if args.output_dir:
        config.training.checkpoint_dir = args.output_dir
        logger.info(f"Output directory overridden to: {args.output_dir}")
    
    if args.epochs:
        config.training.epochs = args.epochs
        logger.info(f"Epochs overridden to: {args.epochs}")
    
    if args.batch_size:
        config.model.batch_size = args.batch_size
        logger.info(f"Batch size overridden to: {args.batch_size}")
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        logger.info(f"Learning rate overridden to: {args.learning_rate}")
    
    # Determine PWM file and temperature
    pwm_file = args.pwm
    if pwm_file is None and hasattr(config, 'pwm') and hasattr(config.pwm, 'pwm_file'):
        pwm_file = config.pwm.pwm_file
    
    if pwm_file:
        logger.info(f"Using PWM file: {pwm_file}")
        if args.temperature is not None:
            logger.info(f"PWM temperature override: {args.temperature}")
            if not hasattr(config, 'pwm'):
                config.pwm = type('obj', (object,), {'temperature': args.temperature})()
            else:
                config.pwm.temperature = args.temperature
    else:
        logger.info("No PWM file specified - ACC sequences will be generated randomly or from patterns")
    
    # Handle unlabeled data path for hybrid training
    unlabeled_path = args.unlabeled or args.unlabeled_dir
    if unlabeled_path:
        unlabeled_path = parse_unlabeled_input(unlabeled_path)
        
        # Print helpful information about the unlabeled data
        path_obj = Path(unlabeled_path)
        if path_obj.is_dir():
            logger.info("Pseudo-labeling will process multiple FASTQ files from directory")
            if args.max_pseudo_per_file:
                logger.info(f"  Max reads per file: {args.max_pseudo_per_file}")
            if args.max_pseudo_total:
                logger.info(f"  Max total reads: {args.max_pseudo_total}")
        elif path_obj.is_file():
            logger.info("Pseudo-labeling will process single FASTQ file")
    
    logger.info("Configuration loaded\n")
    
    # Run pipeline
    try:
        # Prepare data with probabilistic PWM support
        X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads = prepare_data(config, pwm_file)
        
        # Train based on mode
        if args.ensemble:
            # Train ensemble of models
            models = train_ensemble(
                config, X_train, y_train, X_val, y_val, label_to_idx,
                num_models=args.num_models,
                vary_architecture=args.vary_architecture,
                vary_initialization=args.vary_initialization
            )
            model = models[0]  # Return first model for compatibility
            
        elif args.hybrid:
            # Hybrid training
            model = train_hybrid(
                config, 
                train_reads, 
                val_reads, 
                unlabeled_path,
                args.max_pseudo_per_file,
                args.max_pseudo_total
            )
        else:
            # Standard training
            model = train_standard(config, X_train, y_train, X_val, y_val, label_to_idx)
        
        if model is not None:
            # Summary
            print("\n" + "="*80)
            print(" "*30 + "TRAINING COMPLETE")
            print("="*80)
            print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
            
            if args.ensemble:
                print(f"\nEnsemble Training Results:")
                print(f"  - Trained {args.num_models} models")
                print(f"  - Models saved to: {config.training.checkpoint_dir}/ensemble/")
                print(f"  - Architecture variation: {args.vary_architecture}")
                print(f"  - Initialization variation: {args.vary_initialization}")
                print(f"\nNext steps:")
                print(f"  1. Prepare validation data for BMA weight computation")
                print(f"  2. Use 'tempest combine' to create BMA ensemble:")
                print(f"     tempest combine --models-dir {config.training.checkpoint_dir}/ensemble \\")
                print(f"                     --method bayesian_model_averaging \\")
                print(f"                     --validation-data val_data.pkl")
            else:
                print(f"  - model_best.h5: Best model based on validation loss")
                print(f"  - model_final.h5 or model_hybrid_final.h5: Final trained model")
                print(f"  - training_history.csv: Training metrics")
            
            if args.pwm:
                print(f"\nProbabilistic ACC generation used PWM from: {args.pwm}")
                if args.temperature:
                    print(f"  Temperature: {args.temperature}")
            
            if args.hybrid and unlabeled_path:
                path_obj = Path(unlabeled_path)
                if path_obj.is_dir():
                    print(f"\nPseudo-labels were generated from directory: {unlabeled_path}")
                else:
                    print(f"\nPseudo-labels were generated from file: {unlabeled_path}")
            
            print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
