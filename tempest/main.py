#!/usr/bin/env python3
"""
Tempest Training Pipeline with Directory Support for Pseudo-Labeling.

This script supports:
1. Passing a directory of FASTQ files for pseudo-label training
2. Flexible input handling (single file or directory)
3. Configurable batch processing parameters
4. Probabilistic PWM-based ACC generation

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
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np
from tensorflow import keras

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
        logger.info(f"Detected directory input: {path_string}")
        # Check for FASTQ files
        patterns = ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz"]
        file_count = sum(len(list(path.glob(p))) for p in patterns)
        if file_count > 0:
            logger.info(f"Found {file_count} FASTQ files in directory")
        else:
            logger.warning("No FASTQ files found in directory")
    elif path.is_file():
        logger.info(f"Detected single file input: {path_string}")
    
    return path


def prepare_data(config: TempestConfig, pwm_file: Optional[str] = None) -> Tuple:
    """
    Prepare training and validation data with probabilistic PWM support.
    
    Args:
        config: Tempest configuration
        pwm_file: Optional PWM file for ACC generation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads)
    """
    logger.info("="*80)
    logger.info("DATA PREPARATION")
    logger.info("="*80)
    
    # Convert TempestConfig to dict for SequenceSimulator
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
    
    # Update PWM configuration if PWM file provided
    if pwm_file and Path(pwm_file).exists():
        logger.info(f"Configuring PWM-based ACC generation from: {pwm_file}")
        
        # Ensure pwm section exists in config
        if 'pwm' not in config_dict:
            config_dict['pwm'] = {}
        
        # Set PWM file and parameters
        config_dict['pwm']['pwm_file'] = pwm_file
        
        # Use temperature for diversity control (replaces old threshold approach)
        if 'temperature' not in config_dict['pwm']:
            config_dict['pwm']['temperature'] = 1.0  # Default temperature
        if 'min_entropy' not in config_dict['pwm']:
            config_dict['pwm']['min_entropy'] = 0.1  # Minimum diversity
        
        logger.info(f"PWM temperature: {config_dict['pwm']['temperature']}")
        logger.info(f"PWM min_entropy: {config_dict['pwm']['min_entropy']}")
    
    # Ensure simulation section exists
    if 'simulation' not in config_dict:
        config_dict['simulation'] = {}
    
    # Set simulation parameters from config
    sim_config = config_dict['simulation']
    sim_config['n_train'] = getattr(config.simulation, 'n_train', 10000)
    sim_config['n_val'] = getattr(config.simulation, 'n_val', 2000)
    sim_config['random_seed'] = getattr(config.simulation, 'random_seed', 42)
    
    # Set sequence architecture if not present
    if 'sequence_order' not in sim_config:
        sim_config['sequence_order'] = getattr(
            config.simulation, 
            'sequence_order', 
            ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
        )
    
    # Initialize sequence simulator with probabilistic ACC support
    simulator = SequenceSimulator(config=config_dict)
    
    # Log ACC generator status
    if simulator.acc_generator:
        logger.info("Probabilistic ACC generator initialized successfully")
        if hasattr(simulator.acc_generator, 'temperature'):
            logger.info(f"  Temperature: {simulator.acc_generator.temperature}")
        if hasattr(simulator.acc_generator, 'min_entropy'):
            logger.info(f"  Min entropy: {simulator.acc_generator.min_entropy}")
    else:
        logger.info("ACC sequences will be generated randomly or from fixed patterns")
    
    # Generate training data
    logger.info(f"Generating {sim_config['n_train']} training reads...")
    train_reads = simulator.generate_batch(
        n=sim_config['n_train'],
        diversity_schedule='random',  # Use random diversity for training variety
        include_quality=False  # Can enable if quality scores needed
    )
    
    logger.info(f"Generating {sim_config['n_val']} validation reads...")
    val_reads = simulator.generate_batch(
        n=sim_config['n_val'],
        diversity_schedule=None,  # Default diversity for validation
        include_quality=False
    )
    
    # Analyze ACC diversity if ACC sequences present
    if any('ACC' in read.label_regions for read in train_reads[:100]):
        logger.info("Analyzing ACC diversity in generated reads...")
        diversity_metrics = simulator.analyze_acc_diversity(train_reads[:1000])
        if 'error' not in diversity_metrics:
            logger.info(f"  Unique ACC sequences: {diversity_metrics.get('unique_sequences', 'N/A')}")
            logger.info(f"  Uniqueness ratio: {diversity_metrics.get('uniqueness_ratio', 0):.3f}")
            if 'mean_pwm_score' in diversity_metrics:
                logger.info(f"  Mean PWM score: {diversity_metrics['mean_pwm_score']:.3f}")
    
    # Convert to arrays
    logger.info("Converting reads to arrays...")
    X_train, y_train, label_to_idx = reads_to_arrays(train_reads)
    X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx=label_to_idx)
    
    # Pad sequences
    max_len = config.model.max_seq_len
    X_train, y_train = pad_sequences(X_train, y_train, max_len)
    X_val, y_val = pad_sequences(X_val, y_val, max_len)
    
    # Convert labels to categorical
    num_labels = config.model.num_labels
    y_train = convert_labels_to_categorical(y_train, num_labels)
    y_val = convert_labels_to_categorical(y_val, num_labels)
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Label mapping: {label_to_idx}")
    
    return X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads


def train_standard(config: TempestConfig, X_train, y_train, X_val, y_val, 
                  label_to_idx) -> keras.Model:
    """Standard training without hybrid robustness."""
    logger.info("\n" + "="*80)
    logger.info("STANDARD TRAINING MODE")
    logger.info("="*80)
    
    # Build model
    model = build_model_from_config(config)
    print_model_summary(model)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_dir = Path(config.training.checkpoint_dir)
    ensure_dir(str(checkpoint_dir))
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "model_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.training.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(str(checkpoint_dir / "training_history.csv"))
    ]
    
    # Train
    logger.info(f"Training for up to {config.training.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    results = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"\nValidation Loss: {results[0]:.4f}")
    logger.info(f"Validation Accuracy: {results[1]:.4f}")
    
    # Save
    model.save(str(checkpoint_dir / "model_final.h5"))
    logger.info(f"Saved model to: {checkpoint_dir}")
    
    return model


def train_hybrid(config: TempestConfig, train_reads: List[SimulatedRead],
                val_reads: List[SimulatedRead],
                unlabeled_path: Optional[Union[str, Path]] = None,
                max_pseudo_per_file: Optional[int] = None,
                max_pseudo_total: Optional[int] = None) -> keras.Model:
    """
    Hybrid robustness training with directory support.
    
    Args:
        config: Tempest configuration
        train_reads: Training reads
        val_reads: Validation reads
        unlabeled_path: Path to FASTQ file or directory
        max_pseudo_per_file: Maximum reads per file (for directory)
        max_pseudo_total: Maximum total reads (for directory)
        
    Returns:
        Trained model
    """
    logger.info("\n" + "="*80)
    logger.info("HYBRID ROBUSTNESS TRAINING MODE")
    logger.info("="*80)
    
    if not config.hybrid or not config.hybrid.enabled:
        logger.warning("Hybrid training requested but not enabled in config!")
        logger.warning("Add 'hybrid:' section with 'enabled: true' to config")
        return None
    
    # Update config with command-line parameters if provided
    if max_pseudo_per_file is not None:
        config.hybrid.max_pseudo_per_file = max_pseudo_per_file
        logger.info(f"Set max_pseudo_per_file to {max_pseudo_per_file}")
    
    if max_pseudo_total is not None:
        config.hybrid.max_pseudo_total = max_pseudo_total
        logger.info(f"Set max_pseudo_total to {max_pseudo_total}")
    
    # Initialize hybrid trainer
    trainer = HybridTrainer(config)
    
    # Run hybrid training
    model = trainer.train(
        train_reads=train_reads,
        val_reads=val_reads,
        unlabeled_path=unlabeled_path,
        checkpoint_dir=config.training.checkpoint_dir
    )
    
    return model


def main():
    """Main training pipeline with directory support and probabilistic PWM."""
    parser = argparse.ArgumentParser(
        description='Train Tempest sequence annotation model with probabilistic ACC generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--pwm',
        type=str,
        default=None,
        help='Path to PWM file for probabilistic ACC generation (overrides config)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Temperature for PWM diversity (lower=more conservative, higher=more diverse)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for checkpoints (overrides config)'
    )
    
    # Hybrid training arguments
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid robustness training (requires config.hybrid section)'
    )
    
    # Unlabeled data arguments (mutually exclusive)
    unlabeled_group = parser.add_mutually_exclusive_group()
    unlabeled_group.add_argument(
        '--unlabeled',
        type=str,
        default=None,
        help='Path to unlabeled FASTQ file OR directory for pseudo-label training'
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
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*25 + "TEMPEST TRAINING PIPELINE")
    print(" "*20 + "with Probabilistic PWM ACC Generation")
    if args.hybrid:
        print(" "*15 + "(HYBRID ROBUSTNESS MODE WITH DIRECTORY SUPPORT)")
    print("="*80 + "\n")
    
    # Setup GPU
    setup_gpu()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.training.checkpoint_dir = args.output_dir
        logger.info(f"Output directory overridden to: {args.output_dir}")
    
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
    
    # Handle unlabeled data path
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
        if args.hybrid:
            model = train_hybrid(
                config, 
                train_reads, 
                val_reads, 
                unlabeled_path,
                args.max_pseudo_per_file,
                args.max_pseudo_total
            )
        else:
            model = train_standard(config, X_train, y_train, X_val, y_val, label_to_idx)
        
        if model is not None:
            # Summary
            print("\n" + "="*80)
            print(" "*30 + "TRAINING COMPLETE")
            print("="*80)
            print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
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
