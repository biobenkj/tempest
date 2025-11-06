#!/usr/bin/env python3
"""
Main training script for Tempest with hybrid training support.

Supports both standard and hybrid robustness training modes.
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import numpy first
import numpy as np

# Configure TensorFlow suppression if not already done
if os.getenv('TEMPEST_DEBUG', '0') != '1' and os.getenv('TF_CPP_MIN_LOG_LEVEL') != '3':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import TensorFlow
import tensorflow as tf
if os.getenv('TEMPEST_DEBUG', '0') != '1':
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    
    # Try to suppress absl logging if available
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

from tensorflow import keras

# Suppress TensorFlow Addons warnings
warnings.filterwarnings("ignore", message=".*TensorFlow Addons.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

# Import from tempest modules
from tempest.utils import load_config, ensure_dir
from tempest.data import SequenceSimulator, reads_to_arrays
from tempest.core import build_model_from_config, print_model_summary
from tempest.training import HybridTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Configured GPU memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.info("No GPUs found - using CPU")


def pad_sequences(sequences: np.ndarray, labels: np.ndarray, max_length: int) -> tuple:
    """Pad sequences to max_length."""
    num_sequences = sequences.shape[0]
    current_length = sequences.shape[1]
    
    if current_length == max_length:
        return sequences, labels
    
    padded_sequences = np.zeros((num_sequences, max_length), dtype=sequences.dtype)
    padded_labels = np.zeros((num_sequences, max_length), dtype=labels.dtype)
    
    copy_length = min(current_length, max_length)
    padded_sequences[:, :copy_length] = sequences[:, :copy_length]
    padded_labels[:, :copy_length] = labels[:, :copy_length]
    
    return padded_sequences, padded_labels


def convert_labels_to_categorical(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    num_samples, seq_length = labels.shape
    categorical = np.zeros((num_samples, seq_length, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        for j in range(seq_length):
            categorical[i, j, labels[i, j]] = 1.0
    
    return categorical


def prepare_data(config, pwm_file=None):
    """Simulate and prepare training data."""
    logger.info("="*80)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*80)
    
    # Initialize simulator
    simulator = SequenceSimulator(config.simulation, pwm_file=pwm_file)
    
    # Generate data
    train_reads, val_reads = simulator.generate_train_val_split(
        train_fraction=config.training.train_split
    )
    
    # Convert to arrays
    logger.info("Converting reads to arrays...")
    X_train, y_train, label_to_idx = reads_to_arrays(train_reads)
    X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx=label_to_idx)
    
    logger.info(f"  Training set: {X_train.shape}")
    logger.info(f"  Validation set: {X_val.shape}")
    logger.info(f"  Number of labels: {len(label_to_idx)}")
    logger.info(f"  Label mapping: {label_to_idx}")
    
    # Pad to max_seq_len
    max_len = config.model.max_seq_len
    if X_train.shape[1] != max_len:
        logger.info(f"Padding sequences to {max_len}...")
        X_train, y_train = pad_sequences(X_train, y_train, max_len)
        X_val, y_val = pad_sequences(X_val, y_val, max_len)
    
    # Convert labels to categorical
    logger.info("Converting labels to one-hot encoding...")
    y_train = convert_labels_to_categorical(y_train, config.model.num_labels)
    y_val = convert_labels_to_categorical(y_val, config.model.num_labels)
    
    logger.info(f"Data preparation complete")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  y_val: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads


def train_standard(config, X_train, y_train, X_val, y_val, label_to_idx):
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


def train_hybrid(config, train_reads, val_reads, unlabeled_fastq=None):
    """Hybrid robustness training with invalid reads and pseudo-labels."""
    logger.info("\n" + "="*80)
    logger.info("HYBRID ROBUSTNESS TRAINING MODE")
    logger.info("="*80)
    
    if not config.hybrid or not config.hybrid.enabled:
        logger.warning("Hybrid training requested but not enabled in config!")
        logger.warning("Add 'hybrid:' section to config or use --config hybrid_config.yaml")
        return None
    
    # Initialize hybrid trainer
    trainer = HybridTrainer(config)
    
    # Run hybrid training
    model = trainer.train(
        train_reads=train_reads,
        val_reads=val_reads,
        unlabeled_fastq=unlabeled_fastq,
        checkpoint_dir=config.training.checkpoint_dir
    )
    
    return model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Tempest - Modular sequence annotation using length-constrained CRFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEMPEST OVERVIEW:
-----------------
Tempest is a deep learning framework for sequence annotation that combines:
  - Conditional Random Fields (CRFs) for structured prediction
  - Length constraints to enforce biologically meaningful segment sizes
  - Position Weight Matrix (PWM) priors for incorporating domain knowledge
  - Hybrid training modes for improved robustness

TRAINING MODES:
---------------
1. Standard Mode (default):
   - Basic supervised training with CRF layers
   - Uses simulated or provided sequence data
   - Suitable for clean, well-labeled data

2. Hybrid Mode (--hybrid):
   - Advanced training with invalid sequence handling
   - Pseudo-label generation for unlabeled data
   - Improved robustness to noisy real-world sequences
   - Requires hybrid configuration section in config file

CONFIGURATION:
--------------
Training is controlled via YAML configuration files:
  - config.yaml - Standard training configuration
  - hybrid_config.yaml - Hybrid training with robustness features
  - config_with_whitelists.yaml - Training with sequence constraints

Example config files are provided in the config/ directory.

EXAMPLES:
---------
Standard training:
  tempest --config config/train_config.yaml

Hybrid training with PWM:
  tempest --config config/hybrid_config.yaml --hybrid --pwm acc_pwm.txt

Training with unlabeled data:
  tempest --config config/hybrid_config.yaml --hybrid --unlabeled reads.fastq

Custom output directory:
  tempest --config config/train_config.yaml --output-dir ./my_model

For more information, visit: https://github.com/biobenkj/tempest
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (required)'
    )
    parser.add_argument(
        '--pwm',
        type=str,
        default=None,
        help='Path to PWM file for ACC generation (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for model checkpoints (overrides config)'
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable hybrid robustness training mode'
    )
    parser.add_argument(
        '--unlabeled',
        type=str,
        default=None,
        help='Path to unlabeled FASTQ file for pseudo-labeling (hybrid mode only)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*25 + "TEMPEST TRAINING PIPELINE")
    if args.hybrid:
        print(" "*25 + "(HYBRID ROBUSTNESS MODE)")
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
    
    # Determine PWM file
    pwm_file = args.pwm
    if pwm_file is None and hasattr(config, 'pwm') and hasattr(config.pwm, 'pwm_file'):
        pwm_file = config.pwm.pwm_file
    
    if pwm_file:
        logger.info(f"Using PWM file: {pwm_file}")
    else:
        logger.info("No PWM file specified - ACC sequences will be random or from priors")
    
    logger.info("Configuration loaded\n")
    
    # Run pipeline
    try:
        # Prepare data
        X_train, y_train, X_val, y_val, label_to_idx, train_reads, val_reads = prepare_data(config, pwm_file)
        
        # Train based on mode
        if args.hybrid:
            model = train_hybrid(config, train_reads, val_reads, args.unlabeled)
        else:
            model = train_standard(config, X_train, y_train, X_val, y_val, label_to_idx)
        
        # Summary
        print("\n" + "="*80)
        print(" "*30 + "TRAINING COMPLETE")
        print("="*80)
        print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
        print(f"  - model_best.h5: Best model based on validation loss")
        print(f"  - model_final.h5 (or model_hybrid_final.h5): Final trained model")
        print(f"  - training_history.csv: Training metrics")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
