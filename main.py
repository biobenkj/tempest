#!/usr/bin/env python3
"""
Main training script for Tempest.

This script demonstrates the complete pipeline:
1. Load configuration from YAML
2. Simulate training data with ACC PWM
3. Build model architecture
4. Train model
5. Evaluate and save results
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add tempest to path
sys.path.insert(0, str(Path(__file__).parent))

from tempest.utils import load_config, ensure_dir
from tempest.data import SequenceSimulator, reads_to_arrays
from tempest.core.models import build_model_from_config, print_model_summary


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
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✓ Configured GPU memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.info("No GPUs found - using CPU")


def pad_sequences(sequences: np.ndarray, labels: np.ndarray, max_length: int) -> tuple:
    """
    Pad sequences to max_length.
    
    Args:
        sequences: Array of shape (num_sequences, seq_length)
        labels: Array of shape (num_sequences, seq_length)
        max_length: Target length
        
    Returns:
        Tuple of (padded_sequences, padded_labels)
    """
    num_sequences = sequences.shape[0]
    current_length = sequences.shape[1]
    
    if current_length == max_length:
        return sequences, labels
    
    # Create padded arrays
    padded_sequences = np.zeros((num_sequences, max_length), dtype=sequences.dtype)
    padded_labels = np.zeros((num_sequences, max_length), dtype=labels.dtype)
    
    # Copy data
    copy_length = min(current_length, max_length)
    padded_sequences[:, :copy_length] = sequences[:, :copy_length]
    padded_labels[:, :copy_length] = labels[:, :copy_length]
    
    return padded_sequences, padded_labels


def convert_labels_to_categorical(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        labels: Array of shape (num_samples, seq_length) with integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array of shape (num_samples, seq_length, num_classes)
    """
    num_samples, seq_length = labels.shape
    categorical = np.zeros((num_samples, seq_length, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        for j in range(seq_length):
            categorical[i, j, labels[i, j]] = 1.0
    
    return categorical


def prepare_data(config, pwm_file=None):
    """
    Simulate and prepare training data.
    
    Args:
        config: TempestConfig object
        pwm_file: Optional path to PWM file
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, label_to_idx)
    """
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
    
    logger.info(f"✓ Data preparation complete")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  y_val: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, label_to_idx


def build_model(config):
    """
    Build model from configuration.
    
    Args:
        config: TempestConfig object
        
    Returns:
        Keras Model
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: MODEL BUILDING")
    logger.info("="*80)
    
    model = build_model_from_config(config)
    print_model_summary(model)
    
    return model


def compile_model(model, config):
    """
    Compile model with optimizer and loss.
    
    Args:
        model: Keras Model
        config: TempestConfig object
    """
    logger.info("Compiling model...")
    
    # Optimizer
    if config.training.optimizer.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    elif config.training.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=config.training.learning_rate)
    else:
        logger.warning(f"Unknown optimizer '{config.training.optimizer}', using Adam")
        optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    
    # Loss and metrics
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"✓ Model compiled with {config.training.optimizer} optimizer")


def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train model with callbacks.
    
    Args:
        model: Keras Model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: TempestConfig object
        
    Returns:
        Training history
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*80)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    ensure_dir(str(checkpoint_dir))
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = checkpoint_dir / "model_best.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=config.training.reduce_lr_patience,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSV logger
    csv_path = checkpoint_dir / "training_history.csv"
    csv_logger = keras.callbacks.CSVLogger(str(csv_path))
    callbacks.append(csv_logger)
    
    # Train
    logger.info(f"Training for up to {config.training.epochs} epochs...")
    logger.info(f"Batch size: {config.model.batch_size}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("✓ Training complete")
    
    return history


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation set.
    
    Args:
        model: Trained Keras Model
        X_val, y_val: Validation data
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("="*80)
    
    logger.info("Evaluating on validation set...")
    results = model.evaluate(X_val, y_val, verbose=0)
    
    logger.info(f"✓ Validation Loss: {results[0]:.4f}")
    logger.info(f"✓ Validation Accuracy: {results[1]:.4f}")


def save_model(model, config, label_to_idx):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained Keras Model
        config: TempestConfig object
        label_to_idx: Label to index mapping
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SAVING MODEL")
    logger.info("="*80)
    
    checkpoint_dir = Path(config.training.checkpoint_dir)
    
    # Save final model
    final_model_path = checkpoint_dir / "model_final.h5"
    model.save(str(final_model_path))
    logger.info(f"✓ Saved final model to: {final_model_path}")
    
    # Save label mapping
    import json
    label_map_path = checkpoint_dir / "label_mapping.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    logger.info(f"✓ Saved label mapping to: {label_map_path}")
    
    # Save configuration
    config_path = checkpoint_dir / "config.yaml"
    config.to_yaml(str(config_path))
    logger.info(f"✓ Saved configuration to: {config_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train Tempest sequence annotation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
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
        help='Output directory for checkpoints (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*25 + "TEMPEST TRAINING PIPELINE")
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
    if pwm_file is None and config.pwm and config.pwm.pwm_file:
        pwm_file = config.pwm.pwm_file
    
    if pwm_file:
        logger.info(f"Using PWM file: {pwm_file}")
    else:
        logger.info("No PWM file specified - ACC sequences will be random or from priors")
    
    logger.info("✓ Configuration loaded\n")
    
    # Run pipeline
    try:
        # 1. Prepare data
        X_train, y_train, X_val, y_val, label_to_idx = prepare_data(config, pwm_file)
        
        # 2. Build model
        model = build_model(config)
        
        # 3. Compile model
        compile_model(model, config)
        
        # 4. Train model
        history = train_model(model, X_train, y_train, X_val, y_val, config)
        
        # 5. Evaluate model
        evaluate_model(model, X_val, y_val)
        
        # 6. Save model
        save_model(model, config, label_to_idx)
        
        # Print summary
        print("\n" + "="*80)
        print(" "*30 + "TRAINING COMPLETE")
        print("="*80)
        print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
        print(f"  - model_best.h5: Best model based on validation loss")
        print(f"  - model_final.h5: Final model after training")
        print(f"  - training_history.csv: Training metrics")
        print(f"  - label_mapping.json: Label to index mapping")
        print(f"  - config.yaml: Configuration used for training")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
