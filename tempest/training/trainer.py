"""
Single model trainer for Tempest.

Provides a clean training interface with callbacks, metrics tracking,
and model serialization.

Part of: tempest/training/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as keras_callbacks
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime

from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.utils.config import TempestConfig
from tempest.utils.io import ensure_dir
from tempest.training.hybrid_trainer import (
    build_model_from_config,
    pad_sequences,
    convert_labels_to_categorical,
    print_model_summary
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Main trainer class for single model training.
    
    Handles the complete training pipeline including:
    - Model building from config
    - Training and validation loops
    - Callbacks (early stopping, learning rate reduction, checkpointing)
    - Metrics tracking and logging
    - Model saving and loading
    """
    
    def __init__(self, config: TempestConfig, checkpoint_dir: str = "checkpoints"):
        """
        Initialize trainer.
        
        Args:
            config: TempestConfig object with all settings
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        ensure_dir(str(self.checkpoint_dir))
        
        # Training parameters
        self.epochs = config.training.epochs if config.training else 50
        self.batch_size = config.model.batch_size
        self.learning_rate = config.training.learning_rate if config.training else 0.001
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 10)
        self.reduce_lr_patience = getattr(config.training, 'reduce_lr_patience', 5)
        
        # Initialize tracking
        self.training_history = {}
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # Model will be built during training
        self.model = None
        self.label_to_idx = None
        self.idx_to_label = None
    
    def _prepare_data(self, 
                     train_reads: List[SimulatedRead],
                     val_reads: Optional[List[SimulatedRead]] = None) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            train_reads: Training SimulatedRead objects
            val_reads: Optional validation SimulatedRead objects
            
        Returns:
            Tuple of prepared arrays (X_train, y_train, X_val, y_val)
        """
        logger.info("Preparing training data...")
        
        # Convert to arrays
        X_train, y_train, self.label_to_idx = reads_to_arrays(train_reads)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        if val_reads:
            X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx=self.label_to_idx)
        else:
            # Split training data for validation
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            logger.info("No validation data provided, using 20% of training data")
        
        # Pad sequences
        max_len = self.config.model.max_seq_len
        X_train, y_train = pad_sequences(X_train, y_train, max_len)
        X_val, y_val = pad_sequences(X_val, y_val, max_len)
        
        # Convert labels to categorical
        num_labels = self.config.model.num_labels
        y_train_cat = convert_labels_to_categorical(y_train, num_labels)
        y_val_cat = convert_labels_to_categorical(y_val, num_labels)
        
        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        logger.info(f"Number of labels: {num_labels}")
        
        return X_train, y_train_cat, X_val, y_val_cat, y_train, y_val
    
    def _get_callbacks(self) -> List[keras_callbacks.Callback]:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / "model_checkpoint_{epoch:02d}.h5"
        callbacks.append(keras_callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(keras_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Reduce learning rate
        callbacks.append(keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ))
        
        # TensorBoard logging
        log_dir = self.checkpoint_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(keras_callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
        
        # Custom callback for per-label metrics
        callbacks.append(PerLabelMetrics(self.idx_to_label))
        
        return callbacks
    
    def train(self,
             train_reads: List[SimulatedRead],
             val_reads: Optional[List[SimulatedRead]] = None) -> keras.Model:
        """
        Train the model.
        
        Args:
            train_reads: Training SimulatedRead objects
            val_reads: Optional validation SimulatedRead objects
            
        Returns:
            Trained Keras model
        """
        logger.info("="*80)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*80)
        
        # Prepare data
        X_train, y_train_cat, X_val, y_val_cat, y_train_raw, y_val_raw = self._prepare_data(
            train_reads, val_reads
        )
        
        # Build model
        logger.info("Building model from configuration...")
        self.model = build_model_from_config(self.config)
        print_model_summary(self.model)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Check if we should use CRF loss (if available)
        if self.config.model.use_crf:
            try:
                from tensorflow_addons.layers import CRF
                from tensorflow_addons.losses import crf_loss
                loss = crf_loss
                logger.info("Using CRF loss function")
            except ImportError:
                loss = 'categorical_crossentropy'
                logger.warning("tensorflow-addons not available, using categorical crossentropy")
        else:
            loss = 'categorical_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', PerTokenAccuracy()]
        )
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Train model
        logger.info(f"Training for {self.epochs} epochs with batch size {self.batch_size}")
        
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        # Evaluate final model
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION")
        logger.info("="*60)
        
        results = self.model.evaluate(X_val, y_val_cat, verbose=0)
        self.best_val_loss = results[0]
        self.best_val_accuracy = results[1]
        
        logger.info(f"Final Validation Loss: {self.best_val_loss:.4f}")
        logger.info(f"Final Validation Accuracy: {self.best_val_accuracy:.4f}")
        
        # Save final model
        self.save_model()
        
        return self.model
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save the trained model and metadata.
        
        Args:
            filepath: Optional custom filepath for the model
        """
        if not self.model:
            raise ValueError("No model to save. Train a model first.")
        
        # Default filepath
        if filepath is None:
            filepath = self.checkpoint_dir / "model_final.h5"
        else:
            filepath = Path(filepath)
        
        # Save model
        self.model.save(str(filepath))
        logger.info(f"Model saved to: {filepath}")
        
        # Save metadata
        metadata = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'best_val_loss': float(self.best_val_loss),
            'best_val_accuracy': float(self.best_val_accuracy),
            'training_history': {
                k: [float(v) for v in vals] 
                for k, vals in self.training_history.items()
            },
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        metadata_path = filepath.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model and metadata.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        # Load model
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from: {filepath}")
        
        # Load metadata if available
        metadata_path = filepath.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.label_to_idx = metadata.get('label_to_idx', {})
            self.idx_to_label = metadata.get('idx_to_label', {})
            self.best_val_loss = metadata.get('best_val_loss', float('inf'))
            self.best_val_accuracy = metadata.get('best_val_accuracy', 0.0)
            self.training_history = metadata.get('training_history', {})
            
            logger.info(f"Metadata loaded from: {metadata_path}")
    
    def predict(self, reads: List[SimulatedRead]) -> np.ndarray:
        """
        Make predictions on new reads.
        
        Args:
            reads: List of SimulatedRead objects
            
        Returns:
            Predictions array [num_reads, seq_len, num_labels]
        """
        if not self.model:
            raise ValueError("No model available. Train or load a model first.")
        
        # Convert to arrays
        X, _, _ = reads_to_arrays(reads, label_to_idx=self.label_to_idx)
        
        # Pad sequences
        max_len = self.config.model.max_seq_len
        X, _ = pad_sequences(X, np.zeros_like(X), max_len)
        
        # Predict
        predictions = self.model.predict(X, batch_size=self.batch_size)
        
        return predictions


class PerTokenAccuracy(keras.metrics.Metric):
    """
    Custom metric for per-token accuracy in sequence labeling.
    
    Ignores padding tokens in accuracy calculation.
    """
    
    def __init__(self, name='per_token_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get predicted classes
        y_pred_class = tf.argmax(y_pred, axis=-1)
        y_true_class = tf.argmax(y_true, axis=-1)
        
        # Create mask for non-padding tokens (assuming 0 is padding)
        mask = tf.reduce_sum(y_true, axis=-1) > 0
        
        # Calculate accuracy only on non-padding tokens
        matches = tf.equal(y_true_class, y_pred_class)
        matches = tf.cast(matches, tf.float32)
        matches = matches * tf.cast(mask, tf.float32)
        
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))
    
    def result(self):
        return self.total / (self.count + 1e-7)
    
    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


class PerLabelMetrics(keras.callbacks.Callback):
    """
    Callback to track per-label accuracy during training.
    """
    
    def __init__(self, idx_to_label: Dict[int, str]):
        super().__init__()
        self.idx_to_label = idx_to_label
        self.num_labels = len(idx_to_label)
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate and log per-label metrics at epoch end."""
        # Get validation data
        if hasattr(self.model, 'validation_data') and self.model.validation_data:
            x_val, y_val = self.model.validation_data[0], self.model.validation_data[1]
            
            # Make predictions
            y_pred = self.model.predict(x_val, verbose=0)
            
            # Calculate per-label accuracy
            y_pred_class = np.argmax(y_pred, axis=-1)
            y_true_class = np.argmax(y_val, axis=-1)
            
            # Mask for non-padding
            mask = np.sum(y_val, axis=-1) > 0
            
            per_label_acc = {}
            for label_idx, label_name in self.idx_to_label.items():
                # Find positions with this label
                label_mask = (y_true_class == label_idx) & mask
                if np.any(label_mask):
                    correct = np.sum((y_pred_class == label_idx) & label_mask)
                    total = np.sum(label_mask)
                    per_label_acc[label_name] = correct / total if total > 0 else 0.0
            
            # Log results
            if per_label_acc:
                logger.info(f"\nEpoch {epoch + 1} - Per-label accuracy:")
                for label, acc in sorted(per_label_acc.items()):
                    logger.info(f"  {label:15s}: {acc:.4f}")
