"""
Standard Trainer for Tempest models.

This module implements the standard CRF-based training approach without
explicit length constraints, using the TempestConfig structure.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)

from tempest.config import TempestConfig, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class StandardTrainer:
    """
    Standard trainer for Tempest models using CRF architecture.
    
    This trainer implements the basic training pipeline for sequence
    annotation without explicit robustness features.
    """
    
    def __init__(
        self, 
        config: TempestConfig,
        output_dir: Optional[Path] = None,
        verbose: bool = False
    ):
        """
        Initialize the standard trainer.
        
        Args:
            config: TempestConfig object with model and training configuration
            output_dir: Directory to save models and logs
            verbose: Whether to show verbose training output
        """
        self.config = config
        self.model_config = config.model
        self.training_config = config.training if config.training else TrainingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.verbose = verbose
        self.model = None
        self.history = None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        logger.info(f"StandardTrainer initialized with output directory: {self.output_dir}")
    
    def build_model(self) -> Model:
        """
        Build the Tempest model architecture based on ModelConfig.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_seq = layers.Input(
            shape=(self.model_config.max_seq_len,), 
            dtype='int32',
            name='sequence_input'
        )
        
        # Embedding layer
        embedded = layers.Embedding(
            input_dim=5,  # A, C, G, T, N
            output_dim=self.model_config.embedding_dim,
            mask_zero=True,
            name='nucleotide_embedding'
        )(input_seq)
        
        # CNN layers for local pattern detection
        if self.model_config.use_cnn:
            conv1 = layers.Conv1D(
                filters=self.model_config.cnn_filters,
                kernel_size=self.model_config.cnn_kernel_size,
                activation='relu',
                padding='same',
                name='conv1'
            )(embedded)
            conv1 = layers.BatchNormalization()(conv1)
            conv1 = layers.Dropout(self.model_config.dropout * 0.5)(conv1)
            
            conv2 = layers.Conv1D(
                filters=64,
                kernel_size=5,
                activation='relu',
                padding='same',
                name='conv2'
            )(conv1)
            conv2 = layers.BatchNormalization()(conv2)
            conv2 = layers.Dropout(self.model_config.dropout * 0.5)(conv2)
            
            # Concatenate CNN features with embedding
            x = layers.Concatenate()([embedded, conv1, conv2])
        else:
            x = embedded
        
        # BiLSTM/LSTM layers
        for i in range(self.model_config.lstm_layers):
            return_sequences = True  # Always true for sequence labeling
            
            if self.model_config.use_bilstm:
                x = layers.Bidirectional(
                    layers.LSTM(
                        self.model_config.lstm_units,
                        return_sequences=return_sequences,
                        dropout=self.model_config.dropout,
                        recurrent_dropout=self.model_config.dropout * 0.5,
                        name=f'bilstm_{i+1}'
                    ),
                    name=f'bidirectional_{i+1}'
                )(x)
            else:
                x = layers.LSTM(
                    self.model_config.lstm_units,
                    return_sequences=return_sequences,
                    dropout=self.model_config.dropout,
                    recurrent_dropout=self.model_config.dropout * 0.5,
                    name=f'lstm_{i+1}'
                )(x)
            
            # Add batch norm between layers
            if i < self.model_config.lstm_layers - 1:
                x = layers.BatchNormalization()(x)
        
        # Optional attention mechanism
        if self.model_config.use_attention:
            attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.model_config.attention_units // 4,
                name='multi_head_attention'
            )(x, x)
            x = layers.Add()([x, attention])  # Skip connection
            x = layers.LayerNormalization()(x)
        
        # Output layer
        x = layers.Dropout(self.model_config.dropout)(x)
        output = layers.Dense(
            self.model_config.num_labels,
            activation='softmax',
            name='label_output'
        )(x)
        
        # Create model
        model = Model(inputs=input_seq, outputs=output, name='tempest_standard_model')
        
        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if self.verbose:
            model.summary()
        
        logger.info(f"Built model with {model.count_params():,} parameters")
        return model
    
    def _get_optimizer(self):
        """Get optimizer based on training config."""
        lr = self.training_config.learning_rate
        optimizer_name = self.training_config.optimizer.lower()
        
        if optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'adamw':
            return tf.keras.optimizers.AdamW(learning_rate=lr)
        elif optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
            return tf.keras.optimizers.Adam(learning_rate=lr)
    
    def _get_callbacks(self) -> List:
        """Create training callbacks based on config."""
        callbacks = []
        
        # Early stopping
        if self.training_config.early_stopping:
            early_stop_config = self.training_config.early_stopping
            if isinstance(early_stop_config, dict):
                callbacks.append(EarlyStopping(
                    monitor=early_stop_config.get('monitor', 'val_loss'),
                    patience=early_stop_config.get('patience', 10),
                    restore_best_weights=early_stop_config.get('restore_best', True),
                    verbose=1 if self.verbose else 0
                ))
            else:
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1 if self.verbose else 0
                ))
        
        # Model checkpointing
        if self.training_config.checkpoint:
            checkpoint_config = self.training_config.checkpoint
            if isinstance(checkpoint_config, dict):
                checkpoint_path = self.checkpoint_dir / checkpoint_config.get('filename', 'model_{epoch:02d}.h5')
                callbacks.append(ModelCheckpoint(
                    str(checkpoint_path),
                    monitor=checkpoint_config.get('monitor', 'val_loss'),
                    save_best_only=checkpoint_config.get('save_best_only', True),
                    save_weights_only=checkpoint_config.get('save_weights_only', False),
                    verbose=1 if self.verbose else 0
                ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1 if self.verbose else 0
        ))
        
        # TensorBoard
        if self.training_config.tensorboard:
            tb_config = self.training_config.tensorboard
            if isinstance(tb_config, dict):
                log_dir = self.log_dir / tb_config.get('log_dir', 'tensorboard')
                callbacks.append(TensorBoard(
                    log_dir=str(log_dir),
                    histogram_freq=tb_config.get('histogram_freq', 0),
                    write_graph=tb_config.get('write_graph', True),
                    update_freq=tb_config.get('update_freq', 'epoch')
                ))
        
        return callbacks
    
    def train(
        self,
        train_data: Union[np.ndarray, tuple, list],
        val_data: Optional[Union[np.ndarray, tuple, list]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data (X, y) or list of dicts
            val_data: Optional validation data (X, y) or list of dicts
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Process data based on input format
        if isinstance(train_data, (list, tuple)) and len(train_data) == 2:
            X_train, y_train = train_data
        elif isinstance(train_data, list) and isinstance(train_data[0], dict):
            X_train, y_train = self._process_dict_data(train_data)
        else:
            raise ValueError("train_data must be (X, y) tuple or list of dicts")
        
        if val_data is not None:
            if isinstance(val_data, (list, tuple)) and len(val_data) == 2:
                X_val, y_val = val_data
            elif isinstance(val_data, list) and isinstance(val_data[0], dict):
                X_val, y_val = self._process_dict_data(val_data)
            else:
                raise ValueError("val_data must be (X, y) tuple or list of dicts")
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Train model
        logger.info(f"Starting training for {self.training_config.epochs} epochs")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            callbacks=callbacks,
            verbose=1 if self.verbose else 2
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = self.output_dir / 'final_model.h5'
        self.model.save(str(final_model_path))
        logger.info(f"Saved final model to {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        
        # Prepare results
        results = {
            'model_path': str(final_model_path),
            'history': self.history.history,
            'training_time': training_time,
            'final_loss': self.history.history['loss'][-1],
            'final_accuracy': self.history.history['accuracy'][-1]
        }
        
        if validation_data is not None:
            results['final_val_loss'] = self.history.history['val_loss'][-1]
            results['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
        
        # Return metrics for CLI display
        results['metrics'] = {
            'Training Loss': results['final_loss'],
            'Training Accuracy': results['final_accuracy']
        }
        if 'final_val_loss' in results:
            results['metrics']['Validation Loss'] = results['final_val_loss']
            results['metrics']['Validation Accuracy'] = results['final_val_accuracy']
        
        return results
    
    def _process_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process dictionary-based data into arrays.
        
        Args:
            data: List of dictionaries with 'sequence' and 'labels' keys
            
        Returns:
            Tuple of (sequences, labels) arrays
        """
        sequences = []
        labels = []
        
        for item in data:
            seq = item['sequence']
            label = item.get('labels', [])
            
            # Convert sequence to numeric
            seq_numeric = self._sequence_to_numeric(seq)
            sequences.append(seq_numeric)
            
            # Convert labels to numeric if they're strings
            if isinstance(label, list) and len(label) > 0 and isinstance(label[0], str):
                label_numeric = self._labels_to_numeric(label)
                labels.append(label_numeric)
            else:
                labels.append(label)
        
        # Pad sequences
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.model_config.max_seq_len, padding='post'
        )
        
        # Pad labels and convert to categorical
        y = tf.keras.preprocessing.sequence.pad_sequences(
            labels, maxlen=self.model_config.max_seq_len, padding='post'
        )
        y = tf.keras.utils.to_categorical(y, num_classes=self.model_config.num_labels)
        
        return X, y
    
    def _sequence_to_numeric(self, sequence: str) -> List[int]:
        """Convert DNA sequence to numeric representation."""
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
        return [mapping.get(base.upper(), 0) for base in sequence]
    
    def _labels_to_numeric(self, labels: List[str]) -> List[int]:
        """Convert string labels to numeric representation."""
        # Define standard label mapping
        label_mapping = {
            'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4,
            'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10
        }
        return [label_mapping.get(label, 0) for label in labels]
    
    def predict(self, sequences: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Make predictions on new sequences.
        
        Args:
            sequences: Input sequences (numeric array or list of strings)
            
        Returns:
            Predicted label probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Process sequences if they're strings
        if isinstance(sequences, list) and isinstance(sequences[0], str):
            X = []
            for seq in sequences:
                X.append(self._sequence_to_numeric(seq))
            X = tf.keras.preprocessing.sequence.pad_sequences(
                X, maxlen=self.model_config.max_seq_len, padding='post'
            )
        else:
            X = sequences
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, test_data: Union[tuple, list]) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data (X, y) or list of dicts
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Process data
        if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
            X_test, y_test = test_data
        elif isinstance(test_data, list) and isinstance(test_data[0], dict):
            X_test, y_test = self._process_dict_data(test_data)
        else:
            raise ValueError("test_data must be (X, y) tuple or list of dicts")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'test_loss': results[0],
            'test_accuracy': results[1]
        }
    
    def save(self, path: Union[str, Path]):
        """Save model to specified path."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load model from specified path."""
        self.model = tf.keras.models.load_model(str(path))
        logger.info(f"Model loaded from {path}")


def run_training(
    config: TempestConfig,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run standard training based on configuration.
    
    This function is called by the CLI and main.py to execute training.
    
    Args:
        config: TempestConfig object
        output_dir: Output directory for models
        **kwargs: Additional training parameters (train_data, val_data, etc.)
        
    Returns:
        Training results dictionary
    """
    # Extract data from kwargs
    train_data = kwargs.get('train_data')
    val_data = kwargs.get('val_data')
    verbose = kwargs.get('verbose', False)
    
    # Check if we're running hybrid mode
    mode = kwargs.get('mode', 'standard')
    
    if mode == 'hybrid':
        # Import and use HybridTrainer
        from tempest.training.hybrid_trainer import HybridTrainer, run_hybrid_training
        return run_hybrid_training(config, output_dir=output_dir, **kwargs)
    elif mode == 'ensemble':
        # Import and use EnsembleTrainer
        from tempest.training.ensemble import EnsembleTrainer, run_ensemble_training
        return run_ensemble_training(config, output_dir=output_dir, **kwargs)
    else:
        # Use standard trainer
        trainer = StandardTrainer(config, output_dir=output_dir, verbose=verbose)
        
        # If no data provided, try to load from config paths
        if train_data is None and config.training and hasattr(config.training, 'train_data'):
            from tempest.main import load_data
            train_data = load_data(config.training.train_data)
        
        if val_data is None and config.training and hasattr(config.training, 'val_data'):
            from tempest.main import load_data
            val_data = load_data(config.training.val_data)
        
        if train_data is None:
            raise ValueError("No training data provided")
        
        # Run training
        return trainer.train(train_data, val_data, **kwargs)
