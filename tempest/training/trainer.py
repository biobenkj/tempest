"""
Standard Trainer for Tempest models (refactored to use build_model_from_config).

This refactor removes direct architecture construction and instead calls the
core.models.build_model_from_config() factory to centralize all model logic.
All original convenience features (callbacks, IO, preprocessing, etc.) are preserved.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)

from tempest.config import TempestConfig, TrainingConfig
from tempest.core.models import build_model_from_config

logger = logging.getLogger(__name__)


class StandardTrainer:
    def __init__(self, config: TempestConfig, output_dir: Optional[Path] = None, verbose: bool = False):
        self.config = config
        self.model_config = config.model
        self.training_config = config.training if config.training else TrainingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.verbose = verbose
        self.model: Optional[Model] = None
        self.history = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

        logger.info(f"StandardTrainer initialized with output directory: {self.output_dir}")

    def build_model(self) -> Model:
        """Build the Tempest model via the centralized model factory."""
        logger.info("Building model using core.models.build_model_from_config()...")
        try:
            model = build_model_from_config(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to construct model from config: {e}")

        # Compile the model with optimizer, loss, and metrics
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        if self.verbose:
            model.summary()
        logger.info(f"Model constructed: {type(model).__name__}")
        logger.info(f"Model compiled with {model.count_params():,} parameters")
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
        callbacks = []

        if self.training_config.early_stopping:
            cfg = self.training_config.early_stopping
            if isinstance(cfg, dict):
                callbacks.append(EarlyStopping(
                    monitor=cfg.get('monitor', 'val_loss'),
                    patience=cfg.get('patience', 10),
                    restore_best_weights=cfg.get('restore_best', True),
                    verbose=1 if self.verbose else 0))
            else:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

        if self.training_config.checkpoint:
            cfg = self.training_config.checkpoint
            checkpoint_path = self.checkpoint_dir / cfg.get('filename', 'model_{epoch:02d}.h5')
            callbacks.append(ModelCheckpoint(
                str(checkpoint_path),
                monitor=cfg.get('monitor', 'val_loss'),
                save_best_only=cfg.get('save_best_only', True),
                save_weights_only=cfg.get('save_weights_only', False),
                verbose=1 if self.verbose else 0))

        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1 if self.verbose else 0))

        if self.training_config.tensorboard:
            cfg = self.training_config.tensorboard
            log_dir = self.log_dir / cfg.get('log_dir', 'tensorboard')
            callbacks.append(TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=cfg.get('histogram_freq', 0),
                write_graph=cfg.get('write_graph', True),
                update_freq=cfg.get('update_freq', 'epoch')))

        return callbacks

    def train(self, train_data: Union[np.ndarray, tuple, list], val_data: Optional[Union[np.ndarray, tuple, list]] = None, **kwargs) -> Dict[str, Any]:
        if isinstance(train_data, (list, tuple)) and len(train_data) == 2:
            X_train, y_train = train_data
        elif isinstance(train_data, list) and isinstance(train_data[0], dict):
            X_train, y_train = self._process_dict_data(train_data)
        else:
            raise ValueError("train_data must be (X, y) tuple or list of dicts")

        validation_data = None
        if val_data is not None:
            if isinstance(val_data, (list, tuple)) and len(val_data) == 2:
                X_val, y_val = val_data
            elif isinstance(val_data, list) and isinstance(val_data[0], dict):
                X_val, y_val = self._process_dict_data(val_data)
            else:
                raise ValueError("val_data must be (X, y) tuple or list of dicts")
            validation_data = (X_val, y_val)

        if self.model is None:
            self.model = self.build_model()

        callbacks = self._get_callbacks()

        logger.info(f"Starting training for {self.training_config.epochs} epochs")
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            callbacks=callbacks,
            verbose=1 if self.verbose else 2)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        final_model_path = self.output_dir / 'final_model.h5'
        self.model.save(str(final_model_path))
        logger.info(f"Saved final model to {final_model_path}")

        history_path = self.output_dir / 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)

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

        results['metrics'] = {
            'Training Loss': results['final_loss'],
            'Training Accuracy': results['final_accuracy']
        }
        if 'final_val_loss' in results:
            results['metrics']['Validation Loss'] = results['final_val_loss']
            results['metrics']['Validation Accuracy'] = results['final_val_accuracy']

        return results

    def _process_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        sequences, labels = [], []
        for item in data:
            seq_numeric = self._sequence_to_numeric(item['sequence'])
            sequences.append(seq_numeric)
            label = item.get('labels', [])
            if isinstance(label, list) and len(label) > 0 and isinstance(label[0], str):
                labels.append(self._labels_to_numeric(label))
            else:
                labels.append(label)

        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.model_config.max_seq_len, padding='post')
        y = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.model_config.max_seq_len, padding='post')
        y = tf.keras.utils.to_categorical(y, num_classes=self.model_config.num_labels)
        return X, y

    def _sequence_to_numeric(self, sequence: str) -> List[int]:
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
        return [mapping.get(base.upper(), 0) for base in sequence]

    def _labels_to_numeric(self, labels: List[str]) -> List[int]:
        label_mapping = {'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4, 'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10}
        return [label_mapping.get(label, 0) for label in labels]

    def predict(self, sequences: Union[np.ndarray, List[str]]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        if isinstance(sequences, list) and isinstance(sequences[0], str):
            X = [self._sequence_to_numeric(seq) for seq in sequences]
            X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.model_config.max_seq_len, padding='post')
        else:
            X = sequences
        return self.model.predict(X, verbose=0)

    def evaluate(self, test_data: Union[tuple, list]) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained yet")
        if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
            X_test, y_test = test_data
        elif isinstance(test_data, list) and isinstance(test_data[0], dict):
            X_test, y_test = self._process_dict_data(test_data)
        else:
            raise ValueError("test_data must be (X, y) tuple or list of dicts")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {'test_loss': results[0], 'test_accuracy': results[1]}

    def save(self, path: Union[str, Path]):
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        self.model = tf.keras.models.load_model(str(path))
        logger.info(f"Model loaded from {path}")


def run_training(config: TempestConfig, output_dir: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
    train_data = kwargs.get('train_data')
    val_data = kwargs.get('val_data')
    verbose = kwargs.get('verbose', False)
    mode = kwargs.get('mode', 'standard')

    if mode == 'hybrid':
        from tempest.training.hybrid_trainer import run_hybrid_training
        return run_hybrid_training(config, output_dir=output_dir, **kwargs)
    elif mode == 'ensemble':
        from tempest.training.ensemble import run_ensemble_training
        return run_ensemble_training(config, output_dir=output_dir, **kwargs)
    else:
        trainer = StandardTrainer(config, output_dir=output_dir, verbose=verbose)
        if train_data is None and config.training and hasattr(config.training, 'train_data'):
            from tempest.main import load_data
            train_data = load_data(config.training.train_data)
        if val_data is None and config.training and hasattr(config.training, 'val_data'):
            from tempest.main import load_data
            val_data = load_data(config.training.val_data)
        if train_data is None:
            raise ValueError("No training data provided")
        return trainer.train(train_data, val_data, **kwargs)
