"""
Ensemble trainer for Tempest.

Implements Bayesian Model Averaging (BMA) for robust predictions
through training multiple models with variation.

Part of: tempest/training/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from copy import deepcopy
import random

from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.utils.config import TempestConfig
from tempest.utils.io import ensure_dir
from tempest.training.trainer import ModelTrainer
from tempest.training.hybrid_trainer import (
    build_model_from_config,
    pad_sequences,
    convert_labels_to_categorical
)

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Ensemble trainer using Bayesian Model Averaging.
    
    Trains multiple models with variation in:
    - Architecture (different hyperparameters)
    - Initialization (different random seeds)
    
    Combines predictions using BMA with performance-based weights.
    """
    
    def __init__(self,
                 config: TempestConfig,
                 num_models: int = 5,
                 variation_type: str = 'both',
                 checkpoint_dir: str = "ensemble_checkpoints"):
        """
        Initialize ensemble trainer.
        
        Args:
            config: Base TempestConfig for models
            num_models: Number of models in ensemble
            variation_type: Type of variation ('architecture', 'initialization', 'both')
            checkpoint_dir: Directory for saving ensemble checkpoints
        """
        self.base_config = deepcopy(config)
        self.num_models = num_models
        self.variation_type = variation_type
        self.checkpoint_dir = Path(checkpoint_dir)
        ensure_dir(str(self.checkpoint_dir))
        
        # Ensemble components
        self.models = []
        self.model_configs = []
        self.model_weights = []  # BMA weights
        self.model_performances = []
        
        # Training parameters
        self.epochs = config.training.epochs if config.training else 50
        self.batch_size = config.model.batch_size
        
        # BMA parameters
        self.bma_prior = getattr(config.ensemble, 'bma_prior', 'uniform') if config.ensemble else 'uniform'
        self.bma_temperature = getattr(config.ensemble, 'bma_temperature', 1.0) if config.ensemble else 1.0
        
        # Tracking
        self.training_histories = []
        self.label_to_idx = None
        self.idx_to_label = None
    
    def _generate_model_variations(self) -> List[TempestConfig]:
        """
        Generate model configuration variations.
        
        Returns:
            List of varied TempestConfig objects
        """
        configs = []
        
        for i in range(self.num_models):
            config = deepcopy(self.base_config)
            
            if self.variation_type in ['architecture', 'both']:
                # Vary architecture hyperparameters
                config = self._vary_architecture(config, i)
            
            if self.variation_type in ['initialization', 'both']:
                # Set different random seed for initialization
                if not hasattr(config, 'seed'):
                    config.seed = 42 + i
                else:
                    config.seed = config.seed + i
            
            configs.append(config)
            
        logger.info(f"Generated {len(configs)} model configurations with {self.variation_type} variation")
        return configs
    
    def _vary_architecture(self, config: TempestConfig, model_idx: int) -> TempestConfig:
        """
        Apply architecture variations to config.
        
        Args:
            config: Base configuration
            model_idx: Index of model in ensemble
            
        Returns:
            Modified configuration
        """
        # Variation strategies based on model index
        variations = [
            # Model 0: Base configuration (no changes)
            {},
            # Model 1: Deeper LSTM
            {'lstm_layers': 3, 'lstm_units': 256},
            # Model 2: Wider CNN
            {'cnn_filters': [128, 256], 'cnn_kernels': [3, 7]},
            # Model 3: Higher dropout
            {'dropout': 0.5},
            # Model 4: Smaller model
            {'lstm_units': 64, 'embedding_dim': 64},
        ]
        
        # Apply variation if available
        if model_idx < len(variations):
            variation = variations[model_idx]
            for key, value in variation.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
                    logger.info(f"Model {model_idx}: Set {key} = {value}")
        else:
            # Random variations for additional models
            random.seed(42 + model_idx)
            
            # Random LSTM units
            config.model.lstm_units = random.choice([64, 128, 256])
            
            # Random dropout
            config.model.dropout = random.uniform(0.2, 0.5)
            
            # Random CNN filters
            if config.model.use_cnn:
                config.model.cnn_filters = [
                    random.choice([32, 64, 128]),
                    random.choice([64, 128, 256])
                ]
            
            logger.info(f"Model {model_idx}: Random variation applied")
        
        return config
    
    def train(self,
             train_reads: List[SimulatedRead],
             val_reads: Optional[List[SimulatedRead]] = None) -> List[keras.Model]:
        """
        Train ensemble of models.
        
        Args:
            train_reads: Training SimulatedRead objects
            val_reads: Optional validation SimulatedRead objects
            
        Returns:
            List of trained models
        """
        logger.info("="*80)
        logger.info(f"STARTING ENSEMBLE TRAINING ({self.num_models} models)")
        logger.info("="*80)
        
        # Generate model variations
        self.model_configs = self._generate_model_variations()
        
        # Train each model
        for i, config in enumerate(self.model_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING MODEL {i+1}/{self.num_models}")
            logger.info(f"{'='*60}")
            
            # Set random seed if specified
            if hasattr(config, 'seed'):
                tf.random.set_seed(config.seed)
                np.random.seed(config.seed)
                random.seed(config.seed)
            
            # Create trainer for this model
            model_checkpoint_dir = self.checkpoint_dir / f"model_{i}"
            trainer = ModelTrainer(config, checkpoint_dir=str(model_checkpoint_dir))
            
            # Train model
            model = trainer.train(train_reads, val_reads)
            
            # Store model and training info
            self.models.append(model)
            self.training_histories.append(trainer.training_history)
            self.model_performances.append({
                'val_loss': trainer.best_val_loss,
                'val_accuracy': trainer.best_val_accuracy
            })
            
            # Store label mappings from first model
            if i == 0:
                self.label_to_idx = trainer.label_to_idx
                self.idx_to_label = trainer.idx_to_label
            
            logger.info(f"Model {i+1} - Val Loss: {trainer.best_val_loss:.4f}, "
                       f"Val Acc: {trainer.best_val_accuracy:.4f}")
        
        # Compute BMA weights
        self._compute_bma_weights()
        
        # Save ensemble
        self.save_ensemble()
        
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("="*80)
        self._print_ensemble_summary()
        
        return self.models
    
    def _compute_bma_weights(self):
        """
        Compute Bayesian Model Averaging weights based on validation performance.
        """
        if self.bma_prior == 'uniform':
            # Equal weights for all models
            self.model_weights = [1.0 / self.num_models] * self.num_models
            logger.info("Using uniform BMA prior (equal weights)")
            
        elif self.bma_prior == 'performance':
            # Weights based on validation accuracy
            val_accuracies = [perf['val_accuracy'] for perf in self.model_performances]
            
            # Apply temperature scaling
            scaled_accs = np.array(val_accuracies) / self.bma_temperature
            
            # Softmax to get weights
            exp_accs = np.exp(scaled_accs - np.max(scaled_accs))  # Numerical stability
            self.model_weights = (exp_accs / np.sum(exp_accs)).tolist()
            
            logger.info(f"Using performance-based BMA prior (temp={self.bma_temperature})")
            
        else:
            raise ValueError(f"Unknown BMA prior: {self.bma_prior}")
        
        # Log weights
        for i, weight in enumerate(self.model_weights):
            logger.info(f"Model {i+1} BMA weight: {weight:.4f}")
    
    def predict(self, reads: List[SimulatedRead], return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make ensemble predictions using BMA.
        
        Args:
            reads: List of SimulatedRead objects
            return_uncertainty: Whether to return prediction uncertainty
            
        Returns:
            Predictions array, optionally with uncertainty estimates
        """
        if not self.models:
            raise ValueError("No models in ensemble. Train the ensemble first.")
        
        # Convert reads to arrays
        X, _, _ = reads_to_arrays(reads, label_to_idx=self.label_to_idx)
        
        # Pad sequences
        max_len = self.base_config.model.max_seq_len
        X, _ = pad_sequences(X, np.zeros_like(X), max_len)
        
        # Get predictions from each model
        all_predictions = []
        for model in self.models:
            pred = model.predict(X, batch_size=self.batch_size, verbose=0)
            all_predictions.append(pred)
        
        # Stack predictions: [num_models, num_samples, seq_len, num_labels]
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Apply BMA weights
        weighted_predictions = np.zeros_like(all_predictions[0])
        for i, (pred, weight) in enumerate(zip(all_predictions, self.model_weights)):
            weighted_predictions += weight * pred
        
        if return_uncertainty:
            # Calculate prediction uncertainty (entropy or variance)
            # Using predictive entropy as uncertainty measure
            entropy = -np.sum(weighted_predictions * np.log(weighted_predictions + 1e-10), axis=-1)
            
            # Also calculate disagreement between models
            variance = np.var(all_predictions, axis=0)
            mean_variance = np.mean(variance, axis=-1)  # Average over label dimension
            
            uncertainty = {
                'entropy': entropy,
                'model_disagreement': mean_variance
            }
            
            return weighted_predictions, uncertainty
        
        return weighted_predictions
    
    def evaluate(self, val_reads: List[SimulatedRead]) -> Dict[str, float]:
        """
        Evaluate ensemble on validation data.
        
        Args:
            val_reads: Validation SimulatedRead objects
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = self.predict(val_reads)
        
        # Convert validation reads to arrays
        X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx=self.label_to_idx)
        
        # Pad and convert
        max_len = self.base_config.model.max_seq_len
        X_val, y_val = pad_sequences(X_val, y_val, max_len)
        num_labels = self.base_config.model.num_labels
        y_val_cat = convert_labels_to_categorical(y_val, num_labels)
        
        # Calculate metrics
        pred_classes = np.argmax(predictions, axis=-1)
        true_classes = np.argmax(y_val_cat, axis=-1)
        
        # Mask for non-padding
        mask = np.sum(y_val_cat, axis=-1) > 0
        
        # Overall accuracy
        correct = np.sum((pred_classes == true_classes) * mask)
        total = np.sum(mask)
        accuracy = correct / total if total > 0 else 0.0
        
        # Cross-entropy loss
        epsilon = 1e-10
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.sum(y_val_cat * np.log(predictions_clipped) * mask[:, :, np.newaxis])
        loss = loss / total if total > 0 else 0.0
        
        metrics = {
            'ensemble_accuracy': accuracy,
            'ensemble_loss': loss,
            'num_models': self.num_models,
            'effective_models': sum(w > 0.01 for w in self.model_weights)  # Models with >1% weight
        }
        
        # Add individual model performances
        for i, perf in enumerate(self.model_performances):
            metrics[f'model_{i+1}_val_acc'] = perf['val_accuracy']
            metrics[f'model_{i+1}_weight'] = self.model_weights[i]
        
        return metrics
    
    def save_ensemble(self, filepath: Optional[str] = None):
        """
        Save ensemble models and metadata.
        
        Args:
            filepath: Optional custom directory for saving
        """
        if filepath is None:
            filepath = self.checkpoint_dir
        else:
            filepath = Path(filepath)
            ensure_dir(str(filepath))
        
        # Save each model
        for i, model in enumerate(self.models):
            model_path = filepath / f"ensemble_model_{i}.h5"
            model.save(str(model_path))
            logger.info(f"Saved model {i+1} to: {model_path}")
        
        # Save ensemble metadata
        metadata = {
            'num_models': self.num_models,
            'variation_type': self.variation_type,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'bma_prior': self.bma_prior,
            'bma_temperature': self.bma_temperature,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'model_configs': [
                # Save key hyperparameters for each model
                {
                    'lstm_units': config.model.lstm_units,
                    'lstm_layers': config.model.lstm_layers,
                    'dropout': config.model.dropout,
                    'embedding_dim': config.model.embedding_dim,
                    'cnn_filters': config.model.cnn_filters if config.model.use_cnn else None,
                    'seed': getattr(config, 'seed', None)
                }
                for config in self.model_configs
            ]
        }
        
        metadata_path = filepath / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved ensemble metadata to: {metadata_path}")
    
    def load_ensemble(self, filepath: str):
        """
        Load ensemble models and metadata.
        
        Args:
            filepath: Directory containing ensemble files
        """
        filepath = Path(filepath)
        
        # Load metadata
        metadata_path = filepath / "ensemble_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.num_models = metadata['num_models']
        self.variation_type = metadata['variation_type']
        self.model_weights = metadata['model_weights']
        self.model_performances = metadata['model_performances']
        self.bma_prior = metadata['bma_prior']
        self.bma_temperature = metadata['bma_temperature']
        self.label_to_idx = metadata['label_to_idx']
        self.idx_to_label = {str(k): v for k, v in metadata['idx_to_label'].items()}
        
        # Load models
        self.models = []
        for i in range(self.num_models):
            model_path = filepath / f"ensemble_model_{i}.h5"
            model = keras.models.load_model(str(model_path))
            self.models.append(model)
        
        logger.info(f"Loaded ensemble with {self.num_models} models from: {filepath}")
        self._print_ensemble_summary()
    
    def _print_ensemble_summary(self):
        """Print summary of the ensemble."""
        logger.info("\nEnsemble Summary:")
        logger.info(f"  Number of models: {self.num_models}")
        logger.info(f"  Variation type: {self.variation_type}")
        logger.info(f"  BMA prior: {self.bma_prior}")
        
        if self.model_weights:
            logger.info("\nModel Weights (BMA):")
            for i, weight in enumerate(self.model_weights):
                perf = self.model_performances[i] if i < len(self.model_performances) else {}
                val_acc = perf.get('val_accuracy', 0.0)
                logger.info(f"  Model {i+1}: weight={weight:.4f}, val_acc={val_acc:.4f}")
        
        # Calculate effective number of models (based on weight entropy)
        if self.model_weights:
            weights = np.array(self.model_weights)
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            effective_models = np.exp(entropy)
            logger.info(f"\nEffective number of models: {effective_models:.2f}")


class BMAPredictor:
    """
    Standalone predictor for ensemble models using BMA.
    
    Can be used for inference without the full trainer.
    """
    
    def __init__(self, ensemble_dir: str):
        """
        Initialize predictor from saved ensemble.
        
        Args:
            ensemble_dir: Directory containing ensemble files
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.models = []
        self.model_weights = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        self._load_ensemble()
    
    def _load_ensemble(self):
        """Load ensemble models and metadata."""
        # Load metadata
        metadata_path = self.ensemble_dir / "ensemble_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        num_models = metadata['num_models']
        self.model_weights = metadata['model_weights']
        self.label_to_idx = metadata['label_to_idx']
        self.idx_to_label = {str(k): v for k, v in metadata['idx_to_label'].items()}
        
        # Load models
        for i in range(num_models):
            model_path = self.ensemble_dir / f"ensemble_model_{i}.h5"
            model = keras.models.load_model(str(model_path))
            self.models.append(model)
        
        logger.info(f"Loaded {num_models} models for BMA prediction")
    
    def predict(self, sequences: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        """
        Predict labels for sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Tuple of (predicted labels, confidence scores)
        """
        # Encode sequences
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        max_len = max(len(seq) for seq in sequences)
        
        X = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq[:max_len]):
                X[i, j] = base_to_idx.get(base.upper(), 4)
        
        # Get predictions from each model
        all_predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            all_predictions.append(pred)
        
        # Apply BMA weights
        weighted_predictions = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, self.model_weights):
            weighted_predictions += weight * pred
        
        # Convert to labels
        pred_classes = np.argmax(weighted_predictions, axis=-1)
        
        predicted_labels = []
        confidence_scores = []
        
        for i in range(len(sequences)):
            seq_len = len(sequences[i])
            labels = [self.idx_to_label.get(str(pred_classes[i, j]), 'UNKNOWN') 
                     for j in range(seq_len)]
            scores = [np.max(weighted_predictions[i, j]) for j in range(seq_len)]
            
            predicted_labels.append(labels)
            confidence_scores.append(scores)
        
        return predicted_labels, np.array(confidence_scores)
