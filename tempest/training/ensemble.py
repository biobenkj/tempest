"""
Ensemble trainer for Tempest.

Implements Bayesian Model Averaging (BMA) for robust predictions
through training multiple models with variation.

Part of: tempest/training/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
from copy import deepcopy
import random

from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.config import TempestConfig
from tempest.utils.io import ensure_dir
from tempest.training.trainer import StandardTrainer
from tempest.training.hybrid_trainer import HybridTrainer

logger = logging.getLogger(__name__)


# Utility functions (should be moved to tempest.utils in future refactor)
ArrayLike = Union[np.ndarray, tf.Tensor]


def pad_sequences(sequences: ArrayLike, labels: ArrayLike, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences and labels to max_length, handling both NumPy and TensorFlow inputs.
    
    Note: This is a duplicate of the function in hybrid_trainer. Should be moved to
    a shared utils module (e.g., tempest.utils.preprocessing).
    """
    # Convert to NumPy if TensorFlow tensors
    if isinstance(sequences, tf.Tensor):
        sequences = sequences.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    # Ensure NumPy-compatible dtypes
    seq_dtype = getattr(sequences.dtype, "as_numpy_dtype", sequences.dtype)
    lab_dtype = getattr(labels.dtype, "as_numpy_dtype", labels.dtype)

    n, curr_len = sequences.shape
    if curr_len == max_length:
        return sequences, labels

    pad_len = min(curr_len, max_length)
    padded_seq = np.zeros((n, max_length), dtype=seq_dtype)
    padded_lab = np.zeros((n, max_length), dtype=lab_dtype)

    padded_seq[:, :pad_len] = sequences[:, :pad_len]
    padded_lab[:, :pad_len] = labels[:, :pad_len]
    return padded_seq, padded_lab


def convert_labels_to_categorical(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.
    
    Note: This is a duplicate of the function in hybrid_trainer. Should be moved to
    a shared utils module (e.g., tempest.utils.preprocessing).
    
    Args:
        labels: Integer label array [batch, seq_len]
        num_classes: Number of label classes
        
    Returns:
        One-hot encoded labels [batch, seq_len, num_classes]
    """
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


class EnsembleTrainer:
    """
    Enhanced ensemble trainer that supports both standard and hybrid models.
    
    The ensemble can contain:
    - All standard models
    - All hybrid models  
    - Mixed standard and hybrid models
    
    Models are built using the refactored StandardTrainer and HybridTrainer,
    which in turn use the centralized build_model_from_config() factory.
    """
    
    def __init__(self,
                 config: TempestConfig,
                 num_models: int = 5,
                 variation_type: str = 'both',
                 checkpoint_dir: str = "ensemble_checkpoints",
                 model_types: Optional[List[str]] = None,
                 hybrid_ratio: float = 0.0):
        """
        Initialize ensemble trainer with support for mixed model types.
        
        Args:
            config: Base TempestConfig for models
            num_models: Number of models in ensemble
            variation_type: Type of variation ('architecture', 'initialization', 'both')
            checkpoint_dir: Directory for saving ensemble checkpoints
            model_types: List of model types for each model ('standard' or 'hybrid')
                        If None, uses hybrid_ratio to determine
            hybrid_ratio: Fraction of models that should be hybrid (0.0 to 1.0)
                         Only used if model_types is None
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
        if hasattr(config, 'ensemble') and config.ensemble:
            ensemble_config = config.ensemble
        
            # BMA configuration - support both old and new formats
            self.bma_prior = getattr(ensemble_config, 'bma_prior', 'performance')
            self.bma_temperature = getattr(ensemble_config, 'bma_temperature', 1.0)
        
            # Get BMA config details - handle both dict and dataclass
            if hasattr(ensemble_config, 'bma_config') and ensemble_config.bma_config is not None:
                bma_cfg = ensemble_config.bma_config
                
                # Check if it's a dataclass (new format) or dict (old format)
                if hasattr(bma_cfg, '__dataclass_fields__'):
                    # New format: BMAConfig dataclass
                    # Note: The new BMAConfig doesn't have 'method', it's for ModelCombiner inference
                    # For training, we just need min_weight and type_bonus
                    self.bma_method = 'validation_accuracy'  # Default for training
                    self.min_weight = getattr(bma_cfg, 'min_posterior_weight', 0.01)
                    
                    # type_bonus is in weighted_average_config, not bma_config
                    if hasattr(ensemble_config, 'weighted_average_config') and ensemble_config.weighted_average_config:
                        wac = ensemble_config.weighted_average_config
                        if isinstance(wac, dict):
                            self.type_bonus = wac.get('type_bonus', {})
                        else:
                            self.type_bonus = getattr(wac, 'type_bonus', {})
                    else:
                        self.type_bonus = {}
                        
                elif isinstance(bma_cfg, dict):
                    # Old format: dictionary
                    self.bma_method = bma_cfg.get('method', 'validation_accuracy')
                    self.min_weight = bma_cfg.get('min_weight', 0.01)
                    self.type_bonus = bma_cfg.get('type_bonus', {})
                else:
                    # Unknown format, use defaults
                    logger.warning("Unknown bma_config format, using defaults")
                    self.bma_method = 'validation_accuracy'
                    self.min_weight = 0.01
                    self.type_bonus = {}
            else:
                # No bma_config, use defaults
                self.bma_method = 'validation_accuracy'
                self.min_weight = 0.01
                self.type_bonus = {}
        
        # Architecture variations from config
        if hasattr(ensemble_config, 'architecture_variations'):
            self.architecture_variations = ensemble_config.architecture_variations
        else:
            self.architecture_variations = None
        
        # Tracking
        self.training_histories = []
        self.label_to_idx = None
        self.idx_to_label = None

        # Determine model types for ensemble
        if model_types is not None:
            # Use explicitly specified model types
            if len(model_types) != num_models:
                raise ValueError(f"model_types length ({len(model_types)}) must match num_models ({num_models})")
            self.model_types = model_types
        else:
            # Use hybrid_ratio to determine model types
            num_hybrid = int(num_models * hybrid_ratio)
            self.model_types = ['hybrid'] * num_hybrid + ['standard'] * (num_models - num_hybrid)
            # Shuffle for randomness
            import random
            random.shuffle(self.model_types)
        
        # Store trainer instances
        self.trainers = []
        
        logger.info(f"Ensemble configuration: {self.model_types.count('standard')} standard, "
                   f"{self.model_types.count('hybrid')} hybrid models")
    
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
        # Log architecture variation structure for transparency
        if hasattr(self, "architecture_variations") and self.architecture_variations:
            logger.info(
                f"Architecture variation keys: "
                f"{list(self.architecture_variations.__dict__.keys()) if hasattr(self.architecture_variations, '__dict__') else list(self.architecture_variations.keys())}"
            )
        else:
            logger.info("Architecture variation mode: randomized defaults (no config-based variations)")
        return configs
    
    def _vary_architecture(self, config: TempestConfig, model_idx: int) -> TempestConfig:
        """
        Apply architecture variations from config or predefined patterns.
        """
        # Check if we have variations from config
        if hasattr(self, 'architecture_variations') and self.architecture_variations:
            variations_config = self.architecture_variations
        
            # Create variation based on model index
            if model_idx == 0:
                # Base model - no changes
                pass
            else:
                # Apply variations from config
                if hasattr(variations_config, 'vary_lstm_units'):
                    units_options = variations_config.vary_lstm_units
                    config.model.lstm_units = units_options[model_idx % len(units_options)]
                
                if hasattr(variations_config, 'vary_lstm_layers'):
                    layers_options = variations_config.vary_lstm_layers
                    config.model.lstm_layers = layers_options[model_idx % len(layers_options)]
                
                if hasattr(variations_config, 'vary_dropout'):
                    dropout_options = variations_config.vary_dropout
                    config.model.dropout = dropout_options[model_idx % len(dropout_options)]
                
                if hasattr(variations_config, 'vary_embedding_dim'):
                    embed_options = variations_config.vary_embedding_dim
                    config.model.embedding_dim = embed_options[model_idx % len(embed_options)]
                
                if hasattr(variations_config, 'vary_cnn_filters') and config.model.use_cnn:
                    filter_options = variations_config.vary_cnn_filters
                    config.model.cnn_filters = filter_options[model_idx % len(filter_options)]
                
                logger.info(f"Model {model_idx}: Applied config-based variations")
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
              train_data: Union[np.ndarray, tuple, list],
              val_data: Optional[Union[np.ndarray, tuple, list]] = None,
              unlabeled_path: Optional[Union[str, Path]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train ensemble of mixed standard and hybrid models.
        
        Each model is trained using the refactored StandardTrainer or HybridTrainer,
        which internally use the centralized build_model_from_config() factory.
        
        Args:
            train_data: Training data (X, y) or list of dicts
            val_data: Optional validation data (X, y) or list of dicts
            unlabeled_path: Optional path to unlabeled data (used for hybrid models)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing ensemble training results
        """
        logger.info("="*80)
        logger.info(f"STARTING MIXED ENSEMBLE TRAINING ({self.num_models} models)")
        logger.info(f"Model types: {self.model_types}")
        logger.info("="*80)
        
        # Generate model variations
        self.model_configs = self._generate_model_variations()
        
        # Track results
        training_results = []
        
        # Train each model based on its type
        for i, (config, model_type) in enumerate(zip(self.model_configs, self.model_types)):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING MODEL {i+1}/{self.num_models} (Type: {model_type.upper()})")
            logger.info(f"{'='*60}")
            
            # Set random seed if specified
            if hasattr(config, 'seed'):
                tf.random.set_seed(config.seed)
                np.random.seed(config.seed)
                random.seed(config.seed)
            
            # Create appropriate trainer (both use centralized model builder)
            model_checkpoint_dir = self.checkpoint_dir / f"model_{i}_{model_type}"
            
            if model_type == 'standard':
                # StandardTrainer uses build_model_from_config internally
                trainer = StandardTrainer(
                    config=config,
                    output_dir=model_checkpoint_dir,
                    verbose=kwargs.get('verbose', False)
                )
                
                # Train standard model
                result = trainer.train(
                    train_data=train_data,
                    val_data=val_data,
                    **kwargs
                )
                
            elif model_type == 'hybrid':
                # HybridTrainer uses build_model_from_config internally
                trainer = HybridTrainer(
                    config=config,
                    output_dir=model_checkpoint_dir,
                    verbose=kwargs.get('verbose', False)
                )

                # Safety check for missing hybrid configuration
                if not getattr(config, "hybrid", None):
                    logger.warning(
                        f"Hybrid model {i+1} specified but no 'hybrid' block found in config — "
                        f"falling back to StandardTrainer."
                    )
                    model_type = "standard"
                    trainer = StandardTrainer(
                        config=config,
                        output_dir=model_checkpoint_dir,
                        verbose=kwargs.get('verbose', False)
                    )
                    result = trainer.train(
                        train_data=train_data,
                        val_data=val_data,
                        **kwargs
                    )
                    self.trainers.append(trainer)
                    self.models.append(trainer.model if hasattr(trainer, "model") else trainer.base_model)
                    continue  # Skip to next model
                
                # Train hybrid model (with optional unlabeled data)
                result = trainer.train(
                    train_data=train_data,
                    val_data=val_data,
                    unlabeled_path=unlabeled_path,  # Only hybrid models use this
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Store trainer and results
            self.trainers.append(trainer)
            self.models.append(trainer.model if hasattr(trainer, 'model') else trainer.base_model)
            training_results.append(result)
            
            # Extract performance metrics
            if 'final_val_accuracy' in result:
                val_acc = result['final_val_accuracy']
                val_loss = result.get('final_val_loss', float('inf'))
            elif 'metrics' in result:
                val_acc = result['metrics'].get('Validation Accuracy', 0.0)
                val_loss = result['metrics'].get('Validation Loss', float('inf'))
            else:
                val_acc = 0.0
                val_loss = float('inf')
            
            self.model_performances.append({
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'model_type': model_type,
                'model_index': i
            })
            
            # Store label mappings from first model
            if i == 0:
                if hasattr(trainer, 'label_to_idx'):
                    self.label_to_idx = trainer.label_to_idx
                    self.idx_to_label = trainer.idx_to_label
            
            logger.info(f"Model {i+1} ({model_type}) - Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}")
        
        # Compute BMA weights
        self._compute_bma_weights()
        
        # Save ensemble metadata
        self._save_ensemble_metadata()
        
        # Save individual models
        for i, model in enumerate(self.models):
            model_path = self.checkpoint_dir / f"ensemble_model_{i}.h5"
            model.save(str(model_path))
        
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total models trained: {len(self.models)}")
        logger.info(f"Average validation accuracy: {np.mean([p['val_accuracy'] for p in self.model_performances]):.4f}")
        logger.info(f"BMA weights computed using: {self.bma_method}")
        
        results = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'num_models': len(self.models),
            'model_types': self.model_types,
            'model_performances': self.model_performances,
            'bma_weights': self.model_weights.tolist() if isinstance(self.model_weights, np.ndarray) else self.model_weights,
            'training_results': training_results
        }
        
        return results
    
    def _compute_bma_weights(self):
        """
        Compute Bayesian Model Averaging weights based on validation performance.
        
        Supports multiple methods:
        - validation_accuracy: Weight by validation accuracy
        - validation_loss: Weight by inverse validation loss
        - uniform: Equal weights
        """
        if not self.model_performances:
            logger.warning("No model performances available, using uniform weights")
            self.model_weights = np.ones(self.num_models) / self.num_models
            return
        
        # Extract metrics
        val_accs = np.array([p['val_accuracy'] for p in self.model_performances])
        val_losses = np.array([p.get('val_loss', float('inf')) for p in self.model_performances])
        
        # Compute weights based on method
        if self.bma_method == 'validation_accuracy':
            # Weight by validation accuracy
            weights = val_accs ** self.bma_temperature
        elif self.bma_method == 'validation_loss':
            # Weight by inverse validation loss (lower loss = higher weight)
            # Handle infinite losses
            finite_losses = val_losses[np.isfinite(val_losses)]
            if len(finite_losses) > 0:
                max_loss = np.max(finite_losses)
                weights = (max_loss - val_losses) ** self.bma_temperature
                weights[~np.isfinite(val_losses)] = 0
            else:
                weights = np.ones(len(val_losses))
        elif self.bma_method == 'uniform':
            weights = np.ones(len(self.model_performances))
        else:
            logger.warning(f"Unknown BMA method '{self.bma_method}', using validation accuracy")
            weights = val_accs ** self.bma_temperature
        
        # Apply type bonuses if configured
        if self.type_bonus:
            for i, perf in enumerate(self.model_performances):
                model_type = perf['model_type']
                if model_type in self.type_bonus:
                    bonus = self.type_bonus[model_type]
                    weights[i] *= (1.0 + bonus)
                    logger.info(f"Applied {bonus:.2%} bonus to {model_type} model {i}")
        
        # Normalize weights
        weights = np.maximum(weights, self.min_weight)  # Ensure minimum weight
        weights = weights / np.sum(weights)
        
        self.model_weights = weights
        
        logger.info(f"BMA weights computed: {weights}")
        logger.info(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    def _save_ensemble_metadata(self):
        """Save ensemble configuration and weights."""
        metadata = {
            'num_models': self.num_models,
            'model_types': self.model_types,
            'model_weights': self.model_weights.tolist() if isinstance(self.model_weights, np.ndarray) else self.model_weights,
            'model_performances': self.model_performances,
            'bma_method': self.bma_method,
            'bma_temperature': self.bma_temperature,
            'variation_type': self.variation_type,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }
        
        metadata_path = self.checkpoint_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved ensemble metadata to {metadata_path}")
    
    def predict(self,
                reads: List[SimulatedRead],
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Make predictions using BMA ensemble.
        
        Args:
            reads: List of SimulatedRead objects
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            BMA-weighted predictions, optionally with uncertainty dict
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
        
        # Accuracy
        correct = (pred_classes == true_classes) & mask
        accuracy = np.sum(correct) / np.sum(mask)
        
        # Per-label accuracy
        label_accuracies = {}
        for label_name, label_idx in self.label_to_idx.items():
            label_mask = (true_classes == label_idx) & mask
            if np.sum(label_mask) > 0:
                label_correct = correct & label_mask
                label_acc = np.sum(label_correct) / np.sum(label_mask)
                label_accuracies[label_name] = label_acc
        
        results = {
            'accuracy': accuracy,
            'label_accuracies': label_accuracies
        }
        
        return results
    
    def compute_diversity(self, val_data: Union[tuple, List[SimulatedRead]]) -> float:
        """
        Compute ensemble diversity metric.
        
        Args:
            val_data: Validation data (X, y) or list of SimulatedRead objects
            
        Returns:
            Diversity score (pairwise disagreement rate)
        """
        if isinstance(val_data, tuple):
            X_val, _ = val_data
        else:
            X_val, _, _ = reads_to_arrays(val_data, label_to_idx=self.label_to_idx)
            max_len = self.base_config.model.max_seq_len
            X_val, _ = pad_sequences(X_val, np.zeros_like(X_val), max_len)
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(X_val, batch_size=self.batch_size, verbose=0)
            pred_classes = np.argmax(pred, axis=-1)
            predictions.append(pred_classes)
    
        # Compute pairwise disagreement
        disagreement = 0.0
        n_pairs = 0
    
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                disagreement += np.mean(predictions[i] != predictions[j])
                n_pairs += 1
    
        diversity = disagreement / n_pairs if n_pairs > 0 else 0.0
        logger.info(f"Ensemble diversity (disagreement rate): {diversity:.4f}")
    
        return diversity


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
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        num_models = metadata.get("num_models", 1)

        # Gracefully handle missing or empty model_weights
        if "model_weights" in metadata and metadata["model_weights"]:
            self.model_weights = np.array(metadata["model_weights"], dtype=np.float32)
        else:
            # Default: uniform weights
            self.model_weights = np.ones(num_models, dtype=np.float32) / num_models
            logger.warning(
                "No model_weights found in ensemble metadata — using uniform weights."
            )

        self.label_to_idx = metadata.get("label_to_idx", {})
        self.idx_to_label = {
            str(k): v for k, v in metadata.get("idx_to_label", {}).items()
        }

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
    

def run_ensemble_training(
    config: TempestConfig,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced orchestration function for mixed ensemble training.
    
    This function creates an EnsembleTrainer which internally uses
    StandardTrainer and HybridTrainer - both of which have been refactored
    to use the centralized build_model_from_config() factory.
    """
    # Extract parameters
    train_data = kwargs.get('train_data')
    val_data = kwargs.get('val_data')
    unlabeled_path = kwargs.get('unlabeled_path')
    num_models = kwargs.get('num_models', 5)
    verbose = kwargs.get('verbose', False)
    
    # Get ensemble configuration from config
    if hasattr(config, 'ensemble') and config.ensemble:
        ensemble_config = config.ensemble
        
        # Override with config values if not specified in kwargs
        if 'num_models' not in kwargs:
            num_models = getattr(ensemble_config, 'num_models', 5)
        
        # Model types configuration
        model_types = kwargs.get('model_types', getattr(ensemble_config, 'model_types', None))
        hybrid_ratio = kwargs.get('hybrid_ratio', getattr(ensemble_config, 'hybrid_ratio', 0.0))
        variation_type = kwargs.get('variation_type', getattr(ensemble_config, 'variation_type', 'both'))
    else:
        model_types = kwargs.get('model_types', None)
        hybrid_ratio = kwargs.get('hybrid_ratio', 0.0)
        variation_type = kwargs.get('variation_type', 'both')
    
    # Auto-determine hybrid ratio if unlabeled data provided
    if model_types is None and unlabeled_path and hybrid_ratio == 0.0:
        logger.info("Unlabeled data provided, setting hybrid_ratio to 0.4")
        hybrid_ratio = 0.4
    
    # Create output directory
    if output_dir is None:
        output_dir = Path("ensemble_output")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("ENSEMBLE TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Number of models: {num_models}")
    logger.info(f"Hybrid ratio: {hybrid_ratio}")
    logger.info(f"Variation type: {variation_type}")
    logger.info(f"Output directory: {output_dir}")
    if unlabeled_path:
        logger.info(f"Unlabeled data: {unlabeled_path}")
    
    # Create ensemble trainer with corrected parameters
    trainer = EnsembleTrainer(
        config=config,
        num_models=num_models,
        variation_type=variation_type,
        checkpoint_dir=str(output_dir / "checkpoints"),  # Convert to string
        model_types=model_types,
        hybrid_ratio=hybrid_ratio
    )
    
    # Run training
    results = trainer.train(
        train_data=train_data,
        val_data=val_data,
        unlabeled_path=unlabeled_path,
        **kwargs
    )
    
    # Compute diversity if validation data available
    if val_data is not None:
        diversity = trainer.compute_diversity(val_data)
        results['diversity'] = diversity
    
    # Prepare summary for CLI
    if 'metrics' not in results:
        results['metrics'] = {}
    
    results['metrics'].update({
        'Ensemble Size': num_models,
        'Standard Models': trainer.model_types.count('standard'),
        'Hybrid Models': trainer.model_types.count('hybrid'),
        'Average Val Accuracy': f"{np.mean([p.get('val_accuracy', 0) for p in trainer.model_performances]):.4f}",
        'Ensemble Diversity': f"{results.get('diversity', 0):.4f}",
        'Output Directory': str(output_dir)
    })
    
    return results
