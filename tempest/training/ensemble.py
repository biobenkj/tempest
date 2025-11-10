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
from tempest.config import TempestConfig
from tempest.utils.io import ensure_dir
from tempest.training.trainer import StandardTrainer
from tempest.training.hybrid_trainer import (
    HybridTrainer,
    build_model_from_config,
    pad_sequences,
    convert_labels_to_categorical
)

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Enhanced ensemble trainer that supports both standard and hybrid models.
    
    The ensemble can contain:
    - All standard models
    - All hybrid models  
    - Mixed standard and hybrid models
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
        
            # BMA configuration
            self.bma_prior = getattr(ensemble_config, 'bma_prior', 'performance')
            self.bma_temperature = getattr(ensemble_config, 'bma_temperature', 1.0)
        
            # Get BMA config details
            if hasattr(ensemble_config, 'bma_config'):
                self.bma_method = ensemble_config.bma_config.get('method', 'validation_accuracy')
                self.min_weight = ensemble_config.bma_config.get('min_weight', 0.01)
                self.type_bonus = ensemble_config.bma_config.get('type_bonus', {})
            else:
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
            
            # Create appropriate trainer
            model_checkpoint_dir = self.checkpoint_dir / f"model_{i}_{model_type}"
            
            if model_type == 'standard':
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
        
        # Save ensemble with metadata
        self.save_ensemble()
        
        # Prepare comprehensive results
        results = {
            'models': self.models,
            'model_types': self.model_types,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'training_results': training_results,
            'ensemble_dir': str(self.checkpoint_dir),
            'metrics': self._compute_ensemble_metrics()
        }
        
        logger.info("\n" + "="*80)
        logger.info("MIXED ENSEMBLE TRAINING COMPLETE")
        logger.info("="*80)
        self._print_ensemble_summary()
        
        return results
    
    def _compute_bma_weights(self):
        """
        Compute BMA weights with support for type bonuses and performance metrics.
        """
        if self.bma_prior == 'uniform':
            # Equal weights for all models
            self.model_weights = [1.0 / self.num_models] * self.num_models
            logger.info("Using uniform BMA prior (equal weights)")
        
        elif self.bma_prior == 'performance':
            # Get performance metrics
            if self.bma_method == 'validation_accuracy':
                scores = [perf.get('val_accuracy', 0.0) for perf in self.model_performances]
            elif self.bma_method == 'validation_loss':
                # Invert loss (lower is better)
                scores = [1.0 / (perf.get('val_loss', 1.0) + 1e-6) for perf in self.model_performances]
            else:
                # Default to accuracy
                scores = [perf.get('val_accuracy', 0.0) for perf in self.model_performances]
        
            # Apply type bonuses if configured
            if hasattr(self, 'type_bonus') and self.type_bonus:
                for i, model_type in enumerate(self.model_types):
                    bonus = self.type_bonus.get(model_type, 1.0)
                    scores[i] *= bonus
                    if bonus != 1.0:
                        logger.info(f"Applied {bonus}x bonus to {model_type} model {i+1}")
        
            # Apply temperature scaling
            scores_array = np.array(scores) / self.bma_temperature
        
            # Softmax to get weights
            exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
            weights = exp_scores / np.sum(exp_scores)
        
            # Apply minimum weight constraint
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)  # Re-normalize
        
            self.model_weights = weights.tolist()
        
            logger.info(f"Using performance-based BMA (method={self.bma_method}, temp={self.bma_temperature})")
    
        # Log weights with model types
        for i, (weight, model_type) in enumerate(zip(self.model_weights, self.model_types)):
            logger.info(f"Model {i+1} ({model_type}): BMA weight = {weight:.4f}")
    
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
    
    def _compute_ensemble_metrics(self) -> Dict[str, Any]:
        """Compute overall ensemble metrics."""
        metrics = {
            'num_models': self.num_models,
            'num_standard': self.model_types.count('standard'),
            'num_hybrid': self.model_types.count('hybrid'),
            'avg_val_accuracy': np.mean([p['val_accuracy'] for p in self.model_performances]),
            'avg_val_loss': np.mean([p['val_loss'] for p in self.model_performances]),
            'best_val_accuracy': max([p['val_accuracy'] for p in self.model_performances]),
            'worst_val_accuracy': min([p['val_accuracy'] for p in self.model_performances]),
        }
        
        # Add model type specific metrics
        standard_perfs = [p for p in self.model_performances if p['model_type'] == 'standard']
        hybrid_perfs = [p for p in self.model_performances if p['model_type'] == 'hybrid']
        
        if standard_perfs:
            metrics['avg_standard_accuracy'] = np.mean([p['val_accuracy'] for p in standard_perfs])
        if hybrid_perfs:
            metrics['avg_hybrid_accuracy'] = np.mean([p['val_accuracy'] for p in hybrid_perfs])
        
        return metrics
    
    def save_ensemble(self, filepath: Optional[str] = None):
        """
        Save ensemble with enhanced metadata including model types.
        """
        if filepath is None:
            filepath = self.checkpoint_dir
        else:
            filepath = Path(filepath)
            ensure_dir(str(filepath))
    
        # Save each model with type in filename
        for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
            model_path = filepath / f"ensemble_model_{i}_{model_type}.h5"
            model.save(str(model_path))
            logger.info(f"Saved {model_type} model {i+1} to: {model_path}")
    
        # Enhanced metadata
        metadata = {
            'num_models': self.num_models,
            'model_types': self.model_types,
            'variation_type': self.variation_type,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'bma_prior': self.bma_prior,
            'bma_temperature': self.bma_temperature,
            'bma_method': getattr(self, 'bma_method', 'validation_accuracy'),
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'ensemble_statistics': {
                'num_standard': self.model_types.count('standard'),
                'num_hybrid': self.model_types.count('hybrid'),
                'avg_weight_standard': np.mean([w for w, t in zip(self.model_weights, self.model_types) if t == 'standard']) if 'standard' in self.model_types else 0,
                'avg_weight_hybrid': np.mean([w for w, t in zip(self.model_weights, self.model_types) if t == 'hybrid']) if 'hybrid' in self.model_types else 0,
            },
            'model_configs': [
                {
                    'model_type': model_type,
                    'model_index': i,
                    'lstm_units': config.model.lstm_units,
                    'lstm_layers': config.model.lstm_layers,
                    'dropout': config.model.dropout,
                    'embedding_dim': config.model.embedding_dim,
                    'cnn_filters': getattr(config.model, 'cnn_filters', None),
                    'seed': getattr(config, 'seed', None),
                    'weight': self.model_weights[i] if i < len(self.model_weights) else 0
                }
                for i, (config, model_type) in enumerate(zip(self.model_configs, self.model_types))
            ]
        }
    
        # Save metadata
        metadata_path = filepath / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
        logger.info(f"Saved enhanced ensemble metadata to: {metadata_path}")
    
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
        for i, model_type in enumerate(metadata["model_types"]):
            model_path = filepath / f"ensemble_model_{i}_{model_type}.h5"
            model = keras.models.load_model(str(model_path))
            self.models.append(model)
        
        logger.info(f"Loaded ensemble with {self.num_models} models from: {filepath}")
        self._print_ensemble_summary()
    
    def _print_ensemble_summary(self):
        """Ensemble summary that shows model types."""
        logger.info("\nEnsemble Summary:")
        logger.info(f"  Total models: {self.num_models}")
        logger.info(f"  Standard models: {self.model_types.count('standard')}")
        logger.info(f"  Hybrid models: {self.model_types.count('hybrid')}")
        logger.info(f"  Variation type: {self.variation_type}")
        logger.info(f"  BMA prior: {self.bma_prior}")
        
        if self.model_weights:
            logger.info("\nModel Weights and Performance:")
            for i, (weight, perf, model_type) in enumerate(zip(
                self.model_weights, 
                self.model_performances,
                self.model_types
            )):
                val_acc = perf.get('val_accuracy', 0.0)
                logger.info(f"  Model {i+1} ({model_type}): weight={weight:.4f}, val_acc={val_acc:.4f}")
        
        # Calculate effective number of models (based on weight entropy)
        if self.model_weights:
            weights = np.array(self.model_weights)
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            effective_models = np.exp(entropy)
            logger.info(f"\nEffective number of models: {effective_models:.2f}")
    
    def compute_diversity(self, val_data: Optional[Union[np.ndarray, tuple, list]] = None) -> float:
        """
        Compute diversity metric for the ensemble.
        """
        if not self.models or len(self.models) < 2:
            return 0.0
    
        if val_data is None:
            logger.warning("No validation data provided for diversity computation")
            return 0.0
    
        # Get predictions from each model
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(val_data[0] if isinstance(val_data, tuple) else val_data)
                predictions.append(np.argmax(pred, axis=-1))
    
        if len(predictions) < 2:
            return 0.0
    
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
