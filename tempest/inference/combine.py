#!/usr/bin/env python3
"""
Model combination module for Tempest inference.

Implements both Bayesian Model Averaging (BMA) and weighted voting
for combining predictions from multiple models.

Part of: tempest/inference/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.special import logsumexp
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
import yaml

from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.utils.config import TempestConfig, load_config
from tempest.utils.io import ensure_dir
from tempest.training.hybrid_trainer import pad_sequences

logger = logging.getLogger(__name__)


@dataclass
class CombineConfig:
    """Configuration for model combination."""
    method: str = 'bma'  # 'bma' or 'weighted'
    weights: Optional[Dict[str, float]] = None  # For weighted voting
    prior_type: str = 'uniform'  # For BMA: 'uniform', 'performance', 'complexity'
    temperature: float = 1.0  # Temperature scaling for BMA
    use_bic: bool = True  # Use BIC approximation for BMA
    complexity_penalty: bool = True  # Apply complexity penalties in BMA
    save_evidence: bool = True  # Save model evidence calculations
    output_dir: str = './combine_results'


class ModelCombiner:
    """
    Combines predictions from multiple models using BMA or weighted voting.
    
    Supports:
    - True Bayesian Model Averaging with evidence-based weights
    - Simple weighted voting with fixed weights
    - Uncertainty quantification
    - Model evidence calculation
    """
    
    def __init__(self, config: CombineConfig = None):
        """
        Initialize model combiner.
        
        Args:
            config: Configuration for model combination
        """
        self.config = config or CombineConfig()
        self.models = {}
        self.model_paths = {}
        self.model_configs = {}
        self.model_weights = {}
        self.model_evidences = {}
        self.posterior_weights = {}
        self.label_to_idx = None
        self.idx_to_label = None
        
    def load_models(self, model_paths: Union[List[str], Dict[str, str]]):
        """
        Load models from file paths.
        
        Args:
            model_paths: List of paths or dict of name->path mappings
        """
        if isinstance(model_paths, list):
            # Auto-generate names
            model_dict = {f"model_{i}": path for i, path in enumerate(model_paths)}
        else:
            model_dict = model_paths
            
        for name, path in model_dict.items():
            path = Path(path)
            
            if path.is_dir():
                # Load ensemble directory
                self._load_ensemble_directory(name, path)
            elif path.suffix == '.h5':
                # Load single model
                self._load_single_model(name, path)
            else:
                logger.warning(f"Unknown model format for {path}, skipping")
                
        logger.info(f"Loaded {len(self.models)} models")
        
    def _load_single_model(self, name: str, path: Path):
        """Load a single model file."""
        try:
            model = keras.models.load_model(str(path))
            self.models[name] = model
            self.model_paths[name] = str(path)
            
            # Try to load associated config
            config_path = path.parent / f"{path.stem}_config.yaml"
            if config_path.exists():
                self.model_configs[name] = load_config(str(config_path))
            
            # Try to load label mappings
            labels_path = path.parent / f"{path.stem}_labels.pkl"
            if labels_path.exists():
                with open(labels_path, 'rb') as f:
                    labels = pickle.load(f)
                    if self.label_to_idx is None:
                        self.label_to_idx = labels.get('label_to_idx')
                        self.idx_to_label = labels.get('idx_to_label')
                        
            logger.info(f"Loaded model '{name}' from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            
    def _load_ensemble_directory(self, name: str, path: Path):
        """Load an ensemble from a directory."""
        # Load ensemble metadata if available
        metadata_path = path / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Load sub-models
            for i in range(metadata.get('num_models', 0)):
                model_file = path / f"ensemble_model_{i}.h5"
                if model_file.exists():
                    sub_name = f"{name}_sub{i}"
                    self._load_single_model(sub_name, model_file)
                    
                    # Store ensemble weights if using weighted voting
                    if self.config.method == 'weighted' and 'model_weights' in metadata:
                        self.model_weights[sub_name] = metadata['model_weights'][i]
                        
    def compute_weights(self, validation_data: Union[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Compute model combination weights.
        
        Args:
            validation_data: Validation data for computing BMA weights
                            Can be path to pickle file or tuple of (X, y)
        """
        if self.config.method == 'bma':
            self._compute_bma_weights(validation_data)
        elif self.config.method == 'weighted':
            self._setup_weighted_voting()
        else:
            raise ValueError(f"Unknown combination method: {self.config.method}")
            
    def _compute_bma_weights(self, validation_data):
        """
        Compute true Bayesian Model Averaging weights.
        
        P(M_k|D) ∝ P(D|M_k) * P(M_k)
        """
        logger.info("Computing BMA weights...")
        
        # Load validation data
        if isinstance(validation_data, str):
            with open(validation_data, 'rb') as f:
                val_dict = pickle.load(f)
                X_val = val_dict['X']
                y_val = val_dict['y']
        else:
            X_val, y_val = validation_data
            
        # Compute evidence for each model
        for name, model in self.models.items():
            if self.config.use_bic:
                log_evidence = self._compute_bic_evidence(model, X_val, y_val)
            else:
                log_evidence = self._compute_marginal_likelihood(model, X_val, y_val)
                
            self.model_evidences[name] = log_evidence
            logger.info(f"Model '{name}' log evidence: {log_evidence:.4f}")
            
        # Apply priors
        log_priors = self._get_log_priors()
        
        # Compute posterior weights with temperature scaling
        log_posteriors = {}
        for name in self.models:
            log_posteriors[name] = (
                self.model_evidences[name] / self.config.temperature + 
                log_priors[name]
            )
            
        # Normalize to get weights
        log_norm = logsumexp(list(log_posteriors.values()))
        for name in self.models:
            self.posterior_weights[name] = np.exp(log_posteriors[name] - log_norm)
            logger.info(f"Model '{name}' BMA weight: {self.posterior_weights[name]:.4f}")
            
        self.model_weights = self.posterior_weights
        
    def _compute_bic_evidence(self, model, X_val, y_val) -> float:
        """
        Compute BIC approximation to log marginal likelihood.
        
        BIC ≈ log P(D|θ_MLE,M) - (k/2)log(n)
        """
        # Compute log likelihood
        predictions = model.predict(X_val, verbose=0)
        
        # Handle both categorical and sparse labels
        if len(y_val.shape) == 3:  # One-hot encoded
            log_likelihood = np.sum(y_val * np.log(predictions + 1e-10))
        else:  # Sparse labels
            n_classes = predictions.shape[-1]
            y_one_hot = tf.keras.utils.to_categorical(y_val, n_classes)
            log_likelihood = np.sum(y_one_hot * np.log(predictions + 1e-10))
            
        # Count parameters
        n_params = model.count_params()
        n_samples = X_val.shape[0] * X_val.shape[1]  # Total positions
        
        # BIC penalty
        bic_penalty = 0.5 * n_params * np.log(n_samples)
        
        # Apply optional complexity penalty
        if self.config.complexity_penalty:
            complexity = self._get_model_complexity(model)
            bic_penalty += complexity
            
        return log_likelihood - bic_penalty
        
    def _compute_marginal_likelihood(self, model, X_val, y_val) -> float:
        """
        Compute full marginal likelihood (more sophisticated than BIC).
        
        Uses Laplace approximation or variational inference.
        """
        # For now, fall back to BIC
        # Full implementation would require MCMC or variational inference
        return self._compute_bic_evidence(model, X_val, y_val)
        
    def _get_model_complexity(self, model) -> float:
        """
        Compute model-specific complexity penalty.
        
        Returns:
            Complexity penalty value
        """
        n_params = model.count_params()
        n_layers = len(model.layers)
        
        # Heuristic complexity based on architecture
        complexity = 0.0
        
        # Penalty for number of parameters (log scale)
        complexity += 0.1 * np.log(n_params + 1)
        
        # Penalty for depth
        complexity += 0.05 * n_layers
        
        # Check for specific layer types
        for layer in model.layers:
            if isinstance(layer, keras.layers.LSTM):
                complexity += 0.2  # RNNs are complex
            elif isinstance(layer, keras.layers.GRU):
                complexity += 0.15
            elif isinstance(layer, keras.layers.Attention):
                complexity += 0.3  # Attention is very complex
                
        return complexity
        
    def _get_log_priors(self) -> Dict[str, float]:
        """
        Get log prior probabilities for models.
        
        Returns:
            Dictionary of model_name -> log_prior
        """
        n_models = len(self.models)
        
        if self.config.prior_type == 'uniform':
            # Equal priors for all models
            log_prior = -np.log(n_models)
            return {name: log_prior for name in self.models}
            
        elif self.config.prior_type == 'complexity':
            # Prior favors simpler models (Occam's razor)
            complexities = {}
            for name, model in self.models.items():
                complexities[name] = self._get_model_complexity(model)
                
            # Convert to probabilities (lower complexity = higher prior)
            min_complex = min(complexities.values())
            log_priors = {}
            for name in self.models:
                # Exponential prior on simplicity
                log_priors[name] = -(complexities[name] - min_complex)
                
            # Normalize
            log_norm = logsumexp(list(log_priors.values()))
            return {name: lp - log_norm for name, lp in log_priors.items()}
            
        elif self.config.prior_type == 'performance':
            # Prior based on training performance (if available)
            # This would need training history
            logger.warning("Performance priors not available, using uniform")
            log_prior = -np.log(n_models)
            return {name: log_prior for name in self.models}
            
        else:
            # Default to uniform
            log_prior = -np.log(n_models)
            return {name: log_prior for name in self.models}
            
    def _setup_weighted_voting(self):
        """Setup weights for weighted voting."""
        if self.config.weights:
            # Use provided weights
            total_weight = sum(self.config.weights.values())
            self.model_weights = {
                name: self.config.weights.get(name, 1.0) / total_weight
                for name in self.models
            }
        else:
            # Use equal weights
            n_models = len(self.models)
            self.model_weights = {name: 1.0/n_models for name in self.models}
            
        # Log weights
        for name, weight in self.model_weights.items():
            logger.info(f"Model '{name}' weight: {weight:.4f}")
            
    def predict(self, 
                X: np.ndarray,
                return_uncertainty: bool = False,
                return_individual: bool = False) -> Union[np.ndarray, Dict]:
        """
        Make combined predictions.
        
        Args:
            X: Input sequences
            return_uncertainty: Return uncertainty estimates
            return_individual: Return individual model predictions
            
        Returns:
            Combined predictions, optionally with uncertainty and individual predictions
        """
        if not self.models:
            raise ValueError("No models loaded")
            
        if not self.model_weights:
            raise ValueError("Model weights not computed. Run compute_weights() first")
            
        # Get predictions from each model
        individual_predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            individual_predictions[name] = pred
            
        # Combine predictions using weights
        combined = np.zeros_like(next(iter(individual_predictions.values())))
        for name, pred in individual_predictions.items():
            weight = self.model_weights.get(name, 0.0)
            combined += weight * pred
            
        result = {'predictions': combined}
        
        if return_uncertainty:
            # Compute uncertainty metrics
            uncertainty = self._compute_uncertainty(individual_predictions, combined)
            result['uncertainty'] = uncertainty
            
        if return_individual:
            result['individual_predictions'] = individual_predictions
            result['model_weights'] = self.model_weights
            
        if len(result) == 1:
            return result['predictions']
        return result
        
    def _compute_uncertainty(self, 
                            individual_predictions: Dict[str, np.ndarray],
                            combined: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute uncertainty metrics for predictions.
        
        Args:
            individual_predictions: Dict of model predictions
            combined: Combined predictions
            
        Returns:
            Dictionary of uncertainty metrics
        """
        # Convert to array for easier computation
        all_preds = np.stack(list(individual_predictions.values()), axis=0)
        
        # Predictive entropy
        entropy = -np.sum(combined * np.log(combined + 1e-10), axis=-1)
        
        # Model disagreement (variance across models)
        disagreement = np.var(all_preds, axis=0)
        mean_disagreement = np.mean(disagreement, axis=-1)
        
        # Mutual information (epistemic uncertainty)
        # MI = H[E[p]] - E[H[p]]
        expected_entropy = np.mean([
            -np.sum(pred * np.log(pred + 1e-10), axis=-1) 
            for pred in individual_predictions.values()
        ], axis=0)
        mutual_info = entropy - expected_entropy
        
        return {
            'entropy': entropy,
            'model_disagreement': mean_disagreement,
            'mutual_information': mutual_info,
            'epistemic_uncertainty': mutual_info,
            'aleatoric_uncertainty': expected_entropy
        }
        
    def save_results(self, output_dir: Optional[str] = None):
        """
        Save combination results and metadata.
        
        Args:
            output_dir: Directory for saving results
        """
        output_dir = Path(output_dir or self.config.output_dir)
        ensure_dir(str(output_dir))
        
        # Save weights
        weights_file = output_dir / "model_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(self.model_weights, f, indent=2)
        logger.info(f"Saved model weights to {weights_file}")
        
        # Save evidences if BMA
        if self.config.method == 'bma' and self.model_evidences:
            evidence_file = output_dir / "model_evidences.json"
            with open(evidence_file, 'w') as f:
                json.dump(self.model_evidences, f, indent=2)
            logger.info(f"Saved model evidences to {evidence_file}")
            
        # Save configuration
        config_file = output_dir / "combine_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(self.config), f)
        logger.info(f"Saved configuration to {config_file}")
        
        # Save summary
        summary = {
            'method': self.config.method,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_weights': self.model_weights,
            'effective_models': sum(1 for w in self.model_weights.values() if w > 0.01)
        }
        
        if self.config.method == 'bma':
            summary['model_evidences'] = self.model_evidences
            summary['prior_type'] = self.config.prior_type
            summary['temperature'] = self.config.temperature
            
        summary_file = output_dir / "combination_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved combination summary to {summary_file}")
        
    def evaluate_combination(self, test_data: Union[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Evaluate the combined model on test data.
        
        Args:
            test_data: Test data (path or tuple)
            
        Returns:
            Evaluation metrics
        """
        # Load test data
        if isinstance(test_data, str):
            with open(test_data, 'rb') as f:
                test_dict = pickle.load(f)
                X_test = test_dict['X']
                y_test = test_dict['y']
        else:
            X_test, y_test = test_data
            
        # Get combined predictions
        result = self.predict(X_test, return_uncertainty=True, return_individual=True)
        predictions = result['predictions']
        uncertainty = result['uncertainty']
        individual = result['individual_predictions']
        
        # Calculate metrics
        pred_classes = np.argmax(predictions, axis=-1)
        
        # Handle both one-hot and sparse labels
        if len(y_test.shape) == 3:
            true_classes = np.argmax(y_test, axis=-1)
        else:
            true_classes = y_test
            
        # Accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Individual model accuracies
        individual_metrics = {}
        for name, pred in individual.items():
            ind_pred_classes = np.argmax(pred, axis=-1)
            ind_acc = np.mean(ind_pred_classes == true_classes)
            individual_metrics[f"{name}_accuracy"] = ind_acc
            
        # Uncertainty calibration
        # High uncertainty should correlate with errors
        errors = (pred_classes != true_classes).flatten()
        entropy_flat = uncertainty['entropy'].flatten()
        
        # Compute correlation between uncertainty and errors
        if len(errors) > 0:
            uncertainty_error_corr = np.corrcoef(entropy_flat, errors)[0, 1]
        else:
            uncertainty_error_corr = 0.0
            
        metrics = {
            'combined_accuracy': accuracy,
            'mean_entropy': np.mean(uncertainty['entropy']),
            'mean_model_disagreement': np.mean(uncertainty['model_disagreement']),
            'uncertainty_error_correlation': uncertainty_error_corr,
            **individual_metrics
        }
        
        return metrics
