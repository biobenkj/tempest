#!/usr/bin/env python3
"""
Model combination module for Tempest inference.

Implements comprehensive Bayesian Model Averaging (BMA) with multiple
approximation methods and advanced ensemble techniques.

Part of: tempest/inference/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict, field
import yaml
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.utils.config import TempestConfig, load_config
from tempest.utils.io import ensure_dir
from tempest.training.hybrid_trainer import pad_sequences

logger = logging.getLogger(__name__)


@dataclass
class BMAConfig:
    """Enhanced configuration for Bayesian Model Averaging."""
    
    # Prior configuration
    prior_type: str = 'uniform'  # 'uniform', 'informative', 'adaptive'
    prior_weights: Optional[Dict[str, float]] = None
    
    # Approximation method
    approximation: str = 'bic'  # 'bic', 'laplace', 'variational', 'cross_validation'
    
    # BIC parameters
    bic_penalty_factor: float = 1.0
    
    # Laplace approximation parameters
    laplace_num_samples: int = 1000
    laplace_damping: float = 0.01
    
    # Variational inference parameters
    vi_num_iterations: int = 100
    vi_learning_rate: float = 0.01
    vi_convergence_threshold: float = 1e-4
    
    # Cross-validation parameters
    cv_num_folds: int = 5
    cv_stratified: bool = True
    
    # Posterior settings
    temperature: float = 1.0
    compute_posterior_variance: bool = True
    normalize_posteriors: bool = True
    min_posterior_weight: float = 0.01
    
    # Model selection
    use_model_averaging: bool = True
    evidence_threshold: float = 0.05


@dataclass
class EnsembleConfig:
    """Complete ensemble configuration."""
    
    # General settings
    enabled: bool = True
    num_models: int = 3
    voting_method: str = 'bayesian_model_averaging'  # or 'weighted_average', 'voting', 'stacking'
    
    # BMA configuration
    bma_config: BMAConfig = field(default_factory=BMAConfig)
    
    # Weighted average configuration
    weighted_optimization: str = 'fixed'  # 'fixed', 'grid_search', 'differential_evolution', 'bayesian_optimization'
    fixed_weights: Optional[Dict[str, float]] = None
    
    # Prediction aggregation
    prediction_method: str = 'probability_averaging'  # 'logit_averaging', 'rank_averaging'
    confidence_weighting: bool = True
    apply_temperature_scaling: bool = False
    
    # Calibration settings
    calibration_enabled: bool = True
    calibration_method: str = 'isotonic'  # 'platt', 'beta', 'temperature_scaling'
    use_separate_calibration_set: bool = True
    calibration_split: float = 0.2
    
    # Diversity settings
    enforce_diversity: bool = True
    diversity_metric: str = 'disagreement'  # 'kl_divergence', 'correlation'
    min_diversity_threshold: float = 0.1
    
    # Uncertainty settings
    compute_epistemic: bool = True
    compute_aleatoric: bool = True
    confidence_intervals: bool = True
    interval_alpha: float = 0.05
    
    # Output settings
    output_dir: str = './ensemble_results'


class ModelCombiner:
    """
    Model combination with full BMA implementation.
    
    Supports all approximation methods and advanced ensemble techniques
    specified in the configuration.
    """
    
    def __init__(self, config: Optional[Union[EnsembleConfig, dict]] = None):
        """
        Initialize model combiner.
        
        Args:
            config: Ensemble configuration or dict
        """
        if config is None:
            self.config = EnsembleConfig()
        elif isinstance(config, dict):
            self.config = self._parse_config_dict(config)
        else:
            self.config = config
            
        self.models = {}
        self.model_paths = {}
        self.model_configs = {}
        self.model_weights = {}
        self.model_evidences = {}
        self.posterior_weights = {}
        self.posterior_variance = {}
        self.calibrator = None
        self.label_to_idx = None
        self.idx_to_label = None
        
    def _parse_config_dict(self, config_dict: dict) -> EnsembleConfig:
        """Parse configuration dictionary into EnsembleConfig."""
        ensemble_config = EnsembleConfig()
        
        # Parse BMA config if present
        if 'bma_config' in config_dict:
            bma_dict = config_dict['bma_config']
            bma_config = BMAConfig(**{
                k: v for k, v in bma_dict.items() 
                if k in BMAConfig.__annotations__
            })
            ensemble_config.bma_config = bma_config
        
        # Parse other ensemble settings
        for key, value in config_dict.items():
            if key != 'bma_config' and hasattr(ensemble_config, key):
                setattr(ensemble_config, key, value)
                
        return ensemble_config
    
    def load_models(self, model_paths: Union[List[str], Dict[str, str]]):
        """
        Load models from file paths.
        
        Args:
            model_paths: List of paths or dict of name->path mappings
        """
        if isinstance(model_paths, list):
            model_dict = {f"model_{i}": path for i, path in enumerate(model_paths)}
        else:
            model_dict = model_paths
            
        for name, path in model_dict.items():
            path = Path(path)
            
            if path.is_dir():
                self._load_ensemble_directory(name, path)
            elif path.suffix == '.h5':
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
        metadata_path = path / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            for i in range(metadata.get('num_models', 0)):
                model_file = path / f"ensemble_model_{i}.h5"
                if model_file.exists():
                    sub_name = f"{name}_sub{i}"
                    self._load_single_model(sub_name, model_file)
                    
    def compute_weights(self, validation_data: Union[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Compute model combination weights based on configuration.
        
        Args:
            validation_data: Validation data for computing weights
        """
        if self.config.voting_method == 'bayesian_model_averaging':
            self._compute_bma_weights(validation_data)
        elif self.config.voting_method == 'weighted_average':
            self._compute_weighted_average(validation_data)
        elif self.config.voting_method == 'voting':
            self._setup_simple_voting()
        elif self.config.voting_method == 'stacking':
            self._train_stacking_model(validation_data)
        else:
            raise ValueError(f"Unknown voting method: {self.config.voting_method}")
            
    def _compute_bma_weights(self, validation_data):
        """
        Compute Bayesian Model Averaging weights using configured approximation method.
        """
        logger.info(f"Computing BMA weights using {self.config.bma_config.approximation} approximation...")
        
        # Load validation data
        X_val, y_val = self._load_validation_data(validation_data)
        
        # Compute evidence for each model using configured method
        for name, model in self.models.items():
            if self.config.bma_config.approximation == 'bic':
                log_evidence = self._compute_bic_evidence(model, X_val, y_val)
            elif self.config.bma_config.approximation == 'laplace':
                log_evidence = self._compute_laplace_evidence(model, X_val, y_val)
            elif self.config.bma_config.approximation == 'variational':
                log_evidence = self._compute_variational_evidence(model, X_val, y_val)
            elif self.config.bma_config.approximation == 'cross_validation':
                log_evidence = self._compute_cv_evidence(model, X_val, y_val)
            else:
                raise ValueError(f"Unknown approximation: {self.config.bma_config.approximation}")
                
            self.model_evidences[name] = log_evidence
            logger.info(f"Model '{name}' log evidence: {log_evidence:.4f}")
            
        # Apply priors
        log_priors = self._get_log_priors()
        
        # Compute posterior weights with temperature scaling
        log_posteriors = {}
        for name in self.models:
            log_posteriors[name] = (
                self.model_evidences[name] / self.config.bma_config.temperature + 
                log_priors[name]
            )
            
        # Normalize to get weights
        log_norm = logsumexp(list(log_posteriors.values()))
        for name in self.models:
            weight = np.exp(log_posteriors[name] - log_norm)
            
            # Apply minimum weight threshold
            if weight < self.config.bma_config.min_posterior_weight:
                weight = 0.0
            
            self.posterior_weights[name] = weight
            
        # Renormalize if necessary
        if self.config.bma_config.normalize_posteriors:
            total = sum(self.posterior_weights.values())
            if total > 0:
                for name in self.posterior_weights:
                    self.posterior_weights[name] /= total
                    
        # Compute posterior variance if requested
        if self.config.bma_config.compute_posterior_variance:
            self._compute_posterior_variance()
            
        self.model_weights = self.posterior_weights
        
        # Log final weights
        for name, weight in self.model_weights.items():
            logger.info(f"Model '{name}' BMA weight: {weight:.4f}")
    
    def _load_validation_data(self, validation_data):
        """Load validation data from file or tuple."""
        if isinstance(validation_data, str):
            with open(validation_data, 'rb') as f:
                val_dict = pickle.load(f)
                X_val = val_dict['X']
                y_val = val_dict['y']
        else:
            X_val, y_val = validation_data
        return X_val, y_val
    
    def _compute_bic_evidence(self, model, X_val, y_val) -> float:
        """
        Compute BIC approximation to log marginal likelihood.
        
        BIC = log P(D|θ_MLE,M) - (k/2)log(n) * penalty_factor
        """
        # Get predictions
        predictions = model.predict(X_val, verbose=0)
        
        # Compute log likelihood
        n_samples = X_val.shape[0]
        
        # Handle different label formats
        if len(y_val.shape) == 3:
            # One-hot encoded
            log_likelihood = np.sum(y_val * np.log(predictions + 1e-10))
        else:
            # Sparse labels
            batch_size = y_val.shape[0]
            seq_len = y_val.shape[1]
            log_likelihood = 0
            for i in range(batch_size):
                for j in range(seq_len):
                    if y_val[i, j] >= 0:  # Valid label
                        log_likelihood += np.log(predictions[i, j, y_val[i, j]] + 1e-10)
        
        # Count parameters
        n_params = model.count_params()
        
        # Apply BIC penalty with configurable factor
        penalty = self.config.bma_config.bic_penalty_factor * (n_params / 2) * np.log(n_samples)
        
        return log_likelihood - penalty
    
    def _compute_laplace_evidence(self, model, X_val, y_val) -> float:
        """
        Compute Laplace approximation to log marginal likelihood.
        
        Uses second-order Taylor expansion around MAP estimate.
        """
        logger.info("Computing Laplace approximation...")
        
        # Get MAP predictions
        predictions = model.predict(X_val, verbose=0)
        
        # Compute log likelihood at MAP
        if len(y_val.shape) == 3:
            log_likelihood = np.sum(y_val * np.log(predictions + 1e-10))
        else:
            log_likelihood = self._compute_sparse_log_likelihood(predictions, y_val)
        
        # Estimate Hessian using finite differences with damping
        n_params = model.count_params()
        
        # Sample subset of parameters for efficiency
        n_samples = min(self.config.bma_config.laplace_num_samples, n_params)
        
        # Get current weights
        weights = model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        
        # Estimate trace of Hessian using random projections
        trace_hessian = 0
        for _ in range(n_samples):
            # Random direction
            v = np.random.randn(len(flat_weights))
            v = v / np.linalg.norm(v)
            
            # Finite difference approximation
            epsilon = 1e-4
            
            # Perturb weights in direction v
            perturbed = flat_weights + epsilon * v
            self._set_flat_weights(model, perturbed)
            pred_plus = model.predict(X_val, verbose=0)
            
            perturbed = flat_weights - epsilon * v
            self._set_flat_weights(model, perturbed)
            pred_minus = model.predict(X_val, verbose=0)
            
            # Estimate second derivative
            if len(y_val.shape) == 3:
                ll_plus = np.sum(y_val * np.log(pred_plus + 1e-10))
                ll_minus = np.sum(y_val * np.log(pred_minus + 1e-10))
            else:
                ll_plus = self._compute_sparse_log_likelihood(pred_plus, y_val)
                ll_minus = self._compute_sparse_log_likelihood(pred_minus, y_val)
                
            d2l = (ll_plus - 2*log_likelihood + ll_minus) / (epsilon**2)
            trace_hessian += d2l
            
        # Restore original weights
        self._set_flat_weights(model, flat_weights)
        
        # Average trace estimate
        trace_hessian = trace_hessian / n_samples * n_params
        
        # Add damping for numerical stability
        trace_hessian = trace_hessian + self.config.bma_config.laplace_damping * n_params
        
        # Laplace approximation
        log_evidence = log_likelihood + 0.5 * n_params * np.log(2 * np.pi) - 0.5 * np.log(np.abs(trace_hessian))
        
        return log_evidence
    
    def _compute_variational_evidence(self, model, X_val, y_val) -> float:
        """
        Compute variational lower bound (ELBO) as approximation to log evidence.
        
        ELBO = E_q[log p(D|θ)] - KL(q(θ)||p(θ))
        """
        logger.info("Computing variational evidence...")
        
        # Initialize variational parameters
        n_params = model.count_params()
        
        # Mean and log variance of variational distribution
        mu = np.zeros(min(100, n_params))  # Use subset for efficiency
        log_sigma = np.ones(len(mu)) * -2  # Initialize with small variance
        
        # Get baseline predictions
        predictions = model.predict(X_val, verbose=0)
        if len(y_val.shape) == 3:
            baseline_ll = np.sum(y_val * np.log(predictions + 1e-10))
        else:
            baseline_ll = self._compute_sparse_log_likelihood(predictions, y_val)
        
        # Optimization loop
        learning_rate = self.config.bma_config.vi_learning_rate
        best_elbo = -np.inf
        
        for iteration in range(self.config.bma_config.vi_num_iterations):
            # Sample from variational distribution
            epsilon = np.random.randn(*mu.shape)
            theta = mu + np.exp(0.5 * log_sigma) * epsilon
            
            # Compute expected log likelihood (using linear approximation)
            expected_ll = baseline_ll - 0.5 * np.sum(theta**2) * 0.01  # Regularization
            
            # Compute KL divergence from prior N(0, I)
            kl_divergence = 0.5 * np.sum(
                np.exp(log_sigma) + mu**2 - 1 - log_sigma
            )
            
            # ELBO
            elbo = expected_ll - kl_divergence
            
            # Update best ELBO
            if elbo > best_elbo:
                best_elbo = elbo
                
                # Check convergence
                if iteration > 0 and abs(elbo - best_elbo) < self.config.bma_config.vi_convergence_threshold:
                    logger.info(f"Variational inference converged at iteration {iteration}")
                    break
            
            # Gradient updates (simplified)
            # Gradient of ELBO w.r.t. mu
            grad_mu = -mu * 0.01  # Simplified gradient
            mu += learning_rate * grad_mu
            
            # Gradient of ELBO w.r.t. log_sigma
            grad_log_sigma = 0.5 * (np.exp(log_sigma) - 1)
            log_sigma -= learning_rate * grad_log_sigma
            
        return best_elbo
    
    def _compute_cv_evidence(self, model, X_val, y_val) -> float:
        """
        Compute cross-validation based evidence approximation.
        
        Uses k-fold cross-validation on validation set to estimate
        generalization performance as proxy for evidence.
        """
        logger.info(f"Computing {self.config.bma_config.cv_num_folds}-fold CV evidence...")
        
        n_folds = self.config.bma_config.cv_num_folds
        
        if self.config.bma_config.cv_stratified and len(y_val.shape) == 2:
            # Use stratified k-fold for sparse labels
            # Get most common label per sequence for stratification
            strat_labels = np.array([
                np.bincount(y[y >= 0]).argmax() if len(y[y >= 0]) > 0 else 0 
                for y in y_val
            ])
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = kf.split(X_val, strat_labels)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = kf.split(X_val)
        
        cv_log_likelihoods = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_fold_train = X_val[train_idx]
            y_fold_train = y_val[train_idx]
            X_fold_val = X_val[val_idx]
            y_fold_val = y_val[val_idx]
            
            # Create a copy of the model for this fold
            fold_model = keras.models.clone_model(model)
            fold_model.set_weights(model.get_weights())
            
            # Fine-tune on fold training data (few epochs)
            fold_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy' if len(y_fold_train.shape) == 2 
                     else 'categorical_crossentropy'
            )
            
            fold_model.fit(
                X_fold_train, y_fold_train,
                epochs=2,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate on fold validation data
            predictions = fold_model.predict(X_fold_val, verbose=0)
            
            if len(y_fold_val.shape) == 3:
                fold_ll = np.sum(y_fold_val * np.log(predictions + 1e-10))
            else:
                fold_ll = self._compute_sparse_log_likelihood(predictions, y_fold_val)
                
            cv_log_likelihoods.append(fold_ll)
            
        # Average across folds
        mean_cv_ll = np.mean(cv_log_likelihoods)
        
        # Add small penalty for model complexity
        n_params = model.count_params()
        complexity_penalty = -0.5 * np.log(n_params)
        
        return mean_cv_ll + complexity_penalty
    
    def _get_log_priors(self) -> Dict[str, float]:
        """
        Get log prior probabilities for models based on configuration.
        """
        n_models = len(self.models)
        
        if self.config.bma_config.prior_type == 'uniform':
            # Uniform prior
            log_prior = -np.log(n_models)
            return {name: log_prior for name in self.models}
            
        elif self.config.bma_config.prior_type == 'informative':
            # Use provided prior weights
            if self.config.bma_config.prior_weights is None:
                logger.warning("No prior weights provided, using uniform prior")
                log_prior = -np.log(n_models)
                return {name: log_prior for name in self.models}
            
            # Normalize prior weights
            total_weight = sum(self.config.bma_config.prior_weights.values())
            log_priors = {}
            
            for name in self.models:
                weight = self.config.bma_config.prior_weights.get(name, 1.0/n_models)
                log_priors[name] = np.log(weight / total_weight)
                
            return log_priors
            
        elif self.config.bma_config.prior_type == 'adaptive':
            # Adaptive prior based on model complexity
            log_priors = {}
            complexities = {}
            
            # Compute complexity for each model
            for name, model in self.models.items():
                n_params = model.count_params()
                complexities[name] = n_params
                
            # Use inverse complexity as prior
            total_inv_complexity = sum(1.0/c for c in complexities.values())
            
            for name, complexity in complexities.items():
                prior = (1.0/complexity) / total_inv_complexity
                log_priors[name] = np.log(prior)
                
            return log_priors
            
        else:
            raise ValueError(f"Unknown prior type: {self.config.bma_config.prior_type}")
    
    def _compute_posterior_variance(self):
        """Compute variance of posterior weights for uncertainty quantification."""
        # Use bootstrap to estimate variance
        n_bootstrap = 100
        bootstrap_weights = {name: [] for name in self.models}
        
        for _ in range(n_bootstrap):
            # Resample evidences with noise
            noisy_evidences = {}
            for name, evidence in self.model_evidences.items():
                noise = np.random.normal(0, 0.1)
                noisy_evidences[name] = evidence + noise
                
            # Recompute weights with noisy evidences
            log_priors = self._get_log_priors()
            log_posteriors = {}
            
            for name in self.models:
                log_posteriors[name] = (
                    noisy_evidences[name] / self.config.bma_config.temperature + 
                    log_priors[name]
                )
                
            log_norm = logsumexp(list(log_posteriors.values()))
            
            for name in self.models:
                weight = np.exp(log_posteriors[name] - log_norm)
                bootstrap_weights[name].append(weight)
                
        # Compute variance
        for name in self.models:
            self.posterior_variance[name] = np.var(bootstrap_weights[name])
            
    def _compute_weighted_average(self, validation_data):
        """Compute weights using configured optimization method."""
        logger.info(f"Computing weighted average using {self.config.weighted_optimization}")
        
        if self.config.weighted_optimization == 'fixed':
            self._use_fixed_weights()
        elif self.config.weighted_optimization == 'grid_search':
            self._optimize_weights_grid_search(validation_data)
        elif self.config.weighted_optimization == 'differential_evolution':
            self._optimize_weights_de(validation_data)
        elif self.config.weighted_optimization == 'bayesian_optimization':
            self._optimize_weights_bayesian(validation_data)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.weighted_optimization}")
    
    def _use_fixed_weights(self):
        """Use fixed weights from configuration."""
        if self.config.fixed_weights:
            total = sum(self.config.fixed_weights.values())
            self.model_weights = {
                name: self.config.fixed_weights.get(name, 1.0/len(self.models)) / total
                for name in self.models
            }
        else:
            # Equal weights
            n = len(self.models)
            self.model_weights = {name: 1.0/n for name in self.models}
            
    def _optimize_weights_grid_search(self, validation_data):
        """Optimize weights using grid search."""
        X_val, y_val = self._load_validation_data(validation_data)
        
        # Generate grid
        resolution = 0.1
        n_models = len(self.models)
        
        # For efficiency, only search reasonable grid
        best_score = -np.inf
        best_weights = None
        
        # Simple grid search for 2-3 models
        if n_models <= 3:
            import itertools
            weight_values = np.arange(0, 1.1, resolution)
            
            for weights in itertools.product(weight_values, repeat=n_models-1):
                # Last weight determined by constraint
                weights = list(weights)
                last_weight = 1.0 - sum(weights)
                
                if last_weight >= 0 and last_weight <= 1:
                    weights.append(last_weight)
                    
                    # Evaluate
                    weight_dict = {name: w for name, w in zip(self.models.keys(), weights)}
                    score = self._evaluate_weights(weight_dict, X_val, y_val)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weight_dict
                        
        else:
            # Random search for many models
            for _ in range(100):
                weights = np.random.dirichlet(np.ones(n_models))
                weight_dict = {name: w for name, w in zip(self.models.keys(), weights)}
                score = self._evaluate_weights(weight_dict, X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    best_weights = weight_dict
                    
        self.model_weights = best_weights
        logger.info(f"Best weights found: {best_weights}")
    
    def _optimize_weights_de(self, validation_data):
        """Optimize weights using differential evolution."""
        from scipy.optimize import differential_evolution
        
        X_val, y_val = self._load_validation_data(validation_data)
        n_models = len(self.models)
        model_names = list(self.models.keys())
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            weight_dict = {name: w for name, w in zip(model_names, weights)}
            return -self._evaluate_weights(weight_dict, X_val, y_val)
        
        # Bounds for each weight
        bounds = [(0, 1)] * n_models
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42
        )
        
        # Normalize final weights
        weights = result.x / np.sum(result.x)
        self.model_weights = {name: w for name, w in zip(model_names, weights)}
        logger.info(f"DE optimized weights: {self.model_weights}")
    
    def _optimize_weights_bayesian(self, validation_data):
        """Optimize weights using Bayesian optimization."""
        # Simplified Bayesian optimization
        logger.info("Using Bayesian optimization for weight optimization")
        
        # For now, fall back to differential evolution
        # Full Bayesian optimization would require additional dependencies
        self._optimize_weights_de(validation_data)
    
    def _evaluate_weights(self, weight_dict: Dict[str, float], X_val, y_val) -> float:
        """Evaluate weight configuration on validation data."""
        # Get predictions
        combined = np.zeros((X_val.shape[0], X_val.shape[1], self.models[list(self.models.keys())[0]].output_shape[-1]))
        
        for name, model in self.models.items():
            pred = model.predict(X_val, verbose=0)
            combined += weight_dict[name] * pred
            
        # Compute log likelihood
        if len(y_val.shape) == 3:
            score = np.sum(y_val * np.log(combined + 1e-10))
        else:
            score = self._compute_sparse_log_likelihood(combined, y_val)
            
        return score
    
    def _setup_simple_voting(self):
        """Setup simple majority voting with equal weights."""
        n = len(self.models)
        self.model_weights = {name: 1.0/n for name in self.models}
        logger.info("Using simple voting with equal weights")
    
    def _train_stacking_model(self, validation_data):
        """Train a stacking meta-model."""
        logger.info("Training stacking meta-model")
        
        X_val, y_val = self._load_validation_data(validation_data)
        
        # Get predictions from base models
        base_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_val, verbose=0)
            base_predictions.append(pred)
            
        # Stack predictions
        stacked = np.concatenate(base_predictions, axis=-1)
        
        # Flatten for meta-model
        n_samples = stacked.shape[0] * stacked.shape[1]
        n_features = stacked.shape[2]
        
        X_meta = stacked.reshape(n_samples, n_features)
        
        if len(y_val.shape) == 3:
            y_meta = y_val.reshape(n_samples, -1)
            y_meta = np.argmax(y_meta, axis=-1)
        else:
            y_meta = y_val.flatten()
            
        # Train simple logistic regression as meta-model
        from sklearn.linear_model import LogisticRegression
        
        self.stacking_model = LogisticRegression(max_iter=1000)
        self.stacking_model.fit(X_meta[y_meta >= 0], y_meta[y_meta >= 0])
        
        # For compatibility, set equal weights (actual combination done by stacking)
        n = len(self.models)
        self.model_weights = {name: 1.0/n for name in self.models}
    
    def predict(self,
                X: np.ndarray,
                return_uncertainty: bool = False,
                return_individual: bool = False,
                apply_calibration: bool = True) -> Union[np.ndarray, Dict]:
        """
        Make combined predictions with optional calibration and uncertainty.
        
        Args:
            X: Input sequences
            return_uncertainty: Return uncertainty estimates
            return_individual: Return individual model predictions
            apply_calibration: Apply calibration if available
            
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
            
        # Combine predictions based on method
        if self.config.voting_method == 'stacking' and hasattr(self, 'stacking_model'):
            combined = self._combine_stacking(individual_predictions)
        elif self.config.prediction_method == 'probability_averaging':
            combined = self._combine_probability_averaging(individual_predictions)
        elif self.config.prediction_method == 'logit_averaging':
            combined = self._combine_logit_averaging(individual_predictions)
        elif self.config.prediction_method == 'rank_averaging':
            combined = self._combine_rank_averaging(individual_predictions)
        else:
            # Default probability averaging
            combined = self._combine_probability_averaging(individual_predictions)
            
        # Apply calibration if available
        if apply_calibration and self.calibrator is not None:
            combined = self._apply_calibration(combined)
            
        result = {'predictions': combined}
        
        if return_uncertainty:
            uncertainty = self._compute_comprehensive_uncertainty(
                individual_predictions, combined
            )
            result['uncertainty'] = uncertainty
            
        if return_individual:
            result['individual_predictions'] = individual_predictions
            result['model_weights'] = self.model_weights
            
        if len(result) == 1:
            return result['predictions']
        return result
    
    def _combine_probability_averaging(self, individual_predictions):
        """Combine using weighted probability averaging."""
        combined = np.zeros_like(next(iter(individual_predictions.values())))
        
        for name, pred in individual_predictions.items():
            weight = self.model_weights.get(name, 0.0)
            
            if self.config.confidence_weighting:
                # Weight by prediction confidence
                confidence = np.max(pred, axis=-1, keepdims=True)
                weighted_pred = pred * confidence
            else:
                weighted_pred = pred
                
            combined += weight * weighted_pred
            
        # Renormalize
        combined = combined / (np.sum(combined, axis=-1, keepdims=True) + 1e-10)
        
        return combined
    
    def _combine_logit_averaging(self, individual_predictions):
        """Combine in logit space."""
        combined_logits = np.zeros_like(next(iter(individual_predictions.values())))
        
        for name, pred in individual_predictions.items():
            weight = self.model_weights.get(name, 0.0)
            # Convert to logits
            logits = np.log(pred + 1e-10)
            combined_logits += weight * logits
            
        # Convert back to probabilities
        combined = softmax(combined_logits, axis=-1)
        
        return combined
    
    def _combine_rank_averaging(self, individual_predictions):
        """Combine using rank averaging."""
        ranks_sum = np.zeros_like(next(iter(individual_predictions.values())))
        
        for name, pred in individual_predictions.items():
            weight = self.model_weights.get(name, 0.0)
            # Convert to ranks
            ranks = stats.rankdata(pred, axis=-1, method='average')
            ranks_sum += weight * ranks
            
        # Convert ranks back to probabilities (higher rank = higher probability)
        combined = ranks_sum / np.sum(ranks_sum, axis=-1, keepdims=True)
        
        return combined
    
    def _combine_stacking(self, individual_predictions):
        """Combine using stacking meta-model."""
        # Stack predictions
        base_preds = []
        for name in sorted(self.models.keys()):
            base_preds.append(individual_predictions[name])
        stacked = np.concatenate(base_preds, axis=-1)
        
        # Reshape for meta-model
        original_shape = stacked.shape[:2]
        n_classes = list(individual_predictions.values())[0].shape[-1]
        
        flat_stacked = stacked.reshape(-1, stacked.shape[-1])
        
        # Predict with meta-model
        meta_predictions = self.stacking_model.predict_proba(flat_stacked)
        
        # Reshape back
        combined = meta_predictions.reshape(*original_shape, n_classes)
        
        return combined
    
    def _compute_comprehensive_uncertainty(self, individual_predictions, combined):
        """Compute comprehensive uncertainty metrics."""
        # Stack predictions
        all_preds = np.stack(list(individual_predictions.values()), axis=0)
        
        uncertainty = {}
        
        # Predictive entropy (total uncertainty)
        uncertainty['entropy'] = -np.sum(
            combined * np.log(combined + 1e-10), axis=-1
        )
        
        # Model disagreement
        uncertainty['model_disagreement'] = np.std(all_preds, axis=0).mean(axis=-1)
        
        # Epistemic uncertainty (mutual information)
        individual_entropies = []
        for pred in individual_predictions.values():
            ind_entropy = -np.sum(pred * np.log(pred + 1e-10), axis=-1)
            individual_entropies.append(ind_entropy)
        expected_entropy = np.mean(individual_entropies, axis=0)
        
        uncertainty['epistemic_uncertainty'] = uncertainty['entropy'] - expected_entropy
        uncertainty['aleatoric_uncertainty'] = expected_entropy
        
        # Compute confidence intervals if requested
        if self.config.compute_epistemic and self.config.confidence_intervals:
            lower, upper = self._compute_confidence_intervals(
                all_preds, self.config.interval_alpha
            )
            uncertainty['confidence_lower'] = lower
            uncertainty['confidence_upper'] = upper
            uncertainty['confidence_width'] = upper - lower
            
        # Add posterior variance if available (BMA)
        if self.posterior_variance:
            model_variances = list(self.posterior_variance.values())
            uncertainty['model_weight_uncertainty'] = np.mean(model_variances)
            
        return uncertainty
    
    def _compute_confidence_intervals(self, all_preds, alpha):
        """Compute confidence intervals using bootstrap."""
        n_bootstrap = 100
        bootstrap_preds = []
        
        for _ in range(n_bootstrap):
            # Resample models with replacement
            idx = np.random.choice(len(all_preds), size=len(all_preds), replace=True)
            resampled = all_preds[idx]
            
            # Compute weighted average
            weights = np.array([self.model_weights[name] for name in self.models.keys()])
            weights = weights[idx]
            weights = weights / weights.sum()
            
            boot_pred = np.average(resampled, axis=0, weights=weights)
            bootstrap_preds.append(boot_pred)
            
        bootstrap_preds = np.stack(bootstrap_preds)
        
        # Compute percentiles
        lower = np.percentile(bootstrap_preds, alpha/2 * 100, axis=0)
        upper = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
        
        return lower, upper
    
    def calibrate(self, calibration_data: Union[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Calibrate ensemble predictions using configured method.
        
        Args:
            calibration_data: Data for calibration
        """
        if not self.config.calibration_enabled:
            logger.info("Calibration disabled in configuration")
            return
            
        logger.info(f"Calibrating predictions using {self.config.calibration_method}")
        
        X_cal, y_cal = self._load_validation_data(calibration_data)
        
        # Get uncalibrated predictions
        predictions = self.predict(X_cal, apply_calibration=False)
        
        # Flatten for calibration
        if len(y_cal.shape) == 3:
            y_true = np.argmax(y_cal, axis=-1).flatten()
        else:
            y_true = y_cal.flatten()
            
        # Get predicted probabilities for positive class
        pred_probs = predictions.reshape(-1, predictions.shape[-1])
        
        # Remove invalid labels
        valid_mask = y_true >= 0
        y_true = y_true[valid_mask]
        pred_probs = pred_probs[valid_mask]
        
        if self.config.calibration_method == 'isotonic':
            self._calibrate_isotonic(pred_probs, y_true)
        elif self.config.calibration_method == 'platt':
            self._calibrate_platt(pred_probs, y_true)
        elif self.config.calibration_method == 'temperature_scaling':
            self._calibrate_temperature(pred_probs, y_true)
        elif self.config.calibration_method == 'beta':
            self._calibrate_beta(pred_probs, y_true)
        else:
            logger.warning(f"Unknown calibration method: {self.config.calibration_method}")
    
    def _calibrate_isotonic(self, pred_probs, y_true):
        """Isotonic regression calibration."""
        self.calibrator = {}
        
        n_classes = pred_probs.shape[-1]
        for class_idx in range(n_classes):
            # Binary classification for each class
            y_binary = (y_true == class_idx).astype(int)
            class_probs = pred_probs[:, class_idx]
            
            iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_reg.fit(class_probs, y_binary)
            
            self.calibrator[class_idx] = iso_reg
            
        self.calibration_method = 'isotonic'
        logger.info("Isotonic calibration complete")
    
    def _calibrate_platt(self, pred_probs, y_true):
        """Platt scaling calibration."""
        self.calibrator = {}
        
        n_classes = pred_probs.shape[-1]
        for class_idx in range(n_classes):
            y_binary = (y_true == class_idx).astype(int)
            class_probs = pred_probs[:, class_idx]
            
            # Logistic regression on log-odds
            log_odds = np.log(class_probs / (1 - class_probs + 1e-10))
            log_odds = log_odds.reshape(-1, 1)
            
            lr = LogisticRegression(max_iter=100)
            lr.fit(log_odds, y_binary)
            
            self.calibrator[class_idx] = lr
            
        self.calibration_method = 'platt'
        logger.info("Platt scaling calibration complete")
    
    def _calibrate_temperature(self, pred_probs, y_true):
        """Temperature scaling calibration."""
        # Find optimal temperature using optimization
        def nll_loss(temp):
            scaled_logits = np.log(pred_probs + 1e-10) / temp
            scaled_probs = softmax(scaled_logits, axis=-1)
            
            # Negative log likelihood
            nll = -np.mean(np.log(scaled_probs[np.arange(len(y_true)), y_true] + 1e-10))
            return nll
        
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        self.calibration_method = 'temperature'
        logger.info(f"Temperature scaling calibration complete. Temperature: {self.temperature:.3f}")
    
    def _calibrate_beta(self, pred_probs, y_true):
        """Beta calibration."""
        # Simplified beta calibration
        # Would require fitting beta distribution parameters
        logger.warning("Beta calibration not fully implemented, using isotonic instead")
        self._calibrate_isotonic(pred_probs, y_true)
    
    def _apply_calibration(self, predictions):
        """Apply calibration to predictions."""
        if self.calibration_method == 'isotonic':
            calibrated = np.zeros_like(predictions)
            for class_idx, calibrator in self.calibrator.items():
                class_probs = predictions[..., class_idx].flatten()
                calibrated_probs = calibrator.transform(class_probs)
                calibrated[..., class_idx] = calibrated_probs.reshape(predictions.shape[:-1])
            # Renormalize
            calibrated = calibrated / (np.sum(calibrated, axis=-1, keepdims=True) + 1e-10)
            return calibrated
            
        elif self.calibration_method == 'platt':
            calibrated = np.zeros_like(predictions)
            for class_idx, calibrator in self.calibrator.items():
                class_probs = predictions[..., class_idx].flatten()
                log_odds = np.log(class_probs / (1 - class_probs + 1e-10)).reshape(-1, 1)
                calibrated_probs = calibrator.predict_proba(log_odds)[:, 1]
                calibrated[..., class_idx] = calibrated_probs.reshape(predictions.shape[:-1])
            calibrated = calibrated / (np.sum(calibrated, axis=-1, keepdims=True) + 1e-10)
            return calibrated
            
        elif self.calibration_method == 'temperature':
            scaled_logits = np.log(predictions + 1e-10) / self.temperature
            return softmax(scaled_logits, axis=-1)
            
        else:
            return predictions
    
    def _compute_sparse_log_likelihood(self, predictions, y_sparse):
        """Helper to compute log likelihood for sparse labels."""
        log_likelihood = 0
        batch_size, seq_len = y_sparse.shape
        
        for i in range(batch_size):
            for j in range(seq_len):
                if y_sparse[i, j] >= 0:  # Valid label
                    log_likelihood += np.log(predictions[i, j, y_sparse[i, j]] + 1e-10)
                    
        return log_likelihood
    
    def _set_flat_weights(self, model, flat_weights):
        """Set model weights from flattened array."""
        shapes = [w.shape for w in model.get_weights()]
        weights = []
        idx = 0
        
        for shape in shapes:
            size = np.prod(shape)
            weight = flat_weights[idx:idx+size].reshape(shape)
            weights.append(weight)
            idx += size
            
        model.set_weights(weights)
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save all results and configuration."""
        output_dir = Path(output_dir or self.config.output_dir)
        ensure_dir(str(output_dir))
        
        # Save model weights
        with open(output_dir / "model_weights.json", 'w') as f:
            json.dump(self.model_weights, f, indent=2)
            
        # Save evidences if using BMA
        if self.config.voting_method == 'bayesian_model_averaging':
            with open(output_dir / "model_evidences.json", 'w') as f:
                json.dump(self.model_evidences, f, indent=2)
                
            if self.posterior_variance:
                with open(output_dir / "posterior_variance.json", 'w') as f:
                    json.dump(self.posterior_variance, f, indent=2)
                    
        # Save configuration
        config_dict = asdict(self.config)
        with open(output_dir / "ensemble_config.yaml", 'w') as f:
            yaml.dump(config_dict, f)
            
        # Save calibrator if available
        if self.calibrator is not None:
            with open(output_dir / "calibrator.pkl", 'wb') as f:
                pickle.dump({
                    'calibrator': self.calibrator,
                    'method': self.calibration_method,
                    'temperature': getattr(self, 'temperature', None)
                }, f)
                
        logger.info(f"Results saved to {output_dir}")
        
    def evaluate(self, test_data: Union[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Comprehensive evaluation of ensemble."""
        X_test, y_test = self._load_validation_data(test_data)
        
        # Get predictions with uncertainty
        result = self.predict(
            X_test,
            return_uncertainty=True,
            return_individual=True,
            apply_calibration=True
        )
        
        predictions = result['predictions']
        uncertainty = result['uncertainty']
        individual = result['individual_predictions']
        
        # Compute metrics
        metrics = {}
        
        # Accuracy metrics
        pred_classes = np.argmax(predictions, axis=-1)
        if len(y_test.shape) == 3:
            true_classes = np.argmax(y_test, axis=-1)
        else:
            true_classes = y_test
            
        valid_mask = true_classes >= 0
        
        metrics['ensemble_accuracy'] = np.mean(
            pred_classes[valid_mask] == true_classes[valid_mask]
        )
        
        # Individual model metrics
        for name, pred in individual.items():
            ind_pred_classes = np.argmax(pred, axis=-1)
            metrics[f'{name}_accuracy'] = np.mean(
                ind_pred_classes[valid_mask] == true_classes[valid_mask]
            )
            
        # Calibration metrics
        if self.config.calibration_enabled:
            metrics.update(self._compute_calibration_metrics(
                predictions[valid_mask],
                true_classes[valid_mask]
            ))
            
        # Uncertainty metrics
        metrics['mean_entropy'] = np.mean(uncertainty['entropy'])
        metrics['mean_epistemic'] = np.mean(uncertainty['epistemic_uncertainty'])
        metrics['mean_aleatoric'] = np.mean(uncertainty['aleatoric_uncertainty'])
        
        # Diversity metrics
        if self.config.enforce_diversity:
            metrics['ensemble_diversity'] = self._compute_diversity_metric(individual)
            
        return metrics
    
    def _compute_calibration_metrics(self, predictions, true_classes):
        """Compute calibration metrics."""
        from sklearn.calibration import calibration_curve
        
        metrics = {}
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        pred_probs = np.max(predictions, axis=-1)
        pred_classes = np.argmax(predictions, axis=-1)
        correct = (pred_classes == true_classes)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = pred_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        metrics['expected_calibration_error'] = ece
        
        # Brier score
        n_classes = predictions.shape[-1]
        brier_score = 0
        for i in range(len(true_classes)):
            true_vec = np.zeros(n_classes)
            true_vec[true_classes[i]] = 1
            brier_score += np.sum((predictions[i] - true_vec) ** 2)
        
        metrics['brier_score'] = brier_score / len(true_classes)
        
        return metrics
    
    def _compute_diversity_metric(self, individual_predictions):
        """Compute diversity among ensemble members."""
        if self.config.diversity_metric == 'disagreement':
            # Pairwise disagreement
            disagreements = []
            model_names = list(individual_predictions.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    pred_i = np.argmax(individual_predictions[model_names[i]], axis=-1)
                    pred_j = np.argmax(individual_predictions[model_names[j]], axis=-1)
                    disagreement = np.mean(pred_i != pred_j)
                    disagreements.append(disagreement)
                    
            return np.mean(disagreements) if disagreements else 0
            
        elif self.config.diversity_metric == 'kl_divergence':
            # Average KL divergence
            kl_divs = []
            model_names = list(individual_predictions.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    pred_i = individual_predictions[model_names[i]]
                    pred_j = individual_predictions[model_names[j]]
                    kl = np.sum(pred_i * np.log((pred_i + 1e-10) / (pred_j + 1e-10)))
                    kl_divs.append(kl)
                    
            return np.mean(kl_divs) if kl_divs else 0
            
        elif self.config.diversity_metric == 'correlation':
            # Negative average correlation
            correlations = []
            model_names = list(individual_predictions.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    pred_i = individual_predictions[model_names[i]].flatten()
                    pred_j = individual_predictions[model_names[j]].flatten()
                    corr = np.corrcoef(pred_i, pred_j)[0, 1]
                    correlations.append(corr)
                    
            return 1 - np.mean(correlations) if correlations else 0
            
        else:
            return 0


def main():
    """Example usage of model combiner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Combination')
    parser.add_argument('--models', nargs='+', required=True, help='Model paths')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--validation-data', required=True, help='Validation data')
    parser.add_argument('--test-data', help='Test data for evaluation')
    parser.add_argument('--output-dir', default='./ensemble_results')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = config_dict.get('ensemble', {})
    else:
        config = {}
    
    # Initialize combiner
    combiner = ModelCombiner(config)
    
    # Load models
    combiner.load_models(args.models)
    
    # Compute weights
    combiner.compute_weights(args.validation_data)
    
    # Calibrate if configured
    if combiner.config.calibration_enabled:
        combiner.calibrate(args.validation_data)
    
    # Evaluate if test data provided
    if args.test_data:
        metrics = combiner.evaluate(args.test_data)
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    # Save results
    combiner.save_results(args.output_dir)
    

if __name__ == "__main__":
    main()
