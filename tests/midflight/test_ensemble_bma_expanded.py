"""
Comprehensive tests for BMA (Bayesian Model Averaging) and ensemble modeling.

Tests cover:
- Ensemble configuration
- Model diversity
- BMA weight computation (uniform, informative, adaptive priors)
- Evidence approximation methods (BIC, Laplace, variational, cross-validation)
- Posterior computation and tempering
- Calibration methods
- Uncertainty quantification
- Model selection criteria
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path
from scipy import stats

# Add tempest package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

# Import from the tempest package
from tempest.utils.config import EnsembleConfig, BMAConfig
from tempest.training.ensemble import (
    EnsembleModel,
    BayesianModelAveraging,
    compute_bic,
    compute_laplace_evidence,
    compute_variational_evidence,
    compute_cv_evidence
)


class TestEnsembleConfig:
    """Test ensemble configuration handling."""
    
    def test_default_ensemble_config(self):
        """Test default ensemble configuration."""
        config = EnsembleConfig()
        
        assert config.enabled == False
        assert config.num_models == 3
        assert config.voting_method == 'weighted_average'
        assert config.diversity is not None
        assert config.bma_config is not None
    
    def test_custom_ensemble_config(self):
        """Test custom ensemble configuration from dict."""
        config_dict = {
            'enabled': True,
            'num_models': 5,
            'voting_method': 'bayesian_model_averaging',
            'bma_config': {
                'enabled': True,
                'prior_type': 'informative',
                'approximation': 'laplace',
                'temperature': 1.5
            }
        }
        
        config = EnsembleConfig.from_dict(config_dict)
        
        assert config.enabled == True
        assert config.num_models == 5
        assert config.voting_method == 'bayesian_model_averaging'
        assert config.bma_config.enabled == True
        assert config.bma_config.prior_type == 'informative'
        assert config.bma_config.approximation == 'laplace'
        assert config.bma_config.temperature == 1.5


class TestBMAConfiguration:
    """Test Bayesian Model Averaging configuration."""
    
    def test_bma_prior_types(self):
        """Test different prior type configurations."""
        # Uniform prior
        config_uniform = BMAConfig.from_dict({
            'enabled': True,
            'prior_type': 'uniform'
        })
        assert config_uniform.prior_type == 'uniform'
        
        # Informative prior
        config_informative = BMAConfig.from_dict({
            'enabled': True,
            'prior_type': 'informative',
            'prior_weights': {
                'standard_crf': 0.25,
                'hybrid_crf': 0.50,
                'length_constrained': 0.25
            }
        })
        assert config_informative.prior_type == 'informative'
        assert sum(config_informative.prior_weights.values()) == 1.0
        
        # Adaptive prior
        config_adaptive = BMAConfig.from_dict({
            'enabled': True,
            'prior_type': 'adaptive'
        })
        assert config_adaptive.prior_type == 'adaptive'
    
    def test_approximation_methods(self):
        """Test different evidence approximation method configurations."""
        methods = ['bic', 'laplace', 'variational', 'cross_validation']
        
        for method in methods:
            config = BMAConfig.from_dict({
                'enabled': True,
                'approximation': method
            })
            assert config.approximation == method
    
    def test_approximation_parameters(self):
        """Test approximation-specific parameters."""
        config = BMAConfig.from_dict({
            'enabled': True,
            'approximation': 'laplace',
            'approximation_params': {
                'laplace': {
                    'num_samples': 2000,
                    'damping': 0.05
                }
            }
        })
        
        assert config.approximation_params['laplace']['num_samples'] == 2000
        assert config.approximation_params['laplace']['damping'] == 0.05
    
    def test_posterior_tempering(self):
        """Test posterior temperature settings."""
        # Sharp posterior (confident)
        config_sharp = BMAConfig.from_dict({
            'enabled': True,
            'temperature': 0.5
        })
        assert config_sharp.temperature == 0.5
        
        # Standard posterior
        config_standard = BMAConfig.from_dict({
            'enabled': True,
            'temperature': 1.0
        })
        assert config_standard.temperature == 1.0
        
        # Flat posterior (diverse)
        config_flat = BMAConfig.from_dict({
            'enabled': True,
            'temperature': 2.0
        })
        assert config_flat.temperature == 2.0


class TestBayesianModelAveraging:
    """Test Bayesian Model Averaging implementation."""
    
    def setup_method(self):
        """Setup mock models for testing."""
        self.mock_models = []
        for i in range(3):
            model = Mock()
            model.name = f'model_{i}'
            model.predict = Mock(return_value=np.random.randn(10, 5))
            model.evaluate = Mock(return_value={'loss': 0.1 * (i + 1)})
            self.mock_models.append(model)
    
    def test_uniform_prior_weights(self):
        """Test BMA with uniform prior weights."""
        bma = BayesianModelAveraging(
            models=self.mock_models,
            prior_type='uniform'
        )
        
        prior_weights = bma.compute_prior_weights()
        
        # Should be equal for all models
        expected = 1.0 / len(self.mock_models)
        assert all(np.isclose(w, expected) for w in prior_weights.values())
        assert sum(prior_weights.values()) == pytest.approx(1.0)
    
    def test_informative_prior_weights(self):
        """Test BMA with informative prior weights."""
        prior_weights = {
            'model_0': 0.5,
            'model_1': 0.3,
            'model_2': 0.2
        }
        
        bma = BayesianModelAveraging(
            models=self.mock_models,
            prior_type='informative',
            prior_weights=prior_weights
        )
        
        computed_priors = bma.compute_prior_weights()
        
        for model_name, expected_weight in prior_weights.items():
            assert computed_priors[model_name] == pytest.approx(expected_weight)
    
    def test_adaptive_prior_weights(self):
        """Test BMA with adaptive prior weights based on validation performance."""
        val_performances = {
            'model_0': 0.9,  # Best
            'model_1': 0.8,
            'model_2': 0.7   # Worst
        }
        
        bma = BayesianModelAveraging(
            models=self.mock_models,
            prior_type='adaptive'
        )
        
        # Mock validation performance
        bma.validation_performances = val_performances
        
        prior_weights = bma.compute_adaptive_prior_weights()
        
        # Better models should have higher priors
        assert prior_weights['model_0'] > prior_weights['model_1']
        assert prior_weights['model_1'] > prior_weights['model_2']
        assert sum(prior_weights.values()) == pytest.approx(1.0)


class TestEvidenceApproximation:
    """Test different model evidence approximation methods."""
    
    def setup_method(self):
        """Setup for evidence tests."""
        self.n_samples = 100
        self.n_features = 10
        self.n_params = 50
        
        # Mock data
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 5, self.n_samples)
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.random.randn(self.n_samples, 5))
        self.mock_model.evaluate = Mock(return_value={'loss': 0.5})
        self.mock_model.count_params = Mock(return_value=self.n_params)
    
    def test_bic_approximation(self):
        """Test BIC (Bayesian Information Criterion) approximation."""
        log_likelihood = -50.0  # Negative loss
        
        bic = compute_bic(
            log_likelihood=log_likelihood,
            n_params=self.n_params,
            n_samples=self.n_samples,
            penalty_factor=1.0
        )
        
        expected_bic = -2 * log_likelihood + self.n_params * np.log(self.n_samples)
        assert bic == pytest.approx(expected_bic)
        
        # Test with different penalty factor
        bic_strict = compute_bic(
            log_likelihood=log_likelihood,
            n_params=self.n_params,
            n_samples=self.n_samples,
            penalty_factor=2.0  # Stricter penalty
        )
        
        assert bic_strict > bic  # Should be more penalized
    
    def test_laplace_approximation(self):
        """Test Laplace approximation for model evidence."""
        with patch('tempest.training.ensemble.compute_hessian') as mock_hessian:
            # Mock Hessian computation
            mock_hessian.return_value = np.eye(self.n_params) * 0.1
            
            evidence = compute_laplace_evidence(
                model=self.mock_model,
                data=(self.X, self.y),
                num_samples=1000,
                damping=0.01
            )
            
            # Evidence should be a scalar
            assert isinstance(evidence, (float, np.floating))
            assert not np.isnan(evidence)
            assert not np.isinf(evidence)
    
    def test_variational_approximation(self):
        """Test variational inference approximation."""
        evidence = compute_variational_evidence(
            model=self.mock_model,
            data=(self.X, self.y),
            num_iterations=100,
            learning_rate=0.01,
            convergence_threshold=1e-4
        )
        
        # Should return evidence lower bound (ELBO)
        assert isinstance(evidence, (float, np.floating))
        assert not np.isnan(evidence)
    
    def test_cross_validation_evidence(self):
        """Test cross-validation based evidence estimation."""
        evidence = compute_cv_evidence(
            model=self.mock_model,
            data=(self.X, self.y),
            num_folds=5,
            stratified=True
        )
        
        # Should return average CV score
        assert isinstance(evidence, (float, np.floating))
        assert evidence <= 0  # Log likelihood should be negative


class TestPosteriorComputation:
    """Test posterior weight computation and manipulation."""
    
    def setup_method(self):
        """Setup for posterior tests."""
        self.prior_weights = {
            'model_0': 0.4,
            'model_1': 0.3,
            'model_2': 0.3
        }
        
        self.evidences = {
            'model_0': -100.0,  # Best evidence (least negative)
            'model_1': -120.0,
            'model_2': -150.0   # Worst evidence
        }
    
    def test_posterior_computation(self):
        """Test basic posterior weight computation."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            temperature=1.0
        )
        
        posteriors = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=self.evidences,
            temperature=1.0
        )
        
        # Posteriors should sum to 1
        assert sum(posteriors.values()) == pytest.approx(1.0)
        
        # Model with best evidence should have highest posterior
        assert posteriors['model_0'] > posteriors['model_1']
        assert posteriors['model_1'] > posteriors['model_2']
    
    def test_posterior_tempering(self):
        """Test the effect of temperature on posterior weights."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)]
        )
        
        # Sharp posterior (T < 1)
        posteriors_sharp = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=self.evidences,
            temperature=0.5
        )
        
        # Standard posterior (T = 1)
        posteriors_standard = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=self.evidences,
            temperature=1.0
        )
        
        # Flat posterior (T > 1)
        posteriors_flat = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=self.evidences,
            temperature=2.0
        )
        
        # Sharp should be more concentrated on best model
        best_model = 'model_0'
        assert posteriors_sharp[best_model] > posteriors_standard[best_model]
        assert posteriors_standard[best_model] > posteriors_flat[best_model]
        
        # Flat should be more uniform
        variance_sharp = np.var(list(posteriors_sharp.values()))
        variance_flat = np.var(list(posteriors_flat.values()))
        assert variance_sharp > variance_flat
    
    def test_minimum_posterior_weight(self):
        """Test enforcement of minimum posterior weight."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            min_posterior_weight=0.05
        )
        
        # Use extreme evidences to force near-zero weights
        extreme_evidences = {
            'model_0': -10.0,   # Much better
            'model_1': -1000.0,
            'model_2': -1000.0
        }
        
        posteriors = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=extreme_evidences,
            temperature=1.0,
            min_weight=0.05
        )
        
        # All models should have at least min weight
        for weight in posteriors.values():
            assert weight >= 0.05
        
        # Should still sum to 1
        assert sum(posteriors.values()) == pytest.approx(1.0)
    
    def test_posterior_variance_computation(self):
        """Test computation of posterior variance for uncertainty."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            compute_posterior_variance=True
        )
        
        posteriors = bma.compute_posterior_weights(
            prior_weights=self.prior_weights,
            evidences=self.evidences,
            temperature=1.0
        )
        
        # Compute variance
        variance = bma.compute_posterior_variance(posteriors)
        
        assert isinstance(variance, (float, np.floating))
        assert variance >= 0
        
        # More uniform posteriors should have higher variance
        uniform_posteriors = {f'model_{i}': 1/3 for i in range(3)}
        peaked_posteriors = {'model_0': 0.9, 'model_1': 0.05, 'model_2': 0.05}
        
        var_uniform = bma.compute_posterior_variance(uniform_posteriors)
        var_peaked = bma.compute_posterior_variance(peaked_posteriors)
        
        assert var_uniform > var_peaked


class TestModelSelection:
    """Test model selection criteria."""
    
    def test_evidence_threshold_selection(self):
        """Test model selection based on evidence threshold."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(5)]
        )
        
        evidences = {
            'model_0': -100.0,
            'model_1': -102.0,
            'model_2': -105.0,
            'model_3': -150.0,  # Much worse
            'model_4': -200.0   # Much worse
        }
        
        # Select models with evidence ratio above threshold
        selected_models = bma.select_models_by_evidence(
            evidences=evidences,
            evidence_threshold=0.05  # 5% of best evidence
        )
        
        # Should exclude models 3 and 4
        assert 'model_0' in selected_models
        assert 'model_1' in selected_models
        assert 'model_2' in selected_models
        assert 'model_3' not in selected_models
        assert 'model_4' not in selected_models
    
    def test_single_best_model_selection(self):
        """Test selection of single best model instead of averaging."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            use_model_averaging=False
        )
        
        posteriors = {
            'model_0': 0.6,  # Best
            'model_1': 0.3,
            'model_2': 0.1
        }
        
        best_model = bma.select_best_model(posteriors)
        
        assert best_model == 'model_0'


class TestCalibration:
    """Test prediction calibration methods."""
    
    def setup_method(self):
        """Setup for calibration tests."""
        # Mock predictions (probabilities)
        np.random.seed(42)
        self.n_samples = 1000
        self.n_classes = 5
        self.predictions = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
        self.true_labels = np.random.randint(0, self.n_classes, self.n_samples)
    
    def test_isotonic_calibration(self):
        """Test isotonic regression calibration."""
        from sklearn.isotonic import IsotonicRegression
        
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        
        # Calibrate predictions for one class
        class_idx = 0
        calibrator.fit(self.predictions[:800, class_idx], 
                      (self.true_labels[:800] == class_idx).astype(float))
        
        calibrated = calibrator.predict(self.predictions[800:, class_idx])
        
        # Calibrated probabilities should be in [0, 1]
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
    
    def test_platt_scaling(self):
        """Test Platt scaling (sigmoid) calibration."""
        from sklearn.linear_model import LogisticRegression
        
        # Use logistic regression for Platt scaling
        calibrator = LogisticRegression(max_iter=100, solver='lbfgs')
        
        # Binary calibration for one class
        class_idx = 0
        X = self.predictions[:800, class_idx].reshape(-1, 1)
        y = (self.true_labels[:800] == class_idx).astype(int)
        
        calibrator.fit(X, y)
        
        X_test = self.predictions[800:, class_idx].reshape(-1, 1)
        calibrated = calibrator.predict_proba(X_test)[:, 1]
        
        # Should produce valid probabilities
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
    
    def test_temperature_scaling(self):
        """Test temperature scaling calibration."""
        class TemperatureScaling:
            def __init__(self):
                self.temperature = 1.0
            
            def fit(self, logits, labels):
                """Find optimal temperature."""
                from scipy.optimize import minimize
                
                def nll(T):
                    scaled_logits = logits / T[0]
                    probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
                    # Negative log likelihood
                    return -np.log(probs[np.arange(len(labels)), labels] + 1e-10).mean()
                
                result = minimize(nll, [1.0], bounds=[(0.1, 10.0)])
                self.temperature = result.x[0]
                return self
            
            def transform(self, logits):
                scaled_logits = logits / self.temperature
                return np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        
        # Create logits from probabilities
        logits = np.log(self.predictions + 1e-10)
        
        calibrator = TemperatureScaling()
        calibrator.fit(logits[:800], self.true_labels[:800])
        
        calibrated = calibrator.transform(logits[800:])
        
        # Should be valid probability distribution
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert np.all(calibrated >= 0.0)
    
    def test_beta_calibration(self):
        """Test beta calibration with binning."""
        n_bins = 15
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        # For each bin, fit beta distribution
        class_idx = 0
        predictions_class = self.predictions[:800, class_idx]
        labels_class = (self.true_labels[:800] == class_idx).astype(float)
        
        calibration_map = {}
        for i in range(n_bins):
            bin_mask = (predictions_class >= bin_edges[i]) & (predictions_class < bin_edges[i+1])
            if bin_mask.sum() > 0:
                bin_labels = labels_class[bin_mask]
                if len(np.unique(bin_labels)) > 1:
                    # Fit beta distribution
                    alpha, beta, _, _ = stats.beta.fit(bin_labels, floc=0, fscale=1)
                    calibration_map[i] = (alpha, beta)
        
        assert len(calibration_map) > 0
        
        # Apply calibration
        test_predictions = self.predictions[800:, class_idx]
        calibrated = np.zeros_like(test_predictions)
        
        for i in range(n_bins):
            bin_mask = (test_predictions >= bin_edges[i]) & (test_predictions < bin_edges[i+1])
            if bin_mask.sum() > 0 and i in calibration_map:
                alpha, beta = calibration_map[i]
                # Use beta mean as calibrated probability
                calibrated[bin_mask] = alpha / (alpha + beta)
        
        # Should produce valid probabilities
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)


class TestUncertaintyQuantification:
    """Test uncertainty quantification methods."""
    
    def setup_method(self):
        """Setup for uncertainty tests."""
        self.n_models = 5
        self.n_samples = 100
        self.n_classes = 3
        
        # Create diverse model predictions
        np.random.seed(42)
        self.model_predictions = []
        for i in range(self.n_models):
            # Add some diversity between models
            preds = np.random.dirichlet(np.ones(self.n_classes) * (i+1), self.n_samples)
            self.model_predictions.append(preds)
    
    def test_epistemic_uncertainty(self):
        """Test epistemic (model) uncertainty computation."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(self.n_models)],
            compute_epistemic=True
        )
        
        # Compute disagreement between models
        epistemic = bma.compute_epistemic_uncertainty(self.model_predictions)
        
        assert epistemic.shape == (self.n_samples,)
        assert np.all(epistemic >= 0)
        
        # Samples where models disagree more should have higher uncertainty
        # Create test case with high/low agreement
        agreed_predictions = [np.array([[0.9, 0.05, 0.05]]) for _ in range(self.n_models)]
        disagreed_predictions = [
            np.array([[0.9, 0.05, 0.05]]),
            np.array([[0.05, 0.9, 0.05]]),
            np.array([[0.05, 0.05, 0.9]]),
            np.array([[0.33, 0.33, 0.34]]),
            np.array([[0.5, 0.3, 0.2]])
        ]
        
        epistemic_low = bma.compute_epistemic_uncertainty(agreed_predictions)
        epistemic_high = bma.compute_epistemic_uncertainty(disagreed_predictions)
        
        assert epistemic_high[0] > epistemic_low[0]
    
    def test_aleatoric_uncertainty(self):
        """Test aleatoric (data) uncertainty computation."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(self.n_models)],
            compute_aleatoric=True
        )
        
        # Average entropy across models
        aleatoric = bma.compute_aleatoric_uncertainty(self.model_predictions)
        
        assert aleatoric.shape == (self.n_samples,)
        assert np.all(aleatoric >= 0)
        
        # High entropy (uniform) predictions should have high aleatoric uncertainty
        uniform_preds = [np.array([[0.33, 0.33, 0.34]]) for _ in range(self.n_models)]
        peaked_preds = [np.array([[0.95, 0.025, 0.025]]) for _ in range(self.n_models)]
        
        aleatoric_uniform = bma.compute_aleatoric_uncertainty(uniform_preds)
        aleatoric_peaked = bma.compute_aleatoric_uncertainty(peaked_preds)
        
        assert aleatoric_uniform[0] > aleatoric_peaked[0]
    
    def test_confidence_intervals(self):
        """Test computation of confidence intervals."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(self.n_models)],
            confidence_intervals=True,
            interval_alpha=0.05  # 95% CI
        )
        
        # Compute intervals from model predictions
        lower, upper, mean = bma.compute_confidence_intervals(
            self.model_predictions,
            alpha=0.05
        )
        
        assert lower.shape == (self.n_samples, self.n_classes)
        assert upper.shape == (self.n_samples, self.n_classes)
        assert mean.shape == (self.n_samples, self.n_classes)
        
        # Lower <= mean <= upper
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)
        
        # Intervals should be valid probabilities
        assert np.all(lower >= 0)
        assert np.all(upper <= 1)


class TestPredictionAggregation:
    """Test different prediction aggregation methods."""
    
    def setup_method(self):
        """Setup for aggregation tests."""
        self.n_samples = 50
        self.n_classes = 4
        self.model_predictions = []
        self.model_weights = [0.4, 0.3, 0.2, 0.1]  # Different weights
        
        np.random.seed(42)
        for i in range(4):
            preds = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
            self.model_predictions.append(preds)
    
    def test_probability_averaging(self):
        """Test simple probability averaging."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(4)],
            prediction_aggregation_method='probability_averaging'
        )
        
        aggregated = bma.aggregate_predictions(
            self.model_predictions,
            weights=self.model_weights,
            method='probability_averaging'
        )
        
        assert aggregated.shape == (self.n_samples, self.n_classes)
        assert np.allclose(aggregated.sum(axis=1), 1.0)
        
        # Check weighted average
        expected_first = sum(w * preds[0] for w, preds in zip(self.model_weights, self.model_predictions))
        assert np.allclose(aggregated[0], expected_first)
    
    def test_logit_averaging(self):
        """Test averaging in logit space."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(4)],
            prediction_aggregation_method='logit_averaging'
        )
        
        aggregated = bma.aggregate_predictions(
            self.model_predictions,
            weights=self.model_weights,
            method='logit_averaging'
        )
        
        assert aggregated.shape == (self.n_samples, self.n_classes)
        assert np.allclose(aggregated.sum(axis=1), 1.0)
        
        # Should be different from probability averaging
        prob_avg = bma.aggregate_predictions(
            self.model_predictions,
            weights=self.model_weights,
            method='probability_averaging'
        )
        
        assert not np.allclose(aggregated, prob_avg)
    
    def test_rank_averaging(self):
        """Test rank-based averaging."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(4)],
            prediction_aggregation_method='rank_averaging'
        )
        
        aggregated = bma.aggregate_predictions(
            self.model_predictions,
            weights=self.model_weights,
            method='rank_averaging'
        )
        
        assert aggregated.shape == (self.n_samples, self.n_classes)
        
        # Result should be ranks or rank-based scores
        # The exact implementation depends on your system
        assert aggregated.shape == (self.n_samples, self.n_classes)
    
    def test_confidence_weighting(self):
        """Test confidence-weighted aggregation."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(4)],
            confidence_weighting=True
        )
        
        # Create confidences based on entropy
        confidences = []
        for preds in self.model_predictions:
            entropy = -np.sum(preds * np.log(preds + 1e-10), axis=1)
            confidence = 1 / (1 + entropy)  # Lower entropy = higher confidence
            confidences.append(confidence)
        
        aggregated = bma.aggregate_with_confidence(
            self.model_predictions,
            base_weights=self.model_weights,
            confidences=confidences
        )
        
        assert aggregated.shape == (self.n_samples, self.n_classes)
        assert np.allclose(aggregated.sum(axis=1), 1.0)


class TestDiversityMetrics:
    """Test model diversity enforcement and measurement."""
    
    def setup_method(self):
        """Setup for diversity tests."""
        self.n_samples = 100
        self.n_classes = 3
        
        # Create similar and diverse predictions
        np.random.seed(42)
        base_preds = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
        
        # Similar models (small perturbations)
        self.similar_predictions = []
        for i in range(3):
            noise = np.random.normal(0, 0.01, base_preds.shape)
            preds = base_preds + noise
            preds = np.clip(preds, 0, 1)
            preds = preds / preds.sum(axis=1, keepdims=True)
            self.similar_predictions.append(preds)
        
        # Diverse models
        self.diverse_predictions = []
        for i in range(3):
            preds = np.random.dirichlet(np.ones(self.n_classes) * (i+0.5), self.n_samples)
            self.diverse_predictions.append(preds)
    
    def test_disagreement_metric(self):
        """Test disagreement-based diversity metric."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            diversity_metric='disagreement'
        )
        
        diversity_similar = bma.compute_diversity(
            self.similar_predictions,
            metric='disagreement'
        )
        
        diversity_diverse = bma.compute_diversity(
            self.diverse_predictions,
            metric='disagreement'
        )
        
        # Diverse models should have higher disagreement
        assert diversity_diverse > diversity_similar
    
    def test_kl_divergence_metric(self):
        """Test KL divergence diversity metric."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            diversity_metric='kl_divergence'
        )
        
        diversity_similar = bma.compute_diversity(
            self.similar_predictions,
            metric='kl_divergence'
        )
        
        diversity_diverse = bma.compute_diversity(
            self.diverse_predictions,
            metric='kl_divergence'
        )
        
        # Diverse models should have higher KL divergence
        assert diversity_diverse > diversity_similar
    
    def test_correlation_metric(self):
        """Test correlation-based diversity metric."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            diversity_metric='correlation'
        )
        
        diversity_similar = bma.compute_diversity(
            self.similar_predictions,
            metric='correlation'
        )
        
        diversity_diverse = bma.compute_diversity(
            self.diverse_predictions,
            metric='correlation'
        )
        
        # Similar models should have higher correlation (lower diversity score)
        # If using 1 - correlation as diversity metric
        assert diversity_diverse > diversity_similar
    
    def test_minimum_diversity_enforcement(self):
        """Test enforcement of minimum diversity threshold."""
        bma = BayesianModelAveraging(
            models=[Mock(name=f'model_{i}') for i in range(3)],
            enforce_diversity=True,
            min_diversity_threshold=0.1
        )
        
        # Check if models meet diversity threshold
        diversity = bma.compute_diversity(
            self.similar_predictions,
            metric='disagreement'
        )
        
        if diversity < 0.1:
            # Should trigger diversity enforcement mechanism
            assert bma.should_enforce_diversity(diversity)
        
        diversity_high = bma.compute_diversity(
            self.diverse_predictions,
            metric='disagreement'
        )
        
        if diversity_high >= 0.1:
            assert not bma.should_enforce_diversity(diversity_high)


class TestEvaluationMetrics:
    """Test ensemble evaluation metrics."""
    
    def setup_method(self):
        """Setup for evaluation tests."""
        self.n_samples = 200
        self.n_classes = 5
        
        np.random.seed(42)
        self.predictions = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
        self.true_labels = np.random.randint(0, self.n_classes, self.n_samples)
        
        # Make some predictions correct
        for i in range(50):
            self.predictions[i] = np.zeros(self.n_classes)
            self.predictions[i, self.true_labels[i]] = 1.0
    
    def test_negative_log_likelihood(self):
        """Test NLL computation."""
        nll = compute_nll(self.predictions, self.true_labels)
        
        assert isinstance(nll, float)
        assert nll > 0  # NLL should be positive
        
        # Perfect predictions should have lower NLL
        perfect_preds = np.zeros((self.n_samples, self.n_classes))
        for i in range(self.n_samples):
            perfect_preds[i, self.true_labels[i]] = 1.0
        
        nll_perfect = compute_nll(perfect_preds, self.true_labels)
        assert nll_perfect < nll
    
    def test_brier_score(self):
        """Test Brier score computation."""
        brier = compute_brier_score(self.predictions, self.true_labels)
        
        assert isinstance(brier, float)
        assert 0 <= brier <= 2  # Brier score range
        
        # Perfect predictions should have Brier score of 0
        perfect_preds = np.zeros((self.n_samples, self.n_classes))
        for i in range(self.n_samples):
            perfect_preds[i, self.true_labels[i]] = 1.0
        
        brier_perfect = compute_brier_score(perfect_preds, self.true_labels)
        assert brier_perfect < 0.01  # Should be close to 0
    
    def test_expected_calibration_error(self):
        """Test ECE (Expected Calibration Error) computation."""
        ece = compute_ece(self.predictions, self.true_labels, n_bins=10)
        
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
        
        # Well-calibrated predictions should have low ECE
        # Create calibrated predictions
        calibrated_preds = np.zeros((100, 2))
        true_binary = np.random.binomial(1, 0.7, 100)
        
        for i in range(100):
            if true_binary[i] == 1:
                calibrated_preds[i] = [0.3, 0.7]
            else:
                calibrated_preds[i] = [0.7, 0.3]
        
        ece_calibrated = compute_ece(calibrated_preds, true_binary, n_bins=10)
        assert ece_calibrated < 0.2  # Should be relatively low
    
    def test_model_contribution_analysis(self):
        """Test analysis of individual model contributions."""
        n_models = 3
        model_predictions = []
        
        for i in range(n_models):
            preds = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
            model_predictions.append(preds)
        
        contributions = analyze_model_contributions(
            model_predictions,
            self.true_labels,
            metric='accuracy'
        )
        
        assert len(contributions) == n_models
        assert all(0 <= c <= 1 for c in contributions.values())
    
    def test_shapley_values(self):
        """Test computation of Shapley values for model importance."""
        n_models = 3
        model_predictions = []
        
        for i in range(n_models):
            preds = np.random.dirichlet(np.ones(self.n_classes), self.n_samples)
            model_predictions.append(preds)
        
        # This is computationally expensive, so we'll use a small subset
        shapley_values = compute_shapley_values(
            model_predictions[:3],
            self.true_labels[:20],
            metric='accuracy'
        )
        
        assert len(shapley_values) == 3
        # Shapley values should sum to the difference between
        # grand coalition and empty coalition
        assert abs(sum(shapley_values.values())) <= 1.0


# Helper functions for metrics (these would be in the actual implementation)

def compute_nll(predictions, true_labels):
    """Compute negative log-likelihood."""
    eps = 1e-10
    nll = 0.0
    for i, label in enumerate(true_labels):
        nll -= np.log(predictions[i, label] + eps)
    return nll / len(true_labels)


def compute_brier_score(predictions, true_labels):
    """Compute Brier score."""
    n_samples, n_classes = predictions.shape
    brier = 0.0
    
    for i in range(n_samples):
        true_vec = np.zeros(n_classes)
        true_vec[true_labels[i]] = 1.0
        brier += np.sum((predictions[i] - true_vec) ** 2)
    
    return brier / n_samples


def compute_ece(predictions, true_labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracies = (predicted_labels == true_labels).astype(float)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def analyze_model_contributions(model_predictions, true_labels, metric='accuracy'):
    """Analyze individual model contributions."""
    contributions = {}
    
    for i, preds in enumerate(model_predictions):
        if metric == 'accuracy':
            predicted = np.argmax(preds, axis=1)
            accuracy = (predicted == true_labels).mean()
            contributions[f'model_{i}'] = accuracy
    
    return contributions


def compute_shapley_values(model_predictions, true_labels, metric='accuracy'):
    """Compute Shapley values (simplified version)."""
    from itertools import combinations
    
    n_models = len(model_predictions)
    shapley = {f'model_{i}': 0.0 for i in range(n_models)}
    
    # This is a simplified implementation
    # Real implementation would consider all coalitions
    for i in range(n_models):
        marginal_contributions = []
        
        # Consider different coalition sizes
        for size in range(n_models):
            for coalition in combinations([j for j in range(n_models) if j != i], size):
                coalition_with = list(coalition) + [i]
                coalition_without = list(coalition)
                
                if len(coalition_without) > 0:
                    # Compute metric with and without model i
                    preds_with = np.mean([model_predictions[j] for j in coalition_with], axis=0)
                    preds_without = np.mean([model_predictions[j] for j in coalition_without], axis=0)
                    
                    acc_with = (np.argmax(preds_with, axis=1) == true_labels).mean()
                    acc_without = (np.argmax(preds_without, axis=1) == true_labels).mean()
                    
                    marginal_contributions.append(acc_with - acc_without)
        
        if marginal_contributions:
            shapley[f'model_{i}'] = np.mean(marginal_contributions)
    
    return shapley
