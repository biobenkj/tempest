"""
Comprehensive tests for Bayesian Model Averaging (BMA) and ensemble modeling.

Updated for new tempest.config and tempest.training.ensemble APIs.

Tests cover:
- Ensemble configuration (EnsembleConfig, BMAConfig)
- Model diversity and evidence computation
- Bayesian Model Averaging weight computation (uniform, informative, adaptive)
- Evidence approximation (BIC, Laplace, variational, cross-validation)
- Posterior computation and temperature scaling
- Calibration and uncertainty quantification
- Model selection and aggregation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from scipy import stats
import sys
from pathlib import Path

# Add tempest package root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import new config and ensemble classes
from tempest.config import EnsembleConfig, BMAConfig
from tempest.training.ensemble import (
    EnsembleModel,
    BayesianModelAveraging,
    compute_bic,
    compute_laplace_evidence,
    compute_variational_evidence,
    compute_cv_evidence,
)

# ------------------------------------------------------------------------------
# CONFIGURATION TESTS
# ------------------------------------------------------------------------------

class TestConfig:
    """Test EnsembleConfig and BMAConfig initialization and conversion."""

    def test_default_ensemble_config(self):
        cfg = EnsembleConfig()
        assert cfg.enabled is True
        assert cfg.num_models == 3
        assert cfg.voting_method == "bayesian_model_averaging"
        assert isinstance(cfg.bma_config, (BMAConfig, type(None)))

    def test_custom_config_with_bma(self):
        cfg = EnsembleConfig(
            enabled=True,
            num_models=5,
            voting_method="bayesian_model_averaging",
            bma_config=BMAConfig(
                enabled=True,
                prior_type="informative",
                approximation="laplace",
                temperature=1.5,
            ),
        )
        assert cfg.enabled is True
        assert cfg.num_models == 5
        assert cfg.voting_method == "bayesian_model_averaging"
        assert cfg.bma_config.enabled
        assert cfg.bma_config.prior_type == "informative"
        assert cfg.bma_config.approximation == "laplace"
        assert cfg.bma_config.temperature == 1.5

    def test_from_dict_initialization(self):
        config_dict = {
            "enabled": True,
            "num_models": 4,
            "voting_method": "bayesian_model_averaging",
            "bma_config": {
                "enabled": True,
                "prior_type": "adaptive",
                "approximation": "bic",
                "temperature": 0.8,
            },
        }
        cfg = EnsembleConfig.from_dict(config_dict)
        assert cfg.bma_config.prior_type == "adaptive"
        assert cfg.bma_config.approximation == "bic"
        assert cfg.bma_config.temperature == 0.8


# ------------------------------------------------------------------------------
# BMA CONFIGURATION TESTS
# ------------------------------------------------------------------------------

class TestBMAConfig:
    """Test Bayesian Model Averaging configuration behavior."""

    def test_prior_types(self):
        cfg_uniform = BMAConfig(enabled=True, prior_type="uniform")
        cfg_informative = BMAConfig(
            enabled=True,
            prior_type="informative",
            prior_weights={"a": 0.4, "b": 0.3, "c": 0.3},
        )
        cfg_adaptive = BMAConfig(enabled=True, prior_type="adaptive")
        assert cfg_uniform.prior_type == "uniform"
        assert abs(sum(cfg_informative.prior_weights.values()) - 1.0) < 1e-6
        assert cfg_adaptive.prior_type == "adaptive"

    def test_approximation_methods(self):
        for method in ["bic", "laplace", "variational", "cross_validation"]:
            cfg = BMAConfig(enabled=True, approximation=method)
            assert cfg.approximation == method

    def test_temperature_range(self):
        cfg_hot = BMAConfig(temperature=2.0)
        cfg_cold = BMAConfig(temperature=0.5)
        assert cfg_hot.temperature > 1
        assert cfg_cold.temperature < 1


# ------------------------------------------------------------------------------
# BAYESIAN MODEL AVERAGING TESTS
# ------------------------------------------------------------------------------

class TestBayesianModelAveraging:
    """Test Bayesian Model Averaging behavior."""

    def setup_method(self):
        self.models = []
        for i in range(3):
            m = Mock()
            m.name = f"model_{i}"
            m.predict = Mock(return_value=np.random.randn(10, 5))
            m.evaluate = Mock(return_value={"loss": 0.1 * (i + 1)})
            self.models.append(m)

    def test_uniform_prior_weights(self):
        bma = BayesianModelAveraging(self.models, prior_type="uniform")
        priors = bma.compute_prior_weights()
        vals = np.array(list(priors.values()))
        assert np.allclose(vals, 1 / len(self.models))
        assert np.isclose(vals.sum(), 1.0)

    def test_informative_prior_weights(self):
        weights = {"model_0": 0.5, "model_1": 0.3, "model_2": 0.2}
        bma = BayesianModelAveraging(self.models, prior_type="informative", prior_weights=weights)
        priors = bma.compute_prior_weights()
        for k in weights:
            assert np.isclose(priors[k], weights[k])

    def test_adaptive_prior_weights(self):
        perf = {"model_0": 0.9, "model_1": 0.8, "model_2": 0.7}
        bma = BayesianModelAveraging(self.models, prior_type="adaptive")
        bma.validation_performances = perf
        priors = bma.compute_adaptive_prior_weights()
        assert priors["model_0"] > priors["model_1"] > priors["model_2"]
        assert np.isclose(sum(priors.values()), 1.0)


# ------------------------------------------------------------------------------
# EVIDENCE APPROXIMATION TESTS
# ------------------------------------------------------------------------------

class TestEvidenceApproximation:
    """Test BIC, Laplace, variational, and CV evidence functions."""

    def setup_method(self):
        self.n = 100
        self.p = 50
        self.log_like = -40.0
        self.model = Mock()
        self.model.count_params = Mock(return_value=self.p)
        self.model.evaluate = Mock(return_value={"loss": 0.5})
        self.X = np.random.randn(self.n, 10)
        self.y = np.random.randint(0, 5, self.n)

    def test_bic(self):
        bic = compute_bic(log_likelihood=self.log_like, n_params=self.p, n_samples=self.n)
        expected = -2 * self.log_like + self.p * np.log(self.n)
        assert np.isclose(bic, expected)

    def test_laplace(self):
        with patch("tempest.training.ensemble.compute_hessian") as hess:
            hess.return_value = np.eye(self.p) * 0.1
            evidence = compute_laplace_evidence(self.model, (self.X, self.y), num_samples=100)
        assert isinstance(evidence, float)
        assert not np.isnan(evidence)

    def test_variational(self):
        evidence = compute_variational_evidence(
            self.model, (self.X, self.y), num_iterations=10, learning_rate=0.01
        )
        assert isinstance(evidence, float)

    def test_cross_validation(self):
        evidence = compute_cv_evidence(self.model, (self.X, self.y), num_folds=3)
        assert isinstance(evidence, float)
        assert evidence <= 0


# ------------------------------------------------------------------------------
# POSTERIOR COMPUTATION TESTS
# ------------------------------------------------------------------------------

class TestPosteriorComputation:
    """Test posterior weight computation and temperature effects."""

    def setup_method(self):
        self.priors = {"model_0": 0.4, "model_1": 0.3, "model_2": 0.3}
        self.evidences = {"model_0": -100, "model_1": -120, "model_2": -150}
        self.bma = BayesianModelAveraging([Mock(name=f"m{i}") for i in range(3)])

    def test_standard_posterior(self):
        post = self.bma.compute_posterior_weights(
            prior_weights=self.priors, evidences=self.evidences, temperature=1.0
        )
        assert np.isclose(sum(post.values()), 1.0)
        assert post["model_0"] > post["model_1"] > post["model_2"]

    def test_temperature_effect(self):
        p_hot = self.bma.compute_posterior_weights(
            self.priors, self.evidences, temperature=2.0
        )
        p_cold = self.bma.compute_posterior_weights(
            self.priors, self.evidences, temperature=0.5
        )
        var_hot = np.var(list(p_hot.values()))
        var_cold = np.var(list(p_cold.values()))
        assert var_cold > var_hot  # colder = more peaked posterior


# ------------------------------------------------------------------------------
# UNCERTAINTY + DIVERSITY TESTS
# ------------------------------------------------------------------------------

class TestUncertaintyAndDiversity:
    """Test epistemic/aleatoric uncertainty and model diversity."""

    def setup_method(self):
        self.n_models = 4
        self.n_samples = 100
        self.n_classes = 3
        np.random.seed(42)
        self.preds = [
            np.random.dirichlet(np.ones(self.n_classes) * (i + 1), self.n_samples)
            for i in range(self.n_models)
        ]

    def test_epistemic_uncertainty(self):
        bma = BayesianModelAveraging(
            models=[Mock(name=f"m{i}") for i in range(self.n_models)], compute_epistemic=True
        )
        u = bma.compute_epistemic_uncertainty(self.preds)
        assert u.shape == (self.n_samples,)
        assert np.all(u >= 0)

    def test_aleatoric_uncertainty(self):
        bma = BayesianModelAveraging(
            models=[Mock(name=f"m{i}") for i in range(self.n_models)], compute_aleatoric=True
        )
        u = bma.compute_aleatoric_uncertainty(self.preds)
        assert u.shape == (self.n_samples,)
        assert np.all(u >= 0)


# ------------------------------------------------------------------------------
# MODEL SELECTION + AGGREGATION
# ------------------------------------------------------------------------------

class TestModelSelectionAndAggregation:
    """Test model selection criteria and aggregation modes."""

    def setup_method(self):
        self.models = [Mock(name=f"m{i}") for i in range(4)]
        self.bma = BayesianModelAveraging(self.models)
        self.evidences = {"m0": -100, "m1": -102, "m2": -120, "m3": -180}

    def test_select_by_evidence(self):
        selected = self.bma.select_models_by_evidence(self.evidences, evidence_threshold=0.05)
        assert "m0" in selected
        assert "m3" not in selected

    def test_select_best_model(self):
        post = {"m0": 0.6, "m1": 0.3, "m2": 0.1, "m3": 0.0}
        best = self.bma.select_best_model(post)
        assert best == "m0"

    def test_aggregate_predictions(self):
        np.random.seed(42)
        preds = [np.random.dirichlet(np.ones(4), 20) for _ in self.models]
        weights = [0.4, 0.3, 0.2, 0.1]
        agg = self.bma.aggregate_predictions(preds, weights, method="probability_averaging")
        assert agg.shape == (20, 4)
        assert np.allclose(agg.sum(axis=1), 1.0)