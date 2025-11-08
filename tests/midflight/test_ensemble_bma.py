"""
Comprehensive tests for Bayesian Model Averaging (BMA) and ensemble modeling.

Updated for the new EnsembleConfig schema in tempest.config.
Tests cover:
- Ensemble creation and configuration
- Model diversity (architecture and initialization)
- Bayesian Model Averaging weight computation
- Ensemble prediction aggregation
- Performance-based weighting
- Reproducibility with fixed seeds
"""

import pytest
import numpy as np
from collections import defaultdict
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tempest.config import EnsembleConfig, BMAConfig


class MockModel:
    """Mock model for testing ensemble functionality."""

    def __init__(self, model_id, accuracy=0.9, seed=None):
        self.model_id = model_id
        self.accuracy = accuracy
        self.seed = seed
        self.prediction_calls = 0

    def predict(self, x):
        """Mock prediction function."""
        self.prediction_calls += 1
        batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
        seq_len = 100
        num_labels = 5
        if self.seed is not None:
            np.random.seed(self.seed + self.model_id)
        return np.random.random((batch_size, seq_len, num_labels))

    def evaluate(self, x, y):
        """Mock evaluation function."""
        return [0.5, self.accuracy]


# ------------------------------------------------------------------------------
# CONFIG TESTS
# ------------------------------------------------------------------------------

class TestEnsembleConfig:
    """Test EnsembleConfig and BMAConfig structure and defaults."""

    def test_default_ensemble_config(self):
        config = EnsembleConfig()
        assert config.enabled is True
        assert config.num_models == 3
        assert config.voting_method == "bayesian_model_averaging"
        assert config.vary_architecture is True
        assert config.vary_initialization is True
        assert isinstance(config.bma_config, (BMAConfig, type(None)))

        # Default BMA prior_type if present
        if config.bma_config:
            assert config.bma_config.prior_type == "uniform"

    def test_custom_ensemble_config(self):
        """Ensure custom initialization works with nested BMAConfig."""
        bma = BMAConfig(prior_type="performance", approximation="laplace")
        config = EnsembleConfig(
            enabled=True,
            num_models=10,
            vary_architecture=False,
            vary_initialization=True,
            voting_method="voting",
            bma_config=bma
        )

        assert config.voting_method == "voting"
        assert config.num_models == 10
        assert config.vary_architecture is False
        assert config.vary_initialization is True
        assert config.bma_config.prior_type == "performance"
        assert config.bma_config.approximation == "laplace"


# ------------------------------------------------------------------------------
# ENSEMBLE BEHAVIOR TESTS
# ------------------------------------------------------------------------------

class TestEnsembleCreation:
    """Test ensemble creation and model management."""

    def test_create_ensemble_with_models(self):
        num_models = 5
        models = [MockModel(i, accuracy=0.85 + i * 0.01) for i in range(num_models)]
        model_ids = [m.model_id for m in models]
        assert len(set(model_ids)) == num_models
        accuracies = [m.accuracy for m in models]
        assert len(set(accuracies)) == num_models

    def test_model_diversity_through_seeds(self):
        model1 = MockModel(0, seed=42)
        model2 = MockModel(1, seed=43)
        x = np.random.random((10, 100))
        pred1 = model1.predict(x)
        pred2 = model2.predict(x)
        assert not np.allclose(pred1, pred2)


# ------------------------------------------------------------------------------
# BAYESIAN MODEL AVERAGING TESTS
# ------------------------------------------------------------------------------

class TestBayesianModelAveraging:
    """Test Bayesian Model Averaging functionality."""

    def compute_uniform_weights(self, num_models):
        return np.ones(num_models) / num_models

    def compute_performance_weights(self, accuracies):
        exp_acc = np.exp(np.array(accuracies) * 10)
        return exp_acc / np.sum(exp_acc)

    def test_uniform_prior_weights(self):
        weights = self.compute_uniform_weights(5)
        assert np.allclose(weights, 1.0 / 5)
        assert np.isclose(np.sum(weights), 1.0)

    def test_performance_based_weights(self):
        accuracies = [0.85, 0.90, 0.88, 0.92, 0.87]
        weights = self.compute_performance_weights(accuracies)
        assert np.isclose(np.sum(weights), 1.0)
        assert weights[np.argmax(accuracies)] == np.max(weights)
        assert weights[np.argmin(accuracies)] == np.min(weights)

    def test_weight_computation_stability(self):
        accuracies = [0.50, 0.99, 0.51, 0.98, 0.52]
        weights = self.compute_performance_weights(accuracies)
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        assert np.isclose(np.sum(weights), 1.0)


# ------------------------------------------------------------------------------
# ENSEMBLE PREDICTION TESTS
# ------------------------------------------------------------------------------

class TestEnsemblePrediction:
    """Test ensemble prediction aggregation."""

    def aggregate_bma(self, preds, weights):
        weighted = np.array([p * w for p, w in zip(preds, weights)])
        return np.sum(weighted, axis=0)

    def aggregate_voting(self, preds):
        classes = [np.argmax(p, axis=-1) for p in preds]
        stacked = np.stack(classes, axis=0)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked)

    def test_bma_aggregation(self):
        np.random.seed(42)
        preds = [np.random.random((4, 100, 5)) for _ in range(3)]
        weights = np.array([1/3, 1/3, 1/3])
        agg = self.aggregate_bma(preds, weights)
        assert agg.shape == (4, 100, 5)
        assert np.all((agg >= 0) & (agg <= 1))

    def test_voting_aggregation(self):
        pred1 = np.zeros((2, 10, 3))
        pred2 = np.zeros((2, 10, 3))
        pred3 = np.zeros((2, 10, 3))
        pred1[:, :, 0] = 0.9
        pred2[:, :, 0] = 0.9
        pred3[:, :, 1] = 0.9
        voted = self.aggregate_voting([pred1, pred2, pred3])
        assert np.all(voted == 0)


# ------------------------------------------------------------------------------
# EVALUATION + DIVERSITY + REPRODUCIBILITY
# ------------------------------------------------------------------------------

class TestEvaluationAndDiversity:
    """Evaluate ensemble models and diversity."""

    def test_evaluation_tracking(self):
        models = [MockModel(i, accuracy=0.85 + i * 0.01) for i in range(5)]
        perf_log = defaultdict(list)
        for epoch in range(3):
            for i, model in enumerate(models):
                _, acc = model.evaluate(np.random.random((10, 100)), np.random.randint(0, 5, (10, 100)))
                perf_log[f"model_{i}"].append(acc)
        assert len(perf_log) == 5
        for k, v in perf_log.items():
            assert len(v) == 3

    def test_architecture_diversity(self):
        archs = [
            {"lstm_units": 64, "cnn_filters": [32, 64]},
            {"lstm_units": 128, "cnn_filters": [64, 128]},
            {"lstm_units": 256, "cnn_filters": [128, 256]},
        ]
        assert len(archs) == len(set(str(a) for a in archs))

    def test_initialization_diversity(self):
        models = [MockModel(i, seed=s) for i, s in enumerate([42, 43, 44, 45, 46])]
        preds = [m.predict(np.random.random((5, 100))) for m in models]
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                assert not np.allclose(preds[i], preds[j])


# ------------------------------------------------------------------------------
# INTEGRATION TEST
# ------------------------------------------------------------------------------

class TestIntegration:
    """Integration test for complete BMA workflow."""

    def test_bma_end_to_end(self):
        num_models = 5
        models = [MockModel(i, accuracy=0.85 + i * 0.01) for i in range(num_models)]
        accuracies = [m.evaluate(np.random.random((20, 100)), np.random.randint(0, 5, (20, 100)))[1] for m in models]
        exp_acc = np.exp(np.array(accuracies) * 10)
        weights = exp_acc / np.sum(exp_acc)
        x_test = np.random.random((10, 100))
        preds = [m.predict(x_test) for m in models]
        ensemble_pred = np.sum([p * w for p, w in zip(preds, weights)], axis=0)
        assert ensemble_pred.shape == (10, 100, 5)
        assert np.isclose(np.sum(weights), 1.0)
        assert all(m.prediction_calls == 1 for m in models)