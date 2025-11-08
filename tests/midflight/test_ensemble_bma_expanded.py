"""
Expanded tests for Bayesian Model Averaging (BMA) and ensemble modeling.

This suite focuses on the current Tempest implementation:

- EnsembleConfig / BMAConfig (from tempest.config)
- BMA-related behavior inside EnsembleTrainer (weight computation)
- BMA-based prediction aggregation via BMAPredictor
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Make sure the repo root is on sys.path so `import tempest` works
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tempest.config import EnsembleConfig, BMAConfig
from tempest.training.ensemble import EnsembleTrainer, BMAPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trainer_stub(num_models, prior="uniform", temperature=1.0, accuracies=None):
    """
    Create an EnsembleTrainer-like stub that we can use to call _compute_bma_weights
    without invoking the full training pipeline.

    We intentionally bypass __init__ because it wires up a lot of training details
    that aren't needed for these unit tests.
    """
    stub = EnsembleTrainer.__new__(EnsembleTrainer)  # type: ignore[misc]

    stub.num_models = num_models
    stub.bma_prior = prior
    stub.bma_temperature = temperature

    # Model performances, as expected by _compute_bma_weights:
    # a list of dicts, each with a 'val_accuracy' float.
    if accuracies is None:
        stub.model_performances = []
    else:
        stub.model_performances = [
            {"val_accuracy": float(a)} for a in accuracies
        ]

    # Placeholder attributes that _compute_bma_weights will set
    stub.model_weights = None

    return stub


class DummyModel:
    """
    Very simple stand-in for a Keras model used by BMAPredictor.

    It ignores the input X content and just returns a constant
    [batch, seq_len, num_labels] tensor derived from the provided logits.
    """

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype="float32")

    def predict(self, X, **kwargs):
        X = np.asarray(X)
        batch_size, seq_len = X.shape
        num_labels = self.logits.shape[-1]
        # Broadcast logits to every position
        return np.tile(self.logits, (batch_size, seq_len, 1))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    """Tests for EnsembleConfig and BMAConfig wiring."""

    def test_default_bma_config(self):
        cfg = BMAConfig()

        assert cfg.enabled is True
        assert cfg.prior_type == "uniform"
        assert cfg.approximation == "bic"
        assert cfg.temperature == 1.0
        assert cfg.compute_posterior_variance is True
        assert cfg.normalize_posteriors is True
        assert cfg.min_posterior_weight == pytest.approx(0.01)

    def test_bma_config_from_dict_with_approximation_params(self):
        cfg = BMAConfig.from_dict({
            "enabled": True,
            "prior_type": "informative",
            "approximation": "laplace",
            "approximation_params": {
                "bic": {"penalty_factor": 2.0},
                "laplace": {"num_samples": 2000, "damping": 0.05},
                "variational": {
                    "num_iterations": 50,
                    "learning_rate": 0.005,
                    "convergence_threshold": 1e-5,
                },
                "cross_validation": {"num_folds": 7, "stratified": False},
            },
        })

        # Add fields
        assert cfg.enabled is True
        assert cfg.prior_type == "informative"
        assert cfg.approximation == "laplace"

        # Convenience fields extracted from approximation_params
        assert cfg.bic_penalty_factor == pytest.approx(2.0)
        assert cfg.laplace_num_samples == 2000
        assert cfg.laplace_damping == pytest.approx(0.05)

        assert cfg.vi_num_iterations == 50
        assert cfg.vi_learning_rate == pytest.approx(0.005)
        assert cfg.vi_convergence_threshold == pytest.approx(1e-5)

        assert cfg.cv_num_folds == 7
        assert cfg.cv_stratified is False

    def test_default_ensemble_config(self):
        cfg = EnsembleConfig()

        # General defaults
        assert cfg.enabled is True
        assert cfg.num_models == 3
        assert cfg.voting_method == "bayesian_model_averaging"

        # Diversity / variation defaults
        assert cfg.vary_architecture is True
        assert cfg.vary_initialization is True
        assert cfg.vary_training is False

        # BMA config is optional by default
        assert cfg.bma_config is None

    def test_ensemble_config_from_dict_with_nested_bma_config(self):
        cfg_dict = {
            "enabled": True,
            "num_models": 5,
            "voting_method": "bayesian_model_averaging",
            "vary_architecture": False,
            "vary_initialization": True,
            "bma_config": {
                "enabled": True,
                "prior_type": "adaptive",
                "approximation": "bic",
                "temperature": 1.5,
            },
        }

        cfg = EnsembleConfig.from_dict(cfg_dict)

        assert cfg.enabled is True
        assert cfg.num_models == 5
        assert cfg.voting_method == "bayesian_model_averaging"
        assert cfg.vary_architecture is False
        assert cfg.vary_initialization is True

        # Nested BMAConfig converted correctly
        assert isinstance(cfg.bma_config, BMAConfig)
        assert cfg.bma_config.enabled is True
        assert cfg.bma_config.prior_type == "adaptive"
        assert cfg.bma_config.approximation == "bic"
        assert cfg.bma_config.temperature == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# EnsembleTrainer BMA weight computation
# ---------------------------------------------------------------------------

class TestBMAWeightComputation:
    """Tests for the internal _compute_bma_weights in EnsembleTrainer."""

    def test_uniform_prior_produces_equal_weights(self):
        num_models = 4
        trainer = _build_trainer_stub(
            num_models=num_models,
            prior="uniform",
            temperature=1.0,
            accuracies=None,  # not used for uniform
        )

        trainer._compute_bma_weights()

        assert trainer.model_weights is not None
        weights = np.array(trainer.model_weights, dtype="float32")

        # All equal, sum to 1
        assert weights.shape == (num_models,)
        assert np.allclose(weights, 1.0 / num_models)
        assert np.isclose(weights.sum(), 1.0)

    def test_performance_prior_uses_softmax_of_accuracies(self):
        accuracies = [0.80, 0.85, 0.90]
        trainer = _build_trainer_stub(
            num_models=len(accuracies),
            prior="performance",
            temperature=1.0,
            accuracies=accuracies,
        )

        trainer._compute_bma_weights()
        weights = np.array(trainer.model_weights, dtype="float32")

        # Softmax over accuracies with temperature=1
        expected = np.exp(accuracies) / np.exp(accuracies).sum()

        assert weights.shape == expected.shape
        assert np.allclose(weights.sum(), 1.0, atol=1e-6)

        # Same order, and best accuracy gets largest weight
        best_idx = int(np.argmax(accuracies))
        assert np.argmax(weights) == best_idx

        # Roughly softmax-like (not necessarily exact if implementation changes slightly)
        assert weights[best_idx] == pytest.approx(expected[best_idx], rel=1e-2)

    def test_temperature_controls_peakedness(self):
        accuracies = [0.80, 0.85, 0.90]

        # Lower temperature => more peaked distribution
        cold_trainer = _build_trainer_stub(
            num_models=len(accuracies),
            prior="performance",
            temperature=0.5,
            accuracies=accuracies,
        )
        cold_trainer._compute_bma_weights()
        cold_weights = np.array(cold_trainer.model_weights, dtype="float32")

        # Higher temperature => flatter distribution
        hot_trainer = _build_trainer_stub(
            num_models=len(accuracies),
            prior="performance",
            temperature=2.0,
            accuracies=accuracies,
        )
        hot_trainer._compute_bma_weights()
        hot_weights = np.array(hot_trainer.model_weights, dtype="float32")

        # Both should sum to 1
        assert np.isclose(cold_weights.sum(), 1.0)
        assert np.isclose(hot_weights.sum(), 1.0)

        # "Cold" (low temp) distribution is more concentrated on the best model
        best_idx = int(np.argmax(accuracies))
        assert cold_weights[best_idx] > hot_weights[best_idx]


# ---------------------------------------------------------------------------
# BMAPredictor tests (post-training ensemble behavior)
# ---------------------------------------------------------------------------

class TestBMAPredictor:
    """Tests for BMA-based prediction aggregation via BMAPredictor."""

    def _write_metadata(
        self,
        tmp_path,
        num_models,
        model_weights=None,
        label_to_idx=None,
        idx_to_label=None,
    ):
        if model_weights is None:
            model_weights = [1.0 / num_models] * num_models

        if label_to_idx is None:
            label_to_idx = {"L0": 0, "L1": 1}

        if idx_to_label is None:
            idx_to_label = {str(v): k for k, v in label_to_idx.items()}

        meta = {
            "num_models": num_models,
            "model_weights": model_weights,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
        }

        meta_path = tmp_path / "ensemble_metadata.json"
        meta_path.write_text(json.dumps(meta))

        # Create empty .h5 placeholders (paths are still constructed in BMAPredictor)
        for i in range(num_models):
            (tmp_path / f"ensemble_model_{i}.h5").write_text("")

    @patch("tempest.training.ensemble.keras.models.load_model")
    def test_bma_predictor_respects_model_weights(self, mock_load_model, tmp_path):
        """
        Two dummy models, one strongly favoring label 0, the other label 1.
        We assign a larger BMA weight to the second model and check that
        the final predictions consistently pick label 1.
        """
        num_labels = 2

        # Model 0: strong logits for label 0
        logits0 = np.array([5.0, 0.0], dtype="float32")
        # Model 1: strong logits for label 1
        logits1 = np.array([0.0, 5.0], dtype="float32")

        # Heavier weight on model 1
        weights = [0.2, 0.8]

        self._write_metadata(
            tmp_path,
            num_models=2,
            model_weights=weights,
            label_to_idx={"L0": 0, "L1": 1},
            idx_to_label={"0": "L0", "1": "L1"},
        )

        # load_model will be called twice, once per ensemble member
        mock_load_model.side_effect = [
            DummyModel(logits0),
            DummyModel(logits1),
        ]

        predictor = BMAPredictor(tmp_path)

        sequences = ["ACGTACGT", "TTTTAAAA"]
        labels, scores = predictor.predict(sequences)

        # Shape checks
        assert len(labels) == len(sequences)
        assert scores.shape == (len(sequences), max(len(s) for s in sequences))

        # Because model 1 has higher weight, we expect all positions to be L1
        for seq_labels in labels:
            assert len(seq_labels) == len(sequences[labels.index(seq_labels)])
            assert set(seq_labels) == {"L1"}

        # Confidence scores should be positive floats
        assert np.all(scores > 0.0)

    @patch("tempest.training.ensemble.keras.models.load_model")
    def test_bma_predictor_uniform_weights_default(self, mock_load_model, tmp_path):
        """
        If metadata omits model_weights, BMAPredictor should fall back to
        uniform weights across models.
        """
        # Create metadata without explicit model_weights
        label_to_idx = {"L0": 0, "L1": 1}
        idx_to_label = {"0": "L0", "1": "L1"}

        meta = {
            "num_models": 2,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
        }
        (tmp_path / "ensemble_metadata.json").write_text(json.dumps(meta))
        for i in range(2):
            (tmp_path / f"ensemble_model_{i}.h5").write_text("")

        # One model favors L0, the other L1
        logits0 = np.array([5.0, 0.0], dtype="float32")
        logits1 = np.array([0.0, 5.0], dtype="float32")
        mock_load_model.side_effect = [
            DummyModel(logits0),
            DummyModel(logits1),
        ]

        predictor = BMAPredictor(tmp_path)
        sequences = ["ACGT"]
        labels, scores = predictor.predict(sequences)

        # With equal weights and symmetric logits, we expect a tie;
        # np.argmax breaks ties by choosing the first index (L0).
        assert len(labels) == 1
        assert labels[0] == ["L0"] * len(sequences[0])

        assert scores.shape == (1, len(sequences[0]))
        assert np.all(scores > 0.0)

    @patch("tempest.training.ensemble.keras.models.load_model")
    def test_bma_predictor_handles_n_bases(self, mock_load_model, tmp_path):
        """
        Ensure that unknown bases (e.g. 'N') are still handled without error
        and that predictions are returned with the correct sequence lengths.
        """
        logits = np.array([1.0, 2.0, 3.0], dtype="float32")
        label_to_idx = {"SEG0": 0, "SEG1": 1, "SEG2": 2}
        idx_to_label = {"0": "SEG0", "1": "SEG1", "2": "SEG2"}

        self._write_metadata(
            tmp_path,
            num_models=1,
            model_weights=[1.0],
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
        )

        mock_load_model.return_value = DummyModel(logits)

        predictor = BMAPredictor(tmp_path)

        sequences = ["NNNNN", "ACNGT"]
        labels, scores = predictor.predict(sequences)

        assert len(labels) == len(sequences)
        assert scores.shape == (len(sequences), max(len(s) for s in sequences))

        for i, seq in enumerate(sequences):
            assert len(labels[i]) == len(seq)
            # All positions will take the argmax of the logits -> SEG2
            assert set(labels[i]) == {"SEG2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])