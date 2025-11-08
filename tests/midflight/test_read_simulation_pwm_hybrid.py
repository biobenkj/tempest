"""
Hybrid PWM + Invalid Read Generation Tests
(SequenceSimulator + InvalidReadGenerator)
"""

import pytest
import numpy as np
import yaml
import random
from pathlib import Path
from tempest.data.simulator import SequenceSimulator, SimulatedRead
from tempest.data.invalid_generator import InvalidReadGenerator
from tempest.utils import io


@pytest.fixture
def pwm_file():
    """Use the canonical ACC PWM file bundled with Tempest."""
    import tempest

    root_dir = Path(tempest.__file__).resolve().parent.parent
    pwm_path = root_dir / "whitelist" / "acc_pwm.txt"
    assert pwm_path.exists(), f"Expected PWM file not found: {pwm_path}"

    pwm = io.load_pwm(pwm_path)
    assert pwm.shape[1] == 4  # Columns correspond to A, C, G, T
    # Ensure rows are normalized probabilities
    np.testing.assert_allclose(pwm.sum(axis=1), 1.0, atol=1e-6)
    return pwm_path


@pytest.fixture
def hybrid_config(tmp_path, pwm_file):
    """Hybrid simulation configuration YAML for SequenceSimulator + PWM."""
    cfg = {
        "model": {"max_seq_len": 128},
        "simulation": {
            "num_sequences": 100,
            "train_split": 0.8,
            "random_seed": 77,
            "sequence_order": ["ACC", "CDS"],
            "pwm_files": {"ACC": str(pwm_file)},
            "segment_generation": {"CDS": {"min_len": 20, "max_len": 40}},
        },
        "pwm": {
            "pwm_file": str(pwm_file),
            "use_probabilistic": True,
            "temperature": 1.0,
            "min_entropy": 0.1,
            "random_seed": 77,
        },
    }
    cfg_path = tmp_path / "hybrid_pwm_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg_path


# -------------------------------------------------------------------------
# HYBRID SIMULATION TESTS
# -------------------------------------------------------------------------
class TestHybridSimulation:
    """Integration tests for PWM-based read simulation and invalid generation."""

    def test_hybrid_generation_outputs(self, hybrid_config):
        """Verify that simulator produces valid hybrid reads."""
        cfg = yaml.safe_load(open(hybrid_config))
        sim = SequenceSimulator(cfg)
        reads = sim.generate_batch(n=10)
        assert isinstance(reads, list)
        assert all(isinstance(r, SimulatedRead) for r in reads)
        for r in reads:
            assert set(r.sequence).issubset(set("ACGTN"))
            assert len(r.sequence) > 10
            assert any(seg in r.label_regions for seg in ["ACC", "CDS"])

    def test_pwm_bias_preserved(self, hybrid_config):
        """Verify PWM-derived base frequency bias is preserved."""
        cfg = yaml.safe_load(open(hybrid_config))
        sim = SequenceSimulator(cfg)
        reads = [r.sequence for r in sim.generate_batch(n=50)]
        flat = "".join(reads)
        freqs = {b: flat.count(b) / len(flat) for b in "ACGT"}
        dominant = [b for b, p in freqs.items() if p > 0.2]
        assert any(b in dominant for b in ["A", "C", "G"])

    def test_hybrid_invalid_fraction(self, hybrid_config):
        """Combine SequenceSimulator output with InvalidReadGenerator corruption."""
        cfg = yaml.safe_load(open(hybrid_config))
        sim = SequenceSimulator(cfg)
        valid_reads = sim.generate_batch(n=200)
        invalid_gen = InvalidReadGenerator()
        mixed_reads = invalid_gen.generate_batch(valid_reads, invalid_ratio=0.25)
        num_invalid = sum(
            1 for r in mixed_reads if r.metadata.get("error_type", None) is not None
        )
        frac_invalid = num_invalid / len(mixed_reads)
        assert 0.15 <= frac_invalid <= 0.35

    def test_reproducibility_same_seed(self, hybrid_config):
        """Ensure reproducibility when simulator and PWM share seed."""
        seed = 77
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        cfg = yaml.safe_load(open(hybrid_config))
        sim1 = SequenceSimulator(cfg)

        # Reseed everything before second run
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

        sim2 = SequenceSimulator(cfg)

        reads1 = [r.sequence for r in sim1.generate_batch(n=10)]
        reads2 = [r.sequence for r in sim2.generate_batch(n=10)]
        assert reads1 == reads2
        # Optional bitwise reproducibility check
        assert hash("".join(reads1)) == hash("".join(reads2))

    def test_stochasticity_different_seed(self, hybrid_config):
        """Changing seed should alter outputs."""
        cfg = yaml.safe_load(open(hybrid_config))
        cfg["simulation"]["random_seed"] = 999
        cfg["pwm"]["random_seed"] = 999
        alt_path = Path(hybrid_config).parent / "hybrid_alt.yaml"
        with open(alt_path, "w") as f:
            yaml.safe_dump(cfg, f)

        sim1 = SequenceSimulator(yaml.safe_load(open(hybrid_config)))
        sim2 = SequenceSimulator(yaml.safe_load(open(alt_path)))
        reads1 = [r.sequence for r in sim1.generate_batch(n=10)]
        reads2 = [r.sequence for r in sim2.generate_batch(n=10)]
        assert reads1 != reads2

    def test_reproducibility_regression_seed(self, hybrid_config):
        """Regression test: identical seeds should remain bitwise reproducible."""
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        cfg = yaml.safe_load(open(hybrid_config))
        simA = SequenceSimulator(cfg)

        # Reset all RNGs again
        random.seed(seed)
        np.random.seed(seed)
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

        simB = SequenceSimulator(cfg)
        readsA = [r.sequence for r in simA.generate_batch(n=20)]
        readsB = [r.sequence for r in simB.generate_batch(n=20)]
        fpA, fpB = hash("".join(readsA)), hash("".join(readsB))
        assert fpA == fpB, "Regression: identical seeds produced divergent sequences"


# -------------------------------------------------------------------------
# INVALID READ GENERATOR TESTS
# -------------------------------------------------------------------------
class TestInvalidReadGenerator:
    """Direct tests for InvalidReadGenerator segment-level errors."""

    def test_default_probabilities_normalized(self):
        gen = InvalidReadGenerator()
        probs = gen.error_probabilities
        assert pytest.approx(sum(probs.values()), rel=1e-6) == 1.0

    def test_custom_config_dict(self):
        config = {
            "hybrid": {
                "segment_loss_prob": 0.1,
                "segment_dup_prob": 0.1,
                "truncation_prob": 0.7,
                "chimeric_prob": 0.05,
                "scrambled_prob": 0.05,
            }
        }
        gen = InvalidReadGenerator(config)
        assert gen.error_probabilities["truncation"] > 0.5

    def test_generate_invalid_read_random_choice(self, mocker):
        """Ensure random selection correctly dispatches to segment_loss handler."""
        dummy_read = SimulatedRead(
            sequence="ACGT" * 5,
            labels=["UMI"] * 5 + ["ACC"] * 5 + ["CDS"] * 10,
            label_regions={"UMI": [(0, 5)], "ACC": [(5, 10)], "CDS": [(10, 20)]},
            metadata={},
        )
        gen = InvalidReadGenerator()
        mocker.patch("numpy.random.choice", return_value="segment_loss")
        invalid = gen.generate_invalid_read(dummy_read)
        assert isinstance(invalid, SimulatedRead)
        assert invalid.metadata["error_type"] == "segment_loss"

    def test_apply_truncation_reduces_length(self):
        """Truncation must shorten sequence and update metadata."""
        dummy_read = SimulatedRead(
            sequence="A" * 100,
            labels=["CDS"] * 100,
            label_regions={"CDS": [(0, 100)]},
            metadata={},
        )
        gen = InvalidReadGenerator()
        truncated = gen._apply_truncation(dummy_read)
        assert len(truncated.sequence) < len(dummy_read.sequence)
        assert truncated.metadata["error_type"] == "truncation"

    def test_generate_batch_invalid_ratio(self, mocker):
        """Validate correct invalid ratio and that at least one read is corrupted."""
        dummy_reads = [
            SimulatedRead(sequence="A" * 50, labels=["CDS"] * 50, label_regions={}, metadata={})
            for _ in range(20)
        ]

        def fake_choice(a, *args, **kwargs):
            # Return string only when choosing among error types, otherwise numeric indices
            if isinstance(a, (list, tuple, np.ndarray)) and all(isinstance(x, str) for x in a):
                return "segment_loss"
            return np.random.default_rng().choice(a, *args, **kwargs)

        mocker.patch("numpy.random.choice", side_effect=fake_choice)
        gen = InvalidReadGenerator()
        out_reads = gen.generate_batch(dummy_reads, invalid_ratio=0.25)
        assert len(out_reads) == 20
        n_invalid = sum(r.metadata.get("error_type") == "segment_loss" for r in out_reads)
        assert 0 < n_invalid <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])