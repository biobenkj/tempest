"""
Hybrid PWM + Invalid Read Generation Tests

Covers:
- Integration of PWM motif sampling and corruption
- Distributional integrity before and after corruption
- Config-driven hybrid modes (probabilistic + deterministic invalids)
- Reproducibility and stochastic variability
- Edge cases for PWM-corrupted reads
"""

import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tempest.simulator import Simulator
from tempest.invalid_generator import InvalidGenerator
from tempest.config import TempestConfig


@pytest.fixture
def pwm_file(tmp_path):
    """Mock PWM used for hybrid testing."""
    pwm_data = """pos\tbase\tprob
1\tA\t0.9
1\tC\t0.05
1\tG\t0.03
1\tT\t0.02
2\tA\t0.1
2\tC\t0.8
2\tG\t0.05
2\tT\t0.05
3\tA\t0.05
3\tC\t0.1
3\tG\t0.8
3\tT\t0.05
4\tA\t0.05
4\tC\t0.05
4\tG\t0.05
4\tT\t0.85
"""
    pwm_path = tmp_path / "hybrid_pwm.tsv"
    pwm_path.write_text(pwm_data)
    return pwm_path


@pytest.fixture
def hybrid_config(tmp_path, pwm_file):
    """Full TempestConfig YAML combining PWM and invalid generation."""
    cfg = {
        "model": {"max_seq_len": 128},
        "simulation": {
            "num_sequences": 100,
            "train_split": 0.8,
            "random_seed": 77,
            "sequence_order": ["PWM", "CDS"],
            "pwm_files": {"PWM": str(pwm_file)},
            "segment_generation": {
                "CDS": {"min_len": 20, "max_len": 40}
            },
            "pwm": {
                "pwm_file": str(pwm_file),
                "use_probabilistic": True,
                "temperature": 1.0
            },
            "error_injection": {
                "use_hybrid_invalids": True,
                "invalid_fraction": 0.25,
                "mismatch_rate": 0.05,
                "insertion_rate": 0.02,
                "deletion_rate": 0.02,
                "truncation_prob": 0.1
            }
        }
    }
    cfg_path = tmp_path / "hybrid_pwm_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg_path


class TestHybridIntegration:
    """Test combined PWM-driven + Invalid generation."""

    def test_hybrid_generation_shapes(self, hybrid_config):
        """Ensure hybrid generation returns correct output shape."""
        cfg = TempestConfig.from_yaml(hybrid_config)
        sim = Simulator(cfg)
        reads = sim.generate_reads(n=50)
        assert isinstance(reads, list)
        assert all(isinstance(r, str) for r in reads)
        assert all(set(r).issubset(set("ACGTN")) for r in reads)
        assert 10 < np.median([len(r) for r in reads]) < 150

    def test_hybrid_invalid_fraction(self, hybrid_config):
        """Roughly 25% invalids expected per config."""
        cfg = TempestConfig.from_yaml(hybrid_config)
        sim = Simulator(cfg)

        valid_reads, invalid_reads = sim.generate_hybrid_reads(n=200)
        assert len(valid_reads) + len(invalid_reads) == 200
        frac_invalid = len(invalid_reads) / 200
        assert 0.15 <= frac_invalid <= 0.35  # tolerate stochastic variation

    def test_hybrid_pwm_bias_preserved_after_corruption(self, hybrid_config):
        """High PWM-probability bases should remain enriched even after corruption."""
        cfg = TempestConfig.from_yaml(hybrid_config)
        sim = Simulator(cfg)
        valid_reads, invalid_reads = sim.generate_hybrid_reads(n=500)

        def base_freqs(reads):
            flat = "".join(reads)
            counts = np.array([flat.count(b) for b in "ACGT"], dtype=float)
            counts /= counts.sum()
            return dict(zip("ACGT", counts))

        f_valid = base_freqs(valid_reads)
        f_invalid = base_freqs(invalid_reads)

        # PWM is A/C/G heavy; corruption should not invert major composition
        dominant_bases = [b for b, p in f_valid.items() if p > 0.2]
        assert any(b in dominant_bases for b in ["A", "C", "G"])
        # After corruption, dominant bases remain similar
        for b in dominant_bases:
            assert f_invalid[b] > 0.1

    def test_reproducibility_hybrid_mode(self, hybrid_config):
        """Same seed should yield identical hybrid results."""
        cfg = TempestConfig.from_yaml(hybrid_config)
        sim1 = Simulator(cfg)
        sim2 = Simulator(cfg)
        valid1, invalid1 = sim1.generate_hybrid_reads(n=20)
        valid2, invalid2 = sim2.generate_hybrid_reads(n=20)
        assert valid1 == valid2
        assert invalid1 == invalid2

    def test_stochasticity_across_seeds(self, hybrid_config):
        """Different seeds produce statistically different hybrid outcomes."""
        cfg = yaml.safe_load(open(hybrid_config))
        cfg["simulation"]["random_seed"] = 101
        alt_path = Path(hybrid_config).parent / "hybrid_alt.yaml"
        yaml.safe_dump(cfg, open(alt_path, "w"))
        sim1 = Simulator(TempestConfig.from_yaml(hybrid_config))
        sim2 = Simulator(TempestConfig.from_yaml(alt_path))
        valid1, invalid1 = sim1.generate_hybrid_reads(n=100)
        valid2, invalid2 = sim2.generate_hybrid_reads(n=100)
        assert valid1 != valid2 or invalid1 != invalid2


class TestHybridErrorDynamics:
    """Test properties of the InvalidGenerator within hybrid runs."""

    def test_invalid_generator_effect_sizes(self):
        gen = InvalidGenerator(
            mismatch_rate=0.05, insertion_rate=0.02, deletion_rate=0.02
        )
        seq = "ACGT" * 15
        corrupted = [gen.corrupt_sequence(seq) for _ in range(200)]
        # Roughly 5–15% of sequences differ
        diff_ratio = np.mean([s != seq for s in corrupted])
        assert 0.03 < diff_ratio < 0.20

    def test_truncation_probability_behavior(self):
        gen = InvalidGenerator(mismatch_rate=0.0, insertion_rate=0.0,
                               deletion_rate=0.0, truncation_prob=0.5)
        seq = "A" * 50
        truncated = [gen.corrupt_sequence(seq) for _ in range(200)]
        avg_len = np.mean([len(s) for s in truncated])
        assert avg_len < 50  # average shorter than full-length
        assert avg_len > 20  # not catastrophic truncation

    def test_insertion_deletion_balance(self):
        gen = InvalidGenerator(mismatch_rate=0.0,
                               insertion_rate=0.05,
                               deletion_rate=0.05)
        seq = "ACGT" * 10
        corrupted = [gen.corrupt_sequence(seq) for _ in range(100)]
        lens = [len(s) for s in corrupted]
        assert np.std(lens) > 0  # variability introduced
        # Some sequences longer, some shorter
        assert any(l > 40 for l in lens) and any(l < 40 for l in lens)