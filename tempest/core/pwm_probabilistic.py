"""
PWM module for Tempest with probabilistic base generation.

This version uses PWM scores as probability distributions for base generation
rather than hard thresholds for binary decisions. It now supports
reproducible, per-instance random number generation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ProbabilisticPWMGenerator:
    """
    Position Weight Matrix generator that uses probabilistic sampling
    for generating diverse ACC sequences during read simulation.

    Instead of using a threshold to decide if a sequence is ACC or not,
    this class generates sequences by sampling from the PWM probability
    distributions at each position.
    """

    def __init__(
        self,
        pwm: np.ndarray,
        temperature: float = 1.0,
        min_entropy: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize probabilistic PWM generator.

        Args:
            pwm: PWM matrix of shape (length, 4) for [A, C, G, T]
            temperature: Temperature parameter for controlling diversity
                        (higher = more diverse, lower = more conservative)
            min_entropy: Minimum entropy to maintain at each position
            random_seed: Optional random seed for reproducibility
        """
        self.pwm = pwm
        self.temperature = temperature
        self.min_entropy = min_entropy
        self.motif_length = pwm.shape[0]
        self.rng = np.random.default_rng(random_seed)

        # Prepare temperature-scaled and entropy-adjusted PWM
        self.sampling_pwm = self._prepare_sampling_distribution()

        logger.info(
            f"Initialized probabilistic PWM generator: "
            f"length={self.motif_length}, temperature={temperature}, seed={random_seed}"
        )

    def _prepare_sampling_distribution(self) -> np.ndarray:
        """Prepare PWM for probabilistic sampling with temperature scaling and entropy constraints."""
        if self.temperature != 1.0:
            log_pwm = np.log(self.pwm + 1e-10)
            scaled_log_pwm = log_pwm / self.temperature
            exp_pwm = np.exp(scaled_log_pwm - np.max(scaled_log_pwm, axis=1, keepdims=True))
            sampling_pwm = exp_pwm / exp_pwm.sum(axis=1, keepdims=True)
        else:
            sampling_pwm = self.pwm.copy()

        # Apply minimum entropy constraints
        for pos in range(self.motif_length):
            probs = sampling_pwm[pos]
            entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(4)
            if entropy < self.min_entropy:
                uniform = np.ones(4) * 0.25
                mix_weight = (self.min_entropy - entropy) / (1.0 - entropy)
                mix_weight = np.clip(mix_weight, 0, 0.5)
                sampling_pwm[pos] = (1 - mix_weight) * probs + mix_weight * uniform

        return sampling_pwm

    # ----------------------------------------------------------------------
    # Sequence generation methods
    # ----------------------------------------------------------------------

    def generate_sequences(
        self,
        n: int = 1,
        diversity_boost: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate ACC sequences by probabilistic sampling from PWM.

        Args:
            n: Number of sequences to generate
            diversity_boost: Optional temporary multiplier for temperature
            seed: Optional seed for reproducibility override

        Returns:
            List of generated ACC sequences
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        bases = ["A", "C", "G", "T"]

        sampling_dist = self.sampling_pwm
        if diversity_boost is not None and diversity_boost != 1.0:
            temp_pwm = ProbabilisticPWMGenerator(
                self.pwm,
                temperature=self.temperature * diversity_boost,
                min_entropy=self.min_entropy,
            )
            sampling_dist = temp_pwm.sampling_pwm

        sequences = []
        for _ in range(n):
            seq = [bases[rng.choice(4, p=sampling_dist[pos])] for pos in range(self.motif_length)]
            sequences.append("".join(seq))
        return sequences

    def generate_with_quality_scores(self, n: int = 1) -> List[Tuple[str, np.ndarray]]:
        """
        Generate ACC sequences with associated quality scores based on PWM confidence.

        Args:
            n: Number of sequences to generate

        Returns:
            List of (sequence, quality_scores) tuples
        """
        bases = ["A", "C", "G", "T"]
        sequences_with_scores = []

        for _ in range(n):
            seq = []
            scores = []
            for pos in range(self.motif_length):
                probs = self.sampling_pwm[pos]
                base_idx = self.rng.choice(4, p=probs)
                seq.append(bases[base_idx])
                prob = probs[base_idx]
                quality = int(40 * prob)
                scores.append(quality)
            sequences_with_scores.append(("".join(seq), np.array(scores, dtype=np.int16)))

        return sequences_with_scores

    def generate_with_errors(
        self,
        n: int = 1,
        error_rate: float = 0.01,
        error_profile: Optional[Dict] = None,
    ) -> List[Tuple[str, str]]:
        """
        Generate ACC sequences with simulated sequencing errors.

        Args:
            n: Number of sequences to generate
            error_rate: Base error rate
            error_profile: Optional error probability dictionary

        Returns:
            List of (true_sequence, observed_sequence)
        """
        if error_profile is None:
            error_profile = {"substitution": 0.7, "insertion": 0.15, "deletion": 0.15}

        bases = ["A", "C", "G", "T"]
        sequences_with_errors = []

        for _ in range(n):
            true_seq = self.generate_sequences(1)[0]
            observed = list(true_seq)
            i = 0
            while i < len(observed):
                if self.rng.random() < error_rate:
                    error_type = self.rng.choice(
                        list(error_profile.keys()), p=list(error_profile.values())
                    )
                    if error_type == "substitution":
                        current = observed[i]
                        alternatives = [b for b in bases if b != current]
                        observed[i] = self.rng.choice(alternatives)
                    elif error_type == "insertion":
                        observed.insert(i, self.rng.choice(bases))
                        i += 1
                    elif error_type == "deletion" and len(observed) > 1:
                        del observed[i]
                        i -= 1
                i += 1
            sequences_with_errors.append((true_seq, "".join(observed)))
        return sequences_with_errors

    # ----------------------------------------------------------------------
    # Scoring and diversity analysis
    # ----------------------------------------------------------------------

    def score_sequence_probabilistic(self, sequence: str) -> Dict[str, float]:
        """
        Score a sequence against the PWM probabilistically.

        Args:
            sequence: DNA sequence string
        """
        if len(sequence) != self.motif_length:
            raise ValueError(
                f"Sequence length {len(sequence)} does not match PWM length {self.motif_length}"
            )

        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}
        log_likelihood = 0.0
        position_probs = []
        min_prob = 1.0
        geometric_mean_prob = 1.0

        for pos, base in enumerate(sequence.upper()):
            idx = base_to_idx.get(base, -1)
            if idx >= 0:
                prob = self.pwm[pos, idx]
                position_probs.append(prob)
                log_likelihood += np.log(prob + 1e-10)
                min_prob = min(min_prob, prob)
                geometric_mean_prob *= prob ** (1 / self.motif_length)

        entropy_score = 0.0
        for pos, base in enumerate(sequence.upper()):
            idx = base_to_idx.get(base, -1)
            if idx >= 0:
                pos_entropy = -np.sum(self.pwm[pos] * np.log(self.pwm[pos] + 1e-10))
                max_entropy = -np.log(0.25)
                entropy_weight = pos_entropy / max_entropy
                entropy_score += self.pwm[pos, idx] * entropy_weight
        entropy_score /= len(position_probs) if position_probs else 1

        return {
            "log_likelihood": log_likelihood,
            "mean_probability": np.mean(position_probs) if position_probs else 0,
            "min_probability": min_prob if position_probs else 0,
            "geometric_mean": geometric_mean_prob,
            "entropy_weighted_score": entropy_score,
            "position_probabilities": position_probs,
        }

    def calculate_diversity_metrics(self, sequences: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for a set of sequences."""
        if not sequences:
            return {}

        n_seqs = len(sequences)
        length = len(sequences[0])

        # Position-wise entropy
        position_entropies = []
        for pos in range(length):
            counts = np.zeros(4)
            for seq in sequences:
                base = seq[pos].upper()
                if base in "ACGT":
                    counts["ACGT".index(base)] += 1
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(4)
            position_entropies.append(entropy)

        # Pairwise distances
        distances = [
            sum(a != b for a, b in zip(sequences[i], sequences[j])) / length
            for i in range(n_seqs)
            for j in range(i + 1, n_seqs)
        ]

        unique_seqs = len(set(sequences))
        return {
            "mean_position_entropy": float(np.mean(position_entropies)),
            "min_position_entropy": float(np.min(position_entropies)),
            "max_position_entropy": float(np.max(position_entropies)),
            "mean_pairwise_distance": float(np.mean(distances)) if distances else 0.0,
            "unique_sequences": unique_seqs,
            "uniqueness_ratio": unique_seqs / n_seqs,
        }


# ----------------------------------------------------------------------
# Helper for IUPAC-derived PWM creation
# ----------------------------------------------------------------------

def create_acc_pwm_from_pattern(pattern: str = "ACCSSV", pseudocount: float = 0.1) -> np.ndarray:
    """Create a PWM from an IUPAC pattern like 'ACCSSV'."""
    iupac_probs = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "R": [0.5, 0, 0.5, 0],
        "Y": [0, 0.5, 0, 0.5],
        "S": [0, 0.5, 0.5, 0],
        "W": [0.5, 0, 0, 0.5],
        "K": [0, 0, 0.5, 0.5],
        "M": [0.5, 0.5, 0, 0],
        "B": [0, 1 / 3, 1 / 3, 1 / 3],
        "D": [1 / 3, 0, 1 / 3, 1 / 3],
        "H": [1 / 3, 1 / 3, 0, 1 / 3],
        "V": [1 / 3, 1 / 3, 1 / 3, 0],
        "N": [0.25, 0.25, 0.25, 0.25],
    }

    pwm = np.array([iupac_probs.get(base.upper(), [0.25] * 4) for base in pattern])
    pwm = (pwm + pseudocount) / (1 + 4 * pseudocount)
    return pwm