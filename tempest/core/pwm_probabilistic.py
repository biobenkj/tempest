"""
PWM module for Tempest with probabilistic base generation.

This version uses PWM scores as probability distributions for base generation
rather than hard thresholds for binary decisions.
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
    
    def __init__(self, pwm: np.ndarray, 
                 temperature: float = 1.0,
                 min_entropy: float = 0.1):
        """
        Initialize probabilistic PWM generator.
        
        Args:
            pwm: PWM matrix of shape (length, 4) for [A, C, G, T]
            temperature: Temperature parameter for controlling diversity
                        (higher = more diverse, lower = more conservative)
            min_entropy: Minimum entropy to maintain at each position
        """
        self.pwm = pwm
        self.temperature = temperature
        self.min_entropy = min_entropy
        self.motif_length = pwm.shape[0]
        
        # Apply temperature scaling and entropy constraints
        self.sampling_pwm = self._prepare_sampling_distribution()
        
        logger.info(f"Initialized probabilistic PWM generator: "
                   f"length={self.motif_length}, temperature={temperature}")
    
    def _prepare_sampling_distribution(self) -> np.ndarray:
        """
        Prepare PWM for probabilistic sampling with temperature scaling
        and entropy constraints.
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            # Convert to log space, apply temperature, convert back
            log_pwm = np.log(self.pwm + 1e-10)
            scaled_log_pwm = log_pwm / self.temperature
            
            # Normalize back to probabilities
            exp_pwm = np.exp(scaled_log_pwm - np.max(scaled_log_pwm, axis=1, keepdims=True))
            sampling_pwm = exp_pwm / exp_pwm.sum(axis=1, keepdims=True)
        else:
            sampling_pwm = self.pwm.copy()
        
        # Apply minimum entropy constraint
        for pos in range(self.motif_length):
            # Calculate current entropy
            probs = sampling_pwm[pos]
            entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(4)  # Normalized
            
            if entropy < self.min_entropy:
                # Add uniform noise to increase entropy
                uniform = np.ones(4) * 0.25
                mix_weight = (self.min_entropy - entropy) / (1.0 - entropy)
                mix_weight = np.clip(mix_weight, 0, 0.5)  # Don't mix more than 50%
                
                sampling_pwm[pos] = (1 - mix_weight) * probs + mix_weight * uniform
        
        return sampling_pwm
    
    def generate_sequences(self, n: int = 1, 
                          diversity_boost: Optional[float] = None,
                          seed: Optional[int] = None) -> List[str]:
        """
        Generate ACC sequences by probabilistic sampling from PWM.
        
        Args:
            n: Number of sequences to generate
            diversity_boost: Optional temporary boost to diversity (multiplies temperature)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated ACC sequences with natural diversity
        """
        if seed is not None:
            np.random.seed(seed)
        
        bases = ['A', 'C', 'G', 'T']
        sequences = []
        
        # Apply diversity boost if specified
        sampling_dist = self.sampling_pwm
        if diversity_boost is not None and diversity_boost != 1.0:
            temp_pwm = ProbabilisticPWMGenerator(
                self.pwm, 
                temperature=self.temperature * diversity_boost,
                min_entropy=self.min_entropy
            )
            sampling_dist = temp_pwm.sampling_pwm
        
        for _ in range(n):
            seq = []
            for pos in range(self.motif_length):
                # Sample base according to PWM probabilities
                base_idx = np.random.choice(4, p=sampling_dist[pos])
                seq.append(bases[base_idx])
            sequences.append(''.join(seq))
        
        return sequences
    
    def generate_with_quality_scores(self, n: int = 1) -> List[Tuple[str, np.ndarray]]:
        """
        Generate ACC sequences with associated quality scores based on 
        PWM confidence.
        
        Args:
            n: Number of sequences to generate
            
        Returns:
            List of (sequence, quality_scores) tuples where quality scores
            reflect the PWM probability of each base
        """
        sequences_with_scores = []
        bases = ['A', 'C', 'G', 'T']
        
        for _ in range(n):
            seq = []
            scores = []
            
            for pos in range(self.motif_length):
                # Sample base
                probs = self.sampling_pwm[pos]
                base_idx = np.random.choice(4, p=probs)
                seq.append(bases[base_idx])
                
                # Quality score is based on the probability of the chosen base
                # Convert to Phred-like score (higher = more confident)
                prob = probs[base_idx]
                # Map probability [0,1] to quality score [0,40]
                quality = int(40 * prob)
                scores.append(quality)
            
            sequences_with_scores.append((''.join(seq), np.array(scores)))
        
        return sequences_with_scores
    
    def generate_with_errors(self, n: int = 1, 
                            error_rate: float = 0.01,
                            error_profile: Optional[Dict] = None) -> List[Tuple[str, str]]:
        """
        Generate ACC sequences with realistic sequencing errors.
        
        Args:
            n: Number of sequences to generate
            error_rate: Base error rate
            error_profile: Optional dictionary with error type probabilities
            
        Returns:
            List of (true_sequence, observed_sequence) tuples
        """
        if error_profile is None:
            error_profile = {
                'substitution': 0.7,
                'insertion': 0.15,
                'deletion': 0.15
            }
        
        sequences_with_errors = []
        bases = ['A', 'C', 'G', 'T']
        
        for _ in range(n):
            # Generate true sequence
            true_seq = self.generate_sequences(1)[0]
            
            # Apply errors
            observed = list(true_seq)
            i = 0
            while i < len(observed):
                if np.random.random() < error_rate:
                    error_type = np.random.choice(
                        list(error_profile.keys()),
                        p=list(error_profile.values())
                    )
                    
                    if error_type == 'substitution':
                        # Substitute with a different base
                        current = observed[i]
                        alternatives = [b for b in bases if b != current]
                        observed[i] = np.random.choice(alternatives)
                    elif error_type == 'insertion':
                        # Insert a random base
                        observed.insert(i, np.random.choice(bases))
                        i += 1  # Skip the inserted base
                    elif error_type == 'deletion' and len(observed) > 1:
                        # Delete current base
                        del observed[i]
                        i -= 1  # Adjust position
                i += 1
            
            sequences_with_errors.append((true_seq, ''.join(observed)))
        
        return sequences_with_errors
    
    def score_sequence_probabilistic(self, sequence: str) -> Dict[str, float]:
        """
        Score a sequence against the PWM probabilistically.
        
        Instead of returning a single score with a threshold decision,
        return multiple metrics that capture the probabilistic nature
        of the match.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary with multiple scoring metrics
        """
        if len(sequence) != self.motif_length:
            raise ValueError(f"Sequence length {len(sequence)} does not match "
                           f"PWM length {self.motif_length}")
        
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
        
        # Calculate various probabilistic scores
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
                geometric_mean_prob *= prob ** (1/self.motif_length)
        
        # Calculate entropy-based score (how well does it match the distribution)
        entropy_score = 0.0
        for pos, base in enumerate(sequence.upper()):
            idx = base_to_idx.get(base, -1)
            if idx >= 0:
                # Higher score if base is from high-entropy position (more variable)
                pos_entropy = -np.sum(self.pwm[pos] * np.log(self.pwm[pos] + 1e-10))
                max_entropy = -np.log(0.25)  # Maximum possible entropy
                entropy_weight = pos_entropy / max_entropy
                entropy_score += self.pwm[pos, idx] * entropy_weight
        
        entropy_score /= len(position_probs) if position_probs else 1
        
        return {
            'log_likelihood': log_likelihood,
            'mean_probability': np.mean(position_probs) if position_probs else 0,
            'min_probability': min_prob if position_probs else 0,
            'geometric_mean': geometric_mean_prob,
            'entropy_weighted_score': entropy_score,
            'position_probabilities': position_probs
        }
    
    def calculate_diversity_metrics(self, sequences: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a set of generated sequences.
        
        Args:
            sequences: List of generated sequences
            
        Returns:
            Dictionary with diversity metrics
        """
        if not sequences:
            return {}
        
        n_seqs = len(sequences)
        length = len(sequences[0])
        
        # Calculate position-wise entropy
        position_entropies = []
        for pos in range(length):
            base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            for seq in sequences:
                if pos < len(seq):
                    base_counts[seq[pos].upper()] = base_counts.get(seq[pos].upper(), 0) + 1
            
            # Calculate entropy
            counts = np.array(list(base_counts.values()))
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(4)  # Normalized
            position_entropies.append(entropy)
        
        # Calculate pairwise distances
        distances = []
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                distances.append(dist / length)
        
        # Count unique sequences
        unique_seqs = len(set(sequences))
        
        return {
            'mean_position_entropy': np.mean(position_entropies),
            'min_position_entropy': np.min(position_entropies),
            'max_position_entropy': np.max(position_entropies),
            'mean_pairwise_distance': np.mean(distances) if distances else 0,
            'unique_sequences': unique_seqs,
            'uniqueness_ratio': unique_seqs / n_seqs
        }


def create_acc_pwm_from_pattern(pattern: str = "ACCSSV", 
                                pseudocount: float = 0.1) -> np.ndarray:
    """
    Create a PWM from an IUPAC pattern like "ACCSSV".
    
    Args:
        pattern: IUPAC pattern string
        pseudocount: Pseudocount for probability calculation
        
    Returns:
        PWM matrix of shape (length, 4)
    """
    iupac_probs = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'T': [0.0, 0.0, 0.0, 1.0],
        'R': [0.5, 0.0, 0.5, 0.0],  # A or G
        'Y': [0.0, 0.5, 0.0, 0.5],  # C or T
        'S': [0.0, 0.5, 0.5, 0.0],  # G or C
        'W': [0.5, 0.0, 0.0, 0.5],  # A or T
        'K': [0.0, 0.0, 0.5, 0.5],  # G or T
        'M': [0.5, 0.5, 0.0, 0.0],  # A or C
        'B': [0.0, 1/3, 1/3, 1/3],  # C, G, or T
        'D': [1/3, 0.0, 1/3, 1/3],  # A, G, or T
        'H': [1/3, 1/3, 0.0, 1/3],  # A, C, or T
        'V': [1/3, 1/3, 1/3, 0.0],  # A, C, or G
        'N': [0.25, 0.25, 0.25, 0.25]  # Any base
    }
    
    length = len(pattern)
    pwm = np.zeros((length, 4))
    
    for i, base in enumerate(pattern.upper()):
        if base in iupac_probs:
            pwm[i] = iupac_probs[base]
        else:
            # Default to uniform if unknown
            pwm[i] = [0.25, 0.25, 0.25, 0.25]
    
    # Add pseudocount and normalize
    pwm = (pwm + pseudocount) / (1 + 4 * pseudocount)
    
    return pwm


# Example usage and testing
if __name__ == "__main__":
    # Create PWM from IUPAC pattern
    acc_pattern = "ACCSSV"  # A, C, C, S(G/C), S(G/C), V(A/C/G)
    pwm = create_acc_pwm_from_pattern(acc_pattern)
    
    print("PWM from pattern ACCSSV:")
    print("Position\tA\tC\tG\tT")
    for i, row in enumerate(pwm):
        print(f"{i+1}\t" + "\t".join(f"{p:.3f}" for p in row))
    
    # Initialize probabilistic generator
    generator = ProbabilisticPWMGenerator(pwm, temperature=1.0)
    
    # Generate sequences with natural diversity
    print("\nGenerating 10 sequences with natural diversity:")
    sequences = generator.generate_sequences(10)
    for seq in sequences:
        scores = generator.score_sequence_probabilistic(seq)
        print(f"{seq} - Mean prob: {scores['mean_probability']:.3f}")
    
    # Calculate diversity metrics
    diversity = generator.calculate_diversity_metrics(sequences)
    print(f"\nDiversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.3f}")
    
    # Generate with varying temperature
    print("\nEffect of temperature on diversity:")
    for temp in [0.5, 1.0, 2.0]:
        gen = ProbabilisticPWMGenerator(pwm, temperature=temp)
        seqs = gen.generate_sequences(100)
        div = gen.calculate_diversity_metrics(seqs)
        print(f"  Temperature={temp}: uniqueness={div['uniqueness_ratio']:.3f}, "
              f"entropy={div['mean_position_entropy']:.3f}")
    
    # Generate with quality scores
    print("\nGenerating sequences with quality scores:")
    seqs_with_qual = generator.generate_with_quality_scores(3)
    for seq, qual in seqs_with_qual:
        print(f"{seq} - Quality: {qual}")
    
    # Generate with errors
    print("\nGenerating sequences with sequencing errors:")
    seqs_with_errors = generator.generate_with_errors(5, error_rate=0.1)
    for true_seq, obs_seq in seqs_with_errors:
        print(f"True: {true_seq}")
        print(f"Obs:  {obs_seq}")
        print()
