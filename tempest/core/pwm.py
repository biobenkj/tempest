"""
PWM (Position Weight Matrix) module for Tempest.

Provides functions for scoring sequences against PWMs and detecting
ACC sequences in reads.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class PWMScorer:
    """
    Position Weight Matrix scorer for sequence motif detection.
    
    Attributes:
        pwm: PWM matrix of shape (length, 4) for [A, C, G, T]
        threshold: Minimum score threshold for detection
        log_pwm: Log-transformed PWM for scoring
    """
    
    def __init__(self, pwm: np.ndarray, threshold: float = 0.7):
        """
        Initialize PWM scorer.
        
        Args:
            pwm: PWM matrix of shape (length, 4)
            threshold: Score threshold for detection (0-1)
        """
        self.pwm = pwm
        self.threshold = threshold
        self.motif_length = pwm.shape[0]
        
        # Convert to log odds for efficient scoring
        # Add pseudocount to avoid log(0)
        pseudocount = 1e-10
        self.log_pwm = np.log(pwm + pseudocount)
        
        # Background frequencies (uniform)
        background = np.ones(4) * 0.25
        log_background = np.log(background)
        
        # Log odds ratio
        self.log_odds_pwm = self.log_pwm - log_background
        
        logger.info(f"Initialized PWM scorer: length={self.motif_length}, threshold={threshold}")
    
    def score_sequence(self, sequence: str) -> float:
        """
        Score a sequence against the PWM.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Normalized score (0-1), higher is better match
        """
        if len(sequence) != self.motif_length:
            raise ValueError(
                f"Sequence length {len(sequence)} does not match "
                f"PWM length {self.motif_length}"
            )
        
        # Base to index mapping
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
        
        score = 0.0
        valid_positions = 0
        
        for pos, base in enumerate(sequence.upper()):
            idx = base_to_idx.get(base, -1)
            if idx >= 0:
                score += self.log_odds_pwm[pos, idx]
                valid_positions += 1
        
        if valid_positions == 0:
            return 0.0
        
        # Normalize score to 0-1 range
        # Maximum possible score
        max_score = sum(np.max(self.log_odds_pwm[i, :]) for i in range(self.motif_length))
        min_score = sum(np.min(self.log_odds_pwm[i, :]) for i in range(self.motif_length))
        
        # Normalize
        if max_score > min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.0
        
        return max(0.0, min(1.0, normalized_score))
    
    def find_best_match(self, sequence: str, 
                       start: Optional[int] = None,
                       end: Optional[int] = None) -> Tuple[int, float, str]:
        """
        Find the best matching position in a sequence.
        
        Args:
            sequence: DNA sequence string
            start: Start position to search (default: 0)
            end: End position to search (default: len(sequence))
            
        Returns:
            Tuple of (position, score, matched_sequence)
        """
        if start is None:
            start = 0
        if end is None:
            end = len(sequence)
        
        best_pos = -1
        best_score = 0.0
        best_match = ""
        
        # Scan through sequence
        for pos in range(start, min(end, len(sequence) - self.motif_length + 1)):
            subseq = sequence[pos:pos + self.motif_length]
            score = self.score_sequence(subseq)
            
            if score > best_score:
                best_score = score
                best_pos = pos
                best_match = subseq
        
        return best_pos, best_score, best_match
    
    def detect_motif(self, sequence: str,
                    start: Optional[int] = None,
                    end: Optional[int] = None) -> Optional[Tuple[int, int, float, str]]:
        """
        Detect motif in sequence if score exceeds threshold.
        
        Args:
            sequence: DNA sequence string
            start: Start position to search
            end: End position to search
            
        Returns:
            Tuple of (start_pos, end_pos, score, matched_sequence) if detected,
            None otherwise
        """
        pos, score, match = self.find_best_match(sequence, start, end)
        
        if score >= self.threshold and pos >= 0:
            return (pos, pos + self.motif_length, score, match)
        
        return None
    
    def score_multiple_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Score multiple sequences efficiently.
        
        Args:
            sequences: List of DNA sequence strings
            
        Returns:
            Array of scores
        """
        scores = np.array([self.score_sequence(seq) for seq in sequences])
        return scores
    
    def validate_against_iupac(self, sequence: str, iupac_pattern: str) -> bool:
        """
        Check if sequence matches IUPAC pattern.
        
        Args:
            sequence: DNA sequence
            iupac_pattern: IUPAC pattern (e.g., "ACCSSV")
            
        Returns:
            True if sequence matches pattern
        """
        if len(sequence) != len(iupac_pattern):
            return False
        
        # IUPAC codes
        iupac_codes = {
            'A': ['A'],
            'C': ['C'],
            'G': ['G'],
            'T': ['T'],
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            'S': ['G', 'C'],
            'W': ['A', 'T'],
            'K': ['G', 'T'],
            'M': ['A', 'C'],
            'B': ['C', 'G', 'T'],
            'D': ['A', 'G', 'T'],
            'H': ['A', 'C', 'T'],
            'V': ['A', 'C', 'G'],
            'N': ['A', 'C', 'G', 'T']
        }
        
        for base, pattern in zip(sequence.upper(), iupac_pattern.upper()):
            allowed = iupac_codes.get(pattern, [])
            if base not in allowed:
                return False
        
        return True


def generate_acc_from_pwm(pwm: np.ndarray, n: int = 1, random_state: Optional[int] = None) -> List[str]:
    """
    Generate ACC sequences by sampling from PWM.
    
    Args:
        pwm: PWM matrix of shape (length, 4)
        n: Number of sequences to generate
        random_state: Random seed for reproducibility
        
    Returns:
        List of generated ACC sequences
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bases = ['A', 'C', 'G', 'T']
    sequences = []
    
    for _ in range(n):
        seq = []
        for pos in range(pwm.shape[0]):
            # Sample base according to PWM probabilities
            base_idx = np.random.choice(4, p=pwm[pos])
            seq.append(bases[base_idx])
        sequences.append(''.join(seq))
    
    return sequences


def compute_pwm_from_sequences(sequences: List[str]) -> np.ndarray:
    """
    Compute PWM from a list of aligned sequences.
    
    Args:
        sequences: List of aligned DNA sequences
        
    Returns:
        PWM matrix of shape (length, 4)
    """
    if not sequences:
        raise ValueError("Empty sequence list")
    
    length = len(sequences[0])
    if not all(len(seq) == length for seq in sequences):
        raise ValueError("All sequences must have the same length")
    
    # Count bases at each position
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    counts = np.zeros((length, 4))
    
    for seq in sequences:
        for pos, base in enumerate(seq.upper()):
            if base in base_to_idx:
                counts[pos, base_to_idx[base]] += 1
    
    # Add pseudocount
    pseudocount = 1.0
    counts += pseudocount
    
    # Normalize to probabilities
    pwm = counts / counts.sum(axis=1, keepdims=True)
    
    return pwm


def evaluate_pwm_performance(pwm_scorer: PWMScorer, 
                            true_sequences: List[str],
                            false_sequences: List[str]) -> dict:
    """
    Evaluate PWM performance on true vs false sequences.
    
    Args:
        pwm_scorer: PWMScorer instance
        true_sequences: List of true ACC sequences
        false_sequences: List of non-ACC sequences
        
    Returns:
        Dictionary with performance metrics
    """
    true_scores = pwm_scorer.score_multiple_sequences(true_sequences)
    false_scores = pwm_scorer.score_multiple_sequences(false_sequences)
    
    # Compute metrics at current threshold
    threshold = pwm_scorer.threshold
    true_positives = (true_scores >= threshold).sum()
    false_positives = (false_scores >= threshold).sum()
    true_negatives = (false_scores < threshold).sum()
    false_negatives = (true_scores < threshold).sum()
    
    sensitivity = true_positives / len(true_sequences) if len(true_sequences) > 0 else 0
    specificity = true_negatives / len(false_sequences) if len(false_sequences) > 0 else 0
    
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    if sensitivity + precision > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0
    
    return {
        'threshold': threshold,
        'true_positive_rate': sensitivity,
        'false_positive_rate': 1 - specificity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'true_scores_mean': float(true_scores.mean()),
        'false_scores_mean': float(false_scores.mean()),
        'true_scores_std': float(true_scores.std()),
        'false_scores_std': float(false_scores.std())
    }
