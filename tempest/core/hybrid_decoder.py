"""
Hybrid decoder combining soft constraints (training) and hard constraints (inference).

This module provides a unified interface for using both:
1. Soft constraints via length-regularized training (length_crf.py)
2. Hard constraints via constrained Viterbi decoding (constrained_viterbi.py)
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
from .constrained_viterbi import ConstrainedViterbiDecoder
from .length_crf import ModelWithLengthConstrainedCRF


class HybridConstraintDecoder:
    """
    Hybrid decoder that can use both soft and hard constraints.
    
    Soft constraints: Applied during training via regularization
    Hard constraints: Applied during inference via constrained Viterbi
    
    This allows for flexible decoding strategies:
    - Train with soft constraints, decode unconstrained
    - Train with soft constraints, decode with hard constraints  
    - Train unconstrained, decode with hard constraints
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 label_binarizer,
                 length_constraints: Dict[str, Tuple[int, int]],
                 use_hard_constraints: bool = True):
        """
        Initialize hybrid decoder.
        
        Args:
            model: Trained model (may or may not have soft constraints)
            label_binarizer: sklearn LabelBinarizer for label mapping
            length_constraints: Dict like {'UMI': (8, 8), 'ACC': (6, 6)}
            use_hard_constraints: Whether to use hard constraints during decoding
        """
        self.model = model
        self.label_binarizer = label_binarizer
        self.length_constraints = length_constraints
        self.use_hard_constraints = use_hard_constraints
        
        # Check if model has soft constraints
        self.has_soft_constraints = isinstance(model, ModelWithLengthConstrainedCRF)
        
        # Initialize constrained decoder if needed
        if use_hard_constraints:
            self.constrained_decoder = ConstrainedViterbiDecoder(
                label_binarizer, length_constraints
            )
        
        # Extract CRF layer and transition matrix
        self.crf_layer = self._find_crf_layer()
        self.transition_matrix = None
        if self.crf_layer is not None:
            self.transition_matrix = self.crf_layer.get_transition_params()
    
    def _find_crf_layer(self):
        """Find CRF layer in model."""
        if self.has_soft_constraints:
            # For wrapped models, check base_model
            base_model = self.model.base_model
            for layer in base_model.layers:
                if 'crf' in layer.name.lower():
                    return layer
        else:
            # For regular models
            for layer in self.model.layers:
                if 'crf' in layer.name.lower():
                    return layer
        return None
    
    def decode(self, 
               sequences: Union[np.ndarray, tf.Tensor],
               return_scores: bool = False) -> Union[List[List[int]], Tuple]:
        """
        Decode sequences using appropriate strategy.
        
        Args:
            sequences: Input sequences [batch_size, seq_len] or [batch_size, seq_len, features]
            return_scores: Whether to return emission and transition scores
            
        Returns:
            Decoded label sequences (and optionally scores)
        """
        # Get model predictions (emission scores)
        if self.has_soft_constraints:
            # ModelWithLengthConstrainedCRF returns (viterbi, potentials, seq_len, transitions)
            outputs = self.model(sequences, training=False)
            if isinstance(outputs, tuple):
                viterbi_sequence, potentials, _, _ = outputs
                emission_scores = potentials
            else:
                viterbi_sequence = outputs
                emission_scores = None
        else:
            # Regular model
            emission_scores = self.model.predict(sequences)
            viterbi_sequence = None
        
        # Apply decoding strategy
        if self.use_hard_constraints and emission_scores is not None:
            # Use constrained Viterbi
            decoded = self._decode_with_hard_constraints(
                emission_scores, self.transition_matrix
            )
        elif viterbi_sequence is not None:
            # Use model's built-in Viterbi (may have soft constraints)
            decoded = viterbi_sequence.numpy() if hasattr(viterbi_sequence, 'numpy') else viterbi_sequence
        else:
            # Standard Viterbi decoding
            decoded = self._standard_viterbi_batch(
                emission_scores, self.transition_matrix
            )
        
        if return_scores:
            return decoded, emission_scores, self.transition_matrix
        return decoded
    
    def _decode_with_hard_constraints(self, 
                                     emission_scores: np.ndarray,
                                     transition_matrix: np.ndarray) -> np.ndarray:
        """Apply hard constraints during decoding."""
        batch_size = emission_scores.shape[0]
        decoded_paths = []
        
        for b in range(batch_size):
            path = self.constrained_decoder.decode(
                emission_scores[b], transition_matrix
            )
            decoded_paths.append(path)
        
        return np.array(decoded_paths)
    
    def _standard_viterbi_batch(self,
                                emission_scores: np.ndarray,
                                transition_matrix: np.ndarray) -> np.ndarray:
        """Standard batch Viterbi decoding."""
        batch_size, seq_len, num_labels = emission_scores.shape
        decoded = np.zeros((batch_size, seq_len), dtype=np.int32)
        
        for b in range(batch_size):
            decoded[b] = self._standard_viterbi(
                emission_scores[b], transition_matrix
            )
        
        return decoded
    
    def _standard_viterbi(self,
                         emission_scores: np.ndarray,
                         transition_matrix: np.ndarray) -> np.ndarray:
        """Standard Viterbi for single sequence."""
        seq_len, num_labels = emission_scores.shape
        
        # Initialize
        dp = np.zeros((seq_len, num_labels))
        backpointer = np.zeros((seq_len, num_labels), dtype=int)
        
        # First position
        dp[0] = emission_scores[0]
        
        # Forward pass
        for t in range(1, seq_len):
            for curr_label in range(num_labels):
                scores = (dp[t-1] + 
                         transition_matrix[:, curr_label] + 
                         emission_scores[t, curr_label])
                
                best_prev = np.argmax(scores)
                dp[t, curr_label] = scores[best_prev]
                backpointer[t, curr_label] = best_prev
        
        # Backtrack
        path = []
        current_label = np.argmax(dp[seq_len - 1])
        
        for t in range(seq_len - 1, -1, -1):
            path.append(current_label)
            if t > 0:
                current_label = backpointer[t, current_label]
        
        path.reverse()
        return np.array(path)
    
    def evaluate_constraints(self, 
                            sequences: np.ndarray,
                            true_labels: np.ndarray) -> Dict:
        """
        Evaluate how well constraints are satisfied.
        
        Args:
            sequences: Input sequences
            true_labels: True label sequences
            
        Returns:
            Dictionary with constraint satisfaction metrics
        """
        # Decode sequences
        predicted = self.decode(sequences)
        
        metrics = {
            'overall_accuracy': 0.0,
            'constraint_satisfaction': {},
            'length_distribution': {}
        }
        
        # Calculate overall accuracy
        correct = np.sum(predicted == true_labels)
        total = np.prod(true_labels.shape)
        metrics['overall_accuracy'] = correct / total if total > 0 else 0
        
        # Check constraint satisfaction
        for label_name, (min_len, max_len) in self.length_constraints.items():
            label_idx = self.label_binarizer.transform([label_name])[0]
            
            satisfied_count = 0
            total_segments = 0
            length_dist = []
            
            for seq in predicted:
                runs = self._extract_runs(seq, label_idx)
                for run_length in runs:
                    total_segments += 1
                    length_dist.append(run_length)
                    if min_len <= run_length <= max_len:
                        satisfied_count += 1
            
            if total_segments > 0:
                metrics['constraint_satisfaction'][label_name] = satisfied_count / total_segments
                metrics['length_distribution'][label_name] = {
                    'mean': np.mean(length_dist),
                    'std': np.std(length_dist),
                    'min': np.min(length_dist),
                    'max': np.max(length_dist),
                    'expected': (min_len, max_len)
                }
            else:
                metrics['constraint_satisfaction'][label_name] = 0.0
                metrics['length_distribution'][label_name] = None
        
        return metrics
    
    def _extract_runs(self, sequence: np.ndarray, target_label: int) -> List[int]:
        """Extract run lengths of target label in sequence."""
        runs = []
        current_run = 0
        
        for label in sequence:
            if label == target_label:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        return runs
    
    def compare_decoding_strategies(self,
                                   sequences: np.ndarray) -> Dict:
        """
        Compare different decoding strategies.
        
        Args:
            sequences: Input sequences to decode
            
        Returns:
            Dictionary comparing unconstrained vs constrained decoding
        """
        results = {}
        
        # Decode without hard constraints
        self.use_hard_constraints = False
        unconstrained = self.decode(sequences)
        
        # Decode with hard constraints
        self.use_hard_constraints = True
        constrained = self.decode(sequences)
        
        # Compare results
        agreement = np.mean(unconstrained == constrained)
        results['agreement_rate'] = agreement
        
        # Check constraint satisfaction for both
        results['unconstrained'] = self._check_constraints(unconstrained)
        results['constrained'] = self._check_constraints(constrained)
        
        # Measure differences
        results['changed_positions'] = np.sum(unconstrained != constrained)
        results['total_positions'] = np.prod(unconstrained.shape)
        
        return results
    
    def _check_constraints(self, decoded: np.ndarray) -> Dict:
        """Check how well decoded sequences satisfy constraints."""
        satisfaction = {}
        
        for label_name, (min_len, max_len) in self.length_constraints.items():
            label_idx = self.label_binarizer.transform([label_name])[0]
            
            satisfied = 0
            total = 0
            
            for seq in decoded:
                runs = self._extract_runs(seq, label_idx)
                for run_length in runs:
                    total += 1
                    if min_len <= run_length <= max_len:
                        satisfied += 1
            
            satisfaction[label_name] = satisfied / total if total > 0 else 0.0
        
        return satisfaction


def create_hybrid_model(base_model: tf.keras.Model,
                       length_constraints: Dict[str, Tuple[int, int]],
                       label_binarizer,
                       use_soft_constraints: bool = True,
                       use_hard_constraints: bool = True,
                       constraint_weight: float = 5.0,
                       constraint_ramp_epochs: int = 5) -> Tuple[tf.keras.Model, HybridConstraintDecoder]:
    """
    Create a model with both soft and hard constraints.
    
    Args:
        base_model: Base CNN-BiLSTM-CRF model
        length_constraints: Length constraints for specific labels
        label_binarizer: Label encoder
        use_soft_constraints: Whether to add soft constraints (training)
        use_hard_constraints: Whether to use hard constraints (inference)
        constraint_weight: Weight for soft constraints
        constraint_ramp_epochs: Epochs to ramp up constraint weight
        
    Returns:
        Tuple of (model with soft constraints, hybrid decoder)
    """
    # Add soft constraints if requested
    if use_soft_constraints:
        from .length_crf import ModelWithLengthConstrainedCRF
        
        model = ModelWithLengthConstrainedCRF(
            base_model=base_model,
            length_constraints=length_constraints,
            constraint_weight=constraint_weight,
            label_binarizer=label_binarizer,
            constraint_ramp_epochs=constraint_ramp_epochs,
            sparse_target=True
        )
    else:
        model = base_model
    
    # Create hybrid decoder
    decoder = HybridConstraintDecoder(
        model=model,
        label_binarizer=label_binarizer,
        length_constraints=length_constraints,
        use_hard_constraints=use_hard_constraints
    )
    
    return model, decoder
