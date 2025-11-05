"""
Constrained Viterbi decoding for tranquillyzer.
Enforces hard length constraints during CRF inference.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple


class ConstrainedViterbiDecoder:
    """
    Viterbi decoder with hard length constraints.
    
    This decoder finds the most likely sequence of labels while
    ensuring that certain elements (like UMI, CBC) have exact lengths.
    """
    
    def __init__(self, label_binarizer, length_constraints: Dict[str, Tuple[int, int]]):
        """
        Args:
            label_binarizer: sklearn LabelBinarizer for label mapping
            length_constraints: Dict like {'UMI': (8, 8), 'CBC': (16, 16)}
                                (min_length, max_length) for each constrained label
        """
        self.label_binarizer = label_binarizer
        self.length_constraints = length_constraints
        self.num_labels = len(label_binarizer.classes_)
        
        # Create label name to index mapping
        self.label_to_idx = {
            label: idx for idx, label in enumerate(label_binarizer.classes_)
        }
        
        # Create index to label name mapping
        self.idx_to_label = {
            idx: label for label, idx in self.label_to_idx.items()
        }
        
    def decode(self, emission_scores: np.ndarray, 
               transition_scores: np.ndarray) -> List[int]:
        """
        Decode with hard length constraints using modified Viterbi.
        
        Args:
            emission_scores: [seq_len, num_labels] - log probabilities
            transition_scores: [num_labels, num_labels] - log transition probs
            
        Returns:
            List of label indices (most likely constrained path)
        """
        seq_len = emission_scores.shape[0]
        
        # Initialize DP table
        # dp[t][label][run_length] = (score, backpointer)
        dp = [{} for _ in range(seq_len)]
        
        # Initialize first position
        for label_idx in range(self.num_labels):
            dp[0][(label_idx, 1)] = (emission_scores[0, label_idx], None)
        
        # Forward pass with length tracking
        for t in range(1, seq_len):
            for curr_label in range(self.num_labels):
                curr_label_name = self.idx_to_label[curr_label]
                
                for prev_state, (prev_score, _) in dp[t-1].items():
                    prev_label, prev_run_length = prev_state
                    prev_label_name = self.idx_to_label[prev_label]
                    
                    # Case 1: Continue current run
                    if prev_label == curr_label:
                        new_run_length = prev_run_length + 1
                        
                        # Check if this violates max length constraint
                        if curr_label_name in self.length_constraints:
                            _, max_len = self.length_constraints[curr_label_name]
                            if new_run_length > max_len:
                                continue  # Skip this path (hard constraint)
                        
                        new_score = (prev_score + 
                                   emission_scores[t, curr_label] + 
                                   transition_scores[prev_label, curr_label])
                        
                        new_state = (curr_label, new_run_length)
                        
                        if new_state not in dp[t] or new_score > dp[t][new_state][0]:
                            dp[t][new_state] = (new_score, prev_state)
                    
                    # Case 2: Start new run
                    else:
                        # Check if previous run meets minimum length constraint
                        if prev_label_name in self.length_constraints:
                            min_len, _ = self.length_constraints[prev_label_name]
                            if prev_run_length < min_len:
                                continue  # Skip this path (hard constraint)
                        
                        new_score = (prev_score + 
                                   emission_scores[t, curr_label] + 
                                   transition_scores[prev_label, curr_label])
                        
                        new_state = (curr_label, 1)
                        
                        if new_state not in dp[t] or new_score > dp[t][new_state][0]:
                            dp[t][new_state] = (new_score, prev_state)
        
        # Backtrack to find best valid path
        # Find best final state that satisfies constraints
        best_final_state = None
        best_final_score = float('-inf')
        
        for state, (score, _) in dp[seq_len - 1].items():
            label_idx, run_length = state
            label_name = self.idx_to_label[label_idx]
            
            # Check if final state satisfies minimum length
            if label_name in self.length_constraints:
                min_len, _ = self.length_constraints[label_name]
                if run_length < min_len:
                    continue
            
            if score > best_final_score:
                best_final_score = score
                best_final_state = state
        
        # If no valid path found, fall back to unconstrained Viterbi
        if best_final_state is None:
            return self._unconstrained_viterbi(emission_scores, transition_scores)
        
        # Backtrack
        path = []
        current_state = best_final_state
        
        for t in range(seq_len - 1, -1, -1):
            label_idx, _ = current_state
            path.append(label_idx)
            
            if dp[t][current_state][1] is not None:
                current_state = dp[t][current_state][1]
        
        path.reverse()
        return path
    
    def _unconstrained_viterbi(self, emission_scores: np.ndarray,
                               transition_scores: np.ndarray) -> List[int]:
        """
        Standard Viterbi decoding without constraints (fallback).
        
        Args:
            emission_scores: [seq_len, num_labels]
            transition_scores: [num_labels, num_labels]
            
        Returns:
            List of label indices
        """
        seq_len = emission_scores.shape[0]
        
        # Initialize
        dp = np.zeros((seq_len, self.num_labels))
        backpointer = np.zeros((seq_len, self.num_labels), dtype=int)
        
        # First position
        dp[0] = emission_scores[0]
        
        # Forward pass
        for t in range(1, seq_len):
            for curr_label in range(self.num_labels):
                # Find best previous state
                scores = (dp[t-1] + 
                         transition_scores[:, curr_label] + 
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
        return path
    
    def decode_batch(self, emission_scores_batch: np.ndarray,
                     transition_scores: np.ndarray) -> List[List[int]]:
        """
        Decode a batch of sequences.
        
        Args:
            emission_scores_batch: [batch_size, seq_len, num_labels]
            transition_scores: [num_labels, num_labels]
            
        Returns:
            List of decoded paths (one per sequence in batch)
        """
        batch_size = emission_scores_batch.shape[0]
        decoded_paths = []
        
        for b in range(batch_size):
            path = self.decode(emission_scores_batch[b], transition_scores)
            decoded_paths.append(path)
        
        return decoded_paths


def apply_constrained_decoding(model, encoded_data, label_binarizer,
                               length_constraints):
    """
    Apply constrained Viterbi decoding to model predictions.
    
    Args:
        model: Trained CRF model
        encoded_data: Encoded input sequences
        label_binarizer: LabelBinarizer for label mapping
        length_constraints: Dict like {'UMI': (8, 8)}
        
    Returns:
        Decoded label sequences (as label names, not indices)
    """
    # Get emission scores from model
    predictions = model.predict(encoded_data)
    
    # Extract transition matrix from CRF layer
    crf_layer = None
    for layer in model.layers:
        if 'crf' in layer.name.lower():
            crf_layer = layer
            break
    
    if crf_layer is None:
        raise ValueError("No CRF layer found in model")
    
    # Get transition scores
    transition_matrix = crf_layer.transitions.numpy()
    
    # Create decoder
    decoder = ConstrainedViterbiDecoder(label_binarizer, length_constraints)
    
    # Decode each sequence
    decoded_indices = decoder.decode_batch(predictions, transition_matrix)
    
    # Convert indices to label names
    decoded_labels = []
    for indices in decoded_indices:
        labels = [label_binarizer.classes_[idx] for idx in indices]
        decoded_labels.append(labels)
    
    return decoded_labels


def evaluate_constrained_decoding(model, test_data, test_labels,
                                   label_binarizer, length_constraints):
    """
    Evaluate model with constrained decoding.
    
    Args:
        model: Trained CRF model
        test_data: Encoded test sequences
        test_labels: True labels
        label_binarizer: LabelBinarizer
        length_constraints: Dict like {'UMI': (8, 8)}
        
    Returns:
        Dict with evaluation metrics
    """
    # Decode with constraints
    predicted_labels = apply_constrained_decoding(
        model, test_data, label_binarizer, length_constraints
    )
    
    # Calculate metrics
    total_correct = 0
    total_positions = 0
    
    length_accuracy = {}
    for label_name in length_constraints.keys():
        length_accuracy[label_name] = {'correct': 0, 'total': 0}
    
    for pred_seq, true_seq in zip(predicted_labels, test_labels):
        # Overall accuracy
        for pred_label, true_label in zip(pred_seq, true_seq):
            if pred_label == true_label:
                total_correct += 1
            total_positions += 1
        
        # Length accuracy for constrained elements
        for label_name, (expected_len, _) in length_constraints.items():
            pred_runs = _extract_runs(pred_seq, label_name)
            
            for run_len in pred_runs:
                length_accuracy[label_name]['total'] += 1
                if run_len == expected_len:
                    length_accuracy[label_name]['correct'] += 1
    
    # Compute metrics
    overall_accuracy = total_correct / total_positions if total_positions > 0 else 0
    
    length_metrics = {}
    for label_name, counts in length_accuracy.items():
        if counts['total'] > 0:
            length_metrics[label_name] = counts['correct'] / counts['total']
        else:
            length_metrics[label_name] = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'length_accuracy': length_metrics
    }


def _extract_runs(sequence: List[str], target_label: str) -> List[int]:
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


# Example usage:
"""
from constrained_viterbi import apply_constrained_decoding

# Define constraints
length_constraints = {
    'UMI': (8, 8),
    'CBC': (6, 6)
}

# Decode with constraints
decoded_labels = apply_constrained_decoding(
    model=trained_model,
    encoded_data=test_sequences,
    label_binarizer=label_binarizer,
    length_constraints=length_constraints
)

# Evaluate
metrics = evaluate_constrained_decoding(
    model=trained_model,
    test_data=test_sequences,
    test_labels=true_labels,
    label_binarizer=label_binarizer,
    length_constraints=length_constraints
)

print(f"Overall accuracy: {metrics['overall_accuracy']:.3f}")
print(f"UMI length accuracy: {metrics['length_accuracy']['UMI']:.3f}")
"""
