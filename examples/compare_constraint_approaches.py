"""
Example demonstrating the combination of soft and hard constraints for sequence annotation.

This script shows three approaches:
1. Soft constraints only (training regularization)
2. Hard constraints only (inference enforcement)
3. Both soft and hard constraints (hybrid approach)
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tempest.core import build_cnn_bilstm_crf
from tempest.core.hybrid_decoder import create_hybrid_model, HybridConstraintDecoder
from tempest.core.length_crf import ModelWithLengthConstrainedCRF
from tempest.core.constrained_viterbi import ConstrainedViterbiDecoder


def generate_realistic_data(num_sequences=1000, error_rate=0.05):
    """
    Generate synthetic sequence data with some violations of ideal lengths.
    
    Args:
        num_sequences: Number of sequences to generate
        error_rate: Probability of length violations
        
    Returns:
        X: Sequences encoded as integers
        y: Labels encoded as integers
    """
    X = []
    y = []
    
    for _ in range(num_sequences):
        seq = []
        labels = []
        
        # ADAPTER5 (10bp)
        seq.extend(np.random.randint(0, 4, 10))
        labels.extend([0] * 10)
        
        # UMI (should be 8bp, but sometimes varies)
        if np.random.random() < error_rate:
            umi_len = np.random.choice([7, 9])  # Wrong length
        else:
            umi_len = 8  # Correct length
        seq.extend(np.random.randint(0, 4, umi_len))
        labels.extend([1] * umi_len)
        
        # ACC (should be 6bp, but sometimes varies)
        if np.random.random() < error_rate:
            acc_len = np.random.choice([5, 7])  # Wrong length
        else:
            acc_len = 6  # Correct length
        seq.extend(np.random.randint(0, 4, acc_len))
        labels.extend([2] * acc_len)
        
        # BARCODE (should be 16bp, but sometimes varies)
        if np.random.random() < error_rate:
            barcode_len = np.random.choice([15, 17])  # Wrong length
        else:
            barcode_len = 16  # Correct length
        seq.extend(np.random.randint(0, 4, barcode_len))
        labels.extend([3] * barcode_len)
        
        # INSERT (variable 50-100bp)
        insert_len = np.random.randint(50, 100)
        seq.extend(np.random.randint(0, 4, insert_len))
        labels.extend([4] * insert_len)
        
        # ADAPTER3 (10bp)
        seq.extend(np.random.randint(0, 4, 10))
        labels.extend([5] * 10)
        
        # Pad to fixed length
        max_len = 200
        current_len = len(seq)
        if current_len < max_len:
            pad_len = max_len - current_len
            seq.extend([4] * pad_len)
            labels.extend([6] * pad_len)
        else:
            seq = seq[:max_len]
            labels = labels[:max_len]
        
        X.append(seq)
        y.append(labels)
    
    return np.array(X), np.array(y)


def evaluate_approach(approach_name, model, decoder, X_test, y_test, label_binarizer):
    """Evaluate a specific approach and print results."""
    print(f"\n{approach_name}:")
    print("=" * 50)
    
    # Decode
    predictions = decoder.decode(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Check constraint satisfaction
    metrics = decoder.evaluate_constraints(X_test, y_test)
    
    print("\nConstraint satisfaction rates:")
    for label_name, rate in metrics['constraint_satisfaction'].items():
        print(f"  {label_name}: {rate:.4f}")
    
    if metrics['length_distribution']:
        print("\nLength distributions:")
        for label_name, dist in metrics['length_distribution'].items():
            if dist:
                print(f"  {label_name}:")
                print(f"    Mean: {dist['mean']:.2f} (expected: {dist['expected']})")
                print(f"    Std:  {dist['std']:.2f}")
                print(f"    Range: [{dist['min']}, {dist['max']}]")
    
    return metrics


def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 80)
    print("COMPARISON: Soft vs Hard Constraints for Sequence Annotation")
    print("=" * 80)
    
    # Generate data with some length violations
    print("\n1. Generating synthetic data with 5% length violations...")
    X, y = generate_realistic_data(num_sequences=1000, error_rate=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training sequences: {X_train.shape}")
    print(f"   Test sequences: {X_test.shape}")
    print(f"   ~5% of UMI/ACC/BARCODE segments have incorrect lengths")
    
    # Setup label binarizer
    label_names = ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3', 'PAD']
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(label_names)
    
    # Define length constraints
    length_constraints = {
        'UMI': (8, 8),
        'ACC': (6, 6),
        'BARCODE': (16, 16)
    }
    
    print(f"\n2. Length constraints: {length_constraints}")
    
    # Build base model
    print("\n3. Building base CNN-BiLSTM-CRF model...")
    base_model = build_cnn_bilstm_crf(
        vocab_size=5,
        embedding_dim=64,
        cnn_filters=[32, 64],
        cnn_kernels=[3, 5],
        lstm_units=64,
        lstm_layers=2,
        dropout=0.3,
        num_labels=7,
        max_seq_len=200,
        use_cnn=True,
        use_bilstm=True,
        use_crf=True
    )
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    print("\n" + "=" * 80)
    print("APPROACH 1: SOFT CONSTRAINTS ONLY (Training Regularization)")
    print("=" * 80)
    
    # Create model with soft constraints
    soft_model = ModelWithLengthConstrainedCRF(
        base_model=base_model,
        length_constraints=length_constraints,
        constraint_weight=5.0,
        label_binarizer=label_binarizer,
        max_seq_len=200,
        constraint_ramp_epochs=3,
        sparse_target=True
    )
    
    soft_model.compile(optimizer='adam')
    
    print("Training with soft constraints (length regularization)...")
    for epoch in range(5):
        soft_model.on_epoch_begin(epoch)
        history = soft_model.fit(train_dataset, epochs=1, verbose=0)
        print(f"  Epoch {epoch+1}: Loss={history.history['loss'][0]:.4f}, "
              f"Constraint Weight={history.history['constraint_weight'][0]:.2f}")
    
    # Create decoder without hard constraints
    soft_decoder = HybridConstraintDecoder(
        model=soft_model,
        label_binarizer=label_binarizer,
        length_constraints=length_constraints,
        use_hard_constraints=False  # No hard constraints
    )
    
    soft_metrics = evaluate_approach("Soft Constraints Only", 
                                    soft_model, soft_decoder, 
                                    X_test, y_test, label_binarizer)
    
    print("\n" + "=" * 80)
    print("APPROACH 2: HARD CONSTRAINTS ONLY (Inference Enforcement)")
    print("=" * 80)
    
    # Train base model without soft constraints
    base_model_2 = build_cnn_bilstm_crf(
        vocab_size=5,
        embedding_dim=64,
        cnn_filters=[32, 64],
        cnn_kernels=[3, 5],
        lstm_units=64,
        lstm_layers=2,
        dropout=0.3,
        num_labels=7,
        max_seq_len=200,
        use_cnn=True,
        use_bilstm=True,
        use_crf=True
    )
    
    base_model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training without soft constraints (standard CRF)...")
    history = base_model_2.fit(train_dataset, epochs=5, verbose=0)
    print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Create decoder with hard constraints
    hard_decoder = HybridConstraintDecoder(
        model=base_model_2,
        label_binarizer=label_binarizer,
        length_constraints=length_constraints,
        use_hard_constraints=True  # Use hard constraints
    )
    
    hard_metrics = evaluate_approach("Hard Constraints Only",
                                    base_model_2, hard_decoder,
                                    X_test, y_test, label_binarizer)
    
    print("\n" + "=" * 80)
    print("APPROACH 3: HYBRID (Soft Training + Hard Inference)")
    print("=" * 80)
    
    # Create hybrid model with both soft and hard constraints
    hybrid_model, hybrid_decoder = create_hybrid_model(
        base_model=build_cnn_bilstm_crf(
            vocab_size=5,
            embedding_dim=64,
            cnn_filters=[32, 64],
            cnn_kernels=[3, 5],
            lstm_units=64,
            lstm_layers=2,
            dropout=0.3,
            num_labels=7,
            max_seq_len=200,
            use_cnn=True,
            use_bilstm=True,
            use_crf=True
        ),
        length_constraints=length_constraints,
        label_binarizer=label_binarizer,
        use_soft_constraints=True,  # Train with soft constraints
        use_hard_constraints=True,  # Decode with hard constraints
        constraint_weight=5.0,
        constraint_ramp_epochs=3
    )
    
    hybrid_model.compile(optimizer='adam')
    
    print("Training with soft constraints...")
    for epoch in range(5):
        hybrid_model.on_epoch_begin(epoch)
        history = hybrid_model.fit(train_dataset, epochs=1, verbose=0)
        print(f"  Epoch {epoch+1}: Loss={history.history['loss'][0]:.4f}")
    
    print("\nDecoding with hard constraints...")
    hybrid_metrics = evaluate_approach("Hybrid (Soft + Hard)",
                                      hybrid_model, hybrid_decoder,
                                      X_test, y_test, label_binarizer)
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print("\nOverall Accuracy:")
    print(f"  Soft only:   {soft_metrics['overall_accuracy']:.4f}")
    print(f"  Hard only:   {hard_metrics['overall_accuracy']:.4f}")
    print(f"  Hybrid:      {hybrid_metrics['overall_accuracy']:.4f}")
    
    print("\nConstraint Satisfaction (average):")
    soft_avg = np.mean(list(soft_metrics['constraint_satisfaction'].values()))
    hard_avg = np.mean(list(hard_metrics['constraint_satisfaction'].values()))
    hybrid_avg = np.mean(list(hybrid_metrics['constraint_satisfaction'].values()))
    
    print(f"  Soft only:   {soft_avg:.4f}")
    print(f"  Hard only:   {hard_avg:.4f} {'✓ (guaranteed)' if hard_avg > 0.99 else ''}")
    print(f"  Hybrid:      {hybrid_avg:.4f} {'✓ (guaranteed)' if hybrid_avg > 0.99 else ''}")
    
    print("\nKey Insights:")
    print("- Soft constraints improve model's tendency toward correct lengths")
    print("- Hard constraints guarantee exact lengths but may reduce flexibility")
    print("- Hybrid approach combines benefits: learns patterns + guarantees constraints")
    
    # Test decoding strategy comparison
    print("\n" + "=" * 80)
    print("BONUS: Comparing Decoding Strategies on Hybrid Model")
    print("=" * 80)
    
    comparison = hybrid_decoder.compare_decoding_strategies(X_test[:10])
    print(f"\nAgreement rate between unconstrained and constrained: {comparison['agreement_rate']:.4f}")
    print(f"Positions changed by hard constraints: {comparison['changed_positions']}/{comparison['total_positions']}")
    
    print("\nConstraint satisfaction without hard constraints:")
    for label, rate in comparison['unconstrained'].items():
        print(f"  {label}: {rate:.4f}")
    
    print("\nConstraint satisfaction with hard constraints:")
    for label, rate in comparison['constrained'].items():
        print(f"  {label}: {rate:.4f}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
