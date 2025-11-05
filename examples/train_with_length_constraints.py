"""
Example script demonstrating the use of length-constrained CRF for sequence annotation.

This script shows how to:
1. Build a CNN-BiLSTM-CRF model with length constraints
2. Train with constraint weight ramping
3. Evaluate on test data
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tempest.core import build_cnn_bilstm_crf, ModelWithLengthConstrainedCRF


def generate_synthetic_data(num_sequences=1000, max_len=200):
    """
    Generate synthetic sequence data for demonstration.
    
    Returns:
        X: Sequences encoded as integers (batch_size, max_len)
        y: Labels encoded as integers (batch_size, max_len)
    """
    # Vocabulary: A=0, C=1, G=2, T=3, N=4
    vocab_size = 5
    
    # Labels: ADAPTER5=0, UMI=1, ACC=2, BARCODE=3, INSERT=4, ADAPTER3=5, PAD=6
    label_names = ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3', 'PAD']
    num_labels = len(label_names)
    
    X = []
    y = []
    
    for _ in range(num_sequences):
        # Generate a sequence with known structure
        seq = []
        labels = []
        
        # ADAPTER5 (10bp)
        seq.extend(np.random.randint(0, 4, 10))
        labels.extend([0] * 10)
        
        # UMI (8bp - constrained)
        seq.extend(np.random.randint(0, 4, 8))
        labels.extend([1] * 8)
        
        # ACC (6bp - constrained)
        seq.extend(np.random.randint(0, 4, 6))
        labels.extend([2] * 6)
        
        # BARCODE (16bp - constrained)
        seq.extend(np.random.randint(0, 4, 16))
        labels.extend([3] * 16)
        
        # INSERT (variable 50-100bp)
        insert_len = np.random.randint(50, 100)
        seq.extend(np.random.randint(0, 4, insert_len))
        labels.extend([4] * insert_len)
        
        # ADAPTER3 (10bp)
        seq.extend(np.random.randint(0, 4, 10))
        labels.extend([5] * 10)
        
        # Pad to max_len
        current_len = len(seq)
        if current_len < max_len:
            pad_len = max_len - current_len
            seq.extend([4] * pad_len)  # N for padding
            labels.extend([6] * pad_len)  # PAD label
        else:
            seq = seq[:max_len]
            labels = labels[:max_len]
        
        X.append(seq)
        y.append(labels)
    
    return np.array(X), np.array(y)


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("="*80)
    print("TEMPEST: Length-Constrained CRF Training Example")
    print("="*80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic sequence data...")
    X, y = generate_synthetic_data(num_sequences=1000, max_len=200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training sequences: {X_train.shape}")
    print(f"   Test sequences: {X_test.shape}")
    
    # Create label binarizer
    label_names = ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3', 'PAD']
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(label_names)
    
    # Define length constraints
    length_constraints = {
        'UMI': (8, 8),      # UMI must be exactly 8bp
        'ACC': (6, 6),      # ACC must be exactly 6bp
        'BARCODE': (16, 16) # Barcode must be exactly 16bp
    }
    
    print("\n2. Building CNN-BiLSTM-CRF model with length constraints...")
    print(f"   Constraints: {length_constraints}")
    
    # Build base model
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
    
    # Wrap with length constraints
    model = ModelWithLengthConstrainedCRF(
        base_model=base_model,
        length_constraints=length_constraints,
        constraint_weight=5.0,
        label_binarizer=label_binarizer,
        max_seq_len=200,
        constraint_ramp_epochs=5,
        sparse_target=True,  # Our y is already in sparse format
        metric='accuracy'
    )
    
    print(f"   Model parameters: {base_model.count_params():,}")
    
    # Compile model
    print("\n3. Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    # Training with constraint weight ramping
    print("\n4. Training with constraint weight ramping...")
    print("   Ramping constraint weight from 0 to 5.0 over 5 epochs")
    
    batch_size = 32
    epochs = 10
    
    # Convert data to tf.data.Dataset for efficiency
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Training loop with epoch callback
    history = {'loss': [], 'crf_loss': [], 'length_penalty': [], 'constraint_weight': []}
    
    for epoch in range(epochs):
        print(f"\n   Epoch {epoch + 1}/{epochs}")
        
        # Update epoch counter for constraint ramping
        model.on_epoch_begin(epoch)
        
        # Train for one epoch
        epoch_history = model.fit(
            train_dataset,
            epochs=1,
            verbose=1,
            validation_data=test_dataset
        )
        
        # Store metrics
        for key in ['loss', 'crf_loss', 'length_penalty', 'constraint_weight']:
            if key in epoch_history.history:
                history[key].append(epoch_history.history[key][0])
    
    print("\n5. Training Summary:")
    print("   Final metrics:")
    print(f"   - Loss: {history['loss'][-1]:.4f}")
    print(f"   - CRF Loss: {history['crf_loss'][-1]:.4f}")
    print(f"   - Length Penalty: {history['length_penalty'][-1]:.4f}")
    print(f"   - Constraint Weight: {history['constraint_weight'][-1]:.2f}")
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_metrics = model.evaluate(test_dataset, verbose=0)
    print(f"   Test accuracy: {test_metrics[-1]:.4f}")
    
    # Make predictions
    print("\n7. Making predictions on sample sequence...")
    sample_x = X_test[:1]
    sample_y_true = y_test[:1]
    
    # Get predictions (returns tuple during training mode)
    predictions = model.predict(sample_x)
    if isinstance(predictions, tuple):
        viterbi_sequence = predictions[0]
    else:
        viterbi_sequence = predictions
    
    print(f"   Sample sequence length: {np.sum(sample_y_true[0] != 6)}")  # Exclude padding
    print(f"   True labels: {sample_y_true[0][:50]}")
    print(f"   Predicted:   {viterbi_sequence[0][:50]}")
    
    # Check if length constraints are satisfied
    print("\n8. Checking length constraint satisfaction:")
    for label_name, (min_len, max_len) in length_constraints.items():
        label_idx = label_binarizer.classes_.tolist().index(label_name)
        
        # Find runs of this label in predictions
        pred_mask = viterbi_sequence[0] == label_idx
        runs = []
        current_run = 0
        for i in range(len(pred_mask)):
            if pred_mask[i]:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        
        if runs:
            print(f"   {label_name}: Found runs of lengths {runs} (constraint: [{min_len}, {max_len}])")
            satisfied = all(min_len <= r <= max_len for r in runs)
            print(f"              Constraint satisfied: {satisfied}")
        else:
            print(f"   {label_name}: No segments found")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
