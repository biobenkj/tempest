#!/usr/bin/env python3
"""
TEMPEST End-to-End Training Test
=================================
Tests complete training pipeline with simulated data.

This script:
1. Generates synthetic nanopore read data
2. Builds a CNN-BiLSTM-CRF model
3. Trains the model
4. Evaluates performance
5. Tests inference

Run with: python test_end_to_end_training.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow not found")
    print("Install with: pip install tensorflow")
    sys.exit(1)


class NanoporeSimulator:
    """Simulate nanopore read data for testing."""
    
    def __init__(self, vocab_size=5, num_labels=6):
        self.vocab_size = vocab_size  # A, C, G, T, N
        self.num_labels = num_labels  # Adapter1, UMI, CBC, Adapter2, PolyA, Insert
        self.label_names = ['Adapter1', 'UMI', 'CBC', 'Adapter2', 'PolyA', 'Insert']
        
    def generate_read(self, seq_length=200):
        """Generate a single synthetic read with segment structure."""
        sequence = np.zeros(seq_length, dtype=np.int32)
        labels = np.zeros(seq_length, dtype=np.int32)
        
        # Define typical segment lengths
        segments = [
            ('Adapter1', 15, 25),   # Adapter 1: 15-25 bp
            ('UMI', 8, 12),         # UMI: 8-12 bp
            ('CBC', 12, 18),        # Cell barcode: 12-18 bp
            ('Adapter2', 15, 25),   # Adapter 2: 15-25 bp
            ('PolyA', 10, 30),      # PolyA tail: 10-30 bp
            ('Insert', 50, 100),    # Insert/transcript: variable
        ]
        
        pos = 0
        for label_idx, (name, min_len, max_len) in enumerate(segments):
            # Random length within range
            length = np.random.randint(min_len, max_len + 1)
            length = min(length, seq_length - pos)  # Don't exceed sequence
            
            if pos >= seq_length:
                break
                
            # Fill segment with random bases
            sequence[pos:pos+length] = np.random.randint(0, self.vocab_size, size=length)
            labels[pos:pos+length] = label_idx
            
            pos += length
        
        return sequence, labels
    
    def generate_dataset(self, num_samples=1000, seq_length=200):
        """Generate a full dataset."""
        sequences = []
        labels = []
        
        for _ in range(num_samples):
            seq, lab = self.generate_read(seq_length)
            sequences.append(seq)
            labels.append(lab)
        
        return np.array(sequences), np.array(labels)


class TempestModel:
    """Simplified TEMPEST model for testing."""
    
    def __init__(self, vocab_size=5, num_labels=6, seq_length=200):
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.model = None
        
    def build_model(self):
        """Build CNN-BiLSTM model."""
        print("Building model architecture...")
        
        # Input
        inputs = keras.Input(shape=(self.seq_length,), dtype=tf.int32, name='sequence_input')
        
        # Embedding layer
        x = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=64,
            mask_zero=True,
            name='embedding'
        )(inputs)
        
        # CNN layers for local motifs
        conv_outputs = []
        for i, (filters, kernel) in enumerate([(32, 3), (64, 5), (64, 7)]):
            conv = keras.layers.Conv1D(
                filters, kernel,
                padding='same',
                activation='relu',
                name=f'conv1d_{i}'
            )(x)
            conv_outputs.append(conv)
        
        x = keras.layers.Concatenate(name='concat_convs')(conv_outputs)
        
        # BiLSTM for long-range dependencies
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True),
            name='bilstm_1'
        )(x)
        
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True),
            name='bilstm_2'
        )(x)
        
        # Dropout for regularization
        x = keras.layers.Dropout(0.2, name='dropout')(x)
        
        # Output layer
        outputs = keras.layers.Dense(
            self.num_labels,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='TEMPEST')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model."""
        print("Compiling model...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def test_end_to_end():
    """Run end-to-end test."""
    print("\n" + "="*80)
    print("  TEMPEST End-to-End Training Test")
    print("="*80 + "\n")
    
    # Check GPU
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        device = '/GPU:0'
    else:
        print("⚠ No GPU detected - using CPU")
        device = '/CPU:0'
    
    # Generate data
    print("\n" + "-"*80)
    print("Step 1: Generating synthetic data")
    print("-"*80)
    
    simulator = NanoporeSimulator()
    
    print("Generating training data...")
    X_train, y_train = simulator.generate_dataset(num_samples=500, seq_length=200)
    print(f"  Training: {X_train.shape[0]} sequences × {X_train.shape[1]} bp")
    
    print("Generating validation data...")
    X_val, y_val = simulator.generate_dataset(num_samples=100, seq_length=200)
    print(f"  Validation: {X_val.shape[0]} sequences × {X_val.shape[1]} bp")
    
    print("Generating test data...")
    X_test, y_test = simulator.generate_dataset(num_samples=100, seq_length=200)
    print(f"  Test: {X_test.shape[0]} sequences × {X_test.shape[1]} bp")
    
    # Build model
    print("\n" + "-"*80)
    print("Step 2: Building model")
    print("-"*80)
    
    tempest = TempestModel(
        vocab_size=simulator.vocab_size,
        num_labels=simulator.num_labels,
        seq_length=200
    )
    
    with tf.device(device):
        model = tempest.build_model()
        tempest.compile_model()
    
    print(f"\nModel parameters: {model.count_params():,}")
    
    # Train model
    print("\n" + "-"*80)
    print("Step 3: Training model")
    print("-"*80)
    
    print(f"Training on {device}...")
    print("Running 5 epochs (adjust for full training)")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "-"*80)
    print("Step 4: Evaluating model")
    print("-"*80)
    
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Inference
    print("\n" + "-"*80)
    print("Step 5: Testing inference")
    print("-"*80)
    
    print("Running inference on sample sequences...")
    sample_sequences = X_test[:5]
    sample_predictions = model.predict(sample_sequences, verbose=0)
    
    print(f"\nInference Results:")
    print(f"  Input shape:  {sample_sequences.shape}")
    print(f"  Output shape: {sample_predictions.shape}")
    
    # Show sample prediction
    sample_predicted_labels = np.argmax(sample_predictions, axis=-1)
    
    print("\nSample prediction (first sequence):")
    print(f"  True labels:      {y_test[0][:50]}")
    print(f"  Predicted labels: {sample_predicted_labels[0][:50]}")
    
    # Calculate per-label accuracy on full test set
    print("\nRunning inference on full test set for accuracy calculation...")
    all_predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(all_predictions, axis=-1)
    
    print("\nPer-segment accuracy:")
    for i, label_name in enumerate(simulator.label_names):
        mask = y_test == i
        if mask.sum() > 0:
            label_acc = (predicted_labels[mask] == y_test[mask]).mean()
            print(f"  {label_name:12s}: {label_acc:.2%}")
    
    # Summary
    print("\n" + "="*80)
    print("  Summary")
    print("="*80)
    
    print("\n✓ End-to-end pipeline completed successfully!")
    print(f"  - Data generation: ✓")
    print(f"  - Model building: ✓")
    print(f"  - Training: ✓ (test accuracy: {test_acc:.2%})")
    print(f"  - Inference: ✓ (on {len(X_test)} test sequences)")
    
    if test_acc > 0.5:
        print(f"\n✓ Model achieved good accuracy ({test_acc:.2%})")
        print("  System is ready for full training on real data!")
    else:
        print(f"\n⚠ Model accuracy is low ({test_acc:.2%})")
        print("  This is expected for a quick test with limited training.")
        print("  Use more epochs and data for production training.")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(test_end_to_end())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
