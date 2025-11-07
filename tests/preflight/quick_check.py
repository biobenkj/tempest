#!/usr/bin/env python3
"""
Quick TEMPEST Validation
========================
Fast checks for GPU and basic functionality.

Run with: python quick_check.py
"""

import sys
import os

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_gpu():
    """Quick GPU check."""
    print("Checking GPU availability...")
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            
            # Test a simple GPU operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
            
            print(f"✓ GPU computation successful")
            return True
        else:
            print("✗ No GPUs detected")
            print("  Training will run on CPU (slower)")
            return False
            
    except Exception as e:
        print(f"✗ GPU check failed: {str(e)}")
        return False


def check_imports():
    """Check critical imports."""
    print("\nChecking imports...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False


def test_data_flow():
    """Test basic data flow through a model."""
    print("\nTesting data flow...")
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create dummy data
        X = np.random.randint(0, 5, size=(10, 100))
        y = np.random.randint(0, 5, size=(10, 100))
        
        # Build simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(5, 16, input_length=100),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Test forward pass
        predictions = model.predict(X[:2], verbose=0)
        print(f"✓ Model forward pass: input {X.shape} -> output {predictions.shape}")
        
        # Test one training step
        model.fit(X[:5], y[:5], epochs=1, batch_size=2, verbose=0)
        print(f"✓ Training step completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Data flow test failed: {str(e)}")
        return False


def main():
    print("="*60)
    print("  TEMPEST Quick Validation")
    print("="*60)
    
    results = []
    results.append(("Imports", check_imports()))
    results.append(("GPU", check_gpu()))
    results.append(("Data Flow", test_data_flow()))
    
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ System ready for TEMPEST training!")
    else:
        print("\n⚠ Some checks failed - review output above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
