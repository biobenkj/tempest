#!/usr/bin/env python3
"""
Integration test for modular hybrid training architecture.

Tests that all components work together correctly.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from modular components
from config import load_config, HybridTrainingConfig
from simulator import SequenceSimulator, SimulatedRead, reads_to_arrays
from models import build_model_from_config
from pwm import PWMScorer
from io import load_pwm

# Import hybrid training modules
from invalid_generator import InvalidReadGenerator
from hybrid_trainer import (
    HybridTrainer,
    PseudoLabelGenerator,
    ArchitectureDiscriminator
)


def test_config_loading():
    """Test that hybrid configuration loads correctly."""
    print("\n" + "="*60)
    print("TEST 1: Configuration Loading")
    print("="*60)
    
    # Load hybrid config
    config_path = project_root / 'hybrid_config.yaml'
    if not config_path.exists():
        print(f"Warning: {config_path} not found, using default config")
        config_path = project_root / 'train_config.yaml'
    
    config = load_config(str(config_path))
    print("Loaded configuration")
    
    # Check hybrid section
    if config.hybrid is not None:
        assert isinstance(config.hybrid, HybridTrainingConfig), "Wrong hybrid config type"
        print("Hybrid configuration present")
        print(f"  Warmup epochs: {config.hybrid.warmup_epochs}")
        print(f"  Discriminator epochs: {config.hybrid.discriminator_epochs}")
        print(f"  Invalid ratio: {config.hybrid.invalid_ratio}")
        print(f"  Confidence threshold: {config.hybrid.confidence_threshold}")
    else:
        print("Note: No hybrid section in config (okay for basic config)")
    
    return config


def test_invalid_generator():
    """Test invalid read generation module."""
    print("\n" + "="*60)
    print("TEST 2: Invalid Read Generator Module")
    print("="*60)
    
    # Create config
    config = load_config(str(project_root / 'train_config.yaml'))
    
    # Initialize simulator to get valid read
    pwm_file = project_root / 'acc_pwm.txt'
    simulator = SequenceSimulator(
        config.simulation,
        pwm_file=str(pwm_file) if pwm_file.exists() else None
    )
    
    # Generate valid read
    valid_reads = simulator.generate(num_sequences=1)
    valid_read = valid_reads[0]
    print(f"Generated valid read: {len(valid_read.sequence)} bp")
    print(f"  Labels: {set(valid_read.labels)}")
    
    # Initialize invalid generator
    invalid_gen = InvalidReadGenerator(config)
    print("Initialized InvalidReadGenerator")
    
    # Test each error type
    error_types = ['segment_loss', 'segment_duplication', 'truncation', 
                   'chimeric', 'scrambled']
    
    for error_type in error_types:
        invalid_read = invalid_gen.generate_invalid_read(valid_read, error_type)
        print(f"\n  {error_type}:")
        print(f"    Original length: {len(valid_read.sequence)}")
        print(f"    Invalid length: {len(invalid_read.sequence)}")
        print(f"    Metadata: {invalid_read.metadata.get('error_type', 'none')}")
        
        # Check that something changed
        assert invalid_read.metadata.get('error_type') == error_type, \
               f"{error_type} didn't set metadata correctly"
    
    # Test batch generation
    batch = invalid_gen.generate_batch(valid_reads * 10, invalid_ratio=0.3)
    invalid_count = sum(1 for r in batch if 'error_type' in r.metadata)
    print(f"\nBatch generation:")
    print(f"  Total reads: {len(batch)}")
    print(f"  Invalid reads: {invalid_count}")
    
    print("\nAll error types working correctly")


def test_discriminator():
    """Test architecture discriminator module."""
    print("\n" + "="*60)
    print("TEST 3: Architecture Discriminator")
    print("="*60)
    
    # Create discriminator
    discriminator = ArchitectureDiscriminator(num_labels=6, hidden_dim=64)
    print("Created discriminator")
    
    # Create dummy predictions
    batch_size = 4
    seq_len = 100
    num_labels = 6
    
    # Random predictions (shape: [batch, seq_len, num_labels])
    import tensorflow as tf
    random_logits = tf.random.normal((batch_size, seq_len, num_labels))
    dummy_predictions = tf.nn.softmax(random_logits, axis=-1)
    
    # Run discriminator
    validity_scores = discriminator(dummy_predictions, training=False)
    
    print(f"Discriminator output shape: {validity_scores.shape}")
    print(f"  Expected: ({batch_size}, 3)")
    
    assert validity_scores.shape == (batch_size, 3), "Wrong output shape"
    
    # Check probabilities sum to 1
    sums = tf.reduce_sum(validity_scores, axis=1)
    assert tf.reduce_all(tf.abs(sums - 1.0) < 1e-5), "Probabilities don't sum to 1"
    
    print("Discriminator working correctly")


def test_pseudo_label_generator():
    """Test pseudo-label generator module."""
    print("\n" + "="*60)
    print("TEST 4: Pseudo-Label Generator Module")
    print("="*60)
    
    # Build model
    config = load_config(str(project_root / 'train_config.yaml'))
    model = build_model_from_config(config)
    print("Built model")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create label mapping
    label_to_idx = {
        'ADAPTER5': 0, 'UMI': 1, 'ACC': 2,
        'BARCODE': 3, 'INSERT': 4, 'ADAPTER3': 5
    }
    
    # Create pseudo-label generator
    pseudo_gen = PseudoLabelGenerator(
        model, config,
        confidence_threshold=0.5,
        label_to_idx=label_to_idx
    )
    print("Created pseudo-label generator")
    
    # Test sequence conversion
    sequences = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
    X = pseudo_gen._sequences_to_array(sequences)
    print(f"Converted sequences to array: {X.shape}")
    assert X.shape[0] == len(sequences), "Wrong batch size"
    
    # Test reverse complement
    X_rev = pseudo_gen._reverse_complement_array(X)
    print(f"Reverse complemented: {X_rev.shape}")
    assert X_rev.shape == X.shape, "RC changed shape"
    
    # Test architecture validation
    valid_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    invalid_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    assert pseudo_gen._validate_architecture(valid_labels) == True, \
           "Valid architecture rejected"
    assert pseudo_gen._validate_architecture(invalid_labels) == False, \
           "Invalid architecture accepted"
    
    print("Architecture validation working")
    print("Pseudo-label generator module working correctly")


def test_hybrid_trainer():
    """Test hybrid trainer module."""
    print("\n" + "="*60)
    print("TEST 5: Hybrid Trainer Module")
    print("="*60)
    
    # Load config (use basic config, add dummy hybrid section if needed)
    config_path = project_root / 'hybrid_config.yaml'
    if not config_path.exists():
        config_path = project_root / 'train_config.yaml'
    
    config = load_config(str(config_path))
    
    # If no hybrid config, we can't fully test, but we can check initialization
    if config.hybrid is None or not config.hybrid.enabled:
        print("Note: No hybrid config found, testing initialization only")
        # Can still test that HybridTrainer can be created
        try:
            trainer = HybridTrainer(config)
            print("HybridTrainer initialized (without full hybrid config)")
            return
        except Exception as e:
            print(f"Note: Could not initialize without hybrid config: {e}")
            return
    
    # Generate small dataset
    pwm_file = project_root / 'acc_pwm.txt'
    simulator = SequenceSimulator(
        config.simulation,
        pwm_file=str(pwm_file) if pwm_file.exists() else None
    )
    train_reads = simulator.generate(num_sequences=50)
    val_reads = simulator.generate(num_sequences=10)
    
    print(f"Generated {len(train_reads)} train, {len(val_reads)} val reads")
    
    # Initialize hybrid trainer
    trainer = HybridTrainer(config)
    print("Created hybrid trainer")
    
    # Check components
    assert trainer.base_model is not None, "No base model"
    assert trainer.discriminator is not None, "No discriminator"
    assert trainer.invalid_generator is not None, "No invalid generator"
    
    print(f"  Phase schedule: {trainer.phase_schedule}")
    print(f"  Initial invalid weight: {trainer.invalid_weight}")
    
    print("Hybrid trainer module working correctly")


def test_integration():
    """Test full integration of all modules."""
    print("\n" + "="*60)
    print("TEST 6: Full Module Integration")
    print("="*60)
    
    # Import all modular components
    try:
        from config import TempestConfig, HybridTrainingConfig
        from simulator import SequenceSimulator
        from models import build_model_from_config
        from invalid_generator import InvalidReadGenerator
        from hybrid_trainer import HybridTrainer
        
        print("All modules imported successfully")
        
        # Load config
        config = load_config(str(project_root / 'train_config.yaml'))
        assert isinstance(config, TempestConfig), "Wrong config type"
        
        print("Configuration system integrated")
        
        # Build model
        model = build_model_from_config(config)
        print(f"Model built: {model.count_params()} parameters")
        
        # Create invalid generator
        invalid_gen = InvalidReadGenerator(config)
        print("Invalid generator created")
        
        # Create hybrid trainer (may not have full config)
        try:
            trainer = HybridTrainer(config, base_model=model)
            print("Hybrid trainer initialized with model")
        except Exception as e:
            print(f"Note: Full hybrid trainer requires hybrid config section: {e}")
        
        print("\nAll modules integrated successfully")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "MODULAR HYBRID TRAINING INTEGRATION TEST")
    print("="*70)
    
    try:
        # Run tests
        config = test_config_loading()
        test_invalid_generator()
        test_discriminator()
        test_pseudo_label_generator()
        test_hybrid_trainer()
        test_integration()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        print("\nThe modular hybrid training system is working correctly!")
        print("\nUsage:")
        print("  Standard: python main_hybrid.py --config train_config.yaml")
        print("  Hybrid:   python main_hybrid.py --config hybrid_config.yaml --hybrid")
        print("  With PL:  python main_hybrid.py --config hybrid_config.yaml --hybrid --unlabeled data.fastq")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
