#!/usr/bin/env python3
"""
Test script for Tempest simulator with ACC PWM integration.
"""

import sys
sys.path.insert(0, '/home/claude')

from tempest.utils import load_config, load_pwm
from tempest.data import SequenceSimulator
from tempest.core import PWMScorer

def test_pwm_loading():
    """Test PWM loading and scoring."""
    print("\n" + "="*60)
    print("TEST 1: PWM Loading and Scoring")
    print("="*60)
    
    # Load PWM
    pwm = load_pwm('acc_pwm.txt')
    print(f"✓ Loaded PWM with shape: {pwm.shape}")
    
    # Create scorer
    scorer = PWMScorer(pwm, threshold=0.7)
    print(f"✓ Created PWM scorer with threshold 0.7")
    
    # Test scoring some sequences
    test_sequences = [
        'ACCGGG',  # Good match (follows IUPAC ACCSSV pattern)
        'ACCGGC',  # Good match
        'ACCGCG',  # Good match
        'TTTTTT',  # Bad match
        'AAAAAA',  # Bad match
    ]
    
    print("\nTesting sequence scoring:")
    for seq in test_sequences:
        score = scorer.score_sequence(seq)
        print(f"  {seq}: {score:.3f} {'✓' if score >= 0.7 else '✗'}")
    
    return scorer


def test_simulator():
    """Test sequence simulator with PWM."""
    print("\n" + "="*60)
    print("TEST 2: Sequence Simulation with PWM")
    print("="*60)
    
    # Load config
    config = load_config('train_config.yaml')
    print("✓ Loaded configuration")
    
    # Create simulator with PWM
    simulator = SequenceSimulator(config.simulation, pwm_file='acc_pwm.txt')
    print("✓ Created simulator")
    
    # Generate a few reads
    print("\nGenerating 10 test reads...")
    reads = simulator.generate(num_sequences=10)
    print(f"✓ Generated {len(reads)} reads")
    
    # Examine first read
    print("\nFirst read details:")
    read = reads[0]
    print(f"  Length: {len(read.sequence)} bp")
    print(f"  Labels: {set(read.labels)}")
    print(f"  Label regions: {read.label_regions}")
    print(f"\n  Sequence (first 100 bp):")
    print(f"  {read.sequence[:100]}")
    
    # Check ACC sequences
    print("\nACC sequences generated:")
    for i, read in enumerate(reads[:5]):
        if 'ACC' in read.label_regions:
            for start, end in read.label_regions['ACC']:
                acc_seq = read.sequence[start:end]
                print(f"  Read {i+1}: {acc_seq}")
    
    return reads


def test_data_conversion():
    """Test converting reads to arrays."""
    print("\n" + "="*60)
    print("TEST 3: Data Conversion")
    print("="*60)
    
    from tempest.data import reads_to_arrays
    
    # Load config and generate data
    config = load_config('train_config.yaml')
    simulator = SequenceSimulator(config.simulation, pwm_file='acc_pwm.txt')
    reads = simulator.generate(num_sequences=100)
    
    # Convert to arrays
    X, y, label_to_idx = reads_to_arrays(reads)
    
    print(f"✓ Converted {len(reads)} reads to arrays")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Label mapping: {label_to_idx}")
    
    # Show sample
    print(f"\nSample encoding (first 20 bases):")
    print(f"  Bases: {X[0, :20]}")
    print(f"  Labels: {y[0, :20]}")
    
    return X, y, label_to_idx


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "TEMPEST SIMULATOR TEST")
    print("="*70)
    
    try:
        # Test 1: PWM
        scorer = test_pwm_loading()
        
        # Test 2: Simulation
        reads = test_simulator()
        
        # Test 3: Conversion
        X, y, label_to_idx = test_data_conversion()
        
        # Summary
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe simulator is working correctly!")
        print("You can now run main.py to train a model:")
        print("  python main.py --config train_config.yaml --pwm acc_pwm.txt")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
