#!/usr/bin/env python3
"""
Example usage of Tempest Phase 1 components.

This demonstrates how to use the configuration system and I/O utilities
that have been implemented so far.
"""

import sys
sys.path.insert(0, '.')

from tempest.utils import (
    load_config,
    load_pwm,
    load_acc_priors,
    load_barcodes,
    get_base_to_index,
    TempestConfig,
    ModelConfig,
    LengthConstraints
)

def example_1_load_config():
    """Example 1: Load and inspect configuration from YAML."""
    print("\n" + "="*60)
    print("Example 1: Loading Configuration")
    print("="*60)
    
    # Load configuration
    config = load_config('example_config.yaml')
    
    # Access model parameters
    print(f"\nModel Configuration:")
    print(f"  Embedding dim: {config.model.embedding_dim}")
    print(f"  CNN filters: {config.model.cnn_filters}")
    print(f"  LSTM units: {config.model.lstm_units}")
    print(f"  Max sequence length: {config.model.max_seq_len}")
    
    # Access length constraints
    if config.model.length_constraints:
        print(f"\nLength Constraints:")
        for label, (min_len, max_len) in config.model.length_constraints.constraints.items():
            print(f"  {label}: {min_len}-{max_len} bases")
    
    # Access training parameters
    if config.training:
        print(f"\nTraining Configuration:")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Batch size: {config.model.batch_size}")
    
    return config


def example_2_create_config_programmatically():
    """Example 2: Create configuration programmatically."""
    print("\n" + "="*60)
    print("Example 2: Creating Configuration Programmatically")
    print("="*60)
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=5,
        embedding_dim=64,
        lstm_units=64,
        num_labels=8,
        length_constraints=LengthConstraints(
            constraints={
                'UMI': (8, 8),
                'BARCODE': (16, 16)
            },
            constraint_weight=3.0
        )
    )
    
    # Create full config
    config = TempestConfig(model=model_config)
    
    print(f"\nCreated config:")
    print(f"  Embedding dim: {config.model.embedding_dim}")
    print(f"  LSTM units: {config.model.lstm_units}")
    print(f"  Constraints: {config.model.length_constraints.constraints}")
    
    # Save to file
    config.to_yaml('my_custom_config.yaml')
    print(f"\n✓ Saved to 'my_custom_config.yaml'")
    
    return config


def example_3_load_data_files():
    """Example 3: Load various data files."""
    print("\n" + "="*60)
    print("Example 3: Loading Data Files")
    print("="*60)
    
    # Example with dummy data (you can replace with real files)
    
    # 1. Base encoding
    base_to_idx = get_base_to_index()
    print(f"\nBase encoding:")
    for base, idx in base_to_idx.items():
        print(f"  {base} → {idx}")
    
    # 2. Load PWM (if file exists)
    try:
        pwm = load_pwm('acc_pwm.txt')
        print(f"\n✓ Loaded PWM with shape: {pwm.shape}")
        print(f"  First position probabilities: {pwm[0]}")
    except FileNotFoundError:
        print(f"\n⚠ PWM file not found (expected for this demo)")
    
    # 3. Load ACC priors (if file exists)
    try:
        sequences, frequencies = load_acc_priors('acc_priors.tsv', 'model1')
        print(f"\n✓ Loaded {len(sequences)} ACC sequences")
        print(f"  First 3 sequences: {sequences[:3]}")
        print(f"  First 3 frequencies: {frequencies[:3]}")
    except (FileNotFoundError, ValueError):
        print(f"\n⚠ ACC priors file not found (expected for this demo)")
    
    # 4. Load barcodes (if file exists)
    try:
        barcodes = load_barcodes('barcodes.txt')
        print(f"\n✓ Loaded {len(barcodes)} barcodes")
        print(f"  First 3 barcodes: {barcodes[:3]}")
    except FileNotFoundError:
        print(f"\n⚠ Barcodes file not found (expected for this demo)")


def example_4_modify_and_save_config():
    """Example 4: Load, modify, and save configuration."""
    print("\n" + "="*60)
    print("Example 4: Modifying Configuration")
    print("="*60)
    
    # Load existing config
    config = load_config('example_config.yaml')
    
    print(f"\nOriginal learning rate: {config.training.learning_rate}")
    print(f"Original epochs: {config.training.epochs}")
    
    # Modify parameters
    config.training.learning_rate = 0.0005
    config.training.epochs = 30
    config.model.batch_size = 64
    
    print(f"\nModified learning rate: {config.training.learning_rate}")
    print(f"Modified epochs: {config.training.epochs}")
    print(f"Modified batch size: {config.model.batch_size}")
    
    # Save modified config
    config.to_yaml('modified_config.yaml')
    print(f"\n✓ Saved modified config to 'modified_config.yaml'")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "TEMPEST USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_load_config()
    example_2_create_config_programmatically()
    example_3_load_data_files()
    example_4_modify_and_save_config()
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Customize 'example_config.yaml' for your experiment")
    print("  2. Wait for Phase 2: Model architecture implementation")
    print("  3. Then you'll be able to train models and run inference!")
    print()


if __name__ == '__main__':
    main()
