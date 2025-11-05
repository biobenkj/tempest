#!/usr/bin/env python3
"""
Simplified test script for verifying single model and ensemble trainers.

This version has minimal dependencies and focuses on testing the core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Direct imports to avoid circular dependencies
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class SimulatedRead:
    """Container for a simulated read with labels."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict = field(default_factory=dict)  # Additional info


def reads_to_arrays(reads: List[SimulatedRead], 
                    label_to_idx: Optional[Dict[str, int]] = None,
                    max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Convert list of SimulatedRead objects to numpy arrays for training."""
    if not reads:
        raise ValueError("No reads provided")
    
    # Build or use label mapping
    if label_to_idx is None:
        # Collect all unique labels
        unique_labels = set()
        for read in reads:
            unique_labels.update(read.labels)
        
        # Sort for consistency
        sorted_labels = sorted(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        logger.info(f"Created label mapping with {len(label_to_idx)} labels: {sorted_labels}")
    
    # Base encoding
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                   'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
    
    # Find max length if not specified
    if max_length is None:
        max_length = max(len(read.sequence) for read in reads)
    
    # Initialize arrays
    num_reads = len(reads)
    X = np.zeros((num_reads, max_length), dtype=np.int32)
    y = np.zeros((num_reads, max_length), dtype=np.int32)
    
    # Convert each read
    for i, read in enumerate(reads):
        seq_len = min(len(read.sequence), max_length)
        
        # Encode sequence
        for j in range(seq_len):
            base = read.sequence[j]
            X[i, j] = base_to_idx.get(base, 4)  # Default to 'N' index
        
        # Encode labels
        for j in range(min(len(read.labels), max_length)):
            label = read.labels[j]
            if label in label_to_idx:
                y[i, j] = label_to_idx[label]
            else:
                logger.warning(f"Unknown label '{label}' found, skipping")
    
    return X, y, label_to_idx

def create_mock_reads(num_reads=100, seq_length=200):
    """Create mock SimulatedRead objects for testing."""
    reads = []
    
    # Define simple labels
    labels_list = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'POLYA']
    
    for i in range(num_reads):
        # Generate random sequence
        bases = np.random.choice(['A', 'C', 'G', 'T'], seq_length)
        sequence = ''.join(bases)
        
        # Generate simple label pattern
        labels = []
        label_regions = {}
        
        # Simple fixed pattern for testing
        segments = [
            ('ADAPTER', 8),
            ('UMI', 8),
            ('ACC', 6),
            ('BARCODE', 16),
            ('INSERT', seq_length - 8 - 8 - 6 - 16 - 20),
            ('POLYA', 20)
        ]
        
        pos = 0
        for label, length in segments:
            if pos + length <= seq_length:
                labels.extend([label] * length)
                if label not in label_regions:
                    label_regions[label] = []
                label_regions[label].append((pos, pos + length))
                pos += length
        
        # Pad labels if necessary
        while len(labels) < seq_length:
            labels.append('INSERT')
        
        read = SimulatedRead(
            sequence=sequence[:len(labels)],
            labels=labels[:len(sequence)],
            label_regions=label_regions,
            metadata={'read_id': f'read_{i}'}
        )
        reads.append(read)
    
    return reads


def test_reads_to_arrays():
    """Test the reads_to_arrays function."""
    logger.info("\n" + "="*60)
    logger.info("Testing reads_to_arrays function")
    logger.info("="*60)
    
    # Create mock reads
    reads = create_mock_reads(10, 100)
    
    # Convert to arrays
    X, y, label_to_idx = reads_to_arrays(reads)
    
    logger.info(f"Input shape: {X.shape}")
    logger.info(f"Label shape: {y.shape}")
    logger.info(f"Label mapping: {label_to_idx}")
    
    # Verify shapes
    assert X.shape[0] == 10, "Wrong number of samples"
    assert X.shape[1] == 100, "Wrong sequence length"
    assert y.shape == X.shape, "Label shape doesn't match input shape"
    
    # Verify encoding
    assert np.all((X >= 0) & (X <= 4)), "Invalid base encoding"
    assert len(label_to_idx) > 0, "No labels found"
    
    logger.info("✓ reads_to_arrays test passed")
    return True


def test_trainer_imports():
    """Test that trainer modules can be imported."""
    logger.info("\n" + "="*60)
    logger.info("Testing trainer module imports")
    logger.info("="*60)
    
    try:
        # Import trainer modules
        from tempest.training.trainer import ModelTrainer, PerTokenAccuracy, PerLabelMetrics
        logger.info("✓ Successfully imported ModelTrainer and metrics")
        
        from tempest.training.ensemble import EnsembleTrainer, BMAPredictor
        logger.info("✓ Successfully imported EnsembleTrainer and BMAPredictor")
        
        from tempest.training.hybrid_trainer import HybridTrainer
        logger.info("✓ Successfully imported HybridTrainer")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import trainer modules: {e}")
        return False


def test_trainer_initialization():
    """Test trainer initialization without TensorFlow."""
    logger.info("\n" + "="*60)
    logger.info("Testing trainer initialization")
    logger.info("="*60)
    
    try:
        # Create a mock config object
        class MockConfig:
            class model:
                vocab_size = 5
                embedding_dim = 64
                use_cnn = True
                cnn_filters = [32, 64]
                cnn_kernels = [3, 5]
                use_bilstm = True
                lstm_units = 64
                lstm_layers = 1
                dropout = 0.3
                num_labels = 6
                use_crf = False
                max_seq_len = 256
                batch_size = 16
            
            class training:
                epochs = 5
                learning_rate = 0.001
                early_stopping_patience = 3
                reduce_lr_patience = 2
            
            class ensemble:
                num_models = 3
                variation_type = 'both'
                bma_prior = 'performance'
                bma_temperature = 1.0
        
        config = MockConfig()
        
        # Test ModelTrainer initialization
        from tempest.training.trainer import ModelTrainer
        trainer = ModelTrainer(config, checkpoint_dir="test_checkpoint")
        logger.info("✓ ModelTrainer initialized successfully")
        
        # Test EnsembleTrainer initialization
        from tempest.training.ensemble import EnsembleTrainer
        ensemble = EnsembleTrainer(config, num_models=3, checkpoint_dir="test_ensemble")
        logger.info("✓ EnsembleTrainer initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize trainers: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preparation():
    """Test data preparation functions."""
    logger.info("\n" + "="*60)
    logger.info("Testing data preparation functions")
    logger.info("="*60)
    
    try:
        from tempest.training.hybrid_trainer import pad_sequences, convert_labels_to_categorical
        
        # Test pad_sequences
        sequences = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).reshape(2, 3)
        labels = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32).reshape(2, 3)
        
        padded_seq, padded_lab = pad_sequences(sequences, labels, 5)
        
        assert padded_seq.shape == (2, 5), f"Wrong padded shape: {padded_seq.shape}"
        assert padded_lab.shape == (2, 5), f"Wrong padded label shape: {padded_lab.shape}"
        logger.info("✓ pad_sequences test passed")
        
        # Test convert_labels_to_categorical
        categorical = convert_labels_to_categorical(labels, 3)
        assert categorical.shape == (2, 3, 3), f"Wrong categorical shape: {categorical.shape}"
        assert np.allclose(np.sum(categorical, axis=-1), 1.0), "Categorical doesn't sum to 1"
        logger.info("✓ convert_labels_to_categorical test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"Data preparation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_building():
    """Test model building if TensorFlow is available."""
    logger.info("\n" + "="*60)
    logger.info("Testing model building")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        from tempest.training.hybrid_trainer import build_model_from_config
        
        # Create mock config
        class MockConfig:
            class model:
                vocab_size = 5
                embedding_dim = 32
                use_cnn = True
                cnn_filters = [16, 32]
                cnn_kernels = [3, 5]
                use_bilstm = True
                lstm_units = 32
                lstm_layers = 1
                dropout = 0.2
                num_labels = 6
                use_crf = False
                max_seq_len = 100
                batch_size = 8
        
        config = MockConfig()
        
        # Build model
        model = build_model_from_config(config)
        
        # Verify model structure
        assert model.input_shape == (None, 100), f"Wrong input shape: {model.input_shape}"
        assert model.output_shape == (None, 100, 6), f"Wrong output shape: {model.output_shape}"
        
        logger.info("✓ Model building test passed")
        logger.info(f"  Input shape: {model.input_shape}")
        logger.info(f"  Output shape: {model.output_shape}")
        logger.info(f"  Total parameters: {model.count_params()}")
        
        return True
        
    except ImportError:
        logger.warning("TensorFlow not available, skipping model building test")
        return True
    except Exception as e:
        logger.error(f"Model building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting Simplified Tempest Trainer Tests")
    logger.info("="*80)
    
    results = []
    
    # Run tests
    results.append(("reads_to_arrays", test_reads_to_arrays()))
    results.append(("trainer_imports", test_trainer_imports()))
    results.append(("trainer_initialization", test_trainer_initialization()))
    results.append(("data_preparation", test_data_preparation()))
    results.append(("model_building", test_model_building()))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:30s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    # Save summary
    summary = {
        'tests_run': len(results),
        'passed': passed,
        'failed': failed,
        'results': {name: result for name, result in results}
    }
    
    with open('test_summary_simple.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if failed == 0:
        logger.info("\n✓ ALL TESTS PASSED")
        return 0
    else:
        logger.info(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
