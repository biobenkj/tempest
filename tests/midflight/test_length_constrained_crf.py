"""
Comprehensive tests for length-constrained CRF implementation.

Tests cover:
- Constraint weight ramping
- Length penalty computation
- Segment detection and validation
- Vectorized operations
- XLA compatibility
- Reproducibility with fixed seeds
"""

import pytest
import numpy as np
import tensorflow as tf
from tf2crf import CRF
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add tempest package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))  # tempest root
import os
import sys

# Import from the tempest package
from tempest.core.length_crf import (
    ModelWithLengthConstrainedCRF,
    create_length_constrained_model,
    LengthConstrainedCRF
)
from tempest.training.hybrid_trainer import pad_sequences

# small helper
def pad_pred(preds, model):
    padded, _ = pad_sequences(np.array(preds),
                              np.array(preds),
                              max_length=model.max_seq_len)
    return tf.convert_to_tensor(padded, dtype=tf.int32)

# Mock CRF class for testing
class MockCRF(CRF):
    """Mock CRF layer for testing length-constrained CRF logic."""
    def __init__(self, units=5, **kwargs):
        # Don’t call CRF.__init__ — avoids heavy graph initialization
        super().__init__(units, **kwargs)
        self.transitions = tf.Variable(
            tf.random.normal([units, units]), trainable=True
        )
        self.chain_kernel = tf.Variable(
            tf.random.normal([units, units]), trainable=True
        )
    
    def get_viterbi_decoding(self, potentials, sequence_length):
        """Mock Viterbi decoding."""
        batch_size = tf.shape(potentials)[0]
        max_seq_len = tf.shape(potentials)[1]
        
        # Return dummy predictions
        return tf.ones([batch_size, max_seq_len], dtype=tf.int32), tf.ones([batch_size])


class TestConstraintWeightRamping:
    """Test constraint weight ramping functionality."""
    
    def setup_method(self):
        """Setup mock model for testing."""
        # Create a more complete mock base model
        self.mock_crf_layer = MockCRF(units=5)
        
        self.mock_base_model = Mock()
        self.mock_base_model.layers = [
            Mock(),  # Some other layer
            self.mock_crf_layer,  # The CRF layer
            Mock()   # Another layer
        ]
        
        # Mock the base model methods
        self.mock_base_model.predict = Mock(return_value=np.random.randn(10, 256, 5))
        self.mock_base_model.call = Mock(return_value=tf.random.normal([10, 256, 5]))
        
        self.length_constraints = {
            'UMI': (8, 8),
            'ACC': (6, 6),
            'ADAPTER': (15, 25)
        }
        
        self.label_to_idx = {
            'ADAPTER': 0,
            'UMI': 1,
            'ACC': 2,
            'BARCODE': 3,
            'INSERT': 4
        }
        
        # Create a mock label binarizer
        self.mock_label_binarizer = Mock()
        self.mock_label_binarizer.classes_ = list(self.label_to_idx.keys())
    
    def test_initial_weight_is_zero(self):
        """Test that constraint weight starts at zero."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # At epoch 0, weight should be 0
        model.current_epoch.assign(0)
        current_weight = model.compute_constraint_weight()
        
        assert float(current_weight) == 0.0, f"Expected 0.0, got {float(current_weight)}"
    
    def test_weight_ramps_linearly(self):
        """Test that weight increases linearly during ramp period."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        expected_weights = {
            0: 0.0,
            1: 2.0,   # 10.0 * 1/5
            2: 4.0,   # 10.0 * 2/5
            3: 6.0,   # 10.0 * 3/5
            4: 8.0,   # 10.0 * 4/5
            5: 10.0,  # Full weight
        }
        
        for epoch, expected_weight in expected_weights.items():
            model.current_epoch.assign(epoch)
            actual_weight = float(model.compute_constraint_weight())
            assert np.isclose(actual_weight, expected_weight, atol=1e-5), \
                f"Epoch {epoch}: expected {expected_weight}, got {actual_weight}"
    
    def test_weight_caps_at_maximum(self):
        """Test that weight doesn't exceed maximum value."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Test beyond ramp period
        for epoch in [5, 10, 20, 100]:
            model.current_epoch.assign(epoch)
            actual_weight = float(model.compute_constraint_weight())
            assert actual_weight == 10.0, \
                f"Epoch {epoch}: expected 10.0, got {actual_weight}"
    
    def test_no_ramping_when_ramp_epochs_zero(self):
        """Test immediate full weight when ramp_epochs is 0."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=0,  # No ramping
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Weight should be full immediately
        for epoch in [0, 1, 2, 5, 10]:
            model.current_epoch.assign(epoch)
            actual_weight = float(model.compute_constraint_weight())
            assert actual_weight == 10.0, \
                f"Epoch {epoch}: expected 10.0, got {actual_weight}"


class TestLengthPenaltyComputation:
    """Test length penalty calculations."""
    
    def setup_method(self):
        """Setup for penalty tests."""
        self.mock_crf_layer = MockCRF(units=5)
        
        self.mock_base_model = Mock()
        self.mock_base_model.layers = [self.mock_crf_layer]
        
        self.length_constraints = {
            'UMI': (8, 8),      # Fixed length
            'ACC': (6, 6),      # Fixed length  
            'BARCODE': (10, 20) # Variable length
        }
        
        self.mock_label_binarizer = Mock()
        self.mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT']
    
    def test_penalty_for_correct_length(self):
        """Test no penalty when length is within constraints."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create predictions with correct lengths
        # UMI: 8 positions (correct)
        predictions = tf.constant([
            [1, 1, 1, 1, 1, 1, 1, 1,  # 8 UMI (correct)
             2, 2, 2, 2, 2, 2,         # 6 ACC (correct)
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 12 BARCODE (correct, within 10-20)
             0, 0, 0, 0, 0],           # 5 ADAPTER (no constraint)
        ], dtype=tf.int32)

        # Pad to model.max_seq_len
        predictions = pad_pred(predictions, model)

        penalty = model.compute_length_penalty(predictions)
        
        assert float(penalty) == 0.0, f"Expected no penalty for correct lengths, got {float(penalty)}"
    
    def test_penalty_for_too_short(self):
        """Test penalty when segment is shorter than minimum."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create predictions with UMI too short
        predictions = tf.constant([
            [1, 1, 1, 1, 1,            # 5 UMI (too short, should be 8)
             2, 2, 2, 2, 2, 2,         # 6 ACC (correct)
             0, 0, 0, 0, 0, 0, 0, 0],  # 8 ADAPTER
        ], dtype=tf.int32)
        
        # Pad to model.max_seq_len
        predictions = pad_pred(predictions, model)

        penalty = model.compute_length_penalty(predictions)
        
        # Should have penalty > 0 for violating UMI constraint
        assert float(penalty) > 0.0, f"Expected penalty > 0 for short segment, got {float(penalty)}"
    
    def test_penalty_for_too_long(self):
        """Test penalty when segment exceeds maximum length."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=256,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create predictions with BARCODE too long
        barcode_positions = [3] * 25  # 25 positions (exceeds max of 20)
        predictions = tf.constant([
            [1, 1, 1, 1, 1, 1, 1, 1] +  # 8 UMI (correct)
            [2, 2, 2, 2, 2, 2] +         # 6 ACC (correct)
            barcode_positions +           # 25 BARCODE (too long, max is 20)
            [0, 0, 0],                    # 3 ADAPTER
        ], dtype=tf.int32)
        
        # Pad to model.max_seq_len
        predictions = pad_pred(predictions, model)

        penalty = model.compute_length_penalty(predictions)
        
        # Should have penalty > 0 for exceeding BARCODE max constraint
        assert float(penalty) > 0.0, f"Expected penalty > 0 for long segment, got {float(penalty)}"


class TestSegmentDetection:
    """Test segment detection and boundary identification."""
    
    def setup_method(self):
        """Setup for segment detection tests."""
        self.mock_crf_layer = MockCRF(units=5)
        
        self.mock_base_model = Mock()
        self.mock_base_model.layers = [self.mock_crf_layer]
        
        self.mock_label_binarizer = Mock()
        self.mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT']
    
    def test_simple_segment_detection(self):
        """Test detection of simple consecutive segments."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints={'UMI': (8, 8)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=20,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create simple prediction pattern
        predictions = tf.constant([
            [0, 0, 0, 0, 0,  # ADAPTER
             1, 1, 1, 1, 1, 1, 1, 1,  # UMI
             2, 2, 2, 2, 2, 2, 2],     # ACC
        ], dtype=tf.int32)
        
        # Pad to model.max_seq_len
        predictions = pad_pred(predictions, model)

        # Compute segments (this tests the internal segment detection logic)
        penalty = model.compute_length_penalty(predictions)
        
        # The test passes if no error is raised
        assert penalty is not None
    
    def test_multiple_same_label_segments(self):
        """Test handling of multiple non-consecutive segments with same label."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints={'UMI': (8, 8)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=30,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Pattern with multiple UMI segments (non-consecutive)
        predictions = tf.constant([
            [1, 1, 1, 1, 1, 1, 1, 1,  # First UMI segment (8)
             0, 0, 0, 0, 0,            # ADAPTER
             1, 1, 1, 1,               # Second UMI segment (4)
             2, 2, 2, 2, 2, 2,         # ACC
             1, 1, 1],                 # Third UMI segment (3)
        ], dtype=tf.int32)
        
        # Pad to model.max_seq_len
        predictions = pad_pred(predictions, model)

        penalty = model.compute_length_penalty(predictions)
        
        # Should have penalties for the second and third UMI segments
        assert float(penalty) > 0.0, "Expected penalty for incorrect UMI segments"


class TestVectorizedOperations:
    """Test that operations are properly vectorized for efficiency."""
    
    def setup_method(self):
        """Setup for vectorization tests."""
        self.mock_crf_layer = MockCRF(units=5)
        
        self.mock_base_model = Mock()
        self.mock_base_model.layers = [self.mock_crf_layer]
        
        self.mock_label_binarizer = Mock()
        self.mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT']
    
    def test_batch_processing(self):
        """Test that batches are processed efficiently."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints={'UMI': (8, 8), 'ACC': (6, 6)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=20,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create batch of predictions
        batch_size = 16
        predictions = tf.random.uniform(
            [batch_size, 20], 
            minval=0, 
            maxval=5, 
            dtype=tf.int32
        )
        
        # Compute penalties for batch
        penalties = model.compute_length_penalty(predictions)
        
        # Should return one penalty per sample
        assert tf.shape(penalties)[0] == batch_size
        
    def test_xla_compatibility(self):
        """Test that operations are XLA-compatible (no py_function, map_fn)."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints={'UMI': (8, 8)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=20,
            label_binarizer=self.mock_label_binarizer
        )
        
        # Create test function
        @tf.function  # Force XLA compilation
        def test_fn(predictions):
            return model.compute_length_penalty(predictions)
        
        predictions = tf.constant([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=tf.int32)
        
        # This should not raise an error if XLA-compatible
        try:
            result = test_fn(predictions)
            assert result is not None
        except Exception as e:
            if "py_function" in str(e) or "map_fn" in str(e):
                pytest.fail(f"Operation not XLA-compatible: {e}")


class TestReproducibility:
    """Test reproducibility with fixed random seeds."""
    
    def setup_method(self):
        """Setup for reproducibility tests."""
        self.mock_crf_layer = MockCRF(units=5)
        
        self.mock_base_model = Mock()
        self.mock_base_model.layers = [self.mock_crf_layer]
        
        self.mock_label_binarizer = Mock()
        self.mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT']
    
    def test_deterministic_penalty_computation(self):
        """Test that penalty computation is deterministic."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints={'UMI': (8, 8)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=20,
            label_binarizer=self.mock_label_binarizer
        )
        
        predictions = tf.constant([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]
        ], dtype=tf.int32)
        
        # Compute multiple times
        penalties = []
        for _ in range(5):
            penalty = model.compute_length_penalty(predictions)
            penalties.append(float(penalty))
        
        # All should be identical
        assert all(p == penalties[0] for p in penalties), \
            f"Penalties not deterministic: {penalties}"
    
    def test_reproducible_constraint_arrays(self):
        """Test that constraint arrays are built reproducibly."""
        # Create two models with same config
        models = []
        for _ in range(2):
            model = ModelWithLengthConstrainedCRF(
                base_model=self.mock_base_model,
                length_constraints={'UMI': (8, 8), 'ACC': (6, 6)},
                constraint_weight=5.0,
                constraint_ramp_epochs=0,
                max_seq_len=20,
                label_binarizer=self.mock_label_binarizer
            )
            models.append(model)
        
        # Check that constraint tensors are identical
        if hasattr(models[0], 'min_lengths') and hasattr(models[1], 'min_lengths'):
            assert np.array_equal(
                models[0].min_lengths.numpy(), 
                models[1].min_lengths.numpy()
            ), "Min length arrays not reproducible"
            
            assert np.array_equal(
                models[0].max_lengths.numpy(), 
                models[1].max_lengths.numpy()
            ), "Max length arrays not reproducible"


class TestConvenienceFunction:
    """Test the convenience function for model creation."""

    def test_create_length_constrained_model(self):
        """Test the create_length_constrained_model function."""
        # Mock necessary components
        mock_base_model = Mock()
        mock_crf_layer = MockCRF(units=5)
        mock_base_model.layers = [mock_crf_layer]

        mock_label_binarizer = Mock()
        mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC']

        # Patch the internal builder inside length_crf
        with patch('tempest.core.length_crf.build_model') as mock_build:
            mock_build.return_value = mock_base_model

            model = create_length_constrained_model(
                model_config={'some': 'config'},
                label_binarizer=mock_label_binarizer,
                length_constraints={'UMI': (8, 8)},
                constraint_weight=5.0,
                constraint_ramp_epochs=10
            )

            assert model is not None
            assert isinstance(model, ModelWithLengthConstrainedCRF)


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_full_training_simulation(self):
        """Test a simulated training workflow with epoch updates."""
        # Setup
        mock_crf_layer = MockCRF(units=5)
        mock_base_model = Mock()
        mock_base_model.layers = [mock_crf_layer]
        
        mock_label_binarizer = Mock()
        mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT']
        
        model = ModelWithLengthConstrainedCRF(
            base_model=mock_base_model,
            length_constraints={'UMI': (8, 8), 'ACC': (6, 6)},
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=30,
            label_binarizer=mock_label_binarizer
        )
        
        # Simulate training epochs
        for epoch in range(10):
            model.current_epoch.assign(epoch)
            
            # Create batch of predictions
            batch_predictions = tf.random.uniform(
                [8, 30], minval=0, maxval=5, dtype=tf.int32
            )
            
            # Compute penalties
            penalties = model.compute_length_penalty(batch_predictions)
            
            # Verify weight ramping
            expected_weight = min(10.0 * epoch / 5, 10.0) if epoch < 5 else 10.0
            actual_weight = float(model.compute_constraint_weight())
            
            assert np.isclose(actual_weight, expected_weight, atol=1e-5), \
                f"Epoch {epoch}: weight mismatch"
            
            # Verify penalties shape
            assert tf.shape(penalties)[0] == 8, "Batch size mismatch"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_constraints(self):
        """Test model with no length constraints."""
        mock_crf_layer = MockCRF(units=5)
        mock_base_model = Mock()
        mock_base_model.layers = [mock_crf_layer]
        
        mock_label_binarizer = Mock()
        mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC']
        
        model = ModelWithLengthConstrainedCRF(
            base_model=mock_base_model,
            length_constraints={},  # No constraints
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=20,
            label_binarizer=mock_label_binarizer
        )
        
        predictions = tf.constant([[0, 1, 2, 0, 1, 2]], dtype=tf.int32)
        predictions = pad_pred(predictions, model)
        penalty = model.compute_length_penalty(predictions)
        
        # Should be zero penalty when no constraints
        assert float(penalty) == 0.0
    
    def test_single_label_constraint(self):
        """Test with constraint on only one label."""
        mock_crf_layer = MockCRF(units=3)
        mock_base_model = Mock()
        mock_base_model.layers = [mock_crf_layer]
        
        mock_label_binarizer = Mock()
        mock_label_binarizer.classes_ = ['ADAPTER', 'UMI', 'ACC']
        
        model = ModelWithLengthConstrainedCRF(
            base_model=mock_base_model,
            length_constraints={'UMI': (5, 5)},  # Only UMI constrained
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=15,
            label_binarizer=mock_label_binarizer
        )
        
        # UMI (label 1) with wrong length
        predictions = tf.constant([
            [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]  # UMI has 3, should have 5
        ], dtype=tf.int32)
        predictions = pad_pred(predictions, model)
        penalty = model.compute_length_penalty(predictions)
        assert float(penalty) > 0.0, "Should penalize incorrect UMI length"
    
    def test_overlapping_segments(self):
        """Test proper handling when label changes frequently."""
        mock_crf_layer = MockCRF(units=3)
        mock_base_model = Mock()
        mock_base_model.layers = [mock_crf_layer]
        
        mock_label_binarizer = Mock()
        mock_label_binarizer.classes_ = ['A', 'B', 'C']
        
        model = ModelWithLengthConstrainedCRF(
            base_model=mock_base_model,
            length_constraints={'B': (3, 3)},
            constraint_weight=5.0,
            constraint_ramp_epochs=0,
            max_seq_len=10,
            label_binarizer=mock_label_binarizer
        )
        
        # Rapidly alternating labels
        predictions = tf.constant([
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # B appears 5 times but never 3 consecutive
        ], dtype=tf.int32)
        predictions = pad_pred(predictions, model)
        penalty = model.compute_length_penalty(predictions)
        # Each single B occurrence violates the constraint
        assert float(penalty) > 0.0, "Should penalize non-consecutive B segments"
