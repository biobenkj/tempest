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
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add paths - need to add before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))  # tempest root
sys.path.insert(0, '/home/claude/tempest')
sys.path.insert(0, '/mnt/project')

# Now import the module
try:
    from length_constrained_crf_vectorized import (
        ModelWithLengthConstrainedCRF,
        create_length_constrained_model
    )
except ModuleNotFoundError:
    # If running from different location, try alternative import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "length_constrained_crf_vectorized", 
        "/mnt/project/length_constrained_crf_vectorized.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["length_constrained_crf_vectorized"] = module
        spec.loader.exec_module(module)
        ModelWithLengthConstrainedCRF = module.ModelWithLengthConstrainedCRF
        create_length_constrained_model = module.create_length_constrained_model
    else:
        raise


class TestConstraintWeightRamping:
    """Test constraint weight ramping functionality."""
    
    def setup_method(self):
        """Setup mock model for testing."""
        self.mock_base_model = Mock()
        self.mock_base_model.layers = []
        
        self.length_constraints = {
            'UMI': (8, 8),
            'ACC': (6, 6)
        }
        
        self.label_to_idx = {
            'ADAPTER': 0,
            'UMI': 1,
            'ACC': 2,
            'BARCODE': 3,
            'INSERT': 4
        }
    
    def test_initial_weight_is_zero(self):
        """Test that constraint weight starts at zero."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(self.label_to_idx.keys()))
        )
        
        # At epoch 0, weight should be 0
        model.current_epoch.assign(0)
        current_weight = model.compute_constraint_weight()
        
        assert float(current_weight) == 0.0
    
    def test_weight_ramps_linearly(self):
        """Test that weight increases linearly during ramp period."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(self.label_to_idx.keys()))
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
            assert np.isclose(actual_weight, expected_weight, atol=1e-5)
    
    def test_weight_caps_at_maximum(self):
        """Test that weight doesn't exceed maximum value."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(self.label_to_idx.keys()))
        )
        
        # Test beyond ramp period
        for epoch in [5, 10, 20, 100]:
            model.current_epoch.assign(epoch)
            actual_weight = float(model.compute_constraint_weight())
            assert actual_weight == 10.0
    
    def test_no_ramping_when_ramp_epochs_zero(self):
        """Test immediate full weight when ramp_epochs = 0."""
        model = ModelWithLengthConstrainedCRF(
            base_model=self.mock_base_model,
            length_constraints=self.length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(self.label_to_idx.keys()))
        )
        
        # Should be full weight immediately
        model.current_epoch.assign(0)
        assert float(model.compute_constraint_weight()) == 10.0


class TestLengthPenaltyComputation:
    """Test length penalty computation."""
    
    def create_test_predictions(self, sequence_labels, seq_len):
        """Helper to create test prediction sequences."""
        # Create one-hot encoded predictions
        num_labels = 5
        predictions = np.zeros((1, seq_len, num_labels), dtype=np.float32)
        
        for i, label_idx in enumerate(sequence_labels[:seq_len]):
            predictions[0, i, label_idx] = 1.0
        
        return tf.constant(predictions)
    
    def test_penalty_for_correct_length(self):
        """Test that correct lengths have zero penalty."""
        # Segment with correct length: UMI should be 8 bases
        # Label indices: 0=ADAPTER, 1=UMI, 2=ACC, 3=BARCODE, 4=INSERT
        sequence = [0, 0, 0] + [1]*8 + [2]*6 + [3]*16 + [4]*20
        predictions = self.create_test_predictions(sequence, len(sequence))
        
        length_constraints = {'UMI': (8, 8), 'ACC': (6, 6), 'BARCODE': (16, 16)}
        label_to_idx = {'ADAPTER': 0, 'UMI': 1, 'ACC': 2, 'BARCODE': 3, 'INSERT': 4}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Compute penalty
        penalty = model.compute_length_penalty(tf.argmax(predictions, axis=-1))
        
        # Should be zero or very close for correct lengths
        assert float(penalty) < 0.01
    
    def test_penalty_for_too_short(self):
        """Test that segments shorter than min have positive penalty."""
        # UMI should be 8 but is only 5
        sequence = [0, 0, 0] + [1]*5 + [2]*10
        predictions = self.create_test_predictions(sequence, len(sequence))
        
        length_constraints = {'UMI': (8, 8)}
        label_to_idx = {'ADAPTER': 0, 'UMI': 1, 'ACC': 2}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        penalty = float(model.compute_length_penalty(tf.argmax(predictions, axis=-1)))
        
        # Should have positive penalty (8-5)^2 = 9
        assert penalty > 5.0
    
    def test_penalty_for_too_long(self):
        """Test that segments longer than max have positive penalty."""
        # UMI should be 8 but is 12
        sequence = [0, 0, 0] + [1]*12 + [2]*10
        predictions = self.create_test_predictions(sequence, len(sequence))
        
        length_constraints = {'UMI': (8, 8)}
        label_to_idx = {'ADAPTER': 0, 'UMI': 1, 'ACC': 2}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        penalty = float(model.compute_length_penalty(tf.argmax(predictions, axis=-1)))
        
        # Should have positive penalty (12-8)^2 = 16
        assert penalty > 10.0


class TestSegmentDetection:
    """Test segment detection and boundary identification."""
    
    def test_simple_segment_detection(self):
        """Test detection of simple consecutive segments."""
        # Create prediction sequence: AAA UUU CCC
        predictions = tf.constant([[0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=tf.int32)
        
        length_constraints = {'U': (3, 3), 'C': (3, 3)}
        label_to_idx = {'A': 0, 'U': 1, 'C': 2}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # The model should correctly identify 3 segments
        # This is tested indirectly through the penalty calculation
        penalty = float(model.compute_length_penalty(predictions))
        assert penalty < 1.0  # All segments are correct length
    
    def test_multiple_same_label_segments(self):
        """Test handling of multiple segments with same label."""
        # AAA UUU AAA UUU - two ADAPTER segments, two UMI segments
        predictions = tf.constant([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=tf.int32)
        
        length_constraints = {'A': (3, 3), 'U': (3, 3)}
        label_to_idx = {'A': 0, 'U': 1}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        penalty = float(model.compute_length_penalty(predictions))
        # Both U segments should be validated
        assert penalty < 1.0


class TestVectorizedOperations:
    """Test vectorized implementations for performance."""
    
    def test_batch_processing(self):
        """Test that model handles batches correctly."""
        batch_size = 4
        seq_len = 100
        num_labels = 5
        
        # Create batch of predictions
        predictions = tf.constant(
            np.random.randint(0, num_labels, size=(batch_size, seq_len)),
            dtype=tf.int32
        )
        
        length_constraints = {'U': (8, 8)}
        label_to_idx = {'A': 0, 'U': 1, 'C': 2, 'B': 3, 'I': 4}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Should process entire batch
        penalty = model.compute_length_penalty(predictions)
        
        # Should return scalar penalty (averaged over batch)
        assert penalty.shape == ()
    
    def test_xla_compatibility(self):
        """Test that operations are XLA compatible."""
        predictions = tf.constant([[0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]], dtype=tf.int32)
        
        length_constraints = {'U': (8, 8)}
        label_to_idx = {'A': 0, 'U': 1, 'C': 2}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=1.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Wrap in tf.function with XLA compilation
        @tf.function(jit_compile=True)
        def compute_with_xla(preds):
            return model.compute_length_penalty(preds)
        
        try:
            penalty = compute_with_xla(predictions)
            assert penalty.dtype == tf.float32
        except Exception as e:
            pytest.fail(f"XLA compilation failed: {e}")


class TestReproducibility:
    """Test reproducibility of length constraint computations."""
    
    def test_deterministic_penalty_computation(self):
        """Test that penalty computation is deterministic."""
        sequence = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        predictions = tf.constant([sequence], dtype=tf.int32)
        
        length_constraints = {'U': (8, 8), 'A': (6, 6)}
        label_to_idx = {'O': 0, 'U': 1, 'A': 2}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=5.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Compute penalty multiple times
        penalties = [float(model.compute_length_penalty(predictions)) for _ in range(5)]
        
        # All should be identical
        assert all(np.isclose(p, penalties[0]) for p in penalties)
    
    def test_reproducible_constraint_arrays(self):
        """Test that constraint arrays are consistently initialized."""
        length_constraints = {'UMI': (8, 8), 'ACC': (6, 6), 'BARCODE': (16, 16)}
        label_to_idx = {'ADAPTER': 0, 'UMI': 1, 'ACC': 2, 'BARCODE': 3, 'INSERT': 4}
        
        # Create two models with same config
        model1 = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=5.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        model2 = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=5.0,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Constraint arrays should be identical
        assert tf.reduce_all(model1.min_lengths == model2.min_lengths)
        assert tf.reduce_all(model1.max_lengths == model2.max_lengths)
        assert tf.reduce_all(model1.has_constraint == model2.has_constraint)


class TestConvenienceFunction:
    """Test the convenience wrapper function."""
    
    def test_create_length_constrained_model(self):
        """Test model creation via convenience function."""
        base_model = Mock()
        base_model.layers = []
        
        length_constraints = {'UMI': (8, 8), 'ACC': (6, 6)}
        
        model = create_length_constrained_model(
            base_model=base_model,
            length_constraints=length_constraints,
            constraint_weight=5.0,
            max_seq_len=256,
            constraint_ramp_epochs=5,
            label_binarizer=Mock(classes_=['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT'])
        )
        
        assert model.max_constraint_weight == 5.0
        assert model.constraint_ramp_epochs == 5
        assert model.max_seq_len == 256


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_training_simulation(self):
        """Simulate a full training scenario with ramping and penalties."""
        length_constraints = {'UMI': (8, 8), 'ACC': (6, 6)}
        label_to_idx = {'ADAPTER': 0, 'UMI': 1, 'ACC': 2, 'BARCODE': 3, 'INSERT': 4}
        
        model = ModelWithLengthConstrainedCRF(
            base_model=Mock(layers=[]),
            length_constraints=length_constraints,
            constraint_weight=10.0,
            constraint_ramp_epochs=5,
            max_seq_len=256,
            label_binarizer=Mock(classes_=list(label_to_idx.keys()))
        )
        
        # Simulate epochs
        penalties_over_epochs = []
        
        # Create a sequence with slightly wrong UMI length (7 instead of 8)
        sequence = [0, 0, 0] + [1]*7 + [2]*6 + [3]*10 + [4]*20
        predictions = tf.constant([sequence], dtype=tf.int32)
        
        for epoch in range(10):
            model.current_epoch.assign(epoch)
            current_weight = float(model.compute_constraint_weight())
            penalty = float(model.compute_length_penalty(predictions))
            weighted_penalty = penalty * current_weight
            
            penalties_over_epochs.append({
                'epoch': epoch,
                'weight': current_weight,
                'penalty': penalty,
                'weighted_penalty': weighted_penalty
            })
        
        # Verify weight increases as expected
        assert penalties_over_epochs[0]['weight'] == 0.0
        assert penalties_over_epochs[5]['weight'] == 10.0
        
        # Verify weighted penalty increases during ramp
        assert penalties_over_epochs[0]['weighted_penalty'] < penalties_over_epochs[3]['weighted_penalty']
        
        # Verify weighted penalty is constant after ramp
        assert np.isclose(
            penalties_over_epochs[5]['weighted_penalty'],
            penalties_over_epochs[9]['weighted_penalty'],
            atol=0.001
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
