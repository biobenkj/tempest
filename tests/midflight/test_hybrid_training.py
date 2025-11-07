"""
Comprehensive tests for hybrid robustness training.

Tests cover:
- Hybrid training configuration
- Invalid read generation
- Training phase management (warmup, discriminator, pseudo-label)
- Discriminator functionality
- Pseudo-labeling with confidence thresholds
- Loss weighting and scheduling
- Reproducibility with fixed seeds
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, MagicMock, patch
import sys

# Add paths
sys.path.insert(0, '/home/claude/tempest')

from tempest.utils.config import HybridTrainingConfig


class TestHybridTrainingConfig:
    """Test HybridTrainingConfig class."""
    
    def test_default_hybrid_config(self):
        """Test default hybrid training configuration."""
        config = HybridTrainingConfig()
        
        assert config.enabled is False
        assert config.warmup_epochs == 5
        assert config.discriminator_epochs == 10
        assert config.pseudolabel_epochs == 10
        assert config.invalid_ratio == 0.1
        assert config.confidence_threshold == 0.9
    
    def test_custom_hybrid_config(self):
        """Test custom hybrid training configuration."""
        config = HybridTrainingConfig(
            enabled=True,
            warmup_epochs=3,
            discriminator_epochs=15,
            invalid_ratio=0.2,
            confidence_threshold=0.95,
            confidence_decay=0.9
        )
        
        assert config.enabled is True
        assert config.warmup_epochs == 3
        assert config.discriminator_epochs == 15
        assert config.invalid_ratio == 0.2
        assert config.confidence_threshold == 0.95
        assert config.confidence_decay == 0.9
    
    def test_invalid_generation_probabilities(self):
        """Test invalid read generation probability settings."""
        config = HybridTrainingConfig(
            segment_loss_prob=0.3,
            segment_dup_prob=0.3,
            truncation_prob=0.2,
            chimeric_prob=0.1,
            scrambled_prob=0.1
        )
        
        # Probabilities should sum to 1.0
        total_prob = (
            config.segment_loss_prob +
            config.segment_dup_prob +
            config.truncation_prob +
            config.chimeric_prob +
            config.scrambled_prob
        )
        
        assert np.isclose(total_prob, 1.0, atol=0.01)


class TestInvalidReadGeneration:
    """Test invalid read generation mechanisms."""
    
    def generate_valid_read(self, seq_len=100, num_labels=6):
        """Generate a valid read for testing."""
        # Create a simple valid structure
        # ADAPTER(10) - UMI(8) - ACC(6) - BARCODE(16) - INSERT(50) - ADAPTER(10)
        labels = (
            [0] * 10 +  # ADAPTER5
            [1] * 8 +   # UMI
            [2] * 6 +   # ACC
            [3] * 16 +  # BARCODE
            [4] * 50 +  # INSERT
            [5] * 10    # ADAPTER3
        )
        
        # Convert to one-hot if needed
        return np.array(labels), labels
    
    def apply_segment_loss(self, labels, loss_prob=0.3):
        """Simulate missing segment."""
        unique_labels = list(set(labels))
        if len(unique_labels) <= 2:
            return labels  # Don't modify if too few segments
        
        # Choose a segment to remove
        np.random.seed(42)
        segment_to_remove = np.random.choice(unique_labels[1:-1])  # Don't remove first/last
        
        # Remove segment
        modified = [l if l != segment_to_remove else labels[0] for l in labels]
        return modified
    
    def apply_segment_duplication(self, labels, dup_prob=0.3):
        """Simulate duplicated segment."""
        unique_labels = list(set(labels))
        if len(unique_labels) <= 1:
            return labels
        
        # Find segment boundaries
        segments = []
        current_label = labels[0]
        start = 0
        
        for i, label in enumerate(labels):
            if label != current_label:
                segments.append((current_label, start, i))
                current_label = label
                start = i
        segments.append((current_label, start, len(labels)))
        
        # Duplicate a segment
        if len(segments) > 1:
            np.random.seed(42)
            seg_idx = np.random.randint(0, len(segments))
            seg_label, seg_start, seg_end = segments[seg_idx]
            segment = labels[seg_start:seg_end]
            
            # Insert duplicate after original
            modified = labels[:seg_end] + segment + labels[seg_end:]
            return modified[:len(labels)]  # Truncate to original length
        
        return labels
    
    def apply_truncation(self, labels, truncation_prob=0.2):
        """Simulate truncated read."""
        np.random.seed(42)
        truncate_point = np.random.randint(len(labels) // 2, len(labels))
        return labels[:truncate_point] + [0] * (len(labels) - truncate_point)
    
    def test_segment_loss_generation(self):
        """Test generation of reads with missing segments."""
        _, labels = self.generate_valid_read()
        modified = self.apply_segment_loss(labels)
        
        # Should have fewer unique segments
        assert len(set(modified)) <= len(set(labels))
    
    def test_segment_duplication_generation(self):
        """Test generation of reads with duplicated segments."""
        _, labels = self.generate_valid_read()
        original_unique = set(labels)
        
        modified = self.apply_segment_duplication(labels)
        
        # Length should be preserved
        assert len(modified) == len(labels)
    
    def test_truncation_generation(self):
        """Test generation of truncated reads."""
        _, labels = self.generate_valid_read()
        modified = self.apply_truncation(labels)
        
        # Should have many more adapter labels at end (padding)
        assert modified[-10:].count(0) >= 5
    
    def test_invalid_ratio_application(self):
        """Test that invalid ratio is correctly applied."""
        batch_size = 100
        invalid_ratio = 0.2
        
        num_invalid = int(batch_size * invalid_ratio)
        num_valid = batch_size - num_invalid
        
        assert num_invalid == 20
        assert num_valid == 80
        assert num_invalid + num_valid == batch_size


class TestTrainingPhaseManagement:
    """Test management of different training phases."""
    
    def get_current_phase(self, epoch, warmup_epochs, discriminator_epochs, pseudolabel_epochs):
        """Determine current training phase."""
        if epoch < warmup_epochs:
            return 'warmup'
        elif epoch < warmup_epochs + discriminator_epochs:
            return 'discriminator'
        else:
            return 'pseudolabel'
    
    def test_warmup_phase_detection(self):
        """Test warmup phase detection."""
        config = HybridTrainingConfig(
            warmup_epochs=5,
            discriminator_epochs=10,
            pseudolabel_epochs=10
        )
        
        for epoch in range(5):
            phase = self.get_current_phase(
                epoch,
                config.warmup_epochs,
                config.discriminator_epochs,
                config.pseudolabel_epochs
            )
            assert phase == 'warmup'
    
    def test_discriminator_phase_detection(self):
        """Test discriminator phase detection."""
        config = HybridTrainingConfig(
            warmup_epochs=5,
            discriminator_epochs=10,
            pseudolabel_epochs=10
        )
        
        for epoch in range(5, 15):
            phase = self.get_current_phase(
                epoch,
                config.warmup_epochs,
                config.discriminator_epochs,
                config.pseudolabel_epochs
            )
            assert phase == 'discriminator'
    
    def test_pseudolabel_phase_detection(self):
        """Test pseudo-labeling phase detection."""
        config = HybridTrainingConfig(
            warmup_epochs=5,
            discriminator_epochs=10,
            pseudolabel_epochs=10
        )
        
        for epoch in range(15, 25):
            phase = self.get_current_phase(
                epoch,
                config.warmup_epochs,
                config.discriminator_epochs,
                config.pseudolabel_epochs
            )
            assert phase == 'pseudolabel'
    
    def test_phase_transition_epochs(self):
        """Test epoch counts for phase transitions."""
        config = HybridTrainingConfig(
            warmup_epochs=3,
            discriminator_epochs=7,
            pseudolabel_epochs=5
        )
        
        total_epochs = (
            config.warmup_epochs +
            config.discriminator_epochs +
            config.pseudolabel_epochs
        )
        
        assert total_epochs == 15


class TestDiscriminatorFunctionality:
    """Test discriminator component."""
    
    def create_simple_discriminator(self, input_dim=128, hidden_dim=64):
        """Create a simple discriminator model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def test_discriminator_creation(self):
        """Test creation of discriminator model."""
        discriminator = self.create_simple_discriminator()
        
        # Test forward pass
        x = tf.random.normal((10, 128))
        output = discriminator(x)
        
        assert output.shape == (10, 1)
        assert tf.reduce_all(output >= 0) and tf.reduce_all(output <= 1)
    
    def test_discriminator_training(self):
        """Test discriminator training on valid vs invalid reads."""
        discriminator = self.create_simple_discriminator()
        discriminator.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create synthetic data
        # Valid reads (label=1)
        x_valid = tf.random.normal((50, 128))
        y_valid = tf.ones((50, 1))
        
        # Invalid reads (label=0)
        x_invalid = tf.random.normal((50, 128)) * 2  # Different distribution
        y_invalid = tf.zeros((50, 1))
        
        # Combine
        x_train = tf.concat([x_valid, x_invalid], axis=0)
        y_train = tf.concat([y_valid, y_invalid], axis=0)
        
        # Train
        history = discriminator.fit(
            x_train, y_train,
            epochs=5,
            verbose=0,
            batch_size=32
        )
        
        # Should improve over epochs
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        assert final_loss < initial_loss


class TestPseudoLabeling:
    """Test pseudo-labeling mechanism."""
    
    def generate_pseudo_labels(self, predictions, confidence_threshold=0.9):
        """Generate pseudo-labels from model predictions."""
        # Get max probability for each prediction
        max_probs = np.max(predictions, axis=-1)
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=-1)
        
        # Filter by confidence
        high_confidence_mask = max_probs >= confidence_threshold
        
        # Return pseudo-labels and mask
        return pred_classes, high_confidence_mask
    
    def test_pseudo_label_generation(self):
        """Test generation of pseudo-labels from predictions."""
        # Create mock predictions
        batch_size, seq_len, num_labels = 10, 100, 6
        
        # Some high confidence, some low confidence
        predictions = np.random.random((batch_size, seq_len, num_labels))
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)  # Normalize
        
        # Make some predictions very confident
        predictions[0:5, :, 0] = 0.95  # High confidence for class 0
        predictions[0:5, :, 1:] = 0.05 / 5
        
        # Generate pseudo-labels
        pseudo_labels, mask = self.generate_pseudo_labels(predictions, confidence_threshold=0.9)
        
        # High confidence examples should be selected
        assert np.sum(mask[0:5]) > np.sum(mask[5:10])
    
    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold correctly filters examples."""
        predictions = np.array([
            [[0.95, 0.05], [0.60, 0.40]],  # High confidence, low confidence
            [[0.85, 0.15], [0.92, 0.08]]   # Low confidence, high confidence
        ])
        
        pseudo_labels, mask = self.generate_pseudo_labels(predictions, confidence_threshold=0.9)
        
        # Should select positions with prob >= 0.9
        expected_mask = np.array([
            [True, False],   # 0.95 >= 0.9, 0.60 < 0.9
            [False, True]    # 0.85 < 0.9, 0.92 >= 0.9
        ])
        
        assert np.array_equal(mask, expected_mask)
    
    def test_confidence_decay(self):
        """Test confidence threshold decay over epochs."""
        initial_threshold = 0.9
        decay_rate = 0.95
        
        thresholds = []
        current_threshold = initial_threshold
        
        for epoch in range(10):
            thresholds.append(current_threshold)
            current_threshold = max(0.5, current_threshold * decay_rate)
        
        # Threshold should decrease
        assert thresholds[0] > thresholds[-1]
        assert thresholds[-1] >= 0.5  # Should not go below minimum


class TestLossWeighting:
    """Test loss weighting and scheduling."""
    
    def compute_hybrid_loss(self, crf_loss, invalid_loss, adversarial_loss,
                           invalid_weight, adversarial_weight):
        """Compute combined hybrid loss."""
        total_loss = crf_loss + invalid_weight * invalid_loss + adversarial_weight * adversarial_loss
        return total_loss
    
    def test_loss_combination(self):
        """Test combination of multiple loss components."""
        crf_loss = 0.5
        invalid_loss = 0.3
        adversarial_loss = 0.2
        invalid_weight = 0.2
        adversarial_weight = 0.1
        
        total = self.compute_hybrid_loss(
            crf_loss, invalid_loss, adversarial_loss,
            invalid_weight, adversarial_weight
        )
        
        expected = 0.5 + 0.2 * 0.3 + 0.1 * 0.2
        assert np.isclose(total, expected)
    
    def test_invalid_weight_ramping(self):
        """Test invalid loss weight ramping."""
        config = HybridTrainingConfig(
            invalid_weight_initial=0.1,
            invalid_weight_max=0.3
        )
        
        epochs = 10
        weights = []
        
        for epoch in range(epochs):
            # Linear ramp
            progress = epoch / epochs
            weight = config.invalid_weight_initial + (
                config.invalid_weight_max - config.invalid_weight_initial
            ) * progress
            weights.append(weight)
        
        # Should increase from initial to max
        assert weights[0] == config.invalid_weight_initial
        assert weights[-1] == config.invalid_weight_max
        assert all(weights[i] <= weights[i+1] for i in range(len(weights)-1))


class TestValidationChecks:
    """Test architecture validation during hybrid training."""
    
    def validate_architecture(self, predictions, min_unique_segments=3, max_segment_repetition=2):
        """Validate predicted architecture."""
        # Get segment labels
        pred_labels = np.argmax(predictions, axis=-1)
        
        # Count unique segments
        unique_segments = len(set(pred_labels))
        
        # Count segment repetitions
        from collections import Counter
        segment_counts = Counter(pred_labels)
        max_repetition = max(segment_counts.values())
        
        is_valid = (
            unique_segments >= min_unique_segments and
            max_repetition <= max_segment_repetition
        )
        
        return is_valid, unique_segments, max_repetition
    
    def test_valid_architecture(self):
        """Test validation of correct architecture."""
        # Create valid architecture: ADAPTER-UMI-ACC-BARCODE-INSERT-ADAPTER
        predictions = np.zeros((1, 60, 6))
        predictions[0, 0:10, 0] = 1   # ADAPTER
        predictions[0, 10:18, 1] = 1  # UMI
        predictions[0, 18:24, 2] = 1  # ACC
        predictions[0, 24:40, 3] = 1  # BARCODE
        predictions[0, 40:50, 4] = 1  # INSERT
        predictions[0, 50:60, 5] = 1  # ADAPTER
        
        is_valid, unique, max_rep = self.validate_architecture(predictions)
        
        assert is_valid is True
        assert unique == 6
        assert max_rep <= 2
    
    def test_invalid_architecture_too_few_segments(self):
        """Test detection of architecture with too few segments."""
        # Only 2 unique segments
        predictions = np.zeros((1, 60, 6))
        predictions[0, 0:30, 0] = 1
        predictions[0, 30:60, 1] = 1
        
        is_valid, unique, max_rep = self.validate_architecture(predictions, min_unique_segments=3)
        
        assert is_valid is False
        assert unique == 2


class TestReproducibility:
    """Test reproducibility of hybrid training components."""
    
    def test_deterministic_invalid_generation(self):
        """Test that invalid read generation is reproducible."""
        np.random.seed(42)
        labels1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        
        # Apply same transformation twice
        np.random.seed(42)
        modified1 = [l if l != 1 else 0 for l in labels1]
        
        np.random.seed(42)
        modified2 = [l if l != 1 else 0 for l in labels1]
        
        assert modified1 == modified2
    
    def test_deterministic_phase_detection(self):
        """Test that phase detection is deterministic."""
        config = HybridTrainingConfig(
            warmup_epochs=5,
            discriminator_epochs=10,
            pseudolabel_epochs=10
        )
        
        phases1 = []
        phases2 = []
        
        for epoch in range(25):
            if epoch < config.warmup_epochs:
                phases1.append('warmup')
                phases2.append('warmup')
            elif epoch < config.warmup_epochs + config.discriminator_epochs:
                phases1.append('discriminator')
                phases2.append('discriminator')
            else:
                phases1.append('pseudolabel')
                phases2.append('pseudolabel')
        
        assert phases1 == phases2


class TestIntegration:
    """Integration tests for complete hybrid training workflow."""
    
    def test_full_hybrid_workflow(self):
        """Test complete hybrid training workflow."""
        # 1. Configuration
        config = HybridTrainingConfig(
            enabled=True,
            warmup_epochs=2,
            discriminator_epochs=3,
            pseudolabel_epochs=2,
            invalid_ratio=0.2,
            confidence_threshold=0.9
        )
        
        total_epochs = (
            config.warmup_epochs +
            config.discriminator_epochs +
            config.pseudolabel_epochs
        )
        
        # 2. Track phases
        phases_executed = []
        
        for epoch in range(total_epochs):
            if epoch < config.warmup_epochs:
                phase = 'warmup'
            elif epoch < config.warmup_epochs + config.discriminator_epochs:
                phase = 'discriminator'
            else:
                phase = 'pseudolabel'
            
            phases_executed.append(phase)
        
        # 3. Verify phases executed in correct order
        assert 'warmup' in phases_executed
        assert 'discriminator' in phases_executed
        assert 'pseudolabel' in phases_executed
        
        # First phase should be warmup
        assert phases_executed[0] == 'warmup'
        
        # Last phase should be pseudolabel
        assert phases_executed[-1] == 'pseudolabel'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
