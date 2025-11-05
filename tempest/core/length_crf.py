"""
Vectorized Length-Constrained CRF for tranquillyzer.
Fully XLA-compatible with constraint weight ramping over epochs.

Key features:
- Uses Viterbi sequence (not argmax of potentials)
- Fully vectorized (no py_function, no map_fn)
- Fixed shapes for XLA compilation
- Constraint weight ramping over epochs
"""

import tensorflow as tf
import numpy as np
from tf2crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood


class LengthConstrainedCRF(CRF):
    """CRF layer with length constraints for specific labels."""
    
    def __init__(self, units, length_constraints=None, 
                 constraint_weight=5.0, **kwargs):
        """
        Args:
            units: Number of output labels
            length_constraints: Dict mapping label_name -> (min_length, max_length)
                Example: {'UMI': (8, 8), 'ACC': (6, 6)}
            constraint_weight: Initial penalty multiplier for length violations
        """
        super().__init__(units, **kwargs)
        self.length_constraints = length_constraints or {}
        self.constraint_weight = constraint_weight
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'length_constraints': self.length_constraints,
            'constraint_weight': self.constraint_weight
        })
        return config


def unpack_data(data):
    """Unpack data tuple (compatible with tf2crf pattern)."""
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithLengthConstrainedCRF(tf.keras.Model):
    """
    Vectorized wrapper model that adds length penalty to CRF loss.
    
    Features:
    - Fully vectorized (XLA-compatible)
    - Uses Viterbi sequence for penalty
    - Constraint weight ramping over epochs
    - Fixed shapes for optimal compilation
    """
    
    def __init__(self, base_model, length_constraints, 
                 constraint_weight=5.0, label_binarizer=None,
                 max_seq_len=512,
                 constraint_ramp_epochs=5,
                 sparse_target=False, metric='accuracy', **kwargs):
        """
        Args:
            base_model: The base model with CRF layer
            length_constraints: Dict mapping label_name -> (min_length, max_length)
            constraint_weight: Final penalty multiplier (Î» in theory)
            label_binarizer: sklearn LabelBinarizer for label mapping
            max_seq_len: Maximum sequence length (for XLA compilation)
            constraint_ramp_epochs: Number of epochs to ramp constraint from 0 to full
            sparse_target: If True, y is shape (batch, seq_len) with label indices
            metric: Metric for training (default 'accuracy')
        """
        super().__init__(**kwargs)
        self.base_model = base_model
        self.model_layers = [layer for layer in self.base_model.layers]
        self.length_constraints = length_constraints
        self.max_constraint_weight = constraint_weight
        self.max_seq_len = max_seq_len
        self.constraint_ramp_epochs = constraint_ramp_epochs
        self.label_binarizer = label_binarizer
        self.sparse_target = sparse_target
        
        # Epoch counter for ramping
        self.current_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        # Find the CRF layer
        self.crf_layer = None
        for layer in base_model.layers:
            if isinstance(layer, (CRF, LengthConstrainedCRF)):
                self.crf_layer = layer
                break
        
        if self.crf_layer is None:
            raise ValueError("No CRF layer found in base model")
        
        # Build label to index mapping and constraint tensors
        if label_binarizer is not None:
            self.label_to_idx = {
                label: idx 
                for idx, label in enumerate(label_binarizer.classes_)
            }
            self.num_labels = len(label_binarizer.classes_)
            
            # Create constraint tensors for vectorized computation
            self._build_constraint_tensors()
        else:
            self.label_to_idx = {}
            self.num_labels = 0
        
        # Setup metrics
        if isinstance(metric, str):
            if metric == 'accuracy':
                self.metrics_fn = tf.keras.metrics.Accuracy(name='accuracy')
            else:
                raise ValueError(f'Unknown metric name: {metric}')
        else:
            self.metrics_fn = metric
        
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.crf_loss_tracker = tf.keras.metrics.Mean(name='crf_loss')
        self.penalty_tracker = tf.keras.metrics.Mean(name='length_penalty')
        self.constraint_weight_tracker = tf.keras.metrics.Mean(name='constraint_weight')
    
    def _build_constraint_tensors(self):
        """Build tensors for vectorized constraint computation."""
        # Create arrays indexed by label_idx
        min_lengths = np.zeros(self.num_labels, dtype=np.float32)
        max_lengths = np.full(self.num_labels, 1e6, dtype=np.float32)  # Large default
        has_constraint = np.zeros(self.num_labels, dtype=np.float32)
        
        for label_name, (min_len, max_len) in self.length_constraints.items():
            if label_name in self.label_to_idx:
                idx = self.label_to_idx[label_name]
                min_lengths[idx] = float(min_len)
                max_lengths[idx] = float(max_len)
                has_constraint[idx] = 1.0
        
        # Convert to constants for XLA
        self.min_lengths = tf.constant(min_lengths, dtype=tf.float32)
        self.max_lengths = tf.constant(max_lengths, dtype=tf.float32)
        self.has_constraint = tf.constant(has_constraint, dtype=tf.float32)
    
    def call(self, inputs, training=False):
        """Forward pass through base model."""
        output = inputs
        for layer in self.model_layers:
            output = layer(output)
        if training:
            return output  # Returns 4-tuple: (viterbi, potentials, seq_len, transitions)
        else:
            return output[0]  # Just viterbi sequence for inference
    
    def compute_constraint_weight(self):
        """Compute current constraint weight with ramping."""
        if self.constraint_ramp_epochs <= 0:
            return self.max_constraint_weight
        
        # Linear ramp from 0 to max_constraint_weight over ramp_epochs
        epoch_float = tf.cast(self.current_epoch, tf.float32)
        ramp_float = tf.cast(self.constraint_ramp_epochs, tf.float32)
        
        ramp_factor = tf.minimum(epoch_float / ramp_float, 1.0)
        current_weight = self.max_constraint_weight * ramp_factor
        
        return current_weight
    
    def compute_loss(self, x, y, training=False):
        """
        Compute CRF loss + length penalty.
        
        Args:
            x: Input sequences [batch, seq_len]
            y: Target labels [batch, seq_len] or [batch, seq_len, num_labels]
            training: Whether in training mode
            
        Returns:
            viterbi_sequence: Decoded sequence [batch, seq_len]
            sequence_length: Actual lengths [batch]
            total_loss: CRF loss + length penalty
            crf_loss: CRF loss alone
            length_penalty: Length penalty alone
            current_weight: Current constraint weight
        """
        # Get CRF outputs (4-tuple during training)
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(x, training=training)
        
        # Convert y to sparse format if needed
        if not self.sparse_target:
            y = tf.argmax(y, axis=-1)
        
        # Compute CRF loss
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        crf_loss = tf.reduce_mean(crf_loss)
        
        # Compute current constraint weight (with ramping)
        current_weight = self.compute_constraint_weight()
        
        # Compute length penalty using Viterbi sequence (vectorized)
        length_penalty = self._compute_length_penalty_vectorized(
            viterbi_sequence, sequence_length, current_weight
        )
        
        # Total loss
        total_loss = crf_loss + length_penalty
        
        return viterbi_sequence, sequence_length, total_loss, crf_loss, length_penalty, current_weight
    
    @tf.function(jit_compile=True)
    def _compute_length_penalty_vectorized(self, viterbi_sequence, sequence_lengths, weight):
        """
        Vectorized computation of length penalty.
        
        Uses Viterbi sequence and fully vectorized TensorFlow ops (XLA-compatible).
        
        Args:
            viterbi_sequence: [batch_size, max_seq_len] - Viterbi decoded labels
            sequence_lengths: [batch_size] - Actual sequence lengths
            weight: Scalar - Current constraint weight
            
        Returns:
            Scalar penalty term
        """
        if not self.length_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        
        batch_size = tf.shape(viterbi_sequence)[0]
        
        # Create mask for valid positions
        positions = tf.range(self.max_seq_len, dtype=tf.int32)
        positions = tf.expand_dims(positions, 0)  # [1, max_seq_len]
        sequence_lengths_expanded = tf.expand_dims(sequence_lengths, 1)  # [batch_size, 1]
        valid_mask = positions < sequence_lengths_expanded  # [batch_size, max_seq_len]
        
        # Extract run lengths for all labels in vectorized manner
        penalties = []
        
        for label_idx in range(self.num_labels):
            # Skip if this label has no constraint
            if self.has_constraint[label_idx] == 0.0:
                continue
            
            # Create mask for this label
            label_mask = tf.cast(
                tf.equal(viterbi_sequence, label_idx), 
                tf.float32
            )  # [batch_size, max_seq_len]
            
            # Apply valid position mask
            label_mask = label_mask * tf.cast(valid_mask, tf.float32)
            
            # Detect run boundaries (where label changes)
            # Pad with zeros at start to detect first run
            padded_mask = tf.pad(label_mask, [[0, 0], [1, 0]], constant_values=0.0)
            
            # Run starts where prev=0 and current=1
            starts = (padded_mask[:, :-1] == 0.0) & (label_mask == 1.0)
            starts = tf.cast(starts, tf.float32)
            
            # Run ends where current=1 and next=0
            padded_mask_end = tf.pad(label_mask, [[0, 0], [0, 1]], constant_values=0.0)
            ends = (label_mask == 1.0) & (padded_mask_end[:, 1:] == 0.0)
            ends = tf.cast(ends, tf.float32)
            
            # Assign run IDs using cumsum on starts
            run_ids = tf.cumsum(starts, axis=1)  # [batch_size, max_seq_len]
            run_ids = run_ids * label_mask  # Zero out non-label positions
            
            # Compute run lengths by counting positions with same run_id
            # For each unique run_id, sum the label_mask
            max_runs = tf.cast(tf.reduce_max(run_ids), tf.int32) + 1
            
            # Iterate over runs (small number, usually <10 per sequence)
            for run_id in tf.range(1, max_runs):
                run_id_float = tf.cast(run_id, tf.float32)
                run_mask = tf.cast(
                    tf.equal(run_ids, run_id_float), 
                    tf.float32
                )  # [batch_size, max_seq_len]
                
                # Sum to get run length for each sequence in batch
                run_lengths = tf.reduce_sum(run_mask, axis=1)  # [batch_size]
                
                # Only compute penalty where run exists (run_length > 0)
                run_exists = run_lengths > 0.0
                
                # Get constraints for this label
                min_len = self.min_lengths[label_idx]
                max_len = self.max_lengths[label_idx]
                
                # Compute penalties: Î© = [(L_min - L)_+^2 + (L - L_max)_+^2]
                underflow = tf.maximum(0.0, min_len - run_lengths)
                overflow = tf.maximum(0.0, run_lengths - max_len)
                
                run_penalty = underflow ** 2 + overflow ** 2
                
                # Only include penalties where run exists
                run_penalty = tf.where(run_exists, run_penalty, 0.0)
                
                penalties.append(run_penalty)  # [batch_size]
        
        if not penalties:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Stack all penalties and compute mean
        all_penalties = tf.stack(penalties, axis=0)  # [num_penalties, batch_size]
        
        # Mean over all runs and batch
        mean_penalty = tf.reduce_mean(all_penalties)
        
        # Apply constraint weight
        weighted_penalty = weight * mean_penalty
        
        return weighted_penalty
    
    def train_step(self, data):
        """Custom training step with length penalty and constraint ramping."""
        x, y, sample_weight = unpack_data(data)
        
        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, total_loss, crf_loss, length_penalty, current_weight = \
                self.compute_loss(x, y, training=True)
            
            # Add any additional losses from regularization
            total_loss = total_loss + tf.cast(tf.reduce_sum(self.losses), total_loss.dtype)
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.crf_loss_tracker.update_state(crf_loss)
        self.penalty_tracker.update_state(length_penalty)
        self.constraint_weight_tracker.update_state(current_weight)
        
        # Update accuracy metric
        if not self.sparse_target:
            y = tf.argmax(y, axis=-1)
        self.metrics_fn.update_state(
            y, viterbi_sequence, 
            tf.sequence_mask(sequence_length, tf.shape(y)[1])
        )
        
        return {
            'loss': self.loss_tracker.result(),
            'crf_loss': self.crf_loss_tracker.result(),
            'length_penalty': self.penalty_tracker.result(),
            'constraint_weight': self.constraint_weight_tracker.result(),
            self.metrics_fn.name: self.metrics_fn.result()
        }
    
    def test_step(self, data):
        """Custom test step."""
        x, y, sample_weight = unpack_data(data)
        
        viterbi_sequence, sequence_length, total_loss, crf_loss, length_penalty, current_weight = \
            self.compute_loss(x, y, training=True)
        
        # Add regularization losses
        total_loss = total_loss + tf.cast(tf.reduce_sum(self.losses), total_loss.dtype)
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.crf_loss_tracker.update_state(crf_loss)
        self.penalty_tracker.update_state(length_penalty)
        self.constraint_weight_tracker.update_state(current_weight)
        
        # Update accuracy metric
        if not self.sparse_target:
            y = tf.argmax(y, axis=-1)
        self.metrics_fn.update_state(
            y, viterbi_sequence,
            tf.sequence_mask(sequence_length, tf.shape(y)[1])
        )
        
        return {
            'loss_val': self.loss_tracker.result(),
            'crf_loss_val': self.crf_loss_tracker.result(),
            'length_penalty_val': self.penalty_tracker.result(),
            'constraint_weight_val': self.constraint_weight_tracker.result(),
            f'val_{self.metrics_fn.name}': self.metrics_fn.result()
        }
    
    def on_epoch_begin(self, epoch, logs=None):
        """Update epoch counter for constraint ramping."""
        self.current_epoch.assign(epoch)
    
    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [self.loss_tracker, self.crf_loss_tracker, 
                self.penalty_tracker, self.constraint_weight_tracker, 
                self.metrics_fn]


def create_length_constrained_model(base_model, length_constraints, 
                                    constraint_weight=5.0,
                                    max_seq_len=512,
                                    constraint_ramp_epochs=5,
                                    label_binarizer=None,
                                    sparse_target=False,
                                    metric='accuracy'):
    """
    Convenience function to wrap a CRF model with length constraints.
    
    Args:
        base_model: Model ending with CRF layer
        length_constraints: Dict like {'UMI': (8, 8), 'ACC': (6, 6)}
        constraint_weight: Final penalty multiplier (Î» in theory)
        max_seq_len: Maximum sequence length for XLA compilation
        constraint_ramp_epochs: Number of epochs to ramp constraint from 0 to full
        label_binarizer: LabelBinarizer for label mapping
        sparse_target: Whether y is sparse (True) or one-hot (False)
        metric: Metric for training (default 'accuracy')
        
    Returns:
        ModelWithLengthConstrainedCRF instance
    """
    return ModelWithLengthConstrainedCRF(
        base_model=base_model,
        length_constraints=length_constraints,
        constraint_weight=constraint_weight,
        max_seq_len=max_seq_len,
        constraint_ramp_epochs=constraint_ramp_epochs,
        label_binarizer=label_binarizer,
        sparse_target=sparse_target,
        metric=metric
    )


# Example usage:
"""
from length_constrained_crf_vectorized import create_length_constrained_model

# Define constraints
length_constraints = {
    'UMI': (8, 8),
    'ACC': (6, 6)
}

with strategy.scope():
    base_model = ont_read_annotator(
        vocab_size=5,
        embedding_dim=128,
        num_labels=num_labels,
        crf_layer=True
    )
    
    model = create_length_constrained_model(
        base_model=base_model,
        length_constraints=length_constraints,
        constraint_weight=5.0,
        max_seq_len=512,  # For XLA compilation
        constraint_ramp_epochs=5,  # Ramp over first 5 epochs
        label_binarizer=label_binarizer,
        sparse_target=False
    )
    
    model.compile(optimizer=Adam(learning_rate=0.001))

# Initialize
dummy_input = tf.zeros((1, 512), dtype=tf.int32)
_ = model(dummy_input)

# Train - constraint weight will ramp from 0 to 5.0 over first 5 epochs
history = model.fit(train_gen, validation_data=val_gen, epochs=10)
"""
