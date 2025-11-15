"""
Vectorized Length-Constrained CRF for tempest.
Fully XLA-friendly (no py_function/map_fn) with constraint weight ramping over epochs.

Key features:
- Uses Viterbi sequence (not argmax of potentials)
- Fully vectorized TF ops
- Fixed shapes for optimal compilation
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
    - Fully vectorized (no py_function / map_fn)
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
            constraint_weight: Final penalty multiplier (λ in theory)
            label_binarizer: sklearn LabelBinarizer for label mapping
            max_seq_len: Maximum sequence length (for padding / defaults)
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
        self.span_through_errors = {'ERROR_SUB', 'ERROR_INS'}
        self.exclude_errors = {'ERROR_BOUNDARY', 'ERROR_LOSS', 'ERROR_ORDER', 'ERROR_DUP'}

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

        # Store numpy mask for fast Python branching in tf.function loops
        self._has_constraint_np = has_constraint

        # Convert to constants for TF ops
        self.min_lengths = tf.constant(min_lengths, dtype=tf.float32)
        self.max_lengths = tf.constant(max_lengths, dtype=tf.float32)
        self.has_constraint = tf.constant(has_constraint, dtype=tf.float32)
    
    @tf.function
    def _resolve_benign_errors(self, viterbi_sequence):
        """Replace ERROR_SUB and ERROR_INS with preceding valid label."""
        
        # Get error indices
        error_sub_idx = self.label_to_idx.get('ERROR_SUB', -1)
        error_ins_idx = self.label_to_idx.get('ERROR_INS', -1)
        
        # Create mask for positions to resolve
        is_benign_error = tf.logical_or(
            tf.equal(viterbi_sequence, error_sub_idx),
            tf.equal(viterbi_sequence, error_ins_idx)
        )
        
        # Forward fill using iterative replacement (handles consecutive errors)
        resolved = viterbi_sequence
        for _ in range(5):  # Handle up to 5 consecutive errors
            shifted = tf.pad(resolved[:, :-1], [[0,0], [1,0]], 
                           constant_values=resolved[:, 0])
            resolved = tf.where(is_benign_error, shifted, resolved)
        
        return resolved

    def call(self, inputs, training=False):
        """Forward pass through base model."""
        output = inputs
        for layer in self.model_layers:
            output = layer(output)
        if training:
            # Returns 4-tuple: (viterbi, potentials, seq_len, transitions)
            return output
        else:
            # Just viterbi sequence for inference
            return output[0]

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
            total_loss: CRF loss + mean length penalty
            crf_loss: CRF loss alone
            length_penalty: Mean length penalty alone (scalar)
            current_weight: Current constraint weight (scalar)
        """
        # Get CRF outputs (4-tuple during training)
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(
            x, training=training
        )

        # Convert y to sparse format if needed
        if not self.sparse_target:
            y = tf.argmax(y, axis=-1)

        # Compute CRF loss
        crf_loss = -crf_log_likelihood(
            potentials, y, sequence_length, chain_kernel
        )[0]
        crf_loss = tf.reduce_mean(crf_loss)

        # Compute current constraint weight (with ramping)
        current_weight = self.compute_constraint_weight()

        # Compute length penalty using Viterbi sequence (vectorized)
        per_seq_penalty = self._compute_length_penalty_vectorized(
            viterbi_sequence, sequence_length, current_weight
        )  # [batch]

        # Mean penalty over batch for training loss
        length_penalty = tf.reduce_mean(per_seq_penalty)

        # Total loss
        total_loss = crf_loss + length_penalty

        return (
            viterbi_sequence,
            sequence_length,
            total_loss,
            crf_loss,
            length_penalty,
            current_weight,
        )

    @tf.function
    def _compute_length_penalty_vectorized(self, viterbi_sequence, sequence_lengths, weight):
        """
        Vectorized computation of per-sequence length penalties.

        Uses Viterbi sequence and fully vectorized TensorFlow ops.

        Args:
            viterbi_sequence: [batch_size, seq_len] - Viterbi decoded labels
            sequence_lengths: [batch_size] - Actual sequence lengths
            weight: Scalar - Current constraint weight

        Returns:
            penalties: [batch_size] tensor of penalties (already weighted by `weight`)
        """
        batch_size = tf.shape(viterbi_sequence)[0]
        seq_len = tf.shape(viterbi_sequence)[1]

        # No constraints == zero penalties
        if not self.length_constraints:
            return tf.zeros(batch_size, dtype=tf.float32)
        
        # RESOLVE benign errors before computing segment lengths
        viterbi_sequence = self._resolve_benign_errors(viterbi_sequence)

        # Create mask for valid positions
        positions = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]  # [1, seq_len]
        sequence_lengths_expanded = sequence_lengths[:, tf.newaxis]   # [batch, 1]
        valid_mask = positions < sequence_lengths_expanded            # [batch, seq_len]

        # Accumulate per-sequence penalty
        penalties = tf.zeros(batch_size, dtype=tf.float32)

        # Loop over each label index (small, fixed number)
        for label_idx in range(self.num_labels):
            # Use numpy mask for pure-Python branching
            if float(self._has_constraint_np[label_idx]) == 0.0:
                continue

            # Mask for this label
            label_mask = tf.cast(tf.equal(viterbi_sequence, label_idx), tf.float32)
            label_mask = label_mask * tf.cast(valid_mask, tf.float32)  # [batch, seq_len]

            # Detect run starts via transitions 0→1
            padded = tf.pad(label_mask, [[0, 0], [1, 0]], constant_values=0.0)
            starts = tf.cast(
                (padded[:, :-1] == 0.0) & (label_mask == 1.0),
                tf.float32,
            )  # [batch, seq_len]

            # Assign run IDs via cumsum over starts
            run_ids = tf.cumsum(starts, axis=1) * label_mask  # [batch, seq_len]
            max_run_id = tf.cast(tf.reduce_max(run_ids), tf.int32)

            # If no runs at all for this label, skip
            def _process_label():
                def cond(run_id, current_penalties):
                    return run_id <= max_run_id

                def body(run_id, current_penalties):
                    rid = tf.cast(run_id, tf.float32)
                    run_mask = tf.cast(tf.equal(run_ids, rid), tf.float32)  # [batch, seq_len]
                    run_lengths = tf.reduce_sum(run_mask, axis=1)           # [batch]
                    run_exists = run_lengths > 0.0

                    # Constraints for this label
                    min_len = self.min_lengths[label_idx]
                    max_len = self.max_lengths[label_idx]

                    underflow = tf.maximum(0.0, min_len - run_lengths)
                    overflow = tf.maximum(0.0, run_lengths - max_len)
                    run_penalty = underflow ** 2 + overflow ** 2
                    run_penalty = tf.where(run_exists, run_penalty, 0.0)

                    current_penalties = current_penalties + run_penalty
                    return run_id + 1, current_penalties

                init_run_id = tf.constant(1, dtype=tf.int32)
                _, updated_penalties = tf.while_loop(
                    cond,
                    body,
                    loop_vars=(init_run_id, penalties),
                    shape_invariants=(
                        tf.TensorShape([]),
                        tf.TensorShape([None]),
                    ),
                )
                return updated_penalties

            penalties = tf.cond(
                max_run_id > 0,
                true_fn=_process_label,
                false_fn=lambda: penalties,
            )

        # Apply constraint weight per sequence
        penalties = penalties * weight
        return penalties

    def compute_length_penalty(self, predictions, sequence_lengths=None):
        """
        Public wrapper for computing length penalties on predicted label sequences.

        Args:
            predictions: [batch_size, seq_len] integer tensor of predicted labels
            sequence_lengths: optional [batch_size] tensor of actual sequence lengths.
                              If None, uses full seq_len for all.

        Returns:
            - If batch_size == 1: scalar penalty tensor
            - Else: [batch_size] tensor of per-sequence penalties
        """
        # Use actual sequence length dimension when not provided
        if sequence_lengths is None:
            batch_size = tf.shape(predictions)[0]
            seq_len = tf.shape(predictions)[1]
            sequence_lengths = tf.fill([batch_size], seq_len)

        # Current ramped weight
        weight = self.compute_constraint_weight()

        # Per-sequence penalties
        penalties = self._compute_length_penalty_vectorized(
            viterbi_sequence=predictions,
            sequence_lengths=sequence_lengths,
            weight=weight,
        )

        # For most tests, single-sequence calls want a scalar
        if penalties.shape.rank == 1 and penalties.shape[0] == 1:
            return tf.squeeze(penalties, axis=0)

        return penalties

    def train_step(self, data):
        """Custom training step with length penalty and constraint ramping."""
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            (
                viterbi_sequence,
                sequence_length,
                total_loss,
                crf_loss,
                length_penalty,
                current_weight,
            ) = self.compute_loss(x, y, training=True)

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
            y,
            viterbi_sequence,
            tf.sequence_mask(sequence_length, tf.shape(y)[1]),
        )

        return {
            "loss": self.loss_tracker.result(),
            "crf_loss": self.crf_loss_tracker.result(),
            "length_penalty": self.penalty_tracker.result(),
            "constraint_weight": self.constraint_weight_tracker.result(),
            self.metrics_fn.name: self.metrics_fn.result(),
        }

    def test_step(self, data):
        """Custom test step."""
        x, y, sample_weight = unpack_data(data)

        (
            viterbi_sequence,
            sequence_length,
            total_loss,
            crf_loss,
            length_penalty,
            current_weight,
        ) = self.compute_loss(x, y, training=True)

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
            y,
            viterbi_sequence,
            tf.sequence_mask(sequence_length, tf.shape(y)[1]),
        )

        return {
            "loss_val": self.loss_tracker.result(),
            "crf_loss_val": self.crf_loss_tracker.result(),
            "length_penalty_val": self.penalty_tracker.result(),
            "constraint_weight_val": self.constraint_weight_tracker.result(),
            f"val_{self.metrics_fn.name}": self.metrics_fn.result(),
        }

    def on_epoch_begin(self, epoch, logs=None):
        """Update epoch counter for constraint ramping."""
        self.current_epoch.assign(epoch)

    @property
    def metrics(self):
        """Return list of metrics to track."""
        return [
            self.loss_tracker,
            self.crf_loss_tracker,
            self.penalty_tracker,
            self.constraint_weight_tracker,
            self.metrics_fn,
        ]


def create_length_constrained_model(
    base_model,
    length_constraints,
    constraint_weight=5.0,
    max_seq_len=512,
    constraint_ramp_epochs=5,
    label_binarizer=None,
    sparse_target=False,
    metric="accuracy",
):
    """
    Convenience function to wrap a CRF model with length constraints.

    Args:
        base_model: Model ending with CRF layer
        length_constraints: Dict like {'UMI': (8, 8), 'ACC': (6, 6)}
        constraint_weight: Final penalty multiplier (λ in theory)
        max_seq_len: Maximum sequence length for XLA compilation / padding
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
        metric=metric,
    )