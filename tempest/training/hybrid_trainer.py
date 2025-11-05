"""
Hybrid robustness training for Tempest.

Combines discriminator-based adversarial training and pseudo-label
self-training for models robust to segment-level errors.

Part of: tempest/training/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Import from proper tempest modules
from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.data.invalid_generator import InvalidReadGenerator
from tempest.utils.config import TempestConfig
from tempest.utils.io import load_fastq, ensure_dir

logger = logging.getLogger(__name__)


def pad_sequences(sequences: np.ndarray, labels: np.ndarray, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to max_length.
    
    Args:
        sequences: Input sequences array [batch, seq_len]
        labels: Label array [batch, seq_len]
        max_length: Target length for padding
        
    Returns:
        Tuple of padded sequences and labels
    """
    num_sequences = sequences.shape[0]
    current_length = sequences.shape[1]
    
    if current_length == max_length:
        return sequences, labels
    
    padded_sequences = np.zeros((num_sequences, max_length), dtype=sequences.dtype)
    padded_labels = np.zeros((num_sequences, max_length), dtype=labels.dtype)
    
    copy_length = min(current_length, max_length)
    padded_sequences[:, :copy_length] = sequences[:, :copy_length]
    padded_labels[:, :copy_length] = labels[:, :copy_length]
    
    return padded_sequences, padded_labels


def convert_labels_to_categorical(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        labels: Integer label array [batch, seq_len]
        num_classes: Number of label classes
        
    Returns:
        One-hot encoded labels [batch, seq_len, num_classes]
    """
    num_samples, seq_length = labels.shape
    categorical = np.zeros((num_samples, seq_length, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        for j in range(seq_length):
            if labels[i, j] < num_classes:  # Bounds check
                categorical[i, j, labels[i, j]] = 1.0
    
    return categorical


def build_model_from_config(config: TempestConfig) -> keras.Model:
    """
    Build a sequence labeling model from configuration.
    
    Args:
        config: TempestConfig object with model parameters
        
    Returns:
        Compiled Keras model
    """
    # Model parameters from config
    max_seq_len = config.model.max_seq_len
    num_labels = config.model.num_labels
    embedding_dim = config.model.embedding_dim
    lstm_units = config.model.lstm_units
    lstm_layers = config.model.lstm_layers
    dropout = config.model.dropout
    use_cnn = config.model.use_cnn
    use_bilstm = config.model.use_bilstm
    
    # Build model
    inputs = keras.Input(shape=(max_seq_len,), dtype='int32')
    
    # Embedding layer (5 for ACGTN)
    x = layers.Embedding(config.model.vocab_size, embedding_dim, mask_zero=True)(inputs)
    
    # Optional CNN layers
    if use_cnn:
        for filters, kernel in zip(config.model.cnn_filters, config.model.cnn_kernels):
            x = layers.Conv1D(filters, kernel, activation='relu', padding='same')(x)
    
    # BiLSTM layers
    if use_bilstm:
        for i in range(lstm_layers):
            return_sequences = True  # Always return sequences for sequence labeling
            units = lstm_units if i == 0 else lstm_units // 2
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)
            )(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    
    # Output layer
    outputs = layers.Dense(num_labels, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def print_model_summary(model: keras.Model):
    """
    Print a formatted model summary.
    
    Args:
        model: Keras model to summarize
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


class ArchitectureDiscriminator(keras.Model):
    """
    Neural network to classify read architectures as valid or invalid.
    
    Outputs three-way classification:
    - valid_forward: Correct architecture in forward orientation  
    - valid_reverse: Correct architecture in reverse orientation
    - invalid: Incorrect architecture (segment errors)
    """
    
    def __init__(self, num_labels: int = 6, hidden_dim: int = 64):
        """
        Initialize discriminator.
        
        Args:
            num_labels: Number of possible labels
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        # Architecture recognition layers
        self.conv1 = layers.Conv1D(32, 5, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(64, 7, activation='relu', padding='same')
        self.pool = layers.GlobalMaxPooling1D()
        
        # Dense layers
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(32, activation='relu')
        
        # Three-way classification
        self.classifier = layers.Dense(3, activation='softmax')
        
    def call(self, label_predictions, training=False):
        """
        Forward pass.
        
        Args:
            label_predictions: [batch, seq_len, num_labels] from main model
            training: Whether in training mode
            
        Returns:
            [batch, 3] probabilities for (valid_fwd, valid_rev, invalid)
        """
        x = self.conv1(label_predictions)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)


class PseudoLabelGenerator:
    """
    Generate pseudo-labels from unlabeled data using model predictions.
    
    Handles both forward and reverse orientations, validates architectures,
    and filters by confidence threshold.
    """
    
    def __init__(self, model, config: TempestConfig, 
                 confidence_threshold: float = 0.9,
                 label_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize pseudo-label generator.
        
        Args:
            model: Trained model for predictions
            config: Tempest configuration
            confidence_threshold: Minimum confidence for pseudo-labels
            label_to_idx: Mapping from label names to indices
        """
        self.model = model
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.label_to_idx = label_to_idx or self._default_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def _default_label_mapping(self) -> Dict[str, int]:
        """Create default label mapping from config."""
        if self.config and self.config.simulation and self.config.simulation.sequence_order:
            sequence_order = self.config.simulation.sequence_order
            mapping = {label: idx for idx, label in enumerate(sequence_order)}
            # Add PAD and UNKNOWN if not present
            if 'PAD' not in mapping:
                mapping['PAD'] = len(mapping)
            if 'UNKNOWN' not in mapping:
                mapping['UNKNOWN'] = len(mapping)
            return mapping
        
        # Default mapping
        return {
            'ADAPTER5': 0, 'UMI': 1, 'ACC': 2, 
            'BARCODE': 3, 'INSERT': 4, 'ADAPTER3': 5,
            'PAD': 6, 'UNKNOWN': 7
        }
    
    def generate_from_fastq(self, fastq_file: str, 
                           max_reads: int = 1000) -> List[SimulatedRead]:
        """
        Generate pseudo-labeled reads from FASTQ file.
        
        Args:
            fastq_file: Path to unlabeled FASTQ
            max_reads: Maximum reads to process
            
        Returns:
            List of pseudo-labeled SimulatedRead objects
        """
        pseudo_labeled = []
        
        # Process FASTQ in batches
        batch_size = 32
        sequences = []
        read_ids = []
        
        logger.info(f"Generating pseudo-labels from {fastq_file}")
        
        for i, record in enumerate(load_fastq(fastq_file, max_reads)):
            sequences.append(str(record.seq))
            read_ids.append(record.id)
            
            if len(sequences) == batch_size:
                batch_pseudo = self._process_batch(sequences, read_ids)
                pseudo_labeled.extend(batch_pseudo)
                sequences = []
                read_ids = []
        
        # Process remaining
        if sequences:
            batch_pseudo = self._process_batch(sequences, read_ids)
            pseudo_labeled.extend(batch_pseudo)
        
        logger.info(f"Generated {len(pseudo_labeled)} pseudo-labels from "
                   f"{max_reads} reads with confidence >= {self.confidence_threshold}")
        
        return pseudo_labeled
    
    def _process_batch(self, sequences: List[str], 
                      read_ids: List[str]) -> List[SimulatedRead]:
        """Process a batch of sequences to generate pseudo-labels."""
        batch_reads = []
        
        # Convert sequences to input format
        max_len = self.config.model.max_seq_len
        X = self._sequences_to_array(sequences, max_len)
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        for i, (seq, pred) in enumerate(zip(sequences, predictions)):
            # Check confidence
            confidence = np.max(pred, axis=-1)[:len(seq)]
            mean_conf = np.mean(confidence)
            
            if mean_conf >= self.confidence_threshold:
                # Convert predictions to labels
                label_indices = np.argmax(pred, axis=-1)[:len(seq)]
                labels = [self.idx_to_label.get(idx, 'UNKNOWN') 
                         for idx in label_indices]
                
                # Create label regions
                label_regions = self._extract_regions(labels)
                
                batch_reads.append(SimulatedRead(
                    sequence=seq,
                    labels=labels,
                    label_regions=label_regions,
                    metadata={'pseudo_label': True, 
                             'confidence': float(mean_conf),
                             'read_id': read_ids[i]}
                ))
        
        return batch_reads
    
    def _sequences_to_array(self, sequences: List[str], max_len: int) -> np.ndarray:
        """Convert sequences to numpy array."""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        X = np.zeros((len(sequences), max_len), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq[:max_len]):
                X[i, j] = base_to_idx.get(base, 4)
        
        return X
    
    def _extract_regions(self, labels: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """Extract contiguous regions for each label."""
        regions = {}
        if not labels:
            return regions
        
        current_label = labels[0]
        start = 0
        
        for i, label in enumerate(labels[1:], 1):
            if label != current_label:
                if current_label not in regions:
                    regions[current_label] = []
                regions[current_label].append((start, i))
                current_label = label
                start = i
        
        # Add final region
        if current_label not in regions:
            regions[current_label] = []
        regions[current_label].append((start, len(labels)))
        
        return regions


class HybridTrainer:
    """
    Implements hybrid robustness training combining:
    1. Invalid read generation for architecture robustness
    2. Discriminator-based adversarial training
    3. Pseudo-label self-training on unlabeled data
    """
    
    def __init__(self, config: TempestConfig):
        """
        Initialize hybrid trainer.
        
        Args:
            config: TempestConfig with hybrid training parameters
        """
        self.config = config
        self.invalid_generator = InvalidReadGenerator(config)
        
        # Training parameters from config
        if config.hybrid:
            self.invalid_ratio = config.hybrid.invalid_ratio
            self.discriminator_weight = config.hybrid.adversarial_weight
            self.pseudo_label_conf = config.hybrid.confidence_threshold
            self.warmup_epochs = config.hybrid.warmup_epochs
            self.discriminator_epochs = config.hybrid.discriminator_epochs
            self.pseudo_epochs = config.hybrid.pseudolabel_epochs
        else:
            # Defaults
            self.invalid_ratio = 0.1
            self.discriminator_weight = 0.1
            self.pseudo_label_conf = 0.9
            self.warmup_epochs = 5
            self.discriminator_epochs = 10
            self.pseudo_epochs = 5
        
        logger.info(f"Initialized HybridTrainer with invalid_ratio={self.invalid_ratio}, "
                   f"discriminator_weight={self.discriminator_weight}")
    
    def train(self, train_reads: List[SimulatedRead], 
             val_reads: List[SimulatedRead],
             unlabeled_fastq: Optional[str] = None,
             checkpoint_dir: str = "checkpoints") -> keras.Model:
        """
        Run hybrid training pipeline.
        
        Args:
            train_reads: Training SimulatedRead objects
            val_reads: Validation SimulatedRead objects  
            unlabeled_fastq: Optional path to unlabeled FASTQ
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Trained model
        """
        logger.info("="*80)
        logger.info("STARTING HYBRID TRAINING PIPELINE")
        logger.info("="*80)
        
        # Ensure checkpoint directory exists
        checkpoint_path = Path(checkpoint_dir)
        ensure_dir(str(checkpoint_path))
        
        # Convert reads to arrays
        logger.info("Converting reads to arrays...")
        X_train, y_train, label_to_idx = reads_to_arrays(train_reads)
        X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx=label_to_idx)
        
        # Pad sequences
        max_len = self.config.model.max_seq_len
        X_train, y_train = pad_sequences(X_train, y_train, max_len)
        X_val, y_val = pad_sequences(X_val, y_val, max_len)
        
        # Convert to categorical
        num_labels = self.config.model.num_labels
        y_train = convert_labels_to_categorical(y_train, num_labels)
        y_val = convert_labels_to_categorical(y_val, num_labels)
        
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Phase 1: Warm-up training on clean data
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: WARM-UP TRAINING")
        logger.info("="*60)
        
        model = self._warmup_training(X_train, y_train, X_val, y_val, checkpoint_path)
        
        # Phase 2: Invalid read augmentation
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: INVALID READ AUGMENTATION")
        logger.info("="*60)
        
        model = self._invalid_augmentation_training(
            model, train_reads, val_reads, label_to_idx, checkpoint_path
        )
        
        # Phase 3: Pseudo-label training (if unlabeled data provided)
        if unlabeled_fastq and Path(unlabeled_fastq).exists():
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: PSEUDO-LABEL TRAINING")
            logger.info("="*60)
            
            model = self._pseudo_label_training(
                model, unlabeled_fastq, label_to_idx, checkpoint_path
            )
        else:
            logger.info("\nSkipping Phase 3: No unlabeled data provided")
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION")
        logger.info("="*60)
        
        results = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Final Validation Loss: {results[0]:.4f}")
        logger.info(f"Final Validation Accuracy: {results[1]:.4f}")
        
        # Save final model
        final_path = checkpoint_path / "model_hybrid_final.h5"
        model.save(str(final_path))
        logger.info(f"Saved final hybrid model to: {final_path}")
        
        return model
    
    def _warmup_training(self, X_train, y_train, X_val, y_val, 
                        checkpoint_path) -> keras.Model:
        """Phase 1: Standard training on clean data."""
        model = build_model_from_config(self.config)
        print_model_summary(model)
        
        # Compile
        learning_rate = self.config.training.learning_rate if self.config.training else 0.001
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        batch_size = self.config.model.batch_size
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.warmup_epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save warmup checkpoint
        warmup_path = checkpoint_path / "model_warmup.h5"
        model.save(str(warmup_path))
        logger.info(f"Warmup model saved to: {warmup_path}")
        
        return model
    
    def _invalid_augmentation_training(self, model, train_reads, val_reads,
                                      label_to_idx, checkpoint_path) -> keras.Model:
        """Phase 2: Training with invalid read augmentation."""
        
        # Generate invalid reads
        logger.info(f"Generating invalid reads (ratio={self.invalid_ratio})...")
        invalid_train = []
        for read in train_reads[:int(len(train_reads) * self.invalid_ratio)]:
            invalid_train.append(self.invalid_generator.generate_invalid_read(read))
        
        invalid_val = []
        for read in val_reads[:int(len(val_reads) * self.invalid_ratio * 0.5)]:
            invalid_val.append(self.invalid_generator.generate_invalid_read(read))
        
        logger.info(f"Generated {len(invalid_train)} training and "
                   f"{len(invalid_val)} validation invalid reads")
        
        # Combine valid and invalid
        all_train = train_reads + invalid_train
        all_val = val_reads + invalid_val
        
        # Convert to arrays
        X_train_aug, y_train_aug, _ = reads_to_arrays(all_train, label_to_idx=label_to_idx)
        X_val_aug, y_val_aug, _ = reads_to_arrays(all_val, label_to_idx=label_to_idx)
        
        # Pad and convert
        max_len = self.config.model.max_seq_len
        num_labels = self.config.model.num_labels
        
        X_train_aug, y_train_aug = pad_sequences(X_train_aug, y_train_aug, max_len)
        X_val_aug, y_val_aug = pad_sequences(X_val_aug, y_val_aug, max_len)
        
        y_train_aug = convert_labels_to_categorical(y_train_aug, num_labels)
        y_val_aug = convert_labels_to_categorical(y_val_aug, num_labels)
        
        # Continue training
        batch_size = self.config.model.batch_size
        
        history = model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val_aug, y_val_aug),
            epochs=self.discriminator_epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
                )
            ],
            verbose=1
        )
        
        # Save augmentation checkpoint
        aug_path = checkpoint_path / "model_augmented.h5"
        model.save(str(aug_path))
        logger.info(f"Augmented model saved to: {aug_path}")
        
        return model
    
    def _pseudo_label_training(self, model, unlabeled_fastq: str,
                              label_to_idx: Dict[str, int],
                              checkpoint_path) -> keras.Model:
        """Phase 3: Self-training with pseudo-labels."""
        
        # Generate pseudo-labels
        pseudo_generator = PseudoLabelGenerator(
            model, self.config, 
            confidence_threshold=self.pseudo_label_conf,
            label_to_idx=label_to_idx
        )
        
        max_unlabeled = self.config.hybrid.max_pseudo_examples if self.config.hybrid else 1000
        pseudo_reads = pseudo_generator.generate_from_fastq(
            unlabeled_fastq, max_reads=max_unlabeled
        )
        
        if not pseudo_reads:
            logger.warning("No high-confidence pseudo-labels generated")
            return model
        
        logger.info(f"Generated {len(pseudo_reads)} high-confidence pseudo-labels")
        
        # Convert to arrays
        X_pseudo, y_pseudo, _ = reads_to_arrays(pseudo_reads, label_to_idx=label_to_idx)
        
        # Pad and convert
        max_len = self.config.model.max_seq_len
        num_labels = self.config.model.num_labels
        
        X_pseudo, y_pseudo = pad_sequences(X_pseudo, y_pseudo, max_len)
        y_pseudo = convert_labels_to_categorical(y_pseudo, num_labels)
        
        # Fine-tune with pseudo-labels
        batch_size = self.config.model.batch_size
        
        # Use lower learning rate for pseudo-label training
        model.optimizer.learning_rate = 1e-4
        
        history = model.fit(
            X_pseudo, y_pseudo,
            epochs=self.pseudo_epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save pseudo-label checkpoint
        pseudo_path = checkpoint_path / "model_pseudo.h5"
        model.save(str(pseudo_path))
        logger.info(f"Pseudo-label model saved to: {pseudo_path}")
        
        return model
