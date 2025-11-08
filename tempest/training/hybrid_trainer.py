"""
Hybrid robustness training for Tempest with directory support.

Combines discriminator-based adversarial training and pseudo-label
self-training for models robust to segment-level errors.
Supports both single FASTQ files and directories of FASTQ files.

Part of: tempest/training/ module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import glob
from tqdm import tqdm

# Import from proper tempest modules
from tempest.data.simulator import SimulatedRead, reads_to_arrays
from tempest.data.invalid_generator import InvalidReadGenerator
from tempest.config import TempestConfig
from tempest.utils.io import load_fastq, ensure_dir

logger = logging.getLogger(__name__)


ArrayLike = Union[np.ndarray, tf.Tensor]

def pad_sequences(sequences: ArrayLike, labels: ArrayLike, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad sequences and labels to max_length, handling both NumPy and TensorFlow inputs."""
    # Convert to NumPy if TensorFlow tensors
    if isinstance(sequences, tf.Tensor):
        sequences = sequences.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    # Ensure NumPy-compatible dtypes
    seq_dtype = getattr(sequences.dtype, "as_numpy_dtype", sequences.dtype)
    lab_dtype = getattr(labels.dtype, "as_numpy_dtype", labels.dtype)

    n, curr_len = sequences.shape
    if curr_len == max_length:
        return sequences, labels

    pad_len = min(curr_len, max_length)
    padded_seq = np.zeros((n, max_length), dtype=seq_dtype)
    padded_lab = np.zeros((n, max_length), dtype=lab_dtype)

    padded_seq[:, :pad_len] = sequences[:, :pad_len]
    padded_lab[:, :pad_len] = labels[:, :pad_len]
    return padded_seq, padded_lab


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
    x = layers.Embedding(5, embedding_dim, mask_zero=True)(inputs)
    
    # Optional CNN layers
    if use_cnn:
        conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        x = layers.Concatenate()([x, conv1, conv2])
    
    # LSTM layers
    for i in range(lstm_layers):
        return_sequences = True  # Always return sequences for sequence labeling
        if use_bilstm:
            x = layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=dropout)
            )(x)
        else:
            x = layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=dropout)(x)
        
        if i < lstm_layers - 1:  # Add batch norm between layers
            x = layers.BatchNormalization()(x)
    
    # Output layer
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_labels, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='tempest_model')
    
    return model


def print_model_summary(model: keras.Model):
    """Print a formatted model summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


class ArchitectureDiscriminator:
    """
    Discriminator network for distinguishing valid/invalid architectures.
    
    Used in adversarial training to improve model robustness.
    """
    
    def __init__(self, num_labels: int, hidden_dim: int = 64):
        """
        Initialize discriminator.
        
        Args:
            num_labels: Number of label classes
            hidden_dim: Hidden layer dimension
        """
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build discriminator model."""
        inputs = keras.Input(shape=(None, self.num_labels))  # Variable length
        
        # Bidirectional LSTM to capture sequence patterns
        x = layers.Bidirectional(layers.LSTM(self.hidden_dim, return_sequences=True))(inputs)
        x = layers.BatchNormalization()(x)
        
        # Global pooling to get fixed-size representation
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        x = layers.Dense(self.hidden_dim, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Binary output: valid (1) or invalid (0)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
        return model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile discriminator model."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Predict if sequences are valid.
        
        Args:
            predictions: Model predictions [batch, seq_len, num_labels]
            
        Returns:
            Validity scores [batch]
        """
        return self.model.predict(predictions, verbose=0).squeeze()


class PseudoLabelGenerator:
    """
    Pseudo-label generator supporting both single files and directories.
    
    Generates pseudo-labels for unlabeled sequences using a trained model.
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
        if self.config.simulation:
            labels = self.config.simulation.sequence_order
            return {label: idx for idx, label in enumerate(labels)}
        return {'ADAPTER5': 0, 'UMI': 1, 'ACC': 2, 
                'BARCODE': 3, 'INSERT': 4, 'ADAPTER3': 5}
    
    def generate_from_path(self, path: Union[str, Path],
                          max_reads_per_file: int = 1000,
                          max_total_reads: int = 10000) -> List[SimulatedRead]:
        """
        Generate pseudo-labeled reads from either a file or directory.
        
        Args:
            path: Path to FASTQ file or directory
            max_reads_per_file: Maximum reads per file (for directory)
            max_total_reads: Maximum total reads (for directory)
            
        Returns:
            List of pseudo-labeled SimulatedRead objects
        """
        path_obj = Path(path)
        
        if path_obj.is_file():
            return self.generate_from_fastq(str(path), max_reads=max_reads_per_file)
        elif path_obj.is_dir():
            return self.generate_from_directory(
                str(path),
                max_reads_per_file=max_reads_per_file,
                max_total_reads=max_total_reads
            )
        else:
            logger.error(f"Path does not exist or is not accessible: {path}")
            return []
    
    def generate_from_directory(self, directory_path: str,
                               max_reads_per_file: int = 1000,
                               max_total_reads: int = 10000,
                               file_pattern: str = "*.fastq*") -> List[SimulatedRead]:
        """
        Generate pseudo-labeled reads from a directory of FASTQ files.
        
        Args:
            directory_path: Path to directory containing FASTQ files
            max_reads_per_file: Maximum reads to process per file
            max_total_reads: Maximum total reads across all files
            file_pattern: Glob pattern for FASTQ files
            
        Returns:
            List of pseudo-labeled SimulatedRead objects
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        if not directory.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            return []
        
        # Find all FASTQ files
        fastq_files = []
        patterns = ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz"] if file_pattern == "*.fastq*" else [file_pattern]
        
        for pattern in patterns:
            fastq_files.extend(directory.glob(pattern))
        
        if not fastq_files:
            logger.warning(f"No FASTQ files found in {directory_path} with pattern(s): {patterns}")
            return []
        
        logger.info(f"Found {len(fastq_files)} FASTQ files in {directory_path}")
        
        # Process files
        all_pseudo_reads = []
        total_processed = 0
        
        # Use tqdm only if available
        try:
            from tqdm import tqdm
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            logger.debug("tqdm not available, proceeding without progress bar")
        
        if use_progress_bar:
            pbar = tqdm(total=min(max_total_reads, len(fastq_files) * max_reads_per_file),
                       desc="Generating pseudo-labels", unit="reads")
        
        for fastq_file in fastq_files:
            if total_processed >= max_total_reads:
                logger.info(f"Reached maximum total reads ({max_total_reads})")
                break
            
            remaining = max_total_reads - total_processed
            reads_for_this_file = min(max_reads_per_file, remaining)
            
            logger.debug(f"Processing {fastq_file.name} (max {reads_for_this_file} reads)")
            
            try:
                file_pseudo_reads = self.generate_from_fastq(
                    str(fastq_file),
                    max_reads=reads_for_this_file
                )
                
                all_pseudo_reads.extend(file_pseudo_reads)
                processed_count = len(file_pseudo_reads)
                total_processed += processed_count
                
                if use_progress_bar:
                    pbar.update(processed_count)
                
                logger.debug(f"Generated {processed_count} pseudo-labels from {fastq_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {fastq_file}: {e}")
                continue
        
        if use_progress_bar:
            pbar.close()
        
        logger.info(f"Total pseudo-labeled reads generated: {len(all_pseudo_reads)} from {len(fastq_files)} files")
        return all_pseudo_reads
    
    def generate_from_fastq(self, fastq_file: str, 
                           max_reads: int = 1000) -> List[SimulatedRead]:
        """
        Generate pseudo-labeled reads from a single FASTQ file.
        
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
        
        logger.debug(f"Generating pseudo-labels from {fastq_file}")
        
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
        
        logger.debug(f"Generated {len(pseudo_labeled)} pseudo-labels from "
                    f"{min(max_reads, i+1) if 'i' in locals() else 0} sequences")
        return pseudo_labeled
    
    def _process_batch(self, sequences: List[str], 
                      read_ids: List[str]) -> List[SimulatedRead]:
        """Process a batch of sequences."""
        batch_reads = []
        
        # Try both forward and reverse orientations
        for orientation in ['forward', 'reverse']:
            if orientation == 'reverse':
                sequences_to_process = [self._reverse_complement(seq) for seq in sequences]
            else:
                sequences_to_process = sequences
            
            # Convert to array
            max_len = self.config.model.max_seq_len
            X_batch = self._sequences_to_array(sequences_to_process, max_len)
            
            # Predict
            predictions = self.model.predict(X_batch, verbose=0)
            
            # Process each sequence
            for i, (seq, pred) in enumerate(zip(sequences_to_process, predictions)):
                # Calculate confidence
                confidences = np.max(pred, axis=-1)[:len(seq)]
                mean_conf = np.mean(confidences)
                
                if mean_conf < self.confidence_threshold:
                    continue
                
                # Convert predictions to labels
                label_indices = np.argmax(pred, axis=-1)[:len(seq)]
                labels = [self.idx_to_label.get(idx, 'UNKNOWN') 
                         for idx in label_indices]
                
                # Create label regions
                label_regions = self._extract_regions(labels)
                
                # Validate architecture if configured
                if self.config.hybrid and self.config.hybrid.validate_architecture:
                    if not self._validate_architecture(label_regions):
                        continue
                
                batch_reads.append(SimulatedRead(
                    sequence=seq,
                    labels=labels,
                    label_regions=label_regions,
                    metadata={'pseudo_label': True, 
                             'confidence': float(mean_conf),
                             'read_id': read_ids[i],
                             'orientation': orientation}
                ))
        
        return batch_reads
    
    def _sequences_to_array(self, sequences: List[str], max_len: int) -> np.ndarray:
        """Convert sequences to numpy array."""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        X = np.zeros((len(sequences), max_len), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq[:max_len]):
                X[i, j] = base_to_idx.get(base.upper(), 4)
        
        return X
    
    def _reverse_complement(self, seq: str) -> str:
        """Get reverse complement of sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))
    
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
    
    def _validate_architecture(self, label_regions: Dict[str, List[Tuple[int, int]]]) -> bool:
        """
        Validate that the predicted architecture is reasonable.
        
        Args:
            label_regions: Dictionary of label regions
            
        Returns:
            True if architecture is valid
        """
        # Check for minimum unique segments
        if self.config.hybrid.min_unique_segments:
            if len(label_regions) < self.config.hybrid.min_unique_segments:
                return False
        
        # Check for maximum segment repetition
        if self.config.hybrid.max_segment_repetition:
            for label, regions in label_regions.items():
                if len(regions) > self.config.hybrid.max_segment_repetition:
                    return False
        
        # Check for expected segment order if specified
        if self.config.simulation and self.config.simulation.sequence_order:
            expected_order = self.config.simulation.sequence_order
            label_positions = []
            
            for label in expected_order:
                if label in label_regions:
                    first_region = min(label_regions[label], key=lambda x: x[0])
                    label_positions.append((label, first_region[0]))
            
            # Check if labels appear in expected order
            for i in range(len(label_positions) - 1):
                if label_positions[i][1] > label_positions[i+1][1]:
                    return False
        
        return True


class HybridTrainer:
    """
    Implements hybrid robustness training with directory support.
    
    Training proceeds in three phases:
    1. Warmup: Standard training on valid reads
    2. Discriminator: Training with invalid reads and discriminator
    3. Pseudo-label: Fine-tuning with pseudo-labeled unlabeled data
    """
    
    def __init__(self, config: TempestConfig, base_model=None):
        """
        Initialize hybrid trainer.
        
        Args:
            config: Tempest configuration with hybrid settings
            base_model: Optional pre-built model (otherwise built from config)
        """
        self.config = config
        self.base_model = base_model or build_model_from_config(config)
        
        # Initialize components
        self.discriminator = ArchitectureDiscriminator(
            num_labels=config.model.num_labels,
            hidden_dim=config.hybrid.discriminator_hidden_dim if config.hybrid else 64
        )
        self.invalid_generator = InvalidReadGenerator(config)
        self.pseudo_generator = None  # Created after model is trained
        
        # Training state
        self.phase = 'warmup'
        self.current_epoch = 0
        
        # Get phase schedule from config
        if config.hybrid:
            self.phase_schedule = {
                'warmup': config.hybrid.warmup_epochs,
                'discriminator': config.hybrid.discriminator_epochs,
                'pseudo_label': config.hybrid.pseudolabel_epochs
            }
            self.invalid_weight = config.hybrid.invalid_weight_initial
            self.invalid_weight_max = config.hybrid.invalid_weight_max
            self.adversarial_weight = config.hybrid.adversarial_weight
            self.invalid_ratio = config.hybrid.invalid_ratio
            # Directory processing parameters
            self.max_pseudo_per_file = getattr(config.hybrid, 'max_pseudo_per_file', 1000)
            self.max_pseudo_total = getattr(config.hybrid, 'max_pseudo_total', 10000)
        else:
            self.phase_schedule = {
                'warmup': 5,
                'discriminator': 10,
                'pseudo_label': 10
            }
            self.invalid_weight = 0.1
            self.invalid_weight_max = 0.3
            self.adversarial_weight = 0.1
            self.invalid_ratio = 0.1
            self.max_pseudo_per_file = 1000
            self.max_pseudo_total = 10000
        
        logger.info(f"Initialized HybridTrainer with phase schedule: {self.phase_schedule}")
    
    def train(self, train_reads: List[SimulatedRead],
             val_reads: List[SimulatedRead],
             unlabeled_path: Optional[Union[str, Path]] = None,
             checkpoint_dir: str = './checkpoints') -> keras.Model:
        """
        Run complete hybrid training pipeline.
        
        Args:
            train_reads: Training SimulatedRead objects
            val_reads: Validation SimulatedRead objects  
            unlabeled_path: Path to unlabeled FASTQ file or directory
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
        if unlabeled_path:
            path_obj = Path(unlabeled_path) if isinstance(unlabeled_path, str) else unlabeled_path
            
            if path_obj.exists():
                logger.info("\n" + "="*60)
                logger.info("PHASE 3: PSEUDO-LABEL TRAINING")
                
                if path_obj.is_dir():
                    logger.info(f"Processing directory: {path_obj}")
                else:
                    logger.info(f"Processing file: {path_obj}")
                
                logger.info("="*60)
                
                model = self._pseudo_label_training(
                    model, unlabeled_path, label_to_idx, checkpoint_path
                )
            else:
                logger.warning(f"Unlabeled path does not exist: {unlabeled_path}")
                logger.info("Skipping Phase 3: Invalid path provided")
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
            verbose=1
        )
        
        # Save augmentation checkpoint
        aug_path = checkpoint_path / "model_augmented.h5"
        model.save(str(aug_path))
        logger.info(f"Augmented model saved to: {aug_path}")
        
        return model
    
    def _pseudo_label_training(self, model, unlabeled_path: Union[str, Path],
                              label_to_idx: Dict[str, int],
                              checkpoint_path) -> keras.Model:
        """Phase 3: Self-training with pseudo-labels from file or directory."""
        
        # Generate pseudo-labels
        pseudo_generator = PseudoLabelGenerator(
            model, self.config, 
            confidence_threshold=self.pseudo_label_conf,
            label_to_idx=label_to_idx
        )
        
        # Generate from path (handles both files and directories)
        pseudo_reads = pseudo_generator.generate_from_path(
            unlabeled_path,
            max_reads_per_file=self.max_pseudo_per_file,
            max_total_reads=self.max_pseudo_total
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
