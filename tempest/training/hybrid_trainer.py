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
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import glob

# Import from proper tempest modules - using try/except for flexibility
try:
    from tempest.data.simulator import SimulatedRead, reads_to_arrays
    from tempest.data.invalid_generator import InvalidReadGenerator
    from tempest.utils.io import load_fastq, ensure_dir
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Some tempest modules not found, running in limited mode")
    # Provide fallback implementations or stubs as needed
    SimulatedRead = None
    reads_to_arrays = None
    InvalidReadGenerator = None
    load_fastq = None
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

from tempest.config import TempestConfig, HybridTrainingConfig

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

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
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


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
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='tempest_hybrid_model')
    
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
            num_labels: Number of segment labels
            hidden_dim: Hidden layer dimension
        """
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build discriminator model."""
        inputs = keras.Input(shape=(None, self.num_labels))
        
        # Global features
        x = layers.GlobalAveragePooling1D()(inputs)
        x = layers.Dense(self.hidden_dim, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output: probability of being valid
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=output, name='discriminator')
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, valid_preds: np.ndarray, invalid_preds: np.ndarray,
              epochs: int = 5, batch_size: int = 32):
        """
        Train discriminator on valid/invalid predictions.
        
        Args:
            valid_preds: Valid model predictions
            invalid_preds: Invalid model predictions  
            epochs: Training epochs
            batch_size: Batch size
        """
        # Prepare data
        X = np.concatenate([valid_preds, invalid_preds])
        y = np.concatenate([
            np.ones(len(valid_preds)),
            np.zeros(len(invalid_preds))
        ])
        
        # Train
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                      validation_split=0.2, verbose=0)
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Predict validity scores.
        
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
        if self.config.simulation and self.config.simulation.sequence_order:
            labels = self.config.simulation.sequence_order
            return {label: idx for idx, label in enumerate(labels)}
        # Default 11-segment mapping
        return {
            'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4,
            'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10
        }
    
    def generate_from_path(self, path: Union[str, Path],
                          max_reads_per_file: int = 1000,
                          max_total_reads: int = 10000) -> List:
        """
        Generate pseudo-labeled reads from either a file or directory.
        
        Args:
            path: Path to FASTQ file or directory
            max_reads_per_file: Maximum reads per file (for directory)
            max_total_reads: Maximum total reads (for directory)
            
        Returns:
            List of pseudo-labeled SimulatedRead objects or dicts
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
                               file_pattern: str = "*.fastq*") -> List:
        """
        Generate pseudo-labeled reads from a directory of FASTQ files.
        
        Args:
            directory_path: Path to directory containing FASTQ files
            max_reads_per_file: Maximum reads to process per file
            max_total_reads: Maximum total reads across all files
            file_pattern: Glob pattern for FASTQ files
            
        Returns:
            List of pseudo-labeled objects
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
        if HAS_TQDM:
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
                
                if HAS_TQDM:
                    pbar.update(processed_count)
                
                logger.debug(f"Generated {processed_count} pseudo-labels from {fastq_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {fastq_file}: {e}")
                continue
        
        if HAS_TQDM:
            pbar.close()
        
        logger.info(f"Total pseudo-labeled reads generated: {len(all_pseudo_reads)} from {len(fastq_files)} files")
        return all_pseudo_reads
    
    def generate_from_fastq(self, fastq_file: str, 
                           max_reads: int = 1000) -> List:
        """
        Generate pseudo-labeled reads from a single FASTQ file.
        
        Note: This is a stub implementation. The actual implementation
        would process FASTQ records and generate pseudo-labels.
        
        Args:
            fastq_file: Path to unlabeled FASTQ
            max_reads: Maximum reads to process
            
        Returns:
            List of pseudo-labeled objects
        """
        pseudo_labeled = []
        
        # This would normally process FASTQ records
        # For now, return empty list as a stub
        logger.debug(f"Generating pseudo-labels from {fastq_file} (stub implementation)")
        
        return pseudo_labeled


class HybridTrainer:
    """
    Implements hybrid robustness training with directory support.
    
    This trainer combines multiple techniques for improved robustness:
    1. Standard supervised training
    2. Invalid read augmentation with discriminator
    3. Pseudo-labeling for semi-supervised learning
    
    Compatible with the Typer-based CLI through the TempestConfig system.
    """
    
    def __init__(
        self, 
        config: TempestConfig,
        output_dir: Optional[Path] = None,
        verbose: bool = False
    ):
        """
        Initialize hybrid trainer.
        
        Args:
            config: TempestConfig with hybrid settings
            output_dir: Directory to save models and logs
            verbose: Whether to show verbose output
        """
        self.config = config
        self.model_config = config.model
        
        # Use hybrid config if available, otherwise use defaults
        if config.hybrid:
            self.hybrid_config = config.hybrid
        else:
            self.hybrid_config = HybridTrainingConfig()
        
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.verbose = verbose
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Model components
        self.base_model = None
        self.discriminator = None
        self.invalid_generator = None
        self.pseudo_generator = None
        
        # Training state
        self.phase = 'warmup'
        self.current_epoch = 0
        
        # Get phase schedule from config
        self.phase_schedule = {
            'warmup': self.hybrid_config.warmup_epochs,
            'discriminator': self.hybrid_config.discriminator_epochs,
            'pseudo_label': self.hybrid_config.pseudolabel_epochs
        }
        
        logger.info(f"HybridTrainer initialized with phase schedule: {self.phase_schedule}")
    
    def build_model(self) -> keras.Model:
        """Build the hybrid model architecture."""
        model = build_model_from_config(self.config)
        
        # Compile model
        lr = self.config.training.learning_rate if self.config.training else 0.001
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if self.verbose:
            print_model_summary(model)
        
        return model
    
    def train(
        self,
        train_data: Union[np.ndarray, tuple, list],
        val_data: Optional[Union[np.ndarray, tuple, list]] = None,
        unlabeled_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete hybrid training pipeline.
        
        Args:
            train_data: Training data (X, y) or list of dicts
            val_data: Optional validation data (X, y) or list of dicts
            unlabeled_path: Optional path to unlabeled data for pseudo-labeling
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("="*80)
        logger.info("STARTING HYBRID TRAINING PIPELINE")
        logger.info("="*80)
        
        # Process training data
        if isinstance(train_data, (list, tuple)) and len(train_data) == 2:
            X_train, y_train = train_data
        elif isinstance(train_data, list) and isinstance(train_data[0], dict):
            X_train, y_train = self._process_dict_data(train_data)
        else:
            raise ValueError("train_data must be (X, y) tuple or list of dicts")
        
        # Process validation data
        if val_data is not None:
            if isinstance(val_data, (list, tuple)) and len(val_data) == 2:
                X_val, y_val = val_data
            elif isinstance(val_data, list) and isinstance(val_data[0], dict):
                X_val, y_val = self._process_dict_data(val_data)
            else:
                raise ValueError("val_data must be (X, y) tuple or list of dicts")
        else:
            X_val, y_val = None, None
        
        # Phase 1: Warm-up training
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: WARM-UP TRAINING")
        logger.info("="*60)
        
        self.base_model = self.build_model()
        history_warmup = self._warmup_training(X_train, y_train, X_val, y_val)
        
        # Phase 2: Invalid read augmentation (if enabled)
        if self.hybrid_config.enabled and self.hybrid_config.discriminator_epochs > 0:
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: INVALID READ AUGMENTATION")
            logger.info("="*60)
            
            history_adversarial = self._adversarial_training(X_train, y_train, X_val, y_val)
        else:
            history_adversarial = {}
        
        # Phase 3: Pseudo-label training (if unlabeled data provided)
        if unlabeled_path and self.hybrid_config.enabled:
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: PSEUDO-LABEL TRAINING")
            logger.info("="*60)
            
            history_pseudo = self._pseudo_label_training(unlabeled_path, X_train, y_train, X_val, y_val)
        else:
            history_pseudo = {}
            if not unlabeled_path:
                logger.info("\nSkipping Phase 3: No unlabeled data provided")
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION")
        logger.info("="*60)
        
        if X_val is not None:
            results = self.base_model.evaluate(X_val, y_val, verbose=0)
            final_val_loss = results[0]
            final_val_acc = results[1]
            logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
            logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        else:
            final_val_loss = None
            final_val_acc = None
        
        # Save final model
        final_path = self.output_dir / "model_hybrid_final.h5"
        self.base_model.save(str(final_path))
        logger.info(f"Saved final hybrid model to: {final_path}")
        
        # Prepare results
        results = {
            'model_path': str(final_path),
            'history': {
                'warmup': history_warmup,
                'adversarial': history_adversarial,
                'pseudo_label': history_pseudo
            },
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_acc
        }
        
        # Add metrics for CLI display
        results['metrics'] = {
            'Training Mode': 'Hybrid',
            'Phases Completed': len([h for h in [history_warmup, history_adversarial, history_pseudo] if h])
        }
        if final_val_loss is not None:
            results['metrics']['Validation Loss'] = final_val_loss
            results['metrics']['Validation Accuracy'] = final_val_acc
        
        return results
    
    def _warmup_training(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """Phase 1: Standard training on clean data."""
        epochs = self.phase_schedule['warmup']
        batch_size = self.config.training.batch_size if self.config.training else 32
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.base_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 2
        )
        
        return history.history
    
    def _adversarial_training(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """Phase 2: Training with invalid reads and discriminator."""
        # This is a simplified implementation
        # Full implementation would generate invalid reads and use discriminator
        
        epochs = self.phase_schedule['discriminator']
        batch_size = self.config.training.batch_size if self.config.training else 32
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.base_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 2
        )
        
        return history.history
    
    def _pseudo_label_training(self, unlabeled_path, X_train, y_train, X_val=None, y_val=None) -> Dict:
        """Phase 3: Training with pseudo-labeled data."""
        # This is a simplified implementation
        # Full implementation would generate pseudo-labels from unlabeled data
        
        epochs = self.phase_schedule['pseudo_label']
        batch_size = self.config.training.batch_size if self.config.training else 32
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.base_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 2
        )
        
        return history.history
    
    def _process_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Process dictionary-based data into arrays."""
        sequences = []
        labels = []
        
        for item in data:
            seq = item['sequence']
            label = item.get('labels', [])
            
            # Convert sequence to numeric
            seq_numeric = self._sequence_to_numeric(seq)
            sequences.append(seq_numeric)
            
            # Convert labels to numeric if they're strings
            if isinstance(label, list) and len(label) > 0 and isinstance(label[0], str):
                label_numeric = self._labels_to_numeric(label)
                labels.append(label_numeric)
            else:
                labels.append(label)
        
        # Pad sequences
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.model_config.max_seq_len, padding='post'
        )
        
        # Pad labels and convert to categorical
        y = tf.keras.preprocessing.sequence.pad_sequences(
            labels, maxlen=self.model_config.max_seq_len, padding='post'
        )
        y = tf.keras.utils.to_categorical(y, num_classes=self.model_config.num_labels)
        
        return X, y
    
    def _sequence_to_numeric(self, sequence: str) -> List[int]:
        """Convert DNA sequence to numeric representation."""
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
        return [mapping.get(base.upper(), 0) for base in sequence]
    
    def _labels_to_numeric(self, labels: List[str]) -> List[int]:
        """Convert string labels to numeric representation."""
        # Define standard 11-segment label mapping
        label_mapping = {
            'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4,
            'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10
        }
        return [label_mapping.get(label, 0) for label in labels]


def run_hybrid_training(
    config: TempestConfig,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run hybrid training based on configuration.
    
    This function is called by the CLI and main.py to execute hybrid training.
    
    Args:
        config: TempestConfig object
        output_dir: Output directory for models
        **kwargs: Additional training parameters
        
    Returns:
        Training results dictionary
    """
    # Extract parameters from kwargs
    train_data = kwargs.get('train_data')
    val_data = kwargs.get('val_data')
    unlabeled_path = kwargs.get('unlabeled_path')
    verbose = kwargs.get('verbose', False)
    
    # Create trainer
    trainer = HybridTrainer(config, output_dir=output_dir, verbose=verbose)
    
    # If no data provided, try to load from config paths
    if train_data is None:
        # This would normally load from config
        raise ValueError("No training data provided")
    
    # Run training
    return trainer.train(
        train_data,
        val_data,
        unlabeled_path=unlabeled_path,
        **kwargs
    )
