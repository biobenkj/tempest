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

from simulator import SimulatedRead, reads_to_arrays
from models import build_model_from_config, convert_labels_to_categorical, pad_sequences
from config import TempestConfig
from io import load_fastq, ensure_dir
from invalid_generator import InvalidReadGenerator

logger = logging.getLogger(__name__)


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
        if self.config.simulation:
            labels = self.config.simulation.sequence_order
            return {label: idx for idx, label in enumerate(labels)}
        return {'ADAPTER5': 0, 'UMI': 1, 'ACC': 2, 
                'BARCODE': 3, 'INSERT': 4, 'ADAPTER3': 5}
    
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
                   f"{min(max_reads, i+1)} sequences")
        return pseudo_labeled
    
    def _process_batch(self, sequences: List[str], 
                      read_ids: List[str]) -> List[SimulatedRead]:
        """Process a batch of sequences."""
        # Convert to arrays
        X = self._sequences_to_array(sequences)
        
        # Predict forward
        preds_fwd = self.model.predict(X, verbose=0)
        conf_fwd = self._compute_confidence(preds_fwd)
        
        # Predict reverse complement
        X_rev = self._reverse_complement_array(X)
        preds_rev = self.model.predict(X_rev, verbose=0)
        conf_rev = self._compute_confidence(preds_rev)
        
        # Select best orientation and filter by confidence
        pseudo_reads = []
        
        for i in range(len(sequences)):
            if max(conf_fwd[i], conf_rev[i]) > self.confidence_threshold:
                if conf_fwd[i] > conf_rev[i]:
                    # Use forward
                    labels = np.argmax(preds_fwd[i], axis=-1)
                    if self._validate_architecture(labels):
                        pseudo_reads.append(self._create_read(
                            sequences[i], labels, conf_fwd[i], 'forward'
                        ))
                else:
                    # Use reverse
                    labels = np.argmax(preds_rev[i], axis=-1)
                    if self._validate_architecture(labels):
                        rev_seq = self._reverse_complement_string(sequences[i])
                        pseudo_reads.append(self._create_read(
                            rev_seq, labels, conf_rev[i], 'reverse'
                        ))
        
        return pseudo_reads
    
    def _sequences_to_array(self, sequences: List[str]) -> np.ndarray:
        """Convert sequences to numeric array."""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        max_len = self.config.model.max_seq_len
        
        X = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq.upper()[:max_len]):
                X[i, j] = base_to_idx.get(base, 4)
        
        return X
    
    def _reverse_complement_array(self, X: np.ndarray) -> np.ndarray:
        """Reverse complement numeric array."""
        complement = np.array([3, 2, 1, 0, 4])  # A->T, C->G, G->C, T->A, N->N
        X_rev = X[:, ::-1]  # Reverse
        return complement[X_rev]  # Complement
    
    def _reverse_complement_string(self, seq: str) -> str:
        """Reverse complement a sequence string."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in seq.upper()[::-1])
    
    def _compute_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Compute confidence as 1 - normalized entropy."""
        entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=-1)
        max_entropy = np.log(predictions.shape[-1])
        confidence = 1 - (entropy / max_entropy)
        return np.mean(confidence, axis=1)  # Average over sequence
    
    def _validate_architecture(self, labels: np.ndarray) -> bool:
        """Check if predicted architecture is plausible."""
        # Get validation parameters from config
        if self.config.hybrid and self.config.hybrid.validate_architecture:
            min_segments = self.config.hybrid.min_unique_segments
            max_rep = self.config.hybrid.max_segment_repetition
        else:
            min_segments = 3
            max_rep = 2
        
        # Must have minimum number of different segment types
        unique_labels = np.unique(labels)
        if len(unique_labels) < min_segments:
            return False
        
        # Check for excessive repetition
        for label_idx in unique_labels:
            segments = self._count_segments(labels, label_idx)
            if segments > max_rep:
                return False
        
        return True
    
    def _count_segments(self, labels: np.ndarray, label_idx: int) -> int:
        """Count number of segments for a given label."""
        count = 0
        in_segment = False
        
        for label in labels:
            if label == label_idx:
                if not in_segment:
                    count += 1
                    in_segment = True
            else:
                in_segment = False
        
        return count
    
    def _create_read(self, sequence: str, labels: np.ndarray, 
                    confidence: float, orientation: str) -> SimulatedRead:
        """Create SimulatedRead from predictions."""
        # Convert labels to list
        label_list = [self.idx_to_label.get(int(idx), 'UNKNOWN') 
                     for idx in labels[:len(sequence)]]
        
        # Build label regions
        regions = {}
        if label_list:
            current = label_list[0]
            start = 0
            
            for i, label in enumerate(label_list[1:], 1):
                if label != current:
                    if current not in regions:
                        regions[current] = []
                    regions[current].append((start, i))
                    current = label
                    start = i
            
            # Add final region
            if current not in regions:
                regions[current] = []
            regions[current].append((start, len(label_list)))
        
        return SimulatedRead(
            sequence=sequence,
            labels=label_list,
            label_regions=regions,
            metadata={
                'confidence': confidence,
                'orientation': orientation,
                'source': 'pseudo_label'
            }
        )


class HybridTrainer:
    """
    Orchestrate hybrid training with invalid reads and pseudo-labels.
    
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
        
        logger.info(f"Initialized HybridTrainer with phase schedule: {self.phase_schedule}")
    
    def train(self, train_reads: List[SimulatedRead],
             val_reads: List[SimulatedRead],
             unlabeled_fastq: Optional[str] = None,
             checkpoint_dir: str = './checkpoints') -> keras.Model:
        """
        Run complete hybrid training pipeline.
        
        Args:
            train_reads: Labeled training reads
            val_reads: Labeled validation reads
            unlabeled_fastq: Optional path to unlabeled FASTQ for pseudo-labeling
            checkpoint_dir: Directory for model checkpoints
            
        Returns:
            Trained model
        """
        ensure_dir(checkpoint_dir)
        
        # Convert to arrays
        X_train, y_train, label_to_idx = reads_to_arrays(train_reads)
        X_val, y_val, _ = reads_to_arrays(val_reads, label_to_idx)
        
        # Pad sequences
        max_len = self.config.model.max_seq_len
        X_train, y_train = pad_sequences(X_train, y_train, max_len)
        X_val, y_val = pad_sequences(X_val, y_val, max_len)
        
        # Convert to categorical
        y_train = convert_labels_to_categorical(y_train, self.config.model.num_labels)
        y_val = convert_labels_to_categorical(y_val, self.config.model.num_labels)
        
        logger.info("="*80)
        logger.info("HYBRID ROBUSTNESS TRAINING MODE")
        logger.info("="*80)
        
        # Compile model
        self._compile_model()
        
        # Phase 1: Warmup
        logger.info("\n=== Switching to warmup phase ===")
        self.phase = 'warmup'
        for epoch in range(self.phase_schedule['warmup']):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch+1}/{self.phase_schedule['warmup']} [warmup]")
            self.base_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.model.batch_size,
                epochs=1,
                verbose=1
            )
        
        # Phase 2: Discriminator training
        logger.info("\n=== Switching to discriminator phase ===")
        self.phase = 'discriminator'
        for epoch in range(self.phase_schedule['discriminator']):
            self.current_epoch = epoch + self.phase_schedule['warmup']
            logger.info(f"\nEpoch {self.current_epoch+1} [discriminator]")
            
            # Generate invalid reads
            train_with_invalid = self.invalid_generator.generate_batch(
                train_reads, self.invalid_ratio
            )
            X_mixed, y_mixed, _ = reads_to_arrays(train_with_invalid, label_to_idx)
            X_mixed, y_mixed = pad_sequences(X_mixed, y_mixed, max_len)
            y_mixed = convert_labels_to_categorical(y_mixed, self.config.model.num_labels)
            
            # Train
            self.base_model.fit(
                X_mixed, y_mixed,
                validation_data=(X_val, y_val),
                batch_size=self.config.model.batch_size,
                epochs=1,
                verbose=1
            )
            
            # Gradually increase invalid weight
            self.invalid_weight = min(
                self.invalid_weight_max,
                self.invalid_weight * 1.1
            )
        
        # Phase 3: Pseudo-label fine-tuning
        if unlabeled_fastq:
            logger.info("\n=== Switching to pseudo_label phase ===")
            logger.info(f"Using unlabeled data from: {unlabeled_fastq}")
            self.phase = 'pseudo_label'
            
            # Initialize pseudo-label generator
            self.pseudo_generator = PseudoLabelGenerator(
                self.base_model, self.config,
                confidence_threshold=self.config.hybrid.confidence_threshold if self.config.hybrid else 0.9,
                label_to_idx=label_to_idx
            )
            
            for epoch in range(self.phase_schedule['pseudo_label']):
                self.current_epoch = epoch + self.phase_schedule['warmup'] + self.phase_schedule['discriminator']
                logger.info(f"\nEpoch {self.current_epoch+1} [pseudo_label]")
                
                # Generate pseudo-labels
                logger.info("Generating pseudo-labels...")
                max_pseudo = self.config.hybrid.max_pseudo_examples if self.config.hybrid else 1000
                pseudo_reads = self.pseudo_generator.generate_from_fastq(
                    unlabeled_fastq, max_reads=max_pseudo
                )
                logger.info(f"Generated {len(pseudo_reads)} pseudo-labels")
                
                if pseudo_reads:
                    # Combine with training data
                    combined_reads = train_reads + pseudo_reads
                    X_combined, y_combined, _ = reads_to_arrays(combined_reads, label_to_idx)
                    X_combined, y_combined = pad_sequences(X_combined, y_combined, max_len)
                    y_combined = convert_labels_to_categorical(y_combined, self.config.model.num_labels)
                    
                    # Train
                    self.base_model.fit(
                        X_combined, y_combined,
                        validation_data=(X_val, y_val),
                        batch_size=self.config.model.batch_size,
                        epochs=1,
                        verbose=1
                    )
                    
                    # Decay confidence threshold
                    if self.config.hybrid:
                        self.pseudo_generator.confidence_threshold *= self.config.hybrid.confidence_decay
        else:
            logger.info("\nNo unlabeled data provided, skipping pseudo-label phase")
        
        # Save final model
        final_path = Path(checkpoint_dir) / "model_hybrid_final.h5"
        self.base_model.save(str(final_path))
        logger.info(f"\nSaved hybrid-trained model to: {final_path}")
        
        return self.base_model
    
    def _compile_model(self):
        """Compile model with appropriate optimizer and loss."""
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.training.learning_rate
        )
        
        self.base_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model compiled for hybrid training")
