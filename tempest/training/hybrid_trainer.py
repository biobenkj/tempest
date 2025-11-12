"""
Hybrid robustness training for Tempest using the unified core model builder.

Combines:
  1. Standard supervised training (warm-up)
  2. Discriminator-based adversarial robustness
  3. Pseudo-label self-training for unlabeled reads

Part of: tempest/training/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# Proper Tempest imports
try:
    from tempest.config import TempestConfig, HybridTrainingConfig
    from tempest.core import build_model_from_config, print_model_summary
    from tempest.utils import ensure_dir
    from tempest.data import InvalidReadGenerator, SimulatedRead, reads_to_arrays
except ImportError:
    SimulatedRead = None
    reads_to_arrays = None
    InvalidReadGenerator = None
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# Optional tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

logger = logging.getLogger(__name__)
ArrayLike = Union[np.ndarray, tf.Tensor]


# ------------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------------
def pad_sequences(sequences: ArrayLike, labels: ArrayLike, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad sequences and labels to a fixed length."""
    if isinstance(sequences, tf.Tensor): sequences = sequences.numpy()
    if isinstance(labels, tf.Tensor): labels = labels.numpy()
    seq_dtype = getattr(sequences.dtype, "as_numpy_dtype", sequences.dtype)
    lab_dtype = getattr(labels.dtype, "as_numpy_dtype", labels.dtype)
    n, curr_len = sequences.shape
    padded_seq = np.zeros((n, max_length), dtype=seq_dtype)
    padded_lab = np.zeros((n, max_length), dtype=lab_dtype)
    pad_len = min(curr_len, max_length)
    padded_seq[:, :pad_len] = sequences[:, :pad_len]
    padded_lab[:, :pad_len] = labels[:, :pad_len]
    return padded_seq, padded_lab


def convert_labels_to_categorical(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


# ------------------------------------------------------------------------------------
# Adversarial Discriminator
# ------------------------------------------------------------------------------------
class ArchitectureDiscriminator:
    """Adversarial discriminator for distinguishing valid vs invalid architectures."""

    def __init__(self, num_labels: int, hidden_dim: int = 64):
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.model = self._build_model()

    def _build_model(self):
        inputs = keras.Input(shape=(None, self.num_labels))
        x = layers.GlobalAveragePooling1D()(inputs)
        x = layers.Dense(self.hidden_dim, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, output, name="discriminator")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, valid_preds: np.ndarray, invalid_preds: np.ndarray,
              epochs: int = 5, batch_size: int = 32):
        """Train discriminator on valid and invalid predictions."""
        X = np.concatenate([valid_preds, invalid_preds])
        y = np.concatenate([np.ones(len(valid_preds)), np.zeros(len(invalid_preds))])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Predict validity scores."""
        return self.model.predict(predictions, verbose=0).squeeze()


# ------------------------------------------------------------------------------------
# Pseudo-Label Generator
# ------------------------------------------------------------------------------------
class PseudoLabelGenerator:
    """Generates pseudo-labels for unlabeled FASTQ files or directories."""

    def __init__(self, model, config: TempestConfig, confidence_threshold: float = 0.9,
                 label_to_idx: Optional[Dict[str, int]] = None):
        self.model = model
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.label_to_idx = label_to_idx or self._default_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def _default_label_mapping(self):
        if self.config.simulation and self.config.simulation.sequence_order:
            labels = self.config.simulation.sequence_order
            return {label: idx for idx, label in enumerate(labels)}
        return {'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4,
                'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10}

    def generate_from_path(self, path: Union[str, Path],
                           max_reads_per_file: int = 1000,
                           max_total_reads: int = 10000) -> List:
        """Generate pseudo-labeled reads from file or directory."""
        path = Path(path)
        if path.is_file():
            return self.generate_from_fastq(str(path), max_reads=max_reads_per_file)
        if path.is_dir():
            return self._generate_from_directory(str(path), max_reads_per_file, max_total_reads)
        logger.error(f"Invalid path for pseudo-label generation: {path}")
        return []

    def _generate_from_directory(self, directory: str, max_reads_per_file: int, max_total_reads: int):
        """Handle directory-level FASTQ pseudo-labeling."""
        dir_path = Path(directory)
        fastq_files = [f for p in ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz"]
                       for f in dir_path.glob(p)]
        if not fastq_files:
            logger.warning(f"No FASTQ files found in {directory}")
            return []

        total_reads = []
        total_processed = 0
        pbar = tqdm(total=max_total_reads, desc="Pseudo-labeling", unit="reads") if HAS_TQDM else None

        for f in fastq_files:
            if total_processed >= max_total_reads:
                break
            n_reads = min(max_reads_per_file, max_total_reads - total_processed)
            reads = self.generate_from_fastq(str(f), max_reads=n_reads)
            total_reads.extend(reads)
            total_processed += len(reads)
            if HAS_TQDM: pbar.update(len(reads))

        if HAS_TQDM: pbar.close()
        return total_reads

    def generate_from_fastq(self, fastq_file: str, max_reads: int = 1000) -> List:
        """Placeholder pseudo-label generation (extend with FASTQ parsing)."""
        logger.debug(f"Generating pseudo-labels from {fastq_file} (stub)")
        return []


# ------------------------------------------------------------------------------------
# Hybrid Trainer
# ------------------------------------------------------------------------------------
class HybridTrainer:
    """
    Three-phase hybrid training:
      1. Warm-up (supervised)
      2. Adversarial/discriminator robustness
      3. Pseudo-label self-training
    """

    def __init__(self, config: TempestConfig, output_dir: Optional[Path] = None, verbose: bool = False):
        self.config = config
        self.model_config = config.model
        self.hybrid_config = config.hybrid or HybridTrainingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.verbose = verbose

        ensure_dir(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"; ensure_dir(self.checkpoint_dir)
        self.log_dir = self.output_dir / "logs"; ensure_dir(self.log_dir)

        self.base_model = None
        self.discriminator = None
        self.pseudo_generator = None
        self.phase_schedule = {
            "warmup": self.hybrid_config.warmup_epochs,
            "discriminator": self.hybrid_config.discriminator_epochs,
            "pseudo_label": self.hybrid_config.pseudolabel_epochs
        }
        logger.info(f"HybridTrainer initialized with schedule: {self.phase_schedule}")

    # --------------------- Model ---------------------
    def build_model(self) -> keras.Model:
        """Use central core.models factory."""
        model = build_model_from_config(self.config)
        lr = self.config.training.learning_rate if self.config.training else 1e-3
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        if self.verbose:
            print_model_summary(model)
        return model

    # --------------------- Training Loop ---------------------
    def train(self, train_data, val_data=None, unlabeled_path=None, **kwargs) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("HYBRID TRAINING STARTED")
        logger.info("=" * 80)

        X_train, y_train = self._prepare_data(train_data)
        X_val, y_val = (self._prepare_data(val_data) if val_data else (None, None))

        # Phase 1: Warm-up
        logger.info("=== Phase 1: Warm-up ===")
        self.base_model = self.build_model()
        hist_warmup = self._warmup_training(X_train, y_train, X_val, y_val)

        # Phase 2: Adversarial
        hist_adv = {}
        if self.hybrid_config.enabled and self.hybrid_config.discriminator_epochs > 0:
            logger.info("=== Phase 2: Adversarial ===")
            hist_adv = self._adversarial_training(X_train, y_train, X_val, y_val)

        # Phase 3: Pseudo-label
        hist_pseudo = {}
        if unlabeled_path and self.hybrid_config.enabled:
            logger.info("=== Phase 3: Pseudo-label ===")
            hist_pseudo = self._pseudo_label_training(unlabeled_path, X_train, y_train, X_val, y_val)

        # Final eval
        val_loss, val_acc = (None, None)
        if X_val is not None:
            val_loss, val_acc = self.base_model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Validation loss={val_loss:.4f} accuracy={val_acc:.4f}")

        # Save
        final_model_path = self.output_dir / "model_hybrid_final.h5"
        self.base_model.save(str(final_model_path))
        logger.info(f"Saved hybrid model: {final_model_path}")

        return {
            "model_path": str(final_model_path),
            "history": {"warmup": hist_warmup, "adversarial": hist_adv, "pseudo_label": hist_pseudo},
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc
        }

    # --------------------- Phase Routines ---------------------
    def _warmup_training(self, X, y, Xv=None, yv=None) -> Dict:
        epochs = self.phase_schedule["warmup"]
        hist = self.base_model.fit(X, y, validation_data=(Xv, yv) if Xv is not None else None,
                                   epochs=epochs, batch_size=self.config.training.batch_size, verbose=1)
        return hist.history

    def _adversarial_training(self, X, y, Xv=None, yv=None) -> Dict:
        epochs = self.phase_schedule["discriminator"]
        self.discriminator = ArchitectureDiscriminator(self.model_config.num_labels)
        # Example: generate fake/invalid reads if available
        if InvalidReadGenerator:
            invalid_reads = InvalidReadGenerator().generate(X.shape[0])
            invalid_preds = self.base_model.predict(invalid_reads)
            valid_preds = self.base_model.predict(X)
            self.discriminator.train(valid_preds, invalid_preds, epochs=max(1, epochs // 2))
        hist = self.base_model.fit(X, y, validation_data=(Xv, yv) if Xv is not None else None,
                                   epochs=epochs, batch_size=self.config.training.batch_size, verbose=1)
        return hist.history

    def _pseudo_label_training(self, unlabeled_path, X, y, Xv=None, yv=None) -> Dict:
        epochs = self.phase_schedule["pseudo_label"]
        self.pseudo_generator = PseudoLabelGenerator(self.base_model, self.config)
        pseudo_reads = self.pseudo_generator.generate_from_path(unlabeled_path)
        # (stub) would merge pseudo data here
        hist = self.base_model.fit(X, y, validation_data=(Xv, yv) if Xv is not None else None,
                                   epochs=epochs, batch_size=self.config.training.batch_size, verbose=1)
        return hist.history

    # --------------------- Helpers ---------------------
    def _prepare_data(self, data) -> Tuple[np.ndarray, np.ndarray]:
        if data is None:
            return None, None
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return data
        if isinstance(data, list) and isinstance(data[0], dict):
            return self._process_dict_data(data)
        raise ValueError("Unsupported data format for training input")

    def _process_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        seqs, labs = [], []
        for d in data:
            s, l = d["sequence"], d.get("labels", [])
            seqs.append(self._sequence_to_numeric(s))
            labs.append(self._labels_to_numeric(l) if l else [])
        X = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=self.model_config.max_seq_len, padding="post")
        y = tf.keras.preprocessing.sequence.pad_sequences(labs, maxlen=self.model_config.max_seq_len, padding="post")
        y = tf.keras.utils.to_categorical(y, num_classes=self.model_config.num_labels)
        return X, y

    def _sequence_to_numeric(self, seq: str) -> List[int]:
        mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
        return [mapping.get(b.upper(), 0) for b in seq]

    def _labels_to_numeric(self, labels: List[str]) -> List[int]:
        lbl_map = {'p7': 0, 'i7': 1, 'RP2': 2, 'UMI': 3, 'ACC': 4,
                   'cDNA': 5, 'polyA': 6, 'CBC': 7, 'RP1': 8, 'i5': 9, 'p5': 10}
        return [lbl_map.get(l, 0) for l in labels]


# ------------------------------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------------------------------
def run_hybrid_training(config: TempestConfig, output_dir: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
    """Entry point used by main.py or Typer CLI."""
    train_data = kwargs.get("train_data")
    val_data = kwargs.get("val_data")
    unlabeled = kwargs.get("unlabeled_path")
    verbose = kwargs.get("verbose", False)
    trainer = HybridTrainer(config, output_dir=output_dir, verbose=verbose)
    if train_data is None:
        raise ValueError("No training data provided")
    return trainer.train(train_data, val_data, unlabeled_path=unlabeled, **kwargs)