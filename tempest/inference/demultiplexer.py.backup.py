"""
Tempest Demultiplexer for FASTQ file processing.

This module provides the core demultiplexing functionality for processing
FASTQ files using trained Tempest models.
"""

import os
import sys
import json
import logging
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Levenshtein, provide fallback if not available
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    logger.warning("Levenshtein module not available. Using simple edit distance.")


def simple_edit_distance(s1: str, s2: str) -> int:
    """Simple edit distance implementation as fallback."""
    if len(s1) < len(s2):
        return simple_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


@dataclass
class BarcodeWhitelist:
    """Container for barcode whitelists."""
    cbc: Optional[Dict[str, str]] = None
    i5: Optional[Dict[str, str]] = None
    i7: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_files(cls, cbc_file: Optional[str] = None, 
                   i5_file: Optional[str] = None, 
                   i7_file: Optional[str] = None):
        """Load whitelists from files."""
        whitelists = cls()
        
        if cbc_file and Path(cbc_file).exists():
            whitelists.cbc = cls._load_barcode_file(cbc_file)
            logger.info(f"Loaded {len(whitelists.cbc)} CBC barcodes from {cbc_file}")
            
        if i5_file and Path(i5_file).exists():
            whitelists.i5 = cls._load_barcode_file(i5_file)
            logger.info(f"Loaded {len(whitelists.i5)} i5 barcodes from {i5_file}")
            
        if i7_file and Path(i7_file).exists():
            whitelists.i7 = cls._load_barcode_file(i7_file)
            logger.info(f"Loaded {len(whitelists.i7)} i7 barcodes from {i7_file}")
            
        return whitelists
    
    @staticmethod
    def _load_barcode_file(filepath: str) -> Dict[str, str]:
        """Load barcodes from file into dictionary."""
        barcodes = {}
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                barcode = line.strip()
                if barcode and not barcode.startswith('#'):
                    # Use barcode as both key and value, or line number as key
                    barcodes[f"BC{i:04d}"] = barcode
        return barcodes


@dataclass
class DemuxMetrics:
    """Container for demultiplexing metrics."""
    total_reads: int = 0
    annotated_reads: int = 0
    
    # Barcode matching statistics
    cbc_exact_matches: int = 0
    cbc_corrected_matches: int = 0
    cbc_no_matches: int = 0
    
    i5_exact_matches: int = 0
    i5_corrected_matches: int = 0
    i5_no_matches: int = 0
    
    i7_exact_matches: int = 0
    i7_corrected_matches: int = 0
    i7_no_matches: int = 0
    
    # Edit distance distributions
    cbc_edit_distances: List[int] = field(default_factory=list)
    i5_edit_distances: List[int] = field(default_factory=list)
    i7_edit_distances: List[int] = field(default_factory=list)
    
    # Segment length statistics
    segment_lengths: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    
    # Confidence scores
    prediction_confidences: List[float] = field(default_factory=list)
    
    def compute_statistics(self):
        """Compute summary statistics from collected metrics."""
        stats = {
            'total_reads': self.total_reads,
            'annotated_reads': self.annotated_reads,
            'annotation_rate': self.annotated_reads / self.total_reads if self.total_reads > 0 else 0
        }
        
        # Barcode matching rates
        for barcode_type in ['cbc', 'i5', 'i7']:
            exact = getattr(self, f'{barcode_type}_exact_matches')
            corrected = getattr(self, f'{barcode_type}_corrected_matches')
            no_match = getattr(self, f'{barcode_type}_no_matches')
            total = exact + corrected + no_match
            
            if total > 0:
                stats[f'{barcode_type}_exact_rate'] = exact / total
                stats[f'{barcode_type}_corrected_rate'] = corrected / total
                stats[f'{barcode_type}_no_match_rate'] = no_match / total
            
            # Edit distance statistics
            edit_dists = getattr(self, f'{barcode_type}_edit_distances')
            if edit_dists:
                stats[f'{barcode_type}_mean_edit_distance'] = np.mean(edit_dists)
                stats[f'{barcode_type}_median_edit_distance'] = np.median(edit_dists)
                stats[f'{barcode_type}_max_edit_distance'] = np.max(edit_dists)
        
        # Segment length statistics
        for segment, lengths in self.segment_lengths.items():
            if lengths:
                stats[f'{segment}_mean_length'] = np.mean(lengths)
                stats[f'{segment}_std_length'] = np.std(lengths)
                stats[f'{segment}_min_length'] = np.min(lengths)
                stats[f'{segment}_max_length'] = np.max(lengths)
        
        # Prediction confidence
        if self.prediction_confidences:
            stats['mean_confidence'] = np.mean(self.prediction_confidences)
            stats['std_confidence'] = np.std(self.prediction_confidences)
            stats['min_confidence'] = np.min(self.prediction_confidences)
        
        return stats


@dataclass
class AnnotatedRead:
    """Container for an annotated read."""
    read_id: str
    sequence: str
    quality: str
    segments: Dict[str, Tuple[int, int]]  # segment_name -> (start, end)
    barcode_assignments: Dict[str, str] = field(default_factory=dict)  # barcode_type -> assigned_id
    edit_distances: Dict[str, int] = field(default_factory=dict)  # barcode_type -> edit_distance
    confidence: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'read_id': self.read_id,
            'sequence': self.sequence,
            'quality': self.quality,
            'segments': self.segments,
            'barcode_assignments': self.barcode_assignments,
            'edit_distances': self.edit_distances,
            'confidence': self.confidence
        }


class Demultiplexer:
    """Main class for demultiplexing reads using trained models."""
    
    # Standard segment labels used in Tempest
    SEGMENT_LABELS = {
        0: 'PAD',
        1: 'p7',
        2: 'i7', 
        3: 'RP2',
        4: 'UMI',
        5: 'ACC',
        6: 'cDNA',
        7: 'polyA',
        8: 'CBC',
        9: 'RP1',
        10: 'i5',
        11: 'p5'
    }
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[Any] = None,
                 whitelist_cbc: Optional[str] = None,
                 whitelist_i5: Optional[str] = None,
                 whitelist_i7: Optional[str] = None,
                 max_edit_distance: int = 2,
                 batch_size: int = 32):
        """
        Initialize the demultiplexer.
        
        Args:
            model_path: Path to trained model
            config: Optional configuration object
            whitelist_cbc: Path to CBC barcode whitelist
            whitelist_i5: Path to i5 barcode whitelist
            whitelist_i7: Path to i7 barcode whitelist
            max_edit_distance: Maximum edit distance for barcode correction
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.config = config
        self.max_edit_distance = max_edit_distance
        self.batch_size = batch_size
        
        # Load whitelists
        self.whitelists = BarcodeWhitelist.from_files(
            cbc_file=whitelist_cbc,
            i5_file=whitelist_i5,
            i7_file=whitelist_i7
        )
        
        # Load model
        self.model = self._load_model()
        self.metrics = DemuxMetrics()
        
    def _load_model(self):
        """Load the trained Tempest model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            import tensorflow as tf
            if self.model_path.endswith('.h5'):
                # Load Keras model
                model = tf.keras.models.load_model(self.model_path, compile=False)
            elif self.model_path.endswith('.pkl'):
                # Load pickled model
                import pickle
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                # Try loading as SavedModel format
                model = tf.keras.models.load_model(self.model_path, compile=False)
                
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode DNA sequence to numerical representation.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Encoded sequence array
        """
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        encoded = np.array([base_to_idx.get(base.upper(), 4) for base in sequence])
        return encoded
    
    def decode_predictions(self, predictions: np.ndarray) -> List[int]:
        """
        Decode model predictions to segment labels.
        
        Args:
            predictions: Model output predictions
            
        Returns:
            List of segment labels
        """
        if len(predictions.shape) == 3:
            # Softmax output - take argmax
            labels = np.argmax(predictions, axis=-1)
        else:
            # Direct labels
            labels = predictions
            
        return labels[0].tolist()  # Return first batch item as list
    
    def extract_segments(self, sequence: str, labels: List[int]) -> Dict[str, Tuple[int, int]]:
        """
        Extract segment boundaries from predicted labels.
        
        Args:
            sequence: DNA sequence
            labels: Predicted segment labels
            
        Returns:
            Dictionary of segment names to (start, end) positions
        """
        segments = {}
        current_label = labels[0] if labels else None
        start_idx = 0
        
        for i, label in enumerate(labels):
            if label != current_label or i == len(labels) - 1:
                # Segment boundary detected
                end_idx = i if i < len(labels) - 1 else len(labels)
                
                if current_label in self.SEGMENT_LABELS and current_label != 0:  # Skip PAD
                    segment_name = self.SEGMENT_LABELS[current_label]
                    segments[segment_name] = (start_idx, end_idx)
                    
                    # Record segment length
                    self.metrics.segment_lengths[segment_name].append(end_idx - start_idx)
                
                current_label = label
                start_idx = i
        
        return segments
    
    def find_closest_barcode(self, sequence: str, whitelist: Dict[str, str]) -> Tuple[str, int]:
        """
        Find the closest matching barcode from whitelist.
        
        Args:
            sequence: Barcode sequence to match
            whitelist: Dictionary of barcode IDs to sequences
            
        Returns:
            Tuple of (best_match_id, edit_distance)
        """
        if not whitelist:
            return None, -1
            
        best_match = None
        min_distance = float('inf')
        
        for barcode_id, barcode_seq in whitelist.items():
            if HAS_LEVENSHTEIN:
                distance = Levenshtein.distance(sequence, barcode_seq)
            else:
                distance = simple_edit_distance(sequence, barcode_seq)
                
            if distance < min_distance:
                min_distance = distance
                best_match = barcode_id
                
        return best_match, min_distance
    
    def match_barcodes(self, read: AnnotatedRead) -> None:
        """
        Match extracted barcodes against whitelists.
        
        Args:
            read: Annotated read with extracted segments
        """
        sequence = read.sequence
        
        # Match CBC barcode
        if 'CBC' in read.segments and self.whitelists.cbc:
            start, end = read.segments['CBC']
            cbc_seq = sequence[start:end]
            
            match_id, edit_dist = self.find_closest_barcode(cbc_seq, self.whitelists.cbc)
            if match_id:
                read.barcode_assignments['CBC'] = match_id
                read.edit_distances['CBC'] = edit_dist
                self.metrics.cbc_edit_distances.append(edit_dist)
                
                if edit_dist == 0:
                    self.metrics.cbc_exact_matches += 1
                elif edit_dist <= self.max_edit_distance:
                    self.metrics.cbc_corrected_matches += 1
                else:
                    self.metrics.cbc_no_matches += 1
            else:
                self.metrics.cbc_no_matches += 1
        
        # Match i5 barcode
        if 'i5' in read.segments and self.whitelists.i5:
            start, end = read.segments['i5']
            i5_seq = sequence[start:end]
            
            match_id, edit_dist = self.find_closest_barcode(i5_seq, self.whitelists.i5)
            if match_id:
                read.barcode_assignments['i5'] = match_id
                read.edit_distances['i5'] = edit_dist
                self.metrics.i5_edit_distances.append(edit_dist)
                
                if edit_dist == 0:
                    self.metrics.i5_exact_matches += 1
                elif edit_dist <= self.max_edit_distance:
                    self.metrics.i5_corrected_matches += 1
                else:
                    self.metrics.i5_no_matches += 1
            else:
                self.metrics.i5_no_matches += 1
        
        # Match i7 barcode
        if 'i7' in read.segments and self.whitelists.i7:
            start, end = read.segments['i7']
            i7_seq = sequence[start:end]
            
            match_id, edit_dist = self.find_closest_barcode(i7_seq, self.whitelists.i7)
            if match_id:
                read.barcode_assignments['i7'] = match_id
                read.edit_distances['i7'] = edit_dist
                self.metrics.i7_edit_distances.append(edit_dist)
                
                if edit_dist == 0:
                    self.metrics.i7_exact_matches += 1
                elif edit_dist <= self.max_edit_distance:
                    self.metrics.i7_corrected_matches += 1
                else:
                    self.metrics.i7_no_matches += 1
            else:
                self.metrics.i7_no_matches += 1
    
    def process_fastq(self, input_file: str, output_dir: str, output_prefix: str = "demux") -> Dict[str, Any]:
        """
        Process a FASTQ file for demultiplexing.
        
        Args:
            input_file: Path to input FASTQ file
            output_dir: Output directory
            output_prefix: Prefix for output files
            
        Returns:
            Results dictionary
        """
        logger.info(f"Processing FASTQ file: {input_file}")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process file and collect results
        results = {
            'total_reads': 0,
            'passed_reads': 0,
            'failed_reads': 0,
            'barcode_distribution': {}
        }
        
        # Placeholder for actual FASTQ processing
        # This would read the file, process reads, and generate results
        
        # Save results
        results_file = Path(output_dir) / f"{output_prefix}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results