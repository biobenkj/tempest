"""
Sample-based Demultiplexer with Architecture Validation and Barcode Correction.

Processes FASTQ files, validates architecture, extracts barcodes, 
and assigns reads to samples based on a sample sheet.
"""

import os
import gzip
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import Levenshtein

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Container for sample information from sample sheet."""
    sample_name: str
    cbc: str
    i5: str
    i7: str
    
    def barcode_tuple(self) -> Tuple[str, str, str]:
        """Return barcode combination as tuple."""
        return (self.cbc, self.i5, self.i7)
    
    def barcode_string(self) -> str:
        """Return barcode combination as string."""
        return f"{self.i7}+{self.i5}+{self.cbc}"


@dataclass
class BarcodeMatch:
    """Container for barcode matching result."""
    sample_name: str
    cbc_distance: int
    i5_distance: int
    i7_distance: int
    total_distance: int
    
    def is_exact_match(self) -> bool:
        """Check if all barcodes match exactly."""
        return self.total_distance == 0
    
    def is_valid_match(self, max_distance: int = 2) -> bool:
        """Check if match is within acceptable distance."""
        return self.total_distance <= max_distance


class SampleSheet:
    """Handles sample sheet loading and barcode lookups."""
    
    def __init__(self, sample_sheet_path: str, max_edit_distance: int = 2):
        """
        Initialize sample sheet.
        
        Args:
            sample_sheet_path: Path to CSV file with sample information
            max_edit_distance: Maximum allowed edit distance for barcode matching
        """
        self.samples = self._load_sample_sheet(sample_sheet_path)
        self.max_edit_distance = max_edit_distance
        
        # Build lookup structures
        self._build_lookups()
        
        logger.info(f"Loaded {len(self.samples)} samples from sample sheet")
    
    def _load_sample_sheet(self, path: str) -> List[Sample]:
        """Load samples from CSV file."""
        samples = []
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Check for required columns
            required_columns = {'sample_name', 'cbc', 'i5', 'i7'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"Sample sheet missing required columns: {missing}")
            
            for row in reader:
                sample = Sample(
                    sample_name=row['sample_name'].strip(),
                    cbc=row['cbc'].strip().upper(),
                    i5=row['i5'].strip().upper(),
                    i7=row['i7'].strip().upper()
                )
                samples.append(sample)
        
        return samples
    
    def _build_lookups(self):
        """Build lookup structures for fast barcode matching."""
        # Exact match lookup
        self.exact_lookup = {}
        for sample in self.samples:
            key = sample.barcode_tuple()
            if key in self.exact_lookup:
                logger.warning(f"Duplicate barcode combination: {key}")
            self.exact_lookup[key] = sample.sample_name
        
        # Individual barcode sets for validation
        self.valid_cbcs = {s.cbc for s in self.samples}
        self.valid_i5s = {s.i5 for s in self.samples}
        self.valid_i7s = {s.i7 for s in self.samples}
    
    def find_best_match(self, cbc: str, i5: str, i7: str) -> Optional[BarcodeMatch]:
        """
        Find best matching sample for given barcodes.
        
        Args:
            cbc: Cell barcode sequence
            i5: i5 index sequence
            i7: i7 index sequence
            
        Returns:
            BarcodeMatch object or None if no valid match
        """
        # Check exact match first
        if (cbc, i5, i7) in self.exact_lookup:
            return BarcodeMatch(
                sample_name=self.exact_lookup[(cbc, i5, i7)],
                cbc_distance=0,
                i5_distance=0,
                i7_distance=0,
                total_distance=0
            )
        
        # Find best match with error correction
        best_match = None
        min_distance = float('inf')
        
        for sample in self.samples:
            cbc_dist = Levenshtein.distance(cbc, sample.cbc) if cbc else len(sample.cbc)
            i5_dist = Levenshtein.distance(i5, sample.i5) if i5 else len(sample.i5)
            i7_dist = Levenshtein.distance(i7, sample.i7) if i7 else len(sample.i7)
            
            total_dist = cbc_dist + i5_dist + i7_dist
            
            if total_dist < min_distance and total_dist <= self.max_edit_distance:
                min_distance = total_dist
                best_match = BarcodeMatch(
                    sample_name=sample.sample_name,
                    cbc_distance=cbc_dist,
                    i5_distance=i5_dist,
                    i7_distance=i7_dist,
                    total_distance=total_dist
                )
        
        return best_match


class SampleBasedDemultiplexer:
    """
    Demultiplexer that assigns reads to samples based on extracted barcodes.
    
    Combines architecture validation with sample assignment.
    """
    
    # Segment label mapping
    LABEL_TO_SEGMENT = {
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
                 config: Dict[str, Any],
                 sample_sheet: SampleSheet,
                 batch_size: int = 32,
                 confidence_threshold: float = 0.85):
        """
        Initialize sample-based demultiplexer.
        
        Args:
            model_path: Path to trained model
            config: Configuration with demux settings
            sample_sheet: SampleSheet object with sample information
            batch_size: Batch size for inference
            confidence_threshold: Minimum confidence for valid reads
        """
        self.model_path = model_path
        self.config = config
        self.sample_sheet = sample_sheet
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model()
        
        # Initialize architecture validator
        from tempest.inference.demux_validator import ArchitectureValidator
        self.validator = ArchitectureValidator(config.get('demux', {}))
        
        # Statistics
        self.stats = defaultdict(lambda: {
            'total_reads': 0,
            'valid_reads': 0,
            'invalid_reads': 0,
            'exact_matches': 0,
            'corrected_matches': 0,
            'no_matches': 0
        })
        
        # Sample file handles (will be created as needed)
        self.sample_files = {}
    
    def _load_model(self):
        """Load the trained Tempest model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            import tensorflow as tf
            
            if self.model_path.endswith('.h5'):
                model = tf.keras.models.load_model(self.model_path, compile=False)
            else:
                model = tf.keras.models.load_model(self.model_path, compile=False)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_fastq_directory(self,
                               input_dir: str,
                               output_dir: str,
                               file_pattern: str = "*.fastq*",
                               compress_output: bool = True,
                               save_annotations: bool = True) -> Dict[str, Any]:
        """
        Process all FASTQ files in a directory.
        
        Args:
            input_dir: Directory containing FASTQ files
            output_dir: Output directory for per-sample files
            file_pattern: Pattern to match FASTQ files
            compress_output: Whether to gzip output files
            save_annotations: Whether to save JSONL annotations
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all FASTQ files
        fastq_files = list(input_path.glob(file_pattern))
        
        if not fastq_files:
            raise ValueError(f"No files matching pattern '{file_pattern}' found in {input_dir}")
        
        logger.info(f"Found {len(fastq_files)} FASTQ files to process")
        
        # Open annotation file if requested
        self.annotation_file = None
        if save_annotations:
            annotations_path = output_path / "read_annotations.jsonl"
            self.annotation_file = open(annotations_path, 'w')
            logger.info(f"Writing annotations to {annotations_path}")
        
        # Process each file
        for fastq_file in fastq_files:
            logger.info(f"Processing {fastq_file.name}")
            self._process_single_fastq(fastq_file, output_path, compress_output)
        
        # Close all output files
        self._close_sample_files()
        
        # Close annotation file
        if self.annotation_file:
            self.annotation_file.close()
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Save statistics
        stats_file = output_path / "demux_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Demultiplexing complete. Statistics saved to {stats_file}")
        
        return summary
    
    def _process_single_fastq(self, 
                             input_file: Path,
                             output_dir: Path,
                             compress_output: bool):
        """Process a single FASTQ file."""
        # Process reads in batches
        batch_records = []
        batch_sequences = []
        
        # Open input file (handle gzipped)
        if input_file.suffix == '.gz':
            handle = gzip.open(input_file, 'rt')
        else:
            handle = open(input_file, 'r')
        
        try:
            for record in SeqIO.parse(handle, 'fastq'):
                batch_records.append(record)
                batch_sequences.append(str(record.seq))
                
                # Process batch when full
                if len(batch_records) >= self.batch_size:
                    self._process_batch(batch_records, batch_sequences, 
                                      output_dir, compress_output)
                    batch_records = []
                    batch_sequences = []
            
            # Process remaining batch
            if batch_records:
                self._process_batch(batch_records, batch_sequences,
                                  output_dir, compress_output)
        
        finally:
            handle.close()
    
    def _process_batch(self,
                      records: List[SeqRecord],
                      sequences: List[str],
                      output_dir: Path,
                      compress_output: bool):
        """Process a batch of reads."""
        # Get model predictions
        predictions = self._get_predictions(sequences)
        
        # Process each read
        for i, record in enumerate(records):
            # Extract segments from predictions
            pred_labels = self._decode_predictions(predictions[i])
            segments, boundaries = self._extract_segments(pred_labels, len(sequences[i]))
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions[i])
            
            # Validate architecture
            classification, issues, orientation = self.validator.validate(segments)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                classification = 'invalid'
                issues.append(f"Low confidence: {confidence:.2f}")
            
            # Initialize annotation record
            annotation = {
                "read_id": record.id,
                "predicted_segments": segments,
                "segment_boundaries": boundaries,
                "confidence": float(confidence),
                "classification": classification,
                "orientation": orientation,
                "issues": issues,
                "sample_assignment": "undetermined",
                "barcode_match": None
            }
            
            # Process valid reads
            sample_name = 'undetermined'
            
            if classification == 'valid':
                # Extract barcodes from predicted segments
                cbc_seq = self._extract_barcode(sequences[i], boundaries, 'CBC')
                i5_seq = self._extract_barcode(sequences[i], boundaries, 'i5')
                i7_seq = self._extract_barcode(sequences[i], boundaries, 'i7')
                
                # Add extracted barcodes to annotation
                annotation["extracted_barcodes"] = {
                    "CBC": cbc_seq,
                    "i5": i5_seq,
                    "i7": i7_seq
                }
                
                # Find matching sample
                if cbc_seq and i5_seq and i7_seq:
                    match = self.sample_sheet.find_best_match(cbc_seq, i5_seq, i7_seq)
                    
                    if match:
                        sample_name = match.sample_name
                        annotation["sample_assignment"] = sample_name
                        annotation["barcode_match"] = {
                            "sample": sample_name,
                            "cbc_distance": match.cbc_distance,
                            "i5_distance": match.i5_distance,
                            "i7_distance": match.i7_distance,
                            "total_distance": match.total_distance,
                            "exact_match": match.is_exact_match()
                        }
                        
                        # Update statistics
                        if match.is_exact_match():
                            self.stats[sample_name]['exact_matches'] += 1
                        else:
                            self.stats[sample_name]['corrected_matches'] += 1
                    else:
                        self.stats['undetermined']['no_matches'] += 1
                
                self.stats[sample_name]['valid_reads'] += 1
            else:
                # Invalid reads go to undetermined
                self.stats['undetermined']['invalid_reads'] += 1
            
            # Update total reads
            self.stats[sample_name]['total_reads'] += 1
            
            # Write annotation to JSONL if file is open
            if hasattr(self, 'annotation_file') and self.annotation_file:
                self.annotation_file.write(json.dumps(annotation) + '\n')
            
            # Write to appropriate output file
            self._write_read(record, sample_name, output_dir, compress_output)
    
    def _extract_barcode(self, 
                        sequence: str,
                        boundaries: Dict[str, Tuple[int, int]],
                        barcode_type: str) -> Optional[str]:
        """Extract barcode sequence from predicted segment boundaries."""
        if barcode_type in boundaries:
            start, end = boundaries[barcode_type]
            return sequence[start:end]
        return None
    
    def _write_read(self,
                   record: SeqRecord,
                   sample_name: str,
                   output_dir: Path,
                   compress_output: bool):
        """Write read to appropriate sample file."""
        # Get or create file handle for this sample
        if sample_name not in self.sample_files:
            # Create output filename
            if compress_output:
                filename = output_dir / f"{sample_name}.fastq.gz"
                handle = gzip.open(filename, 'wt')
            else:
                filename = output_dir / f"{sample_name}.fastq"
                handle = open(filename, 'w')
            
            self.sample_files[sample_name] = handle
            logger.info(f"Created output file for sample: {sample_name}")
        
        # Write record
        SeqIO.write([record], self.sample_files[sample_name], 'fastq')
    
    def _close_sample_files(self):
        """Close all open sample files."""
        for sample_name, handle in self.sample_files.items():
            handle.close()
            logger.info(f"Closed output file for sample: {sample_name}")
        
        self.sample_files = {}
    
    def _get_predictions(self, sequences: List[str]) -> np.ndarray:
        """Get model predictions for sequences."""
        # Encode sequences
        encoded = self._encode_sequences(sequences)
        
        # Run inference
        import tensorflow as tf
        predictions = self.model(encoded, training=False)
        
        return predictions.numpy()
    
    def _encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """Encode DNA sequences for model input."""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        max_len = max(len(seq) for seq in sequences)
        encoded = np.zeros((len(sequences), max_len), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                encoded[i, j] = base_to_idx.get(base.upper(), 4)
        
        return encoded
    
    def _decode_predictions(self, predictions: np.ndarray) -> List[int]:
        """Decode model predictions to label indices."""
        if len(predictions.shape) == 2:
            return np.argmax(predictions, axis=-1).tolist()
        else:
            return predictions.tolist()
    
    def _extract_segments(self, labels: List[int], seq_len: int) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        """Extract segment names and boundaries from predicted labels."""
        segments = []
        boundaries = {}
        
        if not labels:
            return segments, boundaries
        
        labels = labels[:seq_len]
        
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if current_label in self.LABEL_TO_SEGMENT and current_label != 0:
                    seg_name = self.LABEL_TO_SEGMENT[current_label]
                    segments.append(seg_name)
                    boundaries[seg_name] = (start_idx, i)
                
                current_label = labels[i]
                start_idx = i
        
        # Handle last segment
        if current_label in self.LABEL_TO_SEGMENT and current_label != 0:
            seg_name = self.LABEL_TO_SEGMENT[current_label]
            segments.append(seg_name)
            boundaries[seg_name] = (start_idx, len(labels))
        
        return segments, boundaries
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """Calculate confidence score from predictions."""
        if len(predictions.shape) == 2:
            import tensorflow as tf
            probs = tf.nn.softmax(predictions, axis=-1).numpy()
            max_probs = np.max(probs, axis=-1)
            return float(np.mean(max_probs))
        else:
            return 1.0
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'samples': {},
            'total': {
                'reads': 0,
                'valid': 0,
                'invalid': 0,
                'exact_matches': 0,
                'corrected_matches': 0,
                'no_matches': 0
            }
        }
        
        for sample_name, stats in self.stats.items():
            summary['samples'][sample_name] = stats
            
            summary['total']['reads'] += stats['total_reads']
            summary['total']['valid'] += stats['valid_reads']
            summary['total']['invalid'] += stats['invalid_reads']
            summary['total']['exact_matches'] += stats['exact_matches']
            summary['total']['corrected_matches'] += stats['corrected_matches']
            summary['total']['no_matches'] += stats['no_matches']
        
        # Calculate rates
        if summary['total']['reads'] > 0:
            summary['total']['valid_rate'] = summary['total']['valid'] / summary['total']['reads']
            summary['total']['assignment_rate'] = (
                summary['total']['exact_matches'] + summary['total']['corrected_matches']
            ) / summary['total']['reads']
        
        return summary


def demux_with_samples(
    model_path: str,
    input_dir: str,
    sample_sheet_path: str,
    config: Dict[str, Any],
    output_dir: str = "./demux_output",
    max_edit_distance: int = 2,
    batch_size: int = 32,
    confidence_threshold: float = 0.85,
    compress_output: bool = True,
    file_pattern: str = "*.fastq*"
) -> Dict[str, Any]:
    """
    Main function for sample-based demultiplexing.
    
    Args:
        model_path: Path to trained model
        input_dir: Directory containing FASTQ files
        sample_sheet_path: Path to sample sheet CSV
        config: Configuration dictionary
        output_dir: Output directory for per-sample files
        max_edit_distance: Maximum edit distance for barcode matching
        batch_size: Batch size for inference
        confidence_threshold: Confidence threshold for valid reads
        compress_output: Whether to gzip output files
        file_pattern: Pattern to match FASTQ files
        
    Returns:
        Dictionary with demultiplexing statistics
    """
    # Load sample sheet
    sample_sheet = SampleSheet(sample_sheet_path, max_edit_distance)
    
    # Create demultiplexer
    demux = SampleBasedDemultiplexer(
        model_path=model_path,
        config=config,
        sample_sheet=sample_sheet,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )
    
    # Process directory
    return demux.process_fastq_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        file_pattern=file_pattern,
        compress_output=compress_output
    )
