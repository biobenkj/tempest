#!/usr/bin/env python3
"""
Tempest Demux Module

This module provides functionality for demultiplexing unlabeled FASTQ files using
trained Tempest models. It annotates reads with predicted segment boundaries and
evaluates performance against whitelist barcodes.
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
import tensorflow as tf
from tqdm import tqdm
import Levenshtein

# Import Tempest modules
from tempest.config import load_config
from tempest.utils.io import load_barcodes, load_model_from_checkpoint, ensure_dir

logger = logging.getLogger(__name__)


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
            whitelists.cbc = load_barcodes(cbc_file)
            logger.info(f"Loaded {len(whitelists.cbc)} CBC barcodes from {cbc_file}")
            
        if i5_file and Path(i5_file).exists():
            whitelists.i5 = load_barcodes(i5_file)
            logger.info(f"Loaded {len(whitelists.i5)} i5 barcodes from {i5_file}")
            
        if i7_file and Path(i7_file).exists():
            whitelists.i7 = load_barcodes(i7_file)
            logger.info(f"Loaded {len(whitelists.i7)} i7 barcodes from {i7_file}")
            
        return whitelists


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


class ReadDemultiplexer:
    """Main class for demultiplexing reads using trained models."""
    
    # Standard segment labels used in Tempest
    SEGMENT_LABELS = {
        0: 'PAD',
        1: 'READ1',
        2: 'i7',
        3: 'i5',
        4: 'CBC',
        5: 'UMI',
        6: 'polyA',
        7: 'polyT',
        8: 'TSO',
        9: 'ACC',
        10: 'cDNA',
        11: 'ME',
        12: 'polyG',
        13: 'LINKER'
    }
    
    def __init__(self, 
                 model_path: str,
                 whitelists: Optional[BarcodeWhitelist] = None,
                 max_edit_distance: int = 2,
                 batch_size: int = 32):
        """
        Initialize the demultiplexer.
        
        Args:
            model_path: Path to trained model
            whitelists: Barcode whitelists for matching
            max_edit_distance: Maximum edit distance for barcode correction
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.whitelists = whitelists or BarcodeWhitelist()
        self.max_edit_distance = max_edit_distance
        self.batch_size = batch_size
        
        # Load model
        self.model = self._load_model()
        self.metrics = DemuxMetrics()
        
    def _load_model(self):
        """Load the trained Tempest model."""
        logger.info(f"Loading model from {self.model_path}")
        
        if self.model_path.endswith('.h5'):
            # Load Keras model
            model = tf.keras.models.load_model(self.model_path, compile=False)
        elif self.model_path.endswith('.pkl'):
            # Load pickled model
            import pickle
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Try loading from checkpoint directory
            model = load_model_from_checkpoint(self.model_path)
            
        logger.info("Model loaded successfully")
        return model
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode DNA sequence to numerical representation.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Encoded sequence array
        """
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        encoded = np.array([base_to_idx.get(base, 4) for base in sequence.upper()])
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
            distance = Levenshtein.distance(sequence, barcode_seq)
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
    
    def process_read(self, read_id: str, sequence: str, quality: str) -> AnnotatedRead:
        """
        Process a single read through the model.
        
        Args:
            read_id: Read identifier
            sequence: DNA sequence
            quality: Quality scores
            
        Returns:
            Annotated read with predictions
        """
        # Encode sequence
        encoded = self.encode_sequence(sequence)
        
        # Prepare for batch prediction (add batch dimension)
        input_batch = np.expand_dims(encoded, axis=0)
        
        # Run inference
        predictions = self.model.predict(input_batch, verbose=0)
        
        # Decode predictions
        labels = self.decode_predictions(predictions)
        
        # Extract segments
        segments = self.extract_segments(sequence, labels)
        
        # Calculate confidence (mean of max probabilities)
        if len(predictions.shape) == 3:
            confidence = np.mean(np.max(predictions[0], axis=-1))
        else:
            confidence = 1.0  # Direct predictions, assume high confidence
        
        # Create annotated read
        annotated = AnnotatedRead(
            read_id=read_id,
            sequence=sequence,
            quality=quality,
            segments=segments,
            confidence=float(confidence)
        )
        
        # Match barcodes against whitelists
        self.match_barcodes(annotated)
        
        # Update metrics
        self.metrics.annotated_reads += 1
        self.metrics.prediction_confidences.append(confidence)
        
        return annotated
    
    def process_fastq_batch(self, reads: List[Tuple[str, str, str]]) -> List[AnnotatedRead]:
        """
        Process a batch of reads.
        
        Args:
            reads: List of (read_id, sequence, quality) tuples
            
        Returns:
            List of annotated reads
        """
        annotated_reads = []
        
        # Process in batches for efficiency
        for i in range(0, len(reads), self.batch_size):
            batch = reads[i:i + self.batch_size]
            
            # Encode all sequences in batch
            encoded_batch = []
            for _, seq, _ in batch:
                encoded = self.encode_sequence(seq)
                encoded_batch.append(encoded)
            
            # Pad sequences to same length
            max_len = max(len(seq) for seq in encoded_batch)
            padded_batch = np.array([
                np.pad(seq, (0, max_len - len(seq)), constant_values=0)
                for seq in encoded_batch
            ])
            
            # Run batch inference
            predictions = self.model.predict(padded_batch, verbose=0)
            
            # Process each prediction
            for j, (read_id, seq, qual) in enumerate(batch):
                # Extract predictions for this read
                if len(predictions.shape) == 3:
                    read_pred = predictions[j]
                    labels = np.argmax(read_pred, axis=-1).tolist()
                    confidence = np.mean(np.max(read_pred, axis=-1))
                else:
                    labels = predictions[j].tolist()
                    confidence = 1.0
                
                # Trim padding from labels
                seq_len = len(seq)
                labels = labels[:seq_len]
                
                # Extract segments
                segments = self.extract_segments(seq, labels)
                
                # Create annotated read
                annotated = AnnotatedRead(
                    read_id=read_id,
                    sequence=seq,
                    quality=qual,
                    segments=segments,
                    confidence=float(confidence)
                )
                
                # Match barcodes
                self.match_barcodes(annotated)
                
                annotated_reads.append(annotated)
                
                # Update metrics
                self.metrics.annotated_reads += 1
                self.metrics.prediction_confidences.append(confidence)
        
        return annotated_reads
    
    def process_fastq_file(self, fastq_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process an entire FASTQ file.
        
        Args:
            fastq_path: Path to FASTQ file
            output_dir: Output directory for results
            
        Returns:
            Dictionary of processing results
        """
        logger.info(f"Processing FASTQ file: {fastq_path}")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine if file is gzipped
        is_gzipped = fastq_path.endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        
        # Process reads
        reads_batch = []
        all_annotations = []
        
        with open_func(fastq_path, 'rt') as f:
            line_count = 0
            read_id = None
            sequence = None
            quality = None
            
            for line in f:
                line = line.strip()
                line_count += 1
                
                if line_count % 4 == 1:
                    # Read ID line
                    read_id = line[1:] if line.startswith('@') else line
                elif line_count % 4 == 2:
                    # Sequence line
                    sequence = line
                elif line_count % 4 == 0:
                    # Quality line
                    quality = line
                    
                    # Add to batch
                    reads_batch.append((read_id, sequence, quality))
                    self.metrics.total_reads += 1
                    
                    # Process batch when full
                    if len(reads_batch) >= self.batch_size:
                        annotated = self.process_fastq_batch(reads_batch)
                        all_annotations.extend(annotated)
                        reads_batch = []
                        
                        # Log progress
                        if self.metrics.total_reads % 10000 == 0:
                            logger.info(f"Processed {self.metrics.total_reads} reads")
        
        # Process remaining reads
        if reads_batch:
            annotated = self.process_fastq_batch(reads_batch)
            all_annotations.extend(annotated)
        
        logger.info(f"Finished processing {self.metrics.total_reads} reads")
        
        # Save annotations
        base_name = Path(fastq_path).stem.replace('.fastq', '').replace('.fq', '')
        
        # Save detailed annotations as JSON
        annotations_file = output_path / f"{base_name}_annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump([ann.to_dict() for ann in all_annotations], f, indent=2)
        logger.info(f"Saved annotations to {annotations_file}")
        
        # Save demultiplexed reads by barcode
        if any([self.whitelists.cbc, self.whitelists.i5, self.whitelists.i7]):
            self._save_demultiplexed_reads(all_annotations, output_path, base_name)
        
        # Compute and save metrics
        stats = self.metrics.compute_statistics()
        metrics_file = output_path / f"{base_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        return {
            'total_reads': self.metrics.total_reads,
            'annotated_reads': self.metrics.annotated_reads,
            'annotations_file': str(annotations_file),
            'metrics_file': str(metrics_file),
            'statistics': stats
        }
    
    def _save_demultiplexed_reads(self, annotations: List[AnnotatedRead], 
                                   output_path: Path, base_name: str) -> None:
        """
        Save demultiplexed reads grouped by barcode assignments.
        
        Args:
            annotations: List of annotated reads
            output_path: Output directory
            base_name: Base filename
        """
        # Group reads by barcode combination
        demux_groups = defaultdict(list)
        
        for ann in annotations:
            # Create barcode key
            barcode_key = []
            if 'CBC' in ann.barcode_assignments:
                barcode_key.append(f"CBC_{ann.barcode_assignments['CBC']}")
            if 'i5' in ann.barcode_assignments:
                barcode_key.append(f"i5_{ann.barcode_assignments['i5']}")
            if 'i7' in ann.barcode_assignments:
                barcode_key.append(f"i7_{ann.barcode_assignments['i7']}")
            
            if barcode_key:
                key = "_".join(barcode_key)
                demux_groups[key].append(ann)
        
        # Save each group to a separate file
        demux_dir = output_path / f"{base_name}_demultiplexed"
        demux_dir.mkdir(exist_ok=True)
        
        for barcode_key, reads in demux_groups.items():
            output_file = demux_dir / f"{barcode_key}.fastq"
            with open(output_file, 'w') as f:
                for read in reads:
                    f.write(f"@{read.read_id}\n")
                    f.write(f"{read.sequence}\n")
                    f.write("+\n")
                    f.write(f"{read.quality}\n")
            
            logger.info(f"Saved {len(reads)} reads to {output_file}")
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.fastq*") -> Dict[str, Any]:
        """
        Process all FASTQ files in a directory.
        
        Args:
            input_dir: Input directory containing FASTQ files
            output_dir: Output directory for results
            file_pattern: File pattern to match
            
        Returns:
            Dictionary of processing results
        """
        input_path = Path(input_dir)
        
        # Find all FASTQ files
        fastq_files = list(input_path.glob(file_pattern))
        logger.info(f"Found {len(fastq_files)} FASTQ files to process")
        
        all_results = {}
        
        for fastq_file in tqdm(fastq_files, desc="Processing files"):
            file_results = self.process_fastq_file(str(fastq_file), output_dir)
            all_results[str(fastq_file)] = file_results
        
        # Generate summary report
        summary = {
            'total_files': len(fastq_files),
            'total_reads_processed': sum(r['total_reads'] for r in all_results.values()),
            'total_reads_annotated': sum(r['annotated_reads'] for r in all_results.values()),
            'files': all_results
        }
        
        summary_file = Path(output_dir) / "demux_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary to {summary_file}")
        
        return summary


def main(args):
    """Main entry point for demux command."""
    
    logger.info("="*80)
    logger.info(" " * 30 + "TEMPEST DEMUX")
    logger.info("="*80)
    
    # Load whitelists if provided
    whitelists = BarcodeWhitelist.from_files(
        cbc_file=args.whitelist_cbc,
        i5_file=args.whitelist_i5,
        i7_file=args.whitelist_i7
    )
    
    # Create demultiplexer
    demux = ReadDemultiplexer(
        model_path=args.model,
        whitelists=whitelists,
        max_edit_distance=args.max_edit_distance,
        batch_size=args.batch_size
    )
    
    # Process input
    if args.input_dir:
        # Process directory of FASTQs
        results = demux.process_directory(
            args.input_dir,
            args.output_dir,
            file_pattern=args.file_pattern
        )
    else:
        # Process single FASTQ file
        results = demux.process_fastq_file(
            args.input,
            args.output_dir
        )
    
    # Print summary statistics
    if 'statistics' in results:
        stats = results['statistics']
    else:
        # Compute overall statistics for directory processing
        stats = {}
        logger.info("\nOverall Statistics:")
        logger.info(f"Total files processed: {results['total_files']}")
        logger.info(f"Total reads: {results['total_reads_processed']}")
        logger.info(f"Total annotated: {results['total_reads_annotated']}")
        
    logger.info("\nDemultiplexing Complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    # Print key metrics
    if stats:
        logger.info("\nKey Metrics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demultiplex FASTQ files using trained Tempest model")
    parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    parser.add_argument('--input', '-i', help='Input FASTQ file')
    parser.add_argument('--input-dir', help='Input directory containing FASTQs')
    parser.add_argument('--output-dir', '-o', default='./demux_results', help='Output directory')
    parser.add_argument('--whitelist-cbc', help='CBC whitelist file')
    parser.add_argument('--whitelist-i5', help='i5 whitelist file')
    parser.add_argument('--whitelist-i7', help='i7 whitelist file')
    parser.add_argument('--max-edit-distance', type=int, default=2, help='Max edit distance for correction')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--file-pattern', default='*.fastq*', help='File pattern for directory processing')
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    main(args)
