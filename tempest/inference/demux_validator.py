"""
Tempest Demultiplexer with Architecture Validation based on Model Inference.

Uses trained models (standard, hybrid, or ensemble) to predict segment labels,
then validates if the predicted architecture matches the expected structure.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureValidator:
    """Validates predicted segment architecture against expected structure."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with expected architecture from config.
        
        Args:
            config: Demux configuration containing expected architecture
        """
        self.expected_segments = config.get('expected_segments', 
            ['p7', 'i7', 'RP2', 'UMI', 'ACC', 'cDNA', 'polyA', 'CBC', 'RP1', 'i5', 'p5'])
        self.required_segments = config.get('required_segments', ['UMI', 'ACC', 'CBC'])
        self.optional_segments = config.get('optional_segments', ['polyA', 'RP2', 'RP1'])
        
        # Validation parameters
        self.min_segment_match = config.get('min_segment_match', 0.8)
        self.allow_reverse = config.get('allow_reverse_orientation', True)
        self.strict_order = config.get('strict_order', False)
        
        # Generate reverse architecture if allowed
        if self.allow_reverse:
            self.expected_segments_rev = self.expected_segments[::-1]
        else:
            self.expected_segments_rev = None
    
    def validate(self, predicted_segments: List[str]) -> Tuple[str, List[str], str]:
        """
        Validate predicted segments against expected architecture.
        
        Args:
            predicted_segments: List of predicted segment names in order
            
        Returns:
            Tuple of (classification, issues, orientation)
            - classification: 'valid', 'invalid', or 'ambiguous'
            - issues: List of validation issues
            - orientation: 'forward', 'reverse', or 'unknown'
        """
        issues = []
        
        # Check for required segments
        missing_required = set(self.required_segments) - set(predicted_segments)
        if missing_required:
            issues.append(f"Missing required segments: {', '.join(missing_required)}")
        
        # Remove optional segments for order checking
        filtered_predicted = [s for s in predicted_segments 
                            if s not in self.optional_segments]
        filtered_expected = [s for s in self.expected_segments 
                           if s not in self.optional_segments]
        
        # Check segment order
        orientation = 'unknown'
        forward_score = self._calculate_order_score(filtered_predicted, filtered_expected)
        
        if self.allow_reverse and self.expected_segments_rev:
            filtered_expected_rev = [s for s in self.expected_segments_rev 
                                    if s not in self.optional_segments]
            reverse_score = self._calculate_order_score(filtered_predicted, filtered_expected_rev)
        else:
            reverse_score = 0
        
        # Determine orientation
        if forward_score >= self.min_segment_match:
            orientation = 'forward'
        elif self.allow_reverse and reverse_score >= self.min_segment_match:
            orientation = 'reverse'
        else:
            issues.append(f"Segment order match below threshold (forward: {forward_score:.2f}, reverse: {reverse_score:.2f})")
        
        # Check for unexpected segments
        unexpected = set(predicted_segments) - set(self.expected_segments) - set(self.optional_segments)
        if unexpected:
            issues.append(f"Unexpected segments: {', '.join(unexpected)}")
        
        # Determine classification
        if not issues:
            return 'valid', [], orientation
        elif not missing_required and orientation != 'unknown':
            # Has required segments and reasonable order
            return 'ambiguous', issues, orientation
        else:
            return 'invalid', issues, orientation
    
    def _calculate_order_score(self, predicted: List[str], expected: List[str]) -> float:
        """
        Calculate how well predicted order matches expected order.
        
        Uses longest common subsequence (LCS) approach.
        """
        if not predicted or not expected:
            return 0.0
        
        if self.strict_order:
            # Strict order: must match exactly
            return 1.0 if predicted == expected else 0.0
        else:
            # Flexible order: use LCS
            lcs_length = self._lcs_length(predicted, expected)
            return lcs_length / len(expected)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


@dataclass
class DemuxResult:
    """Container for demultiplexing results with architecture validation."""
    read_id: str
    sequence: str
    quality: str
    predicted_labels: List[int]
    predicted_segments: List[str]
    segment_boundaries: Dict[str, Tuple[int, int]]
    confidence: float
    classification: str  # 'valid', 'invalid', 'ambiguous'
    orientation: str  # 'forward', 'reverse', 'unknown'
    validation_issues: List[str]
    barcode_matches: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'read_id': self.read_id,
            'predicted_segments': self.predicted_segments,
            'segment_boundaries': self.segment_boundaries,
            'confidence': self.confidence,
            'classification': self.classification,
            'orientation': self.orientation,
            'issues': self.validation_issues,
            'barcodes': self.barcode_matches
        }


class ModelBasedDemultiplexer:
    """
    Demultiplexer that uses trained model inference for architecture validation.
    
    Works with any trained Tempest model (standard, hybrid, or ensemble).
    """
    
    # Segment label mapping (standard for Tempest)
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
                 batch_size: int = 32,
                 confidence_threshold: float = 0.85):
        """
        Initialize demultiplexer with trained model.
        
        Args:
            model_path: Path to trained model (h5, pkl, or SavedModel)
            config: Configuration dict with demux settings
            batch_size: Batch size for inference
            confidence_threshold: Minimum confidence for valid classification
        """
        self.model_path = model_path
        self.config = config
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model()
        
        # Initialize architecture validator
        self.validator = ArchitectureValidator(config.get('demux', {}))
        
        # Statistics
        self.stats = {
            'total_reads': 0,
            'valid': 0,
            'invalid': 0,
            'ambiguous': 0,
            'forward_orientation': 0,
            'reverse_orientation': 0,
            'unknown_orientation': 0,
            'common_issues': Counter()
        }
    
    def _load_model(self):
        """Load the trained Tempest model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            import tensorflow as tf
            
            # Try different loading methods
            if self.model_path.endswith('.h5'):
                model = tf.keras.models.load_model(self.model_path, compile=False)
            elif self.model_path.endswith('.pkl'):
                import pickle
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                # Assume SavedModel format
                model = tf.keras.models.load_model(self.model_path, compile=False)
            
            logger.info(f"Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_fastq(self, 
                     input_file: str,
                     output_dir: str,
                     save_valid: bool = True,
                     save_invalid: bool = False) -> Dict[str, Any]:
        """
        Process FASTQ file and classify reads based on predicted architecture.
        
        Args:
            input_file: Path to input FASTQ file
            output_dir: Output directory for results
            save_valid: Save valid reads to separate file
            save_invalid: Save invalid reads to separate file
            
        Returns:
            Dictionary with processing statistics and results
        """
        logger.info(f"Processing {input_file}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Output files
        base_name = Path(input_file).stem
        results_file = Path(output_dir) / f"{base_name}_results.json"
        annotations_file = Path(output_dir) / f"{base_name}_annotations.jsonl"
        
        valid_fastq = Path(output_dir) / f"{base_name}_valid.fastq" if save_valid else None
        invalid_fastq = Path(output_dir) / f"{base_name}_invalid.fastq" if save_invalid else None
        
        # Process reads in batches
        batch_records = []
        batch_sequences = []
        
        valid_records = []
        invalid_records = []
        all_results = []
        
        # Read and process FASTQ
        for record in SeqIO.parse(input_file, "fastq"):
            batch_records.append(record)
            batch_sequences.append(str(record.seq))
            
            # Process batch when full
            if len(batch_records) >= self.batch_size:
                results = self._process_batch(batch_records, batch_sequences)
                
                # Sort results
                for i, result in enumerate(results):
                    all_results.append(result)
                    
                    if result.classification == 'valid':
                        valid_records.append(batch_records[i])
                    elif result.classification == 'invalid' and save_invalid:
                        invalid_records.append(batch_records[i])
                
                # Clear batch
                batch_records = []
                batch_sequences = []
        
        # Process remaining batch
        if batch_records:
            results = self._process_batch(batch_records, batch_sequences)
            
            for i, result in enumerate(results):
                all_results.append(result)
                
                if result.classification == 'valid':
                    valid_records.append(batch_records[i])
                elif result.classification == 'invalid' and save_invalid:
                    invalid_records.append(batch_records[i])
        
        # Write output files
        logger.info(f"Writing results to {output_dir}")
        
        # Write annotations
        with open(annotations_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result.to_dict()) + '\n')
        
        # Write valid reads
        if save_valid and valid_records:
            SeqIO.write(valid_records, valid_fastq, 'fastq')
            logger.info(f"Saved {len(valid_records)} valid reads to {valid_fastq}")
        
        # Write invalid reads
        if save_invalid and invalid_records:
            SeqIO.write(invalid_records, invalid_fastq, 'fastq')
            logger.info(f"Saved {len(invalid_records)} invalid reads to {invalid_fastq}")
        
        # Save statistics
        final_stats = {
            **self.stats,
            'valid_rate': self.stats['valid'] / self.stats['total_reads'] if self.stats['total_reads'] > 0 else 0,
            'common_issues': dict(self.stats['common_issues'].most_common(10))
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Display summary
        self._display_summary()
        
        return final_stats
    
    def _process_batch(self, records: List, sequences: List[str]) -> List[DemuxResult]:
        """Process a batch of sequences through the model."""
        results = []
        
        # Encode sequences
        encoded_seqs = self._encode_sequences(sequences)
        
        # Get model predictions
        predictions = self._get_predictions(encoded_seqs)
        
        # Process each prediction
        for i, record in enumerate(records):
            # Get predicted labels for this sequence
            pred_labels = self._decode_predictions(predictions[i])
            
            # Extract segments from predictions
            segments, boundaries = self._extract_segments(pred_labels, len(sequences[i]))
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions[i])
            
            # Validate architecture
            classification, issues, orientation = self.validator.validate(segments)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                classification = 'invalid'
                issues.append(f"Low confidence: {confidence:.2f}")
            
            # Create result
            result = DemuxResult(
                read_id=record.id,
                sequence=str(record.seq),
                quality="".join([chr(q + 33) for q in record.letter_annotations.get("phred_quality", [])]),
                predicted_labels=pred_labels,
                predicted_segments=segments,
                segment_boundaries=boundaries,
                confidence=confidence,
                classification=classification,
                orientation=orientation,
                validation_issues=issues
            )
            
            results.append(result)
            
            # Update statistics
            self.stats['total_reads'] += 1
            self.stats[classification] += 1
            self.stats[f'{orientation}_orientation'] += 1
            
            for issue in issues:
                self.stats['common_issues'][issue] += 1
        
        return results
    
    def _encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """Encode DNA sequences for model input."""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Find max length for padding
        max_len = max(len(seq) for seq in sequences)
        
        # Encode and pad
        encoded = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                encoded[i, j] = base_to_idx.get(base.upper(), 4)
        
        return encoded
    
    def _get_predictions(self, encoded_seqs: np.ndarray) -> np.ndarray:
        """Get model predictions for encoded sequences."""
        import tensorflow as tf
        
        # Run inference
        predictions = self.model(encoded_seqs, training=False)
        
        return predictions.numpy()
    
    def _decode_predictions(self, predictions: np.ndarray) -> List[int]:
        """Decode model predictions to label indices."""
        if len(predictions.shape) == 2:
            # Softmax output - take argmax
            return np.argmax(predictions, axis=-1).tolist()
        else:
            # Direct labels
            return predictions.tolist()
    
    def _extract_segments(self, labels: List[int], seq_len: int) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        """
        Extract segment names and boundaries from predicted labels.
        
        Returns:
            Tuple of (segment_list, boundary_dict)
            - segment_list: Ordered list of segment names
            - boundary_dict: Map of segment_name -> (start, end)
        """
        segments = []
        boundaries = {}
        
        if not labels:
            return segments, boundaries
        
        # Truncate to sequence length
        labels = labels[:seq_len]
        
        # Find segment boundaries
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # Segment boundary found
                if current_label in self.LABEL_TO_SEGMENT and current_label != 0:  # Skip PAD
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
            # Softmax output - use mean of max probabilities
            import tensorflow as tf
            probs = tf.nn.softmax(predictions, axis=-1).numpy()
            max_probs = np.max(probs, axis=-1)
            return float(np.mean(max_probs))
        else:
            # No probability information available
            return 1.0
    
    def _display_summary(self):
        """Display processing summary."""
        print("\n" + "="*60)
        print("DEMULTIPLEXING SUMMARY")
        print("="*60)
        
        total = self.stats['total_reads']
        if total > 0:
            print(f"Total reads processed: {total:,}")
            print(f"Valid reads: {self.stats['valid']:,} ({100*self.stats['valid']/total:.1f}%)")
            print(f"Invalid reads: {self.stats['invalid']:,} ({100*self.stats['invalid']/total:.1f}%)")
            print(f"Ambiguous reads: {self.stats['ambiguous']:,} ({100*self.stats['ambiguous']/total:.1f}%)")
            
            print(f"\nOrientation distribution:")
            print(f"  Forward: {self.stats['forward_orientation']:,}")
            print(f"  Reverse: {self.stats['reverse_orientation']:,}")
            print(f"  Unknown: {self.stats['unknown_orientation']:,}")
            
            if self.stats['common_issues']:
                print(f"\nMost common issues:")
                for issue, count in self.stats['common_issues'].most_common(5):
                    print(f"  {issue}: {count:,}")
        
        print("="*60 + "\n")


def demux_with_validation(
    model_path: str,
    input_file: str,
    config: Dict[str, Any],
    output_dir: str = "./demux_output",
    batch_size: int = 32,
    confidence_threshold: float = 0.85,
    save_valid: bool = True,
    save_invalid: bool = False
) -> Dict[str, Any]:
    """
    Main function to demultiplex FASTQ with architecture validation.
    
    Args:
        model_path: Path to trained Tempest model
        input_file: Input FASTQ file
        config: Configuration dictionary with demux settings
        output_dir: Output directory
        batch_size: Batch size for inference
        confidence_threshold: Minimum confidence threshold
        save_valid: Save valid reads separately
        save_invalid: Save invalid reads separately
        
    Returns:
        Dictionary with processing statistics
    """
    demux = ModelBasedDemultiplexer(
        model_path=model_path,
        config=config,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )
    
    return demux.process_fastq(
        input_file=input_file,
        output_dir=output_dir,
        save_valid=save_valid,
        save_invalid=save_invalid
    )
