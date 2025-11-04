"""
Invalid read generator for Tempest robustness training.

Generates reads with segment-level architectural errors for training
models that are robust to common sequencing failures.

Part of: tempest/data/ module
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from simulator import SimulatedRead

logger = logging.getLogger(__name__)


class InvalidReadGenerator:
    """
    Generate invalid reads with segment-level errors.
    
    Supports multiple error types:
    - segment_loss: Missing UMI, ACC, or barcode
    - segment_duplication: Repeated segments (PCR artifacts)
    - truncation: Incomplete reads
    - chimeric: Mixed segments from different reads
    - scrambled: Randomized segment order
    """
    
    def __init__(self, config):
        """
        Initialize invalid read generator.
        
        Args:
            config: TempestConfig with hybrid training parameters
        """
        self.config = config
        
        # Default error probabilities
        if config.hybrid:
            self.error_probabilities = {
                'segment_loss': config.hybrid.segment_loss_prob,
                'segment_duplication': config.hybrid.segment_dup_prob,
                'truncation': config.hybrid.truncation_prob,
                'chimeric': config.hybrid.chimeric_prob,
                'scrambled': config.hybrid.scrambled_prob
            }
        else:
            self.error_probabilities = {
                'segment_loss': 0.3,
                'segment_duplication': 0.3,
                'truncation': 0.2,
                'chimeric': 0.1,
                'scrambled': 0.1
            }
        
        # Normalize probabilities
        total = sum(self.error_probabilities.values())
        self.error_probabilities = {
            k: v/total for k, v in self.error_probabilities.items()
        }
        
        logger.info(f"Initialized InvalidReadGenerator with error probabilities: "
                   f"{self.error_probabilities}")
    
    def generate_invalid_read(self, valid_read: SimulatedRead, 
                             error_type: Optional[str] = None) -> SimulatedRead:
        """
        Generate an invalid read from a valid one.
        
        Args:
            valid_read: Valid SimulatedRead object
            error_type: Specific error to apply, or None for random
            
        Returns:
            Invalid SimulatedRead with architectural errors
        """
        if error_type is None:
            error_type = np.random.choice(
                list(self.error_probabilities.keys()),
                p=list(self.error_probabilities.values())
            )
        
        if error_type == 'segment_loss':
            return self._apply_segment_loss(valid_read)
        elif error_type == 'segment_duplication':
            return self._apply_segment_duplication(valid_read)
        elif error_type == 'truncation':
            return self._apply_truncation(valid_read)
        elif error_type == 'chimeric':
            return self._apply_chimeric(valid_read)
        elif error_type == 'scrambled':
            return self._apply_scrambled(valid_read)
        else:
            logger.warning(f"Unknown error type: {error_type}, returning original read")
            return valid_read
    
    def generate_batch(self, valid_reads: List[SimulatedRead], 
                      invalid_ratio: float = 0.1) -> List[SimulatedRead]:
        """
        Generate a batch of invalid reads from valid ones.
        
        Args:
            valid_reads: List of valid SimulatedRead objects
            invalid_ratio: Fraction of reads to make invalid
            
        Returns:
            List combining valid and invalid reads
        """
        n_invalid = int(len(valid_reads) * invalid_ratio)
        
        # Sample reads to corrupt
        indices = np.random.choice(len(valid_reads), n_invalid, replace=False)
        
        result = []
        for i, read in enumerate(valid_reads):
            if i in indices:
                result.append(self.generate_invalid_read(read))
            else:
                result.append(read)
        
        logger.debug(f"Generated batch with {n_invalid}/{len(valid_reads)} invalid reads")
        return result
    
    def _apply_segment_loss(self, read: SimulatedRead) -> SimulatedRead:
        """Remove a segment (common error pattern)."""
        segments = self._extract_segments(read)
        
        # Don't remove adapters or insert
        removable = [s for s in segments 
                    if 'ADAPTER' not in s['type'] and 'INSERT' not in s['type']]
        
        if not removable:
            return read
        
        to_remove = np.random.choice(removable)
        
        # Reconstruct without the segment
        new_sequence = (read.sequence[:to_remove['start']] + 
                       read.sequence[to_remove['end']:])
        new_labels = (read.labels[:to_remove['start']] + 
                     read.labels[to_remove['end']:])
        
        # Update label_regions
        new_regions = {}
        for label, regions in read.label_regions.items():
            if label != to_remove['type']:
                new_regions[label] = regions
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=new_regions,
            metadata={**read.metadata, 'error_type': 'segment_loss',
                     'removed_segment': to_remove['type']}
        )
    
    def _apply_segment_duplication(self, read: SimulatedRead) -> SimulatedRead:
        """Duplicate a segment (PCR artifacts)."""
        segments = self._extract_segments(read)
        
        # UMI and BARCODE commonly duplicated
        duplicatable = [s for s in segments 
                       if s['type'] in ['UMI', 'BARCODE', 'ACC']]
        
        if not duplicatable:
            return read
        
        to_dup = np.random.choice(duplicatable)
        
        # Extract segment
        seg_seq = read.sequence[to_dup['start']:to_dup['end']]
        seg_labels = read.labels[to_dup['start']:to_dup['end']]
        
        # Insert duplicate after original
        new_sequence = (read.sequence[:to_dup['end']] + 
                       seg_seq + 
                       read.sequence[to_dup['end']:])
        new_labels = (read.labels[:to_dup['end']] + 
                     seg_labels + 
                     read.labels[to_dup['end']:])
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=read.label_regions,
            metadata={**read.metadata, 'error_type': 'segment_duplication',
                     'duplicated_segment': to_dup['type']}
        )
    
    def _apply_truncation(self, read: SimulatedRead) -> SimulatedRead:
        """Truncate read (incomplete sequencing)."""
        # Truncate to 50-90% of original length
        cut_point = np.random.randint(
            len(read.sequence) // 2, 
            int(len(read.sequence) * 0.9)
        )
        
        return SimulatedRead(
            sequence=read.sequence[:cut_point],
            labels=read.labels[:cut_point],
            label_regions=self._update_regions_for_truncation(
                read.label_regions, cut_point
            ),
            metadata={**read.metadata, 'error_type': 'truncation',
                     'truncation_point': cut_point}
        )
    
    def _apply_chimeric(self, read: SimulatedRead) -> SimulatedRead:
        """Create chimeric read by mixing segments."""
        segments = self._extract_segments(read)
        
        if len(segments) < 4:
            return read
        
        # Keep adapters, scramble middle
        first = segments[0]
        last = segments[-1]
        middle = segments[1:-1]
        np.random.shuffle(middle)
        
        # Reconstruct
        new_sequence = ""
        new_labels = []
        
        for seg in [first] + middle + [last]:
            new_sequence += read.sequence[seg['start']:seg['end']]
            new_labels.extend(read.labels[seg['start']:seg['end']])
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=read.label_regions,
            metadata={**read.metadata, 'error_type': 'chimeric'}
        )
    
    def _apply_scrambled(self, read: SimulatedRead) -> SimulatedRead:
        """Completely scramble segment order."""
        segments = self._extract_segments(read)
        np.random.shuffle(segments)
        
        new_sequence = ""
        new_labels = []
        
        for seg in segments:
            new_sequence += read.sequence[seg['start']:seg['end']]
            new_labels.extend(read.labels[seg['start']:seg['end']])
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=read.label_regions,
            metadata={**read.metadata, 'error_type': 'scrambled'}
        )
    
    def _extract_segments(self, read: SimulatedRead) -> List[Dict]:
        """Extract segment information from read."""
        segments = []
        if not read.labels:
            return segments
            
        current_label = read.labels[0]
        start = 0
        
        for i, label in enumerate(read.labels[1:], 1):
            if label != current_label:
                segments.append({
                    'type': current_label,
                    'start': start,
                    'end': i
                })
                current_label = label
                start = i
        
        # Add final segment
        segments.append({
            'type': current_label,
            'start': start,
            'end': len(read.labels)
        })
        
        return segments
    
    def _update_regions_for_truncation(self, regions: Dict, cut_point: int) -> Dict:
        """Update label regions after truncation."""
        new_regions = {}
        for label, region_list in regions.items():
            new_list = []
            for start, end in region_list:
                if start < cut_point:
                    new_list.append((start, min(end, cut_point)))
            if new_list:
                new_regions[label] = new_list
        return new_regions
