"""
Invalid read generator for Tempest robustness training with pickle format support.

Generates reads with segment-level architectural errors for training
models that are robust to common sequencing failures. Now supports
efficient pickle format for saving/loading.

Part of: tempest/data module
"""

import numpy as np
import random
import pickle
import gzip
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging

from tempest.data.simulator import SimulatedRead

logger = logging.getLogger(__name__)


class InvalidReadGenerator:
    """
    Generate invalid reads with segment-level architectural errors.
    
    Supports:
    - segment_loss: Missing UMI, ACC, or barcode
    - segment_duplication: Repeated segments (PCR artifacts)
    - truncation: Incomplete reads
    - chimeric: Mixed segments from different reads
    - scrambled: Randomized segment order
    """

    # Init and config handling
    def __init__(self, config=None):
        """Initialize invalid read generator."""
        self.config = config
        self.error_probabilities = self._init_error_probabilities(config)
        logger.info(f"Initialized InvalidReadGenerator with error probabilities: "
                    f"{self.error_probabilities}")
        self.mark_architectural_errors = config.get('mark_architectural_errors', True) if config else True

    @staticmethod
    def _default_probabilities() -> Dict[str, float]:
        """Return default error probabilities."""
        return {
            "segment_loss": 0.3,
            "segment_duplication": 0.3,
            "truncation": 0.2,
            "chimeric": 0.1,
            "scrambled": 0.1,
        }

    def _init_error_probabilities(self, config) -> Dict[str, float]:
        """
        Extract error probabilities from configuration.
        """
        def extract(obj):
            """Extract dict from various object types."""
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dict__"):
                return {
                    k: getattr(obj, k)
                    for k in dir(obj)
                    if not k.startswith("_")
                    and isinstance(getattr(obj, k), (int, float))
                }
            return {}
        
        # Initialize parameter containers
        params_sim = {}
        params_hybrid = {}
        
        # Try to extract from SimulationConfig or TempestConfig
        if hasattr(config, "invalid_params"):
            # Direct SimulationConfig object with invalid_params attribute
            params_sim = extract(config.invalid_params)
            logger.debug(f"Extracted invalid_params from SimulationConfig: {params_sim}")
            
        elif hasattr(config, "simulation"):
            # Full TempestConfig with simulation attribute
            sim = config.simulation
            if hasattr(sim, "invalid_params"):
                params_sim = extract(sim.invalid_params)
                logger.debug(f"Extracted invalid_params from TempestConfig.simulation: {params_sim}")
                
        elif isinstance(config, dict):
            # Dictionary config
            if "invalid_params" in config:
                # Direct invalid_params in dict (SimulationConfig dict)
                params_sim = extract(config.get("invalid_params", {}))
                logger.debug(f"Extracted invalid_params from dict: {params_sim}")
            elif "simulation" in config:
                # Nested under simulation (full config dict)
                sim = config.get("simulation", {})
                if isinstance(sim, dict):
                    params_sim = extract(sim.get("invalid_params", {}))
                    logger.debug(f"Extracted invalid_params from dict['simulation']: {params_sim}")

        # Also check for hybrid training config
        if hasattr(config, "hybrid") and config.hybrid:
            params_hybrid = extract(config.hybrid)
            logger.debug(f"Extracted params from hybrid config: {params_hybrid}")
        elif isinstance(config, dict):
            params_hybrid = extract(config.get("hybrid", {}))
            if params_hybrid:
                logger.debug(f"Extracted params from hybrid dict: {params_hybrid}")

        # Merge parameters with proper priority: simulation > hybrid > defaults
        merged = {
            "segment_loss": params_sim.get("segment_loss_prob",
                            params_hybrid.get("segment_loss_prob", 0.3)),
            "segment_duplication": params_sim.get("segment_dup_prob",
                            params_hybrid.get("segment_dup_prob", 0.3)),
            "truncation": params_sim.get("truncation_prob",
                            params_hybrid.get("truncation_prob", 0.2)),
            "chimeric": params_sim.get("chimeric_prob",
                            params_hybrid.get("chimeric_prob", 0.1)),
            "scrambled": params_sim.get("scrambled_prob",
                            params_hybrid.get("scrambled_prob", 0.1)),
        }
        
        logger.debug(f"Merged error probabilities before normalization: {merged}")

        # Normalize probabilities to sum to 1
        total = sum(merged.values())
        normalized = {k: v / total for k, v in merged.items()}
        
        logger.debug(f"Final normalized error probabilities: {normalized}")
        return normalized

    def _invalid_regions_from_labels(
        self, 
        labels: List[str], 
        exclude: Optional[set] = None,
        span_through: Optional[set] = None
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Recompute compressed segments with smart error handling.
    
        This function performs two key operations:
        1. GHOST RESOLUTION: Replace benign error labels (span_through) with their 
        preceding valid label to maintain segment continuity
        2. RUN-LENGTH ENCODING: Compress consecutive identical labels into 
        (start, end) coordinate tuples
    
        Args:
            labels: List of per-base labels for the entire sequence
            exclude: Labels to skip completely (architectural errors like ERROR_BOUNDARY)
                    These won't appear in the output regions
            span_through: Labels to treat as transparent (benign errors like ERROR_SUB)
                        These get replaced with their context before encoding
        
        Returns:
            Dictionary mapping label names to list of (start, end) tuples
            Example: {'UMI': [(0, 8), (50, 58)], 'ACC': [(8, 14)]}
    
        Example:
            labels = ['UMI', 'UMI', 'ERROR_SUB', 'UMI', 'ACC', 'ACC', 'ERROR_BOUNDARY']
            span_through = {'ERROR_SUB'}
            exclude = {'ERROR_BOUNDARY'}
        
            After ghosting: ['UMI', 'UMI', 'UMI', 'UMI', 'ACC', 'ACC', 'ERROR_BOUNDARY']
            After RLE:      {'UMI': [(0, 4)], 'ACC': [(4, 6)]}
        """
        # Set defaults for error handling
        exclude = exclude or set()
        span_through = span_through or {'ERROR_SUB', 'ERROR_INS'}
    
        # =========================================================================
        # PHASE 1: GHOST RESOLUTION
        # Replace span_through error labels with their preceding valid label
        # This maintains segment continuity despite sequencing errors
        # =========================================================================
        resolved_labels = []
        last_valid_label = None
    
        for label in labels:
            if label in span_through:
                # This is a benign error - "ghost" it by using the previous valid label
                # If we haven't seen a valid label yet, keep the error label as-is
                ghosted_label = last_valid_label if last_valid_label is not None else label
                resolved_labels.append(ghosted_label)
            else:
                # This is a structural label or architectural error
                # Update our tracking of the last valid label (unless it's excluded)
                if label not in exclude:
                    last_valid_label = label
                resolved_labels.append(label)
    
        # =========================================================================
        # PHASE 2: RUN-LENGTH ENCODING
        # Compress consecutive identical labels into coordinate ranges
        # =========================================================================
    
        # Output dictionary: label_name -> list of (start, end) coordinate pairs
        label_regions = {}
    
        # Tracking variables for the scan
        current_position = 0
        total_length = len(resolved_labels)
    
        # Scan through the resolved label sequence
        while current_position < total_length:
            current_label = resolved_labels[current_position]
        
            # Skip labels that should be excluded or are still error labels
            # (error labels would only remain if they appeared at the start before any valid label)
            if current_label in exclude or current_label is None or current_label in span_through:
                current_position += 1
                continue
        
            # Found a valid segment start - now find where this run ends
            # A "run" is a contiguous stretch of identical labels
            segment_start = current_position
            segment_end = current_position + 1
        
            # Extend segment_end while the label matches current_label
            while segment_end < total_length and resolved_labels[segment_end] == current_label:
                segment_end += 1
        
            # Record this segment's coordinates [segment_start, segment_end)
            # Note: segment_end is exclusive (Python slice convention)
            # Example: segment_start=0, segment_end=8 means positions 0,1,2,3,4,5,6,7
            if current_label not in label_regions:
                label_regions[current_label] = []
            label_regions[current_label].append((segment_start, segment_end))
        
            # Move to the next potential segment (start of next different label)
            current_position = segment_end
    
        return label_regions

    # File I/O
    # don't actually think I need this anymore
    def save_invalid_reads(self, 
                          invalid_reads: List[SimulatedRead], 
                          output_path: Path, 
                          format: str = 'pickle',
                          compress: bool = True) -> Dict[str, Any]:
        """
        Save invalid reads to file with metadata.
        
        Args:
            invalid_reads: List of invalid reads to save
            output_path: Output file path
            format: 'pickle', 'json', or 'text'
            compress: Whether to compress pickle files
            
        Returns:
            Metadata dictionary with save statistics
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            if compress:
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump(invalid_reads, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(invalid_reads, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == 'json':
            # Convert to JSON-serializable format
            data = []
            for read in invalid_reads:
                data.append({
                    'sequence': read.sequence,
                    'labels': read.labels,
                    'label_regions': read.label_regions,
                    'metadata': read.metadata
                })
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'text':
            with open(output_path, 'w') as f:
                for read in invalid_reads:
                    labels = ' '.join(read.labels) if read.labels else ''
                    metadata = json.dumps(read.metadata) if read.metadata else ''
                    f.write(f"{read.sequence}\t{labels}\t{metadata}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        save_time = time.time() - start_time
        file_size = output_path.stat().st_size
        
        # Collect error type statistics
        error_types = {}
        for read in invalid_reads:
            if read.metadata and 'error_type' in read.metadata:
                error_type = read.metadata['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'num_reads': len(invalid_reads),
            'file_path': str(output_path),
            'file_size_mb': file_size / (1024 * 1024),
            'save_time_sec': save_time,
            'format': format,
            'compressed': compress if format == 'pickle' else False,
            'error_types': error_types
        }
    
    def load_invalid_reads(self, 
                          input_path: Path, 
                          format: str = 'auto') -> List[SimulatedRead]:
        """
        Load invalid reads from file.
        
        Args:
            input_path: Input file path
            format: 'pickle', 'json', 'text', or 'auto' (detect from extension)
            
        Returns:
            List of SimulatedRead objects
        """
        input_path = Path(input_path)
        
        if format == 'auto':
            if input_path.suffix in ['.pkl', '.pickle']:
                format = 'pickle'
            elif input_path.suffix in ['.pkl.gz', '.pickle.gz']:
                format = 'pickle'
            elif input_path.suffix == '.json':
                format = 'json'
            else:
                format = 'text'
        
        if format == 'pickle':
            opener = gzip.open if input_path.suffix.endswith('.gz') else open
            with opener(input_path, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'json':
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            reads = []
            for item in data:
                labels = item.get('labels', [])
                # Compute label_regions if not provided
                label_regions = item.get('label_regions', 
                                        self._invalid_regions_from_labels(labels, exclude={"ERROR"}))
                
                read = SimulatedRead(
                    sequence=item['sequence'],
                    labels=labels,
                    label_regions=label_regions,
                    metadata=item.get('metadata', {})
                )
                reads.append(read)
            return reads
        
        elif format == 'text':
            reads = []
            with open(input_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        sequence = parts[0]
                        labels = parts[1].split() if len(parts) > 1 and parts[1] else []
                        metadata = json.loads(parts[2]) if len(parts) > 2 and parts[2] else {}
                        
                        # Compute label_regions from labels
                        label_regions = self._invalid_regions_from_labels(labels, exclude={"ERROR"})
                        
                        reads.append(SimulatedRead(
                            sequence=sequence,
                            labels=labels,
                            label_regions=label_regions,
                            metadata=metadata
                        ))
            return reads
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # Error generation methods
    def generate_segment_loss(self, read: SimulatedRead) -> SimulatedRead:
        """Remove a segment and optionally mark the boundary."""
        available = [k for k in read.label_regions.keys() 
                     if k in ['UMI', 'ACC', 'CBC']]
        if not available:
            return read
        
        to_remove = random.choice(available)
        regions = read.label_regions[to_remove]
        
        new_sequence = []
        new_labels = []
        removed_ranges = []
        
        for region_start, region_end in regions:
            removed_ranges.append((region_start, region_end))
        
        last_end = 0
        for start, end in sorted(removed_ranges):
            # Keep everything before this segment
            new_sequence.append(read.sequence[last_end:start])
            new_labels.extend(read.labels[last_end:start])
            
            # OPTIONALLY: Mark the deletion boundary
            if self.mark_architectural_errors:
                # Add a single ERROR_LOSS marker at the boundary
                new_labels.append("ERROR_LOSS")
                # Don't add to sequence (it's a marker, not a base)
            
            last_end = end
        
        # Add remainder
        new_sequence.append(read.sequence[last_end:])
        new_labels.extend(read.labels[last_end:])
        
        final_sequence = ''.join(new_sequence)
        
        # Recompute label_regions with error handling
        label_regions = self._invalid_regions_from_labels(
            new_labels, 
            span_through={'ERROR_SUB', 'ERROR_INS'},  # ← CHANGED
            exclude={'ERROR_LOSS', 'ERROR_BOUNDARY'}
        )
        
        return SimulatedRead(
            sequence=final_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'is_invalid': True,
                'error_type': 'segment_loss',
                'removed_segment': to_remove
            }
        )
    
    def generate_segment_duplication(self, read: SimulatedRead) -> SimulatedRead:
        """
        Duplicate random segments (PCR artifacts).
        
        Optionally marks duplication boundaries with ERROR_DUP to help the model
        learn that these are artifactual repeats rather than biological structure.
        """
        duplicatable = ["UMI", "ACC", "CBC", "i7", "i5"]
        available = [seg for seg in duplicatable if seg in read.labels]
        
        if not available:
            return read
        
        # Duplicate 1-2 segments
        n_duplicated_segments = min(random.randint(1, 2), len(available))
        segments_to_duplicate = set(random.sample(available, n_duplicated_segments))
        
        new_sequence = ""
        new_labels = []
        
        # Iterate over segments properly
        current_position = 0
        total_length = len(read.labels)
        
        while current_position < total_length:
            current_label = read.labels[current_position]
            
            # Find end of current segment (run-length encode on the fly)
            segment_start = current_position
            segment_end = current_position
            while segment_end < total_length and read.labels[segment_end] == current_label:
                segment_end += 1
            
            # Extract the segment sequence and labels
            segment_sequence = read.sequence[segment_start:segment_end]
            segment_length = len(segment_sequence)
            
            # Add original segment
            new_sequence += segment_sequence
            new_labels.extend([current_label] * segment_length)
            
            # Duplicate if this segment was selected
            if current_label in segments_to_duplicate:
                n_copies = random.randint(1, 3)
                
                for copy_num in range(n_copies):
                    # Mark the duplication boundary (junction between original and copy)
                    if self.mark_architectural_errors:
                        # Insert ERROR_DUP marker at the start of each duplicated copy
                        # This marks an unnatural boundary where the segment repeats
                        new_labels.append("ERROR_DUP")
                        # Note: No sequence base added - this is a positional marker
                    
                    # Add the duplicated segment
                    new_sequence += segment_sequence
                    new_labels.extend([current_label] * segment_length)
            
            # Move to next segment
            current_position = segment_end
        
        # Compute label_regions with proper error handling
        # ERROR_DUP markers are excluded (boundaries), benign errors span through
        label_regions = self._invalid_regions_from_labels(
            new_labels,
            span_through={'ERROR_SUB', 'ERROR_INS'},
            exclude={'ERROR_DUP', 'ERROR_BOUNDARY', 'ERROR_LOSS'}
        )
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'is_invalid': True,
                'error_type': 'segment_duplication',
                'duplicated_segments': list(segments_to_duplicate),
                'n_duplications': sum(1 for l in new_labels if l == 'ERROR_DUP')
            }
        )
    
    def generate_truncation(self, read: SimulatedRead) -> SimulatedRead:
        """Truncate read and mark the break point."""
        min_length = max(50, len(read.sequence) // 4)
        max_length = len(read.sequence) - 20
        
        if min_length >= max_length:
            return read

        truncate_pos = random.randint(min_length, max_length)

        if random.random() < 0.5:
            # 5' truncation
            new_sequence = read.sequence[truncate_pos:]
            new_labels = read.labels[truncate_pos:]
            
            # Mark the 5' boundary
            if self.mark_architectural_errors and new_labels:
                new_labels[0] = "ERROR_BOUNDARY"
            
            truncation_type = "5_prime"
        else:
            # 3' truncation
            new_sequence = read.sequence[:truncate_pos]
            new_labels = read.labels[:truncate_pos]
            
            # Mark the 3' boundary
            if self.mark_architectural_errors and new_labels:
                new_labels[-1] = "ERROR_BOUNDARY"
            
            truncation_type = "3_prime"
        
        label_regions = self._invalid_regions_from_labels(
            new_labels,
            span_through={'ERROR_SUB', 'ERROR_INS'},
            exclude={'ERROR_BOUNDARY'}
        )
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'is_invalid': True,
                'error_type': 'truncation',
                'truncation_type': truncation_type
            }
        )
    
    def generate_chimeric(self, read: SimulatedRead, other_read: Optional[SimulatedRead] = None) -> SimulatedRead:
        """
        Create chimeric read from two reads with marked junction point.
        
        Chimeric reads occur when two fragments from different molecules are joined,
        commonly in library prep or PCR. The breakpoint is marked with ERROR_BOUNDARY
        to help the model learn these unnatural junctions.
        """
        if other_read is None:
            # Create a synthetic "other" read by scrambling current read
            other_read = self.generate_scrambled(read)
        
        # Find breakpoint in both reads (avoid very edges)
        min_break_position = len(read.sequence) // 4
        max_break_position = 3 * len(read.sequence) // 4
        breakpoint_read1 = random.randint(min_break_position, max_break_position)
        
        min_break_position_2 = len(other_read.sequence) // 4
        max_break_position_2 = 3 * len(other_read.sequence) // 4
        breakpoint_read2 = random.randint(min_break_position_2, max_break_position_2)
        
        # Take 5' portion of first read
        sequence_5prime = read.sequence[:breakpoint_read1]
        labels_5prime = read.labels[:breakpoint_read1]
        
        # Take 3' portion of second read
        sequence_3prime = other_read.sequence[breakpoint_read2:]
        labels_3prime = other_read.labels[breakpoint_read2:]
        
        # Mark the chimeric junction if enabled
        if self.mark_architectural_errors and labels_3prime:
            # Mark the first base of the 3' portion as the junction point
            # This indicates an unnatural splice between two molecules
            labels_3prime = labels_3prime.copy()  # Avoid modifying original
            labels_3prime[0] = "ERROR_BOUNDARY"
        
        # Combine the two parts
        new_sequence = sequence_5prime + sequence_3prime
        new_labels = labels_5prime + labels_3prime
        
        # Compute label_regions with proper error handling
        # ERROR_BOUNDARY is excluded (marks unnatural junction)
        label_regions = self._invalid_regions_from_labels(
            new_labels,
            span_through={'ERROR_SUB', 'ERROR_INS'},
            exclude={'ERROR_BOUNDARY', 'ERROR_DUP', 'ERROR_LOSS'}
        )
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'is_invalid': True,
                'error_type': 'chimeric',
                'breakpoint_read1': breakpoint_read1,
                'breakpoint_read2': breakpoint_read2,
                'chimeric_junction_marked': self.mark_architectural_errors
            }
        )
    
    def generate_scrambled(self, read: SimulatedRead) -> SimulatedRead:
        """Scramble segments and mark order errors."""
        # Extract segments
        segments = []
        current_segment = []
        current_label = None
        
        for i, label in enumerate(read.labels):
            if label != current_label:
                if current_segment:
                    segments.append((current_label, ''.join(current_segment)))
                current_segment = [read.sequence[i]]
                current_label = label
            else:
                current_segment.append(read.sequence[i])
        
        if current_segment:
            segments.append((current_label, ''.join(current_segment)))
        
        if len(segments) <= 1:
            return read
        
        # Shuffle
        original_order = [seg[0] for seg in segments]
        random.shuffle(segments)
        shuffled_order = [seg[0] for seg in segments]
        
        # Reconstruct with boundary markers
        new_sequence = ""
        new_labels = []
        
        for i, (label, seq) in enumerate(segments):
            # Mark segment boundaries that are out of order
            if self.mark_architectural_errors and i > 0:
                if shuffled_order[i] != original_order[i]:
                    # This segment is out of order - mark first base
                    new_sequence += seq
                    new_labels.append("ERROR_ORDER")
                    new_labels.extend([label] * (len(seq) - 1))
                else:
                    new_sequence += seq
                    new_labels.extend([label] * len(seq))
            else:
                new_sequence += seq
                new_labels.extend([label] * len(seq))
        
        label_regions = self._invalid_regions_from_labels(
            new_labels,
            span_through={'ERROR_SUB', 'ERROR_INS'},
            exclude={'ERROR_ORDER'}
        )
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'is_invalid': True,
                'error_type': 'scrambled',
                'original_order': original_order,
                'shuffled_order': shuffled_order
            }
        )
    
    def generate_invalid_read(self, read: SimulatedRead, error_type: Optional[str] = None) -> SimulatedRead:
        """
        Generate an invalid read with specified or random error type.
        
        Args:
            read: Valid input read
            error_type: Specific error type or None for random selection
            
        Returns:
            Invalid read with error
        """
        if error_type is None:
            # Select error type based on probabilities
            error_type = np.random.choice(
                list(self.error_probabilities.keys()),
                p=list(self.error_probabilities.values())
            )

        if error_type == "segment_loss":
            return self.generate_segment_loss(read)
        elif error_type == "segment_duplication":
            return self.generate_segment_duplication(read)
        elif error_type == "truncation":
            return self.generate_truncation(read)
        elif error_type == "chimeric":
            return self.generate_chimeric(read)
        elif error_type == "scrambled":
            return self.generate_scrambled(read)
        else:
            logger.warning(f"Unknown error type: {error_type}, returning original read")
            return read
    
    def generate_batch(self, reads: List[SimulatedRead], invalid_fraction: float = 0.2) -> List[SimulatedRead]:
        """
        Generate batch of reads with specified fraction of invalid reads.
        
        Args:
            reads: List of valid reads
            invalid_fraction: Fraction of reads to make invalid
            
        Returns:
            Mixed list of valid and invalid reads
        """
        n_invalid = int(len(reads) * invalid_fraction)
        invalid_indices = set(random.sample(range(len(reads)), n_invalid))
        
        result = []
        for i, read in enumerate(reads):
            if i in invalid_indices:
                result.append(self.generate_invalid_read(read))
            else:
                result.append(read)
        
        # Shuffle to mix valid and invalid
        random.shuffle(result)
        return result