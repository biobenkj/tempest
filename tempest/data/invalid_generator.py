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
        
        FIXED: Now properly handles SimulationConfig objects passed directly.
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
        exclude: Optional[set] = None
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Recompute compressed segments from labels after error editing.
        Excludes ephemeral labels such as "ERROR".
        
        Args:
            labels: List of per-base labels
            exclude: Set of labels to exclude (e.g., {"ERROR"})
            
        Returns:
            Dictionary mapping label names to list of (start, end) tuples
        """
        exclude = exclude or set()
        out = {}
        i, n = 0, len(labels)
        while i < n:
            lab = labels[i]
            if lab in exclude:
                i += 1
                continue
            j = i + 1
            while j < n and labels[j] == lab:
                j += 1
            out.setdefault(lab, []).append((i, j))
            i = j
        return out

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
        """Remove random segments from read."""
        # Select segments to remove (prefer variable segments)
        removable = ["UMI", "ACC", "CBC", "i7", "i5", "cDNA", "polyA"]
        available = [seg for seg in removable if seg in read.labels]
        
        if not available:
            return read
        
        # Remove 1-3 segments
        n_remove = min(random.randint(1, 3), len(available))
        to_remove = set(random.sample(available, n_remove))
        
        new_sequence = ""
        new_labels = []
        
        # Iterate over segments properly
        i = 0
        n = len(read.labels)
        while i < n:
            label = read.labels[i]
            # Find end of current segment
            j = i
            while j < n and read.labels[j] == label:
                j += 1
            
            # Keep segment if not in removal list
            if label not in to_remove:
                segment_seq = read.sequence[i:j]
                new_sequence += segment_seq
                new_labels.extend([label] * len(segment_seq))
            
            i = j
        
        # Compute label_regions from modified labels
        label_regions = self._invalid_regions_from_labels(new_labels, exclude={"ERROR"})
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'error_type': 'segment_loss',
                'removed_segments': list(to_remove)
            }
        )
    
    def generate_segment_duplication(self, read: SimulatedRead) -> SimulatedRead:
        """Duplicate random segments (PCR artifacts)."""
        duplicatable = ["UMI", "ACC", "CBC", "i7", "i5"]
        available = [seg for seg in duplicatable if seg in read.labels]
        
        if not available:
            return read
        
        # Duplicate 1-2 segments
        n_dup = min(random.randint(1, 2), len(available))
        to_duplicate = set(random.sample(available, n_dup))
        
        new_sequence = ""
        new_labels = []
        
        # Iterate over segments properly
        i = 0
        n = len(read.labels)
        while i < n:
            label = read.labels[i]
            # Find end of current segment
            j = i
            while j < n and read.labels[j] == label:
                j += 1
            
            segment_seq = read.sequence[i:j]
            new_sequence += segment_seq
            new_labels.extend([label] * len(segment_seq))
            
            # Duplicate if selected
            if label in to_duplicate:
                n_copies = random.randint(1, 3)
                for _ in range(n_copies):
                    new_sequence += segment_seq
                    new_labels.extend([label] * len(segment_seq))
            
            i = j
        
        # Compute label_regions from modified labels
        label_regions = self._invalid_regions_from_labels(new_labels, exclude={"ERROR"})
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'error_type': 'segment_duplication',
                'duplicated_segments': list(to_duplicate)
            }
        )
    
    def generate_truncation(self, read: SimulatedRead) -> SimulatedRead:
        """Truncate read at random position."""
        min_length = max(50, len(read.sequence) // 4)
        max_length = len(read.sequence) - 20
        
        if min_length >= max_length:
            return read

        truncate_pos = random.randint(min_length, max_length)

        # Decide 5' or 3' truncation
        if random.random() < 0.5:
            # 5' truncation
            new_sequence = read.sequence[truncate_pos:]
            new_labels = read.labels[truncate_pos:]
            truncation_type = "5_prime"
        else:
            # 3' truncation
            new_sequence = read.sequence[:truncate_pos]
            new_labels = read.labels[:truncate_pos]
            truncation_type = "3_prime"
        
        # Compute label_regions from modified labels
        label_regions = self._invalid_regions_from_labels(new_labels, exclude={"ERROR"})
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'error_type': 'truncation',
                'truncation_type': truncation_type,
                'original_length': len(read.sequence),
                'truncated_length': len(new_sequence)
            }
        )
    
    def generate_chimeric(self, read: SimulatedRead, other_read: Optional[SimulatedRead] = None) -> SimulatedRead:
        """Create chimeric read from two reads."""
        if other_read is None:
            # Create a synthetic "other" read by scrambling current read
            other_read = self.generate_scrambled(read)
        
        # Find breakpoint in both reads
        break1 = random.randint(len(read.sequence) // 4, 3 * len(read.sequence) // 4)
        break2 = random.randint(len(other_read.sequence) // 4, 3 * len(other_read.sequence) // 4)
        
        # Combine parts
        new_sequence = read.sequence[:break1] + other_read.sequence[break2:]
        new_labels = read.labels[:break1] + other_read.labels[break2:]
        
        # Compute label_regions from modified labels
        label_regions = self._invalid_regions_from_labels(new_labels, exclude={"ERROR"})
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
                'error_type': 'chimeric',
                'break_position_1': break1,
                'break_position_2': break2
            }
        )
    
    def generate_scrambled(self, read: SimulatedRead) -> SimulatedRead:
        """Scramble segment order."""
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
        
        # Shuffle segments
        original_order = [seg[0] for seg in segments]
        random.shuffle(segments)
        shuffled_order = [seg[0] for seg in segments]
        
        # Reconstruct read
        new_sequence = ""
        new_labels = []
        for label, seq in segments:
            new_sequence += seq
            new_labels.extend([label] * len(seq))
        
        # Compute label_regions from modified labels
        label_regions = self._invalid_regions_from_labels(new_labels, exclude={"ERROR"})
        
        return SimulatedRead(
            sequence=new_sequence,
            labels=new_labels,
            label_regions=label_regions,
            metadata={
                **read.metadata,
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
