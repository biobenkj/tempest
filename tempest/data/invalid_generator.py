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

    # ------------------------------------------------------------------ #
    # Initialization & config handling
    # ------------------------------------------------------------------ #
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
        """Initialize error probabilities from config or defaults."""
        if not config:
            probs = self._default_probabilities()
        elif hasattr(config, "hybrid") and config.hybrid:
            hybrid = config.hybrid
            probs = {
                "segment_loss": getattr(hybrid, "segment_loss_prob", 0.3),
                "segment_duplication": getattr(hybrid, "segment_dup_prob", 0.3),
                "truncation": getattr(hybrid, "truncation_prob", 0.2),
                "chimeric": getattr(hybrid, "chimeric_prob", 0.1),
                "scrambled": getattr(hybrid, "scrambled_prob", 0.1),
            }
        elif isinstance(config, dict) and "hybrid" in config:
            hybrid = config["hybrid"]
            probs = {
                "segment_loss": hybrid.get("segment_loss_prob", 0.3),
                "segment_duplication": hybrid.get("segment_dup_prob", 0.3),
                "truncation": hybrid.get("truncation_prob", 0.2),
                "chimeric": hybrid.get("chimeric_prob", 0.1),
                "scrambled": hybrid.get("scrambled_prob", 0.1),
            }
        else:
            probs = self._default_probabilities()

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else probs

    # ------------------------------------------------------------------ #
    # File I/O with pickle format support
    # ------------------------------------------------------------------ #
    def save_batch(
        self,
        reads: List[SimulatedRead],
        output_path: Path,
        format: str = 'pickle',
        compress: bool = True,
        create_preview: bool = True
    ) -> Dict[str, Any]:
        """
        Save a batch of invalid reads to file.
        
        Args:
            reads: List of SimulatedRead objects (may contain invalid reads)
            output_path: Output file path
            format: Output format ('pickle', 'text', 'json')
            compress: Whether to compress pickle files
            create_preview: Whether to create a preview text file
            
        Returns:
            Dictionary with save statistics
        """
        output_path = Path(output_path)
        stats = {'start_time': time.time()}
        
        if format == 'pickle':
            # Adjust filename for compression
            if compress and not output_path.suffix == '.gz':
                if output_path.suffix == '.pkl':
                    output_path = output_path.with_suffix('.pkl.gz')
                else:
                    output_path = output_path.with_suffix(output_path.suffix + '.gz')
            
            # Save pickle file with error type tracking
            save_data = {
                'reads': reads,
                'error_types': self._categorize_reads(reads),
                'error_probabilities': self.error_probabilities,
                'version': '1.0'
            }
            
            if compress:
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create preview file if requested
            if create_preview:
                preview_path = output_path.parent / f"{output_path.stem}_invalid_preview.txt"
                self._create_preview_file(reads, preview_path)
                stats['preview_file'] = str(preview_path)
            
            stats['format'] = 'pickle'
            stats['compressed'] = compress
            
        elif format == 'text':
            # Legacy text format
            with open(output_path, 'w') as f:
                f.write("# Invalid reads generated with segment-level errors\n")
                f.write(f"# Error probabilities: {json.dumps(self.error_probabilities)}\n")
                for read in reads:
                    labels_str = ' '.join(read.labels)
                    error_type = read.metadata.get('error_type', 'none') if read.metadata else 'none'
                    f.write(f"{read.sequence}\t{labels_str}\t{error_type}\n")
            stats['format'] = 'text'
            
        elif format == 'json':
            # JSON format for interoperability
            data = {
                'reads': [],
                'error_probabilities': self.error_probabilities,
                'version': '1.0'
            }
            
            for read in reads:
                data['reads'].append({
                    'sequence': read.sequence,
                    'labels': read.labels,
                    'label_regions': read.label_regions,
                    'metadata': read.metadata
                })
            
            if compress:
                with gzip.open(output_path, 'wt') as f:
                    json.dump(data, f)
            else:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            stats['format'] = 'json'
            stats['compressed'] = compress
        
        # Calculate statistics
        stats['save_time'] = time.time() - stats['start_time']
        stats['file_size'] = output_path.stat().st_size
        stats['file_size_mb'] = stats['file_size'] / (1024 * 1024)
        stats['n_sequences'] = len(reads)
        stats['output_path'] = str(output_path)
        
        # Count error types
        error_counts = self._count_error_types(reads)
        stats['error_counts'] = error_counts
        stats['invalid_ratio'] = sum(error_counts.values()) / len(reads) if reads else 0
        
        logger.info(f"Saved {len(reads)} sequences ({stats['invalid_ratio']:.1%} invalid) "
                   f"to {output_path} ({stats['file_size_mb']:.2f} MB in {stats['save_time']:.2f}s)")
        
        return stats
    
    def load_batch(
        self,
        input_path: Path,
        format: Optional[str] = None
    ) -> Tuple[List[SimulatedRead], Dict[str, Any]]:
        """
        Load a batch of reads from file.
        
        Args:
            input_path: Input file path
            format: Input format (auto-detected if None)
            
        Returns:
            Tuple of (reads, metadata)
        """
        input_path = Path(input_path)
        
        # Auto-detect format
        if format is None:
            if '.pkl' in str(input_path) or '.pickle' in str(input_path):
                format = 'pickle'
            elif '.json' in str(input_path):
                format = 'json'
            else:
                format = 'text'
        
        metadata = {}
        
        if format == 'pickle':
            if input_path.suffix == '.gz' or '.gz' in input_path.suffixes:
                with gzip.open(input_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(input_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Handle both old and new pickle formats
            if isinstance(data, dict) and 'reads' in data:
                reads = data['reads']
                metadata = {k: v for k, v in data.items() if k != 'reads'}
            else:
                reads = data
                    
        elif format == 'text':
            reads = []
            metadata['error_probabilities'] = {}
            
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# Error probabilities:'):
                        try:
                            metadata['error_probabilities'] = json.loads(line.split(':', 1)[1])
                        except:
                            pass
                    elif not line or line.startswith('#'):
                        continue
                    else:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sequence = parts[0]
                            labels = parts[1].split()
                            error_type = parts[2] if len(parts) > 2 else 'none'
                            
                            reads.append(SimulatedRead(
                                sequence=sequence,
                                labels=labels,
                                label_regions={},
                                metadata={'error_type': error_type}
                            ))
                        
        elif format == 'json':
            if input_path.suffix == '.gz':
                with gzip.open(input_path, 'rt') as f:
                    data = json.load(f)
            else:
                with open(input_path, 'r') as f:
                    data = json.load(f)
            
            reads = [SimulatedRead(**item) for item in data['reads']]
            metadata = {k: v for k, v in data.items() if k != 'reads'}
        
        logger.info(f"Loaded {len(reads)} sequences from {input_path}")
        return reads, metadata
    
    def _create_preview_file(self, reads: List[SimulatedRead], preview_path: Path):
        """
        Create a text preview file highlighting invalid reads.
        
        Args:
            reads: List of SimulatedRead objects
            preview_path: Path for preview file
        """
        n_preview = min(10, len(reads))
        
        # Count error types
        error_counts = self._count_error_types(reads)
        invalid_count = sum(error_counts.values())
        
        with open(preview_path, 'w') as f:
            f.write(f"# Invalid Read Preview - First {n_preview} sequences\n")
            f.write(f"# Total sequences: {len(reads)}\n")
            f.write(f"# Invalid sequences: {invalid_count} ({invalid_count/len(reads)*100:.1f}%)\n")
            f.write(f"# Error type distribution: {json.dumps(error_counts)}\n")
            f.write("#" + "="*70 + "\n\n")
            
            for i, read in enumerate(reads[:n_preview], 1):
                error_type = read.metadata.get('error_type', 'valid') if read.metadata else 'valid'
                
                f.write(f"# Read {i} - Type: {error_type.upper()}\n")
                labels_str = ' '.join(read.labels)
                f.write(f"{read.sequence}\t{labels_str}\n")
                
                # Add metadata if present
                if read.metadata:
                    relevant_metadata = {k: v for k, v in read.metadata.items() 
                                       if k != 'error_type'}
                    if relevant_metadata:
                        f.write(f"# Metadata: {json.dumps(relevant_metadata)}\n")
                
                # Highlight the error for invalid reads
                if error_type != 'valid' and error_type != 'none':
                    if error_type == 'segment_loss' and 'removed_segment' in read.metadata:
                        f.write(f"# --> Missing: {read.metadata['removed_segment']}\n")
                    elif error_type == 'segment_duplication' and 'duplicated_segment' in read.metadata:
                        f.write(f"# --> Duplicated: {read.metadata['duplicated_segment']}\n")
                    elif error_type == 'truncation' and 'truncation_point' in read.metadata:
                        f.write(f"# --> Truncated at position: {read.metadata['truncation_point']}\n")
                    elif error_type in ['chimeric', 'scrambled']:
                        f.write(f"# --> Segments reordered\n")
                
                f.write("\n")
        
        logger.info(f"Created invalid read preview: {preview_path}")
    
    def _categorize_reads(self, reads: List[SimulatedRead]) -> Dict[int, str]:
        """
        Categorize reads by their error type.
        
        Returns:
            Dictionary mapping read index to error type
        """
        categories = {}
        for i, read in enumerate(reads):
            if read.metadata and 'error_type' in read.metadata:
                categories[i] = read.metadata['error_type']
            else:
                categories[i] = 'valid'
        return categories
    
    def _count_error_types(self, reads: List[SimulatedRead]) -> Dict[str, int]:
        """
        Count the number of reads for each error type.
        
        Returns:
            Dictionary mapping error type to count
        """
        counts = {'valid': 0}
        for error_type in self.error_probabilities.keys():
            counts[error_type] = 0
        
        for read in reads:
            if read.metadata and 'error_type' in read.metadata:
                error_type = read.metadata['error_type']
                if error_type in counts:
                    counts[error_type] += 1
                else:
                    counts[error_type] = 1
            else:
                counts['valid'] += 1
        
        # Remove zero counts
        return {k: v for k, v in counts.items() if v > 0}
    
    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _pychoice(self, items):
        """Return one random element from a list, preserving dicts."""
        if not items:
            return None
        choice = random.choice(items)
        return choice.copy() if isinstance(choice, dict) else choice

    @staticmethod
    def _clone_metadata(read: SimulatedRead, **updates) -> Dict:
        """Copy metadata and apply updates."""
        meta = dict(read.metadata) if read.metadata else {}
        meta.update(updates)
        return meta

    @staticmethod
    def _reassemble(read: SimulatedRead, segments: List[Dict]) -> tuple[str, list[str], dict]:
        """Rebuild sequence, labels, and regions from given segment definitions."""
        seq, labels, regions, pos = "", [], {}, 0
        for seg in segments:
            s, e = seg["start"], seg["end"]
            seg_seq = read.sequence[s:e]
            seg_labels = read.labels[s:e]
            seq += seg_seq
            labels.extend(seg_labels)
            regions.setdefault(seg["type"], []).append((pos, pos + len(seg_seq)))
            pos += len(seg_seq)
        return seq, labels, regions

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #
    def generate_invalid_read(self, valid_read: SimulatedRead,
                              error_type: Optional[str] = None) -> SimulatedRead:
        """Generate an invalid read from a valid one."""
        if error_type is None:
            # Weighted choice uses numpy since random.choice lacks probability argument
            error_type = np.random.choice(
                list(self.error_probabilities.keys()),
                p=list(self.error_probabilities.values())
            )

        handlers = {
            "segment_loss": self._apply_segment_loss,
            "segment_duplication": self._apply_segment_duplication,
            "truncation": self._apply_truncation,
            "chimeric": self._apply_chimeric,
            "scrambled": self._apply_scrambled,
        }
        return handlers.get(error_type, lambda r: r)(valid_read)

    def generate_batch(self, valid_reads: List[SimulatedRead],
                       invalid_ratio: float = 0.1) -> List[SimulatedRead]:
        """Generate a batch of invalid reads from valid ones."""
        if not valid_reads:
            return []

        # Always produce at least one invalid read if ratio > 0
        n_invalid = max(1, int(len(valid_reads) * invalid_ratio)) if invalid_ratio > 0 else 0
        if n_invalid == 0:
            return valid_reads

        # Use a set for efficient membership check (robust for single-element arrays)
        choice = np.random.choice(len(valid_reads), n_invalid, replace=False)
        indices = set(np.atleast_1d(choice).tolist())

        return [
            self.generate_invalid_read(r) if i in indices else r
            for i, r in enumerate(valid_reads)
        ]
    
    def generate_and_save_batch(
        self,
        valid_reads: List[SimulatedRead],
        output_path: Path,
        invalid_ratio: float = 0.1,
        format: str = 'pickle',
        compress: bool = True,
        create_preview: bool = True
    ) -> Dict[str, Any]:
        """
        Generate invalid reads and save them directly.
        
        Args:
            valid_reads: List of valid SimulatedRead objects
            output_path: Output file path
            invalid_ratio: Fraction of reads to make invalid
            format: Output format
            compress: Whether to compress
            create_preview: Whether to create preview
            
        Returns:
            Dictionary with generation and save statistics
        """
        # Generate batch with invalid reads
        batch = self.generate_batch(valid_reads, invalid_ratio)
        
        # Save the batch
        stats = self.save_batch(batch, output_path, format, compress, create_preview)
        stats['invalid_ratio_requested'] = invalid_ratio
        
        return stats

    # ------------------------------------------------------------------ #
    # Individual corruption strategies (unchanged from original)
    # ------------------------------------------------------------------ #
    def _apply_segment_loss(self, read: SimulatedRead) -> SimulatedRead:
        """Remove a random segment (e.g., missing barcode or UMI)."""
        segments = self._extract_segments(read)
        removable = [s for s in segments
                     if "ADAPTER" not in s["type"] and "INSERT" not in s["type"]]
        if not removable:
            return read

        to_remove = self._pychoice(removable)
        if not isinstance(to_remove, dict):
            return read

        new_seq = read.sequence[:to_remove["start"]] + read.sequence[to_remove["end"]:]
        new_labels = read.labels[:to_remove["start"]] + read.labels[to_remove["end"]:]
        removed_len = to_remove["end"] - to_remove["start"]

        new_regions = {}
        for label, regions in read.label_regions.items():
            if label == to_remove["type"]:
                continue
            adjusted = []
            for start, end in regions:
                if end <= to_remove["start"]:
                    adjusted.append((start, end))
                elif start >= to_remove["end"]:
                    adjusted.append((start - removed_len, end - removed_len))
                elif start < to_remove["start"] and end > to_remove["end"]:
                    adjusted.extend([
                        (start, to_remove["start"]),
                        (to_remove["start"], end - removed_len)
                    ])
            if adjusted:
                new_regions[label] = adjusted

        meta = self._clone_metadata(read,
                                    error_type="segment_loss",
                                    removed_segment=to_remove["type"])
        return SimulatedRead(new_seq, new_labels, new_regions, meta)

    def _apply_segment_duplication(self, read: SimulatedRead) -> SimulatedRead:
        """Duplicate a random segment (PCR-like artifact)."""
        segments = self._extract_segments(read)
        duplicatable = [s for s in segments if s["type"] in ["UMI", "BARCODE", "ACC", "CBC", "i7", "i5"]] \
                       or [s for s in segments if s["type"] != "INSERT"]
        if not duplicatable:
            return read

        to_dup = self._pychoice(duplicatable)
        seg_seq = read.sequence[to_dup["start"]:to_dup["end"]]
        seg_labels = read.labels[to_dup["start"]:to_dup["end"]]
        dup_len = to_dup["end"] - to_dup["start"]

        new_seq = read.sequence[:to_dup["end"]] + seg_seq + read.sequence[to_dup["end"]:]
        new_labels = read.labels[:to_dup["end"]] + seg_labels + read.labels[to_dup["end"]:]

        new_regions = {}
        for label, regions in read.label_regions.items():
            adjusted = []
            for start, end in regions:
                if label == to_dup["type"] and start == to_dup["start"]:
                    adjusted.extend([(start, end), (end, end + dup_len)])
                elif start >= to_dup["end"]:
                    adjusted.append((start + dup_len, end + dup_len))
                else:
                    adjusted.append((start, end))
            if adjusted:
                new_regions[label] = adjusted

        meta = self._clone_metadata(read,
                                    error_type="segment_duplication",
                                    duplicated_segment=to_dup["type"])
        return SimulatedRead(new_seq, new_labels, new_regions, meta)

    def _apply_truncation(self, read: SimulatedRead) -> SimulatedRead:
        """Truncate the read to simulate incomplete sequencing."""
        min_len = max(10, len(read.sequence) // 2)
        max_len = int(len(read.sequence) * 0.9)
        cut = random.randint(min_len, max_len) if min_len < max_len else min_len
        meta = self._clone_metadata(read, error_type="truncation", truncation_point=cut)
        new_regions = self._update_regions_for_truncation(read.label_regions, cut)
        return SimulatedRead(read.sequence[:cut], read.labels[:cut], new_regions, meta)

    def _apply_chimeric(self, read: SimulatedRead,
                        other_read: Optional[SimulatedRead] = None) -> SimulatedRead:
        """Scramble middle segments while keeping adapters intact."""
        segments = self._extract_segments(read)
        if len(segments) < 4:
            return read
        middle = segments[1:-1]
        random.shuffle(middle)
        ordered = [segments[0]] + middle + [segments[-1]]
        seq, labels, regs = self._reassemble(read, ordered)
        meta = self._clone_metadata(read, error_type="chimeric")
        return SimulatedRead(seq, labels, regs, meta)

    def _apply_scrambled(self, read: SimulatedRead) -> SimulatedRead:
        """Completely randomize segment order."""
        segments = self._extract_segments(read)
        if len(segments) <= 1:
            return read
        random.shuffle(segments)
        seq, labels, regs = self._reassemble(read, segments)
        meta = self._clone_metadata(read, error_type="scrambled")
        return SimulatedRead(seq, labels, regs, meta)

    # ------------------------------------------------------------------ #
    # Segment utilities
    # ------------------------------------------------------------------ #
    def _extract_segments(self, read: SimulatedRead) -> List[Dict]:
        """Extract segment boundaries and labels."""
        if not read.labels:
            return []
        segments, start, current = [], 0, read.labels[0]
        for i, label in enumerate(read.labels[1:], 1):
            if label != current:
                segments.append({"type": current, "start": start, "end": i})
                current, start = label, i
        segments.append({"type": current, "start": start, "end": len(read.labels)})
        return segments

    @staticmethod
    def _update_regions_for_truncation(regions: Dict, cut_point: int) -> Dict:
        """Trim region definitions after truncation."""
        out = {}
        for label, lst in regions.items():
            kept = [(s, min(e, cut_point)) for s, e in lst if s < cut_point]
            if kept:
                out[label] = kept
        return out
