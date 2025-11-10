"""
Patch for simulator.py to add pickle format support.

This file contains the modified methods to be added/replaced in the SequenceSimulator class
to support pickle format as the default output format with preview text files.
"""

import pickle
import gzip
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SequenceSimulator:
    """
    Methods to add to SequenceSimulator class for pickle format support.
    These methods should be added to the existing SequenceSimulator class.
    """
    
    def save_reads(
        self, 
        reads: List['SimulatedRead'], 
        output_path: Path,
        format: str = 'pickle',
        compress: bool = True,
        create_preview: bool = True
    ) -> Dict[str, Any]:
        """
        Save simulated reads to file with format options.
        
        Args:
            reads: List of SimulatedRead objects
            output_path: Output file path
            format: Output format ('pickle', 'text', 'json')
            compress: Whether to compress pickle files
            create_preview: Whether to create a preview text file
            
        Returns:
            Dictionary with save statistics
        """
        output_path = Path(output_path)
        stats = {}
        start_time = time.time()
        
        if format == 'pickle':
            # Adjust filename for compression
            if compress and not output_path.suffix == '.gz':
                if output_path.suffix == '.pkl':
                    output_path = output_path.with_suffix('.pkl.gz')
                else:
                    output_path = output_path.with_suffix(output_path.suffix + '.gz')
            
            # Save pickle file
            if compress:
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create preview file if requested
            if create_preview:
                preview_path = output_path.parent / f"{output_path.stem}_preview.txt"
                self._create_preview_file(reads, preview_path)
                stats['preview_file'] = str(preview_path)
            
            stats['format'] = 'pickle'
            stats['compressed'] = compress
            
        elif format == 'text':
            # Legacy text format
            with open(output_path, 'w') as f:
                for read in reads:
                    labels_str = ' '.join(read.labels)
                    f.write(f"{read.sequence}\t{labels_str}\n")
            stats['format'] = 'text'
            
        elif format == 'json':
            # JSON format for interoperability
            data = []
            for read in reads:
                data.append({
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
        stats['save_time'] = time.time() - start_time
        stats['file_size'] = output_path.stat().st_size
        stats['file_size_mb'] = stats['file_size'] / (1024 * 1024)
        stats['n_sequences'] = len(reads)
        stats['output_path'] = str(output_path)
        
        logger.info(f"Saved {len(reads)} sequences to {output_path} "
                   f"({stats['file_size_mb']:.2f} MB in {stats['save_time']:.2f}s)")
        
        return stats
    
    def _create_preview_file(self, reads: List['SimulatedRead'], preview_path: Path):
        """
        Create a text preview file with the first 10 reads.
        
        Args:
            reads: List of SimulatedRead objects
            preview_path: Path for preview file
        """
        n_preview = min(10, len(reads))
        
        with open(preview_path, 'w') as f:
            f.write(f"# Preview of first {n_preview} sequences\n")
            f.write(f"# Total sequences in dataset: {len(reads)}\n")
            f.write(f"# Format: sequence<TAB>labels\n")
            f.write("#" + "="*70 + "\n\n")
            
            for i, read in enumerate(reads[:n_preview], 1):
                f.write(f"# Read {i}\n")
                labels_str = ' '.join(read.labels)
                f.write(f"{read.sequence}\t{labels_str}\n")
                
                # Add metadata if present
                if read.metadata:
                    f.write(f"# Metadata: {json.dumps(read.metadata)}\n")
                
                # Add label regions summary
                if read.label_regions:
                    regions_summary = []
                    for label, regions in read.label_regions.items():
                        ranges = [f"{s}-{e}" for s, e in regions]
                        regions_summary.append(f"{label}:[{','.join(ranges)}]")
                    f.write(f"# Regions: {'; '.join(regions_summary)}\n")
                
                # Show segment boundaries visually
                if i <= 3:  # Only for first 3 reads
                    f.write("# Segments: ")
                    segment_viz = []
                    for label, regions in read.label_regions.items():
                        for start, end in regions:
                            segment_viz.append(f"{label}({end-start}bp)")
                    f.write(" | ".join(segment_viz) + "\n")
                
                f.write("\n")
        
        logger.info(f"Created preview file with {n_preview} sequences: {preview_path}")
    
    def load_reads(
        self, 
        input_path: Path,
        format: Optional[str] = None
    ) -> List['SimulatedRead']:
        """
        Load simulated reads from file.
        
        Args:
            input_path: Input file path
            format: Input format (auto-detected if None)
            
        Returns:
            List of SimulatedRead objects
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
        
        if format == 'pickle':
            if input_path.suffix == '.gz' or '.gz' in input_path.suffixes:
                with gzip.open(input_path, 'rb') as f:
                    reads = pickle.load(f)
            else:
                with open(input_path, 'rb') as f:
                    reads = pickle.load(f)
                    
        elif format == 'text':
            reads = []
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        sequence, labels_str = parts
                        labels = labels_str.split()
                        # Note: We lose label_regions and metadata in text format
                        from tempest.data import SimulatedRead
                        reads.append(SimulatedRead(
                            sequence=sequence,
                            labels=labels,
                            label_regions={},
                            metadata={}
                        ))
                        
        elif format == 'json':
            if input_path.suffix == '.gz':
                with gzip.open(input_path, 'rt') as f:
                    data = json.load(f)
            else:
                with open(input_path, 'r') as f:
                    data = json.load(f)
            
            from tempest.data import SimulatedRead
            reads = [SimulatedRead(**item) for item in data]
        
        logger.info(f"Loaded {len(reads)} sequences from {input_path}")
        return reads
    
    def generate_and_save(
        self,
        n_sequences: int,
        output_path: Path,
        format: str = 'pickle',
        compress: bool = True,
        create_preview: bool = True,
        split: bool = False,
        train_fraction: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate sequences and save them directly to file.
        
        Args:
            n_sequences: Number of sequences to generate
            output_path: Output file/directory path
            format: Output format
            compress: Whether to compress
            create_preview: Whether to create preview file
            split: Whether to create train/val split
            train_fraction: Fraction for training if split
            
        Returns:
            Dictionary with generation and save statistics
        """
        result = {}
        output_path = Path(output_path)
        
        if split:
            # Generate train/val split
            n_train = int(n_sequences * train_fraction)
            n_val = n_sequences - n_train
            
            logger.info(f"Generating {n_train} training sequences...")
            train_reads = self.generate_reads(n_train)
            
            logger.info(f"Generating {n_val} validation sequences...")
            val_reads = self.generate_reads(n_val)
            
            # Determine output paths
            if output_path.is_dir() or not output_path.suffix:
                output_path.mkdir(parents=True, exist_ok=True)
                if format == 'pickle':
                    ext = '.pkl.gz' if compress else '.pkl'
                else:
                    ext = '.txt'
                train_path = output_path / f"train{ext}"
                val_path = output_path / f"val{ext}"
            else:
                # Use stem of provided path
                stem = output_path.stem
                parent = output_path.parent
                parent.mkdir(parents=True, exist_ok=True)
                if format == 'pickle':
                    ext = '.pkl.gz' if compress else '.pkl'
                else:
                    ext = '.txt'
                train_path = parent / f"{stem}_train{ext}"
                val_path = parent / f"{stem}_val{ext}"
            
            # Save both sets
            train_stats = self.save_reads(train_reads, train_path, format, compress, create_preview)
            val_stats = self.save_reads(val_reads, val_path, format, compress, create_preview)
            
            result['train'] = train_stats
            result['val'] = val_stats
            result['n_train'] = n_train
            result['n_val'] = n_val
            
        else:
            # Generate single dataset
            logger.info(f"Generating {n_sequences} sequences...")
            reads = self.generate_reads(n_sequences)
            
            # Save to file
            save_stats = self.save_reads(reads, output_path, format, compress, create_preview)
            result['save_stats'] = save_stats
            result['n_sequences'] = n_sequences
        
        result['format'] = format
        result['compressed'] = compress if format in ['pickle', 'json'] else False
        
        return result


def create_simulator_from_config(config) -> 'SequenceSimulator':
    """
    Enhanced factory function to create simulator from config.
    
    Args:
        config: Either a file path (str/Path) or a TempestConfig object
        
    Returns:
        Configured simulator instance with pickle support
    """
    from tempest.data import SequenceSimulator
    
    # Handle different config types
    if isinstance(config, (str, Path)):
        # Config file path
        simulator = SequenceSimulator(config_file=str(config))
    elif hasattr(config, 'simulation'):
        # TempestConfig object
        simulator = SequenceSimulator(config=config)
    else:
        # Dictionary config
        simulator = SequenceSimulator(config=config)
    
    # Add pickle support methods if not already present
    if not hasattr(simulator, 'save_reads'):
        # Monkey-patch the methods
        simulator.save_reads = SequenceSimulator.save_reads.__get__(simulator)
        simulator._create_preview_file = SequenceSimulator._create_preview_file.__get__(simulator)
        simulator.load_reads = SequenceSimulator.load_reads.__get__(simulator)
        simulator.generate_and_save = SequenceSimulator.generate_and_save.__get__(simulator)
    
    return simulator
