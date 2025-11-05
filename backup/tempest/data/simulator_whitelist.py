"""
Enhanced Data Simulator for Tempest with Whitelist Support.

Generates synthetic training data with configurable sequence structures,
ACC PWM-based generation, error injection, and support for sequence whitelists.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRead:
    """Container for a simulated read with labels."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict  # Additional info


@dataclass
class SimulationConfig:
    """Configuration for sequence simulation."""
    sequence_order: List[str]
    num_sequences: int = 10000
    error_rate: float = 0.02
    random_seed: int = 42
    
    # Component lengths
    umi_length: int = 8
    insert_min_length: int = 50
    insert_max_length: int = 150
    
    # File paths
    acc_priors_file: Optional[str] = None
    barcode_file: Optional[str] = None
    pwm_file: Optional[str] = None
    
    # Whitelist files for specific segments
    i7_whitelist_file: Optional[str] = None
    i5_whitelist_file: Optional[str] = None
    cbc_whitelist_file: Optional[str] = None
    
    # Generic whitelist mapping
    whitelist_files: Optional[Dict[str, str]] = None
    
    # Fixed sequences
    sequences: Optional[Dict[str, Union[str, List[str]]]] = None
    
    # ACC configuration
    acc_sequences: Optional[List[str]] = None
    acc_frequencies: Optional[List[float]] = None


class WhitelistLoader:
    """Handles loading and managing sequence whitelists."""
    
    def __init__(self):
        self.whitelists = {}
    
    def load_whitelist(self, filepath: str, segment_name: str) -> List[str]:
        """
        Load sequences from a whitelist file.
        
        Args:
            filepath: Path to whitelist file (one sequence per line)
            segment_name: Name of the segment (for logging)
            
        Returns:
            List of sequences
        """
        sequences = []
        
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"Whitelist file not found for {segment_name}: {filepath}")
                return sequences
            
            with open(filepath, 'r') as f:
                for line in f:
                    seq = line.strip().upper()
                    # Validate DNA sequence
                    if seq and all(base in 'ACGTN' for base in seq):
                        sequences.append(seq)
                    elif seq:
                        logger.warning(f"Skipping invalid sequence in {segment_name} whitelist: {seq}")
            
            logger.info(f"Loaded {len(sequences)} sequences from {segment_name} whitelist")
            
        except Exception as e:
            logger.error(f"Error loading {segment_name} whitelist from {filepath}: {e}")
        
        return sequences
    
    def add_whitelist(self, segment_name: str, filepath: str):
        """Add a whitelist for a specific segment."""
        sequences = self.load_whitelist(filepath, segment_name)
        if sequences:
            self.whitelists[segment_name.upper()] = sequences
            return True
        return False
    
    def get_sequence(self, segment_name: str, random_state: np.random.RandomState) -> Optional[str]:
        """
        Get a random sequence from whitelist for the specified segment.
        
        Args:
            segment_name: Name of the segment
            random_state: Random state for reproducibility
            
        Returns:
            Random sequence from whitelist or None if not available
        """
        segment_key = segment_name.upper()
        
        # Check various possible keys
        possible_keys = [
            segment_key,
            segment_key.replace('_', ''),
            segment_key.replace('INDEX_', ''),
            segment_key.replace('INDEX', '')
        ]
        
        for key in possible_keys:
            if key in self.whitelists:
                return random_state.choice(self.whitelists[key])
        
        return None


class SequenceSimulator:
    """
    Enhanced sequence simulator with whitelist support.
    
    Generates sequences with defined structure (adapters, UMI, ACC, barcode, indexes, insert)
    with realistic error profiles and support for sequence whitelists.
    """
    
    def __init__(self, config: SimulationConfig, pwm_file: Optional[str] = None):
        """
        Initialize simulator with whitelist support.
        
        Args:
            config: Simulation configuration
            pwm_file: Optional PWM file for ACC generation
        """
        self.config = config
        self.random_state = np.random.RandomState(config.random_seed)
        self.whitelist_loader = WhitelistLoader()
        
        # Load specific whitelists
        if config.i7_whitelist_file:
            self.whitelist_loader.add_whitelist('i7', config.i7_whitelist_file)
            self.whitelist_loader.add_whitelist('INDEX_i7', config.i7_whitelist_file)
        
        if config.i5_whitelist_file:
            self.whitelist_loader.add_whitelist('i5', config.i5_whitelist_file)
            self.whitelist_loader.add_whitelist('INDEX_i5', config.i5_whitelist_file)
        
        if config.cbc_whitelist_file:
            self.whitelist_loader.add_whitelist('CBC', config.cbc_whitelist_file)
            self.whitelist_loader.add_whitelist('BARCODE', config.cbc_whitelist_file)
            self.whitelist_loader.add_whitelist('CELL_BARCODE', config.cbc_whitelist_file)
        
        # Load generic whitelists from mapping
        if config.whitelist_files:
            for segment_name, filepath in config.whitelist_files.items():
                self.whitelist_loader.add_whitelist(segment_name, filepath)
        
        # Load ACC priors if provided
        self.acc_sequences = config.acc_sequences
        self.acc_frequencies = config.acc_frequencies
        
        # Load barcodes if provided (legacy support)
        self.barcodes = None
        if config.barcode_file:
            self.barcodes = self._load_barcodes(config.barcode_file)
        
        # Validate sequence order
        if not config.sequence_order:
            raise ValueError("sequence_order must be specified in config")
        
        logger.info(f"Initialized enhanced simulator with structure: {' → '.join(config.sequence_order)}")
        logger.info(f"Loaded whitelists for: {list(self.whitelist_loader.whitelists.keys())}")
    
    def _load_barcodes(self, filepath: str) -> Optional[List[str]]:
        """Load barcode sequences from file."""
        try:
            barcodes = []
            with open(filepath, 'r') as f:
                for line in f:
                    barcode = line.strip().upper()
                    if barcode:
                        barcodes.append(barcode)
            logger.info(f"Loaded {len(barcodes)} barcodes from {filepath}")
            return barcodes
        except Exception as e:
            logger.warning(f"Could not load barcodes from {filepath}: {e}")
            return None
    
    def generate_random_sequence(self, length: int) -> str:
        """Generate random DNA sequence."""
        bases = ['A', 'C', 'G', 'T']
        return ''.join(self.random_state.choice(bases, size=length))
    
    def generate_umi(self) -> str:
        """Generate UMI sequence (check whitelist first)."""
        # Check whitelist
        seq = self.whitelist_loader.get_sequence('UMI', self.random_state)
        if seq:
            return seq
        
        # Generate random
        return self.generate_random_sequence(self.config.umi_length)
    
    def generate_acc(self) -> str:
        """Generate ACC sequence using priors, PWM, or whitelist."""
        # Check whitelist first
        seq = self.whitelist_loader.get_sequence('ACC', self.random_state)
        if seq:
            return seq
        
        # Use priors if available
        if self.acc_sequences and self.acc_frequencies:
            return self.random_state.choice(self.acc_sequences, p=self.acc_frequencies)
        
        # Fallback: random 6bp sequence
        return self.generate_random_sequence(6)
    
    def generate_index_i7(self) -> str:
        """Generate i7 index sequence."""
        seq = self.whitelist_loader.get_sequence('i7', self.random_state)
        if seq:
            return seq
        
        # Default i7 is typically 8bp
        return self.generate_random_sequence(8)
    
    def generate_index_i5(self) -> str:
        """Generate i5 index sequence."""
        seq = self.whitelist_loader.get_sequence('i5', self.random_state)
        if seq:
            return seq
        
        # Default i5 is typically 8bp
        return self.generate_random_sequence(8)
    
    def generate_cbc(self) -> str:
        """Generate cell barcode (CBC) sequence."""
        seq = self.whitelist_loader.get_sequence('CBC', self.random_state)
        if seq:
            return seq
        
        # Check legacy barcode list
        if self.barcodes:
            return self.random_state.choice(self.barcodes)
        
        # Default CBC is typically 16bp
        return self.generate_random_sequence(16)
    
    def generate_barcode(self) -> str:
        """Generate generic barcode sequence."""
        # Check if CBC whitelist should be used
        seq = self.whitelist_loader.get_sequence('BARCODE', self.random_state)
        if seq:
            return seq
        
        # Check legacy barcode list
        if self.barcodes:
            return self.random_state.choice(self.barcodes)
        
        # Default barcode length
        return self.generate_random_sequence(16)
    
    def generate_insert(self) -> str:
        """Generate random insert sequence with variable length."""
        length = self.random_state.randint(
            self.config.insert_min_length,
            self.config.insert_max_length + 1
        )
        return self.generate_random_sequence(length)
    
    def get_component_sequence(self, component: str) -> str:
        """
        Get sequence for a component, checking whitelists first.
        
        Args:
            component: Component name (e.g., 'ADAPTER5', 'UMI', 'i7', 'CBC')
            
        Returns:
            Sequence string
        """
        # First check whitelist
        seq = self.whitelist_loader.get_sequence(component, self.random_state)
        if seq:
            return seq
        
        # Check if sequence is defined in config
        if self.config.sequences and component in self.config.sequences:
            defined_seq = self.config.sequences[component]
            
            # Handle list of sequences (pick random)
            if isinstance(defined_seq, list):
                return self.random_state.choice(defined_seq)
            
            # Handle 'random' keyword
            if isinstance(defined_seq, str):
                if defined_seq.lower() == 'random':
                    if 'INSERT' in component.upper():
                        return self.generate_insert()
                    else:
                        return self.generate_random_sequence(10)
                else:
                    return defined_seq
        
        # Component-specific generation with whitelist support
        component_upper = component.upper()
        
        if 'I7' in component_upper or 'INDEX_I7' in component_upper:
            return self.generate_index_i7()
        elif 'I5' in component_upper or 'INDEX_I5' in component_upper:
            return self.generate_index_i5()
        elif 'CBC' in component_upper or 'CELL_BARCODE' in component_upper:
            return self.generate_cbc()
        elif 'UMI' in component_upper:
            return self.generate_umi()
        elif 'ACC' in component_upper:
            return self.generate_acc()
        elif 'BARCODE' in component_upper or 'BC' in component_upper:
            return self.generate_barcode()
        elif 'INSERT' in component_upper:
            return self.generate_insert()
        elif 'ADAPTER' in component_upper:
            # Default adapter sequences
            if '5' in component or 'FIVE' in component_upper:
                return 'AGATCGGAAGAGC'
            elif '3' in component or 'THREE' in component_upper:
                return 'AGATCGGAAGAGC'
            else:
                return 'AGATCGGAAGAGC'
        else:
            # Unknown component
            logger.warning(f"Unknown component '{component}', generating 10bp random sequence")
            return self.generate_random_sequence(10)
    
    def inject_errors(self, sequence: str) -> str:
        """
        Inject substitution, insertion, and deletion errors.
        
        Args:
            sequence: Original sequence
            
        Returns:
            Sequence with errors
        """
        if self.config.error_rate == 0:
            return sequence
        
        sequence = list(sequence)
        bases = ['A', 'C', 'G', 'T']
        
        for i in range(len(sequence)):
            if self.random_state.random() < self.config.error_rate:
                error_type = self.random_state.choice(['substitution', 'insertion', 'deletion'])
                
                if error_type == 'substitution':
                    # Replace with different base
                    current = sequence[i]
                    choices = [b for b in bases if b != current]
                    sequence[i] = self.random_state.choice(choices)
                
                elif error_type == 'insertion' and i < len(sequence) - 1:
                    # Insert random base
                    sequence.insert(i + 1, self.random_state.choice(bases))
                
                elif error_type == 'deletion' and len(sequence) > 1:
                    # Delete current base
                    del sequence[i]
                    if i >= len(sequence):
                        break
        
        return ''.join(sequence)
    
    def generate_read(self) -> SimulatedRead:
        """
        Generate a single simulated read with labels.
        
        Returns:
            SimulatedRead object
        """
        components = []
        component_labels = []
        component_sources = []  # Track where each sequence came from
        
        # Generate each component
        for component_name in self.config.sequence_order:
            seq = self.get_component_sequence(component_name)
            components.append(seq)
            component_labels.append(component_name)
            
            # Track if sequence came from whitelist
            if self.whitelist_loader.get_sequence(component_name, self.random_state):
                component_sources.append('whitelist')
            elif self.config.sequences and component_name in self.config.sequences:
                component_sources.append('config')
            else:
                component_sources.append('generated')
        
        # Concatenate to form full sequence
        full_sequence = ''.join(components)
        original_length = len(full_sequence)
        
        # Create labels (one per base)
        labels = []
        label_regions = {}
        current_pos = 0
        
        for component_seq, label in zip(components, component_labels):
            component_len = len(component_seq)
            labels.extend([label] * component_len)
            
            # Track regions
            if label not in label_regions:
                label_regions[label] = []
            label_regions[label].append((current_pos, current_pos + component_len))
            
            current_pos += component_len
        
        # Inject errors
        if self.config.error_rate > 0:
            full_sequence = self.inject_errors(full_sequence)
            
            # Adjust labels if length changed due to indels
            if len(full_sequence) != len(labels):
                if len(full_sequence) > len(labels):
                    # Insertions occurred
                    labels.extend(['UNKNOWN'] * (len(full_sequence) - len(labels)))
                else:
                    # Deletions occurred
                    labels = labels[:len(full_sequence)]
        
        # Create metadata
        metadata = {
            'num_components': len(components),
            'original_length': original_length,
            'final_length': len(full_sequence),
            'error_rate': self.config.error_rate,
            'component_sources': dict(zip(component_labels, component_sources)),
            'has_errors': original_length != len(full_sequence)
        }
        
        return SimulatedRead(
            sequence=full_sequence,
            labels=labels,
            label_regions=label_regions,
            metadata=metadata
        )
    
    def generate(self, num_sequences: Optional[int] = None) -> Tuple[List[SimulatedRead], List[SimulatedRead]]:
        """
        Generate multiple simulated reads.
        
        Args:
            num_sequences: Number of sequences to generate (overrides config)
            
        Returns:
            Tuple of (training_reads, validation_reads) with 80/20 split
        """
        n = num_sequences or self.config.num_sequences
        
        reads = []
        for i in range(n):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Generated {i}/{n} sequences...")
            reads.append(self.generate_read())
        
        logger.info(f"Generated {len(reads)} total sequences")
        
        # Split into train/val
        split_idx = int(len(reads) * 0.8)
        train_reads = reads[:split_idx]
        val_reads = reads[split_idx:]
        
        logger.info(f"Split into {len(train_reads)} training and {len(val_reads)} validation sequences")
        
        return train_reads, val_reads
    
    def get_statistics(self, reads: List[SimulatedRead]) -> Dict:
        """
        Compute statistics about generated reads.
        
        Args:
            reads: List of simulated reads
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_reads': len(reads),
            'sequence_lengths': [len(r.sequence) for r in reads],
            'component_counts': {},
            'whitelist_usage': {},
            'error_injection': {
                'num_with_errors': sum(1 for r in reads if r.metadata.get('has_errors', False)),
                'percent_with_errors': 0
            }
        }
        
        # Count component occurrences and sources
        for read in reads:
            for component in read.metadata.get('component_sources', {}).keys():
                if component not in stats['component_counts']:
                    stats['component_counts'][component] = 0
                stats['component_counts'][component] += 1
            
            # Track whitelist usage
            for component, source in read.metadata.get('component_sources', {}).items():
                if component not in stats['whitelist_usage']:
                    stats['whitelist_usage'][component] = {'whitelist': 0, 'config': 0, 'generated': 0}
                stats['whitelist_usage'][component][source] += 1
        
        # Compute percentages
        if stats['num_reads'] > 0:
            stats['error_injection']['percent_with_errors'] = (
                100 * stats['error_injection']['num_with_errors'] / stats['num_reads']
            )
        
        # Sequence length statistics
        if stats['sequence_lengths']:
            stats['avg_length'] = np.mean(stats['sequence_lengths'])
            stats['min_length'] = min(stats['sequence_lengths'])
            stats['max_length'] = max(stats['sequence_lengths'])
        
        return stats
