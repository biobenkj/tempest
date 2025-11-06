"""
Data Simulator for Tempest.

Combines all features:
- Flexible sequence architecture (any segment names)
- Whitelist support for any segment
- Transcript FASTA support for cDNA inserts
- Variable polyA tail generation
- PWM-based sequence generation
- Comprehensive error simulation
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRead:
    """Container for a simulated read with labels."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict = field(default_factory=dict)  # Additional info


class TranscriptPool:
    """Manages a pool of transcript sequences for cDNA simulation."""
    
    def __init__(self, config: Dict):
        """
        Initialize transcript pool from configuration.
        
        Args:
            config: Transcript configuration dictionary
        """
        self.transcripts = []
        self.transcript_ids = []
        self.config = config
        
        fasta_file = config.get('fasta_file')
        if fasta_file and Path(fasta_file).exists():
            self._load_transcripts(fasta_file)
        else:
            logger.warning(f"Transcript file not found: {fasta_file}")
    
    def _load_transcripts(self, fasta_file: str):
        """Load transcripts from FASTA file."""
        try:
            # Try with Biopython if available
            try:
                from Bio import SeqIO
                for record in SeqIO.parse(fasta_file, "fasta"):
                    seq_len = len(record.seq)
                    min_len = self.config.get('min_length', 100)
                    max_len = self.config.get('max_length', 5000)
                    
                    if min_len <= seq_len <= max_len:
                        self.transcripts.append(str(record.seq).upper())
                        self.transcript_ids.append(record.id)
            except ImportError:
                # Fallback to simple FASTA parsing
                logger.info("Biopython not available, using simple FASTA parser")
                self._simple_fasta_parse(fasta_file)
            
            if self.transcripts:
                logger.info(f"Loaded {len(self.transcripts)} transcripts")
                lengths = [len(t) for t in self.transcripts]
                logger.info(f"Length range: {min(lengths)}-{max(lengths)} bp")
        
        except Exception as e:
            logger.error(f"Failed to load transcripts: {e}")
    
    def _simple_fasta_parse(self, fasta_file: str):
        """Simple FASTA parser without dependencies."""
        with open(fasta_file, 'r') as f:
            seq_id = None
            seq_parts = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if seq_id and seq_parts:
                        seq = ''.join(seq_parts).upper()
                        seq_len = len(seq)
                        min_len = self.config.get('min_length', 100)
                        max_len = self.config.get('max_length', 5000)
                        
                        if min_len <= seq_len <= max_len:
                            self.transcripts.append(seq)
                            self.transcript_ids.append(seq_id)
                    
                    # Start new sequence
                    seq_id = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            
            # Don't forget last sequence
            if seq_id and seq_parts:
                seq = ''.join(seq_parts).upper()
                seq_len = len(seq)
                min_len = self.config.get('min_length', 100)
                max_len = self.config.get('max_length', 5000)
                
                if min_len <= seq_len <= max_len:
                    self.transcripts.append(seq)
                    self.transcript_ids.append(seq_id)
    
    def sample_fragment(self, random_state: np.random.RandomState) -> str:
        """
        Sample a fragment from a random transcript.
        
        Args:
            random_state: Random state for reproducibility
            
        Returns:
            Fragment sequence
        """
        if not self.transcripts:
            # Fallback to random generation
            fallback_mode = self.config.get('fallback_mode', 'random')
            if fallback_mode == 'random':
                length = random_state.randint(
                    self.config.get('fragment_min', 200),
                    self.config.get('fragment_max', 800)
                )
                return self._generate_random_sequence(length, random_state)
            else:
                raise ValueError("No transcripts loaded and no fallback configured")
        
        # Select random transcript
        transcript = random_state.choice(self.transcripts)
        
        # Determine fragment length
        fragment_min = self.config.get('fragment_min', 200)
        fragment_max = self.config.get('fragment_max', 800)
        max_possible = min(len(transcript), fragment_max)
        
        if max_possible < fragment_min:
            # Use full transcript if too short
            fragment = transcript
        else:
            fragment_len = random_state.randint(fragment_min, max_possible + 1)
            max_start = len(transcript) - fragment_len
            start_pos = random_state.randint(0, max_start + 1) if max_start > 0 else 0
            fragment = transcript[start_pos:start_pos + fragment_len]
        
        # Reverse complement with probability
        if random_state.random() < self.config.get('reverse_complement_prob', 0.5):
            fragment = self._reverse_complement(fragment)
        
        return fragment
    
    def _generate_random_sequence(self, length: int, random_state: np.random.RandomState) -> str:
        """Generate random sequence with specified GC content."""
        gc_content = self.config.get('fallback_gc_content', 0.5)
        
        bases = []
        for _ in range(length):
            if random_state.random() < gc_content:
                bases.append(random_state.choice(['G', 'C']))
            else:
                bases.append(random_state.choice(['A', 'T']))
        
        return ''.join(bases)
    
    def _reverse_complement(self, seq: str) -> str:
        """Generate reverse complement of sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(b, 'N') for b in seq[::-1])


class PolyATailGenerator:
    """Generates polyA tails with realistic length distributions."""
    
    def __init__(self, config: Dict):
        """
        Initialize polyA tail generator.
        
        Args:
            config: PolyA configuration dictionary
        """
        self.config = config
        self.distribution = config.get('distribution', 'normal')
        self.empirical_lengths = None
        
        # Load empirical distribution if provided
        empirical_file = config.get('empirical_file')
        if empirical_file and Path(empirical_file).exists():
            self._load_empirical_distribution(empirical_file)
    
    def _load_empirical_distribution(self, filepath: str):
        """Load empirical polyA length distribution."""
        try:
            lengths = []
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        length = int(line.strip())
                        lengths.append(length)
                    except ValueError:
                        continue
            
            if lengths:
                self.empirical_lengths = np.array(lengths)
                logger.info(f"Loaded {len(lengths)} empirical polyA lengths")
        except Exception as e:
            logger.warning(f"Could not load empirical distribution: {e}")
    
    def generate(self, random_state: np.random.RandomState) -> str:
        """Generate a polyA tail sequence."""
        # Determine length based on distribution
        if self.distribution == 'empirical' and self.empirical_lengths is not None:
            length = random_state.choice(self.empirical_lengths)
        elif self.distribution == 'normal':
            mean = self.config.get('mean_length', 150)
            std = self.config.get('std_length', 50)
            length = int(random_state.normal(mean, std))
        elif self.distribution == 'exponential':
            lambda_param = self.config.get('lambda_param', 0.01)
            length = int(random_state.exponential(1.0 / lambda_param))
        elif self.distribution == 'uniform':
            min_len = self.config.get('min_length', 10)
            max_len = self.config.get('max_length', 300)
            length = random_state.randint(min_len, max_len + 1)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # Clip to bounds
        min_len = self.config.get('min_length', 10)
        max_len = self.config.get('max_length', 300)
        length = max(min_len, min(max_len, length))
        
        # Generate polyA with occasional interruptions
        purity = self.config.get('purity', 0.95)
        interruption_bases = self.config.get('interruption_bases', ['T', 'G', 'C'])
        
        bases = []
        for _ in range(length):
            if random_state.random() < purity:
                bases.append('A')
            else:
                bases.append(random_state.choice(interruption_bases))
        
        return ''.join(bases)


class WhitelistManager:
    """Manages sequence whitelists for various segments."""
    
    def __init__(self):
        self.whitelists = {}
        self.whitelist_ids = {}  # Store index IDs when available
        self.usage_stats = {}
    
    def load_whitelist(self, segment_name: str, filepath: str) -> bool:
        """
        Load sequences from a whitelist file.
        Supports both single-column (sequence only) and two-column (index_id, sequence) formats.
        
        Args:
            segment_name: Name of the segment
            filepath: Path to whitelist file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.debug(f"Whitelist file not found for {segment_name}: {filepath}")
                return False
            
            sequences = []
            index_ids = []  # Store index IDs if available
            
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    
                    # Try to parse as two-column format (index_id, sequence)
                    parts = line.split('\t')  # Tab-separated
                    if len(parts) == 1:
                        # Try space-separated if no tabs
                        parts = line.split()
                    
                    if len(parts) == 2:
                        # Two-column format: index_id, sequence
                        index_id, seq = parts
                        seq = seq.upper()
                        # Validate DNA sequence
                        if seq and all(base in 'ACGTN' for base in seq):
                            sequences.append(seq)
                            index_ids.append(index_id)
                        else:
                            logger.warning(f"Invalid sequence at line {line_num} in {filepath}: {seq}")
                    elif len(parts) == 1:
                        # Single-column format: sequence only
                        seq = parts[0].upper()
                        # Validate DNA sequence
                        if seq and all(base in 'ACGTN' for base in seq):
                            sequences.append(seq)
                        else:
                            logger.warning(f"Invalid sequence at line {line_num} in {filepath}: {seq}")
                    else:
                        logger.warning(f"Invalid format at line {line_num} in {filepath}: {line}")
            
            if sequences:
                self.whitelists[segment_name] = sequences
                # Store index IDs if available (for future reference/logging)
                if index_ids:
                    self.whitelist_ids[segment_name] = index_ids
                    logger.info(f"Loaded {len(sequences)} sequences with IDs for {segment_name}")
                else:
                    logger.info(f"Loaded {len(sequences)} sequences for {segment_name}")
                self.usage_stats[segment_name] = {'loaded': len(sequences), 'used': 0}
                return True
            
        except Exception as e:
            logger.error(f"Error loading whitelist for {segment_name}: {e}")
        
        return False
    
    def get_sequence(self, segment_name: str, random_state: np.random.RandomState) -> Optional[str]:
        """
        Get a random sequence from whitelist.
        
        Args:
            segment_name: Name of the segment
            random_state: Random state for reproducibility
            
        Returns:
            Random sequence from whitelist or None
        """
        if segment_name in self.whitelists:
            seq = random_state.choice(self.whitelists[segment_name])
            self.usage_stats[segment_name]['used'] += 1
            return seq
        return None
    
    def get_stats(self) -> Dict:
        """Get whitelist usage statistics."""
        return self.usage_stats.copy()


class SequenceSimulator:
    """
    Sequence simulator with all features:
    - Flexible architecture
    - Whitelist support
    - Transcript pool
    - PolyA generation
    - PWM support
    """
    
    def __init__(self, config_file: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize simulator from config file or dictionary.
        
        Args:
            config_file: Path to YAML configuration file
            config: Configuration dictionary (if not using file)
        """
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_file or config must be provided")
        
        # Get simulation config
        self.sim_config = self.config.get('simulation', {})
        
        # Initialize random state
        seed = self.sim_config.get('random_seed', 42)
        self.random_state = np.random.RandomState(seed)
        
        # Get sequence order (architecture)
        self.sequence_order = self.sim_config.get('sequence_order', [])
        if not self.sequence_order:
            raise ValueError("sequence_order must be defined in configuration")
        
        logger.info(f"Sequence architecture: {' → '.join(self.sequence_order)}")
        
        # Initialize components
        self._init_whitelist_manager()
        self._init_transcript_pool()
        self._init_polya_generator()
        self._init_segment_generators()
    
    def _init_whitelist_manager(self):
        """Initialize whitelist manager."""
        self.whitelist_manager = WhitelistManager()
        
        # Load whitelists from config
        whitelist_files = self.sim_config.get('whitelist_files', {})
        for segment, filepath in whitelist_files.items():
            self.whitelist_manager.load_whitelist(segment, filepath)
    
    def _init_transcript_pool(self):
        """Initialize transcript pool if configured."""
        self.transcript_pool = None
        
        transcript_config = self.sim_config.get('transcript', {})
        if transcript_config.get('fasta_file'):
            self.transcript_pool = TranscriptPool(transcript_config)
    
    def _init_polya_generator(self):
        """Initialize polyA generator."""
        self.polya_generator = None
        
        polya_config = self.sim_config.get('polya_tail', {})
        if polya_config:
            self.polya_generator = PolyATailGenerator(polya_config)
    
    def _init_segment_generators(self):
        """Initialize segment-specific generation parameters."""
        self.segment_generation = self.sim_config.get('segment_generation', {})
        self.fixed_sequences = self.sim_config.get('sequences', {})
        self.fallback_sequences = self.sim_config.get('fallback_sequences', {})
    
    def generate_segment(self, segment_name: str) -> str:
        """
        Generate sequence for a segment using priority system:
        1. Fixed sequences
        2. Whitelist
        3. Fallback sequences
        4. Generated (based on type)
        
        Args:
            segment_name: Name of the segment
            
        Returns:
            Generated sequence
        """
        # 1. Check fixed sequences
        if segment_name in self.fixed_sequences:
            fixed_seq = self.fixed_sequences[segment_name]
            
            # Handle special keywords
            if isinstance(fixed_seq, str):
                if fixed_seq.lower() == 'transcript':
                    return self._generate_transcript()
                elif fixed_seq.lower() == 'polya':
                    return self._generate_polya()
                elif fixed_seq.lower() == 'random':
                    return self._generate_random_segment(segment_name)
                else:
                    return fixed_seq
        
        # 2. Check whitelist
        whitelist_seq = self.whitelist_manager.get_sequence(segment_name, self.random_state)
        if whitelist_seq:
            return whitelist_seq
        
        # 3. Check fallback sequences
        if segment_name in self.fallback_sequences:
            fallback = self.fallback_sequences[segment_name]
            if isinstance(fallback, list):
                return self.random_state.choice(fallback)
            else:
                return fallback
        
        # 4. Generate based on segment type
        return self._generate_by_type(segment_name)
    
    def _generate_transcript(self) -> str:
        """Generate cDNA insert from transcript pool."""
        if self.transcript_pool:
            return self.transcript_pool.sample_fragment(self.random_state)
        else:
            # Fallback to random
            length = self.random_state.randint(200, 800)
            return self._generate_random_dna(length)
    
    def _generate_polya(self) -> str:
        """Generate polyA tail."""
        if self.polya_generator:
            return self.polya_generator.generate(self.random_state)
        else:
            # Simple fallback
            length = self.random_state.randint(10, 200)
            return 'A' * length
    
    def _generate_random_segment(self, segment_name: str) -> str:
        """Generate random segment based on configured length."""
        lengths = self.segment_generation.get('lengths', {})
        
        # Determine length
        if segment_name in lengths:
            length = lengths[segment_name]
        else:
            length = lengths.get('DEFAULT', 20)
        
        return self._generate_random_dna(length)
    
    def _generate_by_type(self, segment_name: str) -> str:
        """Generate segment based on its type/name."""
        segment_upper = segment_name.upper()
        
        # Check generation mode
        gen_modes = self.segment_generation.get('generation_mode', {})
        mode = gen_modes.get(segment_name, gen_modes.get('DEFAULT', 'random'))
        
        if mode == 'transcript':
            return self._generate_transcript()
        elif mode == 'polya':
            return self._generate_polya()
        elif mode == 'pwm':
            return self._generate_pwm_sequence()
        
        # Type-based generation from name patterns
        if 'INSERT' in segment_upper or 'CDNA' in segment_upper:
            return self._generate_transcript()
        elif 'POLYA' in segment_upper or 'POLY_A' in segment_upper:
            return self._generate_polya()
        elif 'UMI' in segment_upper:
            length = self.segment_generation.get('lengths', {}).get('UMI', 12)
            return self._generate_random_dna(length)
        elif 'ACC' in segment_upper:
            return self._generate_acc()
        elif 'BARCODE' in segment_upper or 'BC' in segment_upper or 'CBC' in segment_upper:
            length = self.segment_generation.get('lengths', {}).get(segment_name, 16)
            return self._generate_random_dna(length)
        elif 'INDEX' in segment_upper:
            return self._generate_random_dna(8)
        elif 'ADAPTER' in segment_upper:
            return self._generate_default_adapter(segment_name)
        else:
            # Unknown segment - use configured or default length
            return self._generate_random_segment(segment_name)
    
    def _generate_random_dna(self, length: int) -> str:
        """Generate random DNA sequence."""
        bases = ['A', 'C', 'G', 'T']
        return ''.join(self.random_state.choice(bases, size=length))
    
    def _generate_acc(self) -> str:
        """Generate ACC sequence (simplified for this example)."""
        # Check if ACC sequences are configured
        pwm_config = self.sim_config.get('pwm', {})
        acc_sequences = pwm_config.get('acc_sequences')
        
        if acc_sequences:
            acc_frequencies = pwm_config.get('acc_frequencies')
            if acc_frequencies:
                return self.random_state.choice(acc_sequences, p=acc_frequencies)
            else:
                return self.random_state.choice(acc_sequences)
        
        # Default ACC sequences
        return self.random_state.choice(['GGGGGG', 'AAAAAA', 'CCCCCC', 'TTTTTT'])
    
    def _generate_pwm_sequence(self) -> str:
        """Generate sequence based on PWM (placeholder)."""
        # This would use actual PWM scoring if implemented
        return self._generate_acc()
    
    def _generate_default_adapter(self, segment_name: str) -> str:
        """Generate default adapter sequences."""
        if '5' in segment_name or 'FIVE' in segment_name.upper():
            return 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA'
        elif '3' in segment_name or 'THREE' in segment_name.upper():
            return 'AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT'
        else:
            return 'AGATCGGAAGAGC'
    
    def inject_errors(self, sequence: str) -> Tuple[str, List[str]]:
        """
        Inject sequencing errors into sequence.
        
        Args:
            sequence: Original sequence
            
        Returns:
            Tuple of (modified sequence, error types for each position)
        """
        error_config = self.sim_config.get('error_profile', {})
        overall_rate = error_config.get('error_rate', 0.02)
        
        if overall_rate <= 0:
            return sequence, ['none'] * len(sequence)
        
        # Detailed error rates
        sub_rate = error_config.get('substitution', {}).get('rate', overall_rate * 0.7)
        ins_rate = error_config.get('insertion', {}).get('rate', overall_rate * 0.15)
        del_rate = error_config.get('deletion', {}).get('rate', overall_rate * 0.15)
        
        result = []
        error_types = []
        bases = ['A', 'C', 'G', 'T']
        
        i = 0
        while i < len(sequence):
            rand_val = self.random_state.random()
            
            if rand_val < sub_rate:
                # Substitution
                current = sequence[i]
                sub_matrix = error_config.get('substitution', {}).get('matrix', {})
                
                if current in sub_matrix:
                    # Use transition matrix
                    targets = list(sub_matrix[current].keys())
                    probs = list(sub_matrix[current].values())
                    new_base = self.random_state.choice(targets, p=probs)
                else:
                    # Random substitution
                    choices = [b for b in bases if b != current]
                    new_base = self.random_state.choice(choices)
                
                result.append(new_base)
                error_types.append('substitution')
                i += 1
                
            elif rand_val < sub_rate + ins_rate:
                # Insertion
                max_ins = error_config.get('insertion', {}).get('max_length', 2)
                ins_len = self.random_state.randint(1, max_ins + 1)
                
                result.append(sequence[i])
                error_types.append('none')
                
                for _ in range(ins_len):
                    result.append(self.random_state.choice(bases))
                    error_types.append('insertion')
                i += 1
                
            elif rand_val < sub_rate + ins_rate + del_rate:
                # Deletion
                max_del = error_config.get('deletion', {}).get('max_length', 2)
                del_len = min(self.random_state.randint(1, max_del + 1), len(sequence) - i)
                
                # Skip bases
                i += del_len
                
            else:
                # No error
                result.append(sequence[i])
                error_types.append('none')
                i += 1
        
        return ''.join(result), error_types
    
    def generate_read(self) -> SimulatedRead:
        """
        Generate a single simulated read.
        
        Returns:
            SimulatedRead object
        """
        segments = []
        segment_names = []
        segment_sources = []
        
        # Generate each segment
        for segment_name in self.sequence_order:
            seq = self.generate_segment(segment_name)
            segments.append(seq)
            segment_names.append(segment_name)
            
            # Track source
            if segment_name in self.fixed_sequences:
                segment_sources.append('fixed')
            elif self.whitelist_manager.get_sequence(segment_name, self.random_state):
                segment_sources.append('whitelist')
            elif segment_name in self.fallback_sequences:
                segment_sources.append('fallback')
            else:
                segment_sources.append('generated')
        
        # Concatenate
        full_sequence = ''.join(segments)
        original_sequence = full_sequence
        
        # Create labels
        labels = []
        label_regions = {}
        current_pos = 0
        
        for seg_seq, seg_name in zip(segments, segment_names):
            seg_len = len(seg_seq)
            labels.extend([seg_name] * seg_len)
            
            if seg_name not in label_regions:
                label_regions[seg_name] = []
            label_regions[seg_name].append((current_pos, current_pos + seg_len))
            
            current_pos += seg_len
        
        # Inject errors
        error_config = self.sim_config.get('error_profile', {})
        if error_config.get('error_rate', 0) > 0:
            full_sequence, error_types = self.inject_errors(full_sequence)
            
            # Adjust labels for length changes
            if len(full_sequence) != len(labels):
                if len(full_sequence) > len(labels):
                    labels.extend(['UNKNOWN'] * (len(full_sequence) - len(labels)))
                else:
                    labels = labels[:len(full_sequence)]
        else:
            error_types = ['none'] * len(full_sequence)
        
        # Create metadata
        metadata = {
            'architecture': self.sequence_order,
            'num_segments': len(segments),
            'segment_sources': dict(zip(segment_names, segment_sources)),
            'original_length': len(original_sequence),
            'final_length': len(full_sequence),
            'has_errors': len(original_sequence) != len(full_sequence),
            'error_rate': error_config.get('error_rate', 0),
        }
        
        # Add segment-specific metadata
        for seg_name in segment_names:
            if seg_name in label_regions:
                regions = label_regions[seg_name]
                if regions:
                    metadata[f'{seg_name}_length'] = sum(end - start for start, end in regions)
        
        return SimulatedRead(
            sequence=full_sequence,
            labels=labels,
            label_regions=label_regions,
            metadata=metadata
        )
    
    def generate_batch(self, num_sequences: int) -> List[SimulatedRead]:
        """
        Generate multiple simulated reads.
        
        Args:
            num_sequences: Number of sequences to generate
            
        Returns:
            List of SimulatedRead objects
        """
        reads = []
        
        for i in range(num_sequences):
            reads.append(self.generate_read())
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_sequences} reads")
        
        logger.info(f"Generated {len(reads)} reads successfully")
        
        # Log statistics
        self._log_generation_stats(reads)
        
        return reads
    
    def generate_train_val_split(self) -> Tuple[List[SimulatedRead], List[SimulatedRead]]:
        """
        Generate training and validation datasets.
        
        Returns:
            Tuple of (training_reads, validation_reads)
        """
        total_sequences = self.sim_config.get('num_sequences', 10000)
        train_split = self.sim_config.get('train_split', 0.8)
        
        num_train = int(total_sequences * train_split)
        num_val = total_sequences - num_train
        
        logger.info(f"Generating {num_train} training and {num_val} validation reads")
        
        train_reads = self.generate_batch(num_train)
        val_reads = self.generate_batch(num_val)
        
        return train_reads, val_reads
    
    def _log_generation_stats(self, reads: List[SimulatedRead]):
        """Log statistics about generated reads."""
        if not self.config.get('logging', {}).get('log_generation_stats', True):
            return
        
        stats = {
            'num_reads': len(reads),
            'architecture': self.sequence_order,
            'length_distribution': {},
            'segment_sources': {},
            'error_injection': {'with_errors': 0, 'without_errors': 0},
        }
        
        # Analyze reads
        for read in reads:
            # Length stats
            length = len(read.sequence)
            length_bin = f"{(length // 100) * 100}-{(length // 100 + 1) * 100}"
            stats['length_distribution'][length_bin] = stats['length_distribution'].get(length_bin, 0) + 1
            
            # Source stats
            for seg, source in read.metadata.get('segment_sources', {}).items():
                if seg not in stats['segment_sources']:
                    stats['segment_sources'][seg] = {}
                stats['segment_sources'][seg][source] = stats['segment_sources'][seg].get(source, 0) + 1
            
            # Error stats
            if read.metadata.get('has_errors', False):
                stats['error_injection']['with_errors'] += 1
            else:
                stats['error_injection']['without_errors'] += 1
        
        # Log whitelist usage
        if self.config.get('logging', {}).get('log_whitelist_usage', True):
            stats['whitelist_usage'] = self.whitelist_manager.get_stats()
        
        logger.info("Generation Statistics:")
        logger.info(f"  Total reads: {stats['num_reads']}")
        logger.info(f"  Architecture: {' → '.join(stats['architecture'])}")
        logger.info(f"  Length distribution: {stats['length_distribution']}")
        logger.info(f"  Reads with errors: {stats['error_injection']['with_errors']}")
        
        if 'whitelist_usage' in stats:
            logger.info("  Whitelist usage:")
            for seg, usage in stats['whitelist_usage'].items():
                logger.info(f"    {seg}: loaded={usage['loaded']}, used={usage['used']}")


def create_simulator_from_config(config_file: str) -> SequenceSimulator:
    """
    Convenience function to create simulator from config file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Configured simulator instance
    """
    return SequenceSimulator(config_file=config_file)


def reads_to_arrays(
    reads: List[SimulatedRead],
    label_to_idx: Optional[Dict[str, int]] = None,
    max_len: Optional[int] = None,
    padding_value: int = 4
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert reads to numpy arrays for model training.
    
    Args:
        reads: List of SimulatedRead objects
        label_to_idx: Optional pre-existing label mapping. If None, creates one.
        max_len: Maximum sequence length (pad/truncate to this)
        padding_value: Value to use for padding (4 = N)
        
    Returns:
        X: Encoded sequences (num_reads, max_len)
        y: Labels (num_reads, max_len)
        label_to_idx: Mapping of labels to indices
    """
    # Determine max length
    if max_len is None:
        max_len = max(len(read.sequence) for read in reads)
    
    # Create label mapping if not provided
    if label_to_idx is None:
        all_labels = set()
        for read in reads:
            all_labels.update(read.labels)
        
        # Add special labels
        all_labels.add('PAD')
        all_labels.add('UNKNOWN')
        
        label_to_idx = {label: i for i, label in enumerate(sorted(all_labels))}
    
    # Base encoding
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Initialize arrays
    num_reads = len(reads)
    X = np.full((num_reads, max_len), padding_value, dtype=np.int8)
    y = np.full((num_reads, max_len), label_to_idx['PAD'], dtype=np.int8)
    
    # Fill arrays
    for i, read in enumerate(reads):
        seq_len = min(len(read.sequence), max_len)
        
        # Encode sequence
        for j in range(seq_len):
            base = read.sequence[j]
            X[i, j] = base_to_idx.get(base, 4)
            y[i, j] = label_to_idx.get(read.labels[j], label_to_idx['UNKNOWN'])
    
    return X, y, label_to_idx


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequence Simulator")
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output', help='Output file for generated reads')
    parser.add_argument('--format', choices=['json', 'pickle', 'tsv'], default='json',
                      help='Output format')
    parser.add_argument('--num-sequences', type=int, help='Override number of sequences')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create simulator
    simulator = SequenceSimulator(config_file=args.config)
    
    # Override number of sequences if specified
    if args.num_sequences:
        simulator.sim_config['num_sequences'] = args.num_sequences
    
    # Generate reads
    logger.info("Starting sequence generation...")
    train_reads, val_reads = simulator.generate_train_val_split()
    
    # Save if output specified
    if args.output:
        import json
        import pickle
        
        output_data = {
            'train': [
                {
                    'sequence': read.sequence,
                    'labels': read.labels,
                    'label_regions': read.label_regions,
                    'metadata': read.metadata
                }
                for read in train_reads
            ],
            'validation': [
                {
                    'sequence': read.sequence,
                    'labels': read.labels,
                    'label_regions': read.label_regions,
                    'metadata': read.metadata
                }
                for read in val_reads
            ]
        }
        
        if args.format == 'json':
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        elif args.format == 'pickle':
            with open(args.output, 'wb') as f:
                pickle.dump({'train': train_reads, 'validation': val_reads}, f)
        elif args.format == 'tsv':
            with open(args.output, 'w') as f:
                f.write("split\tsequence\tlabels\n")
                for read in train_reads:
                    f.write(f"train\t{read.sequence}\t{','.join(read.labels)}\n")
                for read in val_reads:
                    f.write(f"validation\t{read.sequence}\t{','.join(read.labels)}\n")
        
        logger.info(f"Saved results to {args.output}")
    
    logger.info("Generation complete!")
