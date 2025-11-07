"""
Data Simulator for Tempest.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
import sys

# Conditional import for PWM generator
try:
    from ..core.pwm_probabilistic import ProbabilisticPWMGenerator, create_acc_pwm_from_pattern
    PWM_AVAILABLE = True
except ImportError:
    PWM_AVAILABLE = False
    logging.warning("ProbabilisticPWMGenerator not available, ACC generation will use fallback methods")

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRead:
    """Container for a simulated read with labels and metadata."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict = field(default_factory=dict)  # Additional info
    quality_scores: Optional[np.ndarray] = None  # Optional quality scores


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
            logger.error(f"Failed to load polyA distribution: {e}")
    
    def generate(self, random_state: np.random.RandomState) -> str:
        """
        Generate a polyA tail.
        
        Args:
            random_state: Random state for reproducibility
            
        Returns:
            PolyA tail sequence
        """
        # Determine length
        if self.empirical_lengths is not None:
            # Sample from empirical distribution
            length = random_state.choice(self.empirical_lengths)
        else:
            # Use parametric distribution
            if self.distribution == 'normal':
                mean = self.config.get('mean_length', 150)
                std = self.config.get('std_length', 50)
                length = int(random_state.normal(mean, std))
            elif self.distribution == 'gamma':
                shape = self.config.get('shape', 2.0)
                scale = self.config.get('scale', 75.0)
                length = int(random_state.gamma(shape, scale))
            elif self.distribution == 'uniform':
                min_len = self.config.get('min_length', 50)
                max_len = self.config.get('max_length', 250)
                length = random_state.randint(min_len, max_len + 1)
            else:
                # Fixed length
                length = self.config.get('fixed_length', 100)
        
        # Ensure reasonable bounds
        min_len = self.config.get('absolute_min', 20)
        max_len = self.config.get('absolute_max', 500)
        length = max(min_len, min(length, max_len))
        
        # Generate tail with optional impurities
        purity = self.config.get('purity', 0.95)
        tail = []
        for _ in range(length):
            if random_state.random() < purity:
                tail.append('A')
            else:
                # Occasional non-A base
                tail.append(random_state.choice(['C', 'G', 'T']))
        
        return ''.join(tail)


class WhitelistManager:
    """Manages whitelisted sequences for different segments."""
    
    def __init__(self, config: Dict):
        """
        Initialize whitelist manager.
        
        Args:
            config: Whitelist configuration dictionary
        """
        self.whitelists = {}
        self.config = config
        self.usage_stats = {}  # Track usage for statistics
        
        # Load whitelists
        whitelist_files = config.get('whitelist_files', {})
        for segment, filepath in whitelist_files.items():
            self._load_whitelist(segment, filepath)
    
    def _load_whitelist(self, segment: str, filepath: str):
        """Load a whitelist from file."""
        if not Path(filepath).exists():
            logger.warning(f"Whitelist file not found for {segment}: {filepath}")
            return
        
        sequences = []
        with open(filepath, 'r') as f:
            for line in f:
                seq = line.strip().upper()
                if seq and not seq.startswith('#'):
                    sequences.append(seq)
        
        if sequences:
            self.whitelists[segment] = sequences
            self.usage_stats[segment] = {'loaded': len(sequences), 'used': 0}
            logger.info(f"Loaded {len(sequences)} whitelisted sequences for {segment}")
    
    def has_whitelist(self, segment: str) -> bool:
        """Check if a segment has a whitelist."""
        return segment in self.whitelists
    
    def sample(self, segment: str, random_state: np.random.RandomState) -> Optional[str]:
        """
        Sample a sequence from a whitelist.
        
        Args:
            segment: Segment name
            random_state: Random state for reproducibility
            
        Returns:
            Sampled sequence or None if no whitelist
        """
        if segment not in self.whitelists:
            return None
        
        seq = random_state.choice(self.whitelists[segment])
        self.usage_stats[segment]['used'] += 1
        return seq
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return self.usage_stats.copy()


class ErrorSimulator:
    """Simulates sequencing errors in reads."""
    
    def __init__(self, config: Dict):
        """
        Initialize error simulator.
        
        Args:
            config: Error simulation configuration
        """
        self.config = config
        self.substitution_rate = config.get('substitution_rate', 0.001)
        self.insertion_rate = config.get('insertion_rate', 0.0001)
        self.deletion_rate = config.get('deletion_rate', 0.0001)
        self.quality_dependent = config.get('quality_dependent', False)
    
    def introduce_errors(self, 
                        sequence: List[str], 
                        labels: List[str],
                        quality_scores: Optional[np.ndarray],
                        random_state: np.random.RandomState) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
        """
        Introduce errors into a sequence.
        
        Args:
            sequence: Original sequence (as list)
            labels: Original labels (as list)
            quality_scores: Optional quality scores
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (modified_sequence, modified_labels, modified_quality_scores)
        """
        if not any([self.substitution_rate, self.insertion_rate, self.deletion_rate]):
            return sequence, labels, quality_scores
        
        new_seq = []
        new_labels = []
        new_qual = [] if quality_scores is not None else None
        
        i = 0
        while i < len(sequence):
            # Calculate error probability
            if self.quality_dependent and quality_scores is not None:
                # Quality-dependent error rate
                qual = quality_scores[i]
                error_prob = 10 ** (-qual / 10)  # Phred scale
            else:
                error_prob = self.substitution_rate + self.insertion_rate + self.deletion_rate
            
            if random_state.random() < error_prob:
                # Decide on error type
                error_type = random_state.choice(['sub', 'ins', 'del'], 
                                                p=[self.substitution_rate,
                                                   self.insertion_rate,
                                                   self.deletion_rate] / np.sum([self.substitution_rate,
                                                                                  self.insertion_rate,
                                                                                  self.deletion_rate]))
                
                if error_type == 'sub':
                    # Substitution
                    original = sequence[i]
                    bases = ['A', 'C', 'G', 'T']
                    bases.remove(original if original in bases else 'A')
                    new_seq.append(random_state.choice(bases))
                    new_labels.append('ERROR')
                    if new_qual is not None:
                        new_qual.append(quality_scores[i] * 0.5)  # Reduce quality
                    i += 1
                
                elif error_type == 'ins':
                    # Insertion
                    new_seq.append(random_state.choice(['A', 'C', 'G', 'T']))
                    new_labels.append('ERROR')
                    if new_qual is not None:
                        new_qual.append(20)  # Low quality for insertion
                    # Don't increment i (process same position again)
                
                else:  # deletion
                    # Skip this position
                    i += 1
            else:
                # No error
                new_seq.append(sequence[i])
                new_labels.append(labels[i])
                if new_qual is not None:
                    new_qual.append(quality_scores[i])
                i += 1
        
        if new_qual is not None:
            new_qual = np.array(new_qual)
        
        return new_seq, new_labels, new_qual


class SequenceSimulator:
    """
    Comprehensive sequence simulator with all features.
    
    Key features:
    - Probabilistic ACC generation with PWM
    - Flexible sequence architecture
    - Whitelist support for any segment
    - Transcript pool for realistic inserts
    - PolyA tail generation
    - Error simulation
    - Quality score tracking
    """
    
    def __init__(self, config: Optional[Dict] = None, config_file: Optional[str] = None):
        """
        Initialize the read simulator.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}
        
        self.sim_config = self.config.get('simulation', {})
        
        # Initialize random state
        seed = self.sim_config.get('random_seed', 42)
        self.random_state = np.random.RandomState(seed)
        random.seed(seed)
        
        # Initialize components
        self.acc_generator = self._initialize_acc_generator()
        self.whitelist_manager = WhitelistManager(self.sim_config)
        self.transcript_pool = self._initialize_transcript_pool()
        self.polya_generator = self._initialize_polya_generator()
        self.error_simulator = self._initialize_error_simulator()
        
        # Load direct whitelists (backward compatibility)
        self.whitelists = self._load_whitelists()
        
        # Get sequence architecture
        self.sequence_order = self.sim_config.get('sequence_order', [])
        self.sequence_configs = self.sim_config.get('sequences', {})
        
        logger.info(f"Initialized comprehensive sequence simulator")
        logger.info(f"Architecture: {' → '.join(self.sequence_order)}")
    
    def _initialize_acc_generator(self) -> Optional['ProbabilisticPWMGenerator']:
        """Initialize the probabilistic ACC generator from configuration."""
        if not PWM_AVAILABLE:
            return None
            
        pwm_config = self.config.get('pwm', {})
        pwm_file = pwm_config.get('pwm_file')
        
        # Check for PWM in different config locations
        if not pwm_file:
            pwm_files = self.sim_config.get('pwm_files', {})
            pwm_file = pwm_files.get('ACC')
        
        if pwm_file and Path(pwm_file).exists():
            try:
                pwm = self._load_pwm_from_file(pwm_file)
                
                # Get temperature setting (controls diversity)
                temperature = pwm_config.get('temperature', 1.0)
                
                # If old threshold config exists, map it to temperature
                if 'threshold' in pwm_config and 'temperature' not in pwm_config:
                    threshold = pwm_config.get('threshold', 0.8)
                    # Map threshold [0.5, 1.0] to temperature [2.0, 0.5]
                    temperature = 2.5 - 2.0 * threshold
                    logger.info(f"Mapped threshold {threshold} to temperature {temperature}")
                
                min_entropy = pwm_config.get('min_entropy', 0.1)
                
                generator = ProbabilisticPWMGenerator(
                    pwm=pwm,
                    temperature=temperature,
                    min_entropy=min_entropy
                )
                
                logger.info(f"Initialized ACC PWM generator with temperature={temperature}")
                return generator
                
            except Exception as e:
                logger.error(f"Failed to load PWM from {pwm_file}: {e}")
        
        # Fallback: Create PWM from IUPAC pattern if specified
        acc_pattern = pwm_config.get('pattern', 'ACCSSV')
        if acc_pattern and PWM_AVAILABLE:
            logger.info(f"Creating ACC PWM from pattern: {acc_pattern}")
            pwm = create_acc_pwm_from_pattern(acc_pattern)
            temperature = pwm_config.get('temperature', 1.0)
            return ProbabilisticPWMGenerator(pwm, temperature=temperature)
        
        return None
    
    def _load_pwm_from_file(self, filepath: str) -> np.ndarray:
        """Load PWM matrix from file."""
        pwm_data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to parse as tab or space separated values
                    values = line.replace('\t', ' ').split()
                    if len(values) >= 4:
                        # Take first 4 values as [A, C, G, T] probabilities
                        try:
                            probs = [float(v) for v in values[:4]]
                            # Normalize if needed
                            total = sum(probs)
                            if total > 0:
                                probs = [p/total for p in probs]
                            pwm_data.append(probs)
                        except ValueError:
                            continue
        
        if not pwm_data:
            raise ValueError(f"No valid PWM data found in {filepath}")
        
        return np.array(pwm_data)
    
    def _load_whitelists(self) -> Dict[str, List[str]]:
        """Load whitelists from configuration (backward compatibility)."""
        whitelists = {}
        whitelist_files = self.sim_config.get('whitelist_files', {})
        
        for segment, filepath in whitelist_files.items():
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]
                whitelists[segment] = sequences
                logger.info(f"Loaded {len(sequences)} sequences for {segment}")
        
        return whitelists
    
    def _initialize_transcript_pool(self) -> Optional[TranscriptPool]:
        """Initialize transcript pool if configured."""
        transcript_config = self.sim_config.get('transcript_pool', {})
        if transcript_config:
            return TranscriptPool(transcript_config)
        return None
    
    def _initialize_polya_generator(self) -> Optional[PolyATailGenerator]:
        """Initialize polyA tail generator if configured."""
        polya_config = self.sim_config.get('polya', {})
        if polya_config:
            return PolyATailGenerator(polya_config)
        return None
    
    def _initialize_error_simulator(self) -> Optional[ErrorSimulator]:
        """Initialize error simulator if configured."""
        error_config = self.sim_config.get('errors', {})
        if error_config.get('enabled', False):
            return ErrorSimulator(error_config)
        return None
    
    def generate_read(self, 
                     diversity_boost: Optional[float] = None,
                     include_quality: bool = False,
                     inject_errors: bool = True) -> SimulatedRead:
        """
        Generate a single simulated read with all features.
        
        Args:
            diversity_boost: Optional multiplier for ACC diversity
            include_quality: Whether to include quality scores
            inject_errors: Whether to inject sequencing errors
            
        Returns:
            SimulatedRead object
        """
        full_sequence = []
        labels = []
        label_regions = {}
        quality_scores = [] if include_quality else None
        segment_sources = {}
        
        for segment_name in self.sequence_order:
            # Generate segment based on type and configuration
            segment_seq, segment_qual, source = self._generate_segment_with_source(
                segment_name, 
                diversity_boost, 
                include_quality
            )
            
            # Track positions
            start = len(full_sequence)
            end = start + len(segment_seq)
            
            # Add to full sequence
            full_sequence.extend(list(segment_seq))
            labels.extend([segment_name] * len(segment_seq))
            segment_sources[segment_name] = source
            
            # Track regions
            if segment_name not in label_regions:
                label_regions[segment_name] = []
            label_regions[segment_name].append((start, end))
            
            # Add quality scores if available
            if include_quality and segment_qual is not None:
                if isinstance(segment_qual, (list, tuple)):
                    quality_scores.extend(segment_qual)
                else:
                    quality_scores.extend([segment_qual] * len(segment_seq))
        
        # Convert quality scores to array
        if include_quality:
            if not quality_scores:
                # Generate default quality scores
                quality_scores = np.full(len(full_sequence), 30.0)  # Phred 30
            else:
                quality_scores = np.array(quality_scores, dtype=np.float32)
        
        # Introduce errors if configured
        has_errors = False
        if inject_errors and self.error_simulator:
            if self.random_state.random() < self.sim_config.get('error_injection_prob', 0.1):
                full_sequence, labels, quality_scores = self.error_simulator.introduce_errors(
                    full_sequence, labels, quality_scores, self.random_state
                )
                has_errors = True
        
        # Apply reverse complement to entire read with probability
        if self.random_state.random() < self.sim_config.get('full_read_reverse_complement_prob', 0.0):
            full_sequence = self._reverse_complement(full_sequence)
            labels = labels[::-1]
            if quality_scores is not None:
                quality_scores = quality_scores[::-1]
            
            # Update label regions for reverse complement
            seq_len = len(full_sequence)
            new_regions = {}
            for label, regions in label_regions.items():
                new_regions[label] = [(seq_len - end, seq_len - start) for start, end in reversed(regions)]
            label_regions = new_regions
        
        # Build metadata
        metadata = {
            'segment_sources': segment_sources,
            'has_errors': has_errors,
            'diversity_boost': diversity_boost
        }
        
        return SimulatedRead(
            sequence=''.join(full_sequence),
            labels=labels,
            label_regions=label_regions,
            metadata=metadata,
            quality_scores=quality_scores
        )
    
    def _generate_segment_with_source(self, 
                                     segment_name: str,
                                     diversity_boost: Optional[float],
                                     include_quality: bool) -> Tuple[str, Optional[Union[List[float], float]], str]:
        """
        Generate a segment and return its source.
        
        Returns:
            Tuple of (sequence, quality_scores, source)
        """
        segment_upper = segment_name.upper()
        
        # Special handling for known segment types
        if segment_name == 'ACC' and self.acc_generator:
            seq, qual = self._generate_acc_segment(diversity_boost, include_quality)
            return seq, qual, 'pwm'
        
        elif 'POLYA' in segment_upper and self.polya_generator:
            seq = self.polya_generator.generate(self.random_state)
            qual = 25.0 if include_quality else None  # Moderate quality for polyA
            return seq, qual, 'polya_generator'
        
        elif 'INSERT' in segment_upper or 'CDNA' in segment_upper or 'TRANSCRIPT' in segment_upper:
            if self.transcript_pool:
                seq = self.transcript_pool.sample_fragment(self.random_state)
                qual = 30.0 if include_quality else None  # High quality for transcript
                return seq, qual, 'transcript_pool'
        
        # Check whitelist manager first
        if self.whitelist_manager.has_whitelist(segment_name):
            seq = self.whitelist_manager.sample(segment_name, self.random_state)
            qual = 35.0 if include_quality else None  # Very high quality for whitelist
            return seq, qual, 'whitelist'
        
        # Check legacy whitelists
        if segment_name in self.whitelists:
            seq = self.random_state.choice(self.whitelists[segment_name])
            qual = 35.0 if include_quality else None
            return seq, qual, 'whitelist'
        
        # Check fixed sequences
        if segment_name in self.sequence_configs:
            seq = self.sequence_configs[segment_name]
            if seq not in ["random", "transcript", "polya"]:
                qual = 40.0 if include_quality else None  # Perfect quality for fixed
                return seq, qual, 'fixed'
        
        # Default: generate random sequence
        seq = self._generate_segment(segment_name)
        qual = 25.0 if include_quality else None  # Moderate quality for random
        return seq, qual, 'random'
    
    def _generate_acc_segment(self, 
                            diversity_boost: Optional[float] = None,
                            include_quality: bool = False) -> Tuple[str, Optional[Union[List[float], float]]]:
        """Generate ACC segment using PWM generator or fallback."""
        # Check for fixed ACC in config
        acc_config = self.sequence_configs.get('ACC', '')
        if acc_config and acc_config not in ['random', 'pwm']:
            # Use fixed ACC sequence
            qual = 40.0 if include_quality else None
            return acc_config, qual
        
        if include_quality and self.acc_generator:
            # Generate with quality scores based on PWM confidence
            seq_qual_pairs = self.acc_generator.generate_with_quality_scores(1)
            seq, qual = seq_qual_pairs[0]
            return seq, qual.tolist()
        elif self.acc_generator:
            # Generate sequence only
            sequences = self.acc_generator.generate_sequences(
                n=1, 
                diversity_boost=diversity_boost
            )
            return sequences[0], None
        else:
            # Fallback to random ACC-like sequence
            bases = []
            for i in range(6):  # Default ACC length
                if i < 3:
                    bases.append('ACC'[i])
                else:
                    bases.append(self.random_state.choice(['C', 'G', 'T', 'A']))
            return ''.join(bases), 25.0 if include_quality else None
    
    def _generate_segment(self, segment_name: str) -> str:
        """Generate non-ACC segments (existing logic)."""
        # Get configured length or use defaults
        lengths = self.sim_config.get('segment_generation', {}).get('lengths', {})
        segment_upper = segment_name.upper()
        
        # Determine length based on segment type
        if 'UMI' in segment_upper:
            length = lengths.get(segment_name, 12)
        elif 'BARCODE' in segment_upper or 'CBC' in segment_upper:
            length = lengths.get(segment_name, 8)
        elif 'ADAPTER' in segment_upper:
            length = lengths.get(segment_name, 22)
        elif 'INSERT' in segment_upper or 'CDNA' in segment_upper:
            length = lengths.get(segment_name, 200)
        else:
            length = lengths.get(segment_name, 20)
        
        return self._generate_random_dna(length)
    
    def _generate_random_dna(self, length: int) -> str:
        """Generate random DNA sequence."""
        bases = ['A', 'C', 'G', 'T']
        return ''.join(self.random_state.choice(bases, size=length))
    
    def _reverse_complement(self, sequence: Union[List[str], str]) -> Union[List[str], str]:
        """Reverse complement a sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        
        if isinstance(sequence, str):
            return ''.join(complement.get(base, 'N') for base in sequence[::-1])
        else:
            return [complement.get(base, 'N') for base in sequence[::-1]]
    
    def generate_batch(self, 
                      n: int = 100,
                      diversity_schedule: Optional[str] = None,
                      include_quality: bool = False,
                      inject_errors: bool = True) -> List[SimulatedRead]:
        """
        Generate a batch of reads with optional diversity scheduling.
        
        Args:
            n: Number of reads to generate
            diversity_schedule: Optional diversity schedule 
                               ('constant', 'increasing', 'decreasing', 'random')
            include_quality: Whether to include quality scores
            inject_errors: Whether to inject sequencing errors
            
        Returns:
            List of SimulatedRead objects
        """
        reads = []
        
        for i in range(n):
            # Determine diversity boost based on schedule
            if diversity_schedule == 'increasing':
                diversity_boost = 0.5 + 1.5 * (i / n)  # 0.5 to 2.0
            elif diversity_schedule == 'decreasing':
                diversity_boost = 2.0 - 1.5 * (i / n)  # 2.0 to 0.5
            elif diversity_schedule == 'random':
                diversity_boost = 0.5 + 1.5 * self.random_state.random()
            else:
                diversity_boost = None  # Use default
            
            read = self.generate_read(
                diversity_boost=diversity_boost,
                include_quality=include_quality,
                inject_errors=inject_errors
            )
            reads.append(read)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{n} reads")
        
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
    
    def analyze_acc_diversity(self, reads: List[SimulatedRead]) -> Dict:
        """
        Analyze the diversity of ACC sequences in generated reads.
        
        Args:
            reads: List of SimulatedRead objects
            
        Returns:
            Dictionary with diversity statistics
        """
        acc_sequences = []
        
        for read in reads:
            if 'ACC' in read.label_regions:
                for start, end in read.label_regions['ACC']:
                    acc_seq = read.sequence[start:end]
                    acc_sequences.append(acc_seq)
        
        if not acc_sequences:
            return {'error': 'No ACC sequences found'}
        
        diversity_metrics = {
            'total_acc_sequences': len(acc_sequences),
            'unique_sequences': len(set(acc_sequences)),
            'uniqueness_ratio': len(set(acc_sequences)) / len(acc_sequences)
        }
        
        # Additional metrics if ACC generator available
        if self.acc_generator and PWM_AVAILABLE:
            detailed_metrics = self.acc_generator.calculate_diversity_metrics(acc_sequences)
            diversity_metrics.update(detailed_metrics)
            
            # Add PWM scores
            scores = []
            for seq in acc_sequences[:100]:  # Limit to first 100 for efficiency
                score_dict = self.acc_generator.score_sequence_probabilistic(seq)
                scores.append(score_dict['mean_probability'])
            
            diversity_metrics['mean_pwm_score'] = np.mean(scores) if scores else 0
            diversity_metrics['std_pwm_score'] = np.std(scores) if scores else 0
        
        return diversity_metrics
    
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
        all_labels.add('ERROR')  # Added for error simulation
        
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


def demonstrate_probabilistic_generation():
    """Demonstrate the read simulator with probabilistic ACC generation."""
    
    # Configuration with probabilistic ACC
    config = {
        'simulation': {
            'random_seed': 42,
            'sequence_order': ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3'],
            'sequences': {
                'ADAPTER5': 'CTACACGACGCTCTTCCGATCT',
                'ADAPTER3': 'AGATCGGAAGAGCACACGTCTG'
            },
            'segment_generation': {
                'lengths': {
                    'ADAPTER5': 22,
                    'UMI': 12,
                    'ACC': 6,
                    'BARCODE': 8,
                    'INSERT': 50,
                    'ADAPTER3': 22
                }
            },
            'full_read_reverse_complement_prob': 0.5,
            'errors': {
                'enabled': True,
                'substitution_rate': 0.001,
                'insertion_rate': 0.0001,
                'deletion_rate': 0.0001
            }
        },
        'pwm': {
            'pattern': 'ACCSSV',  # ACC pattern with degenerate positions
            'temperature': 1.0,    # Instead of threshold, use temperature
            'min_entropy': 0.1     # Minimum diversity at each position
        }
    }
    
    # Initialize simulator
    simulator = SequenceSimulator(config)
    
    print("Tempest Simulator with Probabilistic ACC Generation")
    print("=" * 60)
    
    # Generate reads with different diversity settings
    print("\n1. Generating reads with default diversity:")
    reads_default = simulator.generate_batch(n=10, include_quality=True)
    
    print("\n2. Generating reads with increasing diversity:")
    reads_increasing = simulator.generate_batch(n=10, diversity_schedule='increasing')
    
    print("\n3. Generating reads with random diversity:")
    reads_random = simulator.generate_batch(n=10, diversity_schedule='random')
    
    # Show some examples
    print("\nExample reads with ACC diversity:")
    for i, read in enumerate(reads_default[:3]):
        if 'ACC' in read.label_regions:
            for start, end in read.label_regions['ACC']:
                acc_seq = read.sequence[start:end]
                print(f"  Read {i+1} ACC: {acc_seq}")
                if read.quality_scores is not None:
                    acc_qual = read.quality_scores[start:end]
                    print(f"         Qual: {acc_qual}")
    
    # Analyze diversity
    print("\nDiversity Analysis:")
    for batch_name, batch in [("Default", reads_default), 
                              ("Increasing", reads_increasing),
                              ("Random", reads_random)]:
        diversity = simulator.analyze_acc_diversity(batch)
        print(f"\n  {batch_name} diversity:")
        print(f"    Unique sequences: {diversity.get('unique_sequences', 'N/A')}")
        print(f"    Uniqueness ratio: {diversity.get('uniqueness_ratio', 0):.3f}")
        print(f"    Mean entropy: {diversity.get('mean_position_entropy', 0):.3f}")
        print(f"    Mean PWM score: {diversity.get('mean_pwm_score', 0):.3f}")
    
    return simulator


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequence Simulator")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output file for generated reads')
    parser.add_argument('--format', choices=['json', 'pickle', 'tsv'], default='json',
                      help='Output format')
    parser.add_argument('--num-sequences', type=int, help='Override number of sequences')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.demo:
        # Run demonstration
        simulator = demonstrate_probabilistic_generation()
    elif args.config:
        # Create simulator from config
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
    else:
        parser.print_help()
        print("\nRun with --demo for a demonstration of the simulator")
    
    print("\nSimulation complete!")
