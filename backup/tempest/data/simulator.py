"""
Data simulator for Tempest.

Generates synthetic training data with configurable sequence structures,
ACC PWM-based generation, and error injection.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..utils.config import SimulationConfig
from ..utils.io import load_acc_priors, load_barcodes, load_pwm
from ..core.pwm import PWMScorer, generate_acc_from_pwm

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRead:
    """Container for a simulated read with labels."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict  # Additional info


class SequenceSimulator:
    """
    Simulate training data for sequence annotation.
    
    Generates sequences with defined structure (adapters, UMI, ACC, barcode, insert)
    with realistic error profiles.
    """
    
    def __init__(self, config: SimulationConfig, pwm_file: Optional[str] = None):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration
            pwm_file: Optional PWM file for ACC generation
        """
        self.config = config
        self.random_state = np.random.RandomState(config.random_seed)
        
        # Load ACC priors if provided
        self.acc_sequences = None
        self.acc_frequencies = None
        if config.acc_priors_file:
            try:
                self.acc_sequences, self.acc_frequencies = load_acc_priors(
                    config.acc_priors_file
                )
                logger.info(f"Loaded {len(self.acc_sequences)} ACC sequences from priors")
            except Exception as e:
                logger.warning(f"Could not load ACC priors: {e}")
        
        # If priors not available but sequences/frequencies provided directly
        if self.acc_sequences is None and config.acc_sequences:
            self.acc_sequences = config.acc_sequences
            self.acc_frequencies = config.acc_frequencies
            logger.info(f"Using {len(self.acc_sequences)} ACC sequences from config")
        
        # Load PWM if provided
        self.pwm = None
        self.pwm_scorer = None
        if pwm_file:
            try:
                self.pwm = load_pwm(pwm_file)
                self.pwm_scorer = PWMScorer(self.pwm)
                logger.info(f"Loaded PWM with shape {self.pwm.shape}")
            except Exception as e:
                logger.warning(f"Could not load PWM: {e}")
        
        # Load barcodes if provided
        self.barcodes = None
        if config.barcode_file:
            try:
                self.barcodes = load_barcodes(config.barcode_file)
                logger.info(f"Loaded {len(self.barcodes)} barcodes")
            except Exception as e:
                logger.warning(f"Could not load barcodes: {e}")
        
        # Validate sequence order
        if not config.sequence_order:
            raise ValueError("sequence_order must be specified in config")
        
        logger.info(f"Initialized simulator with structure: {' → '.join(config.sequence_order)}")
    
    def generate_random_sequence(self, length: int) -> str:
        """Generate random DNA sequence."""
        bases = ['A', 'C', 'G', 'T']
        return ''.join(self.random_state.choice(bases, size=length))
    
    def generate_umi(self) -> str:
        """Generate random UMI sequence."""
        return self.generate_random_sequence(self.config.umi_length)
    
    def generate_acc(self) -> str:
        """
        Generate ACC sequence using priors or PWM.
        
        Priority:
        1. Sample from ACC priors if available
        2. Generate from PWM if available
        3. Generate random 6bp sequence as fallback
        """
        # Use priors if available
        if self.acc_sequences and self.acc_frequencies:
            return self.random_state.choice(self.acc_sequences, p=self.acc_frequencies)
        
        # Use PWM if available
        if self.pwm is not None:
            acc_seq = generate_acc_from_pwm(self.pwm, n=1)[0]
            return acc_seq
        
        # Fallback: random sequence
        # Default ACC length is 6
        logger.warning("No ACC priors or PWM available, generating random sequence")
        return self.generate_random_sequence(6)
    
    def generate_barcode(self) -> str:
        """
        Generate barcode sequence.
        
        If barcode list provided, sample from it.
        Otherwise generate random sequence.
        """
        if self.barcodes:
            return self.random_state.choice(self.barcodes)
        
        # Default barcode length is 16
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
        Get sequence for a component based on config.
        
        Args:
            component: Component name (e.g., 'ADAPTER5', 'UMI', 'ACC')
            
        Returns:
            Sequence string
        """
        # Check if sequence is defined in config
        if component in self.config.sequences:
            defined_seq = self.config.sequences[component]
            
            # If 'random', generate appropriate sequence
            if defined_seq.lower() == 'random':
                if 'INSERT' in component.upper():
                    return self.generate_insert()
                else:
                    # For other random components, generate reasonable length
                    return self.generate_random_sequence(10)
            else:
                return defined_seq
        
        # Component-specific generation
        if 'UMI' in component.upper():
            return self.generate_umi()
        elif 'ACC' in component.upper():
            return self.generate_acc()
        elif 'BARCODE' in component.upper() or 'BC' in component.upper():
            return self.generate_barcode()
        elif 'INSERT' in component.upper():
            return self.generate_insert()
        else:
            # Unknown component, generate short random sequence
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
        if self.config.error_rate <= 0:
            return sequence
        
        bases = ['A', 'C', 'G', 'T']
        result = list(sequence)
        
        for i in range(len(result)):
            if self.random_state.random() < self.config.error_rate:
                error_type = self.random_state.choice(['substitution', 'deletion', 'insertion'])
                
                if error_type == 'substitution':
                    # Replace with different base
                    current = result[i]
                    other_bases = [b for b in bases if b != current]
                    result[i] = self.random_state.choice(other_bases)
                
                elif error_type == 'deletion':
                    # Mark for deletion
                    result[i] = ''
                
                elif error_type == 'insertion':
                    # Insert random base
                    result[i] = result[i] + self.random_state.choice(bases)
        
        return ''.join(result)
    
    def generate_single_read(self) -> SimulatedRead:
        """
        Generate a single simulated read.
        
        Returns:
            SimulatedRead object
        """
        components = []
        component_labels = []
        
        # Generate each component
        for component_name in self.config.sequence_order:
            seq = self.get_component_sequence(component_name)
            components.append(seq)
            component_labels.append(component_name)
        
        # Concatenate to form full sequence
        full_sequence = ''.join(components)
        
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
            # Store original before errors
            original_sequence = full_sequence
            full_sequence = self.inject_errors(full_sequence)
            
            # Update labels to match new length
            # This is simplified - proper error handling would adjust labels
            if len(full_sequence) != len(labels):
                # Adjust labels if length changed due to indels
                if len(full_sequence) > len(labels):
                    # Insertions occurred
                    labels.extend(['UNKNOWN'] * (len(full_sequence) - len(labels)))
                else:
                    # Deletions occurred
                    labels = labels[:len(full_sequence)]
        
        metadata = {
            'num_components': len(components),
            'original_length': len(''.join(components)),
            'final_length': len(full_sequence),
            'error_rate': self.config.error_rate
        }
        
        return SimulatedRead(
            sequence=full_sequence,
            labels=labels,
            label_regions=label_regions,
            metadata=metadata
        )
    
    def generate(self, num_sequences: Optional[int] = None) -> List[SimulatedRead]:
        """
        Generate multiple simulated reads.
        
        Args:
            num_sequences: Number of sequences to generate (default: from config)
            
        Returns:
            List of SimulatedRead objects
        """
        if num_sequences is None:
            num_sequences = self.config.num_sequences
        
        logger.info(f"Generating {num_sequences} simulated reads...")
        
        reads = []
        for i in range(num_sequences):
            read = self.generate_single_read()
            reads.append(read)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1}/{num_sequences} reads")
        
        logger.info(f"✓ Generated {len(reads)} reads")
        
        # Log statistics
        total_length = sum(len(r.sequence) for r in reads)
        avg_length = total_length / len(reads)
        logger.info(f"  Average read length: {avg_length:.1f} bp")
        
        # Count labels
        all_labels = set()
        for read in reads:
            all_labels.update(read.labels)
        logger.info(f"  Unique labels: {len(all_labels)} ({', '.join(sorted(all_labels))})")
        
        return reads
    
    def generate_train_val_split(self, train_fraction: float = 0.8) -> Tuple[List[SimulatedRead], List[SimulatedRead]]:
        """
        Generate reads and split into train/validation sets.
        
        Args:
            train_fraction: Fraction of data for training (default: 0.8)
            
        Returns:
            Tuple of (train_reads, val_reads)
        """
        all_reads = self.generate()
        
        # Shuffle
        indices = list(range(len(all_reads)))
        self.random_state.shuffle(indices)
        
        # Split
        split_idx = int(len(all_reads) * train_fraction)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_reads = [all_reads[i] for i in train_indices]
        val_reads = [all_reads[i] for i in val_indices]
        
        logger.info(f"Split: {len(train_reads)} train, {len(val_reads)} validation")
        
        return train_reads, val_reads


def reads_to_arrays(reads: List[SimulatedRead], 
                   label_to_idx: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Convert SimulatedRead objects to numpy arrays for training.
    
    Args:
        reads: List of SimulatedRead objects
        label_to_idx: Optional mapping from label to index
        
    Returns:
        Tuple of (sequences_array, labels_array, label_to_idx)
        - sequences_array: (num_reads, max_length) with base indices
        - labels_array: (num_reads, max_length) with label indices
    """
    # Base encoding
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Create label mapping if not provided
    if label_to_idx is None:
        all_labels = set()
        for read in reads:
            all_labels.update(read.labels)
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    # Find max length
    max_length = max(len(read.sequence) for read in reads)
    
    # Initialize arrays
    num_reads = len(reads)
    sequences_array = np.zeros((num_reads, max_length), dtype=np.int32)
    labels_array = np.zeros((num_reads, max_length), dtype=np.int32)
    
    # Fill arrays
    for i, read in enumerate(reads):
        # Encode sequence
        for j, base in enumerate(read.sequence.upper()):
            sequences_array[i, j] = base_to_idx.get(base, 4)  # 4 for unknown
        
        # Encode labels
        for j, label in enumerate(read.labels):
            if j < max_length:  # Safety check
                labels_array[i, j] = label_to_idx.get(label, 0)
    
    return sequences_array, labels_array, label_to_idx
