"""
Parallel Sequence Simulator for Tempest

High-performance parallel version with proper invalid read generation

"""

import os
import numpy as np
import random
import time
import pickle
import gzip
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Value, Lock
from functools import partial
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import tempfile
import sys

# Control logging in workers
def configure_worker_logging():
    """Configure minimal logging for worker processes to prevent console spam."""
    # Suppress most logs in workers
    logging.getLogger().setLevel(logging.WARNING)
    # Specifically suppress noisy loggers
    logging.getLogger("tempest.data").setLevel(logging.WARNING)
    logging.getLogger("tempest.utils").setLevel(logging.WARNING)
    logging.getLogger("tempest.core").setLevel(logging.WARNING)

# Import the original simulator components
from tempest.data.simulator import (
    SimulatedRead,
    SequenceSimulator,
    create_simulator_from_config
)
from tempest.data.invalid_generator import InvalidReadGenerator
from tempest.config import TempestConfig

# Main process logger
logger = logging.getLogger(__name__)


class ParallelSequenceSimulator(SequenceSimulator):
    """
    Parallel version of SequenceSimulator with multiprocessing support.
    
    Maintains full compatibility with the original SequenceSimulator
    while providing dramatic speedups through parallelization.
    """
    
    def __init__(self, config=None, config_file=None, n_workers=None):
        """
        Initialize parallel simulator.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
            n_workers: Number of worker processes (None = auto-detect)
        """
        super().__init__(config, config_file)
        
        # Determine optimal number of workers
        if n_workers is None:
            # Use 80% of available CPUs, minimum 1, maximum 32
            n_cpus = mp.cpu_count()
            self.n_workers = min(32, max(1, int(n_cpus * 0.8)))
        else:
            self.n_workers = max(1, n_workers)
            
        logger.info(f"Initialized ParallelSequenceSimulator with {self.n_workers} workers")
        
        # Store config for worker processes
        # CRITICAL FIX: Convert config to dict for proper serialization in multiprocessing
        if hasattr(self.config, '_to_dict'):
            # Use the _to_dict method if available (TempestConfig object)
            self._serialized_config = self.config._to_dict()
            logger.debug("Serialized TempestConfig to dict for multiprocessing")
        elif isinstance(self.config, dict):
            # Already a dict
            self._serialized_config = self.config
        else:
            # Try to convert to dict (fallback)
            import copy
            self._serialized_config = copy.deepcopy(self.config)
            logger.warning("Config serialization may not preserve all values")
    
    def generate_batch(
        self,
        n: int = 100,
        diversity_schedule: Optional[str] = None,
        include_quality: bool = False,
        inject_errors: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
        chunk_size: Optional[int] = None
    ) -> List[SimulatedRead]:
        """
        Generate a batch of reads in parallel.
        
        Args:
            n: Number of reads to generate
            diversity_schedule: Optional diversity schedule
            include_quality: Whether to include quality scores
            inject_errors: Whether to inject sequencing errors
            progress_callback: Callback for progress updates
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            List of SimulatedRead objects
        """
        start_time = time.time()
        
        # Determine optimal chunk size
        if chunk_size is None:
            # Balance between parallelism and overhead
            chunk_size = max(10, n // (self.n_workers * 4))
            chunk_size = min(chunk_size, 1000)  # Cap at 1000 for memory efficiency
        
        # Create chunks with proper indices for diversity scheduling
        chunks = []
        for i in range(0, n, chunk_size):
            chunk_end = min(i + chunk_size, n)
            chunks.append({
                'start_idx': i,
                'end_idx': chunk_end,
                'count': chunk_end - i,
                'total_n': n,
                'diversity_schedule': diversity_schedule,
                'include_quality': include_quality,
                'inject_errors': inject_errors,
                'config': self._serialized_config,
                'seed_offset': i  # Ensure different random seeds per chunk
            })
        
        logger.info(f"Generating {n} reads in {len(chunks)} chunks of ~{chunk_size} using {self.n_workers} workers")
        
        # Set up progress tracking
        if progress_callback:
            manager = Manager()
            progress_queue = manager.Queue()
            progress_counter = Value('i', 0)
            
            # Start progress monitor thread
            import threading
            stop_progress = threading.Event()
            
            def progress_monitor():
                while not stop_progress.is_set():
                    try:
                        while not progress_queue.empty():
                            increment = progress_queue.get_nowait()
                            with progress_counter.get_lock():
                                progress_counter.value += increment
                                progress_callback(progress_counter.value)
                    except:
                        pass
                    time.sleep(0.01)
            
            progress_thread = threading.Thread(target=progress_monitor)
            progress_thread.start()
        else:
            progress_queue = None
        
        # Generate reads in parallel with logging control
        try:
            # Use initializer to configure logging in each worker
            with Pool(processes=self.n_workers, initializer=configure_worker_logging) as pool:
                # Use partial to bind the progress queue
                worker_func = partial(_generate_chunk_worker, progress_queue=progress_queue)
                
                # Map chunks to workers
                chunk_results = pool.map(worker_func, chunks)
                
                # Flatten results
                reads = []
                for chunk_reads in chunk_results:
                    reads.extend(chunk_reads)
        
        finally:
            # Clean up progress monitoring
            if progress_callback:
                stop_progress.set()
                progress_thread.join(timeout=1.0)
                # Final update to ensure we show 100%
                progress_callback(n)
        
        # Log statistics
        elapsed = time.time() - start_time
        reads_per_second = n / elapsed if elapsed > 0 else 0
        logger.info(f"Generated {n} reads in {elapsed:.2f}s ({reads_per_second:.0f} reads/sec)")
        
        # Log generation statistics (suppress detailed logging)
        if len(reads) > 0:
            logger.debug(f"Sample read: {len(reads[0].sequence)} bp")
        
        return reads
    
    def generate_train_val_split(
        self,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[List[SimulatedRead], List[SimulatedRead]]:
        """
        Generate training and validation datasets in parallel.
        
        Returns:
            Tuple of (training_reads, validation_reads)
        """
        total_sequences = self.sim_config.get("num_sequences", 10000)
        train_split = self.sim_config.get("train_split", 0.8)
        
        num_train = int(total_sequences * train_split)
        num_val = total_sequences - num_train
        
        logger.info(f"Generating {num_train} training and {num_val} validation reads in parallel")
        
        # Generate both sets in parallel
        train_reads = self.generate_batch(
            num_train, 
            progress_callback=progress_callback if progress_callback else None
        )
        
        # Adjust progress callback for validation set
        if progress_callback:
            def val_progress_callback(n_generated):
                progress_callback(num_train + n_generated)
            val_reads = self.generate_batch(num_val, progress_callback=val_progress_callback)
        else:
            val_reads = self.generate_batch(num_val)
        
        return train_reads, val_reads


def _generate_chunk_worker(chunk_info: Dict[str, Any], progress_queue: Optional[Queue] = None) -> List[SimulatedRead]:
    """
    Worker function for parallel read generation with logging suppression.
    
    This function runs in a separate process and generates a chunk of reads.
    
    Args:
        chunk_info: Dictionary containing chunk parameters
        progress_queue: Optional queue for progress updates
        
    Returns:
        List of SimulatedRead objects
    """
    # Suppress logging in workers (called via initializer, but also here for safety)
    logging.getLogger().setLevel(logging.WARNING)
    
    # Extract parameters
    start_idx = chunk_info['start_idx']
    end_idx = chunk_info['end_idx']
    count = chunk_info['count']
    total_n = chunk_info['total_n']
    diversity_schedule = chunk_info['diversity_schedule']
    include_quality = chunk_info['include_quality']
    inject_errors = chunk_info['inject_errors']
    config = chunk_info['config']
    seed_offset = chunk_info['seed_offset']
    
    # Create a simulator instance for this worker
    # Each worker gets its own random seed to ensure different sequences
    import copy
    worker_config = copy.deepcopy(config)

    sim_section = worker_config.setdefault('simulation', {})
    base_seed = sim_section.get('random_seed')

    # Fall back if missing or None
    if base_seed is None:
        base_seed = 42

    # Ensure ints
    sim_section['random_seed'] = int(base_seed) + int(seed_offset)
    
    # Suppress simulator initialization logging
    simulator = SequenceSimulator(config=worker_config)
    
    # Generate reads for this chunk
    reads = []
    
    for i in range(count):
        global_idx = start_idx + i
        
        # Determine diversity boost based on schedule
        if diversity_schedule == "increasing":
            diversity_boost = 0.5 + 1.5 * (global_idx / total_n)  # 0.5 to 2.0
        elif diversity_schedule == "decreasing":
            diversity_boost = 2.0 - 1.5 * (global_idx / total_n)  # 2.0 to 0.5
        elif diversity_schedule == "random":
            diversity_boost = 0.5 + 1.5 * simulator.random_state.random()
        else:
            diversity_boost = None  # Use default
        
        read = simulator.generate_read(
            diversity_boost=diversity_boost,
            include_quality=include_quality,
            inject_errors=inject_errors,
        )
        reads.append(read)
        
        # Update progress every 10 reads
        if progress_queue and (i + 1) % 10 == 0:
            try:
                progress_queue.put(10)
            except:
                pass  # Ignore queue errors
    
    # Final progress update for remainder
    if progress_queue and count % 10 != 0:
        try:
            progress_queue.put(count % 10)
        except:
            pass
    
    return reads


class ParallelInvalidReadGenerator(InvalidReadGenerator):
    """
    Parallel version of InvalidReadGenerator with multiprocessing support.
    
    Properly handles all corruption methods with parallel execution.
    """
    
    def __init__(self, config=None, n_workers=None):
        """
        Initialize parallel invalid read generator.
        
        Args:
            config: Configuration
            n_workers: Number of worker processes (None = auto-detect)
        """
        super().__init__(config)
        
        # Determine optimal number of workers
        if n_workers is None:
            n_cpus = mp.cpu_count()
            # Use fewer workers for invalid generation as it's less CPU intensive
            self.n_workers = min(16, max(1, int(n_cpus * 0.5)))
        else:
            self.n_workers = max(1, n_workers)
            
        logger.info(f"Initialized ParallelInvalidReadGenerator with {self.n_workers} workers")
    
    def generate_batch(
        self,
        valid_reads: List[SimulatedRead],
        invalid_ratio: Optional[float] = None,
        invalid_fraction: Optional[float] = None,
        chunk_size: Optional[int] = None
    ) -> List[SimulatedRead]:
        """
        Generate invalid reads from valid ones in parallel.
        
        Args:
            valid_reads: List of valid SimulatedRead objects
            invalid_ratio: Fraction of reads to make invalid
            invalid_fraction: Alternative name for invalid_ratio (backward compat)
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Combined list of valid and invalid reads
        """
        # Handle backward compatibility for parameter names
        if invalid_fraction is not None:
            invalid_ratio = invalid_fraction
        if invalid_ratio is None:
            invalid_ratio = 0.1
            
        if invalid_ratio <= 0:
            return valid_reads
        
        n_valid = len(valid_reads)
        n_invalid = max(1, int(n_valid * invalid_ratio)) if invalid_ratio > 0 else 0
        
        if n_invalid == 0:
            return valid_reads
        
        logger.info(f"Generating {n_invalid} invalid reads from {n_valid} valid reads "
                   f"({invalid_ratio:.1%} corruption rate) using {self.n_workers} workers")
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(10, n_invalid // (self.n_workers * 2))
            chunk_size = min(chunk_size, 500)
        
        # Sample reads to corrupt with their indices
        indices_to_corrupt = np.random.choice(n_valid, n_invalid, replace=False)
        reads_to_corrupt = [(idx, valid_reads[idx]) for idx in indices_to_corrupt]
        
        # Create chunks with proper structure
        chunks = []
        for i in range(0, len(reads_to_corrupt), chunk_size):
            chunk_end = min(i + chunk_size, len(reads_to_corrupt))
            chunks.append({
                'read_pairs': reads_to_corrupt[i:chunk_end],
                'error_probabilities': self.error_probabilities,
                'seed_offset': i,
                'config': self.config  # Pass config for proper corruption
            })
        
        # Process chunks in parallel with logging control
        with Pool(processes=self.n_workers, initializer=configure_worker_logging) as pool:
            chunk_results = pool.map(_corrupt_chunk_worker, chunks)
        
        # Build result list
        result_reads = valid_reads.copy()
        
        # Replace corrupted reads at their indices
        for chunk_corruptions in chunk_results:
            for original_idx, corrupted_read in chunk_corruptions:
                result_reads[original_idx] = corrupted_read
        
        # Verify corruption count
        actual_invalid = sum(1 for r in result_reads 
                           if r.metadata and r.metadata.get('is_invalid', False))
        
        if actual_invalid != n_invalid:
            logger.debug(f"Expected {n_invalid} invalid reads, got {actual_invalid}")
        
        return result_reads


def _corrupt_chunk_worker(chunk_info: Dict[str, Any]) -> List[Tuple[int, SimulatedRead]]:
    """
    Worker function for parallel invalid read generation with logging suppression.
    
    This function runs in a separate process and corrupts reads using
    all available corruption methods from InvalidReadGenerator.
    
    Args:
        chunk_info: Dictionary containing chunk parameters with 'read_pairs'
        
    Returns:
        List of (original_index, corrupted_read) tuples
    """
    # Suppress logging in workers
    logging.getLogger().setLevel(logging.WARNING)
    
    read_pairs = chunk_info['read_pairs']
    error_probabilities = chunk_info['error_probabilities']
    seed_offset = chunk_info['seed_offset']
    config = chunk_info.get('config')
    
    # Set random seed for reproducibility
    np.random.seed(42 + seed_offset)
    random.seed(42 + seed_offset)
    
    # Create a local invalid generator instance for this worker
    generator = InvalidReadGenerator(config)
    generator.error_probabilities = error_probabilities
    
    # Corrupt reads
    corrupted_pairs = []
    
    for original_idx, read in read_pairs:
        # Choose error type based on probabilities
        error_type = np.random.choice(
            list(error_probabilities.keys()),
            p=list(error_probabilities.values())
        )
        
        # Apply corruption using actual InvalidReadGenerator methods
        try:
            corrupted = generator.generate_invalid_read(read, error_type)
            # if error_type == "segment_loss":
                # corrupted = generator._apply_segment_loss(read)
            # elif error_type == "segment_duplication":
                # corrupted = generator._apply_segment_duplication(read)
            # elif error_type == "truncation":
                # corrupted = generator._apply_truncation(read)
            # elif error_type == "chimeric":
                # # For chimeric, we'd need another read - use truncation as fallback
                # corrupted = generator._apply_truncation(read)
                # if corrupted.metadata:
                    # corrupted.metadata['error_type'] = 'chimeric'
            # elif error_type == "scrambled":
                # corrupted = generator._apply_scrambled(read)
            # else:
                # corrupted = read
        except Exception as e:
            # If corruption fails, return original with invalid flag
            corrupted = SimulatedRead(
                sequence=read.sequence,
                labels=read.labels[:],
                label_regions=read.label_regions.copy() if read.label_regions else {},
                metadata={'is_invalid': True, 'error_type': 'corruption_failed'}
            )
        
        # Mark as invalid
        if corrupted.metadata is None:
            corrupted.metadata = {}
        corrupted.metadata['is_invalid'] = True
        if 'error_type' not in corrupted.metadata:
            corrupted.metadata['error_type'] = error_type
        
        corrupted_pairs.append((original_idx, corrupted))
    
    return corrupted_pairs


def create_parallel_simulator_from_config(
    config_source,
    n_workers: Optional[int] = None
) -> ParallelSequenceSimulator:
    """
    Factory function to create a ParallelSequenceSimulator from configuration.
    
    Args:
        config_source: TempestConfig, dict, or path to config file
        n_workers: Number of worker processes (None = auto-detect)
        
    Returns:
        ParallelSequenceSimulator instance
    """
    from tempest.config import TempestConfig
    
    if isinstance(config_source, TempestConfig):
        # Convert to dict
        if hasattr(config_source, "_to_dict"):
            cfg_dict = config_source._to_dict()
        elif hasattr(config_source, "to_dict"):
            cfg_dict = config_source.to_dict()
        else:
            try:
                from dataclasses import asdict
                cfg_dict = asdict(config_source)
            except Exception:
                raise TypeError("TempestConfig cannot be converted to dict")
        return ParallelSequenceSimulator(config=cfg_dict, n_workers=n_workers)
    
    elif isinstance(config_source, (str, Path)):
        return ParallelSequenceSimulator(config_file=str(config_source), n_workers=n_workers)
    
    elif isinstance(config_source, dict):
        return ParallelSequenceSimulator(config=config_source, n_workers=n_workers)
    
    else:
        raise TypeError(
            f"Unsupported config source type: {type(config_source)}. "
            "Expected TempestConfig, dict, or path string."
        )


# Export main classes and functions
__all__ = [
    'ParallelSequenceSimulator',
    'ParallelInvalidReadGenerator',
    'create_parallel_simulator_from_config',
    'configure_worker_logging'
]
