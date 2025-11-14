"""
Data Simulator for Tempest
"""

import os
import numpy as np
import random
import time
import pickle
import gzip
import json
import mmap
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import yaml
import hashlib
from contextlib import contextmanager

# Conditional import for PWM generator
try:
    from tempest.core.pwm_probabilistic import (
        ProbabilisticPWMGenerator,
        create_acc_pwm_from_pattern,
    )
    PWM_AVAILABLE = True
except ImportError:
    PWM_AVAILABLE = False
    logging.warning(
        "ProbabilisticPWMGenerator not available, "
        "ACC generation will use fallback methods"
    )

# Unified I/O utilities (for PWM loading, etc.)
from tempest.utils import io

logger = logging.getLogger(__name__)

# Vectorized, thread-safe implementation for reverse complementing sequences
# Initialize LUT at module load time for thread safety
def _init_rc_lut():
    """Initialize reverse complement lookup table."""
    lut = np.full(256, ord("N"), dtype=np.uint8)
    
    # Uppercase mapping
    lut[ord("A")] = ord("T")
    lut[ord("T")] = ord("A")
    lut[ord("C")] = ord("G")
    lut[ord("G")] = ord("C")
    lut[ord("N")] = ord("N")
    
    # Lowercase to uppercase complement
    lut[ord("a")] = ord("T")
    lut[ord("t")] = ord("A")
    lut[ord("c")] = ord("G")
    lut[ord("g")] = ord("C")
    lut[ord("n")] = ord("N")
    
    return lut

_RC_LUT = _init_rc_lut()


def reverse_complement(sequence: Union[str, List[str]]) -> str:
    """
    Thread-safe vectorized reverse complement implementation.
    Accepts list[str] or str; always returns uppercase string.
    """
    # Convert list-of-chars to string
    if isinstance(sequence, list):
        sequence = "".join(sequence)

    # Encode safely (replace invalid chars instead of dropping)
    arr = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8).copy()

    # Sanitize
    invalid_mask = (_RC_LUT[arr] == ord("N")) & (arr != ord("N"))
    arr[invalid_mask] = ord("N")

    # Reverse complement
    rc_arr = _RC_LUT[arr[::-1]]

    return rc_arr.tobytes().decode("ascii")

@dataclass
class SimulatedRead:
    """Container for a simulated read with labels and metadata."""
    sequence: str
    labels: List[str]  # One label per base
    label_regions: Dict[str, List[Tuple[int, int]]]  # Label -> [(start, end), ...]
    metadata: Dict = field(default_factory=dict)  # Additional info
    quality_scores: Optional[np.ndarray] = None  # Optional quality scores


@dataclass
class FastaIndexEntry:
    """Represents an entry from a FASTA index file."""
    name: str
    length: int
    offset: int  # Byte offset in the file
    line_bases: int  # Number of bases per line
    line_width: int  # Number of bytes per line (including newline)
    
    @classmethod
    def from_fai_line(cls, line: str) -> 'FastaIndexEntry':
        """Parse a line from a .fai file."""
        parts = line.strip().split('\t')
        if len(parts) < 5:
            raise ValueError(f"Invalid .fai line: {line}")
        
        return cls(
            name=parts[0],
            length=int(parts[1]),
            offset=int(parts[2]),
            line_bases=int(parts[3]),
            line_width=int(parts[4])
        )


class SizeAwareLRUCache:
    """
    Size-aware LRU cache that evicts based on total sequence bases cached,
    not just the count of sequences. This is more memory-efficient for
    transcripts of varying lengths.
    """
    
    def __init__(self, max_size_mb: float = 100):
        """
        Initialize cache with maximum size in megabytes.
        
        Args:
            max_size_mb: Maximum cache size in MB (default 100MB)
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.sizes = {}  # Track size of each cached item
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[str]:
        """Get item from cache, updating access order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: str) -> None:
        """Add item to cache, evicting LRU items if size limit exceeded."""
        # Approximate size is simply number of characters (ASCII FASTA)
        item_size = len(value)
        # skip caching “whales” that would thrash the cache (>10% of capacity)
        if item_size > 0.10 * self.max_size_bytes:
            return
        
        if key in self.cache:
            # Update existing item
            old_size = self.sizes[key]
            self.current_size = self.current_size - old_size + item_size
            self.cache.move_to_end(key)
            self.cache[key] = value
            self.sizes[key] = item_size
        else:
            # Evict items if necessary to make space
            while self.current_size + item_size > self.max_size_bytes and self.cache:
                # Remove least recently used item
                lru_key = next(iter(self.cache))
                lru_size = self.sizes[lru_key]
                del self.cache[lru_key]
                del self.sizes[lru_key]
                self.current_size -= lru_size
                self.evictions += 1
            
            # Add new item if it fits
            if self.current_size + item_size <= self.max_size_bytes:
                self.cache[key] = value
                self.sizes[key] = item_size
                self.current_size += item_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        avg_size = np.mean(list(self.sizes.values())) if self.sizes else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'num_items': len(self.cache),
            'current_size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self.current_size / self.max_size_bytes,
            'avg_item_size_kb': avg_size / 1024
        }


class GzippedFastaReader:
    """
    Efficient reader for gzipped FASTA files that keeps the file handle open
    and uses an internal index for faster sequential access.
    """
    
    def __init__(self, fasta_file: Path, index_entries: List[FastaIndexEntry]):
        """
        Initialize gzipped FASTA reader.
        
        Args:
            fasta_file: Path to gzipped FASTA file
            index_entries: List of index entries
        """
        self.fasta_file = fasta_file
        self.index_entries = {e.name: e for e in index_entries}
        self.handle = None
        self.current_position = {}  # Track position for each sequence
        self._open_file()
        self._build_sequence_index()
    
    def _open_file(self):
        """Open the gzipped file."""
        self.handle = gzip.open(self.fasta_file, 'rt')
    
    def _build_sequence_index(self):
        """
        Build an index of sequence positions in the gzipped file.
        
        Note: For gzipped files, we cannot build a position-based index because
        gzip streams don't support reliable .tell() after iteration. We rely on the .fai
        index for metadata and do sequential reading when needed.
        This is a no-op placeholder for future non-gzipped optimization.
        """
        logger.debug("Gzipped FASTA: using sequential reading with .fai metadata")
        self.sequence_positions = {}
    
    def read_sequence(self, entry_name: str) -> Optional[str]:
        """
        Read a specific sequence efficiently.
        
        Args:
            entry_name: Name of the sequence to read
            
        Returns:
            Sequence string or None if not found
        """
        if entry_name not in self.index_entries:
            logger.warning(f"Sequence {entry_name} not in index")
            return None
        
        entry = self.index_entries[entry_name]
        
        try:
            # Reset to beginning (gzip doesn't support random seeking well)
            self.handle.seek(0)
            
            # Read through file to find sequence
            found = False
            sequence_parts = []
            
            for line in self.handle:
                line = line.strip()
                if line.startswith('>'):
                    if found:
                        break  # Found next sequence, stop
                    current_name = line[1:].split()[0]
                    if current_name == entry_name:
                        found = True
                elif found:
                    sequence_parts.append(line)
            
            if not sequence_parts:
                logger.error(f"Sequence {entry_name} not found or empty")
                return None
            
            sequence = ''.join(sequence_parts).upper()
            
            # Validate sequence length matches index
            if len(sequence) != entry.length:
                logger.warning(
                    f"Sequence length mismatch for {entry_name}: "
                    f"expected {entry.length}, got {len(sequence)}"
                )
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error reading sequence {entry_name}: {e}")
            return None
    
    def close(self):
        """Close the file handle."""
        if self.handle:
            self.handle.close()
            self.handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed."""
        self.close()
        return False
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()


class TranscriptPool:
    """
    Optimized transcript pool using FASTA index for efficient sampling.
    
    Enhancements in this version:
    - Size-aware caching (evicts by total bases, not count)
    - Persistent gzipped file reader (doesn't reopen for each read)
    - Robust error handling for corrupted entries
    - Progress bars for long operations
    - Validation of sequence lengths against index
    """
    
    def __init__(self, config: Dict, cache_size_mb: float = 100):
        """
        Initialize transcript pool from configuration.
        
        Args:
            config: Transcript configuration dictionary
            cache_size_mb: Maximum cache size in megabytes
        """
        self.config = config
        self.cache = SizeAwareLRUCache(
            max_size_mb=float(self.config.get("cache_size_mb", cache_size_mb))
            )
        self.index_entries = []
        self.filtered_entries = []
        self.fasta_file = None
        self.fasta_handle = None
        self.is_gzipped = False
        self.mmap_handle = None
        self.gzip_reader = None  # Persistent gzipped reader
        self._length_weights = None  # For length-weighted sampling
        
        # Statistics tracking
        self.stats = {
            'sequences_read': 0,
            'validation_errors': 0,
            'corrupted_entries': 0,
            'total_bases_processed': 0
        }
        
        # Initialize if FASTA file is provided
        fasta_file = config.get("fasta_file")
        if fasta_file and Path(fasta_file).exists():
            self._initialize_index(fasta_file)
        else:
            logger.warning(f"Transcript file not found: {fasta_file}")
    
    def _initialize_index(self, fasta_file: str) -> None:
        """Initialize the FASTA index for efficient access."""
        self.fasta_file = Path(fasta_file)
        self.is_gzipped = str(fasta_file).endswith('.gz')
        
        # Look for index file
        fai_file = self._find_index_file(fasta_file)
        
        if not fai_file or not fai_file.exists():
            logger.error(f"FASTA index file not found for: {fasta_file}")
            logger.info("Please create index using: samtools faidx {fasta_file}")
            logger.warning("Falling back to slow loading method...")
            self._load_transcripts_fallback(fasta_file)
            return
        
        # Load and validate index
        try:
            self._load_index(fai_file)
            self.index_entries = [
                e for e in self.index_entries
                if isinstance(e.length, int) and e.length > 0
            ]
            self._validate_index()
            self._filter_transcripts()
            self._open_fasta_handle()
            
            logger.info(
                f"Initialized TranscriptPool with {len(self.filtered_entries)} transcripts "
                f"(filtered from {len(self.index_entries)} total)"
            )
            
            if self.filtered_entries:
                lengths = [e.length for e in self.filtered_entries]
                logger.info(
                    f"Length distribution: min={min(lengths)}, "
                    f"max={max(lengths)}, median={np.median(lengths):.0f}"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize transcript pool: {e}")
            raise
    
    def _find_index_file(self, fasta_file: str) -> Optional[Path]:
        """Find the appropriate index file for a FASTA file."""
        fasta_path = Path(fasta_file)
        
        # Try different index file naming conventions
        possible_indices = [
            Path(str(fasta_file) + '.fai'),  # file.fa.gz.fai
            fasta_path.with_suffix('.fai'),  # file.fai
            fasta_path.parent / (fasta_path.stem + '.fai')  # file.fa.fai (for .fa.gz)
        ]
        
        for idx_path in possible_indices:
            if idx_path.exists():
                logger.info(f"Found index file: {idx_path}")
                return idx_path
        
        return None
    
    def _load_index(self, fai_file: Path) -> None:
        """Load and parse the FASTA index file with error handling."""
        try:
            with open(fai_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = FastaIndexEntry.from_fai_line(line)
                        self.index_entries.append(entry)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid index line {line_num}: {e}")
                        self.stats['corrupted_entries'] += 1
            
            if not self.index_entries:
                raise ValueError("No valid entries found in index file")
            
            logger.info(f"Loaded {len(self.index_entries)} valid index entries")
            
        except Exception as e:
            logger.error(f"Failed to load FASTA index: {e}")
            raise
    def _reload_index_from(self, fai_file: Path) -> None:
        """
        Reload .fai and re-apply filters after the FASTA path changes.
        """
        self.index_entries = []
        self.filtered_entries = []
        self._load_index(fai_file)
        self._validate_index()
        self._filter_transcripts()

    def _validate_index(self):
        """Validate index entries for consistency."""
        logger.debug("Validating index entries...")
        
        # Check for duplicate names
        names = [e.name for e in self.index_entries]
        unique_names = set(names)
        if len(names) != len(unique_names):
            logger.warning(
                f"Found {len(names) - len(unique_names)} duplicate sequence names in index"
            )
        
        # Validate reasonable values
        for entry in self.index_entries:
            if entry.length <= 0:
                logger.warning(f"Invalid length for {entry.name}: {entry.length}")
                self.stats['validation_errors'] += 1
            if entry.line_bases <= 0 or entry.line_width <= entry.line_bases:
                logger.warning(f"Invalid line format for {entry.name}")
                self.stats['validation_errors'] += 1
    
    def _filter_transcripts(self) -> None:
        """Filter transcripts based on length criteria with progress bar."""
        min_length = self.config.get("min_length", 200)
        max_length = self.config.get("max_length", 5000)
        
        logger.info(f"Filtering transcripts to range [{min_length}, {max_length}]...")
        
        self.filtered_entries = [e for e in self.index_entries if min_length <= e.length <= max_length]
        
        # Compute length-based sampling weights
        if self.filtered_entries:
            lens = np.array([e.length for e in self.filtered_entries], dtype=np.float64)
            self._length_weights = (lens / lens.sum())
        
        logger.info(
            f"Retained {len(self.filtered_entries)}/{len(self.index_entries)} transcripts "
            f"({len(self.filtered_entries)/len(self.index_entries)*100:.1f}%)"
        )
        if not self.filtered_entries:
            logger.warning("Filter removed all transcripts; check min/max length settings.")
    
    def _open_fasta_handle(self) -> None:
        """Open the FASTA file for reading with appropriate method."""
        try:
            if self.is_gzipped and self.config.get("decompress_to_tmp", False):
                import shutil
                import tempfile

                tmpdir = Path(self.config.get("tmpdir", tempfile.gettempdir()))
                tmpdir.mkdir(parents=True, exist_ok=True)

                # Create stable temp filename based on source file hash
                source_hash = hashlib.md5(str(self.fasta_file).encode()).hexdigest()[:8]
                tmp_name = f"{Path(self.fasta_file).stem}_{source_hash}.fa"
                tmp_path = tmpdir / tmp_name

                # Check if we need to decompress (file missing or outdated)
                source_mtime = Path(self.fasta_file).stat().st_mtime
                needs_decompress = (
                    not tmp_path.exists()
                    or tmp_path.stat().st_mtime < source_mtime
                )

                lockfile = tmp_path.with_suffix(tmp_path.suffix + ".lock")

                if needs_decompress:
                    # Attempt to acquire the lock atomically
                    try:
                        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        have_lock = True
                        os.close(fd)
                    except FileExistsError:
                        have_lock = False

                    if have_lock:
                        logger.info(f"[LOCK ACQUIRED] Decompressing {self.fasta_file} → {tmp_path}")
                        logger.info("This is a one-time operation that will speed up all subsequent access")

                        # Write into a temporary file, then atomically rename
                        tmp_partial = tmp_path.with_suffix(tmp_path.suffix + ".partial")

                        try:
                            with gzip.open(self.fasta_file, "rb") as src, open(tmp_partial, "wb") as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)

                            # Atomic rename overwrites safely
                            os.replace(tmp_partial, tmp_path)

                            logger.info(f"Decompression complete: {tmp_path.stat().st_size / 1e9:.1f} GB")

                        finally:
                            # Release lock
                            try:
                                os.unlink(lockfile)
                            except FileNotFoundError:
                                pass

                    else:
                        # Another process is decompressing - wait with timeout
                        logger.info(f"Waiting for decompression lock for {tmp_path}...")
                        timeout = 300  # 5 minutes
                        start_time = time.time()
                        while lockfile.exists():
                            if time.time() - start_time > timeout:
                                logger.warning(
                                    f"Lock timeout for {tmp_path}, removing stale lock"
                                )
                                try:
                                    os.unlink(lockfile)
                                except FileNotFoundError:
                                    pass
                                break
                            time.sleep(0.2)

                else:
                    logger.info(f"Using existing decompressed file: {tmp_path}")

                # Only switch to the decompressed file if we have a matching index for it
                tmp_fai = self._find_index_file(str(tmp_path))
                if tmp_fai and tmp_fai.exists():
                    self.fasta_file = tmp_path
                    self.is_gzipped = False  # Treat as uncompressed from here

                    # Reload the index for the decompressed file so offsets match
                    self._reload_index_from(tmp_fai)

                    # Open file handle and mmap for validation
                    self.fasta_handle = open(self.fasta_file, 'rb')
                    self.mmap_handle = mmap.mmap(self.fasta_handle.fileno(), 0, access=mmap.ACCESS_READ)

                    # Safety check: Validate that decompressed FASTA offsets point to headers
                    for e in self.filtered_entries[:5]:
                        try:
                            self.mmap_handle.seek(e.offset)
                            first_char = self.mmap_handle.read(1)
                            if first_char != b'>':
                                logger.warning(
                                    f"Index offset mismatch for {e.name} in decompressed FASTA: "
                                    f"expected '>' at offset {e.offset}"
                                )
                        except Exception:
                            pass

                    logger.info("Memory-mapped decompressed FASTA (with matching index)")

                else:
                    logger.warning("No .fai found for decompressed FASTA; keeping gz streaming to avoid offset mismatches.")
                    self.gzip_reader = GzippedFastaReader(self.fasta_file, self.filtered_entries)
                    return

            elif self.is_gzipped:
                # Keep the streaming reader as fallback
                self.gzip_reader = GzippedFastaReader(
                    self.fasta_file,
                    self.filtered_entries
                )
                logger.info("Opened gzipped FASTA with streaming reader")
                logger.info("Tip: Set config['decompress_to_tmp'] = True for better performance")

            else:
                # Use memory mapping for uncompressed files
                self.fasta_handle = open(self.fasta_file, 'rb')
                self.mmap_handle = mmap.mmap(
                    self.fasta_handle.fileno(), 0,
                    access=mmap.ACCESS_READ
                )
                logger.info("Memory-mapped uncompressed FASTA file")

        except Exception as e:
            logger.error(f"Failed to open FASTA file: {e}")
            raise
    
    def _read_sequence(self, entry: FastaIndexEntry) -> str:
        """Read a transcript safely (never returns None)."""

        cached = self.cache.get(entry.name)
        if cached is not None:
            return cached

        try:
            if self.is_gzipped:
                sequence = self._read_sequence_gzipped(entry)
            else:
                sequence = self._read_sequence_uncompressed(entry)

            if not sequence or not isinstance(sequence, str):
                raise ValueError("Transcript read returned None or invalid type")

            if not self._validate_sequence(sequence, entry):
                logger.warning(f"Sequence validation FAILED for {entry.name}")
                self.stats['validation_errors'] += 1

            # cache & stats
            self.cache.put(entry.name, sequence)
            self.stats['sequences_read'] += 1
            self.stats['total_bases_processed'] += len(sequence)
            return sequence

        except Exception as e:
            logger.error(f"Error reading transcript {entry.name}: {e}")
            self.stats['corrupted_entries'] += 1

            safe_len = entry.length if isinstance(entry.length, int) and entry.length > 0 else 200
            return self._generate_random_sequence(safe_len, np.random.RandomState())
    
    def _read_sequence_gzipped(self, entry: FastaIndexEntry) -> Optional[str]:
        """Read sequence from gzipped file using persistent reader."""
        if self.gzip_reader is None:
            raise ValueError("Gzipped reader not initialized")
        
        sequence = self.gzip_reader.read_sequence(entry.name)
        
        if sequence and self.config.get("trim_polya", False):
            sequence = self._trim_polya(sequence)
        
        return sequence
    
    def _read_sequence_uncompressed(self, entry: FastaIndexEntry) -> str:
        """Read sequence from uncompressed FASTA using mmap."""
        if self.mmap_handle is None:
            raise ValueError("Memory-mapped handle not initialized")
        
        try:
            # Calculate bytes to read (simplified formula that always works)
            full_lines = entry.length // entry.line_bases
            remaining_bases = entry.length % entry.line_bases
            
            if remaining_bases == 0:
                total_bytes = full_lines * entry.line_width
            else:
                total_bytes = full_lines * entry.line_width + min(remaining_bases, entry.line_bases)
            
            # Read from memory-mapped file
            self.mmap_handle.seek(entry.offset)
            raw_sequence = self.mmap_handle.read(total_bytes).decode('ascii', errors="ignore")
            
            # Clean and validate sequence
            sequence = raw_sequence.replace('\n', '').replace('\r', '').upper()
            
            # Validate only expected bases
            valid_bases = set('ACGTN')
            cleaned_sequence = ''.join(b if b in valid_bases else 'N' for b in sequence)
            
            if len(cleaned_sequence) != len(sequence):
                logger.debug(f"Replaced {len(sequence) - len(cleaned_sequence)} invalid bases with N")
            
            # Trim polyA if configured
            if self.config.get("trim_polya", False):
                cleaned_sequence = self._trim_polya(cleaned_sequence)
            
            return cleaned_sequence
            
        except Exception as e:
            logger.error(f"Error reading uncompressed sequence: {e}")
            safe_len = entry.length if isinstance(entry.length, int) and entry.length > 0 else 200
            return self._generate_random_sequence(safe_len, np.random.RandomState())
    
    def _validate_sequence(self, sequence: str, entry: FastaIndexEntry) -> bool:
        """
        Validate that a sequence matches its index entry.
        
        Args:
            sequence: The sequence string
            entry: The index entry
            
        Returns:
            True if valid, False otherwise
        """
        # Check length
        if len(sequence) != entry.length:
            logger.warning(
                f"Length mismatch for {entry.name}: "
                f"index says {entry.length}, got {len(sequence)}"
            )
            return False
        
        # Check for excessive Ns or invalid characters
        n_count = sequence.count('N')
        if n_count / len(sequence) > 0.5:
            logger.warning(f"Sequence {entry.name} is >50% Ns")
            return False
        
        # Check for valid bases
        valid_bases = set('ACGTN')
        invalid_bases = set(sequence) - valid_bases
        if invalid_bases:
            logger.warning(
                f"Sequence {entry.name} contains invalid bases: {invalid_bases}"
            )
            return False
        
        return True
    
    def _trim_polya(self, sequence: str, min_polya_length: int = 10, max_lookback: int = 2000) -> str:
        """
        Trim polyA tail from sequence.
        
        Args:
            sequence: Input sequence
            min_polya_length: Minimum length to consider as polyA tail
            max_lookback: Maximum bases to scan from end (prevents pathological cases)
        
        Returns:
            Sequence with polyA tail trimmed
        """
        # Count trailing As (with allowance for Ns), but limit lookback
        a_count = 0
        lookback_limit = min(len(sequence), max_lookback)
        for i in range(len(sequence) - 1, len(sequence) - lookback_limit - 1, -1):
            if sequence[i] == 'A':
                a_count += 1
            elif sequence[i] != 'N':  # Allow Ns in polyA
                break
        
        # Trim if polyA is long enough
        if a_count >= min_polya_length:
            return sequence[:-a_count]
        
        return sequence
    
    def sample_fragment(self, random_state: np.random.RandomState) -> str:
        if not self.filtered_entries:
            return self._generate_random_fallback(random_state)

        entry = random_state.choice(self.filtered_entries)
        transcript = self._read_sequence(entry)

        # generate fragment
        fragment_min = self.config.get("fragment_min", 200)
        fragment_max = self.config.get("fragment_max", 1000)
        max_possible = min(len(transcript), fragment_max)

        if max_possible < fragment_min:
            fragment = transcript
            fragment_len = len(fragment)
            start_pos = 0
        else:
            fragment_len = random_state.randint(fragment_min, max_possible + 1)
            max_start = len(transcript) - fragment_len
            start_pos = random_state.randint(0, max_start + 1) if max_start > 0 else 0

        fragment = transcript[start_pos:start_pos + fragment_len]

        rc_flag = False
        if random_state.random() < self.config.get("reverse_complement_prob", 0.5):
            fragment = reverse_complement(fragment)
            rc_flag = True

        # now metadata is valid
        self.last_sampled_entry = entry
        self.last_fragment_start = start_pos
        self.last_fragment_end = start_pos + fragment_len
        self.last_fragment_rc = rc_flag

        return fragment
    
    def sample_batch(self, batch_size: int, random_state: np.random.RandomState,
                    show_progress: bool = True) -> List[str]:
        """
        Sample multiple fragments efficiently with optional progress bar.
        
        Args:
            batch_size: Number of fragments to sample
            random_state: Random state for reproducibility
            show_progress: Whether to show progress bar
            
        Returns:
            List of fragment sequences
        """
        if not self.filtered_entries:
            return [self._generate_random_fallback(random_state) for _ in range(batch_size)]
        
        # Pre-select transcripts for better cache utilization
        # Use length-weighted sampling if weights are available
        if getattr(self, "_length_weights", None) is not None:
            selected_entries = random_state.choice(
                self.filtered_entries, size=batch_size, replace=True, p=self._length_weights
            )
        else:
            selected_entries = random_state.choice(
                self.filtered_entries, size=batch_size, replace=True
            )
        
        # No tqdm: rely on external Rich progress logging
        
        fragments = []
        for entry in selected_entries:
            transcript = self._read_sequence(entry)
            
            # Generate fragment
            fragment_min = self.config.get("fragment_min", 200)
            fragment_max = self.config.get("fragment_max", 1000)
            max_possible = min(len(transcript), fragment_max)
            
            if max_possible < fragment_min:
                fragment = transcript
                fragment_len = len(fragment)
                start_pos = 0
            else:
                fragment_len = random_state.randint(fragment_min, max_possible + 1)
                max_start = len(transcript) - fragment_len
                start_pos = random_state.randint(0, max_start + 1) if max_start > 0 else 0
                fragment = transcript[start_pos : start_pos + fragment_len]
            
            # Reverse complement with probability
            if random_state.random() < self.config.get("reverse_complement_prob", 0.5):
                fragment = reverse_complement(fragment)
            
            fragments.append(fragment)
        
        return fragments
    
    def _generate_random_sequence(
        self,
        length: Optional[int],
        random_state: np.random.RandomState
        ) -> str:
        """
        Generate a GC-aware random cDNA sequence.
        SAFETY: length may be None or invalid.
        """
        if not isinstance(length, int) or length <= 0:
            logger.warning(
                f"[SAFETY] Invalid length={length}; falling back to 200bp random sequence"
            )
            length = 200

        gc = float(self.config.get("fallback_gc_content", 0.5))
        vals = random_state.random(length)
        arr = np.empty(length, dtype='S1')

        arr[vals < gc / 2] = b"G"
        arr[(vals >= gc / 2) & (vals < gc)] = b"C"
        arr[(vals >= gc) & (vals < (1 + gc) / 2)] = b"A"
        arr[vals >= (1 + gc) / 2] = b"T"

        return arr.tobytes().decode("ascii")
    
    def _generate_random_fallback(
        self,
        random_state: np.random.RandomState
    ) -> str:
        """
        Decide fallback length (min/max) and generate sequence.
        """
        if self.config.get("fallback_mode", "random") != "random":
            raise ValueError("No transcripts loaded and no fallback configured")

        length = random_state.randint(
            self.config.get("fragment_min", 200),
            self.config.get("fragment_max", 800),
        )

        return self._generate_random_sequence(length, random_state)
    
    def _load_transcripts_fallback(self, fasta_file: str):
        """Fallback method: Load transcripts without index (slower)."""
        self.transcripts = []
        self.transcript_ids = []
        
        logger.warning("Using fallback loading method (this may be slow)...")
        
        try:
            # Handle both gzipped and uncompressed
            if str(fasta_file).endswith('.gz'):
                handle = gzip.open(fasta_file, 'rt')
            else:
                handle = open(fasta_file, 'r')
            
            seq_id = None
            seq_parts = []
            sequences_loaded = 0
            
            # No tqdm - rely on Rich logging
            
            for line in handle:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if seq_id and seq_parts:
                        seq = "".join(seq_parts).upper()
                        seq_len = len(seq)
                        min_len = self.config.get("min_length", 100)
                        max_len = self.config.get("max_length", 5000)
                        
                        if min_len <= seq_len <= max_len:
                            self.transcripts.append(seq)
                            self.transcript_ids.append(seq_id)
                            sequences_loaded += 1
                    
                    # Start new sequence
                    seq_id = line[1:].split()[0]
                    seq_parts = []
                elif seq_id:  # Only accumulate if we have a header
                    seq_parts.append(line)
            
            # Don't forget last sequence
            if seq_id and seq_parts:
                seq = "".join(seq_parts).upper()
                seq_len = len(seq)
                min_len = self.config.get("min_length", 100)
                max_len = self.config.get("max_length", 5000)
                
                if min_len <= seq_len <= max_len:
                    self.transcripts.append(seq)
                    self.transcript_ids.append(seq_id)
                    sequences_loaded += 1
            
            handle.close()
            
            if self.transcripts:
                logger.info(f"Loaded {len(self.transcripts)} transcripts (fallback method)")
                lengths = [len(t) for t in self.transcripts]
                logger.info(f"Length range: {min(lengths)}-{max(lengths)} bp")
            else:
                logger.warning("No transcripts loaded from file")
                
        except Exception as e:
            logger.error(f"Failed to load transcripts: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the transcript pool."""
        stats = {
            'index_entries': len(self.index_entries) if hasattr(self, 'index_entries') else 0,
            'filtered_entries': len(self.filtered_entries) if hasattr(self, 'filtered_entries') else 0,
            'is_gzipped': self.is_gzipped,
            'fasta_file': str(self.fasta_file) if self.fasta_file else None,
            'runtime_stats': self.stats.copy()
        }
        
        # Add cache statistics
        # if hasattr(self, 'cache'):
            # stats['cache'] = self.cache.get_stats()
        
        # Add length distribution for filtered entries
        if self.filtered_entries:
            lengths = [e.length for e in self.filtered_entries]
            stats['length_distribution'] = {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths)
            }
        
        # Add fallback statistics if using in-memory transcripts
        if hasattr(self, 'transcripts') and self.transcripts:
            stats['fallback_mode'] = True
            stats['loaded_transcripts'] = len(self.transcripts)
            lengths = [len(t) for t in self.transcripts]
            stats['fallback_length_distribution'] = {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths)
            }
        
        return stats
    
    def close(self):
        """Close file handles and clean up resources."""
        errors = []
        
        if self.gzip_reader:
            try:
                self.gzip_reader.close()
            except Exception as e:
                errors.append(f"gzip_reader: {e}")
            finally:
                self.gzip_reader = None
        
        if self.mmap_handle:
            try:
                self.mmap_handle.close()
            except Exception as e:
                errors.append(f"mmap_handle: {e}")
            finally:
                self.mmap_handle = None
        
        if self.fasta_handle:
            try:
                self.fasta_handle.close()
            except Exception as e:
                errors.append(f"fasta_handle: {e}")
            finally:
                self.fasta_handle = None
        
        if errors:
            logger.warning(f"Errors during close: {'; '.join(errors)}")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info("TranscriptPool closing with statistics:")
        logger.info(f"  Sequences read: {self.stats['sequences_read']}")
        logger.info(f"  Total bases: {self.stats['total_bases_processed'] / 1e6:.1f} Mb")
        # if 'cache' in stats:
            # cache_stats = stats['cache']
            # logger.info(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
            # logger.info(f"  Cache size: {cache_stats['current_size_mb']:.1f} MB")
            # logger.info(f"  Evictions: {cache_stats['evictions']}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @contextmanager
    def batch_mode(self):
        """
        Context manager for efficient batch processing.
        Temporarily increases cache size and optimizes settings.
        """
        original_max_size = self.cache.max_size_bytes
        try:
            # Temporarily increase cache for batch processing
            self.cache.max_size_bytes *= 2
            logger.debug("Entered batch mode - doubled cache size")
            yield self
        finally:
            # Restore original settings
            self.cache.max_size_bytes = original_max_size
            logger.debug("Exited batch mode - restored cache size")

class PolyATailGenerator:
    """Generates polyA tails with realistic length distributions."""

    def __init__(self, config: Dict):
        """
        Initialize polyA tail generator.

        Args:
            config: PolyA configuration dictionary
        """
        self.config = config
        self.distribution = config.get("distribution", "normal")
        self.empirical_lengths = None

        # Load empirical distribution if provided
        empirical_file = config.get("empirical_file")
        if empirical_file and Path(empirical_file).exists():
            self._load_empirical_distribution(empirical_file)

    def _load_empirical_distribution(self, filepath: str):
        """Load empirical polyA length distribution."""
        try:
            lengths = []
            with open(filepath, "r") as f:
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
            if self.distribution == "normal":
                mean = self.config.get("mean_length", 150)
                std = self.config.get("std_length", 50)
                length = int(random_state.normal(mean, std))
            elif self.distribution == "gamma":
                shape = self.config.get("shape", 2.0)
                scale = self.config.get("scale", 75.0)
                length = int(random_state.gamma(shape, scale))
            elif self.distribution == "uniform":
                min_len = self.config.get("min_length", 50)
                max_len = self.config.get("max_length", 250)
                length = random_state.randint(min_len, max_len + 1)
            else:
                # Fixed length
                length = self.config.get("fixed_length", 100)

        # Ensure reasonable bounds
        min_len = self.config.get("absolute_min", 20)
        max_len = self.config.get("absolute_max", 500)
        length = max(min_len, min(length, max_len))

        # Generate tail with optional impurities
        purity = self.config.get("purity", 0.95)
        tail = []
        for _ in range(length):
            if random_state.random() < purity:
                tail.append("A")
            else:
                # Occasional non-A base
                tail.append(random_state.choice(["C", "G", "T"]))

        return "".join(tail)


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
        whitelist_files = config.get("whitelist_files", {})
        for segment, filepath in whitelist_files.items():
            self._load_whitelist(segment, filepath)

    def _load_whitelist(self, segment: str, filepath: str):
        """Load a whitelist from file."""
        if not Path(filepath).exists():
            logger.warning(f"Whitelist file not found for {segment}: {filepath}")
            return

        # Valid IUPAC codes that can be resolved to actual bases
        valid_iupac = set('ACGTNRYSWKMBDHV')
        
        sequences = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Handle tab-separated format (ID<TAB>SEQUENCE)
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        seq = parts[1].strip().upper()
                    else:
                        continue
                else:
                    # Simple format with just sequences
                    seq = line.upper()
                
                # Validate that sequence contains only valid IUPAC codes
                if seq and all(base in valid_iupac for base in seq):
                    sequences.append(seq)
                else:
                    invalid_chars = set(seq) - valid_iupac
                    logger.warning(
                        f"Skipping invalid sequence in {segment} "
                        f"(contains {invalid_chars}): {seq[:20] if len(seq) > 20 else seq}..."
                    )

        if sequences:
            self.whitelists[segment] = sequences
            self.usage_stats[segment] = {"loaded": len(sequences), "used": 0}
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
        self.usage_stats[segment]["used"] += 1
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
        self.substitution_rate = config.get("substitution_rate", 0.001)
        self.insertion_rate = config.get("insertion_rate", 0.0001)
        self.deletion_rate = config.get("deletion_rate", 0.0001)
        self.quality_dependent = config.get("quality_dependent", False)

    def introduce_errors(
        self,
        sequence: List[str],
        labels: List[str],
        quality_scores: Optional[np.ndarray],
        random_state: np.random.RandomState,
    ) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
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
        if not any(
            [self.substitution_rate, self.insertion_rate, self.deletion_rate]
        ):
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
                error_prob = (
                    self.substitution_rate
                    + self.insertion_rate
                    + self.deletion_rate
                )

            if random_state.random() < error_prob:
                # Decide on error type
                rates = np.array(
                    [
                        self.substitution_rate,
                        self.insertion_rate,
                        self.deletion_rate,
                    ],
                    dtype=float,
                )
                total = rates.sum()
                # Invariant: at least one rate must be > 0 due to guard at start of method
                assert total > 0, "Error rates sum should be > 0 due to earlier guard"

                probs = rates / total
                error_type = random_state.choice(
                    ["sub", "ins", "del"], p=probs
                )

                if error_type == "sub":
                    # Substitution
                    original = sequence[i]
                    bases = ["A", "C", "G", "T"]
                    bases.remove(original if original in bases else "A")
                    new_seq.append(random_state.choice(bases))
                    new_labels.append("ERROR")
                    if new_qual is not None:
                        new_qual.append(quality_scores[i] * 0.5)  # Reduce quality
                    i += 1

                elif error_type == "ins":
                    # Insertion: add extra base without consuming input
                    # Keep current base
                    new_seq.append(sequence[i])
                    new_labels.append(labels[i])
                    if new_qual is not None:
                        new_qual.append(quality_scores[i])
                    # Add the inserted base after it
                    new_seq.append(random_state.choice(["A", "C", "G", "T"]))
                    new_labels.append("ERROR")
                    if new_qual is not None:
                        new_qual.append(20)  # Low quality for insertion
                    i += 1

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
    
    Note: Currently uses np.random.RandomState for backward compatibility.
    Consider migrating to np.random.Generator for improved performance and 
    reproducibility in future versions.
    """

    def __init__(
        self, config: Optional[Dict] = None, config_file: Optional[str] = None
    ):
        """
        Initialize the read simulator.

        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        # Load configuration
        if config_file:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}

        self.sim_config = self.config.get("simulation", {})

        # Initialize random state
        seed = self.sim_config.get("random_seed", 42)
        self.random_state = np.random.RandomState(seed)
        random.seed(seed)

        # Initialize components
        self.acc_generator = self._initialize_acc_generator()
        self.whitelist_manager = WhitelistManager(self.sim_config)
        self.transcript_pool = self._initialize_transcript_pool()
        self.polya_generator = self._initialize_polya_generator()
        self.error_simulator = self._initialize_error_simulator()

        # Get sequence architecture
        self.sequence_order = self.sim_config.get("sequence_order", [])
        self.sequence_configs = self.sim_config.get("sequences", {})

        logger.info("Initialized comprehensive sequence simulator")
        logger.info(f"Architecture: {' → '.join(self.sequence_order)}")

    def _initialize_acc_generator(
        self,
    ) -> Optional["ProbabilisticPWMGenerator"]:
        """Initialize the probabilistic ACC generator from configuration."""
        if not PWM_AVAILABLE:
            return None

        # Handle both top-level and nested PWM configs safely
        pwm_config = self.config.get("pwm") or self.sim_config.get("pwm") or {}
        if pwm_config is None:
            pwm_config = {}

        pwm_file = pwm_config.get("pwm_file")

        # REALLY important for reproducibility and testing
        random_seed = pwm_config.get("random_seed", self.sim_config.get("random_seed", 42))

        # Check for PWM in different config locations
        if not pwm_file:
            pwm_files = self.sim_config.get("pwm_files", {})
            pwm_file = pwm_files.get("ACC")

        if pwm_file and Path(pwm_file).exists():
            try:
                # Use the unified loader from tempest.utils.io
                pwm = io.load_pwm(pwm_file)

                # Get temperature setting (controls diversity)
                temperature = pwm_config.get("temperature", 1.0)

                # If old threshold config exists, map it to temperature
                if "threshold" in pwm_config and "temperature" not in pwm_config:
                    threshold = pwm_config.get("threshold", 0.8)
                    # Map threshold [0.5, 1.0] to temperature [2.0, 0.5]
                    temperature = 2.5 - 2.0 * threshold
                    logger.info(
                        f"Mapped threshold {threshold} to temperature {temperature}"
                    )

                min_entropy = pwm_config.get("min_entropy", 0.1)

                # IMPORTANT: pass random_seed through for reproducibility
                generator = ProbabilisticPWMGenerator(
                    pwm=pwm,
                    temperature=temperature,
                    min_entropy=min_entropy,
                    random_seed=random_seed,
                )

                logger.info(
                    f"Initialized ACC PWM generator with "
                    f"temperature={temperature}, random_seed={random_seed}"
                )
                return generator

            except Exception as e:
                logger.error(f"Failed to load PWM from {pwm_file}: {e}")

        # Fallback: Create PWM from IUPAC pattern if specified
        acc_pattern = pwm_config.get("pattern", "ACCSSV")
        if acc_pattern and PWM_AVAILABLE:
            logger.info(f"Creating ACC PWM from pattern: {acc_pattern}")
            pwm = create_acc_pwm_from_pattern(acc_pattern)
            temperature = pwm_config.get("temperature", 1.0)
            min_entropy = pwm_config.get("min_entropy", 0.1)
            random_seed = pwm_config.get("random_seed", None)
            return ProbabilisticPWMGenerator(
                pwm=pwm,
                temperature=temperature,
                min_entropy=min_entropy,
                random_seed=random_seed,
            )

        return None

    def _initialize_transcript_pool(self) -> Optional[TranscriptPool]:
        """Initialize transcript pool if configured."""
        # Try 'transcript_pool' first (new name), then 'transcript' (config.yaml name)
        transcript_config = self.sim_config.get("transcript_pool") or self.sim_config.get("transcript")
        
        # Only initialize if we have a valid config with fasta_file
        if transcript_config:
            fasta_file = transcript_config.get("fasta_file")
            if fasta_file and fasta_file != "":
                return TranscriptPool(transcript_config)
            else:
                logger.info("Transcript config found but no fasta_file specified, skipping transcript loading")
        return None

    def _initialize_polya_generator(self) -> Optional[PolyATailGenerator]:
        """Initialize polyA tail generator if configured."""
        polya_config = self.sim_config.get("polya", {})
        if polya_config:
            return PolyATailGenerator(polya_config)
        return None

    def _initialize_error_simulator(self) -> Optional[ErrorSimulator]:
        """Initialize error simulator if configured."""
        error_config = self.sim_config.get("errors", {})
        if error_config.get("enabled", False):
            return ErrorSimulator(error_config)
        return None

    def generate_read(
        self,
        diversity_boost: Optional[float] = None,
        include_quality: bool = False,
        inject_errors: bool = True,
    ) -> SimulatedRead:
        """
        Generate a single simulated read with full metadata.

        Args:
            diversity_boost: Optional multiplier for ACC diversity
            include_quality: Whether to include quality scores
            inject_errors: Whether to inject sequencing errors

        Returns:
            SimulatedRead object
        """

        # --- initialized accumulators ---
        full_sequence = []
        labels = []
        label_regions: Dict[str, List[Tuple[int, int]]] = {}

        # unified metadata collectors
        segment_sources = {}
        segment_lengths = {}
        segment_sequences = {}
        segment_meta_list = {}
        segment_order = list(self.sequence_order)
        transcript_info = None

        quality_scores = [] if include_quality else None

        # build segments
        for segment_name in self.sequence_order:

            (
                segment_seq,
                segment_qual,
                source,
                seg_meta,
            ) = self._generate_segment_with_source(
                segment_name, diversity_boost, include_quality
            )

            seg_len = len(segment_seq)

            # Pull in metadata
            segment_sources.setdefault(segment_name, []).append(source)
            segment_lengths.setdefault(segment_name, []).append(seg_len)
            segment_sequences.setdefault(segment_name, []).append(seg_meta["sequence"])
            segment_meta_list.setdefault(segment_name, []).append(seg_meta)

            # Extract transcript fragment metadata if present
            if (
                transcript_info is None
                and "transcript_id" in seg_meta
                and seg_meta["transcript_id"] is not None
            ):
                transcript_info = {
                    "id": seg_meta["transcript_id"],
                    "length": seg_meta["transcript_length"],
                    "fragment_start": seg_meta["fragment_start"],
                    "fragment_end": seg_meta["fragment_end"],
                    "reverse_complemented": seg_meta["reverse_complemented"],
                }

            start = len(full_sequence)
            end = start + seg_len

            full_sequence.extend(segment_seq)
            labels.extend([segment_name] * seg_len)

            label_regions.setdefault(segment_name, []).append((start, end))

            # Quality scores
            if include_quality:
                if segment_qual is not None:
                    if isinstance(segment_qual, (list, tuple)):
                        quality_scores.extend(segment_qual)
                    else:
                        quality_scores.extend([segment_qual] * seg_len)

        # convert qual scores
        if include_quality:
            if not quality_scores:
                quality_scores = np.full(
                    len(full_sequence), 30.0, dtype=np.float32
                )
            else:
                quality_scores = np.array(quality_scores, dtype=np.float32)

        # error injection
        has_errors = False
        if inject_errors and self.error_simulator:
            if self.random_state.random() < self.sim_config.get(
                "error_injection_prob", 0.1
            ):
                full_sequence, labels, quality_scores = self.error_simulator.introduce_errors(
                    full_sequence, labels, quality_scores, self.random_state
                )
                has_errors = True

        # invariant labels
        if len(labels) != len(full_sequence):
            if len(labels) < len(full_sequence):
                labels.extend(["UNKNOWN"] * (len(full_sequence) - len(labels)))
            else:
                labels = labels[:len(full_sequence)]

        # pull back out the labels after segment generation
        label_regions = self._regions_from_labels(labels, exclude={"ERROR"})

        # if full read rc
        if self.random_state.random() < self.sim_config.get(
            "full_read_reverse_complement_prob", 0.0
        ):
            full_sequence = list(reverse_complement(full_sequence))
            labels = labels[::-1]
            if quality_scores is not None:
                quality_scores = quality_scores[::-1]

            label_regions = self._regions_from_labels(labels, exclude={"ERROR"})

        # metadata block
        metadata = {
            "segment_order": segment_order,
            "segment_sources": segment_sources,
            "segment_lengths": segment_lengths,
            "segment_sequences": segment_sequences,
            "segment_meta": segment_meta_list,
            "transcript_info": transcript_info,
            "has_errors": has_errors,
            "diversity_boost": diversity_boost,
            "is_invalid": False,
        }

        # Return SimulatedRead
        return SimulatedRead(
            sequence="".join(full_sequence),
            labels=labels,
            label_regions=label_regions,
            metadata=metadata,
            quality_scores=quality_scores,
        )
    def _get_seq_cfg(self, key: str) -> Optional[str]:
        # Case-insensitive lookup for exact segment configs
        return (self.sequence_configs.get(key) or
                self.sequence_configs.get(key.upper()) or
                self.sequence_configs.get(key.lower()))
    
    def _generate_segment_with_source(
    self,
    segment_name: str,
    diversity_boost: Optional[float],
    include_quality: bool,
    ) -> Tuple[str, Optional[Union[List[float], float]], str, Dict]:
        """
        Generate a segment and return:
            (sequence, quality_scores, source, segment_meta)

        segment_meta is a dict:
            {
                "sequence": str,
                "length": int,
                "source": str,
                ... optional transcript metadata ...
            }
        """
        segment_upper = segment_name.upper()

        # acc
        if segment_name == "ACC" and self.acc_generator:
            seq, qual = self._generate_acc_segment(diversity_boost, include_quality)
            meta = {
                "sequence": seq,
                "length": len(seq),
                "source": "pwm"
            }
            return seq, qual, "pwm", meta

        # polyA
        if "POLYA" in segment_upper and self.polya_generator:
            seq = self.polya_generator.generate(self.random_state)
            qual = 25.0 if include_quality else None
            meta = {
                "sequence": seq,
                "length": len(seq),
                "source": "polya_generator"
            }
            return seq, qual, "polya_generator", meta

        # txp derived insert
        if (
            "INSERT" in segment_upper
            or "CDNA" in segment_upper
            or "TRANSCRIPT" in segment_upper
        ):
            if self.transcript_pool:

                # Sample cDNA fragment
                seq = self.transcript_pool.sample_fragment(self.random_state)
                qual = 30.0 if include_quality else None

                # Grab transcript info if available
                tx_entry = getattr(self.transcript_pool, "last_sampled_entry", None)
                fragment_start = getattr(self.transcript_pool, "last_fragment_start", None)
                fragment_end = getattr(self.transcript_pool, "last_fragment_end", None)

                meta = {
                    "sequence": seq,
                    "length": len(seq),
                    "source": "transcript_pool",
                    "transcript_id": tx_entry.name if tx_entry else None,
                    "transcript_length": tx_entry.length if tx_entry else None,
                    "fragment_start": fragment_start,
                    "fragment_end": fragment_end,
                    "reverse_complemented": getattr(self.transcript_pool, "last_fragment_rc", False)
                }

                return seq, qual, "transcript_pool", meta

        # whitelist manager
        if self.whitelist_manager.has_whitelist(segment_name):
            seq = self.whitelist_manager.sample(segment_name, self.random_state)
            seq = self._resolve_iupac_codes(seq)
            qual = 35.0 if include_quality else None
            meta = {
                "sequence": seq,
                "length": len(seq),
                "source": "whitelist"
            }
            return seq, qual, "whitelist", meta

        # fixed seqs
        cfg_val = self._get_seq_cfg(segment_name)
        if cfg_val is not None:
            seq = cfg_val
            if seq not in ["random", "transcript", "polya"]:
                seq = self._resolve_iupac_codes(seq)
                qual = 40.0 if include_quality else None
                meta = {
                    "sequence": seq,
                    "length": len(seq),
                    "source": "fixed"
                }
                return seq, qual, "fixed", meta

        # random seqs
        seq = self._generate_segment(segment_name)
        qual = 25.0 if include_quality else None
        meta = {
            "sequence": seq,
            "length": len(seq),
            "source": "random"
        }
        return seq, qual, "random", meta

    def _generate_acc_segment(
        self,
        diversity_boost: Optional[float] = None,
        include_quality: bool = False,
    ) -> Tuple[str, Optional[Union[List[float], float]]]:
        """Generate ACC segment using PWM generator or fallback."""
        # Check for fixed ACC in config
        acc_config = self._get_seq_cfg("ACC") or ""
        if acc_config and acc_config not in ["random", "pwm"]:
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
                n=1, diversity_boost=diversity_boost
            )
            return sequences[0], None
        else:
            # Fallback to random ACC-like sequence
            bases = []
            for i in range(6):  # Default ACC length
                if i < 3:
                    bases.append("ACC"[i])
                else:
                    bases.append(
                        self.random_state.choice(["C", "G", "T", "A"])
                    )
            return "".join(bases), 25.0 if include_quality else None

    def _generate_segment(self, segment_name: str) -> str:
        """Generate non-ACC segments (existing logic)."""
        # Get configured length or use defaults
        lengths = self.sim_config.get("segment_generation", {}).get(
            "lengths", {}
        )

        def _len_for(name, default):
            return (lengths.get(name) or lengths.get(name.upper()) or
                    lengths.get(name.lower()) or default)
        segment_upper = segment_name.upper()

        # Determine length based on segment type
        if "UMI" in segment_upper:
            length = _len_for(segment_name, 12)
        elif "BARCODE" in segment_upper or "CBC" in segment_upper:
            length = _len_for(segment_name, 8)
        elif "ADAPTER" in segment_upper:
            length = _len_for(segment_name, 22)
        elif "INSERT" in segment_upper or "CDNA" in segment_upper:
            length = _len_for(segment_name, 200)
        else:
            length = _len_for(segment_name, 20)

        return self._generate_random_dna(length)

    def _generate_random_dna(self, length: int) -> str:
        """Generate random DNA sequence."""
        bases = ["A", "C", "G", "T"]
        return "".join(self.random_state.choice(bases, size=length))
    
    def _resolve_iupac_codes(self, sequence: str) -> str:
        """
        Resolve IUPAC ambiguity codes to actual bases.
        
        IUPAC codes:
        N = A/C/G/T (any)
        R = A/G (purine)
        Y = C/T (pyrimidine)
        S = G/C (strong)
        W = A/T (weak)
        K = G/T (keto)
        M = A/C (amino)
        B = C/G/T (not A)
        D = A/G/T (not C)
        H = A/C/T (not G)
        V = A/C/G (not T)
        
        Args:
            sequence: Sequence that may contain IUPAC codes
            
        Returns:
            Sequence with all ambiguity codes resolved to actual bases
        """
        iupac_map = {
            'N': ['A', 'C', 'G', 'T'],
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            'S': ['G', 'C'],
            'W': ['A', 'T'],
            'K': ['G', 'T'],
            'M': ['A', 'C'],
            'B': ['C', 'G', 'T'],
            'D': ['A', 'G', 'T'],
            'H': ['A', 'C', 'T'],
            'V': ['A', 'C', 'G']
        }
        
        resolved = []
        for base in sequence.upper():
            if base in iupac_map:
                # Resolve ambiguity code to random base from allowed set
                resolved.append(self.random_state.choice(iupac_map[base]))
            elif base in ['A', 'C', 'G', 'T']:
                # Keep standard bases as-is
                resolved.append(base)
            else:
                # Unknown character - replace with random base and warn
                logger.warning(f"Unknown base '{base}' in sequence, replacing with random base")
                resolved.append(self.random_state.choice(['A', 'C', 'G', 'T']))
        
        return "".join(resolved)
    
    def _regions_from_labels(
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

    def generate_batch(
        self,
        n: int = 100,
        diversity_schedule: Optional[str] = None,
        include_quality: bool = False,
        inject_errors: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[SimulatedRead]:
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
        reads: List[SimulatedRead] = []

        for i in range(n):
            # Determine diversity boost based on schedule
            if diversity_schedule == "increasing":
                diversity_boost = 0.5 + 1.5 * (i / n)  # 0.5 to 2.0
            elif diversity_schedule == "decreasing":
                diversity_boost = 2.0 - 1.5 * (i / n)  # 2.0 to 0.5
            elif diversity_schedule == "random":
                diversity_boost = 0.5 + 1.5 * self.random_state.random()
            else:
                diversity_boost = None  # Use default

            read = self.generate_read(
                diversity_boost=diversity_boost,
                include_quality=include_quality,
                inject_errors=inject_errors,
            )
            reads.append(read)

            # Update Rich progress bar if callback is provided
            # Update every 10 sequences for smoother progress
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1)

            if (i + 1) % 1000 == 0 and not progress_callback:
                # Only log if not using an interactive progress bar
                logger.info(f"Generated {i + 1}/{n} reads")
        
        # Final progress update to ensure bar reaches 100%
        if progress_callback:
            progress_callback(n)

        self._log_generation_stats(reads)

        return reads

    def generate_train_val_split(
        self,
    ) -> Tuple[List[SimulatedRead], List[SimulatedRead]]:
        """
        Generate training and validation datasets.

        Returns:
            Tuple of (training_reads, validation_reads)
        """
        total_sequences = self.sim_config.get("num_sequences", 10000)
        train_split = self.sim_config.get("train_split", 0.8)

        num_train = int(total_sequences * train_split)
        num_val = total_sequences - num_train

        logger.info(
            f"Generating {num_train} training and {num_val} validation reads"
        )

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
        acc_sequences: List[str] = []

        for read in reads:
            if "ACC" in read.label_regions:
                for start, end in read.label_regions["ACC"]:
                    acc_seq = read.sequence[start:end]
                    acc_sequences.append(acc_seq)

        if not acc_sequences:
            return {"error": "No ACC sequences found"}

        diversity_metrics: Dict[str, Any] = {
            "total_acc_sequences": len(acc_sequences),
            "unique_sequences": len(set(acc_sequences)),
            "uniqueness_ratio": len(set(acc_sequences)) / len(acc_sequences),
        }

        # Additional metrics if ACC generator available
        if self.acc_generator and PWM_AVAILABLE:
            detailed_metrics = self.acc_generator.calculate_diversity_metrics(
                acc_sequences
            )
            diversity_metrics.update(detailed_metrics)

            # Add PWM scores
            scores = []
            for seq in acc_sequences[:100]:  # Limit to first 100 for efficiency
                score_dict = self.acc_generator.score_sequence_probabilistic(seq)
                scores.append(score_dict["mean_probability"])

            diversity_metrics["mean_pwm_score"] = (
                np.mean(scores) if scores else 0
            )
            diversity_metrics["std_pwm_score"] = (
                np.std(scores) if scores else 0
            )

        return diversity_metrics

    def _log_generation_stats(self, reads: List[SimulatedRead]):
        """Log statistics about generated reads."""
        if not self.config.get("logging", {}).get(
            "log_generation_stats", True
        ):
            return

        stats: Dict[str, Any] = {
            "num_reads": len(reads),
            "architecture": self.sequence_order,
            "length_distribution": {},
            "segment_sources": {},
            "error_injection": {"with_errors": 0, "without_errors": 0},
        }

        # Analyze reads
        for read in reads:
            # Length stats
            length = len(read.sequence)
            length_bin = f"{(length // 100) * 100}-{(length // 100 + 1) * 100}"
            stats["length_distribution"][length_bin] = (
                stats["length_distribution"].get(length_bin, 0) + 1
            )

            # Source stats
            for seg, sources in read.metadata.get("segment_sources", {}).items():
                if seg not in stats["segment_sources"]:
                    stats["segment_sources"][seg] = {}
                # Handle both single source (string) and multiple sources (list)
                source_list = sources if isinstance(sources, list) else [sources]
                for src in source_list:
                    stats["segment_sources"][seg][src] = (
                        stats["segment_sources"][seg].get(src, 0) + 1
                    )

            # Error stats
            if read.metadata.get("has_errors", False):
                stats["error_injection"]["with_errors"] += 1
            else:
                stats["error_injection"]["without_errors"] += 1

        # Log whitelist usage
        if self.config.get("logging", {}).get("log_whitelist_usage", True):
            stats["whitelist_usage"] = self.whitelist_manager.get_stats()

        logger.info("Generation Statistics:")
        logger.info(f"  Total reads: {stats['num_reads']}")
        logger.info(
            f"  Architecture: {' → '.join(stats['architecture'])}"
        )
        logger.info(f"  Length distribution: {stats['length_distribution']}")
        logger.info(
            f"  Reads with errors: {stats['error_injection']['with_errors']}"
        )

        if "whitelist_usage" in stats:
            logger.info("  Whitelist usage:")
            for seg, usage in stats["whitelist_usage"].items():
                logger.info(
                    f"    {seg}: loaded={usage['loaded']}, used={usage['used']}"
                )
    
    def _base_for_preview(self, p: Path) -> str:
        name = p.name
        for suf in ('.pkl.gz', '.json.gz', '.pkl', '.json', '.txt'):
            if name.endswith(suf):
                return name[: -len(suf)]
        return p.stem
    
    def save_reads(
        self, 
        reads: List[SimulatedRead], 
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

        # Capture comprehensive metadata for propagation
        simulator_metadata = {
            "config": self.config,
            "sim_config": self.sim_config,
            "timestamp": time.time(),
            "sequence_order": self.sequence_order,
            "generator_class": self.__class__.__name__,
            "generator_version": getattr(self, 'version', '0.3.0'),
            
            # Add simulation parameters summary
            "simulation_summary": {
                "total_sequences": len(reads),
                "valid_sequences": sum(1 for r in reads if not (hasattr(r, 'metadata') and r.metadata and r.metadata.get('is_invalid', False))),
                "invalid_sequences": sum(1 for r in reads if hasattr(r, 'metadata') and r.metadata and r.metadata.get('is_invalid', False)),
            }
        }
        
        # Add generator configurations if present
        if hasattr(self, 'acc_generator') and self.acc_generator:
            simulator_metadata["acc_generator_config"] = {
                "type": self.acc_generator.__class__.__name__,
                "temperature": getattr(self.acc_generator, 'temperature', None),
                "min_entropy": getattr(self.acc_generator, 'min_entropy', None),
                "diversity_boost": getattr(self.acc_generator, 'diversity_boost', None),
            }
        
        if format == 'pickle':
            # Ensure metadata is fully detached from internal objects
            for r in reads:
                r.metadata = dict(r.metadata)
            # Adjust filename for compression
            if compress and not output_path.suffix == '.gz':
                if output_path.suffix == '.pkl':
                    output_path = output_path.with_suffix('.pkl.gz')
                else:
                    output_path = output_path.with_suffix(output_path.suffix + '.gz')
            # Save pickle file
            if compress:
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump({"reads": reads,
                                 "metadata": simulator_metadata},
                                 f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump({"reads": reads,
                                 "metadata": simulator_metadata},
                                 f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create preview file if requested
            if create_preview:
                base = self._base_for_preview(output_path)
                preview_path = output_path.parent / f"{base}_preview.txt"
                self._create_preview_file(reads, preview_path)
                stats['preview_file'] = str(preview_path)
            
            stats['format'] = 'pickle'
            stats['compressed'] = compress
            
        elif format in ('text', 'tsv'):
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
                if not str(output_path).endswith('.gz'):
                    output_path = output_path.with_suffix(output_path.suffix + '.gz')
                with gzip.open(output_path, 'wt') as f:
                    json.dump({"reads": data,
                               "metadata": simulator_metadata},
                               f)
            else:
                with open(output_path, 'w') as f:
                    json.dump({"reads": data, "metadata": simulator_metadata}, f, indent=2)
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
    
    def _create_preview_file(self, reads: List[SimulatedRead], preview_path: Path):
        """
        Create an enhanced text preview file with comprehensive metadata.
        
        This enhanced version preserves:
        - Complete dataset statistics (valid/invalid counts)
        - Simulator configuration metadata
        - Read-level metadata for all sequences
        - Error type distribution for invalid reads
        - Label distribution statistics
        """
        with open(preview_path, 'w') as f:
            # ============= HEADER SECTION =============
            f.write("# TEMPEST Sequence Preview File\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Generator: {self.__class__.__name__}\n")
            f.write("#" + "=" * 70 + "\n\n")
            
            # ============= DATASET STATISTICS =============
            f.write("# DATASET STATISTICS\n")
            f.write("#" + "-" * 70 + "\n")
            f.write(f"# Total sequences: {len(reads):,}\n")
            
            # Count valid/invalid reads
            valid_count = 0
            invalid_count = 0
            error_types = {}
            
            for read in reads:
                if hasattr(read, 'metadata') and read.metadata:
                    if read.metadata.get('is_invalid', False):
                        invalid_count += 1
                        error_type = read.metadata.get('error_type', 'unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    else:
                        valid_count += 1
                else:
                    valid_count += 1
            
            f.write(f"# Valid reads: {valid_count:,} ({valid_count/len(reads)*100:.1f}%)\n")
            f.write(f"# Invalid reads: {invalid_count:,} ({invalid_count/len(reads)*100:.1f}%)\n")
            
            if error_types:
                f.write("\n# Error Type Distribution:\n")
                for error_type, count in sorted(error_types.items()):
                    f.write(f"#   {error_type}: {count:,} ({count/invalid_count*100:.1f}%)\n")
            
            # Sequence length statistics
            if reads:
                lengths = [len(r.sequence) for r in reads[:1000]]  # Sample first 1000
                f.write(f"\n# Sequence Length Statistics (from sample):\n")
                f.write(f"#   Mean: {sum(lengths)/len(lengths):.1f} bp\n")
                f.write(f"#   Min: {min(lengths)} bp\n")
                f.write(f"#   Max: {max(lengths)} bp\n")
            
            # ============= SIMULATOR CONFIGURATION =============
            f.write("\n# SIMULATOR CONFIGURATION\n")
            f.write("#" + "-" * 70 + "\n")
            
            # Sequence order
            if hasattr(self, 'sequence_order') and self.sequence_order:
                f.write(f"# Sequence order: {' -> '.join(self.sequence_order)}\n")
            
            # Key configuration parameters
            if hasattr(self, 'sim_config') and self.sim_config:
                sim_config = self.sim_config
                
                # Handle both dict and object attribute access
                def get_value(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)
                
                f.write(f"# Random seed: {get_value(sim_config, 'random_seed', 'N/A')}\n")
                f.write(f"# Train split: {get_value(sim_config, 'train_split', 'N/A')}\n")
                
                # Invalid fraction
                invalid_frac = get_value(sim_config, 'invalid_fraction', 0)
                if invalid_frac > 0:
                    f.write(f"# Invalid fraction: {invalid_frac:.3f}\n")
                
                # Error injection
                error_prob = get_value(sim_config, 'error_injection_prob', 0)
                if error_prob > 0:
                    f.write(f"# Error injection probability: {error_prob:.3f}\n")
                
                # Reverse complement probability
                rc_prob = get_value(sim_config, 'full_read_reverse_complement_prob', 0)
                if rc_prob > 0:
                    f.write(f"# Reverse complement probability: {rc_prob:.3f}\n")
            
            # PWM configuration if present
            if hasattr(self, 'acc_generator') and self.acc_generator:
                acc_gen = self.acc_generator
                f.write(f"\n# ACC Generator Configuration:\n")
                if hasattr(acc_gen, 'temperature'):
                    f.write(f"#   Temperature: {acc_gen.temperature}\n")
                if hasattr(acc_gen, 'min_entropy'):
                    f.write(f"#   Min entropy: {acc_gen.min_entropy}\n")
                if hasattr(acc_gen, 'diversity_boost'):
                    f.write(f"#   Diversity boost: {getattr(acc_gen, 'diversity_boost', 'N/A')}\n")
            
            # ============= LABEL DISTRIBUTION =============
            if reads:
                f.write("\n# LABEL DISTRIBUTION (from first 1000 sequences)\n")
                f.write("#" + "-" * 70 + "\n")
                
                label_counts = {}
                total_positions = 0
                
                for read in reads[:1000]:
                    for label in read.labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                        total_positions += 1
                
                # Sort by frequency
                sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                
                for label, count in sorted_labels:
                    percentage = (count / total_positions * 100) if total_positions > 0 else 0
                    f.write(f"# {label:10s}: {count:8,} positions ({percentage:5.2f}%)\n")
            
            # ============= SEQUENCE PREVIEW =============
            f.write("\n# SEQUENCE PREVIEW\n")
            f.write("#" + "=" * 70 + "\n")
            f.write("# Format: sequence<TAB>labels<TAB>is_invalid<TAB>metadata_json\n")
            f.write("#" + "-" * 70 + "\n\n")
            
            # Helper function to safely encode objects for JSON
            def safe_json_encode(obj):
                """Convert non-JSON-serializable objects."""
                if isinstance(obj, (list, tuple)):
                    return [safe_json_encode(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: safe_json_encode(v) for k, v in obj.items()}
                elif hasattr(obj, '__dict__'):
                    return safe_json_encode(obj.__dict__)
                elif isinstance(obj, bytes):
                    return obj.decode('utf-8', errors='replace')
                elif isinstance(obj, (set, frozenset)):
                    return list(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except:
                        return str(obj)
            
            # Show up to 100 sequences (was 10)
            n_preview = min(100, len(reads))
            
            # Write sequences with metadata
            for i, read in enumerate(reads[:n_preview], 1):
                # Basic sequence and labels
                labels_str = ' '.join(read.labels)
                
                # Metadata handling
                is_invalid = 'N'
                metadata_dict = {}
                
                if hasattr(read, 'metadata') and read.metadata:
                    is_invalid = 'Y' if read.metadata.get('is_invalid', False) else 'N'
                    
                    # Include all metadata except redundant fields
                    metadata_dict = {
                        k: safe_json_encode(v) 
                        for k, v in read.metadata.items() 
                        if k not in ['sequence', 'labels']
                    }
                
                # Add label regions if present
                if hasattr(read, 'label_regions') and read.label_regions:
                    region_summary = {}
                    for label, regions in read.label_regions.items():
                        region_summary[label] = [[start, end] for start, end in regions]
                    metadata_dict['regions'] = region_summary
                
                # Compact JSON representation
                try:
                    metadata_json = json.dumps(metadata_dict, separators=(',', ':'))
                except:
                    metadata_json = '{}'
                
                # Write the sequence line
                f.write(f"{read.sequence}\t{labels_str}\t{is_invalid}\t{metadata_json}\n")
                
                # Add human-readable annotation for first few sequences and any invalid reads
                if i <= 5 or (is_invalid == 'Y' and i <= 20):
                    if is_invalid == 'Y' and 'error_type' in metadata_dict:
                        error_type = metadata_dict['error_type']
                        error_desc = f"# -> Error: {error_type}"
                        
                        if error_type == 'segment_loss' and 'removed_segment' in metadata_dict:
                            error_desc += f" (removed: {metadata_dict['removed_segment']})"
                        elif error_type == 'segment_duplication' and 'duplicated_segment' in metadata_dict:
                            error_desc += f" (duplicated: {metadata_dict['duplicated_segment']})"
                        elif error_type == 'truncation' and 'truncation_point' in metadata_dict:
                            error_desc += f" (at position: {metadata_dict['truncation_point']})"
                        
                        f.write(error_desc + "\n")
            
            # Footer
            if len(reads) > n_preview:
                f.write(f"\n# ... {len(reads) - n_preview:,} more sequences not shown ...\n")
            
            f.write("\n# END OF PREVIEW\n")
        
        logger.info(f"Created enhanced preview file: {preview_path}")
        if invalid_count > 0:
            logger.info(f"  - {valid_count:,} valid reads, {invalid_count:,} invalid reads")
            if error_types:
                logger.info(f"  - Error types: {', '.join(f'{k}:{v}' for k, v in error_types.items())}")

        logger.info(f"Created preview file with {n_preview} sequences: {preview_path}")
    
    def load_reads(
        self, 
        input_path: Path,
        format: Optional[str] = None
    ) -> List[SimulatedRead]:
        """
        Load simulated reads from file.
        
        Args:
            input_path: Input file path
            format: Input format (auto-detected if None)
            
        Returns:
            List of SimulatedRead objects
        """
        input_path = Path(input_path)
        wrapped = False

        # Auto-detect format
        if format is None:
            if '.pkl' in str(input_path) or '.pickle' in str(input_path):
                format = 'pickle'
            elif '.json' in str(input_path):
                format = 'json'
            else:
                format = 'text'
        
        if format == 'pickle':
            if '.gz' in input_path.suffixes or str(input_path).endswith('.gz'):
                with gzip.open(input_path, 'rb') as f:
                    obj = pickle.load(f)
                    if isinstance(obj, dict) and "reads" in obj:
                        reads = obj["reads"]
                        wrapped = True
                        file_metadata = obj.get("metadata", {})
                    else:
                        reads = obj
                        file_metadata = {}
            else:
                with open(input_path, 'rb') as f:
                    obj = pickle.load(f)
                    if isinstance(obj, dict) and "reads" in obj:
                        reads = obj["reads"]
                        wrapped = True
                        file_metadata = obj.get("metadata", {})
                    else:
                        reads = obj
                        file_metadata = {}
                    
        elif format in ('text', 'tsv'):
            reads = []
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        sequence, labels_str = parts
                        # Accept comma- or space-separated labels
                        labels = (
                            labels_str.split(',')
                            if ',' in labels_str
                            else labels_str.split()
                        )
                        # Note: We lose label_regions and metadata in text format
                        reads.append(SimulatedRead(
                            sequence=sequence,
                            labels=labels,
                            label_regions={},
                            metadata={}
                        ))
                        
        elif format == 'json':
            if '.gz' in input_path.suffixes or str(input_path).endswith('.gz'):
                with gzip.open(input_path, 'rt') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict) and "reads" in obj:
                        data = obj["reads"]
                        wrapped = True
                        file_metadata = obj.get("metadata", {})
                    else:
                        data = obj
                        file_metadata = {}
            else:
                with open(input_path, 'r') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict) and "reads" in obj:
                        data = obj["reads"]
                        wrapped = True
                        file_metadata = obj.get("metadata", {})
                    else:
                        data = obj
                        file_metadata = {}
            
            # Normalize tuple regions (JSON converts tuples to lists)
            fixed = []
            for item in data:
                if item.get("label_regions"):
                    item["label_regions"] = {
                        k: [tuple(r) for r in v]
                        for k, v in item["label_regions"].items()
                    }
                fixed.append(SimulatedRead(**item))
            reads = fixed
        
        logger.info(f"Loaded {len(reads)} sequences from {input_path}")
        return reads


def create_simulator_from_config(config_source) -> SequenceSimulator:
    """
    Convenience factory for SequenceSimulator from file, dict, or TempestConfig.
    """
    from tempest.config import TempestConfig

    if isinstance(config_source, TempestConfig):
        # Use TempestConfig's own recursive dict converter
        if hasattr(config_source, "_to_dict"):
            cfg_dict = config_source._to_dict()
        elif hasattr(config_source, "to_dict"):
            cfg_dict = config_source.to_dict()
        else:
            # Very conservative fallback via dataclasses.asdict if applicable
            try:
                from dataclasses import asdict
                cfg_dict = asdict(config_source)
            except Exception:
                raise TypeError("TempestConfig cannot be converted to dict")
        return SequenceSimulator(config=cfg_dict)

    elif isinstance(config_source, (str, bytes, Path)):
        return SequenceSimulator(config_file=str(config_source))

    elif isinstance(config_source, dict):
        return SequenceSimulator(config=config_source)

    else:
        raise TypeError(
            f"Unsupported config source type: {type(config_source)}. "
            "Expected TempestConfig, dict, or path string."
        )


def reads_to_arrays_with_mask(
    reads: List[SimulatedRead],
    label_to_idx: Optional[Dict[str, int]] = None,
    max_len: Optional[int] = None,
    padding_value: int = 4,
    return_mask: bool = True,
    return_lengths: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, Dict[str, int]],
    Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray, np.ndarray]
]:
    """
    Convert reads to numpy arrays with masking support for training.
    
    This version provides:
    1. Boolean mask for non-padded positions
    2. Optional sequence lengths for dynamic RNN unrolling
    3. Memory-efficient int8 dtypes
    
    Args:
        reads: List of SimulatedRead objects
        label_to_idx: Optional pre-existing label mapping
        max_len: Maximum sequence length (pad/truncate to this)
        padding_value: Value to use for padding sequences (4 = N)
        return_mask: Whether to return boolean mask
        return_lengths: Whether to return actual sequence lengths
    
    Returns:
        X: Encoded sequences (num_reads, max_len), dtype int8
        y: Labels (num_reads, max_len), dtype int8  
        label_to_idx: Mapping of labels to indices
        mask: Boolean mask (num_reads, max_len) where True = valid position
        lengths: (Optional) Actual sequence lengths (num_reads,), dtype int32
    """
    # Determine max length
    if max_len is None:
        max_len = max(len(read.sequence) for read in reads)
    
    # Create label mapping if not provided
    if label_to_idx is None:
        all_labels = set()
        for read in reads:
            all_labels.update(read.labels)
        
        # Add special labels - PAD should always be 0 for masking
        all_labels.add("PAD")
        all_labels.add("UNKNOWN")
        all_labels.add("ERROR")
        
        # Ensure PAD is index 0
        sorted_labels = sorted(all_labels - {"PAD"})
        label_to_idx = {"PAD": 0}
        for i, label in enumerate(sorted_labels, 1):
            label_to_idx[label] = i
    
    # Base encoding
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    
    # Initialize arrays with appropriate dtypes
    num_reads = len(reads)
    X = np.full((num_reads, max_len), padding_value, dtype=np.uint8)
    y = np.full((num_reads, max_len), label_to_idx.get("PAD", 0), dtype=np.uint16)
    
    # Build LUT once outside loop for speed
    lut = np.full(128, 4, dtype=np.uint8)  # default N=4
    lut[ord('A')] = 0; lut[ord('C')] = 1; lut[ord('G')] = 2; lut[ord('T')] = 3
    lut[ord('a')] = 0; lut[ord('c')] = 1; lut[ord('g')] = 2; lut[ord('t')] = 3
    lut[ord('N')] = 4; lut[ord('n')] = 4
    lut[ord('U')] = 0; lut[ord('u')] = 0  # Treat U as A (for RNA sequences)
    
    # Initialize mask and lengths if requested
    if return_mask:
        mask = np.zeros((num_reads, max_len), dtype=bool)
    
    if return_lengths:
        lengths = np.zeros(num_reads, dtype=np.int32)
    
    # Fill arrays (no tqdm - rely on Rich logging)
    reads_iter = reads
    
    for i, read in enumerate(reads_iter):
        seq_len = min(len(read.sequence), max_len)
        
        # Store actual length
        if return_lengths:
            lengths[i] = seq_len
        
        # Vectorized base encoding for A/C/G/T/N
        seq_slice = read.sequence[:seq_len]
        X[i, :seq_len] = lut[np.frombuffer(seq_slice.encode('ascii', 'ignore'), dtype=np.uint8)]
        
        # Labels (fallback to UNKNOWN if seq longer than labels)
        if read.labels:
            lab = read.labels[:seq_len]
            y[i, :len(lab)] = np.fromiter(
                (label_to_idx.get(L, label_to_idx.get("UNKNOWN", 0)) for L in lab),
                dtype=np.uint16, count=len(lab)
            )
            if len(lab) < seq_len:
                y[i, len(lab):seq_len] = label_to_idx.get("UNKNOWN", 0)
        else:
            y[i, :seq_len] = label_to_idx.get("UNKNOWN", 0)
        
        if return_mask:
            mask[i, :seq_len] = True
    
    # Return based on requested outputs
    if return_mask and return_lengths:
        return X, y, label_to_idx, mask, lengths
    elif return_mask:
        return X, y, label_to_idx, mask
    else:
        return X, y, label_to_idx


# Keep backward compatibility
def reads_to_arrays(
    reads: List[SimulatedRead],
    label_to_idx: Optional[Dict[str, int]] = None,
    max_len: Optional[int] = None,
    padding_value: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Original reads_to_arrays function for backward compatibility.
    Consider using reads_to_arrays_with_mask for better training performance.
    """
    return reads_to_arrays_with_mask(
        reads, label_to_idx, max_len, padding_value, 
        return_mask=False, return_lengths=False
    )


def demonstrate_probabilistic_generation():
    """Demonstrate the read simulator with probabilistic ACC generation."""

    # Configuration with probabilistic ACC
    config = {
        "simulation": {
            "random_seed": 42,
            "sequence_order": ["ADAPTER5", "UMI", "ACC", "BARCODE", "INSERT", "ADAPTER3"],
            "sequences": {
                "ADAPTER5": "CTACACGACGCTCTTCCGATCT",
                "ADAPTER3": "AGATCGGAAGAGCACACGTCTG",
            },
            "segment_generation": {
                "lengths": {
                    "ADAPTER5": 22,
                    "UMI": 12,
                    "ACC": 6,
                    "BARCODE": 8,
                    "INSERT": 50,
                    "ADAPTER3": 22,
                }
            },
            "full_read_reverse_complement_prob": 0.5,
            "errors": {
                "enabled": True,
                "substitution_rate": 0.001,
                "insertion_rate": 0.0001,
                "deletion_rate": 0.0001,
            },
        },
        "pwm": {
            "pattern": "ACCSSV",  # ACC pattern with degenerate positions
            "temperature": 1.0,  # Instead of threshold, use temperature
            "min_entropy": 0.1,  # Minimum diversity at each position
        },
    }

    # Initialize simulator
    simulator = SequenceSimulator(config)

    print("Tempest Simulator with Probabilistic ACC Generation")
    print("=" * 60)

    # Generate reads with different diversity settings
    print("\n1. Generating reads with default diversity:")
    reads_default = simulator.generate_batch(n=10, include_quality=True)

    print("\n2. Generating reads with increasing diversity:")
    reads_increasing = simulator.generate_batch(n=10, diversity_schedule="increasing")

    print("\n3. Generating reads with random diversity:")
    reads_random = simulator.generate_batch(n=10, diversity_schedule="random")

    # Show some examples
    print("\nExample reads with ACC diversity:")
    for i, read in enumerate(reads_default[:3]):
        if "ACC" in read.label_regions:
            for start, end in read.label_regions["ACC"]:
                acc_seq = read.sequence[start:end]
                print(f"  Read {i+1} ACC: {acc_seq}")
                if read.quality_scores is not None:
                    acc_qual = read.quality_scores[start:end]
                    print(f"         Qual: {acc_qual}")

    # Analyze diversity
    print("\nDiversity Analysis:")
    for batch_name, batch in [
        ("Default", reads_default),
        ("Increasing", reads_increasing),
        ("Random", reads_random),
    ]:
        diversity = simulator.analyze_acc_diversity(batch)
        print(f"\n  {batch_name} diversity:")
        print(f"    Unique sequences: {diversity.get('unique_sequences', 'N/A')}")
        print(
            f"    Uniqueness ratio: {diversity.get('uniqueness_ratio', 0):.3f}"
        )
        print(
            f"    Mean entropy: {diversity.get('mean_position_entropy', 0):.3f}"
        )
        print(
            f"    Mean PWM score: {diversity.get('mean_pwm_score', 0):.3f}"
        )

    return simulator


# Example usage
if __name__ == "__main__":
    import argparse
    import json
    import pickle

    parser = argparse.ArgumentParser(description="Sequence Simulator")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for generated reads")
    parser.add_argument(
        "--format",
        choices=["json", "pickle", "tsv"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        help="Override number of sequences",
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.demo:
        # Run demonstration
        simulator = demonstrate_probabilistic_generation()
    elif args.config:
        # Create simulator from config
        simulator = SequenceSimulator(config_file=args.config)

        # Override number of sequences if specified
        if args.num_sequences:
            simulator.sim_config["num_sequences"] = args.num_sequences

        # Generate reads
        logger.info("Starting sequence generation...")
        train_reads, val_reads = simulator.generate_train_val_split()

        # Save if output specified
        if args.output:
            output_data = {
                "train": [
                    {
                        "sequence": read.sequence,
                        "labels": read.labels,
                        "label_regions": read.label_regions,
                        "metadata": read.metadata,
                    }
                    for read in train_reads
                ],
                "validation": [
                    {
                        "sequence": read.sequence,
                        "labels": read.labels,
                        "label_regions": read.label_regions,
                        "metadata": read.metadata,
                    }
                    for read in val_reads
                ],
            }

            if args.format == "json":
                if str(args.output).endswith(".gz"):
                    with gzip.open(args.output, "wt") as f:
                        json.dump(output_data, f)
                else:
                    with open(args.output, "w") as f:
                        json.dump(output_data, f, indent=2)
            elif args.format == "pickle":
                with open(args.output, "wb") as f:
                    pickle.dump(
                        {
                            "train": train_reads,
                            "validation": val_reads,
                        },
                        f,
                    )
            elif args.format == "tsv":
                with open(args.output, "w") as f:
                    f.write("split\tsequence\tlabels\n")
                    for read in train_reads:
                        f.write(
                            f"train\t{read.sequence}\t{','.join(read.labels)}\n"
                        )
                    for read in val_reads:
                        f.write(
                            f"validation\t{read.sequence}\t{','.join(read.labels)}\n"
                        )

            logger.info(f"Saved results to {args.output}")
    else:
        parser.print_help()
        print("\nRun with --demo for a demonstration of the simulator")

    print("\nSimulation complete!")


# Standalone helper functions for backward compatibility
# These are kept to maintain compatibility with existing code that may use them
def save_reads(
    reads: List[SimulatedRead], 
    output_path: Path,
    format: str = 'pickle',
    compress: bool = True,
    create_preview: bool = True
) -> Dict[str, Any]:
    """
    Standalone function to save simulated reads to file.
    
    NOTE: This function creates a dummy simulator instance. 
    It's recommended to use SequenceSimulator.save_reads directly instead.
    
    Args:
        reads: List of SimulatedRead objects
        output_path: Output file path
        format: Output format ('pickle', 'text', 'json')
        compress: Whether to compress pickle files
        create_preview: Whether to create a preview text file
        
    Returns:
        Dictionary with save statistics
    """
    # Create a dummy simulator to use its save_reads method
    simulator = SequenceSimulator()
    return simulator.save_reads(reads, output_path, format, compress, create_preview)


def load_reads(
    input_path: Path,
    format: Optional[str] = None
) -> List[SimulatedRead]:
    """
    Standalone function to load simulated reads from file.
    
    NOTE: This function creates a dummy simulator instance.
    It's recommended to use SequenceSimulator.load_reads directly instead.
    
    Args:
        input_path: Input file path
        format: Input format (auto-detected if None)
        
    Returns:
        List of SimulatedRead objects
    """
    # Create a dummy simulator to use its load_reads method
    simulator = SequenceSimulator()
    return simulator.load_reads(input_path, format)


def generate_and_save(
    config_file: str,
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
    
    NOTE: This function is kept for backward compatibility.
    Consider using the simulate command directly instead.
    
    Args:
        config_file: Configuration file path
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
    simulator = create_simulator_from_config(config_file)
    result = {}
    output_path = Path(output_path)
    
    if split:
        # Generate train/val split
        n_train = int(n_sequences * train_fraction)
        n_val = n_sequences - n_train
        
        logger.info(f"Generating {n_train} training sequences...")
        train_reads = simulator.generate_batch(n_train)
        
        logger.info(f"Generating {n_val} validation sequences...")
        val_reads = simulator.generate_batch(n_val)
        
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
        train_stats = simulator.save_reads(train_reads, train_path, format, compress, create_preview)
        val_stats = simulator.save_reads(val_reads, val_path, format, compress, create_preview)
        
        result['train'] = train_stats
        result['val'] = val_stats
        result['n_train'] = n_train
        result['n_val'] = n_val
        
    else:
        # Generate single dataset
        logger.info(f"Generating {n_sequences} sequences...")
        reads = simulator.generate_batch(n_sequences)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        stats = simulator.save_reads(reads, output_path, format, compress, create_preview)
        result.update(stats)
    
    result['total_sequences'] = n_sequences
    return result

