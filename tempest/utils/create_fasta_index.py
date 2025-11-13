#!/usr/bin/env python3
"""
Utility script to create .fai index files for FASTA files.

This enables fast random access to transcript sequences without loading
the entire FASTA file into memory.

Usage:
    python create_fasta_index.py input.fa
    python create_fasta_index.py input.fa.gz
    python create_fasta_index.py input.fa --output custom_index.fai
"""

import sys
import gzip
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_fasta_index(fasta_file: str, output_fai: str = None, verbose: bool = False) -> str:
    """
    Create a .fai index file for a FASTA file.
    
    The .fai format is tab-delimited with 5 columns:
    1. Sequence name
    2. Sequence length (bp)
    3. Byte offset where sequence starts
    4. Number of bases per line
    5. Number of bytes per line (including newline)
    
    Args:
        fasta_file: Path to FASTA file (.fa, .fasta, .fa.gz, etc.)
        output_fai: Output .fai file path (default: input + '.fai')
        verbose: Print detailed progress
    
    Returns:
        Path to created .fai file
    """
    fasta_path = Path(fasta_file)
    
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    
    # Determine output path
    if output_fai is None:
        output_fai = str(fasta_path) + '.fai'
    
    output_path = Path(output_fai)
    
    logger.info(f"Creating FASTA index for: {fasta_file}")
    logger.info(f"Output index file: {output_fai}")
    
    # Check if gzipped
    is_gzipped = fasta_path.suffix == '.gz' or any(
        s == '.gz' for s in fasta_path.suffixes
    )
    
    if is_gzipped:
        logger.info("Detected gzipped FASTA file")
        opener = gzip.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'
    
    index_entries = []
    sequences_processed = 0
    
    with opener(fasta_path, mode) as f:
        current_name = None
        current_length = 0
        current_offset = 0
        linebases = None
        linewidth = None
        sequence_start_offset = 0
        first_seq_line = True
        
        byte_offset = 0
        
        for line_num, line in enumerate(f, 1):
            # Count bytes in the line (for offset tracking)
            line_bytes = len(line.encode('utf-8'))
            
            if line.startswith('>'):
                # Save previous sequence entry
                if current_name is not None:
                    if linebases is None or linewidth is None:
                        logger.warning(
                            f"Sequence {current_name} has no sequence lines, skipping"
                        )
                    else:
                        index_entries.append(
                            f"{current_name}\t{current_length}\t{sequence_start_offset}\t"
                            f"{linebases}\t{linewidth}"
                        )
                        sequences_processed += 1
                        
                        if verbose and sequences_processed % 10000 == 0:
                            logger.info(f"Processed {sequences_processed:,} sequences...")
                
                # Start new sequence
                current_name = line[1:].split()[0]  # Get first word after '>'
                current_length = 0
                sequence_start_offset = byte_offset + line_bytes
                linebases = None
                linewidth = None
                first_seq_line = True
                
            elif line.strip():  # Non-empty sequence line
                line_seq = line.rstrip('\n\r')
                seq_len = len(line_seq)
                current_length += seq_len
                
                # Record line format from first sequence line
                if first_seq_line:
                    linebases = seq_len
                    linewidth = line_bytes
                    first_seq_line = False
                
                # Verify consistent line format (except possibly last line)
                elif seq_len == linebases:
                    # Consistent line length - good
                    pass
                elif linebases is not None and seq_len < linebases:
                    # Shorter line - probably last line of sequence
                    # Don't update linebases/linewidth
                    pass
            
            byte_offset += line_bytes
        
        # Don't forget the last sequence
        if current_name is not None:
            if linebases is None or linewidth is None:
                logger.warning(
                    f"Last sequence {current_name} has no sequence lines, skipping"
                )
            else:
                index_entries.append(
                    f"{current_name}\t{current_length}\t{sequence_start_offset}\t"
                    f"{linebases}\t{linewidth}"
                )
                sequences_processed += 1
    
    # Write the index file
    with open(output_path, 'w') as f:
        for entry in index_entries:
            f.write(entry + '\n')
    
    logger.info(f"Successfully created index with {sequences_processed:,} sequences")
    logger.info(f"Index file: {output_fai}")
    
    # Print some statistics
    if index_entries:
        lengths = [int(entry.split('\t')[1]) for entry in index_entries]
        logger.info(f"Sequence statistics:")
        logger.info(f"  Min length: {min(lengths):,} bp")
        logger.info(f"  Max length: {max(lengths):,} bp")
        logger.info(f"  Mean length: {sum(lengths)/len(lengths):,.1f} bp")
    
    return str(output_path)


def verify_index(fasta_file: str, fai_file: str, num_tests: int = 5) -> bool:
    """
    Verify that the index file works correctly by sampling random sequences.
    
    Args:
        fasta_file: Path to FASTA file
        fai_file: Path to .fai index file
        num_tests: Number of random sequences to test
    
    Returns:
        True if all tests pass
    """
    import random
    
    logger.info(f"\nVerifying index with {num_tests} random sequences...")
    
    fasta_path = Path(fasta_file)
    is_gzipped = fasta_path.suffix == '.gz' or any(s == '.gz' for s in fasta_path.suffixes)
    
    # Read index
    index_entries = []
    with open(fai_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                name, length, offset, linebases, linewidth = parts[:5]
                index_entries.append((
                    name, int(length), int(offset), int(linebases), int(linewidth)
                ))
    
    if not index_entries:
        logger.error("Index file is empty!")
        return False
    
    # Sample random entries
    test_entries = random.sample(
        index_entries, 
        min(num_tests, len(index_entries))
    )
    
    opener = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    
    all_passed = True
    
    for name, length, offset, linebases, linewidth in test_entries:
        try:
            with opener(fasta_path, mode) as f:
                f.seek(offset)
                
                # Read sequence
                num_lines = (length + linebases - 1) // linebases
                sequence_parts = []
                
                for _ in range(num_lines):
                    line = f.readline().strip()
                    if line and not line.startswith('>'):
                        sequence_parts.append(line)
                
                sequence = ''.join(sequence_parts)
                
                if len(sequence) != length:
                    logger.error(
                        f"Length mismatch for {name}: "
                        f"expected {length}, got {len(sequence)}"
                    )
                    all_passed = False
                else:
                    logger.info(f"✓ {name}: {length} bp verified")
                    
        except Exception as e:
            logger.error(f"Failed to read {name}: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All index verification tests passed!")
    else:
        logger.error("\n✗ Some index verification tests failed")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Create .fai index file for FASTA files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create index for uncompressed FASTA
  python create_fasta_index.py transcripts.fa
  
  # Create index for gzipped FASTA
  python create_fasta_index.py gencode.v49.transcripts.fa.gz
  
  # Specify custom output path
  python create_fasta_index.py input.fa --output my_index.fai
  
  # Verify the created index
  python create_fasta_index.py input.fa --verify
  
  # Verbose mode
  python create_fasta_index.py input.fa --verbose

Note: For very large FASTA files (>1GB), consider using samtools faidx 
which is significantly faster:
  samtools faidx input.fa
        """
    )
    
    parser.add_argument(
        'fasta_file',
        help='Path to FASTA file (.fa, .fasta, .fa.gz, etc.)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output .fai file path (default: input + .fai)',
        default=None
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the index after creation'
    )
    
    parser.add_argument(
        '--num-tests',
        type=int,
        default=5,
        help='Number of sequences to test during verification (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create the index
        fai_path = create_fasta_index(
            args.fasta_file, 
            args.output, 
            args.verbose
        )
        
        # Verify if requested
        if args.verify:
            verify_index(args.fasta_file, fai_path, args.num_tests)
        
        logger.info("\n✓ Index creation complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
