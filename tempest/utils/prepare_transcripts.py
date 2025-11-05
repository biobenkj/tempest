#!/usr/bin/env python3
"""
Utility script for preparing transcript FASTA files for Tempest read simulator.

Supports:
- Filtering transcripts by length
- Extracting coding sequences only
- Sampling representative transcripts
- Converting from various formats
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Process and filter transcript FASTA files for simulation."""
    
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize processor.
        
        Args:
            input_file: Input FASTA file path
            output_file: Output FASTA file path
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
    
    def filter_by_length(self, min_length: int = 200, max_length: int = 5000) -> int:
        """
        Filter transcripts by length.
        
        Args:
            min_length: Minimum transcript length
            max_length: Maximum transcript length
            
        Returns:
            Number of transcripts kept
        """
        logger.info(f"Filtering transcripts by length ({min_length}-{max_length} bp)...")
        
        filtered_records = []
        total_count = 0
        
        for record in SeqIO.parse(self.input_file, "fasta"):
            total_count += 1
            seq_len = len(record.seq)
            
            if min_length <= seq_len <= max_length:
                filtered_records.append(record)
        
        SeqIO.write(filtered_records, self.output_file, "fasta")
        
        logger.info(f"Kept {len(filtered_records)}/{total_count} transcripts")
        return len(filtered_records)
    
    def extract_coding_sequences(self) -> int:
        """
        Extract only coding sequences (CDS) from transcripts.
        
        Looks for CDS annotations in the FASTA headers.
        
        Returns:
            Number of CDS extracted
        """
        logger.info("Extracting coding sequences...")
        
        cds_records = []
        
        for record in SeqIO.parse(self.input_file, "fasta"):
            # Look for CDS markers in description
            if 'CDS' in record.description or 'cds' in record.description:
                # Try to extract CDS coordinates if present
                # Format: CDS:start-end
                import re
                match = re.search(r'CDS[:\s]+(\d+)-(\d+)', record.description)
                
                if match:
                    start = int(match.group(1)) - 1  # Convert to 0-based
                    end = int(match.group(2))
                    cds_seq = record.seq[start:end]
                    
                    new_record = SeqRecord(
                        cds_seq,
                        id=f"{record.id}_CDS",
                        description=f"CDS from {record.id}"
                    )
                    cds_records.append(new_record)
                else:
                    # Assume entire sequence is CDS
                    cds_records.append(record)
        
        if not cds_records:
            logger.warning("No CDS found, using all sequences")
            cds_records = list(SeqIO.parse(self.input_file, "fasta"))
        
        SeqIO.write(cds_records, self.output_file, "fasta")
        logger.info(f"Extracted {len(cds_records)} coding sequences")
        return len(cds_records)
    
    def sample_transcripts(self, n_samples: int = 1000, seed: int = 42) -> int:
        """
        Sample a subset of transcripts randomly.
        
        Args:
            n_samples: Number of transcripts to sample
            seed: Random seed for reproducibility
            
        Returns:
            Number of transcripts sampled
        """
        logger.info(f"Sampling {n_samples} transcripts...")
        
        np.random.seed(seed)
        
        # Load all records
        all_records = list(SeqIO.parse(self.input_file, "fasta"))
        
        if len(all_records) <= n_samples:
            logger.info(f"File has {len(all_records)} transcripts, using all")
            sampled = all_records
        else:
            indices = np.random.choice(len(all_records), n_samples, replace=False)
            sampled = [all_records[i] for i in indices]
        
        SeqIO.write(sampled, self.output_file, "fasta")
        logger.info(f"Sampled {len(sampled)} transcripts")
        return len(sampled)
    
    def deduplicate(self) -> int:
        """
        Remove duplicate sequences.
        
        Returns:
            Number of unique transcripts
        """
        logger.info("Removing duplicate sequences...")
        
        seen_seqs = set()
        unique_records = []
        
        for record in SeqIO.parse(self.input_file, "fasta"):
            seq_str = str(record.seq)
            
            if seq_str not in seen_seqs:
                seen_seqs.add(seq_str)
                unique_records.append(record)
        
        SeqIO.write(unique_records, self.output_file, "fasta")
        
        total = sum(1 for _ in SeqIO.parse(self.input_file, "fasta"))
        logger.info(f"Kept {len(unique_records)}/{total} unique transcripts")
        return len(unique_records)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the transcript file.
        
        Returns:
            Dictionary with statistics
        """
        lengths = []
        gc_contents = []
        
        for record in SeqIO.parse(self.input_file, "fasta"):
            seq = str(record.seq).upper()
            lengths.append(len(seq))
            
            # Calculate GC content
            gc_count = seq.count('G') + seq.count('C')
            total_count = len(seq)
            if total_count > 0:
                gc_contents.append(gc_count / total_count)
        
        stats = {
            'num_transcripts': len(lengths),
            'mean_length': np.mean(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0,
            'min_length': np.min(lengths) if lengths else 0,
            'max_length': np.max(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'mean_gc': np.mean(gc_contents) if gc_contents else 0,
            'std_gc': np.std(gc_contents) if gc_contents else 0
        }
        
        return stats


def download_example_transcripts(output_file: str, organism: str = 'human'):
    """
    Download example transcript sequences.
    
    NOTE: This is a placeholder. In practice, you would download from
    Ensembl, RefSeq, or other databases.
    
    Args:
        output_file: Output FASTA file path
        organism: Organism name
    """
    logger.info(f"Creating example {organism} transcripts...")
    
    # Create synthetic example transcripts
    np.random.seed(42)
    records = []
    
    # Simulate some realistic transcript properties
    transcript_data = [
        ('ACTB', 1800, 0.55),     # Beta-actin
        ('GAPDH', 1400, 0.52),    # GAPDH
        ('EEF1A1', 1600, 0.53),   # Elongation factor
        ('RPL13', 800, 0.58),     # Ribosomal protein
        ('TUBA1A', 1500, 0.54),   # Tubulin
    ]
    
    for gene, length, gc_content in transcript_data:
        # Generate sequence with approximate GC content
        seq = []
        for _ in range(length):
            if np.random.random() < gc_content:
                seq.append(np.random.choice(['G', 'C']))
            else:
                seq.append(np.random.choice(['A', 'T']))
        
        record = SeqRecord(
            Seq(''.join(seq)),
            id=f"{gene}_transcript",
            description=f"{organism} {gene} mRNA"
        )
        records.append(record)
    
    # Add some random transcripts
    for i in range(95):  # Total 100 transcripts
        length = np.random.randint(500, 3000)
        gc = np.random.uniform(0.4, 0.6)
        
        seq = []
        for _ in range(length):
            if np.random.random() < gc:
                seq.append(np.random.choice(['G', 'C']))
            else:
                seq.append(np.random.choice(['A', 'T']))
        
        record = SeqRecord(
            Seq(''.join(seq)),
            id=f"transcript_{i+1:04d}",
            description=f"{organism} transcript {i+1}"
        )
        records.append(record)
    
    SeqIO.write(records, output_file, "fasta")
    logger.info(f"Created {len(records)} example transcripts in {output_file}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Prepare transcript FASTA files for Tempest read simulator"
    )
    
    parser.add_argument(
        'action',
        choices=['filter', 'sample', 'dedupe', 'stats', 'download'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '-i', '--input',
        help='Input FASTA file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output FASTA file'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=200,
        help='Minimum transcript length (default: 200)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=5000,
        help='Maximum transcript length (default: 5000)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of transcripts to sample (default: 1000)'
    )
    
    parser.add_argument(
        '--organism',
        default='human',
        help='Organism for example transcripts (default: human)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'download':
        if not args.output:
            args.output = f"{args.organism}_transcripts.fasta"
        download_example_transcripts(args.output, args.organism)
    
    else:
        if not args.input or not args.output:
            parser.error(f"Action '{args.action}' requires --input and --output")
        
        processor = TranscriptProcessor(args.input, args.output)
        
        if args.action == 'filter':
            processor.filter_by_length(args.min_length, args.max_length)
        
        elif args.action == 'sample':
            processor.sample_transcripts(args.n_samples)
        
        elif args.action == 'dedupe':
            processor.deduplicate()
        
        elif args.action == 'stats':
            stats = processor.get_statistics()
            print("\nTranscript Statistics:")
            print("-" * 40)
            for key, value in stats.items():
                if 'length' in key:
                    print(f"{key}: {value:.0f}")
                elif key == 'num_transcripts':
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    # If run without arguments, show examples
    import sys
    if len(sys.argv) == 1:
        print("""
Transcript Preparation Utility
==============================

Examples:

1. Download example transcripts:
   python prepare_transcripts.py download -o human_transcripts.fasta

2. Filter transcripts by length:
   python prepare_transcripts.py filter -i raw.fasta -o filtered.fasta --min-length 200 --max-length 2000

3. Sample random transcripts:
   python prepare_transcripts.py sample -i all.fasta -o sampled.fasta --n-samples 1000

4. Remove duplicates:
   python prepare_transcripts.py dedupe -i raw.fasta -o unique.fasta

5. Get statistics:
   python prepare_transcripts.py stats -i transcripts.fasta -o /dev/null

For help:
   python prepare_transcripts.py -h
""")
        sys.exit(0)
    
    main()
