"""
Example Usage of Unified Tempest Simulator

Demonstrates various sequence architectures and features:
- Whitelist support
- Transcript-based cDNA inserts
- Variable polyA tails
- Custom architectures
"""

import yaml
from pathlib import Path
from unified_simulator import UnifiedSequenceSimulator, convert_reads_to_arrays
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def example_illumina_rnaseq():
    """Example: Illumina RNA-seq with dual indices and polyA."""
    config = {
        'simulation': {
            'sequence_order': [
                'ADAPTER5',
                'UMI',
                'INDEX_i7',
                'ACC',
                'INDEX_i5',
                'INSERT',  # cDNA from transcripts
                'POLYA',   # Variable length polyA
                'ADAPTER3'
            ],
            
            'sequences': {
                'ADAPTER5': 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA',
                'ADAPTER3': 'AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT',
                'INSERT': 'transcript',  # Use transcript pool
                'POLYA': 'polya'  # Use polyA generator
            },
            
            'whitelist_files': {
                'INDEX_i7': 'whitelists/i7_indices.txt',
                'INDEX_i5': 'whitelists/i5_indices.txt',
                'UMI': 'whitelists/umi_sequences.txt'
            },
            
            'fallback_sequences': {
                'INDEX_i7': ['ATTACTCG', 'TCCGGAGA', 'CGCTCATT', 'GAGATTCC'],
                'INDEX_i5': ['TATAGCCT', 'ATAGAGGC', 'CCTATCCT', 'GGCTCTGA'],
                'ACC': ['GGGGGG', 'AAAAAA', 'CCCCCC', 'TTTTTT']
            },
            
            'segment_generation': {
                'lengths': {
                    'UMI': 12
                }
            },
            
            'transcript': {
                'fasta_file': 'transcripts/human_transcripts.fasta',
                'min_length': 100,
                'max_length': 5000,
                'fragment_min': 200,
                'fragment_max': 800,
                'reverse_complement_prob': 0.5
            },
            
            'polya_tail': {
                'distribution': 'normal',
                'mean_length': 150,
                'std_length': 50,
                'min_length': 10,
                'max_length': 300,
                'purity': 0.95
            },
            
            'error_profile': {
                'error_rate': 0.02,
                'substitution': {'rate': 0.015},
                'insertion': {'rate': 0.0025},
                'deletion': {'rate': 0.0025}
            },
            
            'num_sequences': 1000,
            'random_seed': 42
        }
    }
    
    print("\n=== Illumina RNA-seq Example ===")
    simulator = UnifiedSequenceSimulator(config=config)
    reads = simulator.generate_batch(10)
    
    for i, read in enumerate(reads[:3]):
        print(f"\nRead {i+1}:")
        print(f"  Length: {len(read.sequence)}")
        print(f"  Architecture: {read.metadata['architecture']}")
        print(f"  Segment lengths:")
        for seg in read.metadata['architecture']:
            if f'{seg}_length' in read.metadata:
                print(f"    {seg}: {read.metadata[f'{seg}_length']} bp")


def example_10x_singlecell():
    """Example: 10x Genomics single-cell RNA-seq."""
    config = {
        'simulation': {
            'sequence_order': [
                'TSO',          # Template switch oligo
                'CBC',          # Cell barcode
                'UMI',          # Unique molecular identifier
                'POLY_T',       # PolyT primer
                'INSERT',       # cDNA insert
                'POLYA',        # PolyA tail
                'ADAPTER3'
            ],
            
            'sequences': {
                'TSO': 'AAGCAGTGGTATCAACGCAGAGTAC',
                'POLY_T': 'TTTTTTTTTTTTTTTTTTTT',
                'ADAPTER3': 'CTGTCTCTTATACACATCT',
                'INSERT': 'transcript',
                'POLYA': 'polya'
            },
            
            'whitelist_files': {
                'CBC': 'whitelists/10x_cell_barcodes.txt',
            },
            
            'fallback_sequences': {
                'CBC': [
                    'AAACCCAAGAAACCCT',
                    'AAACCCAAGAAACACT', 
                    'AAACCCAAGAAACCAT',
                    'AAACCCAAGAAACCTA'
                ]
            },
            
            'segment_generation': {
                'lengths': {
                    'UMI': 10,  # 10x uses 10bp UMIs
                    'CBC': 16   # 10x uses 16bp cell barcodes
                }
            },
            
            'transcript': {
                'fasta_file': 'transcripts/mouse_transcripts.fasta',
                'min_length': 200,
                'max_length': 10000,
                'fragment_min': 300,
                'fragment_max': 1000,
                'reverse_complement_prob': 0.0  # Always sense strand for scRNA-seq
            },
            
            'polya_tail': {
                'distribution': 'exponential',
                'lambda_param': 0.01,
                'min_length': 20,
                'max_length': 250,
                'purity': 0.98
            },
            
            'error_profile': {
                'error_rate': 0.01  # Lower error rate for 10x
            },
            
            'num_sequences': 1000,
            'random_seed': 123
        }
    }
    
    print("\n=== 10x Single-Cell Example ===")
    simulator = UnifiedSequenceSimulator(config=config)
    reads = simulator.generate_batch(10)
    
    for i, read in enumerate(reads[:3]):
        print(f"\nRead {i+1}:")
        print(f"  Length: {len(read.sequence)}")
        print(f"  Cell barcode source: {read.metadata['segment_sources'].get('CBC', 'N/A')}")
        if 'POLYA_length' in read.metadata:
            print(f"  PolyA tail length: {read.metadata['POLYA_length']} bp")


def example_nanopore_direct_rna():
    """Example: Oxford Nanopore direct RNA sequencing."""
    config = {
        'simulation': {
            'sequence_order': [
                'ADAPTER5',
                'BARCODE',      # Sample barcode
                'RT_PRIMER',    # Reverse transcription primer
                'INSERT',       # Full-length RNA/cDNA
                'POLYA',        # Native polyA tail
                'RNA_ADAPTER'   # 3' RNA adapter
            ],
            
            'sequences': {
                'ADAPTER5': 'GGCGTCTGCTTGGGTGTTTAACC',
                'RT_PRIMER': 'ACTCTAATTGGACTACTAG',
                'RNA_ADAPTER': 'GAGGGGAAAGAGTGT',
                'INSERT': 'transcript',
                'POLYA': 'polya'
            },
            
            'whitelist_files': {
                'BARCODE': 'whitelists/nanopore_barcodes.txt'
            },
            
            'fallback_sequences': {
                'BARCODE': [
                    'AAGAAAGTTGTCGGTGTC',
                    'TCGATTCCGTTTGTAGTC',
                    'GAGTCTTGTGTCCCAGTT',
                    'CTTTCGATCACGTCCTT'
                ]
            },
            
            'transcript': {
                'fasta_file': 'transcripts/full_length_transcripts.fasta',
                'min_length': 200,
                'max_length': 15000,  # Longer reads for Nanopore
                'fragment_min': 500,   # Full-length preference
                'fragment_max': 10000,
                'reverse_complement_prob': 0.0  # Direct RNA = sense strand
            },
            
            'polya_tail': {
                'distribution': 'empirical',  # Use real distribution if available
                'empirical_file': 'data/nanopore_polya_lengths.txt',
                # Fallback to normal if empirical not available
                'mean_length': 200,
                'std_length': 75,
                'min_length': 20,
                'max_length': 500,  # Longer tails visible with Nanopore
                'purity': 0.90  # Lower purity due to sequencing characteristics
            },
            
            'error_profile': {
                'error_rate': 0.10,  # Higher error rate for Nanopore
                'substitution': {'rate': 0.05},
                'insertion': {'rate': 0.03},
                'deletion': {'rate': 0.02}
            },
            
            'num_sequences': 1000,
            'random_seed': 456
        }
    }
    
    print("\n=== Nanopore Direct RNA Example ===")
    simulator = UnifiedSequenceSimulator(config=config)
    reads = simulator.generate_batch(10)
    
    for i, read in enumerate(reads[:3]):
        print(f"\nRead {i+1}:")
        print(f"  Total length: {len(read.sequence)} bp")
        print(f"  Insert length: {read.metadata.get('INSERT_length', 0)} bp")
        print(f"  PolyA length: {read.metadata.get('POLYA_length', 0)} bp")
        print(f"  Has errors: {read.metadata['has_errors']}")


def example_custom_architecture():
    """Example: Completely custom architecture."""
    config = {
        'simulation': {
            # Define your own custom architecture!
            'sequence_order': [
                'SYNC_MARKER',     # Custom synchronization sequence
                'SAMPLE_ID',       # Sample identifier
                'MOLECULE_TAG',    # Molecular tag
                'RESTRICTION_SITE', # Restriction enzyme site
                'PAYLOAD',         # Main sequence payload
                'QUALITY_CHECK',   # Quality control sequence
                'TERMINATOR'       # Termination sequence
            ],
            
            'sequences': {
                'SYNC_MARKER': 'ATCGATCGATCG',
                'RESTRICTION_SITE': 'GAATTC',  # EcoRI site
                'TERMINATOR': 'GGGGGGGG',
                'PAYLOAD': 'transcript'  # Can still use transcript pool
            },
            
            'whitelist_files': {
                'SAMPLE_ID': 'whitelists/sample_ids.txt',
                'MOLECULE_TAG': 'whitelists/molecule_tags.txt'
            },
            
            'fallback_sequences': {
                'SAMPLE_ID': ['SAMPLE001', 'SAMPLE002', 'SAMPLE003'],
                'MOLECULE_TAG': ['TAGAAA', 'TAGBBB', 'TAGCCC'],
                'QUALITY_CHECK': ['ACACAC', 'TGTGTG', 'GCGCGC']
            },
            
            'segment_generation': {
                'lengths': {
                    'SAMPLE_ID': 9,
                    'MOLECULE_TAG': 6,
                    'QUALITY_CHECK': 6
                }
            },
            
            'transcript': {
                'fasta_file': 'transcripts/custom_sequences.fasta',
                'min_length': 50,
                'max_length': 2000,
                'fragment_min': 100,
                'fragment_max': 500,
                'reverse_complement_prob': 0.25
            },
            
            'error_profile': {
                'error_rate': 0.005  # Very low error rate for custom protocol
            },
            
            'num_sequences': 1000,
            'random_seed': 789
        }
    }
    
    print("\n=== Custom Architecture Example ===")
    simulator = UnifiedSequenceSimulator(config=config)
    reads = simulator.generate_batch(10)
    
    for i, read in enumerate(reads[:3]):
        print(f"\nRead {i+1}:")
        print(f"  Architecture: {' → '.join(read.metadata['architecture'])}")
        print(f"  Length: {len(read.sequence)} bp")
        print(f"  Segment sources:")
        for seg, source in read.metadata['segment_sources'].items():
            print(f"    {seg}: {source}")


def create_example_whitelists():
    """Create example whitelist files for testing."""
    whitelist_dir = Path('whitelists')
    whitelist_dir.mkdir(exist_ok=True)
    
    # i7 indices
    i7_indices = [
        'ATTACTCG', 'TCCGGAGA', 'CGCTCATT', 'GAGATTCC',
        'ATTCAGAA', 'GAATTCGT', 'CTGAAGCT', 'TAATGCGC'
    ]
    with open(whitelist_dir / 'i7_indices.txt', 'w') as f:
        f.write('\n'.join(i7_indices))
    
    # i5 indices
    i5_indices = [
        'TATAGCCT', 'ATAGAGGC', 'CCTATCCT', 'GGCTCTGA',
        'AGGCGAAG', 'TAATCTTA', 'CAGGACGT', 'GTACTGAC'
    ]
    with open(whitelist_dir / 'i5_indices.txt', 'w') as f:
        f.write('\n'.join(i5_indices))
    
    # Cell barcodes (simplified 10x-style)
    cell_barcodes = [
        'AAACCCAAGAAACCCT', 'AAACCCAAGAAACACT',
        'AAACCCAAGAAACCAT', 'AAACCCAAGAAACCTA',
        'AAACCCAAGAAACCTC', 'AAACCCAAGAAACCTT'
    ]
    with open(whitelist_dir / '10x_cell_barcodes.txt', 'w') as f:
        f.write('\n'.join(cell_barcodes))
    
    # Custom tags
    custom_tags = [
        'TAG' + str(i).zfill(3) for i in range(1, 21)
    ]
    with open(whitelist_dir / 'molecule_tags.txt', 'w') as f:
        f.write('\n'.join(custom_tags))
    
    print(f"Created example whitelists in {whitelist_dir}/")


def create_example_transcripts():
    """Create example transcript FASTA file for testing."""
    transcript_dir = Path('transcripts')
    transcript_dir.mkdir(exist_ok=True)
    
    # Create some example transcript sequences
    transcripts = [
        (">TRANSCRIPT_001 Example gene 1",
         "ATGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGC"
         "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
         "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
         "TGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGA"),
        
        (">TRANSCRIPT_002 Example gene 2",
         "ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
         "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
         "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
         "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
         "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
        
        (">TRANSCRIPT_003 Example gene 3", 
         "ATGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCAGTGCA"
         "CTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGA"
         "AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
         "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"
         "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA"
         "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT")
    ]
    
    fasta_file = transcript_dir / 'example_transcripts.fasta'
    with open(fasta_file, 'w') as f:
        for header, seq in transcripts:
            f.write(f"{header}\n")
            # Write sequence in lines of 60 characters (FASTA convention)
            for i in range(0, len(seq), 60):
                f.write(f"{seq[i:i+60]}\n")
    
    print(f"Created example transcript file: {fasta_file}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("UNIFIED TEMPEST SIMULATOR - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create example data files
    print("\nCreating example data files...")
    create_example_whitelists()
    create_example_transcripts()
    
    # Note: These examples will work with fallback sequences if files don't exist
    # In production, you would use real whitelist and transcript files
    
    # Run examples
    try:
        example_illumina_rnaseq()
    except Exception as e:
        print(f"Illumina example error (expected if no transcript file): {e}")
    
    try:
        example_10x_singlecell()
    except Exception as e:
        print(f"10x example error (expected if no transcript file): {e}")
    
    try:
        example_nanopore_direct_rna()
    except Exception as e:
        print(f"Nanopore example error (expected if no transcript file): {e}")
    
    try:
        example_custom_architecture()
    except Exception as e:
        print(f"Custom architecture error (expected if no transcript file): {e}")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("\nTo use with real data:")
    print("1. Replace whitelist files with your actual barcode/index lists")
    print("2. Provide real transcript FASTA files")
    print("3. Adjust parameters in the configuration")
    print("4. Run: simulator = UnifiedSequenceSimulator(config=your_config)")
    print("=" * 60)


if __name__ == "__main__":
    main()
