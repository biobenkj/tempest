#!/usr/bin/env python3
"""
Example usage of the enhanced Tempest simulator with whitelist support.
Demonstrates how to use text file whitelists for i7, i5, and CBC segments.
"""

import logging
from pathlib import Path
from enhanced_simulator import SequenceSimulator, SimulationConfig
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_example_whitelists():
    """Create example whitelist files for demonstration."""
    
    # Create example i7 index sequences (8bp each)
    i7_sequences = [
        'ATTACTCG',
        'TCCGGAGA',
        'CGCTCATT',
        'GAGATTCC',
        'ATTCAGAA',
        'GAATTCGT',
        'CTGAAGCT',
        'TAATGCGC',
        'CGGCTATG',
        'TCCGCGAA',
        'TCTCGCGC',
        'AGCGATAG'
    ]
    
    # Create example i5 index sequences (8bp each)
    i5_sequences = [
        'TATAGCCT',
        'ATAGAGGC',
        'CCTATCCT',
        'GGCTCTGA',
        'AGGCGAAG',
        'TAATCTTA',
        'CAGGACGT',
        'GTACTGAC'
    ]
    
    # Create example CBC sequences (16bp each) - like 10x Genomics barcodes
    cbc_sequences = [
        'AAACCTGAGAAACCAT',
        'AAACCTGAGAAACCGC',
        'AAACCTGAGAAACCTA',
        'AAACCTGAGAAACGAG',
        'AAACCTGAGAAACGCC',
        'AAACCTGAGAAACTAC',
        'AAACCTGAGAAACTGT',
        'AAACCTGAGAAAGAAC',
        'AAACCTGAGAAAGCCT',
        'AAACCTGAGAAAGGTG',
        'AAACCTGAGAAAGTGA',
        'AAACCTGAGAAATAAC',
        'AAACCTGAGAAATACG',
        'AAACCTGAGAAATCGC',
        'AAACCTGAGAAATGCC',
        'AAACCTGAGAAATTCG'
    ]
    
    # Write whitelist files
    with open('i7_whitelist.txt', 'w') as f:
        for seq in i7_sequences:
            f.write(seq + '\n')
    logger.info(f"Created i7_whitelist.txt with {len(i7_sequences)} sequences")
    
    with open('i5_whitelist.txt', 'w') as f:
        for seq in i5_sequences:
            f.write(seq + '\n')
    logger.info(f"Created i5_whitelist.txt with {len(i5_sequences)} sequences")
    
    with open('cbc_whitelist.txt', 'w') as f:
        for seq in cbc_sequences:
            f.write(seq + '\n')
    logger.info(f"Created cbc_whitelist.txt with {len(cbc_sequences)} sequences")
    
    # Create a custom whitelist for another segment
    custom_sequences = [
        'GATCGATC',
        'CTAGCTAG',
        'AGCTAGCT',
        'TCGATCGA'
    ]
    
    with open('custom_whitelist.txt', 'w') as f:
        for seq in custom_sequences:
            f.write(seq + '\n')
    logger.info("Created custom_whitelist.txt for demonstration")


def example_dual_index_configuration():
    """Example configuration for dual-index sequencing (i7 + i5)."""
    
    config = SimulationConfig(
        # Define sequence structure with dual indexes
        sequence_order=[
            'ADAPTER5',
            'UMI',
            'INDEX_i7',
            'ACC',
            'INDEX_i5',
            'INSERT',
            'ADAPTER3'
        ],
        
        # Specify whitelist files
        i7_whitelist_file='i7_whitelist.txt',
        i5_whitelist_file='i5_whitelist.txt',
        
        # Fixed adapter sequences
        sequences={
            'ADAPTER5': 'AGATCGGAAGAGC',
            'ADAPTER3': 'AGATCGGAAGAGC',
            'INSERT': 'random'  # Will generate random insert
        },
        
        # Simulation parameters
        num_sequences=100,
        umi_length=8,
        insert_min_length=50,
        insert_max_length=150,
        error_rate=0.02,
        random_seed=42
    )
    
    return config


def example_single_cell_configuration():
    """Example configuration for single-cell sequencing with CBC."""
    
    config = SimulationConfig(
        # Define sequence structure with cell barcode
        sequence_order=[
            'TSO',  # Template switching oligo
            'CBC',  # Cell barcode
            'UMI',
            'POLY_T',
            'INSERT',
            'ADAPTER3'
        ],
        
        # Specify whitelist files
        cbc_whitelist_file='cbc_whitelist.txt',
        
        # Fixed sequences
        sequences={
            'TSO': 'AAGCAGTGGTATCAACGCAGAGTAC',
            'POLY_T': 'TTTTTTTTTTTTTTTTTTTT',
            'ADAPTER3': 'AGATCGGAAGAGC',
            'INSERT': 'random'
        },
        
        # Simulation parameters
        num_sequences=100,
        umi_length=10,  # 10x uses 10bp UMIs
        insert_min_length=100,
        insert_max_length=300,
        error_rate=0.01,
        random_seed=42
    )
    
    return config


def example_custom_configuration():
    """Example with custom whitelist mapping."""
    
    config = SimulationConfig(
        # Custom sequence structure
        sequence_order=[
            'ADAPTER5',
            'CUSTOM_TAG',
            'UMI',
            'BARCODE',
            'INSERT',
            'ADAPTER3'
        ],
        
        # Generic whitelist mapping
        whitelist_files={
            'CUSTOM_TAG': 'custom_whitelist.txt',
            'BARCODE': 'cbc_whitelist.txt'  # Reuse CBC as barcode
        },
        
        # Fixed sequences
        sequences={
            'ADAPTER5': 'AGATCGGAAGAGC',
            'ADAPTER3': 'AGATCGGAAGAGC',
            'INSERT': 'random'
        },
        
        # Simulation parameters
        num_sequences=50,
        umi_length=8,
        insert_min_length=75,
        insert_max_length=125,
        error_rate=0.015,
        random_seed=42
    )
    
    return config


def run_simulation(config: SimulationConfig, name: str):
    """Run simulation and display results."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running simulation: {name}")
    logger.info(f"{'='*60}")
    
    # Initialize simulator
    simulator = SequenceSimulator(config)
    
    # Generate sequences
    train_reads, val_reads = simulator.generate()
    
    # Get statistics
    all_reads = train_reads + val_reads
    stats = simulator.get_statistics(all_reads)
    
    # Display statistics
    logger.info(f"\nSimulation Statistics:")
    logger.info(f"  Total sequences generated: {stats['num_reads']}")
    logger.info(f"  Average sequence length: {stats.get('avg_length', 0):.1f}")
    logger.info(f"  Length range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)}")
    logger.info(f"  Sequences with errors: {stats['error_injection']['percent_with_errors']:.1f}%")
    
    logger.info(f"\nComponent usage:")
    for component, count in stats['component_counts'].items():
        logger.info(f"  {component}: {count} occurrences")
    
    logger.info(f"\nWhitelist usage:")
    for component, usage in stats['whitelist_usage'].items():
        total = sum(usage.values())
        if total > 0:
            whitelist_pct = 100 * usage['whitelist'] / total
            config_pct = 100 * usage['config'] / total
            generated_pct = 100 * usage['generated'] / total
            logger.info(f"  {component}:")
            if usage['whitelist'] > 0:
                logger.info(f"    - Whitelist: {whitelist_pct:.1f}%")
            if usage['config'] > 0:
                logger.info(f"    - Config: {config_pct:.1f}%")
            if usage['generated'] > 0:
                logger.info(f"    - Generated: {generated_pct:.1f}%")
    
    # Show a few example sequences
    logger.info(f"\nExample sequences (first 3):")
    for i, read in enumerate(train_reads[:3]):
        logger.info(f"\n  Read {i+1}:")
        logger.info(f"    Sequence: {read.sequence[:50]}..." if len(read.sequence) > 50 else f"    Sequence: {read.sequence}")
        logger.info(f"    Length: {len(read.sequence)}")
        logger.info(f"    Components: {list(read.label_regions.keys())}")
        logger.info(f"    Sources: {read.metadata.get('component_sources', {})}")
    
    # Save example to JSON
    output_file = f"{name.lower().replace(' ', '_')}_example.json"
    example_data = {
        'config': {
            'sequence_order': config.sequence_order,
            'num_sequences': config.num_sequences,
            'error_rate': config.error_rate
        },
        'statistics': {
            'total_sequences': stats['num_reads'],
            'avg_length': stats.get('avg_length', 0),
            'whitelist_usage': stats['whitelist_usage']
        },
        'example_reads': [
            {
                'sequence': read.sequence[:100] + '...' if len(read.sequence) > 100 else read.sequence,
                'length': len(read.sequence),
                'components': list(read.label_regions.keys())
            }
            for read in train_reads[:5]
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(example_data, f, indent=2)
    logger.info(f"\nSaved example data to {output_file}")
    
    return train_reads, val_reads


def main():
    """Main demonstration function."""
    
    logger.info("Enhanced Tempest Simulator with Whitelist Support")
    logger.info("=" * 60)
    
    # Create example whitelist files
    logger.info("\nCreating example whitelist files...")
    create_example_whitelists()
    
    # Run dual-index simulation
    config = example_dual_index_configuration()
    train_reads, val_reads = run_simulation(config, "Dual-Index Sequencing")
    
    # Run single-cell simulation
    config = example_single_cell_configuration()
    train_reads, val_reads = run_simulation(config, "Single-Cell Sequencing")
    
    # Run custom simulation
    config = example_custom_configuration()
    train_reads, val_reads = run_simulation(config, "Custom Configuration")
    
    logger.info("\n" + "="*60)
    logger.info("Simulation complete!")
    logger.info("\nWhitelist files created:")
    logger.info("  - i7_whitelist.txt: i7 index sequences")
    logger.info("  - i5_whitelist.txt: i5 index sequences")
    logger.info("  - cbc_whitelist.txt: Cell barcode sequences")
    logger.info("  - custom_whitelist.txt: Custom tag sequences")
    
    logger.info("\nYou can now use these whitelist files in your configuration:")
    logger.info("  config = SimulationConfig(")
    logger.info("      i7_whitelist_file='i7_whitelist.txt',")
    logger.info("      i5_whitelist_file='i5_whitelist.txt',")
    logger.info("      cbc_whitelist_file='cbc_whitelist.txt',")
    logger.info("      # ... other parameters")
    logger.info("  )")


if __name__ == "__main__":
    main()
