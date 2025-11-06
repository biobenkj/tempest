#!/usr/bin/env python3
"""
Simplified test script to validate whitelist handling with two-column format.
Tests loading of udi_i7.txt, udi_i5.txt, and cbc.txt files.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhitelistManager:
    """Simplified WhitelistManager for testing."""
    
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


def test_whitelist_loading():
    """Test loading of whitelist files with two-column format."""
    print("\n" + "="*60)
    print("Testing Whitelist Loading with Two-Column Format")
    print("="*60)
    
    manager = WhitelistManager()
    
    # Test loading each whitelist file
    test_files = [
        ('INDEX_i7', 'udi_i7.txt'),
        ('INDEX_i5', 'udi_i5.txt'),
        ('CBC', 'cbc.txt')
    ]
    
    all_success = True
    
    for segment_name, filepath in test_files:
        print(f"\nLoading {segment_name} from {filepath}...")
        success = manager.load_whitelist(segment_name, filepath)
        
        if success:
            sequences = manager.whitelists.get(segment_name, [])
            print(f"✓ Successfully loaded {len(sequences)} sequences")
            
            # Show first few sequences and IDs if available
            if segment_name in manager.whitelist_ids:
                ids = manager.whitelist_ids[segment_name]
                print(f"  Sample entries (ID -> Sequence):")
                for i in range(min(5, len(sequences))):
                    print(f"    {ids[i]:6} -> {sequences[i]}")
            else:
                print(f"  Sample sequences:")
                for i in range(min(5, len(sequences))):
                    print(f"    {sequences[i]}")
            
            # Check sequence lengths
            lengths = set(len(seq) for seq in sequences)
            print(f"  Unique sequence lengths: {sorted(lengths)}")
            
            # Verify expected characteristics
            if segment_name == 'INDEX_i7':
                expected_count = 384
                expected_length = 8
            elif segment_name == 'INDEX_i5':
                expected_count = 384
                expected_length = 8
            elif segment_name == 'CBC':
                expected_count = 12
                expected_length = 6
            else:
                expected_count = None
                expected_length = None
            
            if expected_count and len(sequences) != expected_count:
                print(f"  ⚠ Warning: Expected {expected_count} sequences, got {len(sequences)}")
                all_success = False
            
            if expected_length and expected_length not in lengths:
                print(f"  ⚠ Warning: Expected length {expected_length}, got {lengths}")
                all_success = False
        else:
            print(f"✗ Failed to load {segment_name}")
            all_success = False
    
    return manager, all_success


def test_whitelist_validation():
    """Validate that all sequences in whitelists are valid DNA sequences."""
    print("\n" + "="*60)
    print("Validating Whitelist File Formats")
    print("="*60)
    
    test_files = [
        ('udi_i7.txt', 8, 384),  # filename, expected_length, expected_count
        ('udi_i5.txt', 8, 384),
        ('cbc.txt', 6, 12)
    ]
    
    all_valid = True
    
    for filepath, expected_length, expected_count in test_files:
        print(f"\nValidating {filepath}...")
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            valid_count = 0
            invalid_count = 0
            length_issues = []
            format_issues = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                
                if len(parts) != 2:
                    format_issues.append(line_num)
                    invalid_count += 1
                    continue
                
                index_id, sequence = parts
                sequence = sequence.upper()
                
                # Check if valid DNA sequence
                if all(base in 'ACGTN' for base in sequence):
                    valid_count += 1
                    
                    # Check length
                    if len(sequence) != expected_length:
                        length_issues.append((line_num, index_id, len(sequence)))
                else:
                    invalid_count += 1
                    if invalid_count <= 3:
                        print(f"  Line {line_num}: Invalid characters in sequence - {index_id}: {sequence}")
            
            # Report results
            print(f"  Results:")
            print(f"    Total valid sequences: {valid_count}")
            
            if valid_count == expected_count:
                print(f"    ✓ Count matches expected: {expected_count}")
            else:
                print(f"    ⚠ Count mismatch: expected {expected_count}, got {valid_count}")
                all_valid = False
            
            if invalid_count > 0:
                print(f"    ✗ Invalid sequences: {invalid_count}")
                all_valid = False
            
            if format_issues:
                print(f"    ✗ Format issues on lines: {format_issues[:5]}{'...' if len(format_issues) > 5 else ''}")
                all_valid = False
            
            if length_issues:
                print(f"    ⚠ Length issues ({expected_length}bp expected):")
                for line_num, id_val, length in length_issues[:3]:
                    print(f"      Line {line_num}: {id_val} has {length}bp")
                if len(length_issues) > 3:
                    print(f"      ... and {len(length_issues)-3} more")
                all_valid = False
                
        except FileNotFoundError:
            print(f"  ✗ File not found: {filepath}")
            all_valid = False
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            all_valid = False
    
    return all_valid


def display_sample_data():
    """Display sample data from each whitelist file."""
    print("\n" + "="*60)
    print("Sample Data from Whitelist Files")
    print("="*60)
    
    files = ['udi_i7.txt', 'udi_i5.txt', 'cbc.txt']
    
    for filepath in files:
        print(f"\n{filepath}:")
        print("-" * 40)
        
        try:
            with open(filepath, 'r') as f:
                # Read first 5 lines and last 2 lines
                lines = f.readlines()
                
            print("First 5 entries:")
            for i, line in enumerate(lines[:5]):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    print(f"  {parts[0]:6} | {parts[1]}")
            
            if len(lines) > 7:
                print("...")
                print("Last 2 entries:")
                for line in lines[-2:]:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        print(f"  {parts[0]:6} | {parts[1]}")
            
            print(f"Total entries: {len([l for l in lines if l.strip()])}")
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("TEMPEST WHITELIST HANDLER TEST")
    print("Testing Two-Column Format: index_id<tab>sequence")
    print("="*60)
    
    # Display sample data
    display_sample_data()
    
    # Run tests
    test_results = []
    
    # Test 1: Validate whitelist file formats
    print("\n" + "-"*60)
    validation_result = test_whitelist_validation()
    test_results.append(("File Format Validation", validation_result))
    
    # Test 2: Load whitelists using the manager
    print("\n" + "-"*60)
    manager, loading_result = test_whitelist_loading()
    test_results.append(("Whitelist Loading", loading_result))
    
    # Test 3: Verify data integrity
    print("\n" + "="*60)
    print("Data Integrity Check")
    print("="*60)
    
    integrity_pass = True
    if manager and loading_result:
        for segment in ['INDEX_i7', 'INDEX_i5', 'CBC']:
            if segment in manager.whitelists and segment in manager.whitelist_ids:
                seqs = manager.whitelists[segment]
                ids = manager.whitelist_ids[segment]
                
                if len(seqs) != len(ids):
                    print(f"✗ {segment}: Mismatch between sequences ({len(seqs)}) and IDs ({len(ids)})")
                    integrity_pass = False
                else:
                    print(f"✓ {segment}: {len(seqs)} sequences with matching IDs")
                    
                # Check for duplicate IDs
                unique_ids = set(ids)
                if len(unique_ids) != len(ids):
                    print(f"  ⚠ Warning: Duplicate IDs found ({len(ids) - len(unique_ids)} duplicates)")
                    
                # Check for duplicate sequences
                unique_seqs = set(seqs)
                if len(unique_seqs) != len(seqs):
                    print(f"  ⚠ Note: Duplicate sequences found ({len(seqs) - len(unique_seqs)} duplicates)")
    
    test_results.append(("Data Integrity", integrity_pass))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} : {status}")
    
    all_passed = all(result for _, result in test_results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe whitelist handler successfully:")
        print("• Reads two-column format (index_id<tab>sequence)")
        print("• Validates DNA sequences")
        print("• Stores both IDs and sequences for reference")
        print("• Handles all three whitelist files correctly")
    else:
        print("⚠ SOME TESTS FAILED")
        print("Please review the output above for details.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
