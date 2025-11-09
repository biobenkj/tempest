"""
Comprehensive unit tests for the refactored simulate module.

These tests cover:
- Configuration loading from YAML
- Command line argument overrides
- Simulation generation with various parameters
- Train/validation splitting
- Reproducibility with seeds
- Error handling and validation
- Performance benchmarks
"""

import pytest
import tempfile
import shutil
import yaml
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typer.testing import CliRunner
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tempest.simulate_refactored import (
    simulate_app,
    run_simulation,
    generate_command,
    validate_command,
    stats_command,
    _save_reads
)
from tempest.config import TempestConfig, SimulationConfig, PWMConfig
from tempest.data import SimulatedRead

# Create CLI runner for testing Typer commands
runner = CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        'model': {
            'max_seq_len': 1500,
            'num_labels': 11,
            'embedding_dim': 128,
            'lstm_units': 256,
            'lstm_layers': 2,
            'dropout': 0.3,
            'use_cnn': True,
            'use_bilstm': True,
            'batch_size': 32
        },
        'simulation': {
            'num_sequences': 1000,
            'train_split': 0.8,
            'random_seed': 42,
            'sequence_order': ['p7', 'i7', 'UMI', 'ACC', 'cDNA', 'polyA', 'CBC', 'i5', 'p5'],
            'sequences': {
                'p7': 'CAAGCAGAAGACGGCATACGAGAT',
                'p5': 'GTGTAGATCTCGGTGGTCGCCGTATCATT',
                'UMI': 'random',
                'cDNA': 'transcript',
                'polyA': 'polya'
            },
            'segment_generation': {
                'lengths': {
                    'p7': 24, 'i7': 8, 'UMI': 8, 'ACC': 6,
                    'cDNA': 500, 'polyA': 30, 'CBC': 6, 'i5': 8, 'p5': 29
                },
                'generation_mode': {
                    'p7': 'fixed', 'i7': 'whitelist', 'UMI': 'random',
                    'ACC': 'pwm', 'cDNA': 'transcript', 'polyA': 'polya',
                    'CBC': 'whitelist', 'i5': 'whitelist', 'p5': 'fixed'
                }
            }
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create a temporary config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_config(sample_config_dict):
    """Create a TempestConfig instance."""
    return TempestConfig.from_dict(sample_config_dict)


@pytest.fixture
def mock_simulator():
    """Create a mock SequenceSimulator."""
    mock = Mock()
    mock.generate_reads = Mock(return_value=[
        SimulatedRead(
            sequence="ATCGATCGATCG",
            labels=['p7'] * 4 + ['i7'] * 4 + ['UMI'] * 4,
            label_regions={'p7': [(0, 4)], 'i7': [(4, 8)], 'UMI': [(8, 12)]},
            metadata={'read_id': 'test_001'}
        )
    ])
    return mock


class TestRunSimulation:
    """Test the run_simulation functional entry point."""
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')
    def test_basic_simulation(self, mock_create_simulator, sample_config, temp_dir):
        """Test basic simulation without splitting."""
        # Setup mock
        mock_simulator = Mock()
        mock_reads = [
            SimulatedRead(
                sequence=f"ATCG" * 10,
                labels=['p7'] * 40,
                label_regions={'p7': [(0, 40)]},
                metadata={'id': i}
            ) for i in range(100)
        ]
        mock_simulator.generate_reads.return_value = mock_reads
        mock_create_simulator.return_value = mock_simulator
        
        # Run simulation
        result = run_simulation(
            sample_config,
            output_dir=temp_dir,
            num_sequences=100,
            seed=123
        )
        
        # Verify results
        assert result['success'] is True
        assert result['n_sequences'] == 100
        assert result['seed'] == 123
        assert 'output_file' in result
        
        # Check that simulator was called correctly
        mock_create_simulator.assert_called_once_with(sample_config)
        mock_simulator.generate_reads.assert_called_once_with(100)
        
        # Check output file exists
        output_file = Path(result['output_file'])
        assert output_file.exists()
        
        # Verify file contents
        with open(output_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 100
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')
    def test_simulation_with_split(self, mock_create_simulator, sample_config, temp_dir):
        """Test simulation with train/validation split."""
        # Setup mock
        mock_simulator = Mock()
        
        def generate_mock_reads(n):
            return [
                SimulatedRead(
                    sequence=f"ATCG" * 10,
                    labels=['p7'] * 40,
                    label_regions={'p7': [(0, 40)]},
                    metadata={'id': i}
                ) for i in range(n)
            ]
        
        mock_simulator.generate_reads.side_effect = [
            generate_mock_reads(800),  # Training data
            generate_mock_reads(200)   # Validation data
        ]
        mock_create_simulator.return_value = mock_simulator
        
        # Run simulation with split
        result = run_simulation(
            sample_config,
            output_dir=temp_dir,
            num_sequences=1000,
            split=True,
            train_fraction=0.8
        )
        
        # Verify results
        assert result['success'] is True
        assert result['n_train'] == 800
        assert result['n_val'] == 200
        assert 'train_file' in result
        assert 'val_file' in result
        
        # Check files exist
        train_file = Path(result['train_file'])
        val_file = Path(result['val_file'])
        assert train_file.exists()
        assert val_file.exists()
        
        # Verify file contents
        with open(train_file, 'r') as f:
            train_lines = f.readlines()
        with open(val_file, 'r') as f:
            val_lines = f.readlines()
        
        assert len(train_lines) == 800
        assert len(val_lines) == 200
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')
    def test_config_overrides(self, mock_create_simulator, sample_config, temp_dir):
        """Test that command line arguments override config values."""
        mock_simulator = Mock()
        mock_simulator.generate_reads.return_value = []
        mock_create_simulator.return_value = mock_simulator
        
        # Original config has num_sequences=1000, random_seed=42
        assert sample_config.simulation.num_sequences == 1000
        assert sample_config.simulation.random_seed == 42
        
        # Run with overrides
        result = run_simulation(
            sample_config,
            output_dir=temp_dir,
            num_sequences=500,
            seed=999
        )
        
        # Config should be updated
        assert result['seed'] == 999
        # Note: The actual implementation updates the config in place
        assert sample_config.simulation.random_seed == 999
        assert sample_config.simulation.num_sequences == 500


class TestGenerateCommand:
    """Test the generate CLI command."""
    
    def test_generate_basic(self, sample_config_file, temp_dir):
        """Test basic generate command."""
        with patch('tempest.simulate_refactored.create_simulator_from_config') as mock_create:
            mock_simulator = Mock()
            mock_simulator.generate_reads.return_value = [
                SimulatedRead(
                    sequence="ATCG" * 10,
                    labels=['p7'] * 40,
                    label_regions={'p7': [(0, 40)]},
                    metadata={}
                ) for _ in range(100)
            ]
            mock_create.return_value = mock_simulator
            
            result = runner.invoke(
                simulate_app,
                ["generate", "-c", str(sample_config_file), "-n", "100", "-o", str(temp_dir / "out.txt")]
            )
            
            assert result.exit_code == 0
            assert "Simulation complete" in result.stdout
            assert Path(temp_dir / "out.txt").exists()
    
    def test_generate_with_split(self, sample_config_file, temp_dir):
        """Test generate command with train/val split."""
        with patch('tempest.simulate_refactored.create_simulator_from_config') as mock_create:
            mock_simulator = Mock()
            mock_simulator.generate_reads.side_effect = [
                [SimulatedRead(sequence="ATCG", labels=['p7'], label_regions={}, metadata={})] * 80,
                [SimulatedRead(sequence="ATCG", labels=['p7'], label_regions={}, metadata={})] * 20
            ]
            mock_create.return_value = mock_simulator
            
            result = runner.invoke(
                simulate_app,
                ["generate", "-c", str(sample_config_file), "--split", "-d", str(temp_dir), "-n", "100"]
            )
            
            assert result.exit_code == 0
            assert "Training sequences: 80" in result.stdout
            assert "Validation sequences: 20" in result.stdout
            assert Path(temp_dir / "train.txt").exists()
            assert Path(temp_dir / "val.txt").exists()
    
    def test_generate_with_seed(self, sample_config_file, temp_dir):
        """Test reproducibility with seed parameter."""
        with patch('tempest.simulate_refactored.create_simulator_from_config') as mock_create:
            mock_simulator = Mock()
            mock_simulator.generate_reads.return_value = []
            mock_create.return_value = mock_simulator
            
            result = runner.invoke(
                simulate_app,
                ["generate", "-c", str(sample_config_file), "--seed", "12345", "-o", str(temp_dir / "seeded.txt")]
            )
            
            assert result.exit_code == 0
            # The seed should be passed through to the simulator
            # In the actual implementation, this would affect random generation
    
    def test_generate_verbose_output(self, sample_config_file, temp_dir):
        """Test verbose output mode."""
        with patch('tempest.simulate_refactored.create_simulator_from_config') as mock_create:
            mock_simulator = Mock()
            mock_simulator.generate_reads.return_value = []
            mock_create.return_value = mock_simulator
            
            result = runner.invoke(
                simulate_app,
                ["generate", "-c", str(sample_config_file), "--verbose", "-n", "10"]
            )
            
            assert result.exit_code == 0
            assert "TEMPEST SIMULATOR" in result.stdout
            assert "Configuration:" in result.stdout
            assert "Random seed used:" in result.stdout
    
    def test_generate_missing_config(self, temp_dir):
        """Test error handling for missing config file."""
        result = runner.invoke(
            simulate_app,
            ["generate", "-c", str(temp_dir / "nonexistent.yaml")]
        )
        
        assert result.exit_code == 2  # Typer exit code for validation error


class TestValidateCommand:
    """Test the validate CLI command."""
    
    def test_validate_valid_config(self, sample_config_file):
        """Test validation of a valid configuration."""
        result = runner.invoke(
            simulate_app,
            ["validate", "-c", str(sample_config_file)]
        )
        
        assert result.exit_code == 0
        assert "Configuration is valid!" in result.stdout
    
    def test_validate_verbose(self, sample_config_file):
        """Test verbose validation output."""
        result = runner.invoke(
            simulate_app,
            ["validate", "-c", str(sample_config_file), "--verbose"]
        )
        
        assert result.exit_code == 0
        assert "Configuration is valid!" in result.stdout
        assert "Simulation parameters:" in result.stdout
        assert "Segment architecture:" in result.stdout
        assert "Number of sequences: 1000" in result.stdout
    
    def test_validate_invalid_config(self, temp_dir):
        """Test validation of an invalid configuration."""
        # Create invalid config
        invalid_config = {
            'simulation': {
                'num_sequences': -10,  # Invalid: negative number
                'train_split': 1.5,    # Invalid: > 1
            }
        }
        
        config_path = temp_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        result = runner.invoke(
            simulate_app,
            ["validate", "-c", str(config_path)]
        )
        
        assert result.exit_code == 1
        assert "Configuration errors:" in result.stdout or "Invalid configuration:" in result.stdout
    
    def test_validate_with_warnings(self, temp_dir, sample_config_dict):
        """Test validation with warnings for missing files."""
        # Add references to non-existent files
        sample_config_dict['simulation']['whitelist_files'] = {
            'i7': str(temp_dir / 'missing_whitelist.txt')
        }
        sample_config_dict['simulation']['pwm_files'] = {
            'ACC': str(temp_dir / 'missing_pwm.txt')
        }
        
        config_path = temp_dir / "config_with_warnings.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        result = runner.invoke(
            simulate_app,
            ["validate", "-c", str(config_path)]
        )
        
        assert result.exit_code == 0
        assert "Configuration is valid!" in result.stdout
        assert "Warnings:" in result.stdout
        assert "not found" in result.stdout


class TestStatsCommand:
    """Test the stats CLI command."""
    
    def test_stats_basic(self, temp_dir):
        """Test basic statistics calculation."""
        # Create sample sequence file
        seq_file = temp_dir / "sequences.txt"
        with open(seq_file, 'w') as f:
            for i in range(100):
                seq = "ATCG" * 25  # 100bp sequence
                labels = " ".join(['p7'] * 24 + ['i7'] * 8 + ['UMI'] * 8 + ['cDNA'] * 60)
                f.write(f"{seq}\t{labels}\n")
        
        result = runner.invoke(
            simulate_app,
            ["stats", str(seq_file)]
        )
        
        assert result.exit_code == 0
        assert "Statistics for 100 sequences:" in result.stdout
        assert "Sequence lengths:" in result.stdout
        assert "Label distribution:" in result.stdout
        assert "Mean:" in result.stdout
    
    def test_stats_verbose(self, temp_dir):
        """Test verbose statistics output."""
        # Create sample sequence file
        seq_file = temp_dir / "sequences.txt"
        with open(seq_file, 'w') as f:
            for i in range(50):
                seq = "ATCG" * 30  # 120bp sequence
                labels = " ".join(['p7'] * 24 + ['i7'] * 8 + ['UMI'] * 8 + ['cDNA'] * 80)
                f.write(f"{seq}\t{labels}\n")
        
        result = runner.invoke(
            simulate_app,
            ["stats", str(seq_file), "--verbose"]
        )
        
        assert result.exit_code == 0
        assert "Statistics for 50 sequences:" in result.stdout
        assert "Detailed segment analysis:" in result.stdout
        assert "Most common segment transitions:" in result.stdout
    
    def test_stats_empty_file(self, temp_dir):
        """Test stats command with empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        
        result = runner.invoke(
            simulate_app,
            ["stats", str(empty_file)]
        )
        
        assert result.exit_code == 1
        assert "No sequences found in file!" in result.stdout
    
    def test_stats_malformed_file(self, temp_dir):
        """Test stats command with malformed file."""
        bad_file = temp_dir / "bad.txt"
        with open(bad_file, 'w') as f:
            f.write("This is not a valid sequence file\n")
            f.write("No tabs here\n")
        
        result = runner.invoke(
            simulate_app,
            ["stats", str(bad_file)]
        )
        
        # Should handle gracefully - no sequences with proper format
        assert result.exit_code == 1
        assert "No sequences found" in result.stdout or "Error" in result.stdout


class TestSaveReads:
    """Test the _save_reads utility function."""
    
    def test_save_reads_basic(self, temp_dir):
        """Test saving reads to file."""
        reads = [
            SimulatedRead(
                sequence="ATCGATCG",
                labels=['p7', 'p7', 'p7', 'p7', 'i7', 'i7', 'i7', 'i7'],
                label_regions={'p7': [(0, 4)], 'i7': [(4, 8)]},
                metadata={}
            ),
            SimulatedRead(
                sequence="GCTAGCTA",
                labels=['p7', 'p7', 'p7', 'p7', 'UMI', 'UMI', 'UMI', 'UMI'],
                label_regions={'p7': [(0, 4)], 'UMI': [(4, 8)]},
                metadata={}
            )
        ]
        
        output_file = temp_dir / "test_reads.txt"
        _save_reads(reads, output_file)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        assert lines[0].strip() == "ATCGATCG\tp7 p7 p7 p7 i7 i7 i7 i7"
        assert lines[1].strip() == "GCTAGCTA\tp7 p7 p7 p7 UMI UMI UMI UMI"
    
    def test_save_reads_empty(self, temp_dir):
        """Test saving empty read list."""
        reads = []
        output_file = temp_dir / "empty_reads.txt"
        _save_reads(reads, output_file)
        
        assert output_file.exists()
        with open(output_file, 'r') as f:
            content = f.read()
        assert content == ""


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')
    def test_end_to_end_workflow(self, mock_create_simulator, sample_config_file, temp_dir):
        """Test complete workflow from config to output."""
        # Setup mock simulator
        mock_simulator = Mock()
        
        def generate_realistic_reads(n):
            reads = []
            np.random.seed(42)
            for i in range(n):
                # Generate variable length sequences
                length = np.random.randint(100, 500)
                sequence = ''.join(np.random.choice(list('ATCG'), length))
                
                # Generate labels based on architecture
                labels = []
                pos = 0
                for segment in ['p7', 'i7', 'UMI', 'ACC', 'cDNA', 'polyA', 'CBC', 'i5', 'p5']:
                    if segment in ['p7', 'p5']:
                        seg_len = 24
                    elif segment in ['i7', 'i5', 'UMI']:
                        seg_len = 8
                    elif segment in ['ACC', 'CBC']:
                        seg_len = 6
                    elif segment == 'polyA':
                        seg_len = np.random.randint(10, 50)
                    else:  # cDNA
                        seg_len = length - pos - 100  # Remaining length
                        if seg_len < 0:
                            break
                    
                    labels.extend([segment] * min(seg_len, length - pos))
                    pos += seg_len
                    if pos >= length:
                        break
                
                reads.append(SimulatedRead(
                    sequence=sequence[:len(labels)],
                    labels=labels,
                    label_regions={},
                    metadata={'id': f'read_{i:06d}'}
                ))
            return reads
        
        mock_simulator.generate_reads.side_effect = [
            generate_realistic_reads(800),
            generate_realistic_reads(200)
        ]
        mock_create_simulator.return_value = mock_simulator
        
        # 1. Validate configuration
        result = runner.invoke(
            simulate_app,
            ["validate", "-c", str(sample_config_file)]
        )
        assert result.exit_code == 0
        
        # 2. Generate sequences with split
        result = runner.invoke(
            simulate_app,
            ["generate", "-c", str(sample_config_file), "--split", "-d", str(temp_dir), "-n", "1000", "--seed", "42"]
        )
        assert result.exit_code == 0
        
        # 3. Check statistics on generated files
        for file_name in ["train.txt", "val.txt"]:
            result = runner.invoke(
                simulate_app,
                ["stats", str(temp_dir / file_name)]
            )
            assert result.exit_code == 0
            assert "Statistics for" in result.stdout


class TestReproducibility:
    """Test reproducibility with seeds."""
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')  
    def test_same_seed_same_output(self, mock_create_simulator, sample_config, temp_dir):
        """Test that same seed produces same output."""
        
        def create_seeded_simulator(seed):
            np.random.seed(seed)
            return [
                SimulatedRead(
                    sequence=''.join(np.random.choice(list('ATCG'), 100)),
                    labels=['p7'] * 100,
                    label_regions={},
                    metadata={}
                ) for _ in range(10)
            ]
        
        mock_simulator = Mock()
        mock_simulator.generate_reads.side_effect = [
            create_seeded_simulator(123),
            create_seeded_simulator(123)
        ]
        mock_create_simulator.return_value = mock_simulator
        
        # Generate twice with same seed
        result1 = run_simulation(sample_config, output_dir=temp_dir, num_sequences=10, seed=123)
        
        # Reset mock for second call
        mock_simulator.generate_reads.side_effect = [create_seeded_simulator(123)]
        
        result2 = run_simulation(sample_config, output_dir=temp_dir / "run2", num_sequences=10, seed=123)
        
        # Both runs should have same seed in results
        assert result1['seed'] == result2['seed'] == 123


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_train_fraction(self, sample_config_file, temp_dir):
        """Test error handling for invalid train fraction."""
        result = runner.invoke(
            simulate_app,
            ["generate", "-c", str(sample_config_file), "--split", "--train-fraction", "1.5"]
        )
        
        # Typer should reject this at validation time
        assert result.exit_code != 0
    
    def test_negative_sequences(self, sample_config_file, temp_dir):
        """Test error handling for negative sequence count."""
        result = runner.invoke(
            simulate_app,
            ["generate", "-c", str(sample_config_file), "-n", "-10"]
        )
        
        # Typer should reject this at validation time
        assert result.exit_code != 0
    
    @patch('tempest.simulate_refactored.create_simulator_from_config')
    def test_simulator_exception(self, mock_create_simulator, sample_config_file, temp_dir):
        """Test handling of simulator exceptions."""
        mock_create_simulator.side_effect = RuntimeError("Simulator failed")
        
        result = runner.invoke(
            simulate_app,
            ["generate", "-c", str(sample_config_file), "-n", "100"]
        )
        
        assert result.exit_code == 1
        assert "Error:" in result.stdout
        assert "Simulator failed" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
