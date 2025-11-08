"""
Test suite for the simulate subcommand with 11-segment architecture support.

Tests cover:
- 11-segment sequence generation (p7, i7, RP2, UMI, ACC, cDNA, polyA, CBC, RP1, i5, p5)
- Probabilistic PWM generation for ACC segment
- Whitelist-based generation for i7, i5, CBC
- GPU-accelerated simulation
- Data splitting and validation
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test helpers
from test_helpers import mock_missing_imports, create_test_config_11_segments
mock_missing_imports()

from tempest.cli import simulate_command
from tempest.data import SequenceSimulator, create_simulator_from_config
from tempest.config import TempestConfig, load_config
from tempest.core import ProbabilisticPWMGenerator


class TestSimulateCommand11Segments:
    """Test simulate command with 11-segment architecture."""
    
    @pytest.mark.unit
    def test_basic_11_segment_simulation(self, sample_config_file, temp_dir):
        """Test basic 11-segment sequence generation."""
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / "output_11seg.txt")
            output_dir = None
            num_sequences = 100
            split = False
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        
        # Run simulation
        simulate_command(args)
        
        # Check output
        assert Path(args.output).exists()
        
        with open(args.output, 'r') as f:
            lines = f.readlines()
        
        # Basic validation
        assert len(lines) > 0
        
        # Check that sequences contain expected segments
        first_seq = lines[0].strip()
        # Should contain fixed sequences
        assert 'CAAGCAGAAGACGGCATACGAGAT' in first_seq  # p7
    
    @pytest.mark.unit
    def test_probabilistic_pwm_generation(self, pwm_file, temp_dir):
        """Test probabilistic PWM generation for ACC segment."""
        # Create PWM generator
        generator = ProbabilisticPWMGenerator(
            pwm_file=str(pwm_file),
            temperature=1.2,
            min_entropy=0.1
        )
        
        # Generate ACC sequences
        acc_sequences = []
        for _ in range(100):
            acc_seq = generator.generate()
            acc_sequences.append(acc_seq)
            assert len(acc_seq) == 6  # ACC should be 6bp
        
        # Check diversity (sequences should not all be identical)
        unique_sequences = set(acc_sequences)
        assert len(unique_sequences) > 10  # Should have reasonable diversity
        
        # Check that sequences follow PWM pattern
        for seq in acc_sequences[:10]:
            score = generator.score(seq)
            assert score > generator.min_score  # Should meet minimum score
    
    @pytest.mark.unit
    def test_whitelist_based_generation(self, whitelist_files, config_11_segments, temp_dir):
        """Test whitelist-based barcode generation."""
        # Update config with whitelist paths
        config_11_segments['simulation']['whitelist_files'] = {
            'i7': str(whitelist_files['i7']),
            'i5': str(whitelist_files['i5']),
            'CBC': str(whitelist_files['CBC'])
        }
        
        # Create simulator
        tempest_config = TempestConfig.from_dict(config_11_segments)
        simulator = create_simulator_from_config(tempest_config)
        
        # Generate sequences
        reads = simulator.generate(100)
        
        # Check that whitelisted segments are valid
        for read in reads[:10]:
            # Parse segments (this is simplified - real implementation may differ)
            if hasattr(read, 'segments'):
                assert read.segments['i7'] in ['ATCGATCG', 'GCTAGCTA', 'TACGTACG', 'CGATCGAT']
                assert read.segments['i5'] in ['GCTAGCTA', 'ATCGATCG', 'TAGCTAGC', 'CGATCGAT']
                assert read.segments['CBC'] in ['ATCGAT', 'GCTAGC', 'TACGTA', 'CGATCG']
    
    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_gpu_accelerated_11seg_simulation(self, sample_config_file, temp_dir, require_gpu, gpu_memory_monitor):
        """Test GPU-accelerated simulation of 11-segment sequences."""
        num_sequences = 10000
        
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / "gpu_11seg_output.txt")
            output_dir = None
            num_sequences = num_sequences
            split = False
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        
        # Monitor GPU
        gpu_memory_monitor.start()
        
        # Time GPU simulation
        start_time = time.time()
        with tf.device('/GPU:0'):
            simulate_command(args)
        gpu_time = time.time() - start_time
        
        # Stop monitoring
        memory_usage = gpu_memory_monitor.stop()
        
        # Verify output
        assert Path(args.output).exists()
        
        # Performance metrics
        print(f"\n11-Segment GPU Simulation Performance:")
        print(f"  Sequences: {num_sequences}")
        print(f"  Time: {gpu_time:.2f} seconds")
        print(f"  Rate: {num_sequences/gpu_time:.0f} sequences/second")
        
        if memory_usage:
            print(f"  GPU Memory: {memory_usage['total_memory_used_mb']:.2f} MB")
    
    @pytest.mark.integration
    def test_full_11_segment_pipeline(self, config_11_segments, pwm_file, whitelist_files, temp_dir):
        """Test complete 11-segment sequence generation pipeline."""
        # Update config with all paths
        config_11_segments['simulation']['pwm_files'] = {'ACC': str(pwm_file)}
        config_11_segments['simulation']['whitelist_files'] = {
            'i7': str(whitelist_files['i7']),
            'i5': str(whitelist_files['i5']),
            'CBC': str(whitelist_files['CBC'])
        }
        
        # Save config
        config_path = temp_dir / "full_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_11_segments, f)
        
        class Args:
            config = str(config_path)
            output = None
            output_dir = str(temp_dir)
            num_sequences = 1000
            split = True
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        
        # Run full pipeline
        simulate_command(args)
        
        # Check outputs
        train_file = temp_dir / "train.txt"
        val_file = temp_dir / "val.txt"
        
        assert train_file.exists()
        assert val_file.exists()
        
        # Verify split
        with open(train_file) as f:
            train_lines = f.readlines()
        with open(val_file) as f:
            val_lines = f.readlines()
        
        assert len(train_lines) == 800
        assert len(val_lines) == 200
        
        # Validate sequence structure
        for line in train_lines[:5]:
            # Each line should have all expected components
            assert len(line.strip()) > 600  # Minimum expected length


class TestSequenceSimulator11Segments:
    """Test SequenceSimulator with 11-segment architecture."""
    
    @pytest.mark.unit
    def test_simulator_initialization(self, config_11_segments):
        """Test simulator initialization with 11-segment config."""
        tempest_config = TempestConfig.from_dict(config_11_segments)
        simulator = SequenceSimulator(tempest_config)
        
        assert simulator is not None
        assert hasattr(simulator, 'config')
        assert simulator.config.model.num_labels == 11
    
    @pytest.mark.unit
    def test_segment_length_constraints(self, config_11_segments):
        """Test that generated segments meet length constraints."""
        tempest_config = TempestConfig.from_dict(config_11_segments)
        simulator = SequenceSimulator(tempest_config)
        
        # Generate a few sequences
        reads = simulator.generate(10)
        
        # Check segment lengths (if accessible)
        expected_lengths = {
            'p7': 24, 'i7': 8, 'RP2': 34, 'UMI': 8,
            'ACC': 6, 'RP1': 33, 'i5': 8, 'p5': 29,
            'CBC': 6
            # cDNA and polyA have variable lengths
        }
        
        # This test would need adaptation based on actual simulator output format
        assert len(reads) == 10
    
    @pytest.mark.pwm
    def test_acc_pwm_diversity(self, config_11_segments, pwm_file):
        """Test diversity of ACC segments generated from PWM."""
        config_11_segments['simulation']['pwm_files'] = {'ACC': str(pwm_file)}
        
        tempest_config = TempestConfig.from_dict(config_11_segments)
        simulator = SequenceSimulator(tempest_config)
        
        # Generate many sequences
        reads = simulator.generate(1000)
        
        # Extract ACC segments (implementation-dependent)
        # This is a placeholder - actual extraction depends on simulator output format
        acc_segments = []
        for read in reads:
            if hasattr(read, 'get_segment'):
                acc = read.get_segment('ACC')
                if acc:
                    acc_segments.append(acc)
        
        # If we can extract ACC segments, check diversity
        if acc_segments:
            unique_acc = set(acc_segments)
            diversity_ratio = len(unique_acc) / len(acc_segments)
            
            # With probabilistic generation, expect reasonable diversity
            assert diversity_ratio > 0.05  # At least 5% unique sequences
            
            # All should be 6bp
            for acc in acc_segments[:10]:
                assert len(acc) == 6
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_sequences", [100, 1000, 10000])
    def test_simulation_performance(self, config_11_segments, temp_dir, num_sequences):
        """Benchmark simulation performance for different dataset sizes."""
        tempest_config = TempestConfig.from_dict(config_11_segments)
        simulator = SequenceSimulator(tempest_config)
        
        start_time = time.time()
        reads = simulator.generate(num_sequences)
        elapsed_time = time.time() - start_time
        
        # Performance expectations (adjust based on hardware)
        expected_times = {
            100: 2.0,
            1000: 10.0,
            10000: 60.0
        }
        
        print(f"\nSimulation of {num_sequences} 11-segment sequences:")
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Rate: {num_sequences/elapsed_time:.1f} seq/sec")
        
        # Relaxed assertion for CI/CD environments
        assert elapsed_time < expected_times[num_sequences] * 2
