"""
Test suite for the simulate subcommand with GPU support.

Tests cover:
- Sequence generation with different configurations
- GPU-accelerated simulation when available
- Data splitting functionality
- Random seed reproducibility
- Performance benchmarks on GPU vs CPU
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

from tempest.cli import simulate_command
from tempest.data import SequenceSimulator
from tempest.utils import load_config


class TestSimulateCommand:
    """Test suite for simulate command functionality."""
    
    @pytest.mark.unit
    def test_basic_simulation(self, sample_config_file, temp_dir):
        """Test basic sequence simulation."""
        # Create mock arguments
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / "output.txt")
            output_dir = None
            num_sequences = 100
            split = False
            train_fraction = 0.8
            seed = 42
            
        args = Args()
        
        # Run simulation
        simulate_command(args)
        
        # Check output file exists
        assert Path(args.output).exists()
        
        # Verify number of sequences
        with open(args.output, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 100
    
    @pytest.mark.unit
    def test_simulation_with_split(self, sample_config_file, temp_dir):
        """Test sequence simulation with train/val split."""
        class Args:
            config = str(sample_config_file)
            output = None
            output_dir = str(temp_dir)
            num_sequences = 1000
            split = True
            train_fraction = 0.8
            seed = 42
            
        args = Args()
        
        # Run simulation
        simulate_command(args)
        
        # Check output files exist
        train_file = temp_dir / "train.txt"
        val_file = temp_dir / "val.txt"
        assert train_file.exists()
        assert val_file.exists()
        
        # Verify split proportions
        with open(train_file, 'r') as f:
            train_lines = f.readlines()
        with open(val_file, 'r') as f:
            val_lines = f.readlines()
        
        assert len(train_lines) == 800
        assert len(val_lines) == 200
    
    @pytest.mark.unit
    def test_reproducibility_with_seed(self, sample_config_file, temp_dir):
        """Test that simulation is reproducible with same seed."""
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / "output1.txt")
            output_dir = None
            num_sequences = 50
            split = False
            train_fraction = 0.8
            seed = 12345
            
        args1 = Args()
        simulate_command(args1)
        
        # Run again with same seed
        args2 = Args()
        args2.output = str(temp_dir / "output2.txt")
        simulate_command(args2)
        
        # Compare outputs
        with open(args1.output, 'r') as f1, open(args2.output, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        
        assert lines1 == lines2
    
    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_gpu_accelerated_simulation(self, sample_config_file, temp_dir, require_gpu, gpu_memory_monitor):
        """Test GPU-accelerated simulation and compare performance."""
        
        # Test configuration
        num_sequences = 10000
        
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / "gpu_output.txt")
            output_dir = None
            num_sequences = num_sequences
            split = False
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        
        # Monitor GPU memory
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
        with open(args.output, 'r') as f:
            lines = f.readlines()
        assert len(lines) == num_sequences
        
        # Log performance metrics
        print(f"\nGPU Simulation Performance:")
        print(f"  Time: {gpu_time:.2f} seconds")
        print(f"  Sequences per second: {num_sequences/gpu_time:.0f}")
        if memory_usage:
            print(f"  GPU Memory used: {memory_usage['memory_used_mb']:.2f} MB")
    
    @pytest.mark.integration
    def test_simulate_with_pwm_constraints(self, temp_dir):
        """Test simulation with PWM constraints."""
        # Create config with PWM settings
        config = {
            'model': {
                'vocab_size': 5,
                'embedding_dim': 64,
                'num_labels': 6,
                'max_seq_len': 150
            },
            'simulation': {
                'num_sequences': 100,
                'random_seed': 42,
                'barcode_length': 16,
                'umi_length': 12,
                'use_pwm': True
            },
            'pwm': {
                'barcode_pwm_path': str(temp_dir / "barcode_pwm.txt"),
                'min_score_threshold': 0.7
            }
        }
        
        # Create simple PWM file
        pwm_file = temp_dir / "barcode_pwm.txt"
        with open(pwm_file, 'w') as f:
            f.write("# Barcode PWM\n")
            f.write("A\t0.25\t0.25\t0.25\t0.25\n")
            f.write("C\t0.25\t0.25\t0.25\t0.25\n")
            f.write("G\t0.25\t0.25\t0.25\t0.25\n")
            f.write("T\t0.25\t0.25\t0.25\t0.25\n")
        
        config_file = temp_dir / "pwm_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        class Args:
            config = str(config_file)
            output = str(temp_dir / "pwm_output.txt")
            output_dir = None
            num_sequences = 100
            split = False
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        simulate_command(args)
        
        # Verify output
        assert Path(args.output).exists()
    
    @pytest.mark.parametrize("num_sequences,expected_time", [
        (100, 1.0),
        (1000, 5.0),
        (10000, 30.0),
    ])
    @pytest.mark.benchmark
    def test_simulation_performance(self, sample_config_file, temp_dir, num_sequences, expected_time):
        """Test simulation performance with different data sizes."""
        class Args:
            config = str(sample_config_file)
            output = str(temp_dir / f"perf_{num_sequences}.txt")
            output_dir = None
            num_sequences = num_sequences
            split = False
            train_fraction = 0.8
            seed = 42
        
        args = Args()
        
        start_time = time.time()
        simulate_command(args)
        elapsed_time = time.time() - start_time
        
        # Check performance
        assert elapsed_time < expected_time, f"Simulation took {elapsed_time:.2f}s, expected < {expected_time}s"
        
        # Verify output
        with open(args.output, 'r') as f:
            lines = f.readlines()
        assert len(lines) == num_sequences


class TestSequenceSimulator:
    """Test the SequenceSimulator class directly."""
    
    @pytest.mark.unit
    def test_simulator_initialization(self, sample_config):
        """Test SequenceSimulator initialization."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        simulator = SequenceSimulator(config)
        
        assert simulator is not None
        assert simulator.config == config
    
    @pytest.mark.unit
    def test_sequence_generation_format(self, sample_config):
        """Test that generated sequences have correct format."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        simulator = SequenceSimulator(config)
        
        sequences = simulator.generate(10)
        
        for seq in sequences:
            # Check format: barcode:umi:variable_region
            parts = seq.split(':')
            assert len(parts) == 3
            
            # Check lengths
            assert len(parts[0]) == config.simulation.barcode_length
            assert len(parts[1]) == config.simulation.umi_length
    
    @pytest.mark.gpu
    def test_gpu_memory_efficiency(self, sample_config, require_gpu, gpu_memory_monitor):
        """Test memory efficiency during large-scale simulation."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        simulator = SequenceSimulator(config)
        
        # Start memory monitoring
        gpu_memory_monitor.start()
        
        # Generate sequences in batches
        batch_size = 1000
        num_batches = 10
        
        all_sequences = []
        for _ in range(num_batches):
            with tf.device('/GPU:0'):
                sequences = simulator.generate(batch_size)
                all_sequences.extend(sequences)
        
        # Stop monitoring
        memory_usage = gpu_memory_monitor.stop()
        
        # Verify sequences
        assert len(all_sequences) == batch_size * num_batches
        
        # Check memory usage is reasonable
        if memory_usage:
            assert memory_usage['memory_used_mb'] < 1000, "Excessive GPU memory usage"
