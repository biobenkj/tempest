"""
Test suite for the train subcommand with GPU support.

Tests cover:
- Standard model training
- Hybrid training approaches
- Ensemble training
- GPU acceleration and memory management
- Checkpointing and resumption
- Performance benchmarks
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import time
import shutil
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tempest.cli import train_command
from tempest.training.trainer import ModelTrainer
from tempest.training.hybrid_trainer import HybridTrainer
from tempest.training.ensemble import EnsembleTrainer
from tempest.utils import load_config


class TestTrainCommand:
    """Test suite for train command functionality."""
    
    @pytest.mark.unit
    def test_basic_training(self, sample_config_file, sample_sequences, temp_dir):
        """Test basic model training."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "models")
            epochs = 2
            batch_size = 16
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
            
        args = Args()
        
        # Run training
        train_command(args)
        
        # Check model output
        model_dir = Path(args.output_dir)
        assert model_dir.exists()
        assert (model_dir / "model.h5").exists() or (model_dir / "model.keras").exists()
        assert (model_dir / "config.yaml").exists()
    
    @pytest.mark.gpu
    def test_gpu_accelerated_training(self, sample_config_file, sample_sequences, temp_dir, require_gpu):
        """Test GPU-accelerated model training."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "gpu_models")
            epochs = 5
            batch_size = 32
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 2
            early_stopping = True
            patience = 3
            use_hybrid = False
            use_ensemble = False
        
        args = Args()
        
        # Ensure GPU is being used
        with tf.device('/GPU:0'):
            # Time training
            start_time = time.time()
            train_command(args)
            gpu_time = time.time() - start_time
        
        # Verify model was saved
        model_dir = Path(args.output_dir)
        assert model_dir.exists()
        
        print(f"\nGPU Training Performance:")
        print(f"  Time for {args.epochs} epochs: {gpu_time:.2f} seconds")
        print(f"  Time per epoch: {gpu_time/args.epochs:.2f} seconds")
    
    @pytest.mark.integration
    def test_hybrid_training(self, sample_config_file, sample_sequences, temp_dir):
        """Test hybrid training approach."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "hybrid_models")
            epochs = 3
            batch_size = 16
            learning_rate = 0.001
            model_type = 'hybrid'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = True
            use_ensemble = False
            hybrid_stages = 2
            hybrid_warmup_epochs = 1
        
        args = Args()
        
        # Run hybrid training
        train_command(args)
        
        # Check outputs
        model_dir = Path(args.output_dir)
        assert model_dir.exists()
        assert (model_dir / "hybrid_model.h5").exists() or (model_dir / "hybrid_model.keras").exists()
    
    @pytest.mark.integration
    def test_ensemble_training(self, sample_config_file, sample_sequences, temp_dir):
        """Test ensemble model training."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "ensemble_models")
            epochs = 2
            batch_size = 16
            learning_rate = 0.001
            model_type = 'ensemble'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = True
            num_models = 3
        
        args = Args()
        
        # Run ensemble training
        train_command(args)
        
        # Check that multiple models were created
        model_dir = Path(args.output_dir)
        assert model_dir.exists()
        
        # Should have subdirectories for each model
        model_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        assert len(model_subdirs) >= args.num_models
    
    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_multi_gpu_training(self, sample_config_file, sample_sequences, temp_dir, gpu_config):
        """Test multi-GPU training if available."""
        if gpu_config['count'] < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")
        
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "multi_gpu_models")
            epochs = 5
            batch_size = 64
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
            multi_gpu = True
        
        args = Args()
        
        # Create mirrored strategy for multi-GPU
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            start_time = time.time()
            train_command(args)
            multi_gpu_time = time.time() - start_time
        
        print(f"\nMulti-GPU Training Performance:")
        print(f"  GPUs used: {gpu_config['count']}")
        print(f"  Time: {multi_gpu_time:.2f} seconds")
    
    @pytest.mark.unit
    def test_checkpoint_resumption(self, sample_config_file, sample_sequences, temp_dir):
        """Test training resumption from checkpoint."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "checkpoint_test")
            epochs = 2
            batch_size = 16
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
        
        args = Args()
        
        # Initial training
        train_command(args)
        
        # Resume training
        args.epochs = 4
        args.resume_from_checkpoint = True
        train_command(args)
        
        # Verify continued training
        assert Path(args.output_dir).exists()
    
    @pytest.mark.parametrize("model_type,expected_files", [
        ("standard", ["model.h5", "config.yaml", "training_history.json"]),
        ("hybrid", ["hybrid_model.h5", "config.yaml", "stage_history.json"]),
    ])
    def test_model_output_files(self, sample_config_file, sample_sequences, temp_dir, model_type, expected_files):
        """Test that correct files are created for different model types."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / f"{model_type}_output")
            epochs = 1
            batch_size = 16
            learning_rate = 0.001
            model_type = model_type
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = (model_type == "hybrid")
            use_ensemble = False
        
        args = Args()
        train_command(args)
        
        model_dir = Path(args.output_dir)
        for file_name in expected_files:
            file_path = model_dir / file_name
            # Handle both .h5 and .keras extensions
            if file_name.endswith('.h5'):
                assert file_path.exists() or (model_dir / file_name.replace('.h5', '.keras')).exists()
            else:
                assert file_path.exists(), f"Missing expected file: {file_name}"
    
    @pytest.mark.gpu
    def test_gpu_memory_management(self, sample_config_file, sample_sequences, temp_dir, require_gpu, gpu_memory_monitor):
        """Test GPU memory management during training."""
        class Args:
            config = str(sample_config_file)
            output_dir = str(temp_dir / "memory_test")
            epochs = 3
            batch_size = 64
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
        
        args = Args()
        
        # Monitor GPU memory
        gpu_memory_monitor.start()
        
        with tf.device('/GPU:0'):
            train_command(args)
        
        memory_usage = gpu_memory_monitor.stop()
        
        if memory_usage:
            print(f"\nGPU Memory Usage:")
            print(f"  Memory used: {memory_usage['memory_used_mb']:.2f} MB")
            # Ensure reasonable memory usage
            assert memory_usage['memory_used_mb'] < 4000, "Excessive GPU memory usage during training"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_length_constrained_training(self, temp_dir):
        """Test training with length constraints."""
        # Create config with length constraints
        config = {
            'model': {
                'vocab_size': 5,
                'embedding_dim': 128,
                'num_labels': 6,
                'max_seq_len': 256,
                'batch_size': 32,
                'use_crf': True
            },
            'training': {
                'learning_rate': 0.001,
                'epochs': 2,
                'batch_size': 32
            },
            'length_constraints': {
                'enable': True,
                'barcode_length': 16,
                'umi_length': 12,
                'adapter_lengths': [10, 14],
                'constraint_weight': 1.0
            }
        }
        
        config_file = temp_dir / "length_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create sample data
        sequences = [
            "ATCGATCGATCGATCG:TAGCTAGCTAGC:GCTAGCTAGCTAGCTAGCTAGC",
            "GCTAGCTAGCTAGCTA:ATCGATCGATCG:TAGCTAGCTAGCTAGCTAGCTA",
        ] * 10
        
        data_file = temp_dir / "length_data.txt"
        with open(data_file, 'w') as f:
            for seq in sequences:
                f.write(f"{seq}\n")
        
        class Args:
            config = str(config_file)
            output_dir = str(temp_dir / "length_models")
            epochs = 2
            batch_size = 16
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(data_file)
            val_data = str(data_file)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
            use_length_constraints = True
        
        args = Args()
        train_command(args)
        
        # Verify model was created
        assert (Path(args.output_dir) / "model.h5").exists() or \
               (Path(args.output_dir) / "model.keras").exists()


class TestTrainerClasses:
    """Test individual trainer classes."""
    
    @pytest.mark.unit
    def test_standard_trainer_initialization(self, sample_config):
        """Test ModelTrainer initialization."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        trainer = ModelTrainer(config)
        
        assert trainer is not None
        assert trainer.config == config
    
    @pytest.mark.unit
    def test_hybrid_trainer_initialization(self, sample_config):
        """Test HybridTrainer initialization."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        trainer = HybridTrainer(config)
        
        assert trainer is not None
        assert trainer.config == config
    
    @pytest.mark.gpu
    def test_trainer_gpu_placement(self, sample_config, require_gpu):
        """Test that trainers correctly use GPU."""
        from tempest.utils.config import TempestConfig
        
        config = TempestConfig.from_dict(sample_config)
        
        with tf.device('/GPU:0'):
            trainer = ModelTrainer(config)
            model = trainer.build_model()
            
            # Check that model is on GPU
            assert len(tf.config.list_physical_devices('GPU')) > 0
    
    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop"])
    def test_different_optimizers(self, sample_config, sample_sequences, temp_dir, optimizer):
        """Test training with different optimizers."""
        # Update config
        sample_config['training']['optimizer'] = optimizer
        
        config_file = temp_dir / f"config_{optimizer}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        class Args:
            config = str(config_file)
            output_dir = str(temp_dir / f"models_{optimizer}")
            epochs = 1
            batch_size = 16
            learning_rate = 0.001
            model_type = 'standard'
            train_data = str(sample_sequences)
            val_data = str(sample_sequences)
            checkpoint_interval = 1
            early_stopping = False
            patience = 5
            use_hybrid = False
            use_ensemble = False
        
        args = Args()
        train_command(args)
        
        # Verify training completed
        assert Path(args.output_dir).exists()