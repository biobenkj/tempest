"""
Comprehensive tests for config.yaml ingestion and validation.

Tests cover:
- Config loading from YAML files
- Parameter validation
- Default value handling
- Nested configuration structures
- Type checking and conversion
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
import sys

# Add tempest to path (adjust as needed for your environment)
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tempest.utils.config import (
    TempestConfig, ModelConfig, SimulationConfig, TrainingConfig,
    EnsembleConfig, InferenceConfig, PWMConfig, LengthConstraints,
    HybridTrainingConfig
)


class TestConfigLoading:
    """Test configuration loading from YAML files."""
    
    def create_temp_config(self, config_dict):
        """Helper to create temporary YAML config file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
    
    def test_basic_config_loading(self):
        """Test loading a basic valid configuration."""
        config_dict = {
            'model': {
                'vocab_size': 5,
                'embedding_dim': 128,
                'num_labels': 6,
                'max_seq_len': 256,
                'batch_size': 32
            }
        }
        
        config_path = self.create_temp_config(config_dict)
        try:
            config = TempestConfig.from_yaml(config_path)
            
            # Assertions
            assert config.model.vocab_size == 5
            assert config.model.embedding_dim == 128
            assert config.model.num_labels == 6
            assert config.model.max_seq_len == 256
            assert config.model.batch_size == 32
        finally:
            os.unlink(config_path)
    
    def test_complete_config_loading(self):
        """Test loading a complete configuration with all sections."""
        config_dict = {
            'model': {
                'vocab_size': 5,
                'embedding_dim': 128,
                'cnn_filters': [64, 128],
                'cnn_kernels': [3, 5],
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'num_labels': 6,
                'max_seq_len': 256,
                'batch_size': 32,
                'length_constraints': {
                    'constraints': {
                        'UMI': [8, 8],
                        'ACC': [6, 6],
                        'BARCODE': [16, 16]
                    },
                    'constraint_weight': 5.0,
                    'ramp_epochs': 5
                }
            },
            'simulation': {
                'sequence_order': ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3'],
                'num_sequences': 5000,
                'umi_length': 8,
                'error_rate': 0.02,
                'random_seed': 42
            },
            'training': {
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'epochs': 20,
                'train_split': 0.8,
                'early_stopping_patience': 5
            },
            'pwm': {
                'pwm_file': 'acc_pwm.txt',
                'use_pwm': True,
                'pwm_threshold': 0.7
            }
        }
        
        config_path = self.create_temp_config(config_dict)
        try:
            config = TempestConfig.from_yaml(config_path)
            
            # Model assertions
            assert config.model.embedding_dim == 128
            assert config.model.cnn_filters == [64, 128]
            assert config.model.lstm_units == 128
            
            # Length constraints assertions
            assert config.model.length_constraints is not None
            assert config.model.length_constraints.constraints['UMI'] == (8, 8)
            assert config.model.length_constraints.constraints['ACC'] == (6, 6)
            assert config.model.length_constraints.constraint_weight == 5.0
            
            # Simulation assertions
            assert config.simulation is not None
            assert config.simulation.num_sequences == 5000
            assert config.simulation.error_rate == 0.02
            assert config.simulation.random_seed == 42
            
            # Training assertions
            assert config.training is not None
            assert config.training.learning_rate == 0.001
            assert config.training.epochs == 20
            
            # PWM assertions
            assert config.pwm is not None
            assert config.pwm.use_pwm is True
            assert config.pwm.pwm_threshold == 0.7
        finally:
            os.unlink(config_path)


class TestModelConfig:
    """Test ModelConfig class and validation."""
    
    def test_default_model_config(self):
        """Test model config with default values."""
        config = ModelConfig()
        
        assert config.vocab_size == 5
        assert config.embedding_dim == 128
        assert config.use_cnn is True
        assert config.use_bilstm is True
        assert config.dropout == 0.3
        assert config.num_labels == 10
        assert config.max_seq_len == 512
        assert config.batch_size == 32
    
    def test_custom_model_config(self):
        """Test model config with custom values."""
        config = ModelConfig(
            vocab_size=4,
            embedding_dim=256,
            cnn_filters=[128, 256],
            lstm_units=256,
            num_labels=8
        )
        
        assert config.vocab_size == 4
        assert config.embedding_dim == 256
        assert config.cnn_filters == [128, 256]
        assert config.lstm_units == 256
        assert config.num_labels == 8
    
    def test_length_constraints_integration(self):
        """Test integration of length constraints in model config."""
        constraints = {
            'UMI': (8, 8),
            'ACC': (6, 6),
            'BARCODE': (12, 16)
        }
        
        length_config = LengthConstraints(
            constraints=constraints,
            constraint_weight=10.0,
            ramp_epochs=3
        )
        
        model_config = ModelConfig(length_constraints=length_config)
        
        assert model_config.length_constraints is not None
        assert model_config.length_constraints.constraints['UMI'] == (8, 8)
        assert model_config.length_constraints.constraint_weight == 10.0
        assert model_config.length_constraints.ramp_epochs == 3


class TestLengthConstraints:
    """Test LengthConstraints configuration."""
    
    def test_basic_constraints(self):
        """Test basic length constraint creation."""
        constraints = {'UMI': (8, 8), 'ACC': (6, 6)}
        config = LengthConstraints(constraints=constraints)
        
        assert config.constraints['UMI'] == (8, 8)
        assert config.constraints['ACC'] == (6, 6)
        assert config.constraint_weight == 5.0  # default
        assert config.ramp_epochs == 5  # default
    
    def test_constraints_from_dict(self):
        """Test creating constraints from dictionary (as in YAML)."""
        config_dict = {
            'constraints': {
                'UMI': [8, 8],  # Lists from YAML
                'ACC': [6, 6],
                'BARCODE': [16, 16]
            },
            'constraint_weight': 7.0,
            'ramp_epochs': 10
        }
        
        config = LengthConstraints.from_dict(config_dict)
        
        # Should convert lists to tuples
        assert isinstance(config.constraints['UMI'], tuple)
        assert config.constraints['UMI'] == (8, 8)
        assert config.constraints['ACC'] == (6, 6)
        assert config.constraint_weight == 7.0
        assert config.ramp_epochs == 10


class TestSimulationConfig:
    """Test SimulationConfig class."""
    
    def test_basic_simulation_config(self):
        """Test basic simulation configuration."""
        config = SimulationConfig(
            sequence_order=['ADAPTER', 'UMI', 'INSERT', 'ADAPTER'],
            num_sequences=1000,
            umi_length=10,
            error_rate=0.05
        )
        
        assert len(config.sequence_order) == 4
        assert config.num_sequences == 1000
        assert config.umi_length == 10
        assert config.error_rate == 0.05
    
    def test_simulation_with_seed(self):
        """Test simulation config with random seed for reproducibility."""
        config = SimulationConfig(
            num_sequences=5000,
            random_seed=42
        )
        
        assert config.random_seed == 42


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_training_config(self):
        """Test training config with defaults."""
        config = TrainingConfig()
        
        assert config.learning_rate == 0.001
        assert config.optimizer == 'adam'
        assert config.epochs == 20
        assert config.train_split == 0.8
        assert config.validation_split == 0.1
        assert config.early_stopping_patience == 5
    
    def test_custom_training_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=0.0001,
            epochs=50,
            early_stopping_patience=10,
            checkpoint_dir='./my_checkpoints'
        )
        
        assert config.learning_rate == 0.0001
        assert config.epochs == 50
        assert config.early_stopping_patience == 10
        assert config.checkpoint_dir == './my_checkpoints'


class TestEnsembleConfig:
    """Test EnsembleConfig class for BMA."""
    
    def test_default_ensemble_config(self):
        """Test default ensemble/BMA configuration."""
        config = EnsembleConfig()
        
        assert config.method == 'bma'
        assert config.num_models == 5
        assert config.vary_architecture is True
        assert config.vary_initialization is True
        assert config.prior_type == 'uniform'
    
    def test_custom_ensemble_config(self):
        """Test custom ensemble configuration."""
        config = EnsembleConfig(
            method='voting',
            num_models=10,
            vary_architecture=False,
            prior_type='performance'
        )
        
        assert config.method == 'voting'
        assert config.num_models == 10
        assert config.vary_architecture is False
        assert config.prior_type == 'performance'


class TestHybridTrainingConfig:
    """Test HybridTrainingConfig class."""
    
    def test_default_hybrid_config(self):
        """Test default hybrid training configuration."""
        config = HybridTrainingConfig()
        
        assert config.enabled is False
        assert config.warmup_epochs == 5
        assert config.discriminator_epochs == 10
        assert config.pseudolabel_epochs == 10
    
    def test_enabled_hybrid_config(self):
        """Test enabled hybrid training with custom values."""
        config = HybridTrainingConfig(
            enabled=True,
            warmup_epochs=3,
            discriminator_epochs=15,
            invalid_ratio=0.2,
            confidence_threshold=0.95
        )
        
        assert config.enabled is True
        assert config.warmup_epochs == 3
        assert config.discriminator_epochs == 15
        assert config.invalid_ratio == 0.2
        assert config.confidence_threshold == 0.95


class TestPWMConfig:
    """Test PWMConfig class."""
    
    def test_default_pwm_config(self):
        """Test default PWM configuration."""
        config = PWMConfig()
        
        assert config.pwm_file is None
        assert config.use_pwm is True
        assert config.pwm_threshold == 0.7
    
    def test_custom_pwm_config(self):
        """Test custom PWM configuration."""
        config = PWMConfig(
            pwm_file='my_acc_pwm.txt',
            use_pwm=True,
            pwm_threshold=0.8
        )
        
        assert config.pwm_file == 'my_acc_pwm.txt'
        assert config.use_pwm is True
        assert config.pwm_threshold == 0.8


class TestConfigReproducibility:
    """Test configuration reproducibility and serialization."""
    
    def test_config_round_trip(self):
        """Test saving and loading config maintains values."""
        original_config = {
            'model': {
                'vocab_size': 5,
                'embedding_dim': 128,
                'num_labels': 6
            },
            'simulation': {
                'num_sequences': 5000,
                'random_seed': 42
            },
            'training': {
                'learning_rate': 0.001,
                'epochs': 20
            }
        }
        
        # Save to temporary file
        temp_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False).name
        with open(temp_path, 'w') as f:
            yaml.dump(original_config, f)
        
        try:
            # Load back
            loaded_config = TempestConfig.from_yaml(temp_path)
            
            # Verify values match
            assert loaded_config.model.vocab_size == 5
            assert loaded_config.model.embedding_dim == 128
            assert loaded_config.simulation.random_seed == 42
            assert loaded_config.training.learning_rate == 0.001
        finally:
            os.unlink(temp_path)
    
    def test_seed_reproducibility(self):
        """Test that random seed is properly preserved."""
        config_dict = {
            'model': {'num_labels': 6},
            'simulation': {'random_seed': 12345}
        }
        
        config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False).name
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        try:
            config1 = TempestConfig.from_yaml(config_path)
            config2 = TempestConfig.from_yaml(config_path)
            
            assert config1.simulation.random_seed == config2.simulation.random_seed
            assert config1.simulation.random_seed == 12345
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
