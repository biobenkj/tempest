"""
Pytest configuration for Tempest tests with GPU support.

This module provides shared fixtures, GPU detection, and configuration
for all test modules in the Tempest test suite.
"""

import pytest
import os
import sys
import logging
import tempfile
from pathlib import Path
import yaml
import tensorflow as tf
import torch
import numpy as np
import warnings
from typing import Dict, Any, Optional

# Add tempest to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# GPU Configuration and Detection
class GPUConfig:
    """GPU configuration and detection utilities."""
    
    @staticmethod
    def detect_gpus():
        """Detect available GPUs using both TensorFlow and PyTorch."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'framework': None,
            'memory_info': []
        }
        
        # Check TensorFlow GPUs
        try:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                gpu_info['available'] = True
                gpu_info['count'] = len(tf_gpus)
                gpu_info['framework'] = 'tensorflow'
                
                for gpu in tf_gpus:
                    gpu_info['devices'].append({
                        'name': gpu.name,
                        'device_type': 'GPU',
                        'framework': 'tensorflow'
                    })
                    
                    # Get memory info if possible
                    try:
                        memory_info = tf.config.experimental.get_memory_info(gpu)
                        gpu_info['memory_info'].append(memory_info)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"TensorFlow GPU detection failed: {e}")
        
        # Check PyTorch GPUs (if needed for certain components)
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                cuda_count = torch.cuda.device_count()
                if cuda_count > gpu_info['count']:
                    gpu_info['count'] = cuda_count
                
                for i in range(cuda_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'name': props.name,
                        'device_type': 'CUDA',
                        'framework': 'pytorch',
                        'total_memory': props.total_memory,
                        'major': props.major,
                        'minor': props.minor
                    })
        except Exception as e:
            logger.warning(f"PyTorch GPU detection failed: {e}")
        
        return gpu_info
    
    @staticmethod
    def configure_tensorflow_gpus(memory_growth=True, memory_limit=None):
        """Configure TensorFlow GPU settings."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    if memory_limit:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )]
                        )
                logger.info(f"Configured {len(gpus)} TensorFlow GPU(s)")
                return True
            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")
                return False
        return False


# Pytest Fixtures
@pytest.fixture(scope="session")
def gpu_config():
    """Session-wide GPU configuration fixture."""
    config = GPUConfig()
    gpu_info = config.detect_gpus()
    
    # Configure TensorFlow GPUs if available
    if gpu_info['available']:
        config.configure_tensorflow_gpus(memory_growth=True)
    
    logger.info(f"GPU Detection Results: {gpu_info}")
    return gpu_info


@pytest.fixture(scope="session")
def require_gpu(gpu_config):
    """Skip test if GPU is not available."""
    if not gpu_config['available']:
        pytest.skip("GPU not available")
    return gpu_config


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'model': {
            'vocab_size': 5,
            'embedding_dim': 128,
            'num_labels': 6,
            'max_seq_len': 256,
            'batch_size': 32,
            'dropout_rate': 0.1,
            'l2_reg': 0.01,
            'use_crf': True,
            'crf_loss_weight': 1.0,
            'lstm_units': 256,
            'num_lstm_layers': 2,
            'attention_heads': 8
        },
        'training': {
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'optimizer': 'adam',
            'early_stopping_patience': 5,
            'reduce_lr_patience': 3,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs'
        },
        'simulation': {
            'num_sequences': 1000,
            'random_seed': 42,
            'barcode_length': 16,
            'umi_length': 12,
            'variable_region_length': 30,
            'read_length': 150,
            'error_rate': 0.01
        },
        'pwm': {
            'barcode_pwm_path': None,
            'umi_pwm_path': None,
            'adapter_pwm_path': None,
            'min_score_threshold': 0.8
        },
        'length_constraints': {
            'enable': True,
            'barcode_length': 16,
            'umi_length': 12,
            'adapter_lengths': [10, 14]
        },
        'ensemble': {
            'num_models': 3,
            'model_types': ['standard', 'hybrid'],
            'combination_method': 'bma',
            'approximation': 'laplace'
        }
    }


@pytest.fixture
def sample_config_file(sample_config, temp_dir):
    """Create a temporary config file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_sequences(temp_dir):
    """Create sample sequence data for testing."""
    sequences = [
        "ATCGATCGATCGATCG:TAGCTAGCTAGC:GCTAGCTAGCTAGCTAGCTAGC",
        "GCTAGCTAGCTAGCTA:ATCGATCGATCG:TAGCTAGCTAGCTAGCTAGCTA",
        "TAGCTAGCTAGCTAGC:GCTAGCTAGCTA:ATCGATCGATCGATCGATCGAT",
    ]
    
    data_file = temp_dir / "sample_sequences.txt"
    with open(data_file, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    
    return data_file


@pytest.fixture
def mock_model_path(temp_dir):
    """Create a mock model file for testing."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir(exist_ok=True)
    
    # Create dummy model files
    (model_dir / "model.h5").touch()
    (model_dir / "config.yaml").touch()
    
    # Create a simple config
    config = {
        'model_type': 'standard',
        'vocab_size': 5,
        'num_labels': 6
    }
    
    with open(model_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return model_dir


@pytest.fixture
def gpu_memory_monitor():
    """Monitor GPU memory usage during tests."""
    class GPUMemoryMonitor:
        def __init__(self):
            self.initial_memory = None
            self.peak_memory = None
            
        def start(self):
            """Start monitoring GPU memory."""
            if tf.config.list_physical_devices('GPU'):
                try:
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Get initial memory usage
                    gpu = tf.config.list_physical_devices('GPU')[0]
                    self.initial_memory = tf.config.experimental.get_memory_info(gpu)
                except:
                    pass
        
        def stop(self):
            """Stop monitoring and return memory usage."""
            if tf.config.list_physical_devices('GPU'):
                try:
                    gpu = tf.config.list_physical_devices('GPU')[0]
                    final_memory = tf.config.experimental.get_memory_info(gpu)
                    
                    if self.initial_memory and final_memory:
                        memory_used = final_memory['current'] - self.initial_memory['current']
                        return {
                            'memory_used_bytes': memory_used,
                            'memory_used_mb': memory_used / (1024 * 1024),
                            'peak_memory': final_memory.get('peak', None)
                        }
                except:
                    pass
            return None
    
    return GPUMemoryMonitor()


# Pytest Markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )


# Test Collection Hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark GPU tests
        if 'gpu' in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Auto-mark slow tests
        if 'slow' in item.nodeid.lower() or 'integration' in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
