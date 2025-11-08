"""
Pytest configuration for Tempest tests with 11-segment architecture support.

Provides fixtures and configuration for testing the complex read structure:
p7, i7, RP2, UMI, ACC, cDNA, polyA, CBC, RP1, i5, p5
"""

import pytest
import os
import sys
import logging
import tempfile
from pathlib import Path
import yaml
import tensorflow as tf
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
    """GPU configuration and detection utilities for TensorFlow."""
    
    @staticmethod
    def detect_gpus():
        """Detect available GPUs using TensorFlow."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'framework': 'tensorflow',
            'memory_info': []
        }
        
        # Check TensorFlow GPUs
        try:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                gpu_info['available'] = True
                gpu_info['count'] = len(tf_gpus)
                
                for gpu in tf_gpus:
                    gpu_details = {
                        'name': gpu.name,
                        'device_type': gpu.device_type
                    }
                    
                    # Get detailed GPU info if available
                    try:
                        # Try to get memory info
                        memory_info = tf.config.experimental.get_memory_info(gpu.name)
                        gpu_details['memory'] = memory_info
                        gpu_info['memory_info'].append(memory_info)
                    except Exception:
                        # Memory info might not be available in all configurations
                        pass
                    
                    gpu_info['devices'].append(gpu_details)
                    
                logger.info(f"Detected {len(tf_gpus)} TensorFlow GPU(s)")
            else:
                logger.info("No GPUs detected by TensorFlow")
        except Exception as e:
            logger.warning(f"TensorFlow GPU detection failed: {e}")
        
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
    
    @staticmethod
    def get_gpu_memory_usage():
        """Get current GPU memory usage from TensorFlow."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            memory_info = []
            for gpu in gpus:
                try:
                    info = tf.config.experimental.get_memory_info(gpu.name)
                    memory_info.append({
                        'device': gpu.name,
                        'current': info.get('current', 0),
                        'peak': info.get('peak', 0)
                    })
                except Exception:
                    pass
            return memory_info
        return []


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
    """Monitor GPU memory usage during tests using TensorFlow."""
    class GPUMemoryMonitor:
        def __init__(self):
            self.initial_memory = None
            self.peak_memory = None
            self.gpus = tf.config.list_physical_devices('GPU')
            
        def start(self):
            """Start monitoring GPU memory."""
            if self.gpus:
                try:
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear TensorFlow session
                    tf.keras.backend.clear_session()
                    
                    # Get initial memory usage for all GPUs
                    self.initial_memory = {}
                    for gpu in self.gpus:
                        try:
                            memory_info = tf.config.experimental.get_memory_info(gpu.name)
                            self.initial_memory[gpu.name] = memory_info
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Could not start memory monitoring: {e}")
        
        def stop(self):
            """Stop monitoring and return memory usage."""
            if self.gpus and self.initial_memory:
                try:
                    memory_stats = []
                    
                    for gpu in self.gpus:
                        if gpu.name in self.initial_memory:
                            try:
                                final_memory = tf.config.experimental.get_memory_info(gpu.name)
                                initial = self.initial_memory[gpu.name]
                                
                                memory_used = final_memory.get('current', 0) - initial.get('current', 0)
                                peak = final_memory.get('peak', 0)
                                
                                stats = {
                                    'device': gpu.name,
                                    'memory_used_bytes': memory_used,
                                    'memory_used_mb': memory_used / (1024 * 1024),
                                    'peak_memory_bytes': peak,
                                    'peak_memory_mb': peak / (1024 * 1024)
                                }
                                memory_stats.append(stats)
                            except Exception:
                                pass
                    
                    # Return aggregated stats if multiple GPUs
                    if memory_stats:
                        total_used = sum(s['memory_used_bytes'] for s in memory_stats)
                        max_peak = max(s['peak_memory_bytes'] for s in memory_stats)
                        
                        return {
                            'devices': memory_stats,
                            'total_memory_used_bytes': total_used,
                            'total_memory_used_mb': total_used / (1024 * 1024),
                            'max_peak_memory_bytes': max_peak,
                            'max_peak_memory_mb': max_peak / (1024 * 1024),
                            'num_gpus': len(memory_stats)
                        }
                except Exception as e:
                    logger.warning(f"Could not get final memory stats: {e}")
            
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
