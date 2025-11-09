#!/usr/bin/env python3
"""
Test runner and integration demonstration for the refactored simulate module.

This script demonstrates:
1. Running unit tests for the simulate module
2. Integration between main.py -> cli.py -> simulate.py
3. Using config.yaml vs command line arguments
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import yaml
import subprocess
import logging
from typing import Dict, Any

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config(output_dir: Path) -> Path:
    """Create a test configuration file."""
    config = {
        'model': {
            'max_seq_len': 1500,
            'num_labels': 9,  # Simplified without RP1/RP2
            'embedding_dim': 128,
            'lstm_units': 256,
            'lstm_layers': 2,
            'dropout': 0.3,
            'use_cnn': True,
            'use_bilstm': True,
            'batch_size': 32
        },
        'simulation': {
            'num_sequences': 100,
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
                    'p7': 24,
                    'i7': 8,
                    'UMI': 8,
                    'ACC': 6,
                    'cDNA': 500,
                    'polyA': 30,
                    'CBC': 6,
                    'i5': 8,
                    'p5': 29
                },
                'generation_mode': {
                    'p7': 'fixed',
                    'i7': 'whitelist',
                    'UMI': 'random',
                    'ACC': 'pwm',
                    'cDNA': 'transcript',
                    'polyA': 'polya',
                    'CBC': 'whitelist',
                    'i5': 'whitelist',
                    'p5': 'fixed'
                }
            },
            'whitelist_files': {},  # Empty for testing
            'pwm_files': {},  # Empty for testing
            'pwm': {
                'temperature': 1.2,
                'min_entropy': 0.1,
                'diversity_boost': 1.0,
                'pattern': 'ACCSSV'
            }
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
    }
    
    config_path = output_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def test_unit_tests():
    """Run the unit tests for the simulate module."""
    logger.info("=" * 60)
    logger.info("Running Unit Tests")
    logger.info("=" * 60)
    
    test_file = PROJECT_ROOT / "tests" / "commands" / "test_simulate_refactored.py"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✓ All unit tests passed!")
        return True
    else:
        logger.error("✗ Unit tests failed!")
        logger.error(result.stdout)
        logger.error(result.stderr)
        return False


def test_cli_integration():
    """Test CLI integration using the refactored simulate module."""
    logger.info("=" * 60)
    logger.info("Testing CLI Integration")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config
        config_path = create_test_config(temp_path)
        logger.info(f"Created test config: {config_path}")
        
        # Test 1: Validate configuration
        logger.info("\n1. Testing configuration validation...")
        cmd = [
            sys.executable, "-m", "tempest.cli",
            "simulate", "validate",
            "-c", str(config_path),
            "--verbose"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if "Configuration is valid!" in result.stdout:
            logger.info("✓ Configuration validation successful")
        else:
            logger.error("✗ Configuration validation failed")
            logger.error(result.stdout)
            return False
        
        # Test 2: Generate sequences
        logger.info("\n2. Testing sequence generation...")
        output_file = temp_path / "sequences.txt"
        cmd = [
            sys.executable, "-m", "tempest.cli",
            "simulate", "generate",
            "-c", str(config_path),
            "-o", str(output_file),
            "-n", "50",
            "--seed", "12345"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if output_file.exists() and "Simulation complete!" in result.stdout:
            logger.info("✓ Sequence generation successful")
            with open(output_file, 'r') as f:
                lines = f.readlines()
            logger.info(f"  Generated {len(lines)} sequences")
        else:
            logger.error("✗ Sequence generation failed")
            logger.error(result.stdout)
            return False
        
        # Test 3: Generate with train/val split
        logger.info("\n3. Testing train/validation split...")
        split_dir = temp_path / "split"
        split_dir.mkdir()
        cmd = [
            sys.executable, "-m", "tempest.cli",
            "simulate", "generate",
            "-c", str(config_path),
            "--split",
            "-d", str(split_dir),
            "-n", "100",
            "--train-fraction", "0.7",
            "--seed", "42"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        train_file = split_dir / "train.txt"
        val_file = split_dir / "val.txt"
        
        if train_file.exists() and val_file.exists():
            logger.info("✓ Train/validation split successful")
            with open(train_file, 'r') as f:
                train_lines = f.readlines()
            with open(val_file, 'r') as f:
                val_lines = f.readlines()
            logger.info(f"  Training: {len(train_lines)} sequences")
            logger.info(f"  Validation: {len(val_lines)} sequences")
        else:
            logger.error("✗ Train/validation split failed")
            logger.error(result.stdout)
            return False
        
        # Test 4: Run statistics on generated sequences
        logger.info("\n4. Testing sequence statistics...")
        cmd = [
            sys.executable, "-m", "tempest.cli",
            "simulate", "stats",
            str(output_file),
            "--verbose"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if "Statistics for" in result.stdout and "Sequence lengths:" in result.stdout:
            logger.info("✓ Statistics calculation successful")
        else:
            logger.error("✗ Statistics calculation failed")
            logger.error(result.stdout)
            return False
    
    return True


def test_main_integration():
    """Test integration through main.py entry point."""
    logger.info("=" * 60)
    logger.info("Testing main.py Integration")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config
        config_path = create_test_config(temp_path)
        logger.info(f"Created test config: {config_path}")
        
        # Test using main.py directly
        logger.info("\nTesting direct main.py invocation...")
        
        # Import and use main.py
        from tempest.main import run_pipeline, load_config
        
        try:
            # Load config
            config = load_config(config_path)
            logger.info("✓ Config loaded successfully")
            
            # Run simulation through main.py
            output_dir = temp_path / "main_output"
            output_dir.mkdir()
            
            result = run_pipeline(
                command="simulate",
                config_path=config_path,
                output_dir=output_dir,
                extra_args={
                    'num_sequences': 25,
                    'seed': 999,
                    'split': False,
                    'output_file': 'main_sequences.txt'
                }
            )
            
            output_file = output_dir / 'main_sequences.txt'
            if output_file.exists():
                logger.info("✓ main.py pipeline execution successful")
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                logger.info(f"  Generated {len(lines)} sequences through main.py")
            else:
                logger.error("✗ main.py pipeline execution failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ main.py integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_config_vs_cli_args():
    """Test that CLI arguments properly override config values."""
    logger.info("=" * 60)
    logger.info("Testing Config vs CLI Arguments")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create config with specific values
        config_path = create_test_config(temp_path)
        
        # Modify config to have different values
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['simulation']['num_sequences'] = 1000
        config['simulation']['random_seed'] = 111
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info("Config values:")
        logger.info(f"  num_sequences: {config['simulation']['num_sequences']}")
        logger.info(f"  random_seed: {config['simulation']['random_seed']}")
        
        # Run with CLI overrides
        output_file = temp_path / "override_test.txt"
        cmd = [
            sys.executable, "-m", "tempest.cli",
            "simulate", "generate",
            "-c", str(config_path),
            "-o", str(output_file),
            "-n", "50",  # Override num_sequences
            "--seed", "999"  # Override random_seed
        ]
        
        logger.info("\nCLI override values:")
        logger.info("  num_sequences: 50 (should override 1000)")
        logger.info("  random_seed: 999 (should override 111)")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 50:
                logger.info("✓ CLI arguments successfully overrode config values")
                logger.info(f"  Generated {len(lines)} sequences (as per CLI, not config's 1000)")
            else:
                logger.error("✗ CLI arguments did not override properly")
                logger.error(f"  Expected 50 sequences, got {len(lines)}")
                return False
        else:
            logger.error("✗ Command failed to generate output")
            logger.error(result.stdout)
            return False
    
    return True


def main():
    """Main test runner."""
    logger.info("=" * 60)
    logger.info("TEMPEST Simulate Module Test Suite")
    logger.info("=" * 60)
    
    # Track results
    results = {}
    
    # Run tests
    logger.info("\nStarting test suite...\n")
    
    # 1. Unit tests
    results['unit_tests'] = test_unit_tests()
    
    # 2. CLI integration
    results['cli_integration'] = test_cli_integration()
    
    # 3. Main.py integration
    results['main_integration'] = test_main_integration()
    
    # 4. Config vs CLI arguments
    results['config_vs_cli'] = test_config_vs_cli_args()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
