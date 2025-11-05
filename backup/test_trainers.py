#!/usr/bin/env python3
"""
Test script for verifying single model and ensemble trainers.

This script:
1. Generates simulated training data
2. Tests the single model trainer
3. Tests the ensemble trainer
4. Compares performance between single model and ensemble
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from pathlib import Path
import json
import time

# Tempest imports
from tempest.data.simulator import SequenceSimulator, SimulatedRead
from tempest.training.trainer import ModelTrainer
from tempest.training.ensemble import EnsembleTrainer
from tempest.utils.config import TempestConfig
from tempest.utils.io import ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a test configuration for training."""
    config = {
        'model': {
            'vocab_size': 5,
            'embedding_dim': 64,
            'use_cnn': True,
            'cnn_filters': [32, 64],
            'cnn_kernels': [3, 5],
            'use_bilstm': True,
            'lstm_units': 64,
            'lstm_layers': 1,
            'dropout': 0.3,
            'num_labels': 6,  # ADAPTER, UMI, ACC, BARCODE, INSERT, POLYA
            'use_crf': False,  # Disable CRF for simplicity
            'max_seq_len': 256,
            'batch_size': 16
        },
        'simulation': {
            'num_train': 100,  # Small dataset for testing
            'num_val': 20,
            'sequence_order': ['ADAPTER', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'POLYA'],
            'segments': {
                'ADAPTER': {
                    'type': 'fixed',
                    'sequence': 'ATCGATCG',
                    'label': 'ADAPTER'
                },
                'UMI': {
                    'type': 'random',
                    'length': 8,
                    'label': 'UMI'
                },
                'ACC': {
                    'type': 'pwm',
                    'length': 6,
                    'pwm_file': None,  # Will use random for testing
                    'label': 'ACC'
                },
                'BARCODE': {
                    'type': 'whitelist',
                    'whitelist': ['AAACCCGGGTTTTTT', 'CCCGGGTTTAAACCC', 'GGGAAATTTCCCGGG'],
                    'label': 'BARCODE'
                },
                'INSERT': {
                    'type': 'random',
                    'min_length': 50,
                    'max_length': 100,
                    'label': 'INSERT'
                },
                'POLYA': {
                    'type': 'polya',
                    'min_length': 10,
                    'max_length': 30,
                    'label': 'POLYA'
                }
            },
            'error_rate': 0.01,
            'indel_rate': 0.001
        },
        'training': {
            'epochs': 5,  # Few epochs for testing
            'learning_rate': 0.001,
            'early_stopping_patience': 3,
            'reduce_lr_patience': 2
        },
        'ensemble': {
            'num_models': 3,  # Small ensemble for testing
            'variation_type': 'both',
            'bma_prior': 'performance',
            'bma_temperature': 1.0
        }
    }
    
    # Create TempestConfig object
    config_obj = TempestConfig()
    
    # Set attributes from dictionary
    for key, value in config.items():
        if isinstance(value, dict):
            # Create nested config objects
            nested_obj = type('Config', (), {})()
            for k, v in value.items():
                setattr(nested_obj, k, v)
            setattr(config_obj, key, nested_obj)
        else:
            setattr(config_obj, key, value)
    
    return config_obj


def generate_test_data(config):
    """Generate test data for training."""
    logger.info("Generating test data...")
    
    # Create simulator
    simulator = SequenceSimulator(config.simulation.__dict__)
    
    # Generate training data
    train_reads = []
    for i in range(config.simulation.num_train):
        read = simulator.generate_read()
        train_reads.append(read)
    
    # Generate validation data
    val_reads = []
    for i in range(config.simulation.num_val):
        read = simulator.generate_read()
        val_reads.append(read)
    
    logger.info(f"Generated {len(train_reads)} training and {len(val_reads)} validation reads")
    
    # Print sample read
    if train_reads:
        sample = train_reads[0]
        logger.info("\nSample read:")
        logger.info(f"  Sequence (first 50bp): {sample.sequence[:50]}...")
        logger.info(f"  Labels (first 50): {sample.labels[:50]}")
        logger.info(f"  Label regions: {list(sample.label_regions.keys())}")
    
    return train_reads, val_reads


def test_single_model_trainer(config, train_reads, val_reads):
    """Test the single model trainer."""
    logger.info("\n" + "="*80)
    logger.info("TESTING SINGLE MODEL TRAINER")
    logger.info("="*80)
    
    # Create trainer
    checkpoint_dir = "test_single_model"
    trainer = ModelTrainer(config, checkpoint_dir=checkpoint_dir)
    
    # Train model
    start_time = time.time()
    model = trainer.train(train_reads, val_reads)
    training_time = time.time() - start_time
    
    logger.info(f"\nSingle model training completed in {training_time:.2f} seconds")
    
    # Make predictions on validation data
    predictions = trainer.predict(val_reads[:5])  # Test on first 5 validation reads
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    
    # Test model saving and loading
    logger.info("\nTesting model save/load...")
    model_path = Path(checkpoint_dir) / "test_model.h5"
    trainer.save_model(str(model_path))
    
    # Create new trainer and load model
    new_trainer = ModelTrainer(config, checkpoint_dir=checkpoint_dir)
    new_trainer.load_model(str(model_path))
    
    logger.info("Model successfully saved and loaded")
    
    return trainer.best_val_accuracy


def test_ensemble_trainer(config, train_reads, val_reads):
    """Test the ensemble trainer."""
    logger.info("\n" + "="*80)
    logger.info("TESTING ENSEMBLE TRAINER")
    logger.info("="*80)
    
    # Create ensemble trainer
    checkpoint_dir = "test_ensemble"
    ensemble_trainer = EnsembleTrainer(
        config,
        num_models=config.ensemble.num_models,
        variation_type=config.ensemble.variation_type,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train ensemble
    start_time = time.time()
    models = ensemble_trainer.train(train_reads, val_reads)
    training_time = time.time() - start_time
    
    logger.info(f"\nEnsemble training completed in {training_time:.2f} seconds")
    logger.info(f"Trained {len(models)} models")
    
    # Evaluate ensemble
    metrics = ensemble_trainer.evaluate(val_reads)
    
    logger.info("\nEnsemble Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Test predictions with uncertainty
    predictions, uncertainty = ensemble_trainer.predict(
        val_reads[:5], 
        return_uncertainty=True
    )
    
    logger.info(f"\nPredictions shape: {predictions.shape}")
    logger.info(f"Mean entropy: {np.mean(uncertainty['entropy']):.4f}")
    logger.info(f"Mean model disagreement: {np.mean(uncertainty['model_disagreement']):.6f}")
    
    # Test ensemble saving and loading
    logger.info("\nTesting ensemble save/load...")
    ensemble_trainer.save_ensemble()
    
    # Create new trainer and load ensemble
    new_ensemble = EnsembleTrainer(config, checkpoint_dir=checkpoint_dir)
    new_ensemble.load_ensemble(checkpoint_dir)
    
    logger.info("Ensemble successfully saved and loaded")
    
    return metrics['ensemble_accuracy']


def compare_performance(single_acc, ensemble_acc):
    """Compare single model vs ensemble performance."""
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    logger.info(f"Single Model Accuracy: {single_acc:.4f}")
    logger.info(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    
    improvement = (ensemble_acc - single_acc) / single_acc * 100
    if improvement > 0:
        logger.info(f"Ensemble improvement: +{improvement:.2f}%")
    else:
        logger.info(f"Ensemble change: {improvement:.2f}%")
    
    # Create summary report
    summary = {
        'single_model_accuracy': float(single_acc),
        'ensemble_accuracy': float(ensemble_acc),
        'improvement_percent': float(improvement),
        'test_status': 'PASSED'
    }
    
    # Save summary
    with open('test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTest summary saved to test_summary.json")


def main():
    """Main test function."""
    logger.info("Starting Tempest Trainer Tests")
    logger.info("="*80)
    
    try:
        # Create test configuration
        config = create_test_config()
        
        # Generate test data
        train_reads, val_reads = generate_test_data(config)
        
        # Test single model trainer
        single_acc = test_single_model_trainer(config, train_reads, val_reads)
        
        # Test ensemble trainer
        ensemble_acc = test_ensemble_trainer(config, train_reads, val_reads)
        
        # Compare performance
        compare_performance(single_acc, ensemble_acc)
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
