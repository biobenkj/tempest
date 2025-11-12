"""
Tempest Model Evaluator and Comparison Tools.

This module provides specialized evaluation capabilities for Tempest models,
including standard, constraint-aware, hybrid, and ensemble approaches.
Now integrates Tempest's unified data loader (from main.py) and batch predictor
(from inference_utils) for consistent CLI compatibility.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import logging

from tempest.main import load_data
from tempest.inference import predict_sequences
from tempest.compare.evaluation_framework import ModelEvaluationFramework
from tempest.core.models import build_cnn_bilstm_crf
from tempest.core.hybrid_decoder import HybridConstraintDecoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Specialized evaluator for Tempest models incorporating domain-specific metrics."""

    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """Initialize evaluator with Tempest configuration and optional model path."""
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.setup_evaluation_params()

    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        # Default configuration
        return {
            'label_names': ['p7', 'i7', 'RP2', 'UMI', 'ACC', 'cDNA', 'polyA', 'CBC', 'RP1', 'i5', 'p5', 'PAD'],
            'length_constraints': {
                'UMI': (8, 8),
                'ACC': (6, 6),
                'CBC': (16, 16)
            },
            'model': {
                'max_seq_len': 512,
                'num_labels': 12,
                'vocab_size': 5
            }
        }

    def setup_evaluation_params(self):
        """Set up evaluation parameters from configuration."""
        self.label_names = self.config['label_names']
        self.length_constraints = self.config.get('length_constraints', {})
        self.max_seq_len = self.config['model']['max_seq_len']
        self.num_labels = self.config['model']['num_labels']
        self.label_binarizer = LabelBinarizer().fit(self.label_names)

    def load_test_data(self, data_path: str, format: str = "auto"):
        """
        Wrapper around Tempest's unified data loader.
        
        Args:
            data_path: Path to test data
            format: Data format (auto, pickle, npz, fastq, etc.)
            
        Returns:
            Loaded test dataset
        """
        return load_data(data_path, format)

    def evaluate(self, test_dataset: Union[Dict, np.ndarray, List], 
                 batch_size: int = 32, 
                 metrics: Optional[List[str]] = None, 
                 per_segment: bool = False) -> Dict[str, Any]:
        """
        Simplified single-model evaluation entry point for CLI.
        
        Args:
            test_dataset: Test data (various formats supported)
            batch_size: Batch size for prediction
            metrics: Specific metrics to compute
            per_segment: Whether to compute per-segment metrics
            
        Returns:
            Dictionary containing evaluation results
        """
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # Load model
        model = keras.models.load_model(self.model_path, compile=False)

        # Prepare input data
        X_test, y_test = self._prepare_test_data(test_dataset)

        # Predict using batch predictor
        y_pred = self.predict_batch(X_test, batch_size=batch_size)

        # Compute metrics
        results = self._compute_metrics(y_test, y_pred, metrics)

        # Add per-segment metrics if requested
        if per_segment:
            results['per_segment'] = self._compute_segment_metrics(y_test, y_pred)

        return results

    def predict_batch(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict a batch of sequences using Tempest's inference utilities.
        
        Args:
            sequences: Input sequences
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array
        """
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        model = keras.models.load_model(self.model_path, compile=False)
        return predict_sequences(model, sequences, batch_size=batch_size)

    def _prepare_test_data(self, test_dataset: Union[Dict, np.ndarray, List]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare test data from various input formats.
        
        Args:
            test_dataset: Input test data in various formats
            
        Returns:
            Tuple of (X_test, y_test) where y_test may be None
        """
        # Handle dictionary format
        if isinstance(test_dataset, dict):
            X_test = test_dataset.get('X_test', test_dataset.get('sequences'))
            y_test = test_dataset.get('y_test', test_dataset.get('labels'))
            
        # Handle list of dictionaries format
        elif isinstance(test_dataset, list) and test_dataset and isinstance(test_dataset[0], dict):
            X_test = np.array([x['sequence'] for x in test_dataset])
            y_test = np.array([x.get('labels') for x in test_dataset]) if 'labels' in test_dataset[0] else None
            
        # Handle numpy array format
        elif isinstance(test_dataset, np.ndarray):
            X_test = test_dataset
            y_test = None
            
        # Handle tuple format (X, y)
        elif isinstance(test_dataset, tuple) and len(test_dataset) == 2:
            X_test, y_test = test_dataset
            
        else:
            raise ValueError(f"Unsupported test dataset format: {type(test_dataset)}")
        
        return X_test, y_test

    def _compute_metrics(self, y_true: Optional[np.ndarray], 
                        y_pred: np.ndarray, 
                        metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels (may be None)
            y_pred: Predicted labels
            metrics: Specific metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        
        if y_true is None:
            logger.warning("No ground truth labels provided - skipping metrics computation")
            return {'predictions_shape': y_pred.shape}
        
        # Flatten arrays for metrics computation
        mask = y_true > 0  # Ignore padding
        y_true_flat = y_true[mask]
        y_pred_flat = y_pred[mask] if len(y_pred.shape) == len(y_true.shape) else y_pred.flatten()
        
        # Default metrics
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        # Compute requested metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        
        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        
        if 'precision' in metrics:
            results['precision'] = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        
        if 'recall' in metrics:
            results['recall'] = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        
        return results

    def _compute_segment_metrics(self, y_true: Optional[np.ndarray], 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute per-segment accuracy metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with per-segment accuracies
        """
        if y_true is None:
            return {}
        
        segment_metrics = {}
        
        for i, label_name in enumerate(self.label_names):
            if label_name == 'PAD':
                continue
            
            # Find positions with this label
            mask = y_true == i
            if np.any(mask):
                segment_acc = accuracy_score(y_true[mask], y_pred[mask])
                segment_metrics[label_name] = segment_acc
        
        return segment_metrics

    def load_models(self, models_dir: str) -> Dict:
        """Load multiple models from a directory for comparison."""
        models_dir = Path(models_dir)
        models = {}
        
        # Standard model
        standard_path = models_dir / 'standard_model.h5'
        if standard_path.exists():
            logger.info(f"Loading standard model from {standard_path}")
            model = keras.models.load_model(standard_path, compile=False)
            decoder = self._create_standard_decoder(model)
            models['Standard'] = (model, decoder, 'standard')
        
        # Soft constraint model
        soft_path = models_dir / 'soft_constraint_model.h5'
        if soft_path.exists():
            logger.info(f"Loading soft constraint model from {soft_path}")
            model = keras.models.load_model(soft_path, compile=False)
            decoder = HybridConstraintDecoder(
                model=model,
                label_binarizer=self.label_binarizer,
                length_constraints=self.length_constraints,
                use_hard_constraints=False
            )
            models['Soft Constraints'] = (model, decoder, 'soft_constraint')
        
        # Hard constraint model
        hard_path = models_dir / 'hard_constraint_model.h5'
        if hard_path.exists():
            logger.info(f"Loading hard constraint model from {hard_path}")
            model = keras.models.load_model(hard_path, compile=False)
            decoder = HybridConstraintDecoder(
                model=model,
                label_binarizer=self.label_binarizer,
                length_constraints=self.length_constraints,
                use_hard_constraints=True
            )
            models['Hard Constraints'] = (model, decoder, 'hard_constraint')
        
        # Hybrid model
        hybrid_path = models_dir / 'hybrid_model.h5'
        if hybrid_path.exists():
            logger.info(f"Loading hybrid model from {hybrid_path}")
            model = keras.models.load_model(hybrid_path, compile=False)
            decoder = HybridConstraintDecoder(
                model=model,
                label_binarizer=self.label_binarizer,
                length_constraints=self.length_constraints,
                use_hard_constraints=True
            )
            models['Hybrid'] = (model, decoder, 'hybrid')
        
        # Ensemble models
        ensemble_path = models_dir / 'ensemble'
        if ensemble_path.exists():
            logger.info(f"Loading ensemble models from {ensemble_path}")
            ensemble_model = self._load_ensemble_models(ensemble_path)
            if ensemble_model:
                models['Ensemble'] = (ensemble_model, ensemble_model, 'ensemble')
        
        logger.info(f"Loaded {len(models)} models")
        return models

    def _create_standard_decoder(self, model):
        """Create a standard decoder wrapper for consistency."""
        class StandardDecoder:
            def __init__(self, model):
                self.model = model
            
            def decode(self, X):
                predictions = self.model.predict(X)
                if len(predictions.shape) == 3:
                    return np.argmax(predictions, axis=-1)
                return predictions
        
        return StandardDecoder(model)

    def _load_ensemble_models(self, ensemble_dir):
        """Load ensemble models from directory."""
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
            
            def decode(self, X):
                predictions = []
                for m in self.models:
                    pred = m.predict(X)
                    if len(pred.shape) == 3:
                        pred = np.argmax(pred, axis=-1)
                    predictions.append(pred)
                
                # Majority voting
                stacked = np.stack(predictions)
                from scipy import stats
                return stats.mode(stacked, axis=0)[0].squeeze()
        
        models = []
        for model_file in sorted(ensemble_dir.glob('model_*.h5')):
            logger.info(f"  Loading ensemble member: {model_file.name}")
            model = keras.models.load_model(model_file, compile=False)
            models.append(model)
        
        return EnsembleModel(models) if models else None

    def evaluate_all_models(self, models: Dict, X_test: np.ndarray, 
                           y_test: Optional[np.ndarray] = None) -> ModelEvaluationFramework:
        """
        Evaluate all loaded models using the evaluation framework.
        
        Args:
            models: Dictionary of loaded models
            X_test: Test sequences
            y_test: Test labels (optional)
            
        Returns:
            ModelEvaluationFramework with results
        """
        framework = ModelEvaluationFramework(
            label_names=self.label_names,
            length_constraints=self.length_constraints
        )
        
        for model_name, (model, decoder, model_type) in models.items():
            logger.info(f"Evaluating {model_name}...")
            framework.evaluate_model(
                model_name=model_name,
                model=decoder,
                X_test=X_test,
                y_test=y_test,
                model_type=model_type
            )
        
        return framework

    def generate_detailed_comparison(self, framework: ModelEvaluationFramework, 
                                    output_dir: str):
        """
        Generate detailed comparison reports and visualizations.
        
        Args:
            framework: Evaluation framework with results
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save report
        report = framework.generate_report()
        
        # Save JSON report
        json_path = output_dir / 'comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report to {json_path}")
        
        # Save CSV summary
        if hasattr(framework, 'results_df') and framework.results_df is not None:
            csv_path = output_dir / 'comparison_summary.csv'
            framework.results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV summary to {csv_path}")
        
        # Generate visualizations
        self._generate_comparison_plots(framework, output_dir)

    def _generate_comparison_plots(self, framework: ModelEvaluationFramework, 
                                   output_dir: Path):
        """Generate comparison visualizations."""
        try:
            # Performance comparison bar chart
            if hasattr(framework, 'results_df') and framework.results_df is not None:
                df = framework.results_df
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Accuracy comparison
                ax = axes[0, 0]
                df.plot(x='model', y='accuracy', kind='bar', ax=ax, legend=False)
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                
                # F1 Score comparison
                ax = axes[0, 1]
                df.plot(x='model', y='f1_score', kind='bar', ax=ax, legend=False, color='orange')
                ax.set_title('F1 Score Comparison')
                ax.set_ylabel('F1 Score')
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                
                # Precision comparison
                ax = axes[1, 0]
                df.plot(x='model', y='precision', kind='bar', ax=ax, legend=False, color='green')
                ax.set_title('Precision Comparison')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                
                # Recall comparison
                ax = axes[1, 1]
                df.plot(x='model', y='recall', kind='bar', ax=ax, legend=False, color='red')
                ax.set_title('Recall Comparison')
                ax.set_ylabel('Recall')
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                
                plt.tight_layout()
                plot_path = output_dir / 'model_comparison.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved comparison plot to {plot_path}")
                
        except Exception as e:
            logger.warning(f"Could not generate comparison plots: {e}")


def compare_models(models_dir: str, test_data_path: str, 
                   config_path: Optional[str] = None, 
                   output_dir: str = './evaluation_results') -> ModelEvaluationFramework:
    """
    Main function to compare multiple models.
    
    Args:
        models_dir: Directory containing models
        test_data_path: Path to test data
        config_path: Configuration file path
        output_dir: Output directory for results
        
    Returns:
        ModelEvaluationFramework with comparison results
    """
    logger.info("Initializing evaluator...")
    evaluator = ModelEvaluator(config_path=config_path)
    
    logger.info(f"Loading models from {models_dir}...")
    models = evaluator.load_models(models_dir)
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    
    logger.info(f"Loading test data from {test_data_path}...")
    test_data = load_data(test_data_path)
    
    # Prepare test data
    X_test, y_test = evaluator._prepare_test_data(test_data)
    
    # Evaluate all models
    framework = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # Generate detailed comparison
    evaluator.generate_detailed_comparison(framework, output_dir)
    
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")
    return framework
