"""
Tempest Model Evaluator and Comparison Tools.

This module provides specialized evaluation capabilities for Tempest models,
including standard, constraint-aware, hybrid, and ensemble approaches.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import logging

from .evaluation_framework import ModelEvaluationFramework
from ..core.models import build_cnn_bilstm_crf
from ..core.hybrid_decoder import HybridConstraintDecoder
from sklearn.preprocessing import LabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TempestModelEvaluator:
    """
    Specialized evaluator for Tempest models incorporating domain-specific metrics.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize evaluator with Tempest configuration.
        
        Args:
            config_path: Path to Tempest config file
        """
        self.config = self._load_config(config_path)
        self.setup_evaluation_params()
        
    def _load_config(self, config_path):
        """Load Tempest configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'label_names': ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3', 'PAD'],
                'length_constraints': {
                    'UMI': (8, 8),
                    'ACC': (6, 6),
                    'BARCODE': (16, 16)
                },
                'model': {
                    'max_seq_len': 512,
                    'num_labels': 7,
                    'vocab_size': 5
                }
            }
    
    def setup_evaluation_params(self):
        """Setup evaluation parameters from config."""
        self.label_names = self.config['label_names']
        self.length_constraints = self.config.get('length_constraints', {})
        self.max_seq_len = self.config['model']['max_seq_len']
        self.num_labels = self.config['model']['num_labels']
        
        # Initialize label binarizer
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(self.label_names)
        
    def load_models(self, models_dir: str) -> Dict:
        """
        Load all trained models from directory.
        
        Args:
            models_dir: Directory containing trained models
            
        Returns:
            Dictionary mapping model_name -> (model, decoder, type)
        """
        models_dir = Path(models_dir)
        models = {}
        
        # Load standard model
        standard_path = models_dir / 'standard_model.h5'
        if standard_path.exists():
            logger.info(f"Loading standard model from {standard_path}")
            model = keras.models.load_model(standard_path, compile=False)
            decoder = self._create_standard_decoder(model)
            models['Standard'] = (model, decoder, 'standard')
        
        # Load soft constraint model
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
        
        # Load hard constraint model
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
        
        # Load hybrid model
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
        
        # Load ensemble models
        ensemble_path = models_dir / 'ensemble'
        if ensemble_path.exists():
            logger.info(f"Loading ensemble models from {ensemble_path}")
            ensemble_model = self._load_ensemble_models(ensemble_path)
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
        """Load and wrap ensemble models."""
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
            
            def decode(self, X):
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    if len(pred.shape) == 3:
                        pred = np.argmax(pred, axis=-1)
                    predictions.append(pred)
                
                # Majority voting
                stacked = np.stack(predictions)
                from scipy import stats
                return stats.mode(stacked, axis=0)[0].squeeze()
            
            def get_model_predictions(self, X):
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    if len(pred.shape) == 3:
                        pred = np.argmax(pred, axis=-1)
                    predictions.append(pred)
                return predictions
            
            def get_prediction_uncertainty(self, X):
                predictions = self.get_model_predictions(X)
                stacked = np.stack(predictions)
                # Calculate entropy as uncertainty
                from scipy import stats
                entropy = stats.entropy(stacked, axis=0)
                return entropy
        
        # Load individual ensemble members
        models = []
        for model_file in sorted(ensemble_dir.glob('model_*.h5')):
            logger.info(f"  Loading ensemble member: {model_file.name}")
            model = keras.models.load_model(model_file, compile=False)
            models.append(model)
        
        return EnsembleModel(models) if models else None
    
    def evaluate_all_models(self, models: Dict, X_test: np.ndarray, 
                           y_test: np.ndarray) -> ModelEvaluationFramework:
        """
        Evaluate all loaded models.
        
        Args:
            models: Dictionary of loaded models
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation framework with results
        """
        # Initialize evaluation framework
        evaluator = ModelEvaluationFramework(
            label_names=self.label_names,
            length_constraints=self.length_constraints
        )
        
        # Evaluate each model
        for model_name, (model, decoder, model_type) in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # Wrap decoder for consistent interface
            class DecoderWrapper:
                def __init__(self, decoder):
                    self.decoder = decoder
                
                def decode(self, X):
                    return self.decoder.decode(X)
                
                def predict(self, X):
                    return self.decoder.decode(X)
            
            wrapped_model = DecoderWrapper(decoder)
            
            # Evaluate
            evaluator.evaluate_model(
                model_name=model_name,
                model=wrapped_model,
                X_test=X_test,
                y_test=y_test,
                model_type=model_type
            )
        
        return evaluator
    
    def generate_detailed_comparison(self, evaluator: ModelEvaluationFramework, 
                                    output_dir: str = './evaluation_results'):
        """
        Generate detailed comparison reports and visualizations.
        
        Args:
            evaluator: Evaluation framework with results
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate comparison table
        comparison_df = evaluator.compare_models()
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        logger.info(f"Comparison table saved to {output_dir / 'model_comparison.csv'}")
        
        # 2. Generate detailed report
        report = evaluator.generate_report(str(output_dir / 'evaluation_report.json'))
        
        # 3. Create comprehensive visualization
        self._create_comprehensive_plots(evaluator, output_dir)
        
        # 4. Generate markdown report
        self._generate_markdown_report(evaluator, output_dir)
        
        logger.info(f"All evaluation results saved to {output_dir}")
    
    def _create_comprehensive_plots(self, evaluator: ModelEvaluationFramework, 
                                   output_dir: Path):
        """Create comprehensive visualization plots."""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Overall metrics comparison
        ax1 = plt.subplot(2, 3, 1)
        df = evaluator.compare_models()
        metrics = ['Accuracy', 'F1']
        x = np.arange(len(df))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, df[metric], width, label=metric)
        
        ax1.set_xlabel('Model')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Constraint satisfaction
        ax2 = plt.subplot(2, 3, 2)
        if 'Constraint Sat.' in df.columns:
            ax2.bar(range(len(df)), df['Constraint Sat.'])
            ax2.set_xticks(range(len(df)))
            ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
            ax2.set_ylabel('Satisfaction Rate')
            ax2.set_title('Constraint Satisfaction')
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Robustness comparison
        ax3 = plt.subplot(2, 3, 3)
        if 'Robustness' in df.columns:
            ax3.bar(range(len(df)), df['Robustness'])
            ax3.set_xticks(range(len(df)))
            ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
            ax3.set_ylabel('Robustness Score')
            ax3.set_title('Model Robustness')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Inference time
        ax4 = plt.subplot(2, 3, 4)
        ax4.bar(range(len(df)), df['Inference (ms)'])
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Inference Speed')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Radar chart for multi-metric comparison
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        # Normalize metrics to 0-1 scale for radar chart
        metrics_for_radar = ['Accuracy', 'F1']
        if 'Constraint Sat.' in df.columns:
            metrics_for_radar.append('Constraint Sat.')
        if 'Robustness' in df.columns:
            metrics_for_radar.append('Robustness')
        
        angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for idx, row in df.iterrows():
            values = [row[m] for m in metrics_for_radar]
            values += [values[0]]
            ax5.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics_for_radar)
        ax5.set_ylim(0, 1.1)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax5.set_title('Multi-Metric Comparison')
        ax5.grid(True)
        
        # 6. Performance vs Efficiency trade-off
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(df['Inference (ms)'], df['F1'], s=100)
        for idx, row in df.iterrows():
            ax6.annotate(row['Model'], (row['Inference (ms)'], row['F1']),
                        xytext=(5, 5), textcoords='offset points')
        ax6.set_xlabel('Inference Time (ms)')
        ax6.set_ylabel('F1 Score')
        ax6.set_title('Performance vs Efficiency Trade-off')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
        logger.info(f"Comprehensive plot saved to {output_dir / 'comprehensive_evaluation.png'}")
        plt.close()
    
    def _generate_markdown_report(self, evaluator: ModelEvaluationFramework, 
                                 output_dir: Path):
        """Generate a markdown report summarizing results."""
        df = evaluator.compare_models()
        report = evaluator.generate_report()
        
        markdown_content = f"""# Tempest Model Evaluation Report

## Executive Summary

Evaluated {len(df)} models on sequence annotation task with length constraints.

### Best Models by Metric:
"""
        
        # Add recommendations
        for key, value in report['recommendations'].items():
            markdown_content += f"- **{key.replace('_', ' ').title()}**: {value['model']} ({value['reason']})\n"
        
        markdown_content += f"""

## Model Comparison Table

{df.to_markdown(index=False)}

## Detailed Analysis

### 1. Performance Metrics

The evaluation considered multiple performance dimensions:
- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Balanced measure of precision and recall
- **Constraint Satisfaction**: Adherence to specified length constraints
- **Robustness**: Performance degradation under various error conditions
- **Inference Speed**: Computational efficiency

### 2. Model Types Evaluated

1. **Standard Model**: Baseline CNN-BiLSTM-CRF without constraints
2. **Soft Constraints**: Length regularization during training
3. **Hard Constraints**: Enforcement during inference
4. **Hybrid**: Combination of soft training and hard inference
5. **Ensemble**: Multiple models with voting/averaging

### 3. Key Findings

"""
        
        # Add key findings based on results
        best_f1_model = df.loc[df['F1'].idxmax()]
        markdown_content += f"- Best overall performance: **{best_f1_model['Model']}** (F1: {best_f1_model['F1']:.4f})\n"
        
        if 'Constraint Sat.' in df.columns:
            best_constraint = df.loc[df['Constraint Sat.'].idxmax()]
            markdown_content += f"- Best constraint satisfaction: **{best_constraint['Model']}** ({best_constraint['Constraint Sat.']:.4f})\n"
        
        fastest = df.loc[df['Inference (ms)'].idxmin()]
        markdown_content += f"- Fastest inference: **{fastest['Model']}** ({fastest['Inference (ms)']:.2f} ms)\n"
        
        markdown_content += """

### 4. Recommendations

Based on the evaluation results:

"""
        
        # Add use-case specific recommendations
        if 'Constraint Sat.' in df.columns:
            constraint_models = df[df['Constraint Sat.'] > 0.95]
            if not constraint_models.empty:
                best_constrained = constraint_models.loc[constraint_models['F1'].idxmax()]
                markdown_content += f"""
**For applications requiring strict length constraints:**
- Use **{best_constrained['Model']}** model
- Achieves {best_constrained['Constraint Sat.']:.1%} constraint satisfaction
- Maintains {best_constrained['F1']:.4f} F1 score
"""
        
        # Speed-critical applications
        speed_threshold = df['Inference (ms)'].quantile(0.25)
        fast_models = df[df['Inference (ms)'] <= speed_threshold]
        if not fast_models.empty:
            best_fast = fast_models.loc[fast_models['F1'].idxmax()]
            markdown_content += f"""
**For real-time/high-throughput applications:**
- Use **{best_fast['Model']}** model
- Inference time: {best_fast['Inference (ms)']:.2f} ms
- F1 score: {best_fast['F1']:.4f}
"""
        
        markdown_content += """

### 5. Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
"""
        
        # Add selection guide
        use_cases = {
            "High accuracy required": df.loc[df['F1'].idxmax()]['Model'],
            "Strict constraints": df.loc[df.get('Constraint Sat.', pd.Series()).idxmax()]['Model'] if 'Constraint Sat.' in df.columns else 'N/A',
            "Real-time processing": df.loc[df['Inference (ms)'].idxmin()]['Model'],
            "Robust to errors": df.loc[df.get('Robustness', pd.Series()).idxmax()]['Model'] if 'Robustness' in df.columns else 'N/A'
        }
        
        for use_case, model in use_cases.items():
            if model != 'N/A':
                model_data = df[df['Model'] == model].iloc[0]
                reason = f"F1: {model_data['F1']:.3f}"
                if use_case == "Real-time processing":
                    reason = f"Time: {model_data['Inference (ms)']:.1f}ms"
                elif use_case == "Strict constraints" and 'Constraint Sat.' in model_data:
                    reason = f"Satisfaction: {model_data['Constraint Sat.']:.1%}"
                markdown_content += f"| {use_case} | {model} | {reason} |\n"
        
        markdown_content += """

## Conclusion

The evaluation demonstrates clear trade-offs between different modeling approaches:
- Standard models offer baseline performance
- Constraint-aware models improve adherence to specifications
- Hybrid approaches balance multiple objectives
- Ensemble methods may provide improved robustness

Select the appropriate model based on your specific requirements for accuracy, 
constraint satisfaction, robustness, and computational efficiency.

---
*Generated by Tempest Model Evaluation Framework*
"""
        
        # Save markdown report
        report_path = output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to {report_path}")


def compare_models(models_dir: str, test_data_path: str, 
                  config_path: Optional[str] = None,
                  output_dir: str = './evaluation_results') -> ModelEvaluationFramework:
    """
    Main function to compare Tempest models.
    
    Args:
        models_dir: Directory containing trained models
        test_data_path: Path to test data (pickled X_test, y_test)
        config_path: Path to Tempest config file (optional)
        output_dir: Directory for evaluation outputs
        
    Returns:
        ModelEvaluationFramework with results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = TempestModelEvaluator(config_path=config_path)
    
    # Load models
    logger.info(f"Loading models from {models_dir}...")
    models = evaluator.load_models(models_dir)
    
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Evaluate models
    logger.info("Starting model evaluation...")
    framework = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # Generate comprehensive comparison
    logger.info("Generating comparison reports...")
    evaluator.generate_detailed_comparison(framework, output_dir)
    
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")
    
    return framework
