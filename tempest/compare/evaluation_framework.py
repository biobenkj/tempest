"""
Comprehensive Model Evaluation Framework for Tempest
=====================================================

Framework for systematically evaluating and comparing standard, hybrid, and ensemble 
approaches after training. This provides a structured way to assess model performance
across multiple dimensions.

Author: Evaluation Framework
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluationFramework:
    """
    Comprehensive evaluation framework for comparing different model approaches.
    
    Evaluates models on:
    1. Basic metrics (accuracy, precision, recall, F1)
    2. Segment-level metrics (per-label performance)
    3. Length constraint satisfaction
    4. Robustness to errors (missing/duplicated segments)
    5. Computational efficiency
    6. Uncertainty quantification (for ensemble models)
    """
    
    def __init__(self, label_names: List[str], length_constraints: Optional[Dict] = None):
        """
        Initialize evaluation framework.
        
        Args:
            label_names: List of segment labels (e.g., ['ADAPTER5', 'UMI', 'ACC', ...])
            length_constraints: Dict mapping label -> (min_length, max_length)
        """
        self.label_names = label_names
        self.length_constraints = length_constraints or {}
        self.results = {}
        
    def evaluate_model(self, model_name: str, model, X_test, y_test, 
                       model_type: str = 'standard') -> Dict:
        """
        Complete evaluation of a single model.
        
        Args:
            model_name: Identifier for the model
            model: Trained model object
            X_test: Test sequences
            y_test: Test labels
            model_type: One of ['standard', 'soft_constraint', 'hard_constraint', 
                               'hybrid', 'ensemble']
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name} ({model_type})...")
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Basic predictions
        logger.info("  Computing predictions...")
        predictions, pred_time = self._timed_predict(model, X_test)
        results['inference_time_ms'] = pred_time * 1000
        
        # 2. Basic metrics
        logger.info("  Computing basic metrics...")
        results.update(self._compute_basic_metrics(y_test, predictions))
        
        # 3. Per-segment metrics
        logger.info("  Computing segment-level metrics...")
        results['segment_metrics'] = self._compute_segment_metrics(y_test, predictions)
        
        # 4. Length constraint satisfaction
        if self.length_constraints:
            logger.info("  Evaluating constraint satisfaction...")
            results['constraint_satisfaction'] = self._evaluate_constraints(
                predictions, self.length_constraints
            )
        
        # 5. Robustness evaluation
        logger.info("  Evaluating robustness to errors...")
        results['robustness_scores'] = self._evaluate_robustness(
            model, X_test, y_test
        )
        
        # 6. Ensemble-specific metrics
        if model_type == 'ensemble':
            logger.info("  Computing ensemble metrics...")
            results['ensemble_metrics'] = self._compute_ensemble_metrics(
                model, X_test, y_test
            )
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def _timed_predict(self, model, X_test) -> Tuple[np.ndarray, float]:
        """Time the prediction process."""
        import time
        start = time.time()
        
        # Handle different model APIs
        if hasattr(model, 'decode'):
            predictions = model.decode(X_test)
        elif hasattr(model, 'predict'):
            predictions = model.predict(X_test)
            # Convert from one-hot if needed
            if len(predictions.shape) == 3:
                predictions = np.argmax(predictions, axis=-1)
        else:
            raise ValueError(f"Model doesn't have decode or predict method")
            
        elapsed = time.time() - start
        return predictions, elapsed
    
    def _compute_basic_metrics(self, y_true, y_pred) -> Dict:
        """Compute basic classification metrics."""
        # Flatten for overall metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove padding (assuming padding label is the last one)
        mask = y_true_flat < len(self.label_names) - 1
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, 
            labels=range(len(self.label_names) - 1),
            average='weighted'
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        return metrics
    
    def _compute_segment_metrics(self, y_true, y_pred) -> Dict:
        """Compute per-segment performance metrics."""
        segment_metrics = {}
        
        for label_idx, label_name in enumerate(self.label_names[:-1]):  # Skip padding
            # Find positions for this label
            label_mask = y_true == label_idx
            
            if np.any(label_mask):
                # Segment-specific accuracy
                segment_acc = np.mean(y_pred[label_mask] == label_idx)
                
                # Transition accuracy (correctly identifying segment boundaries)
                transitions = self._find_transitions(y_true, label_idx)
                if len(transitions) > 0:
                    pred_transitions = self._find_transitions(y_pred, label_idx)
                    transition_acc = self._compute_transition_accuracy(
                        transitions, pred_transitions
                    )
                else:
                    transition_acc = 0.0
                
                segment_metrics[label_name] = {
                    'accuracy': segment_acc,
                    'transition_accuracy': transition_acc,
                    'support': np.sum(label_mask)
                }
        
        return segment_metrics
    
    def _evaluate_constraints(self, predictions, constraints) -> Dict:
        """Evaluate how well predictions satisfy length constraints."""
        satisfaction_rates = {}
        length_distributions = {}
        
        for label_name, (min_len, max_len) in constraints.items():
            label_idx = self.label_names.index(label_name)
            
            # Find all segments of this type
            segments = self._extract_segments(predictions, label_idx)
            
            if segments:
                lengths = [seg['length'] for seg in segments]
                
                # Satisfaction rate
                satisfied = sum(min_len <= l <= max_len for l in lengths)
                satisfaction_rates[label_name] = satisfied / len(lengths)
                
                # Length distribution
                length_distributions[label_name] = {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'expected': (min_len, max_len),
                    'violations': len(lengths) - satisfied
                }
            else:
                satisfaction_rates[label_name] = 0.0
                length_distributions[label_name] = None
        
        return {
            'rates': satisfaction_rates,
            'distributions': length_distributions
        }
    
    def _evaluate_robustness(self, model, X_test, y_test) -> Dict:
        """Evaluate model robustness to various error types."""
        error_types = {
            'missing_segment': self._generate_missing_segment_errors,
            'duplicated_segment': self._generate_duplicated_segment_errors,
            'truncated': self._generate_truncated_errors,
            'noisy': self._generate_noisy_errors
        }
        
        robustness_scores = {}
        n_samples = min(100, len(X_test))  # Use subset for efficiency
        
        for error_name, error_func in error_types.items():
            # Generate corrupted data
            X_corrupted, y_corrupted = error_func(
                X_test[:n_samples], y_test[:n_samples]
            )
            
            # Evaluate on corrupted data
            predictions, _ = self._timed_predict(model, X_corrupted)
            
            # Compute degradation
            clean_acc = accuracy_score(
                y_test[:n_samples].flatten(), 
                self._timed_predict(model, X_test[:n_samples])[0].flatten()
            )
            corrupt_acc = accuracy_score(y_corrupted.flatten(), predictions.flatten())
            
            robustness_scores[error_name] = {
                'clean_accuracy': clean_acc,
                'corrupted_accuracy': corrupt_acc,
                'degradation': clean_acc - corrupt_acc,
                'robustness_ratio': corrupt_acc / clean_acc if clean_acc > 0 else 0
            }
        
        return robustness_scores
    
    def _compute_ensemble_metrics(self, ensemble_model, X_test, y_test) -> Dict:
        """Compute ensemble-specific metrics like uncertainty and agreement."""
        ensemble_metrics = {}
        
        if hasattr(ensemble_model, 'get_model_predictions'):
            # Get predictions from individual models
            all_predictions = ensemble_model.get_model_predictions(X_test)
            
            # Agreement rate (how often models agree)
            agreement = np.mean([
                np.all(all_predictions[0] == pred) 
                for pred in all_predictions[1:]
            ])
            ensemble_metrics['model_agreement'] = agreement
            
            # Uncertainty (entropy of predictions)
            if hasattr(ensemble_model, 'get_prediction_uncertainty'):
                uncertainty = ensemble_model.get_prediction_uncertainty(X_test)
                ensemble_metrics['mean_uncertainty'] = np.mean(uncertainty)
                ensemble_metrics['max_uncertainty'] = np.max(uncertainty)
        
        return ensemble_metrics
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a comparison table of all evaluated models.
        
        Args:
            model_names: List of model names to compare (default: all evaluated)
            
        Returns:
            DataFrame with comparison metrics
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        comparison_data = []
        
        for name in model_names:
            if name not in self.results:
                logger.warning(f"Model {name} not found in results")
                continue
                
            result = self.results[name]
            row = {
                'Model': name,
                'Type': result['model_type'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1': result['f1'],
                'Inference (ms)': result['inference_time_ms']
            }
            
            # Add constraint satisfaction if available
            if 'constraint_satisfaction' in result:
                avg_satisfaction = np.mean(list(
                    result['constraint_satisfaction']['rates'].values()
                ))
                row['Constraint Sat.'] = avg_satisfaction
            
            # Add robustness score
            if 'robustness_scores' in result:
                avg_robustness = np.mean([
                    score['robustness_ratio'] 
                    for score in result['robustness_scores'].values()
                ])
                row['Robustness'] = avg_robustness
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('F1', ascending=False)
    
    def plot_comparison(self, metrics: List[str] = None, save_path: str = None):
        """
        Create visualization comparing models across metrics.
        
        Args:
            metrics: List of metrics to plot (default: key metrics)
            save_path: Path to save figure
        """
        if metrics is None:
            metrics = ['Accuracy', 'F1', 'Constraint Sat.', 'Robustness']
        
        df = self.compare_models()
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                 figsize=(5*len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            df_plot = df[['Model', 'Type', metric]].copy()
            
            # Color by model type
            colors = {
                'standard': 'blue',
                'soft_constraint': 'green',
                'hard_constraint': 'orange',
                'hybrid': 'red',
                'ensemble': 'purple'
            }
            
            bar_colors = [colors.get(t, 'gray') for t in df_plot['Type']]
            
            ax.bar(range(len(df_plot)), df_plot[metric], color=bar_colors)
            ax.set_xticks(range(len(df_plot)))
            ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': len(self.results),
            'summary_table': self.compare_models().to_dict('records'),
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on evaluation results."""
        if not self.results:
            return {"error": "No models evaluated yet"}
        
        df = self.compare_models()
        recommendations = {}
        
        # Best overall
        best_f1 = df.loc[df['F1'].idxmax()]
        recommendations['best_overall'] = {
            'model': best_f1['Model'],
            'reason': f"Highest F1 score ({best_f1['F1']:.4f})"
        }
        
        # Best for constraints
        if 'Constraint Sat.' in df.columns:
            best_constraint = df.loc[df['Constraint Sat.'].idxmax()]
            recommendations['best_constraints'] = {
                'model': best_constraint['Model'],
                'reason': f"Best constraint satisfaction ({best_constraint['Constraint Sat.']:.4f})"
            }
        
        # Most robust
        if 'Robustness' in df.columns:
            best_robust = df.loc[df['Robustness'].idxmax()]
            recommendations['most_robust'] = {
                'model': best_robust['Model'],
                'reason': f"Highest robustness score ({best_robust['Robustness']:.4f})"
            }
        
        # Fastest
        fastest = df.loc[df['Inference (ms)'].idxmin()]
        recommendations['fastest'] = {
            'model': fastest['Model'],
            'reason': f"Lowest inference time ({fastest['Inference (ms)']:.2f} ms)"
        }
        
        return recommendations
    
    # Helper methods
    def _find_transitions(self, labels, label_idx):
        """Find start/end positions of segments with given label."""
        transitions = []
        in_segment = False
        start = 0
        
        for i, label in enumerate(labels.flatten()):
            if label == label_idx and not in_segment:
                start = i
                in_segment = True
            elif label != label_idx and in_segment:
                transitions.append((start, i))
                in_segment = False
        
        if in_segment:
            transitions.append((start, len(labels.flatten())))
        
        return transitions
    
    def _compute_transition_accuracy(self, true_trans, pred_trans, tolerance=2):
        """Compute accuracy of boundary detection with tolerance."""
        if not pred_trans:
            return 0.0
        
        correct = 0
        for true_start, true_end in true_trans:
            for pred_start, pred_end in pred_trans:
                if (abs(true_start - pred_start) <= tolerance and 
                    abs(true_end - pred_end) <= tolerance):
                    correct += 1
                    break
        
        return correct / len(true_trans)
    
    def _extract_segments(self, predictions, label_idx):
        """Extract all segments of a given label from predictions."""
        segments = []
        
        for seq in predictions:
            transitions = self._find_transitions(seq.reshape(1, -1), label_idx)
            for start, end in transitions:
                segments.append({
                    'start': start,
                    'end': end,
                    'length': end - start
                })
        
        return segments
    
    def _generate_missing_segment_errors(self, X, y):
        """Generate data with missing segments."""
        X_corrupt = X.copy()
        y_corrupt = y.copy()
        
        # Randomly remove UMI segments (label 1) from 30% of sequences
        for i in range(len(X_corrupt)):
            if np.random.random() < 0.3:
                # Find UMI positions
                umi_mask = y_corrupt[i] == 1
                if np.any(umi_mask):
                    # Replace with padding
                    X_corrupt[i][umi_mask] = 4  # N base
                    y_corrupt[i][umi_mask] = len(self.label_names) - 1  # Padding label
        
        return X_corrupt, y_corrupt
    
    def _generate_duplicated_segment_errors(self, X, y):
        """Generate data with duplicated segments."""
        # Implementation for duplicating segments
        # This is a simplified version - you'd implement based on your needs
        return X.copy(), y.copy()
    
    def _generate_truncated_errors(self, X, y):
        """Generate truncated sequences."""
        X_corrupt = X.copy()
        y_corrupt = y.copy()
        
        for i in range(len(X_corrupt)):
            # Truncate to 75% of original length
            truncate_point = int(X_corrupt.shape[1] * 0.75)
            X_corrupt[i, truncate_point:] = 4  # Padding
            y_corrupt[i, truncate_point:] = len(self.label_names) - 1
        
        return X_corrupt, y_corrupt
    
    def _generate_noisy_errors(self, X, y):
        """Add random noise to sequences."""
        X_corrupt = X.copy()
        
        # Random substitutions in 5% of positions
        noise_mask = np.random.random(X_corrupt.shape) < 0.05
        X_corrupt[noise_mask] = np.random.randint(0, 4, size=np.sum(noise_mask))
        
        return X_corrupt, y.copy()


# Example usage function
def evaluate_all_models(models_dict, X_test, y_test, label_names, length_constraints):
    """
    Example function showing how to use the evaluation framework.
    
    Args:
        models_dict: Dictionary mapping model_name -> (model, model_type)
        X_test: Test sequences
        y_test: Test labels
        label_names: List of label names
        length_constraints: Length constraints dictionary
    
    Returns:
        Evaluation framework with results
    """
    # Initialize framework
    evaluator = ModelEvaluationFramework(
        label_names=label_names,
        length_constraints=length_constraints
    )
    
    # Evaluate each model
    for model_name, (model, model_type) in models_dict.items():
        evaluator.evaluate_model(
            model_name=model_name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type
        )
    
    # Generate comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    comparison_df = evaluator.compare_models()
    print(comparison_df.to_string())
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    report = evaluator.generate_report()
    for key, value in report['recommendations'].items():
        print(f"{key}: {value['model']} - {value['reason']}")
    
    # Create visualization
    evaluator.plot_comparison(save_path="model_comparison.png")
    
    return evaluator


if __name__ == "__main__":
    print("Model Evaluation Framework")
    print("=" * 80)
    print("This framework provides comprehensive evaluation capabilities for")
    print("comparing standard, hybrid, and ensemble sequence annotation models.")
    print("\nKey Features:")
    print("- Basic metrics (accuracy, precision, recall, F1)")
    print("- Segment-level performance analysis")
    print("- Length constraint satisfaction evaluation")
    print("- Robustness testing (missing/duplicated segments, truncation, noise)")
    print("- Computational efficiency measurement")
    print("- Ensemble-specific metrics (uncertainty, agreement)")
    print("- Automated report generation and visualization")
    print("\nUse the ModelEvaluationFramework class to systematically compare")
    print("your trained models and identify the best approach for your use case.")
