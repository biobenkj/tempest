#!/usr/bin/env python3
"""
Evaluate model robustness on segment-level errors.

Tests trained models on:
1. Segment loss (missing UMI, ACC, barcode)
2. Segment duplication (repeated segments)
3. Truncation (incomplete reads)
4. Chimeric reads (mixed architectures)

Usage:
    python evaluate_segment_robustness.py --model model.h5 --config config.yaml
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentRobustnessEvaluator:
    """Comprehensive evaluation of model robustness to segment errors."""
    
    def __init__(self, model_path: str, config: Dict):
        """
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.model = keras.models.load_model(model_path)
        self.config = config
        
        # Expected architecture
        self.expected_segments = ['ADAPTER5', 'UMI', 'ACC', 'BARCODE', 'INSERT', 'ADAPTER3']
        self.segment_to_idx = {seg: i for i, seg in enumerate(self.expected_segments)}
        
        # Error types to test
        self.error_types = [
            'valid',
            'missing_umi',
            'missing_acc',
            'missing_barcode',
            'double_umi',
            'double_barcode',
            'truncated_50',
            'truncated_75',
            'chimeric'
        ]
        
    def evaluate_all_errors(self, test_sequences: np.ndarray, 
                           test_labels: np.ndarray,
                           n_samples: int = 100) -> pd.DataFrame:
        """
        Evaluate model on all error types.
        
        Returns:
            DataFrame with results for each error type
        """
        results = []
        
        for error_type in self.error_types:
            logger.info(f"Evaluating {error_type}...")
            
            # Generate test cases
            error_sequences, error_labels = self._generate_error_cases(
                test_sequences[:n_samples], 
                test_labels[:n_samples],
                error_type
            )
            
            # Evaluate
            metrics = self._evaluate_batch(error_sequences, error_labels)
            metrics['error_type'] = error_type
            metrics['n_samples'] = n_samples
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _generate_error_cases(self, sequences: np.ndarray, 
                             labels: np.ndarray,
                             error_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate specific error patterns."""
        
        if error_type == 'valid':
            return sequences, labels
        
        error_seqs = []
        error_labs = []
        
        for seq, lab in zip(sequences, labels):
            if error_type == 'missing_umi':
                err_seq, err_lab = self._remove_segment(seq, lab, 'UMI')
            elif error_type == 'missing_acc':
                err_seq, err_lab = self._remove_segment(seq, lab, 'ACC')
            elif error_type == 'missing_barcode':
                err_seq, err_lab = self._remove_segment(seq, lab, 'BARCODE')
            elif error_type == 'double_umi':
                err_seq, err_lab = self._duplicate_segment(seq, lab, 'UMI')
            elif error_type == 'double_barcode':
                err_seq, err_lab = self._duplicate_segment(seq, lab, 'BARCODE')
            elif error_type == 'truncated_50':
                err_seq, err_lab = self._truncate(seq, lab, 0.5)
            elif error_type == 'truncated_75':
                err_seq, err_lab = self._truncate(seq, lab, 0.75)
            elif error_type == 'chimeric':
                err_seq, err_lab = self._create_chimera(seq, lab)
            else:
                err_seq, err_lab = seq, lab
            
            error_seqs.append(err_seq)
            error_labs.append(err_lab)
        
        # Pad to consistent length
        return self._pad_batch(error_seqs, error_labs)
    
    def _remove_segment(self, sequence: np.ndarray, labels: np.ndarray, 
                       segment_name: str) -> Tuple:
        """Remove a specific segment."""
        segments = self._identify_segments(labels)
        
        for seg in segments:
            if seg['name'] == segment_name:
                # Remove segment
                new_seq = np.concatenate([
                    sequence[:seg['start']], 
                    sequence[seg['end']:]
                ])
                new_lab = np.concatenate([
                    labels[:seg['start']], 
                    labels[seg['end']:]
                ])
                return new_seq, new_lab
        
        return sequence, labels
    
    def _duplicate_segment(self, sequence: np.ndarray, labels: np.ndarray,
                          segment_name: str) -> Tuple:
        """Duplicate a specific segment."""
        segments = self._identify_segments(labels)
        
        for seg in segments:
            if seg['name'] == segment_name:
                # Extract segment
                seg_seq = sequence[seg['start']:seg['end']]
                seg_lab = labels[seg['start']:seg['end']]
                
                # Insert duplicate after original
                new_seq = np.concatenate([
                    sequence[:seg['end']],
                    seg_seq,
                    sequence[seg['end']:]
                ])
                new_lab = np.concatenate([
                    labels[:seg['end']],
                    seg_lab,
                    labels[seg['end']:]
                ])
                return new_seq, new_lab
        
        return sequence, labels
    
    def _truncate(self, sequence: np.ndarray, labels: np.ndarray, 
                 fraction: float) -> Tuple:
        """Truncate sequence to fraction of original length."""
        cut_point = int(len(sequence) * fraction)
        return sequence[:cut_point], labels[:cut_point]
    
    def _create_chimera(self, sequence: np.ndarray, labels: np.ndarray) -> Tuple:
        """Create chimeric read by scrambling middle segments."""
        segments = self._identify_segments(labels)
        
        if len(segments) < 4:
            return sequence, labels
        
        # Keep adapters, scramble middle
        first = segments[0]
        last = segments[-1]
        middle = segments[1:-1]
        np.random.shuffle(middle)
        
        # Reconstruct
        new_seq = []
        new_lab = []
        
        for seg in [first] + middle + [last]:
            new_seq.append(sequence[seg['start']:seg['end']])
            new_lab.append(labels[seg['start']:seg['end']])
        
        return np.concatenate(new_seq), np.concatenate(new_lab)
    
    def _identify_segments(self, labels: np.ndarray) -> List[Dict]:
        """Identify segment boundaries and names."""
        segments = []
        current = labels[0]
        start = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current:
                # Map label index to segment name
                seg_name = self.expected_segments[current] if current < len(self.expected_segments) else 'UNKNOWN'
                segments.append({
                    'name': seg_name,
                    'start': start,
                    'end': i,
                    'label': current
                })
                current = labels[i]
                start = i
        
        # Add final segment
        seg_name = self.expected_segments[current] if current < len(self.expected_segments) else 'UNKNOWN'
        segments.append({
            'name': seg_name,
            'start': start,
            'end': len(labels),
            'label': current
        })
        
        return segments
    
    def _pad_batch(self, sequences: List, labels: List) -> Tuple:
        """Pad sequences to consistent length."""
        max_len = max(len(s) for s in sequences)
        
        padded_seqs = np.zeros((len(sequences), max_len), dtype=np.int32)
        padded_labs = np.zeros((len(labels), max_len), dtype=np.int32)
        
        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            padded_seqs[i, :len(seq)] = seq
            padded_labs[i, :len(lab)] = lab
        
        return padded_seqs, padded_labs
    
    def _evaluate_batch(self, sequences: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate model performance on a batch."""
        # Get predictions
        predictions = self.model.predict(sequences, verbose=0)
        pred_labels = np.argmax(predictions, axis=-1)
        
        # Compute metrics
        metrics = {}
        
        # Overall accuracy
        mask = labels > 0  # Ignore padding
        metrics['accuracy'] = np.mean(pred_labels[mask] == labels[mask])
        
        # Per-segment accuracy
        for seg_name in self.expected_segments:
            seg_idx = self.segment_to_idx[seg_name]
            seg_mask = labels == seg_idx
            if np.any(seg_mask):
                metrics[f'acc_{seg_name.lower()}'] = np.mean(
                    pred_labels[seg_mask] == labels[seg_mask]
                )
        
        # Confidence metrics
        confidence = np.max(predictions, axis=-1)
        metrics['mean_confidence'] = np.mean(confidence[mask])
        metrics['min_confidence'] = np.min(confidence[mask])
        
        # Segment boundary accuracy
        boundaries = self._find_boundaries(labels)
        pred_boundaries = self._find_boundaries(pred_labels)
        if boundaries:
            metrics['boundary_precision'] = self._boundary_precision(
                pred_boundaries, boundaries
            )
        
        return metrics
    
    def _find_boundaries(self, labels: np.ndarray) -> List[int]:
        """Find segment boundary positions."""
        boundaries = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1] and labels[i] > 0:
                boundaries.append(i)
        return boundaries
    
    def _boundary_precision(self, predicted: List[int], true: List[int], 
                           tolerance: int = 3) -> float:
        """Calculate boundary detection precision with tolerance."""
        if not predicted:
            return 0.0
        
        correct = 0
        for pred_pos in predicted:
            for true_pos in true:
                if abs(pred_pos - true_pos) <= tolerance:
                    correct += 1
                    break
        
        return correct / len(predicted)
    
    def plot_results(self, results_df: pd.DataFrame, output_path: str):
        """Create visualization of robustness results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Overall accuracy by error type
        ax = axes[0, 0]
        results_df.plot(x='error_type', y='accuracy', kind='bar', ax=ax)
        ax.set_title('Overall Accuracy by Error Type')
        ax.set_ylabel('Accuracy')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target')
        
        # 2. Segment-specific accuracy heatmap
        ax = axes[0, 1]
        seg_cols = [c for c in results_df.columns if c.startswith('acc_')]
        seg_data = results_df.set_index('error_type')[seg_cols]
        seg_data.columns = [c.replace('acc_', '') for c in seg_data.columns]
        
        sns.heatmap(seg_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax)
        ax.set_title('Segment-Specific Accuracy')
        
        # 3. Confidence distribution
        ax = axes[0, 2]
        results_df.plot(x='error_type', y=['mean_confidence', 'min_confidence'], 
                       kind='bar', ax=ax)
        ax.set_title('Prediction Confidence')
        ax.set_ylabel('Confidence')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Accuracy degradation from baseline
        ax = axes[1, 0]
        baseline_acc = results_df[results_df['error_type'] == 'valid']['accuracy'].values[0]
        results_df['degradation'] = baseline_acc - results_df['accuracy']
        
        results_df.plot(x='error_type', y='degradation', kind='bar', 
                       color='coral', ax=ax)
        ax.set_title('Accuracy Degradation from Baseline')
        ax.set_ylabel('Degradation')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Boundary detection precision
        ax = axes[1, 1]
        if 'boundary_precision' in results_df.columns:
            results_df.plot(x='error_type', y='boundary_precision', 
                          kind='bar', color='steelblue', ax=ax)
            ax.set_title('Segment Boundary Detection')
            ax.set_ylabel('Precision')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Summary Statistics
        ==================
        
        Best Performance: {results_df.loc[results_df['accuracy'].idxmax(), 'error_type']}
        Worst Performance: {results_df.loc[results_df['accuracy'].idxmin(), 'error_type']}
        
        Average Accuracy: {results_df['accuracy'].mean():.3f}
        Std Dev: {results_df['accuracy'].std():.3f}
        
        Most Robust Segment: {self._most_robust_segment(results_df)}
        Least Robust Segment: {self._least_robust_segment(results_df)}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, 
               verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
    
    def _most_robust_segment(self, df: pd.DataFrame) -> str:
        """Find segment with least accuracy degradation."""
        seg_cols = [c for c in df.columns if c.startswith('acc_')]
        if not seg_cols:
            return "N/A"
        
        seg_variance = df[seg_cols].var()
        return seg_variance.idxmin().replace('acc_', '')
    
    def _least_robust_segment(self, df: pd.DataFrame) -> str:
        """Find segment with most accuracy degradation."""
        seg_cols = [c for c in df.columns if c.startswith('acc_')]
        if not seg_cols:
            return "N/A"
        
        seg_variance = df[seg_cols].var()
        return seg_variance.idxmax().replace('acc_', '')
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str):
        """Generate detailed robustness report."""
        report = []
        report.append("="*80)
        report.append("SEGMENT ROBUSTNESS EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-"*40)
        for _, row in results_df.iterrows():
            report.append(f"{row['error_type']:20s}: {row['accuracy']:.3f}")
        report.append("")
        
        # Segment-specific performance
        report.append("SEGMENT-SPECIFIC ACCURACY")
        report.append("-"*40)
        seg_cols = [c for c in results_df.columns if c.startswith('acc_')]
        
        for col in seg_cols:
            seg_name = col.replace('acc_', '').upper()
            report.append(f"\n{seg_name}:")
            for _, row in results_df.iterrows():
                if col in row and not pd.isna(row[col]):
                    report.append(f"  {row['error_type']:20s}: {row[col]:.3f}")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-"*40)
        
        # Generate recommendations based on results
        worst_error = results_df.loc[results_df['accuracy'].idxmin(), 'error_type']
        worst_acc = results_df.loc[results_df['accuracy'].idxmin(), 'accuracy']
        
        if worst_acc < 0.7:
            report.append(f"⚠️  Critical: Model performs poorly on {worst_error} (acc={worst_acc:.3f})")
            report.append("   Recommendation: Increase training examples with this error pattern")
        
        if 'missing_acc' in results_df['error_type'].values:
            acc_missing = results_df[results_df['error_type'] == 'missing_acc']['accuracy'].values[0]
            if acc_missing < 0.8:
                report.append(f"⚠️  ACC detection needs improvement (acc={acc_missing:.3f})")
                report.append("   Recommendation: Increase PWM weight or add more ACC training examples")
        
        truncation_errors = results_df[results_df['error_type'].str.contains('truncated')]
        if not truncation_errors.empty:
            mean_trunc_acc = truncation_errors['accuracy'].mean()
            if mean_trunc_acc < 0.75:
                report.append(f"⚠️  Poor truncation handling (avg acc={mean_trunc_acc:.3f})")
                report.append("   Recommendation: Add more truncated examples during training")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved report to {output_path}")
        
        # Print summary to console
        print('\n'.join(report))


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output', default='robustness_eval/', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=100, help='Samples per error type')
    
    args = parser.parse_args()
    
    # Load configuration
    from tempest.utils import load_config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    from tempest.data import SequenceSimulator, reads_to_arrays
    simulator = SequenceSimulator(config.simulation)
    test_reads = simulator.generate(num_sequences=args.n_samples * 2)
    X_test, y_test, _ = reads_to_arrays(test_reads)
    
    # Initialize evaluator
    evaluator = SegmentRobustnessEvaluator(args.model, config)
    
    # Run evaluation
    logger.info("Starting robustness evaluation...")
    results = evaluator.evaluate_all_errors(X_test, y_test, n_samples=args.n_samples)
    
    # Save results
    results_path = output_dir / 'robustness_results.csv'
    results.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Generate visualizations
    plot_path = output_dir / 'robustness_plots.png'
    evaluator.plot_results(results, str(plot_path))
    
    # Generate report
    report_path = output_dir / 'robustness_report.txt'
    evaluator.generate_report(results, str(report_path))


if __name__ == '__main__':
    main()
