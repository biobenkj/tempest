#!/usr/bin/env python3
"""
PWM Validation Read Generation Checker for Tempest

This script specifically checks that PWM is effectively used when generating
validation reads after model training, ensuring consistency between training
and validation data.

Author: Computational Biology Assistant
"""

import sys
import os
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add tempest to path
sys.path.insert(0, os.path.abspath('.'))

class ValidationReadChecker:
    """Check PWM implementation in validation read generation."""
    
    def __init__(self, config_path: str = 'train_config.yaml', 
                 pwm_path: str = 'acc_pwm.txt'):
        self.config_path = config_path
        self.pwm_path = pwm_path
        self.results = {}
    
    def check_training_validation_consistency(self):
        """Verify that training and validation reads use the same PWM."""
        print("\n" + "="*80)
        print("CHECKING TRAINING/VALIDATION DATA CONSISTENCY")
        print("="*80)
        
        try:
            from tempest.utils import load_config, load_pwm
            from tempest.data import SequenceSimulator
            from tempest.core import PWMScorer
            
            # Load components
            config = load_config(self.config_path)
            pwm = load_pwm(self.pwm_path)
            scorer = PWMScorer(pwm, threshold=0.7)
            
            # Create simulator with PWM
            simulator = SequenceSimulator(config.simulation, pwm_file=self.pwm_path)
            
            # Generate train/val split
            print("\nGenerating training and validation data...")
            train_reads, val_reads = simulator.generate_train_val_split(
                train_fraction=0.8
            )
            
            print(f"✓ Generated {len(train_reads)} training reads")
            print(f"✓ Generated {len(val_reads)} validation reads")
            
            # Extract ACC sequences from both sets
            train_acc = self._extract_acc_sequences(train_reads)
            val_acc = self._extract_acc_sequences(val_reads)
            
            print(f"\nExtracted ACC sequences:")
            print(f"  Training: {len(train_acc)} ACC sequences")
            print(f"  Validation: {len(val_acc)} ACC sequences")
            
            # Score ACC sequences
            train_scores = np.array([scorer.score_sequence(seq) for seq in train_acc])
            val_scores = np.array([scorer.score_sequence(seq) for seq in val_acc])
            
            # Compare statistics
            print("\nPWM Score Statistics:")
            print(f"  Training ACC:")
            print(f"    Mean: {train_scores.mean():.3f}")
            print(f"    Std:  {train_scores.std():.3f}")
            print(f"    Min:  {train_scores.min():.3f}")
            print(f"    Max:  {train_scores.max():.3f}")
            print(f"    >0.7: {(train_scores >= 0.7).mean() * 100:.1f}%")
            
            print(f"\n  Validation ACC:")
            print(f"    Mean: {val_scores.mean():.3f}")
            print(f"    Std:  {val_scores.std():.3f}")
            print(f"    Min:  {val_scores.min():.3f}")
            print(f"    Max:  {val_scores.max():.3f}")
            print(f"    >0.7: {(val_scores >= 0.7).mean() * 100:.1f}%")
            
            # Statistical test for similarity
            from scipy import stats
            statistic, p_value = stats.ks_2samp(train_scores, val_scores)
            
            print(f"\nKolmogorov-Smirnov test:")
            print(f"  Statistic: {statistic:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value > 0.05:
                print("✓ Training and validation ACC distributions are similar (p > 0.05)")
                self.results['consistency'] = True
            else:
                print("⚠ Training and validation ACC distributions differ (p < 0.05)")
                self.results['consistency'] = False
            
            # Check sequence diversity
            train_unique = len(set(train_acc))
            val_unique = len(set(val_acc))
            
            print(f"\nSequence Diversity:")
            print(f"  Training: {train_unique} unique ACC sequences")
            print(f"  Validation: {val_unique} unique ACC sequences")
            
            overlap = len(set(train_acc) & set(val_acc))
            print(f"  Overlap: {overlap} sequences in both sets")
            
            # Save results
            self.results.update({
                'train_mean_score': train_scores.mean(),
                'val_mean_score': val_scores.mean(),
                'train_above_threshold': (train_scores >= 0.7).mean(),
                'val_above_threshold': (val_scores >= 0.7).mean(),
                'ks_statistic': statistic,
                'ks_p_value': p_value,
                'train_unique': train_unique,
                'val_unique': val_unique,
                'overlap': overlap
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return False
    
    def check_model_validation_process(self):
        """Check PWM usage in the actual model validation process."""
        print("\n" + "="*80)
        print("CHECKING MODEL VALIDATION PROCESS")
        print("="*80)
        
        try:
            # Check if model training preserves PWM characteristics
            from tempest.utils import load_config
            from tempest.data import SequenceSimulator, reads_to_arrays
            
            config = load_config(self.config_path)
            
            # Generate small dataset for testing
            simulator = SequenceSimulator(config.simulation, pwm_file=self.pwm_path)
            test_reads = simulator.generate(num_sequences=100)
            
            # Convert to arrays (as done in training)
            X_test, y_test, label_to_idx = reads_to_arrays(test_reads)
            
            print(f"✓ Generated test data: {X_test.shape}")
            print(f"✓ Labels shape: {y_test.shape}")
            print(f"✓ Label mapping: {label_to_idx}")
            
            # Verify ACC label exists
            if 'ACC' not in label_to_idx:
                print("⚠ WARNING: ACC not in label mapping!")
                self.results['acc_in_labels'] = False
            else:
                print(f"✓ ACC label index: {label_to_idx['ACC']}")
                self.results['acc_in_labels'] = True
                
                # Check ACC label frequency
                acc_label_idx = label_to_idx['ACC']
                acc_positions = (y_test == acc_label_idx).sum()
                total_positions = y_test.size
                acc_percentage = (acc_positions / total_positions) * 100
                
                print(f"✓ ACC label frequency: {acc_percentage:.2f}% of positions")
                self.results['acc_frequency'] = acc_percentage
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation check failed: {e}")
            return False
    
    def visualize_pwm_scores(self):
        """Create visualization of PWM scores in train/val data."""
        print("\n" + "="*80)
        print("GENERATING PWM SCORE VISUALIZATIONS")
        print("="*80)
        
        try:
            from tempest.utils import load_config, load_pwm
            from tempest.data import SequenceSimulator
            from tempest.core import PWMScorer
            
            # Setup
            config = load_config(self.config_path)
            pwm = load_pwm(self.pwm_path)
            scorer = PWMScorer(pwm, threshold=0.7)
            
            # Generate data
            simulator = SequenceSimulator(config.simulation, pwm_file=self.pwm_path)
            train_reads, val_reads = simulator.generate_train_val_split()
            
            # Extract and score ACC sequences
            train_acc = self._extract_acc_sequences(train_reads)
            val_acc = self._extract_acc_sequences(val_reads)
            
            train_scores = [scorer.score_sequence(seq) for seq in train_acc]
            val_scores = [scorer.score_sequence(seq) for seq in val_acc]
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram of scores
            ax = axes[0, 0]
            ax.hist(train_scores, bins=30, alpha=0.5, label='Training', color='blue')
            ax.hist(val_scores, bins=30, alpha=0.5, label='Validation', color='red')
            ax.axvline(0.7, color='black', linestyle='--', label='Threshold')
            ax.set_xlabel('PWM Score')
            ax.set_ylabel('Count')
            ax.set_title('ACC Sequence PWM Score Distribution')
            ax.legend()
            
            # Box plot comparison
            ax = axes[0, 1]
            ax.boxplot([train_scores, val_scores], labels=['Training', 'Validation'])
            ax.axhline(0.7, color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('PWM Score')
            ax.set_title('Score Distribution Comparison')
            
            # Score over time (first 100 sequences)
            ax = axes[1, 0]
            ax.plot(train_scores[:100], alpha=0.7, label='Training')
            ax.plot(val_scores[:100], alpha=0.7, label='Validation')
            ax.axhline(0.7, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Sequence Index')
            ax.set_ylabel('PWM Score')
            ax.set_title('PWM Scores of First 100 ACC Sequences')
            ax.legend()
            
            # Cumulative distribution
            ax = axes[1, 1]
            train_sorted = np.sort(train_scores)
            val_sorted = np.sort(val_scores)
            train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
            val_cdf = np.arange(1, len(val_sorted) + 1) / len(val_sorted)
            ax.plot(train_sorted, train_cdf, label='Training')
            ax.plot(val_sorted, val_cdf, label='Validation')
            ax.axvline(0.7, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('PWM Score')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Cumulative Distribution Function')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('pwm_validation_scores.png', dpi=150, bbox_inches='tight')
            print("✓ Saved visualization to pwm_validation_scores.png")
            
            return True
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False
    
    def _extract_acc_sequences(self, reads: List) -> List[str]:
        """Extract ACC sequences from reads."""
        acc_sequences = []
        for read in reads:
            if 'ACC' in read.label_regions:
                for start, end in read.label_regions['ACC']:
                    acc_sequences.append(read.sequence[start:end])
        return acc_sequences
    
    def generate_summary_report(self):
        """Generate summary report of findings."""
        print("\n" + "="*80)
        print("PWM VALIDATION READ GENERATION SUMMARY")
        print("="*80)
        
        if self.results.get('consistency'):
            print("✓ Training and validation data are consistent")
        else:
            print("⚠ Training and validation data show differences")
        
        if self.results.get('acc_in_labels'):
            print(f"✓ ACC is properly labeled ({self.results.get('acc_frequency', 0):.1f}% of positions)")
        else:
            print("✗ ACC labeling issue detected")
        
        train_mean = self.results.get('train_mean_score', 0)
        val_mean = self.results.get('val_mean_score', 0)
        
        print(f"\nPWM Score Summary:")
        print(f"  Training mean:   {train_mean:.3f}")
        print(f"  Validation mean: {val_mean:.3f}")
        print(f"  Difference:      {abs(train_mean - val_mean):.3f}")
        
        if abs(train_mean - val_mean) < 0.05:
            print("  ✓ Excellent consistency")
        elif abs(train_mean - val_mean) < 0.1:
            print("  ✓ Good consistency")
        else:
            print("  ⚠ Significant difference detected")
        
        print(f"\nDiversity Metrics:")
        print(f"  Training unique:   {self.results.get('train_unique', 0)}")
        print(f"  Validation unique: {self.results.get('val_unique', 0)}")
        print(f"  Overlap:          {self.results.get('overlap', 0)}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT:")
        
        if (train_mean >= 0.7 and val_mean >= 0.7 and 
            abs(train_mean - val_mean) < 0.1 and
            self.results.get('consistency')):
            print("✓ PWM is EFFECTIVELY IMPLEMENTED in validation read generation")
            print("  - Validation reads maintain PWM characteristics")
            print("  - Consistent scoring between training and validation")
            print("  - Proper ACC sequence generation throughout pipeline")
        else:
            print("⚠ PWM implementation needs attention:")
            if train_mean < 0.7:
                print("  - Training ACC scores are low")
            if val_mean < 0.7:
                print("  - Validation ACC scores are low")
            if abs(train_mean - val_mean) >= 0.1:
                print("  - Inconsistent scores between train/val")
        
        return self.results


def main():
    """Run validation read generation checks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check PWM implementation in validation reads'
    )
    parser.add_argument('--config', default='train_config.yaml')
    parser.add_argument('--pwm', default='acc_pwm.txt')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    checker = ValidationReadChecker(
        config_path=args.config,
        pwm_path=args.pwm
    )
    
    # Run checks
    print("PWM Validation Read Generation Checker")
    print("=" * 80)
    
    success = True
    
    # Check consistency
    if not checker.check_training_validation_consistency():
        success = False
    
    # Check model validation
    if not checker.check_model_validation_process():
        success = False
    
    # Generate visualizations if requested
    if args.visualize:
        checker.visualize_pwm_scores()
    
    # Generate report
    results = checker.generate_summary_report()
    
    # Save results
    import json
    with open('pwm_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: pwm_validation_results.json")
    
    if success:
        print("\n✓ PWM validation check completed successfully")
        sys.exit(0)
    else:
        print("\n✗ PWM validation check found issues")
        sys.exit(1)


if __name__ == '__main__':
    main()
