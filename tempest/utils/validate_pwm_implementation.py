#!/usr/bin/env python3
"""
Comprehensive PWM Implementation Validation Script for Tempest

This script validates that the PWM matrix is effectively implemented during:
1. Read simulation
2. ACC sequence generation
3. Model training
4. Validation read generation

Author: Computational Biology Assistant
"""

import sys
import os
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add tempest to path (adjust if needed)
sys.path.insert(0, os.path.abspath('.'))

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.ENDC}")

def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message."""
    print(f"  {message}")


class PWMValidator:
    """Comprehensive PWM implementation validator."""
    
    def __init__(self, config_path: str = 'train_config.yaml', 
                 pwm_path: str = 'acc_pwm.txt'):
        """
        Initialize validator.
        
        Args:
            config_path: Path to configuration file
            pwm_path: Path to PWM file
        """
        self.config_path = config_path
        self.pwm_path = pwm_path
        self.validation_results = {}
        
    def validate_pwm_loading(self) -> bool:
        """Validate that PWM can be loaded correctly."""
        print_header("TEST 1: PWM Loading and Structure")
        
        try:
            from tempest.utils import load_pwm
            
            # Load PWM
            pwm = load_pwm(self.pwm_path)
            print_success(f"PWM loaded successfully from {self.pwm_path}")
            
            # Check structure
            expected_shape = (6, 4)  # 6 positions, 4 bases
            if pwm.shape != expected_shape:
                print_error(f"PWM shape {pwm.shape} != expected {expected_shape}")
                return False
            print_success(f"PWM shape correct: {pwm.shape}")
            
            # Check probabilities sum to 1
            for pos in range(pwm.shape[0]):
                prob_sum = pwm[pos].sum()
                if not np.isclose(prob_sum, 1.0, rtol=1e-5):
                    print_error(f"Position {pos+1} probabilities sum to {prob_sum:.5f}, not 1.0")
                    return False
            print_success("All position probabilities sum to 1.0")
            
            # Check for expected ACC pattern
            # Position 1: A dominant (>90%)
            if pwm[0, 0] < 0.9:
                print_warning(f"Position 1: A probability {pwm[0, 0]:.3f} < 0.9")
            else:
                print_success(f"Position 1: A dominant ({pwm[0, 0]:.3f})")
            
            # Positions 2-3: C dominant (>80%)
            for pos in [1, 2]:
                if pwm[pos, 1] < 0.8:
                    print_warning(f"Position {pos+1}: C probability {pwm[pos, 1]:.3f} < 0.8")
                else:
                    print_success(f"Position {pos+1}: C dominant ({pwm[pos, 1]:.3f})")
            
            self.validation_results['pwm_loading'] = True
            return True
            
        except Exception as e:
            print_error(f"Failed to load PWM: {e}")
            self.validation_results['pwm_loading'] = False
            return False
    
    def validate_pwm_scoring(self) -> bool:
        """Validate PWM scoring functionality."""
        print_header("TEST 2: PWM Scoring")
        
        try:
            from tempest.utils import load_pwm
            from tempest.core import PWMScorer
            
            # Load PWM and create scorer
            pwm = load_pwm(self.pwm_path)
            scorer = PWMScorer(pwm, threshold=0.7)
            print_success("PWM scorer initialized")
            
            # Test sequences (IUPAC pattern: ACCSSV)
            test_cases = [
                # (sequence, expected_high_score, description)
                ('ACCGGG', True, 'Perfect ACCGGG match'),
                ('ACCGGC', True, 'Good ACCGGC match'),
                ('ACCGCG', True, 'Good ACCGCG match'),
                ('ACCCGG', True, 'Alternative ACCCGG'),
                ('TTTTTT', False, 'Bad all-T sequence'),
                ('AAAAAA', False, 'Bad all-A sequence'),
                ('GGGGGG', False, 'Bad all-G sequence'),
            ]
            
            all_passed = True
            for seq, should_pass, description in test_cases:
                score = scorer.score_sequence(seq)
                passed = (score >= 0.7) == should_pass
                
                if passed:
                    print_success(f"{seq}: {score:.3f} - {description}")
                else:
                    print_error(f"{seq}: {score:.3f} - {description} (FAILED)")
                    all_passed = False
            
            self.validation_results['pwm_scoring'] = all_passed
            return all_passed
            
        except Exception as e:
            print_error(f"PWM scoring validation failed: {e}")
            self.validation_results['pwm_scoring'] = False
            return False
    
    def validate_acc_generation(self) -> bool:
        """Validate ACC sequence generation using PWM."""
        print_header("TEST 3: ACC Generation from PWM")
        
        try:
            from tempest.utils import load_pwm
            from tempest.core import PWMScorer, generate_acc_from_pwm
            
            # Load PWM
            pwm = load_pwm(self.pwm_path)
            scorer = PWMScorer(pwm, threshold=0.7)
            
            # Generate ACC sequences
            n_sequences = 100
            acc_sequences = generate_acc_from_pwm(pwm, n=n_sequences)
            print_success(f"Generated {len(acc_sequences)} ACC sequences")
            
            # Score them
            scores = []
            for seq in acc_sequences:
                score = scorer.score_sequence(seq)
                scores.append(score)
            
            scores = np.array(scores)
            
            # Statistics
            mean_score = scores.mean()
            min_score = scores.min()
            max_score = scores.max()
            above_threshold = (scores >= 0.7).mean() * 100
            
            print_info(f"Score statistics:")
            print_info(f"  Mean: {mean_score:.3f}")
            print_info(f"  Min:  {min_score:.3f}")
            print_info(f"  Max:  {max_score:.3f}")
            print_info(f"  Above 0.7: {above_threshold:.1f}%")
            
            # Check quality
            if mean_score < 0.8:
                print_warning(f"Mean score {mean_score:.3f} < 0.8")
            else:
                print_success(f"High mean score: {mean_score:.3f}")
            
            if above_threshold < 90:
                print_warning(f"Only {above_threshold:.1f}% above threshold")
            else:
                print_success(f"{above_threshold:.1f}% sequences above threshold")
            
            # Show sample sequences
            print_info("\nSample generated ACC sequences:")
            for i in range(min(5, len(acc_sequences))):
                print_info(f"  {acc_sequences[i]}: {scores[i]:.3f}")
            
            self.validation_results['acc_generation'] = mean_score >= 0.75
            return mean_score >= 0.75
            
        except Exception as e:
            print_error(f"ACC generation validation failed: {e}")
            self.validation_results['acc_generation'] = False
            return False
    
    def validate_simulation_integration(self) -> bool:
        """Validate PWM integration in sequence simulation."""
        print_header("TEST 4: PWM Integration in Read Simulation")
        
        try:
            from tempest.utils import load_config, load_pwm
            from tempest.data import SequenceSimulator
            from tempest.core import PWMScorer
            
            # Load config and PWM
            config = load_config(self.config_path)
            pwm = load_pwm(self.pwm_path)
            scorer = PWMScorer(pwm, threshold=0.7)
            
            # Create simulator WITH PWM
            simulator_with_pwm = SequenceSimulator(
                config.simulation, 
                pwm_file=self.pwm_path
            )
            print_success("Simulator initialized with PWM")
            
            # Generate reads
            n_reads = 50
            reads = simulator_with_pwm.generate(num_sequences=n_reads)
            print_success(f"Generated {len(reads)} reads")
            
            # Extract and score ACC sequences
            acc_sequences = []
            for read in reads:
                if 'ACC' in read.label_regions:
                    for start, end in read.label_regions['ACC']:
                        acc_seq = read.sequence[start:end]
                        acc_sequences.append(acc_seq)
            
            if not acc_sequences:
                print_error("No ACC sequences found in generated reads!")
                self.validation_results['simulation_integration'] = False
                return False
            
            print_success(f"Found {len(acc_sequences)} ACC sequences in reads")
            
            # Score ACC sequences
            scores = []
            for seq in acc_sequences:
                score = scorer.score_sequence(seq)
                scores.append(score)
            
            scores = np.array(scores)
            mean_score = scores.mean()
            above_threshold = (scores >= 0.7).mean() * 100
            
            print_info(f"\nACC sequences from simulated reads:")
            print_info(f"  Mean score: {mean_score:.3f}")
            print_info(f"  Above threshold: {above_threshold:.1f}%")
            
            # Show samples
            print_info("\nSample ACC sequences from reads:")
            for i in range(min(5, len(acc_sequences))):
                print_info(f"  {acc_sequences[i]}: {scores[i]:.3f}")
            
            # Compare with random generation (without PWM)
            print_info("\nComparing with random generation (no PWM):")
            simulator_no_pwm = SequenceSimulator(config.simulation, pwm_file=None)
            reads_random = simulator_no_pwm.generate(num_sequences=20)
            
            acc_random = []
            for read in reads_random:
                if 'ACC' in read.label_regions:
                    for start, end in read.label_regions['ACC']:
                        acc_random.append(read.sequence[start:end])
            
            if acc_random:
                scores_random = np.array([scorer.score_sequence(seq) for seq in acc_random[:5]])
                print_info(f"  Random ACC mean score: {scores_random.mean():.3f}")
                print_info(f"  PWM ACC mean score: {mean_score:.3f}")
                
                if mean_score > scores_random.mean() + 0.1:
                    print_success("PWM-generated ACC scores significantly higher than random")
                else:
                    print_warning("PWM advantage not significant")
            
            self.validation_results['simulation_integration'] = mean_score >= 0.7
            return mean_score >= 0.7
            
        except Exception as e:
            print_error(f"Simulation integration validation failed: {e}")
            self.validation_results['simulation_integration'] = False
            return False
    
    def validate_training_integration(self) -> bool:
        """Validate PWM usage during model training."""
        print_header("TEST 5: PWM Integration in Training Pipeline")
        
        try:
            from tempest.utils import load_config
            
            # Check if training uses PWM
            config = load_config(self.config_path)
            
            # Check configuration
            checks = []
            
            # Check if PWM file is specified
            if hasattr(config, 'pwm') and config.pwm.get('use_pwm'):
                print_success("PWM enabled in configuration")
                checks.append(True)
            else:
                print_warning("PWM not explicitly enabled in config")
                checks.append(False)
            
            # Check if ACC priors are configured
            if hasattr(config.simulation, 'acc_priors_file'):
                print_success(f"ACC priors configured: {config.simulation.acc_priors_file}")
                checks.append(True)
            else:
                print_info("ACC priors not configured (will use PWM if available)")
            
            # Check main.py integration
            main_py = Path('main.py')
            if main_py.exists():
                with open(main_py, 'r') as f:
                    main_content = f.read()
                    if 'pwm_file' in main_content:
                        print_success("main.py includes PWM file handling")
                        checks.append(True)
                    else:
                        print_warning("main.py may not handle PWM files")
                        checks.append(False)
            
            # Simulate mini training to verify
            print_info("\nSimulating mini training run...")
            try:
                from tempest.data import SequenceSimulator
                simulator = SequenceSimulator(config.simulation, pwm_file=self.pwm_path)
                train_reads, val_reads = simulator.generate_train_val_split(train_fraction=0.8)
                
                print_success(f"Training data generated: {len(train_reads)} train, {len(val_reads)} val")
                
                # Check if validation reads also use PWM
                val_acc_sequences = []
                for read in val_reads[:10]:
                    if 'ACC' in read.label_regions:
                        for start, end in read.label_regions['ACC']:
                            val_acc_sequences.append(read.sequence[start:end])
                
                if val_acc_sequences:
                    from tempest.utils import load_pwm
                    from tempest.core import PWMScorer
                    
                    pwm = load_pwm(self.pwm_path)
                    scorer = PWMScorer(pwm)
                    val_scores = [scorer.score_sequence(seq) for seq in val_acc_sequences]
                    val_mean_score = np.mean(val_scores)
                    
                    print_success(f"Validation ACC mean score: {val_mean_score:.3f}")
                    checks.append(val_mean_score >= 0.7)
                
            except Exception as e:
                print_warning(f"Could not simulate training: {e}")
            
            success = all(checks) if checks else False
            self.validation_results['training_integration'] = success
            return success
            
        except Exception as e:
            print_error(f"Training integration validation failed: {e}")
            self.validation_results['training_integration'] = False
            return False
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        print_header("VALIDATION REPORT")
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for v in self.validation_results.values() if v)
        
        print_info(f"Total tests: {total_tests}")
        print_info(f"Passed: {passed_tests}")
        print_info(f"Failed: {total_tests - passed_tests}")
        
        print_info("\nTest Results:")
        for test_name, passed in self.validation_results.items():
            status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
            print(f"  {test_name}: {status}")
        
        # Overall assessment
        print_header("OVERALL ASSESSMENT")
        
        if passed_tests == total_tests:
            print_success("PWM is FULLY IMPLEMENTED and working correctly!")
            print_info("The PWM matrix is effectively integrated throughout:")
            print_info("  • Read simulation generates realistic ACC sequences")
            print_info("  • ACC sequences score highly against the PWM")
            print_info("  • Training pipeline uses PWM-based generation")
            print_info("  • Validation reads maintain PWM characteristics")
        elif passed_tests >= total_tests * 0.7:
            print_warning("PWM is MOSTLY IMPLEMENTED with some issues")
            print_info("Recommendations:")
            if not self.validation_results.get('pwm_loading'):
                print_info("  • Fix PWM loading issues")
            if not self.validation_results.get('simulation_integration'):
                print_info("  • Ensure simulator uses PWM file")
            if not self.validation_results.get('training_integration'):
                print_info("  • Verify training pipeline uses PWM")
        else:
            print_error("PWM implementation has SIGNIFICANT ISSUES")
            print_info("Critical fixes needed:")
            print_info("  • Review PWM file format and loading")
            print_info("  • Check integration points in simulator")
            print_info("  • Verify configuration settings")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'test_results': self.validation_results,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
    
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        print(f"\n{Colors.BOLD}PWM Implementation Validation Suite{Colors.ENDC}")
        print(f"Configuration: {self.config_path}")
        print(f"PWM File: {self.pwm_path}")
        
        # Run all tests
        tests = [
            self.validate_pwm_loading,
            self.validate_pwm_scoring,
            self.validate_acc_generation,
            self.validate_simulation_integration,
            self.validate_training_integration
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print_error(f"Test failed with exception: {e}")
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_file = 'pwm_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print_info(f"\nReport saved to: {report_file}")
        
        return report['success_rate'] >= 0.8


def main():
    """Main validation runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate PWM implementation in Tempest')
    parser.add_argument('--config', default='train_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--pwm', default='acc_pwm.txt',
                       help='Path to PWM file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation only')
    
    args = parser.parse_args()
    
    # Create validator
    validator = PWMValidator(config_path=args.config, pwm_path=args.pwm)
    
    if args.quick:
        # Quick validation - just check loading and scoring
        print("Running quick validation...")
        validator.validate_pwm_loading()
        validator.validate_pwm_scoring()
    else:
        # Full validation
        success = validator.run_all_validations()
        
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ PWM IMPLEMENTATION VALIDATED{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ PWM IMPLEMENTATION NEEDS ATTENTION{Colors.ENDC}")
            sys.exit(1)


if __name__ == '__main__':
    main()
