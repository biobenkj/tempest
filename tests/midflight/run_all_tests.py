"""
Master test runner for Tempest comprehensive test suite.

This script runs all test modules and generates a summary report.
"""

import subprocess
import sys
from pathlib import Path


def run_test_module(test_file, description):
    """Run a single test module and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Module: {test_file}")
    print('='*80)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short', '--color=yes'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    """Run all tests and generate summary."""
    print("Tempest Comprehensive Test Suite")
    print("="*80)
    
    test_modules = [
        ('test_config_ingestion.py', 'Config.yaml Ingestion Tests'),
        ('test_length_constrained_crf.py', 'Length-Constrained CRF Tests'),
        ('test_ensemble_bma.py', 'BMA and Ensemble Modeling Tests'),
        ('test_hybrid_training.py', 'Hybrid Training Tests')
    ]
    
    results = {}
    
    for test_file, description in test_modules:
        test_path = Path(__file__).parent / test_file
        
        if not test_path.exists():
            print(f"\n⚠️  Warning: {test_file} not found, skipping...")
            results[description] = False
            continue
        
        success = run_test_module(str(test_path), description)
        results[description] = success
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    print("\n" + "-"*80)
    print(f"Total: {total_tests} test modules")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print("-"*80)
    
    if failed_tests > 0:
        print("\n⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)
    else:
        print("\n✅ All test modules passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
