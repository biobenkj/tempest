#!/usr/bin/env python3
"""
Comprehensive test runner for Tempest with GPU support.

This script provides various test execution modes including:
- Full test suite
- GPU-specific tests
- Performance benchmarks
- Quick unit tests
- Coverage reports
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Manages test execution with various configurations."""
    
    def __init__(self, base_dir=None):
        """Initialize test runner."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.test_dir = self.base_dir
        self.results_dir = self.test_dir / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def check_gpu_availability(self):
        """Check if GPU is available for testing."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s) available for testing")
                for gpu in gpus:
                    logger.info(f"  - {gpu.name}")
                return True
            else:
                logger.warning("No GPUs found. GPU tests will be skipped.")
                return False
        except Exception as e:
            logger.error(f"Error checking GPU availability: {e}")
            return False
    
    def run_command(self, cmd, capture_output=False):
        """Run a command and return the result."""
        logger.info(f"Running: {' '.join(cmd)}")
        
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result
        else:
            result = subprocess.run(cmd)
            return result.returncode == 0
    
    def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("="*60)
        logger.info("Running COMPLETE TEST SUITE")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit.xml",
            f"--html={self.results_dir}/report.html",
            "--self-contained-html"
        ]
        
        return self.run_command(cmd)
    
    def run_unit_tests(self):
        """Run only unit tests (fast)."""
        logger.info("="*60)
        logger.info("Running UNIT TESTS")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-m", "unit",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_unit.xml"
        ]
        
        return self.run_command(cmd)
    
    def run_integration_tests(self):
        """Run integration tests."""
        logger.info("="*60)
        logger.info("Running INTEGRATION TESTS")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-m", "integration",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_integration.xml"
        ]
        
        return self.run_command(cmd)
    
    def run_gpu_tests(self):
        """Run GPU-specific tests."""
        if not self.check_gpu_availability():
            logger.warning("Skipping GPU tests - no GPU available")
            return False
        
        logger.info("="*60)
        logger.info("Running GPU TESTS")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-m", "gpu",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_gpu.xml"
        ]
        
        return self.run_command(cmd)
    
    def run_benchmark_tests(self):
        """Run performance benchmark tests."""
        logger.info("="*60)
        logger.info("Running BENCHMARK TESTS")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-m", "benchmark",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_benchmark.xml",
            "--benchmark-only",
            "--benchmark-json={self.results_dir}/benchmark.json"
        ]
        
        return self.run_command(cmd)
    
    def run_subcommand_tests(self, subcommand):
        """Run tests for a specific subcommand."""
        logger.info("="*60)
        logger.info(f"Running tests for subcommand: {subcommand}")
        logger.info("="*60)
        
        test_file = self.test_dir / "commands" / f"test_{subcommand}.py"
        
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return False
        
        cmd = [
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_{subcommand}.xml"
        ]
        
        return self.run_command(cmd)
    
    def run_with_coverage(self):
        """Run tests with coverage report."""
        logger.info("="*60)
        logger.info("Running TESTS WITH COVERAGE")
        logger.info("="*60)
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "--cov=tempest",
            "--cov-report=html:" + str(self.results_dir / "coverage_html"),
            "--cov-report=term-missing",
            "--cov-report=json:" + str(self.results_dir / "coverage.json"),
            f"--junitxml={self.results_dir}/junit_coverage.xml"
        ]
        
        success = self.run_command(cmd)
        
        if success:
            logger.info(f"Coverage report saved to: {self.results_dir / 'coverage_html/index.html'}")
        
        return success
    
    def run_parallel_tests(self, num_workers=None):
        """Run tests in parallel."""
        logger.info("="*60)
        logger.info("Running PARALLEL TESTS")
        logger.info("="*60)
        
        workers = num_workers or "auto"
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-n", str(workers),
            "--tb=short",
            f"--junitxml={self.results_dir}/junit_parallel.xml"
        ]
        
        return self.run_command(cmd)
    
    def run_quick_check(self):
        """Run a quick smoke test."""
        logger.info("="*60)
        logger.info("Running QUICK CHECK (smoke tests)")
        logger.info("="*60)
        
        # Run only fast unit tests
        cmd = [
            "pytest",
            str(self.test_dir),
            "-v",
            "-m", "unit and not slow",
            "--tb=short",
            "--maxfail=3",
            "-x"  # Stop on first failure
        ]
        
        return self.run_command(cmd)
    
    def generate_report(self):
        """Generate a summary report of all test results."""
        logger.info("Generating test summary report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_available": self.check_gpu_availability(),
            "test_results": {}
        }
        
        # Check for various result files
        result_files = {
            "unit": "junit_unit.xml",
            "integration": "junit_integration.xml",
            "gpu": "junit_gpu.xml",
            "benchmark": "benchmark.json",
            "coverage": "coverage.json"
        }
        
        for test_type, filename in result_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                report["test_results"][test_type] = "completed"
            else:
                report["test_results"][test_type] = "not_run"
        
        # Save report
        report_file = self.results_dir / "test_summary.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test summary saved to: {report_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        for test_type, status in report["test_results"].items():
            logger.info(f"  {test_type:15} : {status}")
        logger.info("="*60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Tempest Test Runner with GPU Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run only GPU tests
  python run_tests.py --gpu
  
  # Run tests for specific subcommand
  python run_tests.py --subcommand train
  
  # Run with coverage
  python run_tests.py --coverage
  
  # Run quick smoke tests
  python run_tests.py --quick
  
  # Run tests in parallel
  python run_tests.py --parallel -n 4
        """
    )
    
    # Test selection arguments
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    parser.add_argument('--unit', action='store_true',
                        help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                        help='Run integration tests only')
    parser.add_argument('--gpu', action='store_true',
                        help='Run GPU tests only')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark tests only')
    parser.add_argument('--subcommand', type=str,
                        choices=['simulate', 'train', 'evaluate', 'visualize', 
                                'compare', 'combine', 'demux'],
                        help='Run tests for specific subcommand')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick smoke tests')
    
    # Test configuration arguments
    parser.add_argument('--coverage', action='store_true',
                        help='Run with coverage report')
    parser.add_argument('--parallel', action='store_true',
                        help='Run tests in parallel')
    parser.add_argument('-n', '--num-workers', type=int,
                        help='Number of parallel workers')
    parser.add_argument('--report', action='store_true',
                        help='Generate test report only')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    # Handle report generation
    if args.report:
        runner.generate_report()
        return 0
    
    # Determine what to run
    success = True
    tests_run = False
    
    if args.all:
        success = runner.run_all_tests()
        tests_run = True
    
    if args.unit:
        success = runner.run_unit_tests() and success
        tests_run = True
    
    if args.integration:
        success = runner.run_integration_tests() and success
        tests_run = True
    
    if args.gpu:
        success = runner.run_gpu_tests() and success
        tests_run = True
    
    if args.benchmark:
        success = runner.run_benchmark_tests() and success
        tests_run = True
    
    if args.subcommand:
        success = runner.run_subcommand_tests(args.subcommand) and success
        tests_run = True
    
    if args.coverage:
        success = runner.run_with_coverage() and success
        tests_run = True
    
    if args.parallel:
        success = runner.run_parallel_tests(args.num_workers) and success
        tests_run = True
    
    if args.quick:
        success = runner.run_quick_check() and success
        tests_run = True
    
    # If no specific tests selected, run quick check
    if not tests_run:
        logger.info("No specific tests selected. Running quick check...")
        success = runner.run_quick_check()
    
    # Generate report
    runner.generate_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
