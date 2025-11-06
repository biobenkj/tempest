#!/usr/bin/env python3
"""
Tempest CLI wrapper - checks for help before importing ANYTHING.

This script is the entry point for the 'tempest' command and handles
--help requests without importing TensorFlow or the tempest package.
"""

import sys
import os


def show_help():
    """Display help message without importing any modules."""
    help_text = """usage: tempest [-h] --config CONFIG [--pwm PWM] [--output-dir OUTPUT_DIR] [--hybrid]
               [--unlabeled UNLABELED]

Tempest - Modular sequence annotation using length-constrained CRFs

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration YAML file (required)
  --pwm PWM             Path to PWM file for ACC generation (overrides config)
  --output-dir OUTPUT_DIR
                        Output directory for model checkpoints (overrides config)
  --hybrid              Enable hybrid robustness training mode
  --unlabeled UNLABELED
                        Path to unlabeled FASTQ file for pseudo-labeling (hybrid mode only)

TEMPEST OVERVIEW:
-----------------
Tempest is a deep learning framework for sequence annotation that combines:
  • Conditional Random Fields (CRFs) for structured prediction
  • Length constraints to enforce biologically meaningful segment sizes
  • Position Weight Matrix (PWM) priors for incorporating domain knowledge
  • Hybrid training modes for improved robustness

TRAINING MODES:
---------------
1. Standard Mode (default):
   - Basic supervised training with CRF layers
   - Uses simulated or provided sequence data
   - Suitable for clean, well-labeled data

2. Hybrid Mode (--hybrid):
   - Advanced training with invalid sequence handling
   - Pseudo-label generation for unlabeled data
   - Improved robustness to noisy real-world sequences
   - Requires hybrid configuration section in config file

CONFIGURATION:
--------------
Training is controlled via YAML configuration files:
  • config.yaml - Standard training configuration
  • hybrid_config.yaml - Hybrid training with robustness features
  • config_with_whitelists.yaml - Training with sequence constraints

Example config files are provided in the config/ directory.

EXAMPLES:
---------
Standard training:
  tempest --config config/train_config.yaml

Hybrid training with PWM:
  tempest --config config/hybrid_config.yaml --hybrid --pwm acc_pwm.txt

Training with unlabeled data:
  tempest --config config/hybrid_config.yaml --hybrid --unlabeled reads.fastq

Custom output directory:
  tempest --config config/train_config.yaml --output-dir ./my_model

For more information, visit: https://github.com/biobenkj/tempest
"""
    print(help_text)
    sys.exit(0)


def main():
    """Main entry point that checks for help before ANY imports."""
    # Check for help FIRST, before any imports from tempest package
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
    
    # Only now do we set up TensorFlow suppression and imports
    # ============================================================================
    # TENSORFLOW WARNING SUPPRESSION
    # ============================================================================
    if os.getenv('TEMPEST_DEBUG', '0') != '1':
        # Suppress TensorFlow C++ logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ.setdefault('TF_DISABLE_PLUGIN_REGISTRATION', '1')
        os.environ.setdefault('TF_ENABLE_DEPRECATION_WARNINGS', '0')
        os.environ.setdefault('TF_TRT_ALLOW_ENGINE_CACHING', '0')
        
        # Set up warning filters before imports
        import warnings
        warnings.filterwarnings("ignore")
        
        # Configure logging
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('tensorflow').propagate = False
        
        # Suppress absl logging
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
    
    # NOW we can import tempest - but we do it carefully
    # We import ONLY what we need, not the whole package
    # This avoids triggering the __init__.py imports
    import importlib.util
    
    # Load main.py directly without going through __init__.py
    spec = importlib.util.spec_from_file_location(
        "tempest_main", 
        os.path.join(os.path.dirname(__file__), "main.py")
    )
    tempest_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tempest_main)
    
    # Run the actual tempest main
    tempest_main.main()


if __name__ == '__main__':
    main()
