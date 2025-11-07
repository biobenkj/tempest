"""
Tempest Model Comparison Module.

This module provides tools for evaluating and comparing different model approaches:
- Standard models
- Soft constraint models (training regularization)
- Hard constraint models (inference enforcement)  
- Hybrid models (soft + hard constraints)
- Ensemble models

Main components:
- ModelEvaluationFramework: Core evaluation class
- TempestModelEvaluator: Tempest-specific evaluator
- compare_models: Main comparison function
"""

from .evaluation_framework import ModelEvaluationFramework
from .evaluator import TempestModelEvaluator, compare_models

__all__ = [
    'ModelEvaluationFramework',
    'TempestModelEvaluator', 
    'compare_models'
]
