"""
Tempest: Modular sequence annotation using length-constrained CRFs

A deep learning framework for annotating biological sequences with structured
elements using CNN-BiLSTM-CRF architecture with semi-Markov approximation.
"""

__version__ = "0.2.0"
__author__ = "Ben Johnson"
__email__ = "ben.johnson@vai.org"

from . import core
from . import data  
from . import training
from . import inference
from . import visualization
from . import utils
from . import demux

# Core model building functions
from .core import (
    build_cnn_bilstm_crf,
    build_model_with_length_constraints,
    build_model_from_config,
    LengthConstrainedCRF,
    ModelWithLengthConstrainedCRF
)

__all__ = [
    'core',
    'data',
    'training', 
    'inference',
    'visualization',
    'utils',
    'demux',
    'build_cnn_bilstm_crf',
    'build_model_with_length_constraints',
    'build_model_from_config',
    'LengthConstrainedCRF',
    'ModelWithLengthConstrainedCRF'
]
