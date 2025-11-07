"""
Data module for Tempest.

Contains data simulation, preprocessing, and generators.
"""

from .simulator import (
    SequenceSimulator,
    SimulatedRead,
    TranscriptPool,
    PolyATailGenerator,
    WhitelistManager,
    ErrorSimulator,
    reads_to_arrays,
    create_simulator_from_config,
    demonstrate_probabilistic_generation
)

from .invalid_generator import (
    InvalidReadGenerator
)

__all__ = [
    'SequenceSimulator',
    'SimulatedRead', 
    'TranscriptPool',
    'PolyATailGenerator',
    'WhitelistManager',
    'ErrorSimulator',
    'reads_to_arrays',
    'create_simulator_from_config',
    'demonstrate_probabilistic_generation',
    'InvalidReadGenerator'
]
