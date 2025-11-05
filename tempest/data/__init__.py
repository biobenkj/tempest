"""
Data module for Tempest.

Contains data simulation, preprocessing, and generators.
"""

from .simulator import (
    SequenceSimulator,
    SimulatedRead,
    TranscriptPool,
    PolyATailGenerator,
    reads_to_arrays
)

__all__ = [
    'SequenceSimulator',
    'SimulatedRead', 
    'TranscriptPool',
    'PolyATailGenerator',
    'reads_to_arrays'
]
