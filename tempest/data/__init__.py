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

try:
    from .simulator import (
        save_reads,
        load_reads,
        generate_and_save
    )
except ImportError:
    save_reads = None
    load_reads = None
    generate_and_save = None

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

if save_reads is not None:
    __all__.extend(['save_reads', 'load_reads', 'generate_and_save'])
