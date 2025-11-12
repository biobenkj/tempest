"""
Tempest inference module for model predictions and analysis.
"""

# Import core inference utilities
from .inference_utils import (
    predict_sequences,
    encode_sequences,
    decode_sequences,
    batch_process_fasta
)

from .combiner import (
    ModelCombiner
)

from .demux_validator import (
    ArchitectureValidator,
    DemuxResult,
    ModelBasedDemultiplexer,
    demux_with_validation
)

from .sample_demultiplexer import (
    Sample,
    BarcodeMatch,
    SampleSheet,
    SampleBasedDemultiplexer,
    demux_with_samples
)


# Module exports
__all__ = [
    # Core inference functions
    'predict_sequences',
    'encode_sequences',
    'decode_sequences',
    'batch_process_fasta',
    'ModelCombiner',
    'ArchitectureValidator',
    'DemuxResult',
    'ModelBasedDemultiplexer',
    'demux_with_validation',
    'Sample',
    'BarcodeMatch',
    'SampleSheet',
    'SampleBasedDemultiplexer',
    'demux_with_samples',
]
