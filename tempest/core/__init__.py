"""
Tempest Core Modules

Contains model architectures and layers for sequence annotation with both
soft and hard length constraints. Additionally, PWM scoring support.
"""

from .models import (
    build_cnn_bilstm_crf,
    build_model_with_length_constraints,
    build_model_from_config,
    create_hybrid_model,
    print_model_summary,
    get_crf_layer,
    get_transition_matrix,
    validate_length_constraints
)

from .length_crf import (
    LengthConstrainedCRF,
    ModelWithLengthConstrainedCRF,
    unpack_data,
    create_length_constrained_model
)

from .constrained_viterbi import (
    ConstrainedViterbiDecoder,
    apply_constrained_decoding,
    evaluate_constrained_decoding
)

from .hybrid_decoder import (
    HybridConstraintDecoder
    # Note: create_hybrid_model is imported from models to avoid duplication
)

from .pwm import (
    PWMScorer,
    generate_acc_from_pwm,
    compute_pwm_from_sequences
)

from .pwm_probabilistic import (
    ProbabilisticPWMGenerator,
    create_acc_pwm_from_pattern
)

__all__ = [
    # Model building functions
    'build_cnn_bilstm_crf',
    'build_model_with_length_constraints', 
    'build_model_from_config',
    'create_hybrid_model',
    'print_model_summary',
    
    # Model utility functions
    'get_crf_layer',
    'get_transition_matrix',
    'validate_length_constraints',
    
    # Length CRF components
    'LengthConstrainedCRF',
    'ModelWithLengthConstrainedCRF',
    'unpack_data',
    'create_length_constrained_model',
    
    # Constrained Viterbi components
    'ConstrainedViterbiDecoder',
    'apply_constrained_decoding',
    'evaluate_constrained_decoding',
    
    # Hybrid decoder
    'HybridConstraintDecoder',
    
    # PWM components
    'PWMScorer',
    'generate_acc_from_pwm',
    'compute_pwm_from_sequences',
    'ProbabilisticPWMGenerator',
    'create_acc_pwm_from_pattern'
]