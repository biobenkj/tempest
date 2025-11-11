"""
Model architecture builder for Tempest.

Builds CNN-BiLSTM-CRF models for sequence annotation with length constraints.
Properly integrates the length-constrained CRF layer for semi-Markov approximation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, Tuple, Optional, List, Union

logger = logging.getLogger(__name__)


def build_cnn_bilstm_crf(
    vocab_size: int = 5,
    embedding_dim: int = 128,
    cnn_filters: Union[List[int], int] = [64, 128],
    cnn_kernels: Union[List[int], int] = [3, 5],
    lstm_units: int = 128,
    lstm_layers: int = 2,
    dropout: float = 0.3,
    num_labels: int = 10,
    max_seq_len: int = 512,
    use_cnn: bool = True,
    use_bilstm: bool = True,
    use_crf: bool = True,
    use_attention: bool = False,
    attention_units: int = 64,
    length_constraints: Optional[Dict[str, Tuple[int, int]]] = None,
    constraint_weight: float = 5.0
) -> keras.Model:
    """
    Build CNN-BiLSTM-CRF model for sequence annotation with optional length constraints.
    
    Architecture:
    1. Embedding: Convert base indices to dense vectors
    2. CNN (optional): Extract local motifs and k-mer features  
    3. BiLSTM (optional): Model long-range dependencies
    4. Attention (optional): Focus on important regions
    5. CRF (optional): Structured prediction with transition constraints
    6. Length constraints: Semi-Markov approximation for segment length control
    
    Args:
        vocab_size: Size of vocabulary (5 for A, C, G, T, N)
        embedding_dim: Dimension of embedding vectors
        cnn_filters: List of filter sizes for CNN layers or single int
        cnn_kernels: List of kernel sizes for CNN layers or single int
        lstm_units: Number of LSTM units
        lstm_layers: Number of BiLSTM layers
        dropout: Dropout rate
        num_labels: Number of output labels
        max_seq_len: Maximum sequence length
        use_cnn: Whether to include CNN layers
        use_bilstm: Whether to include BiLSTM layers
        use_crf: Whether to use CRF output layer
        use_attention: Whether to use attention mechanism
        attention_units: Number of attention units
        length_constraints: Dict mapping label_name -> (min_length, max_length)
                          Example: {'UMI': (8, 8), 'ACC': (6, 6), 'BARCODE': (16, 16)}
        constraint_weight: Penalty multiplier for length violations
        
    Returns:
        Keras Model (base model without constraint wrapper)
    """
    logger.info("Building CNN-BiLSTM-CRF model...")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  CNN: {use_cnn} (filters={cnn_filters}, kernels={cnn_kernels})")
    logger.info(f"  BiLSTM: {use_bilstm} (units={lstm_units}, layers={lstm_layers})")
    logger.info(f"  Attention: {use_attention} (units={attention_units})")
    logger.info(f"  CRF: {use_crf}")
    logger.info(f"  Length constraints: {length_constraints}")
    logger.info(f"  Number of labels: {num_labels}")
    
    # Handle single int inputs for CNN parameters
    if isinstance(cnn_filters, int):
        cnn_filters = [cnn_filters]
    if isinstance(cnn_kernels, int):
        cnn_kernels = [cnn_kernels]
    
    # Input layer
    inputs = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input')
    
    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='embedding'
    )(inputs)
    
    # CNN layers
    if use_cnn and cnn_filters and cnn_kernels:
        cnn_outputs = []
        for i, (filters, kernel) in enumerate(zip(cnn_filters, cnn_kernels)):
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel,
                padding='same',
                activation='relu',
                name=f'conv_{i+1}'
            )(x)
            cnn_outputs.append(conv)
        
        if len(cnn_outputs) > 1:
            x = layers.Concatenate(name='cnn_concat')(cnn_outputs)
        else:
            x = cnn_outputs[0]
        
        x = layers.Dropout(dropout, name='cnn_dropout')(x)
    
    # BiLSTM layers
    if use_bilstm:
        for i in range(lstm_layers):
            return_sequences = True  # Always true for sequence labeling
            
            if use_bilstm:
                x = layers.Bidirectional(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=0.0,  # Avoid issues with CuDNN
                        name=f'lstm_{i+1}'
                    ),
                    name=f'bilstm_{i+1}'
                )(x)
            else:
                x = layers.LSTM(
                    lstm_units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=0.0,
                    name=f'lstm_{i+1}'
                )(x)
    
    # Attention layer (optional)
    if use_attention:
        # Self-attention mechanism
        attention_scores = layers.Dense(1, activation='tanh', name='attention_scores')(x)
        attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_scores)
        x = layers.Multiply(name='attention_output')([x, attention_weights])
    
    # Dense layer for emissions
    x = layers.Dense(num_labels, name='emissions')(x)
    
    # Output layer with CRF
    if use_crf:
        # Import CRF layers here to avoid circular imports
        try:
            from tempest.core.length_crf import LengthConstrainedCRF
        except ImportError:
            # Fallback to relative import
            from .length_crf import LengthConstrainedCRF
        
        # Use Length-Constrained CRF if constraints are provided
        if length_constraints:
            crf_layer = LengthConstrainedCRF(
                units=num_labels,
                length_constraints=length_constraints,
                constraint_weight=constraint_weight,
                name='length_crf'
            )
        else:
            # Standard CRF without length constraints
            try:
                from tf2crf import CRF
                crf_layer = CRF(units=num_labels, name='crf')
            except ImportError:
                logger.warning("tf2crf not available, using Dense layer instead of CRF")
                outputs = layers.Activation('softmax', name='output')(x)
                model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_bilstm')
                logger.info(f"Model built with {model.count_params():,} parameters")
                return model
        
        outputs = crf_layer(x)
    else:
        # Simple softmax output without CRF
        outputs = layers.Activation('softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_bilstm_crf')
    
    logger.info(f"Model built with {model.count_params():,} parameters")
    
    return model


def build_model_with_length_constraints(
    base_model: keras.Model,
    length_constraints: Dict[str, Tuple[int, int]],
    constraint_weight: float = 5.0,
    label_binarizer=None,
    max_seq_len: int = 512,
    constraint_ramp_epochs: int = 5,
    sparse_target: bool = False,
    metric: str = 'accuracy'
) -> keras.Model:
    """
    Wrap an existing model with length constraint penalties.
    
    This function takes a model with a CRF layer and adds the length penalty
    computation to create a semi-Markov CRF approximation.
    
    Args:
        base_model: The base model with CRF layer
        length_constraints: Dict mapping label_name -> (min_length, max_length)
        constraint_weight: Final penalty multiplier (λ in theory)
        label_binarizer: sklearn LabelBinarizer for label mapping
        max_seq_len: Maximum sequence length (for XLA compilation)
        constraint_ramp_epochs: Number of epochs to ramp constraint from 0 to full
        sparse_target: If True, y is shape (batch, seq_len) with label indices
        metric: Metric for training (default 'accuracy')
        
    Returns:
        ModelWithLengthConstrainedCRF wrapper model
    """
    try:
        from tempest.core.length_crf import ModelWithLengthConstrainedCRF
    except ImportError:
        # Fallback to relative import
        from .length_crf import ModelWithLengthConstrainedCRF
    
    return ModelWithLengthConstrainedCRF(
        base_model=base_model,
        length_constraints=length_constraints,
        constraint_weight=constraint_weight,
        label_binarizer=label_binarizer,
        max_seq_len=max_seq_len,
        constraint_ramp_epochs=constraint_ramp_epochs,
        sparse_target=sparse_target,
        metric=metric
    )


def build_model_from_config(config) -> keras.Model:
    """
    Build model from configuration object.
    
    Args:
        config: ModelConfig or TempestConfig object
        
    Returns:
        Keras Model with optional length constraints
    """
    # Handle both ModelConfig and TempestConfig
    if hasattr(config, 'model'):
        model_config = config.model
    else:
        model_config = config
    
    # Extract length constraints if present
    length_constraints = getattr(model_config, 'length_constraints', None)
    constraint_weight = getattr(model_config, 'constraint_weight', 5.0)
    
    # Handle CNN filter/kernel configuration
    cnn_filters = getattr(model_config, 'cnn_filters', [64, 128])
    cnn_kernels = getattr(model_config, 'cnn_kernels', None)
    if cnn_kernels is None:
        cnn_kernels = getattr(model_config, 'cnn_kernel_size', [3, 5])
    
    # Build base model
    base_model = build_cnn_bilstm_crf(
        vocab_size=getattr(model_config, 'vocab_size', 5),
        embedding_dim=model_config.embedding_dim,
        cnn_filters=cnn_filters,
        cnn_kernels=cnn_kernels,
        lstm_units=model_config.lstm_units,
        lstm_layers=model_config.lstm_layers,
        dropout=model_config.dropout,
        num_labels=model_config.num_labels,
        max_seq_len=model_config.max_seq_len,
        use_cnn=model_config.use_cnn,
        use_bilstm=model_config.use_bilstm,
        use_crf=getattr(model_config, 'use_crf', True),
        use_attention=getattr(model_config, 'use_attention', False),
        attention_units=getattr(model_config, 'attention_units', 64),
        length_constraints=length_constraints,
        constraint_weight=constraint_weight
    )
    
    # If length constraints and we need to wrap the model
    if length_constraints and getattr(model_config, 'use_crf', True):
        label_binarizer = getattr(config, 'label_binarizer', None)
        constraint_ramp_epochs = getattr(model_config, 'constraint_ramp_epochs', 5)
        
        model = build_model_with_length_constraints(
            base_model=base_model,
            length_constraints=length_constraints,
            constraint_weight=constraint_weight,
            label_binarizer=label_binarizer,
            max_seq_len=model_config.max_seq_len,
            constraint_ramp_epochs=constraint_ramp_epochs
        )
        
        logger.info("Model wrapped with length constraint penalties")
        return model
    
    return base_model


def create_hybrid_model(
    base_model: keras.Model,
    length_constraints: Dict[str, Tuple[int, int]],
    label_binarizer,
    use_soft_constraints: bool = True,
    use_hard_constraints: bool = True,
    constraint_weight: float = 5.0,
    constraint_ramp_epochs: int = 5
) -> Tuple[keras.Model, object]:
    """
    Create a model with both soft and hard constraints.
    
    Args:
        base_model: Base CNN-BiLSTM-CRF model
        length_constraints: Length constraints for specific labels
        label_binarizer: Label encoder
        use_soft_constraints: Whether to add soft constraints (training)
        use_hard_constraints: Whether to use hard constraints (inference)
        constraint_weight: Weight for soft constraints
        constraint_ramp_epochs: Epochs to ramp up constraint weight
        
    Returns:
        Tuple of (model with soft constraints, hybrid decoder)
    """
    # Add soft constraints if requested
    if use_soft_constraints:
        model = build_model_with_length_constraints(
            base_model=base_model,
            length_constraints=length_constraints,
            constraint_weight=constraint_weight,
            label_binarizer=label_binarizer,
            constraint_ramp_epochs=constraint_ramp_epochs,
            sparse_target=True
        )
    else:
        model = base_model
    
    # Create hybrid decoder
    try:
        from tempest.core.hybrid_decoder import HybridConstraintDecoder
    except ImportError:
        # Fallback to relative import
        from .hybrid_decoder import HybridConstraintDecoder
    
    decoder = HybridConstraintDecoder(
        model=model,
        label_binarizer=label_binarizer,
        length_constraints=length_constraints,
        use_hard_constraints=use_hard_constraints
    )
    
    return model, decoder


def print_model_summary(model: keras.Model):
    """Print model architecture summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


# Utility functions for model inspection
def get_crf_layer(model: keras.Model):
    """Extract CRF layer from model."""
    for layer in model.layers:
        if 'crf' in layer.name.lower():
            return layer
    # Check if it's a wrapped model
    if hasattr(model, 'base_model'):
        for layer in model.base_model.layers:
            if 'crf' in layer.name.lower():
                return layer
    return None


def get_transition_matrix(model: keras.Model):
    """Extract transition matrix from CRF layer."""
    crf_layer = get_crf_layer(model)
    if crf_layer is not None:
        if hasattr(crf_layer, 'transitions'):
            return crf_layer.transitions.numpy()
        elif hasattr(crf_layer, 'get_transition_params'):
            return crf_layer.get_transition_params()
    return None


def validate_length_constraints(
    length_constraints: Dict[str, Tuple[int, int]], 
    label_names: List[str]
) -> Dict[str, Tuple[int, int]]:
    """
    Validate that length constraints match available labels.
    
    Args:
        length_constraints: Dict mapping label_name -> (min_length, max_length)
        label_names: List of available label names
        
    Returns:
        Validated length constraints dict
        
    Raises:
        ValueError: If constraint label not in available labels
    """
    if not length_constraints:
        return {}
    
    validated = {}
    for label_name, (min_len, max_len) in length_constraints.items():
        if label_name not in label_names:
            logger.warning(f"Label '{label_name}' in constraints not found in label set. Skipping.")
            continue
        if min_len > max_len:
            raise ValueError(f"Invalid constraint for '{label_name}': min_length ({min_len}) > max_length ({max_len})")
        validated[label_name] = (min_len, max_len)
    
    return validated
