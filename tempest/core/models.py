"""
Model architecture builder for Tempest.

Builds CNN-BiLSTM-CRF models for sequence annotation with length constraints.
Integrates the length-constrained CRF layer for semi-Markov approximation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from .length_crf import LengthConstrainedCRF, ModelWithLengthConstrainedCRF

logger = logging.getLogger(__name__)


def build_cnn_bilstm_crf(
    vocab_size: int = 5,
    embedding_dim: int = 128,
    cnn_filters: list = [64, 128],
    cnn_kernels: list = [3, 5],
    lstm_units: int = 128,
    lstm_layers: int = 2,
    dropout: float = 0.3,
    num_labels: int = 10,
    max_seq_len: int = 512,
    use_cnn: bool = True,
    use_bilstm: bool = True,
    use_crf: bool = True,
    length_constraints: dict = None,
    constraint_weight: float = 5.0
) -> keras.Model:
    """
    Build CNN-BiLSTM-CRF model for sequence annotation with optional length constraints.
    
    Architecture:
    1. Embedding: Convert base indices to dense vectors
    2. CNN (optional): Extract local motifs and k-mer features  
    3. BiLSTM (optional): Model long-range dependencies
    4. CRF (optional): Structured prediction with transition constraints
    5. Length constraints: Semi-Markov approximation for segment length control
    
    Args:
        vocab_size: Size of vocabulary (5 for A, C, G, T, N)
        embedding_dim: Dimension of embedding vectors
        cnn_filters: List of filter sizes for CNN layers
        cnn_kernels: List of kernel sizes for CNN layers
        lstm_units: Number of LSTM units
        lstm_layers: Number of BiLSTM layers
        dropout: Dropout rate
        num_labels: Number of output labels
        max_seq_len: Maximum sequence length
        use_cnn: Whether to include CNN layers
        use_bilstm: Whether to include BiLSTM layers
        use_crf: Whether to use CRF output layer
        length_constraints: Dict mapping label_name -> (min_length, max_length)
                          Example: {'UMI': (8, 8), 'ACC': (6, 6), 'BARCODE': (16, 16)}
        constraint_weight: Penalty multiplier for length violations
        
    Returns:
        Keras Model (wrapped with length constraints if specified)
    """
    logger.info("Building CNN-BiLSTM-CRF model...")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  CNN: {use_cnn} (filters={cnn_filters}, kernels={cnn_kernels})")
    logger.info(f"  BiLSTM: {use_bilstm} (units={lstm_units}, layers={lstm_layers})")
    logger.info(f"  CRF: {use_crf}")
    logger.info(f"  Length constraints: {length_constraints}")
    logger.info(f"  Number of labels: {num_labels}")
    
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
            x = layers.Bidirectional(
                layers.LSTM(
                    lstm_units,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=0.0,  # Avoid issues with CuDNN
                    name=f'lstm_{i+1}'
                ),
                name=f'bilstm_{i+1}'
            )(x)
    
    # Dense layer for emissions
    x = layers.Dense(num_labels, name='emissions')(x)
    
    # Output layer with CRF
    if use_crf:
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
            from tf2crf import CRF
            crf_layer = CRF(units=num_labels, name='crf')
        
        outputs = crf_layer(x)
    else:
        # Simple softmax output without CRF
        outputs = layers.Activation('softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_bilstm_crf')
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    
    return model


def build_model_with_length_constraints(
    base_model: keras.Model,
    length_constraints: dict,
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
    
    # Build base model
    base_model = build_cnn_bilstm_crf(
        vocab_size=model_config.vocab_size,
        embedding_dim=model_config.embedding_dim,
        cnn_filters=model_config.cnn_filters,
        cnn_kernels=model_config.cnn_kernels,
        lstm_units=model_config.lstm_units,
        lstm_layers=model_config.lstm_layers,
        dropout=model_config.dropout,
        num_labels=model_config.num_labels,
        max_seq_len=model_config.max_seq_len,
        use_cnn=model_config.use_cnn,
        use_bilstm=model_config.use_bilstm,
        use_crf=model_config.use_crf,
        length_constraints=length_constraints,
        constraint_weight=constraint_weight
    )
    
    # If length constraints and we need to wrap the model
    if length_constraints and model_config.use_crf:
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
        
        logger.info("✓ Model wrapped with length constraint penalties")
        return model
    
    return base_model


def print_model_summary(model: keras.Model):
    """Print model architecture summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")
