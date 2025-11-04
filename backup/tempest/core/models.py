"""
Model architecture builder for Tempest.

Builds CNN-BiLSTM-CRF models for sequence annotation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

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
    use_crf: bool = True
) -> keras.Model:
    """
    Build CNN-BiLSTM-CRF model for sequence annotation.
    
    Architecture:
    1. Embedding: Convert base indices to dense vectors
    2. CNN (optional): Extract local motifs and k-mer features
    3. BiLSTM (optional): Model long-range dependencies
    4. CRF (optional): Structured prediction with transition constraints
    
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
        
    Returns:
        Keras Model
    """
    logger.info("Building CNN-BiLSTM-CRF model...")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  CNN: {use_cnn} (filters={cnn_filters}, kernels={cnn_kernels})")
    logger.info(f"  BiLSTM: {use_bilstm} (units={lstm_units}, layers={lstm_layers})")
    logger.info(f"  CRF: {use_crf}")
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
    
    # Output layer
    if use_crf:
        # CRF requires special handling - for now, we'll use a dense layer
        # with softmax and add proper CRF layer integration later
        logger.warning("CRF layer not yet integrated - using Dense + Softmax")
        outputs = layers.Dense(num_labels, activation='softmax', name='output')(x)
    else:
        outputs = layers.Dense(num_labels, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_bilstm_crf')
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    
    return model


def build_model_from_config(config) -> keras.Model:
    """
    Build model from configuration object.
    
    Args:
        config: ModelConfig or TempestConfig object
        
    Returns:
        Keras Model
    """
    # Handle both ModelConfig and TempestConfig
    if hasattr(config, 'model'):
        model_config = config.model
    else:
        model_config = config
    
    return build_cnn_bilstm_crf(
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
        use_crf=model_config.use_crf
    )


def print_model_summary(model: keras.Model):
    """Print model architecture summary."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")
