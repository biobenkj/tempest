# Tempest Visualization Module

## Overview

The Tempest visualization module provides comprehensive tools for visualizing annotated reads and model predictions. This module was inspired by and adapted from tranquillyzer's visualization capabilities, integrated seamlessly with Tempest's architecture.

## Features

### Core Visualization Capabilities

1. **Annotated Read Visualization**
   - Color-coded sequence annotations
   - Multi-page PDF output for long sequences
   - Customizable colors for different labels
   - Metadata display for each read

2. **Prediction Statistics**
   - Label distribution charts
   - Proportion analysis
   - Batch-level statistics

3. **Model Integration**
   - Direct visualization from model outputs
   - Support for CRF and standard models
   - Batch processing capabilities

## Installation

The visualization module is included with Tempest. Ensure you have the required dependencies:

```bash
pip install matplotlib>=3.5.0
```

## Quick Start

### Basic Usage

```python
from tempest.visualization import save_plots_to_pdf

# Your sequences and predictions
sequences = ["ATCGATCGATCG...", "GCTAGCTAGCTA..."]
predictions = [pred_array1, pred_array2]
read_names = ["Read_001", "Read_002"]
label_names = ["adapter", "barcode", "UMI", "cDNA", "polyA"]

# Generate visualization PDF
save_plots_to_pdf(
    sequences=sequences,
    predictions=predictions,
    read_names=read_names,
    filename="output.pdf",
    label_names=label_names
)
```

### Using TempestVisualizer Class

```python
from tempest.visualization import TempestVisualizer

# Initialize visualizer
visualizer = TempestVisualizer(
    label_names=["adapter", "barcode", "UMI", "cDNA", "polyA"],
    output_dir="./visualizations"
)

# Visualize predictions
output_path = visualizer.visualize_predictions(
    sequences=sequences,
    predictions=predictions,
    output_filename="predictions.pdf",
    include_statistics=True
)
```

### Integration with Model Inference

```python
from tempest.inference.visualize_predictions import TempestInferenceVisualizer

# Initialize inference visualizer
viz = TempestInferenceVisualizer(
    config_path="model_config.json",
    model_path="model_weights.h5",
    output_dir="./results"
)

# Process and visualize FASTA file
viz.process_fasta_file(
    fasta_path="sequences.fasta",
    visualize=True,
    save_predictions=True
)
```

## Command Line Interface

### Visualize Model Predictions

```bash
python -m tempest.inference.visualize_predictions \
    input.fasta \
    --config model_config.json \
    --model model_weights.h5 \
    --output-dir ./results \
    --batch-size 32
```

### Run Visualization Examples

```bash
# Run all examples
python examples/visualization_demo.py --example all

# Run specific example
python examples/visualization_demo.py --example basic
```

## Customization Options

### Custom Color Schemes

Define custom colors for your labels:

```python
custom_colors = {
    'adapter': '#FF6B6B',    # Coral red
    'barcode': '#4ECDC4',    # Teal
    'UMI': '#45B7D1',        # Sky blue
    'cDNA': '#FFEAA7'        # Pale yellow
}

visualizer = TempestVisualizer(
    label_names=label_names,
    colors=custom_colors
)
```

### Visualization Parameters

- `chars_per_line`: Number of characters per line (default: 100)
- `max_chunks_per_page`: Maximum chunks per PDF page (default: 50)
- `header_max_length`: Maximum header text length (default: 100)

## API Reference

### Main Functions

#### `save_plots_to_pdf()`

Save visualization plots to a PDF file.

**Parameters:**
- `sequences`: List of sequence strings
- `predictions`: List of prediction arrays
- `read_names`: List of read names
- `filename`: Output PDF filename
- `colors`: Optional color mapping dictionary
- `label_names`: List of label names
- `chars_per_line`: Characters per line (default: 100)
- `max_chunks_per_page`: Max chunks per page (default: 50)
- `metadata_list`: Optional metadata for each sequence

#### `plot_annotation_statistics()`

Generate statistics plots for predictions.

**Parameters:**
- `predictions`: List of prediction arrays
- `label_names`: List of label names
- `output_file`: Output filename for plot
- `title`: Plot title

### TempestVisualizer Class

#### `__init__(label_names, colors=None, output_dir="./visualizations")`

Initialize the visualizer.

#### `visualize_predictions(sequences, predictions, ...)`

Visualize model predictions for sequences.

**Parameters:**
- `sequences`: Input sequences (strings or encoded arrays)
- `predictions`: Model predictions
- `read_names`: Optional sequence names
- `output_filename`: PDF filename
- `include_statistics`: Generate statistics plot
- `metadata_list`: Optional metadata

**Returns:** Path to generated PDF file

#### `visualize_model_outputs(model, sequences, ...)`

Visualize outputs directly from a Tempest model.

**Parameters:**
- `model`: Trained Tempest model
- `sequences`: Encoded input sequences
- `batch_size`: Batch size for prediction
- `output_filename`: Output PDF filename

## Examples

### Example 1: Basic Visualization

```python
from tempest.visualization import save_plots_to_pdf

# Simple example with 3 sequences
sequences = [
    "ATCGATCGATCGATCG" * 10,
    "GCTAGCTAGCTAGCTA" * 10,
    "TTAATTAATTAATTAA" * 10
]

# Mock predictions (as integer labels)
predictions = [
    np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 17)[:160],
    np.array([1, 1, 1, 2, 2, 2, 0, 0, 0] * 17)[:160],
    np.array([2, 2, 2, 0, 0, 0, 1, 1, 1] * 17)[:160]
]

read_names = ["Read_A", "Read_B", "Read_C"]
label_names = ["adapter", "barcode", "UMI"]

save_plots_to_pdf(
    sequences=sequences,
    predictions=predictions,
    read_names=read_names,
    filename="example_output.pdf",
    label_names=label_names
)
```

### Example 2: Batch Processing

```python
from tempest.visualization import visualize_batch_predictions

# Visualize predictions for multiple batches
output_files = visualize_batch_predictions(
    model=trained_model,
    data_generator=test_generator,
    label_names=label_names,
    num_batches=5,
    output_dir="./batch_visualizations"
)
```

### Example 3: Full Pipeline Integration

```python
import tempest
from tempest.visualization import TempestVisualizer

# Load model
model = tempest.load_model("model_path")

# Prepare data
sequences = load_sequences("data.fasta")
encoded = encode_sequences(sequences)

# Get predictions
predictions = model.predict(encoded)

# Visualize with metadata
metadata = [{"GC_content": calculate_gc(seq)} for seq in sequences]

visualizer = TempestVisualizer(label_names=model.label_names)
visualizer.visualize_predictions(
    sequences=sequences,
    predictions=predictions,
    metadata_list=metadata,
    output_filename="full_analysis.pdf"
)
```

## Tips and Best Practices

1. **Memory Management**: For large datasets, process sequences in batches to avoid memory issues.

2. **Color Selection**: Use contrasting colors for adjacent labels to improve readability.

3. **PDF Size**: Long sequences generate large PDFs. Consider limiting `max_chunks_per_page` for very long reads.

4. **Performance**: Enable GPU acceleration for model inference before visualization:
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

5. **Label Names**: Use descriptive label names for clearer visualizations.

## Troubleshooting

### Common Issues

1. **Memory Error with Large PDFs**
   - Solution: Reduce `max_chunks_per_page` or process fewer sequences at once

2. **Missing Labels in Visualization**
   - Check that `label_names` matches the number of classes in predictions

3. **Slow Performance**
   - Process sequences in smaller batches
   - Use GPU for model inference
   - Reduce DPI in visualization (modify in code)

## Integration with Existing Workflows

The visualization module is designed to integrate seamlessly with existing Tempest workflows:

```python
# Standard Tempest workflow
from tempest import load_model, process_sequences
from tempest.visualization import TempestVisualizer

# Your existing code
model = load_model("model.h5")
sequences = load_sequences("data.fasta")
predictions = model.predict(sequences)

# Add visualization
visualizer = TempestVisualizer(label_names=model.get_labels())
visualizer.visualize_predictions(
    sequences=sequences,
    predictions=predictions,
    output_filename="results.pdf"
)
```

## Future Enhancements

Planned features for future versions:
- Interactive HTML visualizations
- Real-time visualization during training
- Comparative visualization between models
- Export to other formats (SVG, PNG)
- Integration with sequence browsers

## Credits

This visualization module was inspired by and incorporates concepts from the tranquillyzer package's visualization capabilities, adapted and enhanced for Tempest's architecture and requirements.
