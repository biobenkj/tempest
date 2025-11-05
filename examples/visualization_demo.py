#!/usr/bin/env python3
"""
Example script demonstrating the use of Tempest's visualization capabilities
for annotated reads.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tempest.visualization import (
    TempestVisualizer,
    visualize_predictions,
    save_plots_to_pdf,
    plot_annotation_statistics
)


def generate_example_data(num_sequences=5, seq_length=200, num_labels=8):
    """
    Generate example data for demonstration.
    
    Args:
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        num_labels: Number of distinct labels
        
    Returns:
        Tuple of (sequences, predictions, label_names)
    """
    # Generate random sequences
    bases = ['A', 'C', 'G', 'T']
    sequences = []
    for i in range(num_sequences):
        seq = ''.join(np.random.choice(bases, seq_length))
        sequences.append(seq)
    
    # Generate mock predictions (as probabilities)
    predictions = []
    for i in range(num_sequences):
        # Create blocks of different labels to simulate real annotations
        pred = np.zeros((seq_length, num_labels))
        
        # Create segments with different labels
        segment_starts = [0, 20, 50, 100, 150, seq_length]
        segment_labels = np.random.choice(num_labels, len(segment_starts) - 1)
        
        for j in range(len(segment_starts) - 1):
            start = segment_starts[j]
            end = segment_starts[j + 1]
            label = segment_labels[j]
            pred[start:end, label] = 1.0
        
        predictions.append(pred)
    
    # Define label names
    label_names = [
        'adapter', 'barcode', 'UMI', 'primer',
        'cDNA', 'polyA', 'spacer', 'random'
    ][:num_labels]
    
    return sequences, predictions, label_names


def example_basic_visualization():
    """
    Demonstrate basic visualization of annotated reads.
    """
    print("Generating basic visualization example...")
    
    # Generate example data
    sequences, predictions, label_names = generate_example_data(
        num_sequences=3,
        seq_length=300,
        num_labels=6
    )
    
    # Create read names
    read_names = [f"Example_Read_{i:03d}" for i in range(len(sequences))]
    
    # Create metadata for each read
    metadata_list = [
        {'length': len(seq), 'quality': np.random.uniform(20, 40)}
        for seq in sequences
    ]
    
    # Save visualization to PDF
    output_file = "example_basic_visualization.pdf"
    save_plots_to_pdf(
        sequences=sequences,
        predictions=predictions,
        read_names=read_names,
        filename=output_file,
        label_names=label_names,
        metadata_list=metadata_list
    )
    
    print(f"  Saved visualization to {output_file}")
    
    # Also create statistics plot
    stats_file = "example_annotation_statistics.png"
    plot_annotation_statistics(
        predictions=predictions,
        label_names=label_names,
        output_file=stats_file,
        title="Example Annotation Statistics"
    )
    
    print(f"  Saved statistics to {stats_file}")


def example_visualizer_class():
    """
    Demonstrate using the TempestVisualizer class.
    """
    print("\nUsing TempestVisualizer class...")
    
    # Generate example data
    sequences, predictions, label_names = generate_example_data(
        num_sequences=5,
        seq_length=250,
        num_labels=7
    )
    
    # Create visualizer
    visualizer = TempestVisualizer(
        label_names=label_names,
        output_dir="./visualization_output"
    )
    
    # Visualize predictions
    output_path = visualizer.visualize_predictions(
        sequences=sequences,
        predictions=predictions,
        output_filename="tempest_predictions.pdf",
        include_statistics=True
    )
    
    print(f"  Saved visualization to {output_path}")


def example_custom_colors():
    """
    Demonstrate custom color schemes for visualization.
    """
    print("\nGenerating visualization with custom colors...")
    
    # Generate example data
    sequences, predictions, label_names = generate_example_data(
        num_sequences=2,
        seq_length=200,
        num_labels=5
    )
    
    # Define custom colors
    custom_colors = {
        'adapter': '#FF6B6B',    # Coral red
        'barcode': '#4ECDC4',    # Teal
        'UMI': '#45B7D1',        # Sky blue
        'primer': '#96CEB4',     # Sage green
        'cDNA': '#FFEAA7'        # Pale yellow
    }
    
    # Create visualizer with custom colors
    visualizer = TempestVisualizer(
        label_names=label_names,
        colors=custom_colors,
        output_dir="./visualization_output"
    )
    
    # Visualize
    output_path = visualizer.visualize_predictions(
        sequences=sequences,
        predictions=predictions,
        output_filename="custom_colors_example.pdf"
    )
    
    print(f"  Saved custom color visualization to {output_path}")


def example_encoded_sequences():
    """
    Demonstrate visualization with encoded sequences (as from model input).
    """
    print("\nVisualizing encoded sequences...")
    
    # Generate encoded sequences (as integers)
    num_sequences = 3
    seq_length = 150
    
    # Encode as integers (0=pad, 1=A, 2=C, 3=G, 4=T)
    encoded_sequences = np.random.randint(1, 5, size=(num_sequences, seq_length))
    
    # Generate predictions
    num_labels = 6
    predictions = np.random.rand(num_sequences, seq_length, num_labels)
    
    # Label names
    label_names = ['adapter', 'barcode', 'UMI', 'primer', 'cDNA', 'polyA']
    
    # Create visualizer
    visualizer = TempestVisualizer(
        label_names=label_names,
        output_dir="./visualization_output"
    )
    
    # Visualize (will automatically decode sequences)
    output_path = visualizer.visualize_predictions(
        sequences=encoded_sequences,
        predictions=predictions,
        output_filename="encoded_sequences_example.pdf"
    )
    
    print(f"  Saved encoded sequence visualization to {output_path}")


def main():
    """
    Main function to run all examples.
    """
    parser = argparse.ArgumentParser(
        description="Tempest visualization examples"
    )
    parser.add_argument(
        '--example',
        choices=['all', 'basic', 'class', 'colors', 'encoded'],
        default='all',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEMPEST VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    # Create output directory
    Path("./visualization_output").mkdir(exist_ok=True)
    
    if args.example == 'all':
        example_basic_visualization()
        example_visualizer_class()
        example_custom_colors()
        example_encoded_sequences()
    elif args.example == 'basic':
        example_basic_visualization()
    elif args.example == 'class':
        example_visualizer_class()
    elif args.example == 'colors':
        example_custom_colors()
    elif args.example == 'encoded':
        example_encoded_sequences()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("Check the current directory and ./visualization_output/ for outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
