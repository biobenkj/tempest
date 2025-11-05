"""
Visualization module for annotated reads in Tempest.
Adapted from tranquillyzer's visualization capabilities.
"""

import numpy as np
from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def visualize_sequence_annotations(
    colors: Dict[str, str],
    read_name: str,
    read: str,
    predicted_labels: List[str],
    architecture: Optional[str] = None,
    reason: Optional[str] = None,
    chars_per_line: int = 100,
    header_max_length: int = 100,
    max_chunks_per_page: int = 50,
    metadata: Optional[Dict[str, Any]] = None
) -> List[plt.Figure]:
    """
    Visualize sequence annotations for a single read.
    
    Args:
        colors: Dictionary mapping labels to colors
        read_name: Name/ID of the read
        read: The sequence string
        predicted_labels: List of predicted labels for each position
        architecture: Optional architecture information
        reason: Optional reason/classification
        chars_per_line: Number of characters per line in visualization
        header_max_length: Maximum length for header text
        max_chunks_per_page: Maximum number of chunks per page
        metadata: Optional additional metadata to display
        
    Returns:
        List of matplotlib figure objects
    """
    if not read:
        logger.warning(f"Empty read for {read_name}. Skipping this read.")
        return []
    
    # Ensure read and predicted_labels have the same length
    predicted_labels = predicted_labels[0:len(read)]
    if len(read) != len(predicted_labels):
        logger.error(f"Length mismatch between read and predicted_labels for {read_name}. Skipping this read.")
        return []
    
    num_chunks = int(np.ceil(len(read) / chars_per_line))
    read_chunks = [read[i * chars_per_line: (i + 1) * chars_per_line] for i in range(num_chunks)]
    label_chunks = [predicted_labels[i * chars_per_line: (i + 1) * chars_per_line] for i in range(num_chunks)]
    
    # Build header text
    header_parts = [read_name]
    if architecture:
        header_parts.append(f"Architecture: {architecture}")
    if reason:
        header_parts.append(f"Reason: {reason}")
    if metadata:
        for key, value in metadata.items():
            header_parts.append(f"{key}: {value}")
    
    header_text = " | ".join(header_parts)
    wrapped_header = fill(header_text, width=header_max_length)
    
    figures = []
    
    for page_start in range(0, num_chunks, max_chunks_per_page):
        # Select the chunks for the current page
        page_end = min(page_start + max_chunks_per_page, num_chunks)
        page_read_chunks = read_chunks[page_start:page_end]
        page_label_chunks = label_chunks[page_start:page_end]
        
        # Calculate the figure height dynamically
        fixed_font_size = 10
        chunk_height = 0.6
        header_height = 1.0
        num_page_chunks = page_end - page_start
        
        fig_height = header_height + num_page_chunks * chunk_height
        fig, axs = plt.subplots(
            num_page_chunks + 1, 1,
            figsize=(15, fig_height),
            gridspec_kw={'height_ratios': [header_height] + [1] * num_page_chunks},
            dpi=300
        )
        
        # Handle single subplot case
        if num_page_chunks == 0:
            axs = [axs]
        elif num_page_chunks == 1:
            axs = [axs[0], axs[1]]
        
        # Turn off axis for each subplot
        for ax in axs:
            ax.axis('off')
        
        # Display the header
        axs[0].text(
            0.5, 0.5, wrapped_header,
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            transform=axs[0].transAxes
        )
        
        # Display the read sequence and labels chunk by chunk
        for ax_idx, (read_chunk, label_chunk) in enumerate(zip(page_read_chunks, page_label_chunks), start=1):
            ax = axs[ax_idx]
            start_idx = 0
            current_label = label_chunk[0] if label_chunk else None
            
            for idx, (base, label) in enumerate(zip(read_chunk, label_chunk)):
                if label:
                    color = colors.get(label, 'black')
                    ax.text(
                        idx / chars_per_line, 1, base,
                        ha='center', va='center',
                        color=color,
                        fontsize=fixed_font_size,
                        fontweight='medium'
                    )
                
                # Handle label positioning and separation
                if current_label != label or idx == len(read_chunk) - 1:
                    if current_label:
                        label_position = start_idx / chars_per_line + (idx - start_idx) / (2 * chars_per_line)
                        color = colors.get(current_label, 'black')
                        ax.text(
                            label_position, 0.5, current_label,
                            ha='center', va='center',
                            color=color,
                            fontsize=fixed_font_size
                        )
                    
                    start_idx = idx
                    current_label = label
        
        plt.subplots_adjust(hspace=0.1)
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def save_plots_to_pdf(
    sequences: List[str],
    predictions: List[np.ndarray],
    read_names: List[str],
    filename: str,
    colors: Optional[Dict[str, str]] = None,
    label_names: Optional[List[str]] = None,
    chars_per_line: int = 100,
    max_chunks_per_page: int = 50,
    metadata_list: Optional[List[Dict[str, Any]]] = None
):
    """
    Save visualization plots to a PDF file.
    
    Args:
        sequences: List of sequence strings
        predictions: List of prediction arrays (can be integer labels or probabilities)
        read_names: List of read names
        filename: Output PDF filename
        colors: Dictionary mapping labels to colors (if None, uses default)
        label_names: List of label names for index to label conversion
        chars_per_line: Number of characters per line
        max_chunks_per_page: Maximum chunks per page
        metadata_list: Optional list of metadata dictionaries for each sequence
    """
    if colors is None:
        colors = get_default_colors(label_names)
    
    if metadata_list is None:
        metadata_list = [{}] * len(sequences)
    
    with PdfPages(filename) as pdf:
        for idx, (sequence, prediction, read_name, metadata) in enumerate(
            zip(sequences, predictions, read_names, metadata_list)
        ):
            if not sequence:
                logger.warning(f"Empty sequence for {read_name}. Skipping.")
                continue
            
            # Convert predictions to labels if necessary
            if len(prediction.shape) > 1:  # Probability matrix
                predicted_indices = np.argmax(prediction, axis=-1)
            else:  # Already integer labels
                predicted_indices = prediction
            
            # Convert indices to label names
            if label_names:
                predicted_labels = [
                    label_names[idx] if 0 <= idx < len(label_names) else 'unknown'
                    for idx in predicted_indices
                ]
            else:
                predicted_labels = [str(idx) for idx in predicted_indices]
            
            # Visualize and generate figures for each read
            figures = visualize_sequence_annotations(
                colors, read_name,
                sequence,
                predicted_labels,
                metadata=metadata,
                chars_per_line=chars_per_line,
                max_chunks_per_page=max_chunks_per_page
            )
            
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
    
    logger.info(f"Saved visualization to {filename}")


def get_default_colors(label_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Generate default color palette for labels.
    
    Args:
        label_names: Optional list of label names
        
    Returns:
        Dictionary mapping labels to colors
    """
    # Base palette
    palette = ['red', 'blue', 'green', 'purple', 'pink',
               'cyan', 'magenta', 'orange', 'brown', 'olive',
               'navy', 'teal', 'lime', 'indigo', 'coral']
    
    # Special colors for common biological elements
    special_colors = {
        'random_s': 'black',
        'random_e': 'black',
        'cDNA': 'gray',
        'polyT': 'orange',
        'polyA': 'orange',
        'adapter': 'red',
        'barcode': 'blue',
        'UMI': 'green',
        'primer': 'purple',
        'spacer': 'lightgray',
        'background': 'white',
        'unknown': 'darkgray'
    }
    
    colors = {}
    
    if label_names:
        color_idx = 0
        for label in label_names:
            if label in special_colors:
                colors[label] = special_colors[label]
            else:
                colors[label] = palette[color_idx % len(palette)]
                color_idx += 1
    else:
        colors = special_colors.copy()
    
    return colors


def plot_annotation_statistics(
    predictions: List[np.ndarray],
    label_names: List[str],
    output_file: str,
    title: str = "Annotation Statistics"
):
    """
    Plot statistics about the predicted annotations.
    
    Args:
        predictions: List of prediction arrays
        label_names: List of label names
        output_file: Output filename for the plot
        title: Title for the plot
    """
    # Collect label counts
    label_counts = {label: 0 for label in label_names}
    total_positions = 0
    
    for prediction in predictions:
        if len(prediction.shape) > 1:  # Probability matrix
            predicted_indices = np.argmax(prediction, axis=-1)
        else:  # Already integer labels
            predicted_indices = prediction
        
        for idx in predicted_indices:
            if 0 <= idx < len(label_names):
                label_counts[label_names[idx]] += 1
                total_positions += 1
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart of counts
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    ax1.bar(labels, counts)
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Label Counts')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart of proportions
    non_zero_labels = [l for l, c in zip(labels, counts) if c > 0]
    non_zero_counts = [c for c in counts if c > 0]
    
    if non_zero_counts:
        ax2.pie(non_zero_counts, labels=non_zero_labels, autopct='%1.1f%%')
        ax2.set_title(f'{title} - Label Proportions')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved annotation statistics to {output_file}")
