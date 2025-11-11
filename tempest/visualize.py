"""
Tempest visualization.

This module provides CLI commands for visualizing model predictions,
training history, segment distributions, and other analysis outputs.
"""

import typer
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from rich.console import Console
import numpy as np

from tempest.visualization import TempestVisualizer
from tempest.inference.inference_utils import encode_sequences
from tempest.main import load_config, load_data

logger = logging.getLogger(__name__)

# Create the visualize sub-application
visualize_app = typer.Typer(help="Create visualizations for models and results")

console = Console()


def get_custom_objects() -> Dict[str, Any]:
    """
    Get custom objects for loading Tempest models.
    
    Collects all custom layers from tempest.core that may be needed
    for loading trained models.
    
    Returns
    -------
    dict
        Dictionary of custom object names to classes
    """
    custom_objects = {}
    
    # Try to import custom CRF layers from tempest.core
    try:
        from tempest.core.length_crf import LengthConstrainedCRF
        custom_objects['LengthConstrainedCRF'] = LengthConstrainedCRF
        logger.debug("Added LengthConstrainedCRF to custom objects")
    except ImportError:
        logger.debug("LengthConstrainedCRF not available")
    
    try:
        from tempest.core.length_crf import ModelWithLengthConstrainedCRF
        custom_objects['ModelWithLengthConstrainedCRF'] = ModelWithLengthConstrainedCRF
        logger.debug("Added ModelWithLengthConstrainedCRF to custom objects")
    except ImportError:
        logger.debug("ModelWithLengthConstrainedCRF not available")
    
    # Try standard CRF from tf2crf as fallback
    try:
        from tf2crf import CRF
        custom_objects['CRF'] = CRF
        logger.debug("Added CRF (tf2crf) to custom objects")
    except ImportError:
        logger.debug("tf2crf CRF not available")
    
    # Try hybrid decoder if present
    try:
        from tempest.core.hybrid_decoder import HybridConstraintDecoder
        custom_objects['HybridConstraintDecoder'] = HybridConstraintDecoder
        logger.debug("Added HybridConstraintDecoder to custom objects")
    except ImportError:
        logger.debug("HybridConstraintDecoder not available")
    
    return custom_objects


def load_model_with_custom_objects(model_path: str):
    """
    Load a Tempest model with proper custom object handling.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    
    Returns
    -------
    tf.keras.Model
        Loaded model
    """
    import tensorflow as tf
    
    # Get all available custom objects
    custom_objects = get_custom_objects()
    
    if custom_objects:
        logger.info(f"Loading model with custom objects: {list(custom_objects.keys())}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
    else:
        logger.info("Loading model without custom objects")
        model = tf.keras.models.load_model(model_path, compile=False)
    
    return model


def run_visualization(
    config,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main visualization pipeline runner called by main.py dispatch.
    
    This function serves as the entry point for the visualization subcommand
    when called through the main pipeline runner.
    
    Parameters
    ----------
    config : TempestConfig
        Configuration object with visualization settings
    output_dir : str, optional
        Directory for output visualizations
    **kwargs : dict
        Additional arguments (model_path, data_path, etc.)
    
    Returns
    -------
    dict
        Results including paths to generated visualizations
    """
    logger.info("Running Tempest visualization pipeline")
    
    # Set up output directory
    if output_dir is None:
        output_dir = config.output.save_dir if hasattr(config, 'output') else "./visualizations"
    
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'output_dir': str(vis_dir),
        'generated_files': []
    }
    
    # Extract visualization parameters from config or kwargs
    model_path = kwargs.get('model_path') or kwargs.get('model')
    data_path = kwargs.get('data_path') or kwargs.get('data')
    
    if not model_path:
        logger.error("No model path provided for visualization")
        raise ValueError("model_path is required for visualization")
    
    # Initialize visualizer
    label_names = config.model.label_names if hasattr(config.model, 'label_names') else None
    
    if not label_names and hasattr(config, 'simulation'):
        # Infer from sequence order
        label_names = config.simulation.sequence_order
    
    if not label_names:
        raise ValueError("Could not determine label names from config")
    
    visualizer = TempestVisualizer(
        label_names=label_names,
        output_dir=str(vis_dir)
    )
    
    # If data path provided, visualize predictions
    if data_path:
        logger.info(f"Loading data from {data_path}")
        
        # Load the data
        data = load_data(data_path)
        
        # Extract sequences and prepare for visualization
        if isinstance(data, list):
            sequences = [item.get('sequence', '') for item in data]
            read_names = [item.get('name', f'Read_{i:04d}') for i in range(len(data))]
        else:
            sequences = data.get('sequences', [])
            read_names = data.get('names', [f'Read_{i:04d}' for i in range(len(sequences))])
        
        # Load model with custom objects
        logger.info(f"Loading model from {model_path}")
        
        try:
            model = load_model_with_custom_objects(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Encode sequences using inference module
        logger.info("Encoding sequences...")
        encoded_sequences = encode_sequences(sequences[:100])  # Limit to first 100
        
        # Get predictions
        logger.info("Running inference...")
        predictions = model.predict(
            encoded_sequences,
            batch_size=config.training.batch_size if hasattr(config, 'training') else 32
        )
        
        # Create visualization
        output_file = visualizer.visualize_predictions(
            sequences=sequences[:100],
            predictions=predictions,
            read_names=read_names[:100],
            output_filename="predictions.pdf",
            include_statistics=True
        )
        
        results['generated_files'].append(output_file)
        logger.info(f"Generated prediction visualization: {output_file}")
    
    logger.info(f"Visualization complete. Files saved to {vis_dir}")
    return results


@visualize_app.command(name="predictions")
def visualize_predictions_cmd(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model (.h5 or SavedModel)",
        exists=True
    ),
    data: Path = typer.Option(
        ...,
        "--data", "-d",
        help="Path to data file (pickle, text, or FASTQ)",
        exists=True
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file (for label names and settings)",
        exists=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output PDF file (default: predictions.pdf)"
    ),
    num_samples: int = typer.Option(
        100,
        "--num-samples", "-n",
        help="Number of samples to visualize",
        min=1
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for model prediction"
    ),
    include_stats: bool = typer.Option(
        True,
        "--stats/--no-stats",
        help="Include annotation statistics plot"
    )
):
    """
    Visualize model predictions on sequences.
    
    This command loads a trained model, runs predictions on provided data,
    and generates detailed sequence-level visualizations showing the predicted
    segment annotations.
    
    Examples:
    
        tempest visualize predictions -m model.h5 -d test_data.pkl
        
        tempest visualize predictions -m model.h5 -d test.txt -c config.yaml
        
        tempest visualize predictions -m model.h5 -d test.pkl -n 50 --no-stats
    """
    console.print("[bold blue]Visualizing Model Predictions[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Data: {data}")
    
    # Load configuration if provided
    if config:
        cfg = load_config(config)
        label_names = cfg.simulation.sequence_order
    else:
        # Try to infer from model or use defaults
        console.print("[yellow]Warning: No config provided, using default labels[/yellow]")
        label_names = ["p7", "i7", "RP2", "UMI", "ACC", "cDNA", "polyA", "CBC", "RP1", "i5", "p5"]
    
    # Load data
    console.print(f"Loading data...")
    data_obj = load_data(data)
    
    if isinstance(data_obj, list):
        sequences = [item.get('sequence', '') for item in data_obj]
        read_names = [item.get('name', f'Read_{i:04d}') for i in range(len(data_obj))]
    else:
        sequences = data_obj.get('sequences', [])
        read_names = data_obj.get('names', [f'Read_{i:04d}' for i in range(len(sequences))])
    
    # Limit to requested number of samples
    sequences = sequences[:num_samples]
    read_names = read_names[:num_samples]
    
    console.print(f"Loaded {len(sequences)} sequences")
    
    # Load model with custom objects
    console.print(f"Loading model...")
    try:
        model_obj = load_model_with_custom_objects(str(model))
        console.print(f"[green]Model loaded successfully[/green]")
        
        # Show which custom layers were loaded
        custom_objects = get_custom_objects()
        if custom_objects:
            console.print(f"[dim]Custom layers: {', '.join(custom_objects.keys())}[/dim]")
    
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        logger.exception("Model loading failed")
        raise typer.Exit(1)
    
    # Set up output
    output_dir = Path(output).parent if output else Path("./visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output or (output_dir / "predictions.pdf")
    
    # Initialize visualizer
    visualizer = TempestVisualizer(
        label_names=label_names,
        output_dir=str(output_dir)
    )
    
    # Generate predictions and visualize
    console.print(f"Encoding sequences and generating predictions...")
    
    try:
        # Encode sequences using inference module
        encoded = encode_sequences(sequences)
        console.print(f"[dim]Encoded shape: {encoded.shape}[/dim]")
        
        # Get predictions
        predictions = model_obj.predict(encoded, batch_size=batch_size, verbose=0)
        console.print(f"[dim]Predictions shape: {predictions.shape}[/dim]")
        
        # Create visualization
        result_path = visualizer.visualize_predictions(
            sequences=sequences,
            predictions=predictions,
            read_names=read_names,
            output_filename=output_file.name,
            include_statistics=include_stats
        )
        
        console.print(f"[green]Visualization saved to: {result_path}[/green]")
        
        if include_stats:
            stats_file = str(output_file).replace('.pdf', '_stats.png')
            console.print(f"[green]Statistics saved to: {stats_file}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during visualization: {e}[/red]")
        logger.exception("Visualization failed")
        raise typer.Exit(1)


@visualize_app.command(name="training-history")
def training_history(
    log_file: Path = typer.Option(
        ...,
        "--log-file", "-l",
        help="Path to training log file (JSON or CSV)",
        exists=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for plot (default: training_history.png)"
    ),
    metrics: List[str] = typer.Option(
        ["loss", "accuracy"],
        "--metric", "-m",
        help="Metrics to plot (can specify multiple times)"
    ),
    smooth: float = typer.Option(
        0.0,
        "--smooth",
        help="Smoothing factor (0-1)",
        min=0.0,
        max=1.0
    )
):
    """
    Plot training history from log files.
    
    Examples:
    
        tempest visualize training-history -l training.log
        
        tempest visualize training-history -l history.json -m loss -m accuracy --smooth 0.8
    """
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    console.print("[bold blue]Plotting Training History[/bold blue]")
    
    # Load training history
    try:
        with open(log_file) as f:
            if log_file.suffix == '.json':
                history = json.load(f)
            else:
                # Try to parse as JSON anyway
                history = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading log file: {e}[/red]")
        raise typer.Exit(1)
    
    # Create plots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in history:
            console.print(f"[yellow]Warning: Metric '{metric}' not found in history[/yellow]")
            continue
            
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        # Apply smoothing if requested
        if smooth > 0:
            smoothed = []
            for i, v in enumerate(values):
                if i == 0:
                    smoothed.append(v)
                else:
                    smoothed.append(smooth * smoothed[-1] + (1 - smooth) * v)
            axes[idx].plot(epochs, smoothed, label=f'{metric} (smoothed)', linewidth=2)
            axes[idx].plot(epochs, values, alpha=0.3, label=f'{metric} (raw)', linewidth=1)
        else:
            axes[idx].plot(epochs, values, label=metric, linewidth=2)
        
        # Add validation if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            val_values = history[val_metric]
            if smooth > 0:
                val_smoothed = []
                for i, v in enumerate(val_values):
                    if i == 0:
                        val_smoothed.append(v)
                    else:
                        val_smoothed.append(smooth * val_smoothed[-1] + (1 - smooth) * v)
                axes[idx].plot(epochs, val_smoothed, label=f'{val_metric} (smoothed)', 
                             linestyle='--', linewidth=2)
                axes[idx].plot(epochs, val_values, alpha=0.3, label=f'{val_metric} (raw)', 
                             linestyle='--', linewidth=1)
            else:
                axes[idx].plot(epochs, val_values, label=val_metric, linestyle='--', linewidth=2)
        
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'Training {metric.capitalize()}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output or Path("training_history.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Training history plot saved to: {output_path}[/green]")


@visualize_app.command(name="segment-distribution")
def segment_distribution(
    data: Path = typer.Option(
        ...,
        "--data", "-d",
        help="Path to data file with segment annotations",
        exists=True
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file for segment definitions"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for plot (default: segment_distributions.png)"
    ),
    bins: int = typer.Option(
        50,
        "--bins",
        help="Number of bins for histograms",
        min=10,
        max=200
    )
):
    """
    Visualize segment length distributions in the dataset.
    
    Examples:
    
        tempest visualize segment-distribution -d train.pkl -c config.yaml
        
        tempest visualize segment-distribution -d train.pkl --bins 100
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    console.print("[bold blue]Analyzing Segment Distributions[/bold blue]")
    
    # Load config for segment definitions
    if config:
        cfg = load_config(config)
        segment_names = cfg.simulation.sequence_order
    else:
        segment_names = ["p7", "i7", "RP2", "UMI", "ACC", "cDNA", "polyA", "CBC", "RP1", "i5", "p5"]
    
    # Load data
    console.print("Loading data...")
    data_obj = load_data(data)
    
    # Analyze segment lengths
    segment_lengths = {seg: [] for seg in segment_names}
    
    if isinstance(data_obj, list):
        for item in data_obj:
            labels = item.get('labels', [])
            if isinstance(labels, str):
                labels = labels.split()
            
            # Count consecutive segments
            current_seg = None
            current_count = 0
            
            for label in labels:
                if label == current_seg:
                    current_count += 1
                else:
                    if current_seg and current_seg in segment_lengths:
                        segment_lengths[current_seg].append(current_count)
                    current_seg = label
                    current_count = 1
            
            # Add final segment
            if current_seg and current_seg in segment_lengths:
                segment_lengths[current_seg].append(current_count)
    
    # Filter out segments with no data
    segment_lengths = {k: v for k, v in segment_lengths.items() if len(v) > 0}
    
    if not segment_lengths:
        console.print("[red]No segment length data found[/red]")
        raise typer.Exit(1)
    
    # Create visualization
    n_segments = len(segment_lengths)
    ncols = 3
    nrows = (n_segments + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = axes.flatten() if n_segments > 1 else [axes]
    
    for idx, (segment_name, lengths) in enumerate(segment_lengths.items()):
        axes[idx].hist(lengths, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
        axes[idx].set_title(f'{segment_name} Length Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Length (bp)', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        median_len = np.median(lengths)
        
        axes[idx].axvline(mean_len, color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {mean_len:.1f}')
        axes[idx].axvline(median_len, color='green', linestyle='-.', linewidth=2,
                         label=f'Median: {median_len:.1f}')
        
        # Add text with stats
        stats_text = f'n={len(lengths)}\nmean={mean_len:.1f}\nstd={std_len:.1f}'
        axes[idx].text(0.98, 0.98, stats_text, transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=9)
        
        axes[idx].legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_segments, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Segment Length Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output or Path("segment_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Segment distribution plot saved to: {output_path}[/green]")


@visualize_app.command(name="confusion-matrix")
def confusion_matrix(
    predictions: Path = typer.Option(
        ...,
        "--predictions", "-p",
        help="Path to predictions file (pickle or text)",
        exists=True
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file for label names"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for plot (default: confusion_matrix.png)"
    ),
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="Normalize confusion matrix"
    )
):
    """
    Generate confusion matrix from predictions.
    
    Examples:
    
        tempest visualize confusion-matrix -p predictions.pkl -c config.yaml
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix as compute_cm
    
    console.print("[bold blue]Generating Confusion Matrix[/bold blue]")
    
    # Load configuration
    if config:
        cfg = load_config(config)
        label_names = cfg.simulation.sequence_order
    else:
        label_names = ["p7", "i7", "RP2", "UMI", "ACC", "cDNA", "polyA", "CBC", "RP1", "i5", "p5"]
    
    # Load predictions
    pred_data = load_data(predictions)
    
    # Extract true and predicted labels
    if isinstance(pred_data, dict):
        y_true = pred_data.get('true_labels', [])
        y_pred = pred_data.get('predicted_labels', [])
    else:
        console.print("[red]Predictions file format not recognized[/red]")
        raise typer.Exit(1)
    
    # Flatten if needed
    if isinstance(y_true[0], list):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    
    # Compute confusion matrix
    cm = compute_cm(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=label_names,
           yticklabels=label_names,
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''),
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    output_path = output or Path("confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Confusion matrix saved to: {output_path}[/green]")
