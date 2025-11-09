"""
Tempest visualization commands using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console

from tempest.visualization import Visualizer
from tempest.utils import load_config

# Create the visualize sub-application
visualize_app = typer.Typer(help="Create visualizations for models and results")

console = Console()


@visualize_app.command()
def embeddings(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model",
        exists=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for visualization (default: embeddings.png)"
    ),
    method: str = typer.Option(
        "tsne",
        "--method",
        help="Dimensionality reduction method",
        callback=lambda v: v if v in ["tsne", "umap", "pca"] else typer.BadParameter(f"Invalid method: {v}")
    ),
    perplexity: int = typer.Option(
        30,
        "--perplexity",
        help="t-SNE perplexity parameter"
    ),
    n_components: int = typer.Option(
        2,
        "--n-components",
        help="Number of components for visualization",
        min=2,
        max=3
    )
):
    """
    Visualize learned embeddings using dimensionality reduction.
    
    Examples:
        # Visualize with t-SNE
        tempest visualize embeddings --model model.h5 --method tsne
        
        # 3D visualization with UMAP
        tempest visualize embeddings --model model.h5 --method umap --n-components 3
    """
    console.print(f"[bold blue]Visualizing Embeddings[/bold blue]")
    console.print(f"Method: {method.upper()}")
    
    visualizer = Visualizer(model_path=str(model))
    
    output_path = output or Path(f"embeddings_{method}.png")
    visualizer.plot_embeddings(
        output_path=str(output_path),
        method=method,
        n_components=n_components,
        perplexity=perplexity if method == "tsne" else None
    )
    
    console.print(f"[green]✓[/green] Embeddings visualization saved to: {output_path}")


@visualize_app.command()
def attention(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model",
        exists=True
    ),
    test_data: Path = typer.Option(
        ...,
        "--test-data", "-t",
        help="Path to test data file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for visualizations"
    ),
    num_samples: int = typer.Option(
        5,
        "--num-samples", "-n",
        help="Number of samples to visualize",
        min=1,
        max=50
    ),
    layer: Optional[int] = typer.Option(
        None,
        "--layer",
        help="Specific layer to visualize (default: last attention layer)"
    )
):
    """
    Visualize attention weights for sample sequences.
    
    Examples:
        # Visualize attention for 10 samples
        tempest visualize attention --model model.h5 --test-data test.txt -n 10
    """
    console.print(f"[bold blue]Visualizing Attention Weights[/bold blue]")
    
    visualizer = Visualizer(model_path=str(model))
    test_dataset = visualizer.load_data(str(test_data))
    
    output_path = Path(output_dir or "./attention_plots")
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(num_samples, len(test_dataset))):
        console.print(f"Processing sample {i+1}/{num_samples}...")
        
        sample_output = output_path / f"attention_sample_{i+1}.png"
        visualizer.plot_attention(
            sequence=test_dataset[i],
            output_path=str(sample_output),
            layer=layer
        )
    
    console.print(f"[green]✓[/green] Attention visualizations saved to: {output_path}")


@visualize_app.command()
def training_history(
    log_file: Path = typer.Option(
        ...,
        "--log-file", "-l",
        help="Path to training log file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for plot (default: training_history.png)"
    ),
    metrics: List[str] = typer.Option(
        ["loss", "accuracy"],
        "--metrics",
        help="Metrics to plot"
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
        # Plot training curves
        tempest visualize training-history --log-file training.log
        
        # Plot with smoothing
        tempest visualize training-history --log-file training.log --smooth 0.8
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    
    console.print(f"[bold blue]Plotting Training History[/bold blue]")
    
    # Load training history
    with open(log_file) as f:
        history = json.load(f)
    
    # Create plots
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in history:
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
                axes[idx].plot(epochs, smoothed, label=f'{metric} (smoothed)')
                axes[idx].plot(epochs, values, alpha=0.3, label=f'{metric} (raw)')
            else:
                axes[idx].plot(epochs, values, label=metric)
            
            # Add validation if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                axes[idx].plot(epochs, val_values, label=val_metric, linestyle='--')
            
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'Training {metric.capitalize()}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output or Path("training_history.png")
    plt.savefig(output_path)
    plt.close()
    
    console.print(f"[green]✓[/green] Training history plot saved to: {output_path}")


@visualize_app.command()
def predictions(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model",
        exists=True
    ),
    test_data: Path = typer.Option(
        ...,
        "--test-data", "-t",
        help="Path to test data file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for visualization"
    ),
    num_samples: int = typer.Option(
        10,
        "--num-samples", "-n",
        help="Number of samples to visualize",
        min=1,
        max=100
    ),
    show_confidence: bool = typer.Option(
        False,
        "--show-confidence",
        help="Display prediction confidence"
    )
):
    """
    Visualize model predictions on test sequences.
    
    Examples:
        # Basic prediction visualization
        tempest visualize predictions --model model.h5 --test-data test.txt
        
        # Show with confidence scores
        tempest visualize predictions --model model.h5 --test-data test.txt --show-confidence
    """
    from tempest.compare.evaluate import ModelEvaluator
    
    console.print(f"[bold blue]Visualizing Predictions[/bold blue]")
    
    evaluator = ModelEvaluator(model_path=str(model))
    test_dataset = evaluator.load_test_data(str(test_data))
    
    # Get predictions
    samples = test_dataset[:num_samples]
    predictions = evaluator.predict_batch(samples)
    
    # Create visualization
    output_path = output or Path("predictions.html")
    
    # Generate HTML visualization
    html_content = _generate_prediction_html(samples, predictions, show_confidence)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    console.print(f"[green]✓[/green] Predictions visualization saved to: {output_path}")


@visualize_app.command()
def segment_distribution(
    data: Path = typer.Option(
        ...,
        "--data", "-d",
        help="Path to data file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for plot"
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
        # Visualize segment distributions
        tempest visualize segment-distribution --data train.txt --bins 100
    """
    import matplotlib.pyplot as plt
    from tempest.data import SequenceDataset
    
    console.print(f"[bold blue]Analyzing Segment Distributions[/bold blue]")
    
    # Load data
    dataset = SequenceDataset.from_file(str(data))
    
    # Analyze segment lengths
    segment_lengths = dataset.analyze_segment_lengths()
    
    # Create subplots for each segment
    n_segments = len(segment_lengths)
    fig, axes = plt.subplots(2, (n_segments + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (segment_name, lengths) in enumerate(segment_lengths.items()):
        axes[idx].hist(lengths, bins=bins, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{segment_name} Length Distribution')
        axes[idx].set_xlabel('Length (bp)')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        axes[idx].axvline(mean_len, color='red', linestyle='--', 
                         label=f'Mean: {mean_len:.1f}±{std_len:.1f}')
        axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(n_segments, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Segment Length Distributions', fontsize=16)
    plt.tight_layout()
    
    output_path = output or Path("segment_distributions.png")
    plt.savefig(output_path)
    plt.close()
    
    console.print(f"[green]✓[/green] Segment distribution plot saved to: {output_path}")


def _generate_prediction_html(samples, predictions, show_confidence):
    """Generate HTML visualization for predictions."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tempest Predictions</title>
        <style>
            body { font-family: monospace; margin: 20px; }
            .sample { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }
            .sequence { background: #f0f0f0; padding: 10px; word-wrap: break-word; }
            .true-label { color: green; font-weight: bold; }
            .pred-label { color: blue; font-weight: bold; }
            .mismatch { background: #ffcccc; }
            .confidence { color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Tempest Model Predictions</h1>
    """
    
    for i, (sample, pred) in enumerate(zip(samples, predictions)):
        html += f"""
        <div class="sample">
            <h3>Sample {i+1}</h3>
            <div class="sequence">{sample['sequence']}</div>
            <div class="true-label">True: {sample['label']}</div>
            <div class="pred-label">Predicted: {pred['label']}</div>
        """
        
        if show_confidence and 'confidence' in pred:
            html += f'<div class="confidence">Confidence: {pred["confidence"]:.3f}</div>'
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html
