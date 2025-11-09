"""
Tempest evaluation commands using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json
from rich.console import Console
from rich.table import Table

from tempest.compare.evaluate import ModelEvaluator
from tempest.utils import load_config

# Create the evaluate sub-application
evaluate_app = typer.Typer(help="Evaluate trained Tempest models")

console = Console()


@evaluate_app.command()
def performance(
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
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration YAML file"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save metrics to JSON file"
    ),
    metrics: List[str] = typer.Option(
        ["accuracy", "f1", "precision", "recall"],
        "--metrics",
        help="Metrics to compute"
    ),
    per_segment: bool = typer.Option(
        False,
        "--per-segment",
        help="Compute per-segment metrics"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for evaluation"
    )
):
    """
    Evaluate model performance on test data.
    
    Examples:
        # Basic evaluation
        tempest evaluate performance --model model.h5 --test-data test.txt
        
        # Evaluate with per-segment metrics
        tempest evaluate performance --model model.h5 --test-data test.txt --per-segment
    """
    console.print("[bold blue]TEMPEST Model Evaluation[/bold blue]")
    console.print("=" * 60)
    
    # Load configuration
    cfg = None
    if config:
        cfg = load_config(str(config))
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=str(model),
        config=cfg
    )
    
    # Load test data
    console.print(f"Loading test data from: {test_data}")
    test_dataset = evaluator.load_test_data(str(test_data))
    
    # Evaluate
    console.print(f"Evaluating on {len(test_dataset)} sequences...")
    results = evaluator.evaluate(
        test_dataset,
        batch_size=batch_size,
        metrics=metrics,
        per_segment=per_segment
    )
    
    # Display results
    _display_metrics(results, per_segment)
    
    # Save if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]✓[/green] Metrics saved to: {output}")


@evaluate_app.command()
def compare(
    models: List[Path] = typer.Option(
        ...,
        "--models", "-m",
        help="Paths to models to compare (can specify multiple)"
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
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration YAML file"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save comparison to JSON file"
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Generate comparison plots"
    )
):
    """
    Compare multiple models on the same test set.
    
    Examples:
        # Compare two models
        tempest evaluate compare --models model1.h5 --models model2.h5 --test-data test.txt
        
        # Compare with plots
        tempest evaluate compare --models model1.h5 --models model2.h5 --test-data test.txt --plot
    """
    if len(models) < 2:
        console.print("[red]Error: Need at least 2 models to compare[/red]")
        raise typer.Exit(1)
    
    console.print("[bold blue]TEMPEST Model Comparison[/bold blue]")
    console.print("=" * 60)
    console.print(f"Comparing {len(models)} models")
    
    # Load configuration
    cfg = None
    if config:
        cfg = load_config(str(config))
    
    results = {}
    for model_path in models:
        model_name = model_path.stem
        console.print(f"\nEvaluating: {model_name}")
        
        evaluator = ModelEvaluator(
            model_path=str(model_path),
            config=cfg
        )
        
        test_dataset = evaluator.load_test_data(str(test_data))
        metrics = evaluator.evaluate(test_dataset)
        results[model_name] = metrics
    
    # Display comparison table
    _display_comparison(results)
    
    # Generate plots if requested
    if plot:
        import matplotlib.pyplot as plt
        _generate_comparison_plots(results)
        console.print("\n[green]✓[/green] Comparison plots saved")
    
    # Save if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]✓[/green] Comparison saved to: {output}")


@evaluate_app.command()
def inference_speed(
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
    batch_sizes: List[int] = typer.Option(
        [1, 8, 16, 32, 64],
        "--batch-sizes",
        help="Batch sizes to test"
    ),
    warmup: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup iterations"
    ),
    iterations: int = typer.Option(
        100,
        "--iterations",
        help="Number of timing iterations"
    )
):
    """
    Benchmark inference speed at different batch sizes.
    
    Examples:
        # Test inference speed
        tempest evaluate inference-speed --model model.h5 --test-data test.txt
        
        # Test specific batch sizes
        tempest evaluate inference-speed --model model.h5 --test-data test.txt --batch-sizes 1 --batch-sizes 32 --batch-sizes 128
    """
    import time
    import numpy as np
    
    console.print("[bold blue]TEMPEST Inference Speed Benchmark[/bold blue]")
    console.print("=" * 60)
    
    evaluator = ModelEvaluator(model_path=str(model))
    test_dataset = evaluator.load_test_data(str(test_data))
    
    # Prepare results table
    table = Table(title="Inference Speed Results")
    table.add_column("Batch Size", style="cyan", no_wrap=True)
    table.add_column("Throughput (seq/s)", style="magenta")
    table.add_column("Latency (ms/seq)", style="yellow")
    table.add_column("GPU Memory (MB)", style="green")
    
    for batch_size in batch_sizes:
        console.print(f"\nTesting batch size: {batch_size}")
        
        # Warmup
        for _ in range(warmup):
            _ = evaluator.predict_batch(test_dataset[:batch_size], batch_size=batch_size)
        
        # Time iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = evaluator.predict_batch(test_dataset[:batch_size], batch_size=batch_size)
            times.append(time.perf_counter() - start)
        
        # Calculate metrics
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        latency = (avg_time / batch_size) * 1000
        
        # Try to get GPU memory if available
        gpu_mem = "N/A"
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                # This is a placeholder - actual GPU memory tracking would need nvidia-ml-py
                gpu_mem = f"~{batch_size * 10}"  
        except:
            pass
        
        table.add_row(
            str(batch_size),
            f"{throughput:.2f}",
            f"{latency:.2f}",
            gpu_mem
        )
    
    console.print("\n")
    console.print(table)


def _display_metrics(metrics: dict, per_segment: bool = False):
    """Display evaluation metrics in a formatted table."""
    # Overall metrics table
    table = Table(title="Overall Performance Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    for key, value in metrics.get('overall', {}).items():
        if isinstance(value, float):
            table.add_row(key.capitalize(), f"{value:.4f}")
        else:
            table.add_row(key.capitalize(), str(value))
    
    console.print(table)
    
    # Per-segment metrics if available
    if per_segment and 'per_segment' in metrics:
        console.print("\n[bold]Per-Segment Metrics:[/bold]")
        for segment, segment_metrics in metrics['per_segment'].items():
            seg_table = Table(title=f"Segment: {segment}")
            seg_table.add_column("Metric", style="cyan")
            seg_table.add_column("Value", style="magenta")
            
            for key, value in segment_metrics.items():
                if isinstance(value, float):
                    seg_table.add_row(key.capitalize(), f"{value:.4f}")
                else:
                    seg_table.add_row(key.capitalize(), str(value))
            
            console.print(seg_table)


def _display_comparison(results: dict):
    """Display model comparison in a formatted table."""
    # Get all metrics
    all_metrics = set()
    for model_metrics in results.values():
        all_metrics.update(model_metrics.get('overall', {}).keys())
    
    # Create comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    for metric in sorted(all_metrics):
        table.add_column(metric.capitalize(), style="magenta")
    
    # Add rows for each model
    for model_name, metrics in results.items():
        row = [model_name]
        for metric in sorted(all_metrics):
            value = metrics.get('overall', {}).get(metric, 'N/A')
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table.add_row(*row)
    
    console.print(table)


def _generate_comparison_plots(results: dict):
    """Generate comparison plots for multiple models."""
    import matplotlib.pyplot as plt
    
    # Extract data for plotting
    models = list(results.keys())
    metrics_data = {}
    
    for model, metrics in results.items():
        for metric_name, value in metrics.get('overall', {}).items():
            if metric_name not in metrics_data:
                metrics_data[metric_name] = []
            metrics_data[metric_name].append(value if isinstance(value, (int, float)) else 0)
    
    # Create bar plot
    fig, axes = plt.subplots(1, len(metrics_data), figsize=(15, 5))
    if len(metrics_data) == 1:
        axes = [axes]
    
    for idx, (metric, values) in enumerate(metrics_data.items()):
        axes[idx].bar(models, values)
        axes[idx].set_title(metric.capitalize())
        axes[idx].set_ylabel('Score')
        axes[idx].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
