"""
Tempest model comparison commands using Typer.
"""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
import json

from tempest.compare import ModelComparator
from tempest.utils import load_config

# Create the compare sub-application
compare_app = typer.Typer(help="Compare multiple Tempest models")

console = Console()


@compare_app.command()
def models(
    model_paths: List[Path] = typer.Option(
        ...,
        "--models", "-m",
        help="Paths to models to compare (specify multiple times)",
        exists=True
    ),
    test_data: Path = typer.Option(
        ...,
        "--test-data", "-t",
        help="Path to test data",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    metrics: List[str] = typer.Option(
        ["accuracy", "f1", "precision", "recall", "loss"],
        "--metrics",
        help="Metrics to compare"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save comparison results to JSON"
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Generate comparison plots"
    ),
    statistical_test: bool = typer.Option(
        False,
        "--statistical-test",
        help="Perform statistical significance tests"
    )
):
    """
    Compare multiple models on the same test set.
    
    Examples:
        # Basic comparison
        tempest compare models --models model1.h5 --models model2.h5 --test-data test.txt
        
        # With statistical tests and plots
        tempest compare models --models m1.h5 --models m2.h5 --models m3.h5 \\
            --test-data test.txt --statistical-test --plot
    """
    if len(model_paths) < 2:
        console.print("[red]Error: Need at least 2 models to compare[/red]")
        raise typer.Exit(1)
    
    console.print("[bold blue]Tempest Model Comparison[/bold blue]")
    console.print("=" * 60)
    console.print(f"Comparing {len(model_paths)} models")
    
    # Create comparator
    comparator = ModelComparator(
        model_paths=[str(p) for p in model_paths],
        test_data=str(test_data)
    )
    
    # Run comparison
    results = comparator.compare(
        metrics=metrics,
        statistical_test=statistical_test
    )
    
    # Display results
    _display_comparison_results(results, statistical_test)
    
    # Generate plots if requested
    if plot:
        plot_path = Path("model_comparison_plots")
        plot_path.mkdir(exist_ok=True)
        comparator.generate_plots(results, output_dir=str(plot_path))
        console.print(f"\n[green]✓[/green] Plots saved to: {plot_path}")
    
    # Save results if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]✓[/green] Results saved to: {output}")


@compare_app.command()
def architectures(
    configs: List[Path] = typer.Option(
        ...,
        "--configs", "-c",
        help="Configuration files to compare (specify multiple times)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save comparison to file"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        help="Show detailed architecture differences"
    )
):
    """
    Compare model architectures from configuration files.
    
    Examples:
        # Compare two architectures
        tempest compare architectures --configs config1.yaml --configs config2.yaml
        
        # Detailed comparison
        tempest compare architectures --configs c1.yaml --configs c2.yaml --detailed
    """
    console.print("[bold blue]Architecture Comparison[/bold blue]")
    console.print("=" * 60)
    
    # Load configurations
    configurations = {}
    for config_path in configs:
        cfg = load_config(str(config_path))
        configurations[config_path.stem] = cfg
    
    # Create comparison table
    table = Table(title="Architecture Comparison")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    
    for config_name in configurations.keys():
        table.add_column(config_name, style="magenta")
    
    # Extract and compare key parameters
    params_to_compare = [
        ('model.type', 'Model Type'),
        ('model.hidden_units', 'Hidden Units'),
        ('model.num_layers', 'Layers'),
        ('model.dropout_rate', 'Dropout'),
        ('model.use_attention', 'Attention'),
        ('training.batch_size', 'Batch Size'),
        ('training.learning_rate', 'Learning Rate'),
        ('training.epochs', 'Epochs'),
    ]
    
    for param_path, param_name in params_to_compare:
        row = [param_name]
        for config_name, cfg in configurations.items():
            # Navigate through nested config
            value = cfg
            for key in param_path.split('.'):
                value = getattr(value, key, 'N/A') if hasattr(value, key) else 'N/A'
            row.append(str(value))
        table.add_row(*row)
    
    console.print(table)
    
    if detailed:
        console.print("\n[bold]Detailed Differences:[/bold]")
        _show_detailed_differences(configurations)
    
    # Save if requested
    if output:
        comparison_data = _extract_comparison_data(configurations)
        with open(output, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        console.print(f"\n[green]✓[/green] Comparison saved to: {output}")


@compare_app.command()
def checkpoints(
    checkpoint_dir: Path = typer.Option(
        ...,
        "--checkpoint-dir", "-d",
        help="Directory containing checkpoints",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    metric: str = typer.Option(
        "val_loss",
        "--metric",
        help="Metric to compare"
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="Show top K checkpoints",
        min=1,
        max=20
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Plot checkpoint progression"
    )
):
    """
    Compare checkpoints from training to find the best model.
    
    Examples:
        # Find best checkpoint by validation loss
        tempest compare checkpoints --checkpoint-dir ./checkpoints --metric val_loss
        
        # Show top 10 by accuracy with plot
        tempest compare checkpoints --checkpoint-dir ./checkpoints \\
            --metric val_accuracy --top-k 10 --plot
    """
    import glob
    
    console.print("[bold blue]Checkpoint Comparison[/bold blue]")
    console.print("=" * 60)
    
    # Find all checkpoint files
    checkpoint_files = list(Path(checkpoint_dir).glob("*.ckpt"))
    if not checkpoint_files:
        console.print("[red]No checkpoint files found![/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Load checkpoint metrics
    checkpoint_metrics = []
    for ckpt_file in checkpoint_files:
        # Try to load associated metrics file
        metrics_file = ckpt_file.with_suffix('.json')
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                checkpoint_metrics.append({
                    'file': ckpt_file.name,
                    'epoch': metrics.get('epoch', 0),
                    metric: metrics.get(metric, float('inf'))
                })
    
    # Sort by metric (ascending for loss, descending for accuracy/f1)
    reverse = 'loss' not in metric.lower()
    checkpoint_metrics.sort(key=lambda x: x[metric], reverse=reverse)
    
    # Display top K
    table = Table(title=f"Top {top_k} Checkpoints by {metric}")
    table.add_column("Rank", style="cyan")
    table.add_column("Checkpoint", style="magenta")
    table.add_column("Epoch", style="yellow")
    table.add_column(metric, style="green")
    
    for i, ckpt in enumerate(checkpoint_metrics[:top_k], 1):
        table.add_row(
            str(i),
            ckpt['file'],
            str(ckpt['epoch']),
            f"{ckpt[metric]:.4f}"
        )
    
    console.print(table)
    
    # Best checkpoint
    best = checkpoint_metrics[0]
    console.print(f"\n[bold green]Best checkpoint:[/bold green] {best['file']}")
    console.print(f"Epoch {best['epoch']}, {metric}: {best[metric]:.4f}")
    
    if plot:
        _plot_checkpoint_progression(checkpoint_metrics, metric)
        console.print("\n[green]✓[/green] Checkpoint progression plot saved")


def _display_comparison_results(results: dict, statistical_test: bool):
    """Display model comparison results."""
    # Main comparison table
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    
    # Add columns for each metric
    metrics = list(next(iter(results['models'].values())).keys())
    for metric in metrics:
        table.add_column(metric.capitalize(), style="magenta")
    
    # Add rows for each model
    for model_name, model_metrics in results['models'].items():
        row = [model_name]
        for metric in metrics:
            value = model_metrics.get(metric, 0)
            row.append(f"{value:.4f}")
        table.add_row(*row)
    
    # Add best model row
    if 'best_model' in results:
        table.add_row("", "", "", "", style="dim")
        best_row = [f"[bold green]Best: {results['best_model']}[/bold green]"]
        for metric in metrics:
            best_row.append("")
        table.add_row(*best_row)
    
    console.print(table)
    
    # Statistical test results if available
    if statistical_test and 'statistical_tests' in results:
        console.print("\n[bold]Statistical Significance Tests:[/bold]")
        stats_table = Table()
        stats_table.add_column("Comparison", style="cyan")
        stats_table.add_column("p-value", style="magenta")
        stats_table.add_column("Significant", style="yellow")
        
        for test in results['statistical_tests']:
            stats_table.add_row(
                test['comparison'],
                f"{test['p_value']:.4f}",
                "Yes" if test['significant'] else "No"
            )
        
        console.print(stats_table)


def _show_detailed_differences(configurations: dict):
    """Show detailed differences between configurations."""
    # Find all unique parameters
    all_params = set()
    for cfg in configurations.values():
        all_params.update(_flatten_config(cfg))
    
    # Show differences
    for param in sorted(all_params):
        values = []
        for cfg_name, cfg in configurations.items():
            flat_cfg = _flatten_config(cfg)
            values.append(flat_cfg.get(param, 'N/A'))
        
        # Only show if there are differences
        if len(set(values)) > 1:
            console.print(f"\n[yellow]{param}:[/yellow]")
            for cfg_name, value in zip(configurations.keys(), values):
                console.print(f"  {cfg_name}: {value}")


def _flatten_config(cfg, prefix=''):
    """Flatten nested configuration to dot-notation keys."""
    flat = {}
    for key, value in cfg.__dict__.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if hasattr(value, '__dict__') and not isinstance(value, (list, tuple, str)):
            flat.update(_flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _extract_comparison_data(configurations: dict) -> dict:
    """Extract comparison data for saving."""
    comparison = {}
    for cfg_name, cfg in configurations.items():
        comparison[cfg_name] = _flatten_config(cfg)
    return comparison


def _plot_checkpoint_progression(checkpoint_metrics: list, metric: str):
    """Plot checkpoint progression over epochs."""
    import matplotlib.pyplot as plt
    
    # Sort by epoch
    checkpoint_metrics.sort(key=lambda x: x['epoch'])
    
    epochs = [ckpt['epoch'] for ckpt in checkpoint_metrics]
    values = [ckpt[metric] for ckpt in checkpoint_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} Progression Over Training')
    plt.grid(True, alpha=0.3)
    
    # Mark best checkpoint
    best_idx = values.index(min(values)) if 'loss' in metric.lower() else values.index(max(values))
    plt.scatter(epochs[best_idx], values[best_idx], color='red', s=100, zorder=5)
    plt.annotate('Best', xy=(epochs[best_idx], values[best_idx]), 
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('checkpoint_progression.png')
    plt.close()
