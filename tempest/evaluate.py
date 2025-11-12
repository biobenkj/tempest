"""
Tempest Evaluate Command - Model evaluation and comparison CLI.

This module provides the 'evaluate' subcommand for assessing model performance,
comparing multiple models, and generating evaluation reports.
"""

import typer
from pathlib import Path
from typing import Optional, List
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()
evaluate_app = typer.Typer(help="Evaluate trained Tempest models")


@app.command("single")
def evaluate_single_model(
    model: Path = typer.Argument(..., help="Path to trained model file"),
    test_data: Path = typer.Argument(..., help="Path to test data (pickle, npz, or directory)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file (optional)"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for evaluation"),
    output_dir: Path = typer.Option("./evaluation_results", "--output", "-o", help="Output directory"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", "-m", help="Specific metrics to compute"),
    per_segment: bool = typer.Option(False, "--per-segment", help="Compute per-segment metrics"),
    format: str = typer.Option("auto", "--format", "-f", help="Input data format (auto, pickle, npz, fastq)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output")
):
    """
    Evaluate a single trained model on test data.
    
    Examples:
        tempest evaluate single model.h5 test_data.pkl
        tempest evaluate single model.h5 test_data.pkl --per-segment --output results/
        tempest evaluate single model.h5 test_fastq/ --format fastq --config config.yaml
    """
    from tempest.compare.evaluator import ModelEvaluator
    
    if not quiet:
        console.print("[bold green]Evaluating single model...[/bold green]")
    
    # Initialize evaluator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet
    ) as progress:
        task = progress.add_task("Loading model and configuration...", total=None)
        
        try:
            evaluator = ModelEvaluator(
                config_path=str(config) if config else None,
                model_path=str(model)
            )
            
            # Load test data
            progress.update(task, description="Loading test data...")
            test_dataset = evaluator.load_test_data(str(test_data), format=format)
            
            # Run evaluation
            progress.update(task, description="Running evaluation...")
            results = evaluator.evaluate(
                test_dataset,
                batch_size=batch_size,
                metrics=metrics,
                per_segment=per_segment
            )
            
            # Save results
            progress.update(task, description="Saving results...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_file = output_dir / "evaluation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Display results
            if not quiet:
                _display_single_model_results(results)
            
            console.print(f"\n[green]✓[/green] Results saved to: {output_dir}")
            
        except Exception as e:
            console.print(f"[red]✗ Evaluation failed:[/red] {str(e)}")
            raise typer.Exit(1)


@app.command("compare")
def compare_models(
    models_dir: Path = typer.Argument(..., help="Directory containing models to compare"),
    test_data: Path = typer.Argument(..., help="Path to test data"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    output_dir: Path = typer.Option("./comparison_results", "--output", "-o", help="Output directory"),
    format: str = typer.Option("auto", "--format", "-f", help="Input data format"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output")
):
    """
    Compare multiple models on the same test dataset.
    
    Expects models_dir to contain:
    - standard_model.h5
    - soft_constraint_model.h5
    - hard_constraint_model.h5
    - hybrid_model.h5
    - ensemble/ (directory with ensemble models)
    
    Examples:
        tempest evaluate compare models/ test_data.pkl
        tempest evaluate compare models/ test_data.pkl --config config.yaml
    """
    from tempest.compare.evaluator import compare_models as run_comparison
    
    if not quiet:
        console.print("[bold green]Comparing models...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet
    ) as progress:
        task = progress.add_task("Running model comparison...", total=None)
        
        try:
            # Run comparison
            framework = run_comparison(
                models_dir=str(models_dir),
                test_data_path=str(test_data),
                config_path=str(config) if config else None,
                output_dir=str(output_dir)
            )
            
            # Display comparison results
            if not quiet:
                _display_comparison_results(framework)
            
            console.print(f"\n[green]✓[/green] Comparison complete. Results saved to: {output_dir}")
            
        except Exception as e:
            console.print(f"[red]✗ Comparison failed:[/red] {str(e)}")
            raise typer.Exit(1)


@app.command("batch")
def batch_predict(
    model: Path = typer.Argument(..., help="Path to trained model"),
    sequences: Path = typer.Argument(..., help="Path to sequences file (FASTA/FASTQ/pickle)"),
    output: Path = typer.Option("predictions.json", "--output", "-o", help="Output file"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    format: str = typer.Option("auto", "--format", "-f", help="Input format"),
    confidence: bool = typer.Option(False, "--confidence", help="Include confidence scores"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output")
):
    """
    Run batch prediction on sequences using a trained model.
    
    Examples:
        tempest evaluate batch model.h5 sequences.fasta -o predictions.json
        tempest evaluate batch model.h5 sequences.pkl --confidence
    """
    from tempest.compare.evaluator import ModelEvaluator
    from tempest.main import load_data
    
    if not quiet:
        console.print("[bold green]Running batch predictions...[/bold green]")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            config_path=str(config) if config else None,
            model_path=str(model)
        )
        
        # Load sequences
        data = load_data(str(sequences), format=format)
        
        # Extract sequences
        if isinstance(data, dict) and 'X_test' in data:
            sequences_array = data['X_test']
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            sequences_array = np.array([x['sequence'] for x in data])
        else:
            sequences_array = np.array(data)
        
        # Run predictions
        with console.status("Predicting...") if not quiet else nullcontext():
            predictions = evaluator.predict_batch(sequences_array, batch_size=batch_size)
        
        # Format output
        output_data = {
            'model': str(model),
            'num_sequences': len(sequences_array),
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        }
        
        if confidence:
            # Add confidence scores if available
            output_data['confidence'] = "Not implemented yet"  # TODO: Add confidence extraction
        
        # Save predictions
        output = Path(output)
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓[/green] Predictions saved to: {output}")
        
    except Exception as e:
        console.print(f"[red]✗ Batch prediction failed:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("metrics")
def show_metrics(
    results_file: Path = typer.Argument(..., help="Path to evaluation results JSON file"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Specific metric to display"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, csv)")
):
    """
    Display metrics from a previous evaluation.
    
    Examples:
        tempest evaluate metrics evaluation_results/metrics.json
        tempest evaluate metrics results.json --metric accuracy
        tempest evaluate metrics results.json --format csv
    """
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        if metric:
            if metric in results:
                console.print(f"{metric}: {results[metric]}")
            else:
                console.print(f"[red]Metric '{metric}' not found[/red]")
                console.print(f"Available metrics: {', '.join(results.keys())}")
        else:
            if format == "table":
                _display_metrics_table(results)
            elif format == "json":
                console.print_json(json.dumps(results, indent=2, default=str))
            elif format == "csv":
                import csv
                import sys
                writer = csv.writer(sys.stdout)
                writer.writerow(["Metric", "Value"])
                for k, v in results.items():
                    writer.writerow([k, v])
    
    except Exception as e:
        console.print(f"[red]Failed to load metrics:[/red] {str(e)}")
        raise typer.Exit(1)


# Helper functions for displaying results

def _display_single_model_results(results: dict):
    """Display evaluation results for a single model."""
    console.print("\n[bold]Evaluation Results:[/bold]")
    
    # Main metrics table
    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add main metrics
    for key in ['accuracy', 'f1_score', 'precision', 'recall']:
        if key in results:
            value = results[key]
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)
    
    # Per-segment metrics if available
    if 'per_segment' in results:
        seg_table = Table(title="Per-Segment Accuracy")
        seg_table.add_column("Segment", style="cyan")
        seg_table.add_column("Accuracy", style="green")
        
        for segment, acc in results['per_segment'].items():
            seg_table.add_row(segment, f"{acc:.4f}")
        
        console.print(seg_table)


def _display_comparison_results(framework):
    """Display model comparison results."""
    console.print("\n[bold]Model Comparison Results:[/bold]")
    
    # Get summary from framework
    if hasattr(framework, 'results_df'):
        import pandas as pd
        df = framework.results_df
        
        # Create comparison table
        table = Table(title="Model Performance Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("F1 Score", style="green")
        table.add_column("Precision", style="yellow")
        table.add_column("Recall", style="yellow")
        
        for _, row in df.iterrows():
            table.add_row(
                row.get('model', 'Unknown'),
                f"{row.get('accuracy', 0):.4f}",
                f"{row.get('f1_score', 0):.4f}",
                f"{row.get('precision', 0):.4f}",
                f"{row.get('recall', 0):.4f}"
            )
        
        console.print(table)


def _display_metrics_table(metrics: dict):
    """Display metrics in a formatted table."""
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table.add_row(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
        elif isinstance(value, dict):
            # Nested metrics
            for sub_key, sub_value in value.items():
                formatted_key = f"{key.replace('_', ' ').title()} - {sub_key}"
                if isinstance(sub_value, float):
                    table.add_row(formatted_key, f"{sub_value:.4f}")
                else:
                    table.add_row(formatted_key, str(sub_value))
    
    console.print(table)


# For importing nullcontext if Python < 3.7
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext():
        yield


if __name__ == "__main__":
    evaluate_app()
