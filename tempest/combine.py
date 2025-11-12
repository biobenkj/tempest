"""
Tempest model combination/ensemble commands using Typer.

Integrates with tempest.inference.combiner.ModelCombiner for BMA and ensemble methods.
"""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import pickle
import numpy as np

from tempest.inference.combiner import ModelCombiner
from tempest.config import load_config, EnsembleConfig

# Create the combine sub-application
combine_app = typer.Typer(help="Combine models using BMA/ensemble methods")

console = Console()


@combine_app.command(name="combine")
def combine_models(
    model_paths: List[Path] = typer.Option(
        ...,
        "--models", "-m",
        help="Paths to models to combine (can specify multiple times)",
        exists=True
    ),
    validation_data: Path = typer.Option(
        ...,
        "--validation-data", "-v",
        help="Validation data for computing weights (.pkl or .txt)",
        exists=True
    ),
    output_dir: Path = typer.Option(
        "./ensemble_results",
        "--output", "-o",
        help="Output directory for ensemble results"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file with ensemble settings"
    ),
    method: str = typer.Option(
        "bayesian_model_averaging",
        "--method",
        help="Combination method: bayesian_model_averaging, weighted_average, voting, stacking"
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--test-data", "-t",
        help="Optional test data for evaluation"
    ),
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        help="Enable prediction calibration"
    )
):
    """
    Combine multiple trained models into an ensemble using BMA or other methods.
    
    This command implements the full Bayesian Model Averaging workflow with
    support for multiple approximation methods (BIC, Laplace, Variational, CV).
    
    Examples:
        # BMA with default settings (BIC approximation)
        tempest combine combine --models m1.h5 --models m2.h5 --models m3.h5 \\
            --validation-data val.pkl --output results/
        
        # BMA with config file specifying Laplace approximation
        tempest combine combine --models model*.h5 \\
            --validation-data val.pkl \\
            --config config.yaml \\
            --output ensemble_output/
        
        # With calibration and test evaluation
        tempest combine combine --models m1.h5 --models m2.h5 \\
            --validation-data val.pkl \\
            --test-data test.pkl \\
            --calibrate \\
            --output results/
        
        # Simple weighted average
        tempest combine combine --models m1.h5 --models m2.h5 \\
            --validation-data val.pkl \\
            --method weighted_average \\
            --output results/
    """
    console.print("[bold blue]Tempest Model Combination[/bold blue]")
    console.print("=" * 70)
    console.print(f"Method: {method}")
    console.print(f"Number of models: {len(model_paths)}")
    console.print(f"Output directory: {output_dir}")
    
    # Load configuration
    if config:
        console.print(f"Loading configuration from: {config}")
        full_config = load_config(str(config))
        ensemble_config = full_config.ensemble
        
        # Override method if specified in CLI
        if method != "bayesian_model_averaging":
            ensemble_config.voting_method = method
    else:
        # Create minimal config
        console.print("No config provided, using defaults")
        ensemble_config = EnsembleConfig(
            voting_method=method,
            num_models=len(model_paths)
        )
    
    # Display ensemble configuration
    _display_config(ensemble_config, method)
    
    # Initialize ModelCombiner
    console.print("\n[bold]Step 1: Initializing ModelCombiner[/bold]")
    combiner = ModelCombiner(config=ensemble_config)
    
    # Load models
    console.print("[bold]Step 2: Loading models[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        combiner.load_models([str(p) for p in model_paths])
        progress.update(task, completed=True)
    
    console.print(f" Loaded {len(combiner.models)} models")
    
    # Compute weights
    console.print("[bold]Step 3: Computing ensemble weights[/bold]")
    console.print(f"Using validation data: {validation_data}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Computing weights...", total=None)
        combiner.compute_weights(str(validation_data))
        progress.update(task, completed=True)
    
    # Display computed weights
    _display_weights(combiner, method)
    
    # Calibrate if requested
    if calibrate:
        console.print("\n[bold]Step 4: Calibrating predictions[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Calibrating...", total=None)
            combiner.calibrate(str(validation_data))
            progress.update(task, completed=True)
        console.print(f" Calibration complete using {ensemble_config.calibration_method}")
    
    # Evaluate on test data if provided
    if test_data:
        console.print("\n[bold]Step 5: Evaluating on test data[/bold]")
        console.print(f"Test data: {test_data}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating...", total=None)
            metrics = combiner.evaluate(str(test_data))
            progress.update(task, completed=True)
        
        # Display metrics
        _display_metrics(metrics)
    
    # Save results
    console.print(f"\n[bold]Step {'6' if test_data else '5'}: Saving results[/bold]")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Saving ensemble...", total=None)
        combiner.save_results(str(output_dir))
        progress.update(task, completed=True)
    
    console.print(f"Results saved to: {output_dir}")
    console.print("\n[green] Ensemble combination complete![/green]")
    console.print("\nSaved files:")
    console.print(f"  - {output_dir}/ensemble_weights.json - Model weights")
    console.print(f"  - {output_dir}/ensemble_config.yaml - Configuration used")
    console.print(f"  - {output_dir}/ensemble_metadata.json - Ensemble metadata")
    if test_data:
        console.print(f"  - {output_dir}/evaluation_metrics.json - Test metrics")


@combine_app.command(name="predict")
def predict(
    ensemble_dir: Path = typer.Option(
        ...,
        "--ensemble", "-e",
        help="Directory containing ensemble results",
        exists=True,
        dir_okay=True,
        file_okay=False
    ),
    input_data: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Input data file (.pkl or .txt)",
        exists=True
    ),
    output: Path = typer.Option(
        "predictions.pkl",
        "--output", "-o",
        help="Output file for predictions"
    ),
    uncertainty: bool = typer.Option(
        True,
        "--uncertainty/--no-uncertainty",
        help="Include uncertainty estimates"
    ),
    individual: bool = typer.Option(
        False,
        "--individual",
        help="Include individual model predictions"
    )
):
    """
    Make predictions using a saved ensemble.
    
    Examples:
        # Basic prediction with uncertainty
        tempest combine predict --ensemble results/ --input new_data.pkl --output pred.pkl
        
        # Include individual model predictions
        tempest combine predict --ensemble results/ --input data.pkl \\
            --output pred.pkl --individual
    """
    console.print("[bold blue]Ensemble Prediction[/bold blue]")
    console.print("=" * 70)
    
    # Load ensemble
    console.print(f"Loading ensemble from: {ensemble_dir}")
    
    # Load config
    config_path = ensemble_dir / "ensemble_config.yaml"
    if not config_path.exists():
        console.print("[red]Error: ensemble_config.yaml not found in ensemble directory[/red]")
        raise typer.Exit(1)
    
    full_config = load_config(str(config_path))
    combiner = ModelCombiner(config=full_config.ensemble)
    
    # Load models from ensemble directory
    metadata_path = ensemble_dir / "ensemble_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        model_paths = metadata.get('model_paths', [])
        
        if not model_paths:
            console.print("[red]Error: No model paths in ensemble metadata[/red]")
            raise typer.Exit(1)
        
        combiner.load_models(model_paths)
    else:
        console.print("[red]Error: ensemble_metadata.json not found[/red]")
        raise typer.Exit(1)
    
    # Load weights
    weights_path = ensemble_dir / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            weights_data = json.load(f)
        combiner.posterior_weights = weights_data.get('posterior_weights', {})
    
    # Load calibrator if exists
    calibrator_path = ensemble_dir / "calibrator.pkl"
    if calibrator_path.exists():
        with open(calibrator_path, 'rb') as f:
            calib_data = pickle.load(f)
        combiner.calibrator = calib_data.get('calibrator')
    
    # Load input data
    console.print(f"Loading input data from: {input_data}")
    
    if input_data.suffix == '.pkl':
        with open(input_data, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            X_input = data[0]
        else:
            X_input = data
    else:
        # Handle text format
        console.print("[red]Error: Text format not yet implemented, use .pkl[/red]")
        raise typer.Exit(1)
    
    # Make predictions
    console.print("Making predictions...")
    
    result = combiner.predict(
        X_input,
        return_uncertainty=uncertainty,
        return_individual=individual,
        apply_calibration=(calibrator_path.exists())
    )
    
    # Save predictions
    console.print(f"Saving predictions to: {output}")
    with open(output, 'wb') as f:
        pickle.dump(result, f)
    
    # Display summary
    console.print("\n[green] Predictions complete![/green]")
    console.print(f"Predictions shape: {result['predictions'].shape}")
    
    if uncertainty:
        console.print("\nUncertainty estimates:")
        unc = result['uncertainty']
        console.print(f"  Mean entropy: {np.mean(unc['entropy']):.4f}")
        console.print(f"  Mean epistemic: {np.mean(unc['epistemic_uncertainty']):.4f}")
        console.print(f"  Mean aleatoric: {np.mean(unc['aleatoric_uncertainty']):.4f}")


@combine_app.command(name="info")
def info(
    ensemble_dir: Path = typer.Argument(
        ...,
        help="Directory containing ensemble results",
        exists=True
    )
):
    """
    Display information about a saved ensemble.
    
    Example:
        tempest combine info results/
    """
    console.print("[bold blue]Ensemble Information[/bold blue]")
    console.print("=" * 70)
    
    # Load metadata
    metadata_path = ensemble_dir / "ensemble_metadata.json"
    if not metadata_path.exists():
        console.print("[red]Error: ensemble_metadata.json not found[/red]")
        raise typer.Exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Display general info
    console.print("\n[bold]General Information:[/bold]")
    console.print(f"  Number of models: {metadata.get('num_models', 'Unknown')}")
    console.print(f"  Method: {metadata.get('voting_method', 'Unknown')}")
    console.print(f"  Created: {metadata.get('created_at', 'Unknown')}")
    
    # Display model paths
    console.print("\n[bold]Models:[/bold]")
    model_paths = metadata.get('model_paths', [])
    for i, path in enumerate(model_paths, 1):
        console.print(f"  {i}. {Path(path).name}")
    
    # Load and display weights
    weights_path = ensemble_dir / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            weights_data = json.load(f)
        
        console.print("\n[bold]Ensemble Weights:[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Weight", justify="right", style="green")
        table.add_column("Evidence", justify="right", style="yellow")
        
        weights = weights_data.get('posterior_weights', {})
        evidences = weights_data.get('model_evidences', {})
        
        for name in weights:
            weight = weights[name]
            evidence = evidences.get(name, 'N/A')
            ev_str = f"{evidence:.4f}" if isinstance(evidence, float) else str(evidence)
            table.add_row(name, f"{weight:.6f}", ev_str)
        
        console.print(table)
        
        # BMA specific info
        if 'bma_config' in weights_data:
            bma_config = weights_data['bma_config']
            console.print("\n[bold]BMA Configuration:[/bold]")
            console.print(f"  Approximation: {bma_config.get('approximation', 'Unknown')}")
            console.print(f"  Temperature: {bma_config.get('temperature', 'Unknown')}")
            console.print(f"  Prior type: {bma_config.get('prior_type', 'Unknown')}")
    
    # Check for evaluation metrics
    metrics_path = ensemble_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        console.print("\n[bold]Evaluation Metrics:[/bold]")
        for key, value in metrics.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.4f}")
            else:
                console.print(f"  {key}: {value}")
    
    # Check for calibrator
    calibrator_path = ensemble_dir / "calibrator.pkl"
    if calibrator_path.exists():
        console.print("\n[green] Calibration enabled[/green]")
    else:
        console.print("\n[yellow]✗ No calibration[/yellow]")


def _display_config(config: EnsembleConfig, method: str):
    """Display ensemble configuration."""
    console.print("\n[bold]Ensemble Configuration:[/bold]")
    console.print(f"  Voting method: {config.voting_method}")
    
    if method == "bayesian_model_averaging" and config.bma_config:
        console.print(f"  BMA approximation: {config.bma_config.approximation}")
        console.print(f"  BMA temperature: {config.bma_config.temperature}")
        console.print(f"  Min posterior weight: {config.bma_config.min_posterior_weight}")
    
    if config.calibration_enabled:
        console.print(f"  Calibration: {config.calibration_method}")
    
    if config.compute_epistemic or config.compute_aleatoric:
        unc_types = []
        if config.compute_epistemic:
            unc_types.append("epistemic")
        if config.compute_aleatoric:
            unc_types.append("aleatoric")
        console.print(f"  Uncertainty: {', '.join(unc_types)}")


def _display_weights(combiner: ModelCombiner, method: str):
    """Display computed weights."""
    console.print("\n[bold]Computed Weights:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", width=30)
    table.add_column("Weight", justify="right", style="green")
    
    if method == "bayesian_model_averaging":
        table.add_column("Log Evidence", justify="right", style="yellow")
        
        for name in sorted(combiner.posterior_weights.keys()):
            weight = combiner.posterior_weights[name]
            evidence = combiner.model_evidences.get(name, 0.0)
            table.add_row(name, f"{weight:.6f}", f"{evidence:.4f}")
    else:
        for name in sorted(combiner.model_weights.keys()):
            weight = combiner.model_weights[name]
            table.add_row(name, f"{weight:.6f}")
    
    console.print(table)


def _display_metrics(metrics: dict):
    """Display evaluation metrics."""
    console.print("\n[bold]Evaluation Results:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    # Display main metrics
    main_metrics = [
        'ensemble_accuracy', 'mean_entropy', 
        'mean_epistemic', 'mean_aleatoric'
    ]
    
    for metric in main_metrics:
        if metric in metrics:
            value = metrics[metric]
            table.add_row(metric, f"{value:.4f}")
    
    console.print(table)
    
    # Display individual model accuracies if present
    individual = {k: v for k, v in metrics.items() if k.endswith('_accuracy') and k != 'ensemble_accuracy'}
    if individual:
        console.print("\n[bold]Individual Model Accuracies:[/bold]")
        for model_name, acc in individual.items():
            console.print(f"  {model_name}: {acc:.4f}")

if __name__ == "__main__":
    combine_app()