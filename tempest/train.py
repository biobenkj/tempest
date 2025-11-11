"""
Tempest training subcommand using Typer.

This module provides the training CLI interface that integrates with the 
Typer-based CLI structure and supports various input formats including pickle.
"""

import typer
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import config and data loading from main
from tempest.main import main as tempest_main
from tempest.main import load_config, load_data
from tempest.config import TempestConfig

# Create the train sub-application
train_app = typer.Typer(
    help="Train Tempest models with various approaches",
    rich_markup_mode="rich"
)

logger = logging.getLogger(__name__)
console = Console()


@train_app.command("standard")
def standard_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    train_data: Optional[Path] = typer.Option(
        None,
        "--train-data",
        help="Path to training data (pickle or text format)"
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val-data",
        help="Path to validation data (pickle or text format)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for trained models"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of training epochs (overrides config)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Training batch size (overrides config)"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate", "--lr",
        help="Learning rate (overrides config)"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None,
        "--checkpoint-every",
        help="Save checkpoint every N epochs"
    ),
    early_stopping: bool = typer.Option(
        True,
        "--early-stopping/--no-early-stopping",
        help="Enable/disable early stopping"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Early stopping patience (epochs)"
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed training progress"
    )
):
    """
    Train a standard Tempest model.
    
    The standard training approach uses a CRF-based architecture for
    sequence annotation without explicit length constraints.
    
    [bold cyan]Examples:[/bold cyan]
    
    Train with default settings using pickle data:
    ```
    tempest train standard --config config.yaml --train-data train.pkl.gz
    ```
    
    Train with custom hyperparameters:
    ```
    tempest train standard --config config.yaml --train-data train.pkl.gz --epochs 100 --batch-size 64
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    console.print("[yellow]Training Mode:[/yellow] Standard CRF")
    
    # Load configuration using TempestConfig
    config_obj = load_config(config)
    
    # Override config with CLI arguments if provided
    if epochs is not None and config_obj.training:
        config_obj.training.epochs = epochs
    if batch_size is not None:
        config_obj.model.batch_size = batch_size
    if learning_rate is not None and config_obj.training:
        config_obj.training.learning_rate = learning_rate
    
    # Load training data if specified
    train_dataset = None
    val_dataset = None
    
    if train_data:
        console.print(f"[cyan]Loading training data:[/cyan] {train_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            train_dataset = load_data(train_data)
        console.print(f"[green][/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green][/green] Loaded {len(val_dataset)} validation sequences")
    
    # Build training arguments
    train_args = {
        'train_data': train_dataset,
        'val_data': val_dataset,
        'checkpoint_every': checkpoint_every,
        'early_stopping': early_stopping,
        'patience': patience,
        'use_gpu': use_gpu,
        'verbose': verbose,
    }

    # Dispatch via unified main entrypoint
    result = tempest_main("train", config, output=output_dir, subcommand="standard", **train_args)
    
    # Display results
    if result and 'metrics' in result:
        table = Table(title="Training Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
    
    console.print("\n[bold green] Training complete![/bold green]")
    if output_dir:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")


@train_app.command("hybrid")
def hybrid_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    train_data: Optional[Path] = typer.Option(
        None,
        "--train-data",
        help="Path to training data (pickle or text format)"
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val-data",
        help="Path to validation data (pickle or text format)"
    ),
    unlabeled_data: Optional[Path] = typer.Option(
        None,
        "--unlabeled-data",
        help="Path to unlabeled data for semi-supervised hybrid training"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for trained models"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of training epochs (overrides config)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Training batch size (overrides config)"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate", "--lr",
        help="Learning rate (overrides config)"
    ),
    constraint_weight: float = typer.Option(
        1.0,
        "--constraint-weight",
        help="Weight for length constraint loss",
        min=0.0
    ),
    constraint_type: str = typer.Option(
        "soft",
        "--constraint-type",
        help="Type of constraints: 'soft' or 'hard'"
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed training progress"
    )
):
    """
    Train a hybrid Tempest model with length constraints.
    
    The hybrid approach combines standard CRF training with
    length constraint enforcement for improved accuracy on
    structured sequences.
    
    [bold cyan]Examples:[/bold cyan]
    
    Train hybrid model with soft constraints:
    ```
    tempest train hybrid --config config.yaml --train-data train.pkl.gz
    ```
    
    Train with stronger constraint enforcement:
    ```
    tempest train hybrid --config config.yaml --train-data train.pkl.gz --constraint-weight 2.0
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    console.print("[yellow]Training Mode:[/yellow] Hybrid (with length constraints)")
    console.print(f"[yellow]Constraint Type:[/yellow] {constraint_type}")
    console.print(f"[yellow]Constraint Weight:[/yellow] {constraint_weight}")
    
    # Load configuration using TempestConfig
    config_obj = load_config(config)
    
    # Ensure hybrid block exists
    if not hasattr(config_obj, "hybrid") or config_obj.hybrid is None:
        from types import SimpleNamespace
        config_obj.hybrid = SimpleNamespace()

    # Override config with CLI arguments if provided
    if epochs is not None and config_obj.training:
        config_obj.training.epochs = epochs
    if batch_size is not None:
        config_obj.model.batch_size = batch_size
    if learning_rate is not None and config_obj.training:
        config_obj.training.learning_rate = learning_rate
    
    # Load training data if specified
    train_dataset = None
    val_dataset = None
    
    if train_data:
        console.print(f"[cyan]Loading training data:[/cyan] {train_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            train_dataset = load_data(train_data)
        console.print(f"[green][/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green][/green] Loaded {len(val_dataset)} validation sequences")
    
    # Build training arguments
    train_args = {
        'train_data': train_dataset,
        'val_data': val_dataset,
        'constraint_weight': constraint_weight,
        'constraint_type': constraint_type,
        'use_gpu': use_gpu,
        'verbose': verbose,
    }
    if unlabeled_data:
        train_args["unlabeled_path"] = unlabeled_data

    # Dispatch via unified main entrypoint
    result = tempest_main("train", config, output=output_dir, subcommand="hybrid", **train_args)
    
    # Display results
    if result and 'metrics' in result:
        table = Table(title="Training Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
    
    console.print("\n[bold green] Hybrid training complete![/bold green]")
    if output_dir:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")


@train_app.command("ensemble")
def ensemble_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    train_data: Optional[Path] = typer.Option(
        None,
        "--train-data",
        help="Path to training data (pickle or text format)"
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val-data",
        help="Path to validation data (pickle or text format)"
    ),
    unlabeled_data: Optional[Path] = typer.Option(
        None,
        "--unlabeled-data",
        help="Path to unlabeled data for hybrid models in ensemble"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for trained models"
    ),
    num_models: int = typer.Option(
        5,
        "--num-models", "-n",
        help="Number of models in ensemble",
        min=2,
        max=20
    ),
    hybrid_ratio: float = typer.Option(
        0.0,
        "--hybrid-ratio",
        help="Fraction of models that should be hybrid (0.0-1.0)",
        min=0.0,
        max=1.0
    ),
    model_types: Optional[str] = typer.Option(
        None,
        "--model-types",
        help="Comma-separated list of model types (e.g., 'standard,hybrid,standard,hybrid,standard')"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of training epochs per model (overrides config)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Training batch size (overrides config)"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate", "--lr",
        help="Learning rate (overrides config)"
    ),
    variation_type: str = typer.Option(
        "both",
        "--variation-type",
        help="Type of variation: 'architecture', 'initialization', or 'both'"
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available"
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Train models in parallel (requires multiple GPUs)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed training progress"
    )
):
    """
    Train an ensemble of Tempest models with optional mixing of standard and hybrid models.
    
    The ensemble can consist of:
    - All standard models (default)
    - All hybrid models (set --hybrid-ratio 1.0)
    - Mixed standard and hybrid (set --hybrid-ratio between 0 and 1)
    
    Hybrid models will use unlabeled data if provided via --unlabeled-data.
    
    [bold cyan]Examples:[/bold cyan]
    
    Train standard-only ensemble:
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz --num-models 5
    ```
    
    Train mixed ensemble (40% hybrid, 60% standard):
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz \\
        --num-models 5 --hybrid-ratio 0.4 --unlabeled-data unlabeled.fastq
    ```
    
    Train with explicit model types:
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz \\
        --num-models 5 --model-types "standard,hybrid,standard,hybrid,standard"
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST ENSEMBLE TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    
    # Process model types if specified as string
    model_types_list = None
    if model_types:
        model_types_list = [t.strip() for t in model_types.split(',')]
        if len(model_types_list) != num_models:
            console.print(f"[red]Error: model-types list length ({len(model_types_list)}) must match num-models ({num_models})[/red]")
            raise typer.Exit(1)
        # Validate model types
        for mt in model_types_list:
            if mt not in ['standard', 'hybrid']:
                console.print(f"[red]Error: Invalid model type '{mt}'. Must be 'standard' or 'hybrid'[/red]")
                raise typer.Exit(1)
    
    # Determine model composition
    if model_types_list:
        num_standard = model_types_list.count('standard')
        num_hybrid = model_types_list.count('hybrid')
        console.print(f"[yellow]Ensemble Composition:[/yellow] {num_standard} standard, {num_hybrid} hybrid models (explicit)")
    else:
        num_hybrid = int(num_models * hybrid_ratio)
        num_standard = num_models - num_hybrid
        console.print(f"[yellow]Ensemble Composition:[/yellow] {num_standard} standard, {num_hybrid} hybrid models (ratio={hybrid_ratio})")
    
    console.print(f"[yellow]Total Models:[/yellow] {num_models}")
    console.print(f"[yellow]Variation Type:[/yellow] {variation_type}")
    
    if unlabeled_data and num_hybrid > 0:
        console.print(f"[yellow]Unlabeled Data:[/yellow] {unlabeled_data} (for hybrid models)")
    elif num_hybrid > 0 and not unlabeled_data:
        console.print("[yellow] Warning:[/yellow] Hybrid models requested but no unlabeled data provided")
    
    # Load configuration using TempestConfig
    config_obj = load_config(config)
    
    # Override config with CLI arguments if provided
    if epochs is not None and config_obj.training:
        config_obj.training.epochs = epochs
    if batch_size is not None:
        config_obj.model.batch_size = batch_size
    if learning_rate is not None and config_obj.training:
        config_obj.training.learning_rate = learning_rate
    
    # Update ensemble config if it exists
    if config_obj.ensemble:
        config_obj.ensemble.num_models = num_models
        if not model_types_list:  # Only set hybrid_ratio if not using explicit types
            config_obj.ensemble.hybrid_ratio = hybrid_ratio
        config_obj.ensemble.variation_type = variation_type
    
    # Load training data if specified
    train_dataset = None
    val_dataset = None
    
    if train_data:
        console.print(f"\n[cyan]Loading training data:[/cyan] {train_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            train_dataset = load_data(train_data)
        console.print(f"[green][/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green][/green] Loaded {len(val_dataset)} validation sequences")
    
    # Build training arguments
    train_args = {
        'train_data': train_dataset,
        'val_data': val_dataset,
        'num_models': num_models,
        'model_types': model_types_list,
        'hybrid_ratio': hybrid_ratio,
        'variation_type': variation_type,
        'use_gpu': use_gpu,
        'parallel': parallel,
        'verbose': verbose,
    }
    if unlabeled_data:
        train_args["unlabeled_path"] = str(unlabeled_data)

    # Dispatch via unified main entrypoint
    result = tempest_main("train", config, output=output_dir, subcommand="ensemble", **train_args)
    
    # Display results
    if result and 'metrics' in result:
        table = Table(title="Ensemble Training Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print("\n", table)
        
        # Additional ensemble-specific info
        if 'model_weights' in result:
            weights_table = Table(title="Model Weights (BMA)", show_header=True, header_style="bold cyan")
            weights_table.add_column("Model", style="cyan")
            weights_table.add_column("Type", style="yellow")
            weights_table.add_column("Weight", style="green")
            
            for i, (weight, model_type) in enumerate(zip(result['model_weights'], result.get('model_types', ['unknown']*num_models))):
                weights_table.add_row(f"Model {i+1}", model_type, f"{weight:.4f}")
            
            console.print("\n", weights_table)
    
    console.print(f"\n[bold green] Ensemble training complete![/bold green]")
    console.print(f"[green]Trained {num_models} models ({num_standard} standard, {num_hybrid} hybrid)[/green]")
    if output_dir:
        console.print(f"[dim]Models saved to: {output_dir}[/dim]")


@train_app.command("resume")
def resume_command(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        help="Path to checkpoint to resume from",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration YAML file (optional, uses checkpoint config if not provided)"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Additional epochs to train"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for trained models"
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed training progress"
    )
):
    """
    Resume training from a checkpoint.
    
    [bold cyan]Examples:[/bold cyan]
    
    Resume training for 50 more epochs:
    ```
    tempest train resume --checkpoint model_checkpoint.ckpt --epochs 50
    ```
    
    Resume with modified config:
    ```
    tempest train resume --checkpoint model_checkpoint.ckpt --config new_config.yaml
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    console.print(f"[yellow]Resuming from checkpoint:[/yellow] {checkpoint}")
    
    # Load configuration if provided
    config_obj = None
    if config:
        config_obj = load_config(config)
        console.print(f"[cyan]Using configuration:[/cyan] {config}")
    
    # Build training arguments
    train_args = {
        'checkpoint_path': checkpoint,
        'additional_epochs': epochs,
        'use_gpu': use_gpu,
        'verbose': verbose
    }
    
    # Run training
    result = tempest_main("train",
                          config or checkpoint,
                          output=output_dir,
                          subcommand="resume",
                          **train_args)
    
    # Display results
    if result and 'metrics' in result:
        table = Table(title="Training Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in result['metrics'].items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
    
    console.print("\n[bold green] Training resumed and complete![/bold green]")
    if output_dir:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")
