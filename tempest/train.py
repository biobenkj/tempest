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
        console.print(f"[green]✓[/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green]✓[/green] Loaded {len(val_dataset)} validation sequences")
    
    # Import and run the actual training
    from tempest.training import run_training
    
    # Build training arguments
    train_args = {
        'mode': 'standard',
        'train_data': train_dataset,
        'val_data': val_dataset,
        'checkpoint_every': checkpoint_every,
        'early_stopping': early_stopping,
        'patience': patience,
        'use_gpu': use_gpu,
        'verbose': verbose
    }
    
    # Run training
    result = run_training(config_obj, output_dir=output_dir, **train_args)
    
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
    
    console.print("\n[bold green]✓ Training complete![/bold green]")
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
        console.print(f"[green]✓[/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green]✓[/green] Loaded {len(val_dataset)} validation sequences")
    
    # Import and run the actual training
    from tempest.training import run_training
    
    # Build training arguments
    train_args = {
        'mode': 'hybrid',
        'train_data': train_dataset,
        'val_data': val_dataset,
        'constraint_weight': constraint_weight,
        'constraint_type': constraint_type,
        'use_gpu': use_gpu,
        'verbose': verbose
    }
    
    # Run training
    result = run_training(config_obj, output_dir=output_dir, **train_args)
    
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
    
    console.print("\n[bold green]✓ Hybrid training complete![/bold green]")
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
    diversity_weight: float = typer.Option(
        0.1,
        "--diversity-weight",
        help="Weight for diversity regularization",
        min=0.0,
        max=1.0
    ),
    bma_method: str = typer.Option(
        "BIC",
        "--bma-method",
        help="BMA approximation method: BIC, Laplace, or Variational"
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
    Train an ensemble of Tempest models using Bayesian Model Averaging.
    
    Ensemble training creates multiple models with different
    initializations and combines their predictions using BMA for improved
    accuracy and uncertainty estimation.
    
    [bold cyan]Examples:[/bold cyan]
    
    Train ensemble with 5 models:
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz --num-models 5
    ```
    
    Train larger ensemble with diversity:
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz --num-models 10 --diversity-weight 0.2
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    console.print(f"[yellow]Training Mode:[/yellow] Ensemble ({num_models} models)")
    console.print(f"[yellow]BMA Method:[/yellow] {bma_method}")
    console.print(f"[yellow]Diversity Weight:[/yellow] {diversity_weight}")
    
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
        if config_obj.ensemble.bma_config:
            config_obj.ensemble.bma_config.approximation = bma_method.lower()
    
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
        console.print(f"[green]✓[/green] Loaded {len(train_dataset)} training sequences")
    
    if val_data:
        console.print(f"[cyan]Loading validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            val_dataset = load_data(val_data)
        console.print(f"[green]✓[/green] Loaded {len(val_dataset)} validation sequences")
    
    # Import and run the actual training
    from tempest.training import run_training
    
    # Build training arguments
    train_args = {
        'mode': 'ensemble',
        'train_data': train_dataset,
        'val_data': val_dataset,
        'num_models': num_models,
        'diversity_weight': diversity_weight,
        'bma_method': bma_method,
        'use_gpu': use_gpu,
        'parallel': parallel,
        'verbose': verbose
    }
    
    # Run training
    result = run_training(config_obj, output_dir=output_dir, **train_args)
    
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
        
        console.print(table)
    
    console.print(f"\n[bold green]✓ Ensemble training complete! Trained {num_models} models.[/bold green]")
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
    
    # Import and run the actual training
    from tempest.training import resume_training
    
    # Build training arguments
    train_args = {
        'checkpoint_path': checkpoint,
        'additional_epochs': epochs,
        'use_gpu': use_gpu,
        'verbose': verbose
    }
    
    # Run training
    result = resume_training(
        config=config_obj, 
        output_dir=output_dir,
        **train_args
    )
    
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
    
    console.print("\n[bold green]✓ Training resumed and complete![/bold green]")
    if output_dir:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")
