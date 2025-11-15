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
        help="Path to default training data (used for all phases unless phase-specific data provided)"
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val-data",
        help="Path to default validation data (used for all phases unless phase-specific data provided)"
    ),
    unlabeled_data: Optional[Path] = typer.Option(
        None,
        "--unlabeled-data",
        help="Path to unlabeled data for semi-supervised hybrid training"
    ),
    # Phase-specific data options
    warmup_train: Optional[Path] = typer.Option(
        None,
        "--warmup-train",
        help="Training data for warmup phase (Phase 1)"
    ),
    warmup_val: Optional[Path] = typer.Option(
        None,
        "--warmup-val",
        help="Validation data for warmup phase (Phase 1)"
    ),
    adversarial_train: Optional[Path] = typer.Option(
        None,
        "--adversarial-train",
        help="Training data for adversarial phase (Phase 2)"
    ),
    adversarial_val: Optional[Path] = typer.Option(
        None,
        "--adversarial-val",
        help="Validation data for adversarial phase (Phase 2)"
    ),
    pseudolabel_train: Optional[Path] = typer.Option(
        None,
        "--pseudolabel-train",
        help="Training data for pseudo-label phase (Phase 3)"
    ),
    pseudolabel_val: Optional[Path] = typer.Option(
        None,
        "--pseudolabel-val",
        help="Validation data for pseudo-label phase (Phase 3)"
    ),
    pseudolabel_unlabeled: Optional[Path] = typer.Option(
        None,
        "--pseudolabel-unlabeled",
        help="Unlabeled data for pseudo-label phase (Phase 3)"
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
    Train a hybrid Tempest model with three-phase training strategy.
    
    The hybrid training approach includes:
    1. Warmup phase: Standard supervised training
    2. Adversarial phase: Discriminator-based robustness training
    3. Pseudo-label phase: Self-training with unlabeled data
    
    You can specify different datasets for each phase using --warmup-train,
    --adversarial-train, --pseudolabel-train, etc. If not specified, uses
    the default --train-data for all phases.
    
    [bold cyan]Examples:[/bold cyan]
    
    Basic hybrid training with single dataset:
    ```
    tempest train hybrid --config config.yaml --train-data train.pkl.gz --val-data val.pkl.gz
    ```
    
    Hybrid training with phase-specific datasets:
    ```
    tempest train hybrid --config config.yaml \\
        --warmup-train data/simulated/warmup/train.pkl.gz \\
        --warmup-val data/simulated/warmup/val.pkl.gz \\
        --adversarial-train data/simulated/adversarial_p2/train.pkl.gz \\
        --adversarial-val data/simulated/adversarial_p2/val.pkl.gz \\
        --pseudolabel-unlabeled data/unlabeled/reads.pkl.gz
    ```
    
    Hybrid training with mixed approach (some phases use default, others use specific):
    ```
    tempest train hybrid --config config.yaml \\
        --train-data data/default/train.pkl.gz \\
        --val-data data/default/val.pkl.gz \\
        --adversarial-train data/simulated/adversarial_p2/train.pkl.gz \\
        --adversarial-val data/simulated/adversarial_p2/val.pkl.gz
    ```
    """
    console.print("\n[bold blue]═" * 80 + "[/bold blue]")
    console.print(" " * 30 + "[bold cyan]TEMPEST HYBRID TRAINER[/bold cyan]")
    console.print("[bold blue]═" * 80 + "[/bold blue]\n")
    console.print("[yellow]Training Mode:[/yellow] Hybrid (3-phase)")
    
    # Load configuration
    config_obj = load_config(config)
    
    # Override config with CLI arguments if provided
    if epochs is not None and config_obj.training:
        config_obj.training.epochs = epochs
    if batch_size is not None:
        config_obj.model.batch_size = batch_size
    if learning_rate is not None and config_obj.training:
        config_obj.training.learning_rate = learning_rate
    
    # Display phase configuration
    if config_obj.hybrid:
        console.print(f"[yellow]Phase 1 (Warmup):[/yellow] {config_obj.hybrid.warmup_epochs} epochs")
        console.print(f"[yellow]Phase 2 (Adversarial):[/yellow] {config_obj.hybrid.discriminator_epochs} epochs")
        console.print(f"[yellow]Phase 3 (Pseudo-label):[/yellow] {config_obj.hybrid.pseudolabel_epochs} epochs")
    
    # Load default training data
    default_train_dataset = None
    default_val_dataset = None
    
    if train_data:
        console.print(f"\n[cyan]Loading default training data:[/cyan] {train_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            default_train_dataset = load_data(train_data)
        console.print(f"[green]✓[/green] Loaded {len(default_train_dataset)} sequences")
    
    if val_data:
        console.print(f"[cyan]Loading default validation data:[/cyan] {val_data}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading...", total=None)
            default_val_dataset = load_data(val_data)
        console.print(f"[green]✓[/green] Loaded {len(default_val_dataset)} sequences")
    
    # Load phase-specific data
    phase_data = {}
    
    # Warmup phase
    if warmup_train or warmup_val:
        console.print("\n[bold cyan]Phase 1: Warmup Data[/bold cyan]")
        phase_data['warmup'] = {}
        
        if warmup_train:
            console.print(f"[cyan]Loading warmup training data:[/cyan] {warmup_train}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['warmup']['train'] = load_data(warmup_train)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['warmup']['train'])} sequences")
        
        if warmup_val:
            console.print(f"[cyan]Loading warmup validation data:[/cyan] {warmup_val}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['warmup']['val'] = load_data(warmup_val)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['warmup']['val'])} sequences")
    
    # Adversarial phase
    if adversarial_train or adversarial_val:
        console.print("\n[bold cyan]Phase 2: Adversarial Data[/bold cyan]")
        phase_data['adversarial'] = {}
        
        if adversarial_train:
            console.print(f"[cyan]Loading adversarial training data:[/cyan] {adversarial_train}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['adversarial']['train'] = load_data(adversarial_train)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['adversarial']['train'])} sequences")
        
        if adversarial_val:
            console.print(f"[cyan]Loading adversarial validation data:[/cyan] {adversarial_val}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['adversarial']['val'] = load_data(adversarial_val)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['adversarial']['val'])} sequences")
    
    # Pseudo-label phase
    if pseudolabel_train or pseudolabel_val or pseudolabel_unlabeled:
        console.print("\n[bold cyan]Phase 3: Pseudo-label Data[/bold cyan]")
        phase_data['pseudolabel'] = {}
        
        if pseudolabel_train:
            console.print(f"[cyan]Loading pseudo-label training data:[/cyan] {pseudolabel_train}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['pseudolabel']['train'] = load_data(pseudolabel_train)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['pseudolabel']['train'])} sequences")
        
        if pseudolabel_val:
            console.print(f"[cyan]Loading pseudo-label validation data:[/cyan] {pseudolabel_val}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Loading...", total=None)
                phase_data['pseudolabel']['val'] = load_data(pseudolabel_val)
            console.print(f"[green]✓[/green] Loaded {len(phase_data['pseudolabel']['val'])} sequences")
        
        if pseudolabel_unlabeled:
            console.print(f"[cyan]Loading pseudo-label unlabeled data:[/cyan] {pseudolabel_unlabeled}")
            # Note: unlabeled data path stored as string for lazy loading
            phase_data['pseudolabel']['unlabeled_path'] = str(pseudolabel_unlabeled)
            console.print(f"[green]✓[/green] Path stored (will be loaded during training)")
    
    # Build training arguments
    train_args = {
        'train_data': default_train_dataset,
        'val_data': default_val_dataset,
        'unlabeled_path': str(unlabeled_data) if unlabeled_data else None,
        'phase_data': phase_data if phase_data else None,
        'use_gpu': use_gpu,
        'verbose': verbose,
    }
    
    # Dispatch via unified main entrypoint
    result = tempest_main("train", config, output=output_dir, subcommand="hybrid", **train_args)
    
    # Display results
    if result and 'history' in result:
        console.print("\n[bold magenta]Training History by Phase:[/bold magenta]\n")
        
        # Create summary table
        for phase_name, phase_hist in result['history'].items():
            if not phase_hist:
                continue
                
            table = Table(title=f"{phase_name.title()} Phase", show_header=True, header_style="bold cyan")
            table.add_column("Epoch", style="dim")
            table.add_column("Train Loss", style="yellow")
            table.add_column("Train Acc", style="yellow")
            table.add_column("Val Loss", style="green")
            table.add_column("Val Acc", style="green")
            
            # Show first, middle, and last epochs
            epochs_to_show = []
            if 'loss' in phase_hist and len(phase_hist['loss']) > 0:
                n_epochs = len(phase_hist['loss'])
                if n_epochs <= 5:
                    epochs_to_show = list(range(n_epochs))
                else:
                    epochs_to_show = [0, n_epochs//2, n_epochs-1]
                
                for epoch_idx in epochs_to_show:
                    train_loss = phase_hist.get('loss', [None])[epoch_idx]
                    train_acc = phase_hist.get('accuracy', [None])[epoch_idx]
                    val_loss = phase_hist.get('val_loss', [None])[epoch_idx] if 'val_loss' in phase_hist else None
                    val_acc = phase_hist.get('val_accuracy', [None])[epoch_idx] if 'val_accuracy' in phase_hist else None
                    
                    table.add_row(
                        str(epoch_idx + 1),
                        f"{train_loss:.4f}" if train_loss is not None else "N/A",
                        f"{train_acc:.4f}" if train_acc is not None else "N/A",
                        f"{val_loss:.4f}" if val_loss is not None else "N/A",
                        f"{val_acc:.4f}" if val_acc is not None else "N/A"
                    )
            
            console.print(table)
    
    if result and 'final_val_accuracy' in result and result['final_val_accuracy'] is not None:
        console.print(f"\n[bold green]Final Validation Accuracy: {result['final_val_accuracy']:.4f}[/bold green]")
    
    console.print(f"\n[bold green]✓ Hybrid training complete![/bold green]")
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
    num_models: int = typer.Option(
        5,
        "--num-models", "-n",
        help="Number of models in ensemble"
    ),
    model_types: Optional[str] = typer.Option(
        None,
        "--model-types",
        help="Comma-separated list of model types (e.g., 'standard,hybrid,standard,hybrid,standard')"
    ),
    hybrid_ratio: float = typer.Option(
        0.4,
        "--hybrid-ratio",
        help="Ratio of hybrid models in ensemble (0.0 to 1.0)"
    ),
    variation_type: str = typer.Option(
        "both",
        "--variation-type",
        help="Type of model variation: 'architecture', 'initialization', or 'both'"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for trained ensemble"
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
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available"
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Train models in parallel (experimental)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed training progress"
    )
):
    """
    Train an ensemble of Tempest models with Bayesian Model Averaging.
    
    The ensemble combines multiple models (standard and/or hybrid) with
    architectural variations and different initializations for improved
    robustness and uncertainty quantification.
    
    [bold cyan]Examples:[/bold cyan]
    
    Train 5-model ensemble with default 40% hybrid ratio:
    ```
    tempest train ensemble --config config.yaml --train-data train.pkl.gz --num-models 5
    ```
    
    Train ensemble with specific model types:
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
        console.print("[yellow]⚠ Warning:[/yellow] Hybrid models requested but no unlabeled data provided")
    
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
    
    console.print(f"\n[bold green]✓ Ensemble training complete![/bold green]")
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
    
    console.print("\n[bold green]✓ Training resumed and complete![/bold green]")
    if output_dir:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")
