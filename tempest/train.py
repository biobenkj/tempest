"""
Tempest training commands using Typer.
"""

import typer
from pathlib import Path
from typing import Optional
import logging

from tempest.main import main as train_main
from tempest.utils import load_config

# Create the train sub-application
train_app = typer.Typer(help="Train Tempest models with various approaches")

logger = logging.getLogger(__name__)


@train_app.command()
def standard(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for trained models"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Learning rate"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None,
        "--checkpoint-every",
        help="Save checkpoint every N epochs"
    ),
    early_stopping: bool = typer.Option(
        False,
        "--early-stopping",
        help="Enable early stopping"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Early stopping patience (epochs)"
    )
):
    """
    Train a standard Tempest model.
    
    Examples:
        # Train with default settings
        tempest train standard --config config.yaml
        
        # Train with custom hyperparameters
        tempest train standard --config config.yaml --epochs 100 --batch-size 64
    """
    typer.echo("=" * 80)
    typer.echo(" " * 30 + "TEMPEST TRAINER")
    typer.echo("=" * 80)
    typer.echo("Training mode: Standard")
    
    # Build arguments dictionary for train_main
    train_args = {
        'config': str(config),
        'command': 'train',
        'hybrid': False,
        'ensemble': False
    }
    
    # Add optional parameters
    if output_dir:
        train_args['output_dir'] = str(output_dir)
    if epochs is not None:
        train_args['epochs'] = epochs
    if batch_size is not None:
        train_args['batch_size'] = batch_size
    if learning_rate is not None:
        train_args['learning_rate'] = learning_rate
    if checkpoint_every is not None:
        train_args['checkpoint_every'] = checkpoint_every
    if early_stopping:
        train_args['early_stopping'] = early_stopping
        train_args['patience'] = patience
    
    # Call the training main function
    train_main(train_args)
    
    typer.secho("Training complete!", fg=typer.colors.GREEN, bold=True)


@train_app.command()
def hybrid(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for trained models"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Learning rate"
    ),
    constraint_weight: float = typer.Option(
        1.0,
        "--constraint-weight",
        help="Weight for constraint loss",
        min=0.0
    )
):
    """
    Train a hybrid Tempest model with constraints.
    
    The hybrid approach combines standard CRF training with
    length constraint enforcement for improved accuracy.
    
    Examples:
        # Train hybrid model
        tempest train hybrid --config config.yaml
        
        # Train with stronger constraint enforcement
        tempest train hybrid --config config.yaml --constraint-weight 2.0
    """
    typer.echo("=" * 80)
    typer.echo(" " * 30 + "TEMPEST TRAINER")
    typer.echo("=" * 80)
    typer.echo("Training mode: Hybrid (with constraints)")
    
    # Build arguments dictionary for train_main
    train_args = {
        'config': str(config),
        'command': 'train',
        'hybrid': True,
        'ensemble': False,
        'constraint_weight': constraint_weight
    }
    
    # Add optional parameters
    if output_dir:
        train_args['output_dir'] = str(output_dir)
    if epochs is not None:
        train_args['epochs'] = epochs
    if batch_size is not None:
        train_args['batch_size'] = batch_size
    if learning_rate is not None:
        train_args['learning_rate'] = learning_rate
    
    # Call the training main function
    train_main(train_args)
    
    typer.secho("Hybrid training complete!", fg=typer.colors.GREEN, bold=True)


@train_app.command()
def ensemble(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for trained models"
    ),
    num_models: int = typer.Option(
        5,
        "--num-models",
        help="Number of models in ensemble",
        min=2,
        max=20
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs per model"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Learning rate"
    ),
    diversity_weight: float = typer.Option(
        0.1,
        "--diversity-weight",
        help="Weight for diversity regularization",
        min=0.0,
        max=1.0
    )
):
    """
    Train an ensemble of Tempest models.
    
    Ensemble training creates multiple models with different
    initializations and combines their predictions for improved
    accuracy and uncertainty estimation.
    
    Examples:
        # Train ensemble with 5 models
        tempest train ensemble --config config.yaml --num-models 5
        
        # Train larger ensemble with diversity
        tempest train ensemble --config config.yaml --num-models 10 --diversity-weight 0.2
    """
    typer.echo("=" * 80)
    typer.echo(" " * 30 + "TEMPEST TRAINER")
    typer.echo("=" * 80)
    typer.echo(f"Training mode: Ensemble ({num_models} models)")
    
    # Build arguments dictionary for train_main
    train_args = {
        'config': str(config),
        'command': 'train',
        'hybrid': False,
        'ensemble': True,
        'num_models': num_models,
        'diversity_weight': diversity_weight
    }
    
    # Add optional parameters
    if output_dir:
        train_args['output_dir'] = str(output_dir)
    if epochs is not None:
        train_args['epochs'] = epochs
    if batch_size is not None:
        train_args['batch_size'] = batch_size
    if learning_rate is not None:
        train_args['learning_rate'] = learning_rate
    
    # Call the training main function
    train_main(train_args)
    
    typer.secho(f"Ensemble training complete! Trained {num_models} models.", 
                fg=typer.colors.GREEN, bold=True)


@train_app.command()
def resume(
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
    )
):
    """
    Resume training from a checkpoint.
    
    Examples:
        # Resume training for 50 more epochs
        tempest train resume --checkpoint model_checkpoint.ckpt --epochs 50
    """
    typer.echo("=" * 80)
    typer.echo(" " * 30 + "TEMPEST TRAINER")
    typer.echo("=" * 80)
    typer.echo(f"Resuming from checkpoint: {checkpoint}")
    
    # Build arguments dictionary for train_main
    train_args = {
        'command': 'train',
        'resume': True,
        'checkpoint': str(checkpoint)
    }
    
    if config:
        train_args['config'] = str(config)
    if epochs is not None:
        train_args['additional_epochs'] = epochs
    
    # Call the training main function
    train_main(train_args)
    
    typer.secho("Training resumed and complete!", fg=typer.colors.GREEN, bold=True)
