"""
Tempest model combination/ensemble commands using Typer.
"""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console

from tempest.inference.combiner import ModelCombiner
from tempest.main import load_config

# Create the combine sub-application
combine_app = typer.Typer(help="Combine models using BMA/ensemble methods")

console = Console()


@combine_app.command()
def ensemble(
    model_paths: List[Path] = typer.Option(
        ...,
        "--models", "-m",
        help="Paths to models to combine (can specify multiple times)",
        exists=True
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output path for combined model"
    ),
    method: str = typer.Option(
        "voting",
        "--method",
        help="Combine method",
        callback=lambda v: v if v in ["voting", "averaging", "stacking", "bma"] 
                          else typer.BadParameter(f"Invalid method: {v}")
    ),
    weights: Optional[List[float]] = typer.Option(
        None,
        "--weights", "-w",
        help="Model weights (must match number of models)"
    ),
    validation_data: Optional[Path] = typer.Option(
        None,
        "--validation-data",
        help="Validation data for weight optimization"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file for ensemble settings"
    )
):
    """
    Combine multiple models into an ensemble.
    
    Examples:
        # Simple voting ensemble
        tempest combine ensemble --models m1.h5 --models m2.h5 --models m3.h5 \\
            --output ensemble.h5 --method voting
        
        # Weighted average with custom weights
        tempest combine ensemble --models m1.h5 --models m2.h5 \\
            --output ensemble.h5 --method averaging --weights 0.6 --weights 0.4
        
        # BMA ensemble with validation data for weight learning
        tempest combine ensemble --models m1.h5 --models m2.h5 --models m3.h5 \\
            --output ensemble.h5 --method bma --validation-data val.txt
    """
    console.print("[bold blue]Creating Model Ensemble[/bold blue]")
    console.print("=" * 60)
    console.print(f"Method: {method.upper()}")
    console.print(f"Number of models: {len(model_paths)}")
    
    # Validate weights if provided
    if weights:
        if len(weights) != len(model_paths):
            console.print("[red]Error: Number of weights must match number of models[/red]")
            raise typer.Exit(1)
        if abs(sum(weights) - 1.0) > 0.01:
            console.print("[yellow]Warning: Weights do not sum to 1.0, normalizing...[/yellow]")
            total = sum(weights)
            weights = [w/total for w in weights]
    
    # Load configuration if provided
    cfg = None
    if config:
        cfg = load_config(str(config))
    
    # Create ensemble builder
    builder = ModelCombiner(
        model_paths=[str(p) for p in model_paths],
        config=cfg
    )
    
    # Build ensemble based on method
    if method == "voting":
        console.print("Building voting ensemble...")
        ensemble_model = builder.build_voting_ensemble(weights=weights)
        
    elif method == "averaging":
        console.print("Building averaging ensemble...")
        ensemble_model = builder.build_averaging_ensemble(weights=weights)
        
    elif method == "stacking":
        if not validation_data:
            console.print("[red]Error: Stacking requires validation data[/red]")
            raise typer.Exit(1)
        console.print("Building stacking ensemble...")
        console.print("Training meta-learner on validation data...")
        ensemble_model = builder.build_stacking_ensemble(
            validation_data=str(validation_data)
        )
        
    elif method == "bma":
        console.print("Building Bayesian Model Averaging ensemble...")
        if validation_data:
            console.print("Learning BMA weights from validation data...")
            ensemble_model = builder.build_bma_ensemble(
                validation_data=str(validation_data)
            )
        else:
            console.print("Using uniform BMA weights...")
            ensemble_model = builder.build_bma_ensemble()
    
    # Save ensemble
    console.print(f"Saving ensemble to: {output}")
    builder.save_ensemble(ensemble_model, str(output))
    
    # Display ensemble information
    _display_ensemble_info(method, model_paths, weights)
    
    console.print(f"\n[green]✓[/green] Ensemble created successfully: {output}")


@combine_app.command()
def optimize_weights(
    model_paths: List[Path] = typer.Option(
        ...,
        "--models", "-m",
        help="Paths to models to combine (specify multiple times)",
        exists=True
    ),
    validation_data: Path = typer.Option(
        ...,
        "--validation-data", "-v",
        help="Validation data for weight optimization",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    method: str = typer.Option(
        "grid_search",
        "--method",
        help="Optimization method",
        callback=lambda v: v if v in ["grid_search", "random_search", "bayesian"] 
                          else typer.BadParameter(f"Invalid method: {v}")
    ),
    metric: str = typer.Option(
        "accuracy",
        "--metric",
        help="Metric to optimize"
    ),
    n_trials: int = typer.Option(
        100,
        "--n-trials",
        help="Number of optimization trials",
        min=10,
        max=1000
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save optimal weights to file"
    )
):
    """
    Find optimal ensemble weights using validation data.
    
    Examples:
        # Grid search for optimal weights
        tempest combine optimize-weights --models m1.h5 --models m2.h5 --models m3.h5 \\
            --validation-data val.txt --method grid_search
        
        # Bayesian optimization
        tempest combine optimize-weights --models m1.h5 --models m2.h5 \\
            --validation-data val.txt --method bayesian --n-trials 200
    """
    console.print("[bold blue]Optimizing Ensemble Weights[/bold blue]")
    console.print("=" * 60)
    console.print(f"Method: {method}")
    console.print(f"Metric: {metric}")
    console.print(f"Trials: {n_trials}")
    
    from tempest.training.ensemble import WeightOptimizer
    
    # Create optimizer
    optimizer = WeightOptimizer(
        model_paths=[str(p) for p in model_paths],
        validation_data=str(validation_data)
    )
    
    # Run optimization
    with console.status(f"Running {method} optimization..."):
        if method == "grid_search":
            optimal_weights, best_score = optimizer.grid_search(
                metric=metric,
                n_points=n_trials
            )
        elif method == "random_search":
            optimal_weights, best_score = optimizer.random_search(
                metric=metric,
                n_trials=n_trials
            )
        elif method == "bayesian":
            optimal_weights, best_score = optimizer.bayesian_optimization(
                metric=metric,
                n_trials=n_trials
            )
    
    # Display results
    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"Best {metric}: {best_score:.4f}")
    console.print("\nOptimal weights:")
    
    for i, (path, weight) in enumerate(zip(model_paths, optimal_weights)):
        console.print(f"  Model {i+1} ({path.name}): {weight:.4f}")
    
    # Save weights if requested
    if output:
        import json
        weights_data = {
            'method': method,
            'metric': metric,
            'best_score': best_score,
            'weights': {
                str(path): weight 
                for path, weight in zip(model_paths, optimal_weights)
            }
        }
        with open(output, 'w') as f:
            json.dump(weights_data, f, indent=2)
        console.print(f"\n[green]✓[/green] Weights saved to: {output}")


@combine_app.command()
def stack(
    base_models: List[Path] = typer.Option(
        ...,
        "--base-models", "-b",
        help="Base model paths (specify multiple times)",
        exists=True
    ),
    meta_learner: str = typer.Option(
        "logistic",
        "--meta-learner",
        help="Type of meta-learner",
        callback=lambda v: v if v in ["logistic", "neural", "random_forest", "gradient_boost"] 
                          else typer.BadParameter(f"Invalid meta-learner: {v}")
    ),
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        help="Training data for meta-learner",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    validation_data: Optional[Path] = typer.Option(
        None,
        "--validation-data",
        help="Validation data for meta-learner"
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output path for stacked model"
    ),
    cv_folds: int = typer.Option(
        5,
        "--cv-folds",
        help="Cross-validation folds for meta-learner training",
        min=2,
        max=20
    )
):
    """
    Create a stacked ensemble with a meta-learner.
    
    Stacking trains a meta-learner to combine base model predictions
    optimally based on their individual strengths.
    
    Examples:
        # Stack with logistic regression meta-learner
        tempest combine stack --base-models m1.h5 --base-models m2.h5 --base-models m3.h5 \\
            --train-data train.txt --meta-learner logistic --output stacked.h5
        
        # Stack with neural network meta-learner
        tempest combine stack --base-models m1.h5 --base-models m2.h5 \\
            --train-data train.txt --validation-data val.txt \\
            --meta-learner neural --output stacked.h5
    """
    console.print("[bold blue]Creating Stacked Ensemble[/bold blue]")
    console.print("=" * 60)
    console.print(f"Base models: {len(base_models)}")
    console.print(f"Meta-learner: {meta_learner}")
    console.print(f"CV folds: {cv_folds}")
    
    from tempest.training.ensemble import StackingEnsemble
    
    # Create stacking ensemble
    stacker = StackingEnsemble(
        base_model_paths=[str(p) for p in base_models],
        meta_learner_type=meta_learner
    )
    
    # Train meta-learner
    console.print("\n[bold]Phase 1:[/bold] Generating base model predictions...")
    with console.status("Processing training data..."):
        stacker.generate_meta_features(str(train_data))
    
    console.print("[bold]Phase 2:[/bold] Training meta-learner...")
    with console.status(f"Training {meta_learner} meta-learner..."):
        if validation_data:
            stacker.train_meta_learner(
                cv_folds=cv_folds,
                validation_data=str(validation_data)
            )
        else:
            stacker.train_meta_learner(cv_folds=cv_folds)
    
    # Evaluate if validation data provided
    if validation_data:
        console.print("[bold]Phase 3:[/bold] Evaluating on validation set...")
        val_score = stacker.evaluate(str(validation_data))
        console.print(f"Validation accuracy: {val_score:.4f}")
    
    # Save stacked model
    console.print(f"\nSaving stacked ensemble to: {output}")
    stacker.save(str(output))
    
    console.print(f"\n[green]✓[/green] Stacked ensemble created successfully!")


def _display_ensemble_info(method: str, model_paths: List[Path], weights: Optional[List[float]]):
    """Display ensemble information."""
    from rich.table import Table
    
    table = Table(title="Ensemble Configuration")
    table.add_column("Model", style="cyan")
    table.add_column("Weight", style="magenta")
    
    if weights:
        for path, weight in zip(model_paths, weights):
            table.add_row(path.name, f"{weight:.4f}")
    else:
        # Equal weights
        equal_weight = 1.0 / len(model_paths)
        for path in model_paths:
            table.add_row(path.name, f"{equal_weight:.4f}")
    
    console.print(table)
    
    # Method-specific information
    info_messages = {
        "voting": "Using majority voting for predictions",
        "averaging": "Using weighted average of model outputs",
        "stacking": "Using meta-learner to combine predictions",
        "bma": "Using Bayesian Model Averaging with uncertainty estimation"
    }
    
    console.print(f"\n[yellow]Method:[/yellow] {info_messages.get(method, method)}")
