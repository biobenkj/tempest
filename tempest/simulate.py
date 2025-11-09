"""
Tempest simulation commands using Typer.
"""

import typer
from pathlib import Path
from typing import Optional
import logging

from tempest.utils import load_config
from tempest.data import SequenceSimulator

# Create the simulate sub-application
simulate_app = typer.Typer(help="Generate synthetic sequence reads for training and testing")

logger = logging.getLogger(__name__)


@simulate_app.command()
def generate(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for generated sequences"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for split datasets"
    ),
    num_sequences: Optional[int] = typer.Option(
        None,
        "--num-sequences", "-n",
        help="Number of sequences to generate"
    ),
    split: bool = typer.Option(
        False,
        "--split",
        help="Generate train/validation split"
    ),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction",
        help="Fraction of data for training (default: 0.8)",
        min=0.1,
        max=0.9
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility"
    )
):
    """
    Generate synthetic sequence reads with configurable architecture.
    
    Examples:
        # Generate 1000 sequences to a single file
        tempest simulate generate --config config.yaml --output sequences.txt -n 1000
        
        # Generate train/validation split
        tempest simulate generate --config config.yaml --split --output-dir ./data -n 5000
    """
    typer.echo("=" * 80)
    typer.echo(" " * 30 + "TEMPEST SIMULATOR")
    typer.echo("=" * 80)
    
    # Load base configuration
    cfg = load_config(str(config))
    
    # Override simulation parameters from command line
    if num_sequences:
        cfg.simulation.num_sequences = num_sequences
    if seed:
        cfg.simulation.random_seed = seed
    
    # Create simulator
    simulator = SequenceSimulator(cfg)
    
    # Generate sequences
    if split:
        # Generate train/validation split
        typer.echo(f"Generating {cfg.simulation.num_sequences} sequences")
        typer.echo(f"Split: {train_fraction:.0%} train, {1-train_fraction:.0%} validation")
        
        train_data, val_data = simulator.generate_split(
            num_sequences=cfg.simulation.num_sequences,
            train_fraction=train_fraction
        )
        
        # Save to output directory
        output_path = Path(output_dir or "./data")
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "train.txt"
        val_file = output_path / "val.txt"
        
        simulator.save_sequences(train_data, train_file)
        simulator.save_sequences(val_data, val_file)
        
        typer.secho(f"✓ Train data saved to: {train_file}", fg=typer.colors.GREEN)
        typer.secho(f"✓ Validation data saved to: {val_file}", fg=typer.colors.GREEN)
    else:
        # Generate single dataset
        typer.echo(f"Generating {cfg.simulation.num_sequences} sequences")
        sequences = simulator.generate(cfg.simulation.num_sequences)
        
        output_file = Path(output or "sequences.txt")
        simulator.save_sequences(sequences, output_file)
        typer.secho(f"✓ Sequences saved to: {output_file}", fg=typer.colors.GREEN)
    
    typer.secho("Simulation complete!", fg=typer.colors.GREEN, bold=True)


@simulate_app.command()
def validate(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed configuration"
    )
):
    """
    Validate simulation configuration without generating sequences.
    
    This command loads and validates the configuration file,
    showing the parameters that would be used for simulation.
    """
    typer.echo("Validating simulation configuration...")
    
    try:
        cfg = load_config(str(config))
        typer.secho("✓ Configuration is valid!", fg=typer.colors.GREEN)
        
        if verbose:
            typer.echo("\nSimulation parameters:")
            typer.echo(f"  - Number of sequences: {cfg.simulation.num_sequences}")
            typer.echo(f"  - Random seed: {cfg.simulation.random_seed}")
            typer.echo(f"  - Segment architecture: {len(cfg.segments)} segments")
            for segment in cfg.segments:
                typer.echo(f"    • {segment.name}: {segment.min_length}-{segment.max_length} bp")
    except Exception as e:
        typer.secho(f"✗ Configuration invalid: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
