"""
Tempest simulation module - refactored with Typer approach.

This module provides functionality for generating synthetic sequence reads
for training and testing TEMPEST models.
"""

import typer
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import yaml
import json
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the data simulator components
from tempest.data import (
    SequenceSimulator,
    SimulatedRead,
    create_simulator_from_config
)
from tempest.config import TempestConfig, SimulationConfig

logger = logging.getLogger(__name__)
console = Console()

# Create the simulate Typer sub-application
simulate_app = typer.Typer(
    help="Generate synthetic sequence reads for training and testing",
    rich_markup_mode="rich"
)


def run_simulation(
    config: TempestConfig,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Functional entry point for simulation from main.py.
    
    Args:
        config: Loaded TempestConfig instance
        output_dir: Optional output directory
        **kwargs: Additional arguments to override config
        
    Returns:
        Dictionary with simulation results and statistics
    """
    # Update config with any overrides from kwargs
    sim_config = config.simulation
    
    if 'num_sequences' in kwargs:
        sim_config.num_sequences = kwargs['num_sequences']
    if 'seed' in kwargs:
        sim_config.random_seed = kwargs['seed']
    if 'train_fraction' in kwargs:
        sim_config.train_split = kwargs['train_fraction']
    
    # Determine output paths
    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    simulator = create_simulator_from_config(config)
    
    # Generate sequences based on configuration
    result = {}
    
    if kwargs.get('split', False):
        # Generate train/validation split
        logger.info(f"Generating {sim_config.num_sequences} sequences with train/val split")
        
        n_train = int(sim_config.num_sequences * sim_config.train_split)
        n_val = sim_config.num_sequences - n_train
        
        train_reads = simulator.generate_reads(n_train)
        val_reads = simulator.generate_reads(n_val)
        
        # Save sequences
        train_file = output_path / "train.txt"
        val_file = output_path / "val.txt"
        
        _save_reads(train_reads, train_file)
        _save_reads(val_reads, val_file)
        
        result['train_file'] = str(train_file)
        result['val_file'] = str(val_file)
        result['n_train'] = n_train
        result['n_val'] = n_val
        
        logger.info(f"Saved {n_train} training sequences to {train_file}")
        logger.info(f"Saved {n_val} validation sequences to {val_file}")
        
    else:
        # Generate single dataset
        logger.info(f"Generating {sim_config.num_sequences} sequences")
        
        reads = simulator.generate_reads(sim_config.num_sequences)
        
        output_file = output_path / (kwargs.get('output_file', 'sequences.txt'))
        _save_reads(reads, output_file)
        
        result['output_file'] = str(output_file)
        result['n_sequences'] = sim_config.num_sequences
        
        logger.info(f"Saved {sim_config.num_sequences} sequences to {output_file}")
    
    # Add statistics
    result['config'] = config
    result['seed'] = sim_config.random_seed
    result['success'] = True
    
    return result


def _save_reads(reads: List[SimulatedRead], output_file: Path):
    """
    Save simulated reads to file.
    
    Args:
        reads: List of SimulatedRead objects
        output_file: Path to save file
    """
    with open(output_file, 'w') as f:
        for read in reads:
            # Save in the format: sequence<TAB>labels
            labels_str = ' '.join(read.labels)
            f.write(f"{read.sequence}\t{labels_str}\n")


@simulate_app.command("generate")
def generate_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for generated sequences"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-d",
        help="Output directory for generated files"
    ),
    num_sequences: Optional[int] = typer.Option(
        None,
        "--num-sequences", "-n",
        help="Number of sequences to generate (overrides config)",
        min=1
    ),
    split: bool = typer.Option(
        False,
        "--split", "-s",
        help="Generate train/validation split"
    ),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction", "-t",
        help="Fraction of data for training when using --split",
        min=0.1,
        max=0.9
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-r",
        help="Random seed for reproducibility (overrides config)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed progress and statistics"
    )
):
    """
    Generate synthetic sequence reads with configurable architecture.
    
    This command generates synthetic sequences based on the architecture
    defined in the configuration file. The sequences can be saved as
    a single dataset or split into training and validation sets.
    
    Examples:
        
        # Generate 1000 sequences to a single file
        tempest simulate generate -c config.yaml -o sequences.txt -n 1000
        
        # Generate train/validation split with custom ratio
        tempest simulate generate -c config.yaml --split -d ./data -t 0.8
        
        # Generate with specific seed for reproducibility
        tempest simulate generate -c config.yaml -o test.txt --seed 12345
    """
    # Print header
    if verbose:
        console.print("=" * 60)
        console.print(" " * 20 + "[bold blue]TEMPEST SIMULATOR[/bold blue]")
        console.print("=" * 60)
    
    try:
        # Load configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading configuration...", total=None)
            cfg = TempestConfig.from_yaml(str(config))
            progress.update(task, completed=True, description="Configuration loaded")
        
        # Override parameters from command line
        kwargs = {}
        if num_sequences is not None:
            kwargs['num_sequences'] = num_sequences
        if seed is not None:
            kwargs['seed'] = seed
        if train_fraction != 0.8:
            kwargs['train_fraction'] = train_fraction
        if output:
            kwargs['output_file'] = output.name
        kwargs['split'] = split
        
        # Determine output directory
        if output_dir:
            out_dir = output_dir
        elif output:
            out_dir = output.parent
        else:
            out_dir = Path('.')
        
        # Show configuration if verbose
        if verbose:
            console.print("\n[cyan]Configuration:[/cyan]")
            console.print(f"  Config file: {config}")
            console.print(f"  Sequences: {kwargs.get('num_sequences', cfg.simulation.num_sequences)}")
            console.print(f"  Random seed: {kwargs.get('seed', cfg.simulation.random_seed)}")
            console.print(f"  Architecture: {len(cfg.simulation.sequence_order)} segments")
            if split:
                console.print(f"  Train fraction: {train_fraction:.0%}")
            console.print()
        
        # Run simulation with progress tracking
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[cyan]Generating sequences...",
                total=kwargs.get('num_sequences', cfg.simulation.num_sequences)
            )
            
            # Run simulation
            result = run_simulation(cfg, output_dir=out_dir, **kwargs)
            
            progress.update(task, completed=True)
        
        # Report results
        console.print("\n[bold green]✓ Simulation complete![/bold green]")
        
        if split:
            console.print(f"  Training sequences: {result['n_train']}")
            console.print(f"  Validation sequences: {result['n_val']}")
            console.print(f"  Training file: [green]{result['train_file']}[/green]")
            console.print(f"  Validation file: [green]{result['val_file']}[/green]")
        else:
            console.print(f"  Total sequences: {result['n_sequences']}")
            console.print(f"  Output file: [green]{result['output_file']}[/green]")
        
        if verbose:
            console.print(f"\n[dim]Random seed used: {result['seed']}[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@simulate_app.command("validate")
def validate_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed configuration"
    )
):
    """
    Validate simulation configuration without generating sequences.
    
    This command loads and validates the configuration file, checking
    for errors and showing the parameters that would be used for simulation.
    
    Examples:
        
        # Quick validation
        tempest simulate validate -c config.yaml
        
        # Detailed validation with all parameters
        tempest simulate validate -c config.yaml --verbose
    """
    try:
        console.print("[cyan]Validating simulation configuration...[/cyan]")
        
        # Load configuration
        cfg = TempestConfig.from_yaml(str(config))
        
        # Basic validation checks
        errors = []
        warnings = []
        
        # Check simulation parameters
        sim = cfg.simulation
        if sim.num_sequences < 1:
            errors.append("num_sequences must be at least 1")
        
        if sim.train_split <= 0 or sim.train_split >= 1:
            errors.append("train_split must be between 0 and 1")
        
        # Check sequence order
        if not sim.sequence_order:
            errors.append("sequence_order must be defined")
        
        # Check for required files
        if sim.whitelist_files:
            for name, path in sim.whitelist_files.items():
                if not Path(path).exists():
                    warnings.append(f"Whitelist file not found: {name} -> {path}")
        
        if sim.pwm_files:
            for name, path in sim.pwm_files.items():
                if not Path(path).exists():
                    warnings.append(f"PWM file not found: {name} -> {path}")
        
        # Report results
        if errors:
            console.print("\n[bold red]Configuration errors:[/bold red]")
            for error in errors:
                console.print(f"  ✗ {error}")
            raise typer.Exit(1)
        
        console.print("[bold green]✓ Configuration is valid![/bold green]")
        
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  ⚠ {warning}")
        
        if verbose:
            console.print("\n[cyan]Simulation parameters:[/cyan]")
            console.print(f"  Number of sequences: {sim.num_sequences}")
            console.print(f"  Train/test split: {sim.train_split:.0%}")
            console.print(f"  Random seed: {sim.random_seed}")
            console.print(f"\n[cyan]Segment architecture:[/cyan]")
            
            if sim.sequence_order:
                for i, segment in enumerate(sim.sequence_order, 1):
                    # Get length info if available
                    if sim.segment_generation and 'lengths' in sim.segment_generation:
                        length = sim.segment_generation['lengths'].get(segment, 'variable')
                    else:
                        length = 'variable'
                        
                    # Get generation mode if available  
                    if sim.segment_generation and 'generation_mode' in sim.segment_generation:
                        mode = sim.segment_generation['generation_mode'].get(segment, 'unknown')
                    else:
                        mode = 'unknown'
                    
                    console.print(f"  {i:2}. {segment:10s} - length: {str(length):8s} mode: {mode}")
            
            # Show PWM configuration if present
            if sim.pwm:
                console.print(f"\n[cyan]PWM configuration:[/cyan]")
                console.print(f"  Temperature: {sim.pwm.temperature}")
                console.print(f"  Min entropy: {sim.pwm.min_entropy}")
                console.print(f"  Diversity boost: {sim.pwm.diversity_boost}")
                if sim.pwm.pattern:
                    console.print(f"  Pattern: {sim.pwm.pattern}")
        
        console.print()
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Configuration file not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Invalid configuration: {e}")
        raise typer.Exit(1)


@simulate_app.command("stats")
def stats_command(
    input_file: Path = typer.Argument(
        ...,
        help="Input file containing generated sequences",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed statistics per segment"
    )
):
    """
    Analyze statistics of generated sequences.
    
    This command analyzes a file of generated sequences and provides
    statistics about sequence lengths, segment distributions, and
    label frequencies.
    
    Examples:
        
        # Basic statistics
        tempest simulate stats sequences.txt
        
        # Detailed statistics per segment  
        tempest simulate stats sequences.txt --verbose
    """
    try:
        console.print(f"[cyan]Analyzing sequences from {input_file}...[/cyan]")
        
        # Read sequences
        sequences = []
        labels_list = []
        
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    seq, labels = parts
                    sequences.append(seq)
                    labels_list.append(labels.split())
        
        n_sequences = len(sequences)
        
        if n_sequences == 0:
            console.print("[red]No sequences found in file![/red]")
            raise typer.Exit(1)
        
        # Calculate statistics
        console.print(f"\n[bold]Statistics for {n_sequences} sequences:[/bold]")
        
        # Sequence length statistics
        lengths = [len(seq) for seq in sequences]
        console.print(f"\n[cyan]Sequence lengths:[/cyan]")
        console.print(f"  Mean: {np.mean(lengths):.1f} bp")
        console.print(f"  Std: {np.std(lengths):.1f} bp")
        console.print(f"  Min: {np.min(lengths)} bp")
        console.print(f"  Max: {np.max(lengths)} bp")
        console.print(f"  Median: {np.median(lengths):.1f} bp")
        
        # Label statistics
        if labels_list and labels_list[0]:
            # Count label frequencies
            label_counts = {}
            for labels in labels_list:
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            console.print(f"\n[cyan]Label distribution:[/cyan]")
            total_labels = sum(label_counts.values())
            
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                percentage = (count / total_labels) * 100
                console.print(f"  {label:10s}: {count:8d} ({percentage:5.2f}%)")
        
        if verbose:
            # Detailed segment analysis
            console.print(f"\n[cyan]Detailed segment analysis:[/cyan]")
            
            # Analyze segment transitions
            transitions = {}
            for labels in labels_list:
                for i in range(len(labels) - 1):
                    transition = f"{labels[i]} -> {labels[i+1]}"
                    transitions[transition] = transitions.get(transition, 0) + 1
            
            console.print("\nMost common segment transitions:")
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            for transition, count in sorted_transitions[:10]:
                console.print(f"  {transition:20s}: {count:6d}")
        
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# Backwards compatibility for direct module import
def simulate_command(args):
    """Legacy function for backwards compatibility."""
    logger.warning("Using deprecated simulate_command function, please use the Typer commands directly")
    
    # Convert args object to appropriate Typer command call
    if hasattr(args, 'split') and args.split:
        generate_command(
            config=Path(args.config),
            output_dir=Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else None,
            num_sequences=args.num_sequences if hasattr(args, 'num_sequences') else None,
            split=True,
            train_fraction=args.train_fraction if hasattr(args, 'train_fraction') else 0.8,
            seed=args.seed if hasattr(args, 'seed') else None,
            verbose=False
        )
    else:
        generate_command(
            config=Path(args.config),
            output=Path(args.output) if hasattr(args, 'output') and args.output else None,
            num_sequences=args.num_sequences if hasattr(args, 'num_sequences') else None,
            split=False,
            seed=args.seed if hasattr(args, 'seed') else None,
            verbose=False
        )
