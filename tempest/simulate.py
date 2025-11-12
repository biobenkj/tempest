"""
Tempest simulation module - refactored with pickle format support.

This module provides functionality for generating synthetic sequence reads
for training and testing TEMPEST models. Now saves to pickle format by default
for efficiency, with a preview text file for inspection.
"""

import typer
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import yaml
import json
import pickle
import gzip
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import the data simulator components
from tempest.data import (
    SequenceSimulator,
    SimulatedRead,
    InvalidReadGenerator,
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
    
    # Get format preference (default to pickle)
    output_format = kwargs.get('format', 'pickle')
    compress = kwargs.get('compress', True)
    
    # Determine output paths
    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    simulator = create_simulator_from_config(config)
    
    # Generate sequences based on configuration
    result = {}
    
    if kwargs.get('split', False):
        # Generate train/validation split
        logger.info(f"Generating {sim_config.num_sequences} sequences with {sim_config.train_split}/{1.0 - sim_config.train_split} train/val split")
        
        n_train = int(sim_config.num_sequences * sim_config.train_split)
        n_val = sim_config.num_sequences - n_train
        
        train_reads = simulator.generate_reads(n_train)
        val_reads = simulator.generate_reads(n_val)

        if getattr(sim_config, "invalid_fraction", 0.0) > 0.0:
            invalid_ratio = sim_config.invalid_fraction
            logger.info(
                f"Applying invalid read corruption to train/val with ratio={invalid_ratio:.3f}"
            )
            invalid_gen = InvalidReadGenerator(sim_config)

            for r in train_reads:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata.setdefault("is_invalid", False)

            for r in val_reads:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata.setdefault("is_invalid", False)

            train_reads = invalid_gen.generate_batch(train_reads, invalid_ratio=invalid_ratio)
            val_reads = invalid_gen.generate_batch(val_reads, invalid_ratio=invalid_ratio)
        
        # Save sequences based on format
        if output_format == 'pickle':
            train_file = output_path / ("train.pkl.gz" if compress else "train.pkl")
            val_file = output_path / ("val.pkl.gz" if compress else "val.pkl")
            train_preview = output_path / "train_preview.txt"
            val_preview = output_path / "val_preview.txt"
            
            save_stats_train = _save_reads_pickle(train_reads, train_file, train_preview, compress)
            save_stats_val = _save_reads_pickle(val_reads, val_file, val_preview, compress)
            
            result['save_stats'] = {
                'train': save_stats_train,
                'val': save_stats_val
            }
        else:
            train_file = output_path / "train.txt"
            val_file = output_path / "val.txt"
            _save_reads_text(train_reads, train_file)
            _save_reads_text(val_reads, val_file)
        
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

        # Generate invalid reads and log
        if getattr(sim_config, "invalid_fraction", 0.0) > 0.0:
            logger.info(
                f"Generating invalid reads (fraction={sim_config.invalid_fraction:.2f})"
            )
            invalid_gen = InvalidReadGenerator(sim_config)
            # Ensure metadata exists so generator can safely mark invalids
            for r in reads:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata.setdefault("is_invalid", False)

            reads = invalid_gen.generate_batch(reads, invalid_fraction=sim_config.invalid_fraction)

            # Combine valid + invalid reads
            logger.info(f"Total combined reads: {len(reads)} (valid + invalid)")
        
        # Determine output filename based on format
        if output_format == 'pickle':
            output_file = output_path / (kwargs.get('output_file', 'sequences.pkl.gz' if compress else 'sequences.pkl'))
            preview_file = output_path / kwargs.get('preview_file', 'sequences_preview.txt')
            save_stats = _save_reads_pickle(reads, output_file, preview_file, compress)
            result['save_stats'] = save_stats
        else:
            output_file = output_path / kwargs.get('output_file', 'sequences.txt')
            _save_reads_text(reads, output_file)
        
        result['output_file'] = str(output_file)
        result['n_sequences'] = sim_config.num_sequences
        
        logger.info(f"Saved {sim_config.num_sequences} sequences to {output_file}")
    
    # Add statistics
    result['config'] = config
    result['seed'] = sim_config.random_seed
    result['success'] = True
    result['format'] = output_format
    
    return result


def _save_reads_pickle(
    reads: List[SimulatedRead], 
    output_file: Path, 
    preview_file: Path,
    compress: bool = True
) -> Dict[str, Any]:
    """
    Save simulated reads to pickle format with preview text file.
    
    Args:
        reads: List of SimulatedRead objects
        output_file: Path to save pickle file
        preview_file: Path to save preview text file
        compress: Whether to compress the pickle file
        
    Returns:
        Dictionary with save statistics
    """
    import time
    start_time = time.time()
    
    # Save pickle file (compressed or uncompressed)
    if compress:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create preview text file with first 10 reads
    n_preview = min(10, len(reads))
    with open(preview_file, 'w') as f:
        f.write(f"# Preview of first {n_preview} sequences from {output_file.name}\n")
        f.write(f"# Total sequences: {len(reads)}\n")
        f.write(f"# Format: sequence<TAB>labels\n")
        f.write("#" + "="*70 + "\n\n")
        
        for i, read in enumerate(reads[:n_preview]):
            labels_str = ' '.join(read.labels)
            f.write(f"{read.sequence}\t{labels_str}\n")
            
            # Add metadata as comment if present
            if read.metadata:
                f.write(f"# Metadata: {json.dumps(read.metadata)}\n")
            
            # Add label regions summary
            if read.label_regions:
                regions_summary = ", ".join([f"{k}:{v}" for k, v in read.label_regions.items()])
                f.write(f"# Regions: {regions_summary}\n")
            
            f.write("\n")
    
    # Calculate statistics
    save_time = time.time() - start_time
    file_size = output_file.stat().st_size
    
    stats = {
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'save_time': save_time,
        'sequences_per_second': len(reads) / save_time if save_time > 0 else 0,
        'compressed': compress,
        'preview_file': str(preview_file),
        'n_preview': n_preview
    }
    
    logger.info(f"Saved {len(reads)} sequences in {save_time:.2f}s ({stats['file_size_mb']:.2f} MB)")
    logger.info(f"Preview saved to: {preview_file}")
    
    return stats


def _save_reads_text(reads: List[SimulatedRead], output_file: Path):
    """
    Save simulated reads to text file (legacy format).
    
    Args:
        reads: List of SimulatedRead objects
        output_file: Path to save file
    """
    with open(output_file, 'w') as f:
        for read in reads:
            # Save in the format: sequence<TAB>labels
            labels_str = ' '.join(read.labels)
            f.write(f"{read.sequence}\t{labels_str}\n")


def _load_reads_pickle(input_file: Path) -> List[SimulatedRead]:
    """
    Load simulated reads from pickle format.
    
    Args:
        input_file: Path to pickle file
        
    Returns:
        List of SimulatedRead objects
    """
    if input_file.suffix == '.gz' or '.gz' in input_file.suffixes:
        with gzip.open(input_file, 'rb') as f:
            return pickle.load(f)
    else:
        with open(input_file, 'rb') as f:
            return pickle.load(f)


def _load_reads_text(input_file: Path) -> List[SimulatedRead]:
    """
    Load simulated reads from text format.
    
    Args:
        input_file: Path to text file
        
    Returns:
        List of SimulatedRead objects
    """
    reads = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                sequence, labels_str = parts
                labels = labels_str.split()
                reads.append(SimulatedRead(
                    sequence=sequence,
                    labels=labels,
                    label_regions={},  # Would need to reconstruct
                    metadata={}
                ))
    return reads


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
    format: str = typer.Option(
        "pickle",
        "--format", "-f",
        help="Output format: 'pickle' (default, efficient) or 'text' (human-readable)"
    ),
    no_compress: bool = typer.Option(
        False,
        "--no-compress",
        help="Don't compress pickle files (saves faster but uses more disk)"
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
    defined in the configuration file. By default, sequences are saved in
    compressed pickle format for efficiency, with a preview text file for
    inspection.
    
    Examples:
        
        # Generate 1000 sequences in pickle format (default)
        tempest simulate generate -c config.yaml -n 1000
        
        # Generate in text format for compatibility
        tempest simulate generate -c config.yaml -n 1000 --format text
        
        # Generate train/validation split
        tempest simulate generate -c config.yaml --split -d ./data -t 0.8
        
        # Generate without compression for faster I/O
        tempest simulate generate -c config.yaml --no-compress
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
        kwargs = {
            'format': format.lower(),
            'compress': not no_compress
        }
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
            console.print(f"  Output format: {format} {'(compressed)' if not no_compress else '(uncompressed)'}")
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
            
            if 'save_stats' in result and verbose:
                console.print("\n[cyan]Save Statistics:[/cyan]")
                for dataset, stats in result['save_stats'].items():
                    console.print(f"  {dataset}:")
                    console.print(f"    File size: {stats['file_size_mb']:.2f} MB")
                    console.print(f"    Save time: {stats['save_time']:.2f}s")
                    console.print(f"    Preview: {stats['preview_file']}")
        else:
            console.print(f"  Total sequences: {result['n_sequences']}")
            console.print(f"  Output file: [green]{result['output_file']}[/green]")
            
            if 'save_stats' in result:
                stats = result['save_stats']
                console.print(f"  File size: {stats['file_size_mb']:.2f} MB")
                if format == 'pickle':
                    console.print(f"  Preview file: [green]{stats['preview_file']}[/green]")
                
                if verbose:
                    console.print(f"\n[dim]Performance:[/dim]")
                    console.print(f"  Save time: {stats['save_time']:.2f}s")
                    console.print(f"  Throughput: {stats['sequences_per_second']:.0f} seq/s")
        
        if verbose:
            console.print(f"\n[dim]Random seed used: {result['seed']}[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@simulate_app.command("convert")
def convert_command(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to convert",
        exists=True,
        file_okay=True,
        readable=True
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output file path"
    ),
    input_format: Optional[str] = typer.Option(
        None,
        "--input-format", "-i",
        help="Input format (auto-detected if not specified)"
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output-format", "-o",
        help="Output format (auto-detected from extension if not specified)"
    ),
    compress: bool = typer.Option(
        True,
        "--compress/--no-compress",
        help="Compress output if using pickle format"
    )
):
    """
    Convert between sequence file formats.
    
    Examples:
        # Convert text to pickle
        tempest simulate convert sequences.txt sequences.pkl.gz
        
        # Convert pickle to text  
        tempest simulate convert sequences.pkl.gz sequences.txt
    """
    try:
        console.print(f"[cyan]Converting {input_file} to {output_file}...[/cyan]")
        
        # Auto-detect formats if not specified
        if input_format is None:
            if '.pkl' in str(input_file) or '.pickle' in str(input_file):
                input_format = 'pickle'
            else:
                input_format = 'text'
        
        if output_format is None:
            if '.pkl' in str(output_file) or '.pickle' in str(output_file):
                output_format = 'pickle'
            else:
                output_format = 'text'
        
        # Load sequences
        if input_format == 'pickle':
            reads = _load_reads_pickle(input_file)
        else:
            reads = _load_reads_text(input_file)
        
        console.print(f"  Loaded {len(reads)} sequences")
        
        # Save in new format
        if output_format == 'pickle':
            preview_file = output_file.parent / f"{output_file.stem}_preview.txt"
            stats = _save_reads_pickle(reads, output_file, preview_file, compress)
            console.print(f"  Saved to: [green]{output_file}[/green]")
            console.print(f"  Preview: [green]{preview_file}[/green]")
            console.print(f"  Size: {stats['file_size_mb']:.2f} MB")
        else:
            _save_reads_text(reads, output_file)
            console.print(f"  Saved to: [green]{output_file}[/green]")
        
        console.print("[bold green]✓ Conversion complete![/bold green]")
        
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
        
        # Check if invalid raction is non-existent or the majority
        if sim.invalid_fraction < 0 or sim.invalid_fraction > 0.5:
            warnings.append(
                f"invalid_fraction={sim.invalid_fraction} is outside the recommended [0, 0.5] range"
                )
        
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
                table = Table(title="Sequence Architecture")
                table.add_column("#", style="cyan")
                table.add_column("Segment", style="green")
                table.add_column("Length", style="yellow")
                table.add_column("Mode", style="magenta")
                
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
                    
                    table.add_row(str(i), segment, str(length), mode)
                
                console.print(table)
            
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
    format: Optional[str] = typer.Option(
        None,
        "--format", "-f",
        help="File format (auto-detected if not specified)"
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
    label frequencies. Supports both pickle and text formats.
    
    Examples:
        
        # Basic statistics from pickle file
        tempest simulate stats sequences.pkl.gz
        
        # Detailed statistics from text file
        tempest simulate stats sequences.txt --verbose
    """
    try:
        console.print(f"[cyan]Analyzing sequences from {input_file}...[/cyan]")
        
        # Auto-detect format if not specified
        if format is None:
            if '.pkl' in str(input_file) or '.pickle' in str(input_file):
                format = 'pickle'
            else:
                format = 'text'
        
        # Load sequences
        if format == 'pickle':
            reads = _load_reads_pickle(input_file)
            sequences = [r.sequence for r in reads]
            labels_list = [r.labels for r in reads]
        else:
            sequences = []
            labels_list = []
            
            with open(input_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
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
            
            # Create a table for better display
            table = Table(title="Label Distribution")
            table.add_column("Label", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Percentage", style="yellow")
            
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                percentage = (count / total_labels) * 100
                table.add_row(label, f"{count:,}", f"{percentage:.2f}%")
            
            console.print(table)
        
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
            
            trans_table = Table(title="Segment Transitions")
            trans_table.add_column("Transition", style="cyan")
            trans_table.add_column("Count", style="green")
            
            for transition, count in sorted_transitions[:10]:
                trans_table.add_row(transition, f"{count:,}")
            
            console.print(trans_table)
            
            # Additional analysis for pickle format
            if format == 'pickle' and isinstance(reads[0], SimulatedRead):
                # Analyze metadata if present
                metadata_keys = set()
                for read in reads[:100]:  # Sample first 100
                    if read.metadata:
                        metadata_keys.update(read.metadata.keys())
                
                if metadata_keys:
                    console.print(f"\n[cyan]Metadata fields:[/cyan] {', '.join(metadata_keys)}")
                
                # Analyze label regions
                if reads[0].label_regions:
                    console.print(f"\n[cyan]Label regions present:[/cyan] {', '.join(reads[0].label_regions.keys())}")
        
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
            format='pickle',  # Default to pickle
            no_compress=False,
            verbose=False
        )
    else:
        generate_command(
            config=Path(args.config),
            output=Path(args.output) if hasattr(args, 'output') and args.output else None,
            num_sequences=args.num_sequences if hasattr(args, 'num_sequences') else None,
            split=False,
            seed=args.seed if hasattr(args, 'seed') else None,
            format='pickle',  # Default to pickle
            no_compress=False,
            verbose=False
        )
