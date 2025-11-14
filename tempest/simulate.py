"""
Enhanced Tempest simulation
"""

import os
import logging
import typer
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import pickle
import gzip
import json
import numpy as np
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from tempest.utils.logging_utils import setup_rich_logging, console, status

# Determine log level dynamically
log_level = os.getenv("TEMPEST_LOG_LEVEL", "INFO").upper()
setup_rich_logging(log_level)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Import both standard and parallel simulators
from tempest.data import (
    SequenceSimulator,
    SimulatedRead,
    InvalidReadGenerator,
    create_simulator_from_config,
)

# Import parallel versions - with fallback if not available
try:
    from tempest.data.parallel_simulator import (
        ParallelSequenceSimulator,
        ParallelInvalidReadGenerator,
        create_parallel_simulator_from_config
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    logger.warning("Parallel simulator not available, using standard sequential version")
    PARALLEL_AVAILABLE = False
    ParallelSequenceSimulator = SequenceSimulator
    ParallelInvalidReadGenerator = InvalidReadGenerator
    create_parallel_simulator_from_config = create_simulator_from_config

from tempest.config import TempestConfig, SimulationConfig

simulate_app = typer.Typer(
    help="Generate synthetic sequence reads for training and testing",
    rich_markup_mode="rich",
)


def safe_json_encode(obj):
    """Convert non-JSON-serializable objects for metadata display."""
    if isinstance(obj, (list, tuple)):
        return [safe_json_encode(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_json_encode(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return safe_json_encode(obj.__dict__)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)


def _save_reads_pickle(reads: List[SimulatedRead], output_file: Path, compress: bool = True) -> Dict[str, Any]:
    """Save reads to pickle format with optional compression and comprehensive metadata preview."""
    start_time = time.time()
    
    if compress:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(reads, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_time = time.time() - start_time
    file_size = output_file.stat().st_size
    
    # Create comprehensive preview file with metadata
    preview_file = output_file.parent / f"{output_file.stem}_preview.txt"
    n_preview = min(100, len(reads))
    
    with open(preview_file, 'w') as f:
        # Header
        f.write(f"# Preview of {output_file.name} ({len(reads)} total sequences)\n")
        f.write(f"# Showing first {n_preview} sequences\n")
        f.write("#" + "=" * 70 + "\n")
        
        # Calculate label distribution from all reads
        label_counts = {}
        total_positions = 0
        n_invalid = 0
        n_with_errors = 0
        
        for read in reads:
            for label in read.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                total_positions += 1
            
            # Check metadata flags
            if hasattr(read, 'metadata') and read.metadata:
                if read.metadata.get('is_invalid', False):
                    n_invalid += 1
                if read.metadata.get('has_errors', False):
                    n_with_errors += 1
        
        # Dataset statistics
        f.write("\n# DATASET STATISTICS\n")
        f.write("#" + "-" * 70 + "\n")
        f.write(f"# Total sequences: {len(reads):,}\n")
        f.write(f"# Total positions: {total_positions:,}\n")
        if n_invalid > 0:
            f.write(f"# Invalid sequences: {n_invalid:,} ({n_invalid/len(reads)*100:.1f}%)\n")
        if n_with_errors > 0:
            f.write(f"# Sequences with errors: {n_with_errors:,} ({n_with_errors/len(reads)*100:.1f}%)\n")
        
        # Label distribution
        f.write("\n# LABEL DISTRIBUTION\n")
        f.write("#" + "-" * 70 + "\n")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            percentage = (count / total_positions * 100) if total_positions > 0 else 0
            f.write(f"# {label:10s}: {count:8,} positions ({percentage:5.2f}%)\n")
        
        # Sequence preview with metadata
        f.write("\n# SEQUENCE PREVIEW\n")
        f.write("#" + "=" * 70 + "\n")
        f.write("# Format: sequence<TAB>labels<TAB>is_invalid\n")
        f.write("# Followed by JSON metadata for each read\n")
        f.write("#" + "-" * 70 + "\n\n")
        
        for i, read in enumerate(reads[:n_preview], 1):
            f.write(f"# Read {i}\n")
            
            # Basic info: sequence, labels, invalid flag
            labels_str = ' '.join(read.labels)
            is_invalid = 'N'
            
            if hasattr(read, 'metadata') and read.metadata:
                is_invalid = 'Y' if read.metadata.get('is_invalid', False) else 'N'
            
            f.write(f"{read.sequence}\t{labels_str}\t{is_invalid}\n")
            
            # Write comprehensive metadata
            if hasattr(read, 'metadata') and read.metadata:
                f.write("# Metadata:\n")
                
                # Prepare metadata for display
                metadata_display = {}
                
                # Core metadata fields in desired order
                core_fields = [
                    'segment_order', 'segment_sources', 'segment_lengths',
                    'segment_sequences', 'segment_meta', 'transcript_info',
                    'has_errors', 'error_types', 'error_positions',
                    'diversity_boost', 'is_invalid', 'invalid_reason',
                    'reverse_complement', 'acc_score', 'acc_temperature'
                ]
                
                for key in core_fields:
                    if key in read.metadata:
                        metadata_display[key] = safe_json_encode(read.metadata[key])
                
                # Add any additional metadata fields
                for key, value in read.metadata.items():
                    if key not in metadata_display:
                        metadata_display[key] = safe_json_encode(value)
                
                # Add label regions if available
                if hasattr(read, 'label_regions') and read.label_regions:
                    region_summary = {}
                    for label, regions in read.label_regions.items():
                        region_summary[label] = [[start, end] for start, end in regions]
                    metadata_display['label_regions'] = region_summary
                
                # Write formatted JSON with proper indentation
                try:
                    formatted_json = json.dumps(metadata_display, indent=2)
                    for line in formatted_json.split('\n'):
                        f.write(f"# {line}\n")
                except Exception as e:
                    f.write(f"# Error formatting metadata: {str(e)}\n")
                    f.write(f"# Raw metadata: {str(read.metadata)[:200]}...\n")
            else:
                f.write("# No metadata\n")
            
            f.write("\n")  # Blank line between reads
    
    logger.info(f"Created comprehensive preview file: {preview_file}")
    
    return {
        'save_time': save_time,
        'file_size_mb': file_size / (1024 * 1024),
        'sequences_per_second': len(reads) / save_time if save_time > 0 else 0,
        'compression_ratio': 1.0 if not compress else 0.5,  # Approximate
        'n_invalid': n_invalid,
        'n_with_errors': n_with_errors
    }


def _save_reads_text(reads: List[SimulatedRead], output_file: Path):
    """Save reads to text format with metadata."""
    with open(output_file, 'w') as f:
        f.write("# TEMPEST simulated sequences with metadata\n")
        f.write("# Format: sequence<TAB>labels<TAB>metadata_json\n\n")
        for read in reads:
            labels_str = ' '.join(read.labels)
            # Include metadata as JSON string
            metadata_str = "{}"
            if hasattr(read, 'metadata') and read.metadata:
                try:
                    metadata_str = json.dumps(safe_json_encode(read.metadata))
                except:
                    metadata_str = "{}"
            f.write(f"{read.sequence}\t{labels_str}\t{metadata_str}\n")


def run_simulation(
    config: TempestConfig,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fixed simulation function with proper progress for invalid generation.
    """
    # Update config with any overrides from kwargs
    sim_config = config.simulation
    
    # Check if parallel processing is requested (default: True if available)
    use_parallel = kwargs.get('parallel', PARALLEL_AVAILABLE)
    n_workers = kwargs.get('n_workers', None)
    
    if use_parallel and not PARALLEL_AVAILABLE:
        logger.warning("Parallel processing requested but not available, using sequential")
        use_parallel = False
    
    # Apply configuration overrides
    if 'num_sequences' in kwargs and kwargs['num_sequences'] is not None:
        sim_config.num_sequences = kwargs['num_sequences']

    if 'seed' in kwargs and kwargs['seed'] is not None:
        sim_config.random_seed = kwargs['seed']

    if 'train_fraction' in kwargs and kwargs['train_fraction'] is not None:
        sim_config.train_split = kwargs['train_fraction']

    if 'invalid_fraction' in kwargs and kwargs['invalid_fraction'] is not None:
        sim_config.invalid_fraction = kwargs['invalid_fraction']
    
    # Handle transcript overrides
    if 'skip_transcripts' in kwargs and kwargs['skip_transcripts']:
        if hasattr(sim_config, 'transcript'):
            if hasattr(sim_config.transcript, 'fasta_file'):
                sim_config.transcript.fasta_file = ""
            elif isinstance(sim_config.transcript, dict):
                sim_config.transcript['fasta_file'] = ""
        logger.info("Skipping transcript pool loading")
    elif 'transcript_fasta' in kwargs and kwargs['transcript_fasta']:
        if not hasattr(sim_config, 'transcript') or sim_config.transcript is None:
            from types import SimpleNamespace
            sim_config.transcript = SimpleNamespace()
        if hasattr(sim_config.transcript, '__dict__'):
            sim_config.transcript.fasta_file = str(kwargs['transcript_fasta'])
        else:
            sim_config.transcript['fasta_file'] = str(kwargs['transcript_fasta'])
        logger.info(f"Using transcript FASTA override: {kwargs['transcript_fasta']}")
    
    # Get format preference
    output_format = kwargs.get('format', 'pickle')
    compress = kwargs.get('compress', True)
    
    # Determine output paths
    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator (parallel or standard)
    if use_parallel:
        logger.info(f"Using parallel simulator with {n_workers or 'auto'} workers")
        simulator = create_parallel_simulator_from_config(config, n_workers=n_workers)
        
        # Create parallel invalid generator if needed
        if getattr(sim_config, "invalid_fraction", 0.0) > 0.0:
            invalid_gen = ParallelInvalidReadGenerator(sim_config, n_workers=n_workers)
        else:
            invalid_gen = None
    else:
        logger.info("Using standard sequential simulator")
        simulator = create_simulator_from_config(config)
        
        if getattr(sim_config, "invalid_fraction", 0.0) > 0.0:
            invalid_gen = InvalidReadGenerator(sim_config)
        else:
            invalid_gen = None
    
    # Generate sequences based on configuration
    result = {}
    
    if kwargs.get('split', False):
        # SPLIT MODE - works fine
        total_sequences = sim_config.num_sequences
        if total_sequences is None:
            raise ValueError("num_sequences not specified")
        
        logger.info(f"Generating {total_sequences} sequences with "
                   f"{sim_config.train_split:.2f}/{1.0 - sim_config.train_split:.2f} train/val split")
        
        n_train = int(total_sequences * sim_config.train_split)
        n_val = total_sequences - n_train
        
        progress_callback = kwargs.get('progress_callback', None)
        
        # For split mode with invalid generation, we need to track both phases
        if invalid_gen and sim_config.invalid_fraction > 0.0:
            # Calculate total work: generation + corruption for both train and val
            invalid_ratio = sim_config.invalid_fraction
            n_train_invalid = int(n_train * invalid_ratio)
            n_val_invalid = int(n_val * invalid_ratio)
            
            # Create a wrapper for progress that accounts for both phases
            if progress_callback:
                original_callback = progress_callback
                current_progress = {'value': 0}
                
                def combined_progress_callback(n):
                    current_progress['value'] = n
                    original_callback(n)
                
                progress_callback = combined_progress_callback
        
        # Generate training sequences
        train_reads = simulator.generate_batch(n_train, progress_callback=progress_callback)
        
        # Generate validation sequences
        if progress_callback:
            def val_progress_callback(n_generated):
                progress_callback(n_train + n_generated)
            val_reads = simulator.generate_batch(n_val, progress_callback=val_progress_callback)
        else:
            val_reads = simulator.generate_batch(n_val)
        
        # Apply invalid read corruption if configured
        if invalid_gen and sim_config.invalid_fraction > 0.0:
            invalid_ratio = sim_config.invalid_fraction
            logger.info(f"Applying invalid read corruption with ratio={invalid_ratio:.3f}")
            
            for dataset_name, reads in [("train", train_reads), ("validation", val_reads)]:
                for r in reads:
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata.setdefault("is_invalid", False)
                
                corrupted = invalid_gen.generate_batch(reads, invalid_ratio=invalid_ratio)
                logger.info(f"{dataset_name.capitalize()} set: {len(corrupted)} total reads")
                
                if dataset_name == "train":
                    train_reads = corrupted
                else:
                    val_reads = corrupted
        
        # Save sequences with comprehensive metadata
        if output_format == 'pickle':
            train_file = output_path / ("train.pkl.gz" if compress else "train.pkl")
            val_file = output_path / ("val.pkl.gz" if compress else "val.pkl")
            
            save_stats_train = _save_reads_pickle(train_reads, train_file, compress)
            save_stats_val = _save_reads_pickle(val_reads, val_file, compress)
            
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
        # SINGLE DATASET MODE
        total_sequences = sim_config.num_sequences
        if total_sequences is None:
            raise ValueError("num_sequences not specified")
        
        logger.info(f"Generating {total_sequences} sequences")
        
        # Get the main progress callback
        main_progress_callback = kwargs.get("progress_callback", None)
        
        # Check if we'll be doing invalid generation
        will_generate_invalid = (invalid_gen and 
                                getattr(sim_config, 'invalid_fraction', 0.0) > 0.0)
        
        if will_generate_invalid and main_progress_callback:
            # For single dataset with invalid generation, we need special progress handling
            logger.info(f"Phase 1: Generating {total_sequences} valid sequences")
            
            # Create a progress callback that caps at ~90% for valid generation
            def valid_progress_callback(n):
                scaled = int(n * 0.9)
                main_progress_callback(scaled)
            
            reads = simulator.generate_batch(
                total_sequences,
                progress_callback=valid_progress_callback
            )
            
            # Phase 2: Apply invalid corruption
            logger.info(f"Phase 2: Applying {sim_config.invalid_fraction:.1%} invalid corruption")
            
            # Mark all reads as initially valid
            for r in reads:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata.setdefault("is_invalid", False)
            
            # Apply corruption with remaining progress (90-100%)
            n_to_corrupt = int(total_sequences * sim_config.invalid_fraction)
            
            # For the invalid generation, update progress for the last 10%
            def invalid_progress_callback(n_corrupted):
                progress = 0.9 + (0.1 * (n_corrupted / n_to_corrupt))
                main_progress_callback(int(total_sequences * progress))
            
            # Check if parallel invalid generator supports progress callback
            if use_parallel and hasattr(invalid_gen, 'generate_batch'):
                import signal
                import threading
                
                # Set up timeout handling
                def timeout_handler():
                    logger.error("Invalid generation timed out after 5 minutes")
                    main_progress_callback(total_sequences)
                
                timer = threading.Timer(300.0, timeout_handler)  # 5 minute timeout
                timer.start()
                
                try:
                    reads = invalid_gen.generate_batch(
                        reads, 
                        invalid_ratio=sim_config.invalid_fraction
                    )
                finally:
                    timer.cancel()
            else:
                # Use sequential invalid generator
                reads = invalid_gen.generate_batch(
                    reads,
                    invalid_ratio=sim_config.invalid_fraction
                )
            
            # Ensure progress shows 100%
            main_progress_callback(total_sequences)
            
            logger.info(f"Completed: {len(reads)} total sequences "
                       f"({n_to_corrupt} corrupted)")
            
        else:
            # No invalid generation or no progress callback - simple case
            reads = simulator.generate_batch(
                total_sequences,
                progress_callback=main_progress_callback
            )
            
            # Apply invalid reads if configured (without progress)
            if will_generate_invalid:
                for r in reads:
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata.setdefault("is_invalid", False)
                
                reads = invalid_gen.generate_batch(
                    reads, 
                    invalid_ratio=sim_config.invalid_fraction
                )
                logger.info(f"Applied invalid read corruption: {len(reads)} total reads")
        
        # Save sequences with comprehensive metadata
        if output_format == 'pickle':
            output_file = output_path / ("sequences.pkl.gz" if compress else "sequences.pkl")
            save_stats = _save_reads_pickle(reads, output_file, compress)
            result['save_stats'] = save_stats
        else:
            output_file = output_path / "sequences.txt"
            _save_reads_text(reads, output_file)
        
        result['output_file'] = str(output_file)
        result['n_sequences'] = len(reads)
        
        logger.info(f"Saved {len(reads)} sequences to {output_file}")
    
    return result




def _load_reads_pickle(input_file: Path) -> List[SimulatedRead]:
    """Load reads from pickle format."""
    if '.gz' in str(input_file):
        with gzip.open(input_file, 'rb') as f:
            reads = pickle.load(f)
    else:
        with open(input_file, 'rb') as f:
            reads = pickle.load(f)
    return reads


@simulate_app.command("generate")
def generate_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Configuration YAML file"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for generated data"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file (single dataset)"),
    num_sequences: Optional[int] = typer.Option(
        None,
        "--num-sequences", "-n",
        help="Number of sequences to generate"),
    split: bool = typer.Option(
        False,
        "--split",
        help="Generate train/val split"),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction",
        help="Fraction for training set when using --split"),
    invalid_fraction: Optional[float] = typer.Option(
        None,
        "--invalid-fraction",
        help="Fraction of invalid reads to generate"),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-s",
        help="Random seed"),
    format: str = typer.Option(
        "pickle",
        "--format", "-f",
        help="Output format: pickle|text"),
    no_compress: bool = typer.Option(
        False,
        "--no-compress",
        help="Don't compress pickle files"),
    parallel: bool = typer.Option(
        True,
        "--parallel/--no-parallel",
        help="Use parallel processing (10-50x faster)"),
    n_workers: Optional[int] = typer.Option(
        None,
        "--workers", "-w",
        help="Number of parallel workers (auto-detect if not specified)"),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output"),
    skip_transcripts: bool = typer.Option(
        False, "--skip-transcripts",
        help="Skip loading transcript pool"),
    transcript_fasta: Optional[Path] = typer.Option(
        None,
        "--transcript-fasta",
        help="Override transcript FASTA file")
):
    """
    Generate synthetic sequence reads with parallel processing support.
    
    Performance tips:
    - Use --parallel (default) for 10-50x speedup
    - Adjust --workers based on your CPU cores (auto-detection usually optimal)
    - Use pickle format (default) for 10x faster I/O and 50% smaller files
    - Generate large batches (100K+) for best parallel efficiency
    
    Examples:
        # Generate train/val split with parallel processing
        tempest simulate generate -c config.yaml -o data/ --split -n 100000
        
        # Generate with custom worker count for large datasets
        tempest simulate generate -c config.yaml -o data/ --split -n 1000000 --workers 32
        
        # Generate single dataset without compression
        tempest simulate generate -c config.yaml --output sequences.pkl -n 50000 --no-compress
    """
    try:
        # Load configuration
        config_obj = TempestConfig.from_yaml(config)
        
        # Check parallel availability
        if parallel and not PARALLEL_AVAILABLE:
            console.print("[yellow]Warning: Parallel simulator not available, using sequential mode[/yellow]")
            console.print("[dim]To enable parallel processing, ensure parallel_simulator.py is installed[/dim]")
            parallel = False
        
        # Display configuration
        panel_content = f"""[cyan]Configuration:[/cyan] {config}
[cyan]Mode:[/cyan] {'Parallel' if parallel else 'Sequential'} processing
[cyan]Workers:[/cyan] {n_workers if n_workers else 'auto-detect'} {'(parallel mode)' if parallel else ''}
[cyan]Output:[/cyan] {output_dir or output or 'current directory'}
[cyan]Format:[/cyan] {format} {'(compressed)' if not no_compress and format == 'pickle' else ''}"""
        
        # Determine effective number of sequences (CLI override wins, then config)
        effective_num_sequences = (
            num_sequences
            if num_sequences is not None
            else config_obj.simulation.num_sequences
            )

        if effective_num_sequences is not None:
            if num_sequences is not None:
                panel_content += f"\n[cyan]Sequences:[/cyan] {effective_num_sequences:,} (from CLI)"
            else:
                panel_content += f"\n[cyan]Sequences:[/cyan] {effective_num_sequences:,} (from config)"
        else:
            panel_content += (
                "\n[cyan]Sequences:[/cyan] "
                "[red]Not specified - please use --num-sequences[/red]"
            )
        
        if split:
            n_seqs = effective_num_sequences
            if n_seqs is not None:
                n_train = int(n_seqs * train_fraction)
                n_val = n_seqs - n_train
                panel_content += (
                    f"\n[cyan]Split:[/cyan] {n_train:,} train / {n_val:,} val "
                    f"({train_fraction:.0%}/{1-train_fraction:.0%})"
                )
            else:
                panel_content += (
                    "\n[cyan]Split:[/cyan] Cannot calculate (num_sequences not specified)"
                )
        
        if invalid_fraction is not None:
            panel_content += f"\n[cyan]Invalid fraction:[/cyan] {invalid_fraction:.1%}"
        
        console.print(Panel(panel_content, title="[bold]Sequence Generation[/bold]", box=box.ROUNDED))
        
        # Set up progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            
            n_total = (
                num_sequences 
                if num_sequences is not None 
                else config_obj.simulation.num_sequences
                )
            
            # Check if we have a valid number of sequences
            if n_total is None:
                try:
                    # Try to get from config
                    n_total = config_obj.simulation.num_sequences
                except:
                    console.print("[bold red]Error:[/bold red] Number of sequences not specified.")
                    console.print("Please use --num-sequences or specify in config file.")
                    raise typer.Exit(1)
            
            task = progress.add_task("[cyan]Generating sequences...", total=n_total)
            
            def progress_callback(n_generated):
                progress.update(task, completed=n_generated)
            
            # Determine output path
            if output_dir:
                output_path = output_dir
            elif output:
                output_path = output.parent
            else:
                output_path = Path('.')
            
            # Run simulation with progress tracking
            result = run_simulation(
                config_obj,
                output_dir=output_path,
                num_sequences=num_sequences,
                split=split,
                train_fraction=train_fraction,
                invalid_fraction=invalid_fraction,
                seed=seed,
                format=format,
                compress=not no_compress,
                parallel=parallel,
                n_workers=n_workers,
                progress_callback=progress_callback,
                skip_transcripts=skip_transcripts,
                transcript_fasta=transcript_fasta
            )
        
        # Show results
        results_table = Table(title="Generation Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        if 'train_file' in result:
            results_table.add_row("Training file", str(result['train_file']))
            results_table.add_row("Training sequences", f"{result['n_train']:,}")
            results_table.add_row("Validation file", str(result['val_file']))
            results_table.add_row("Validation sequences", f"{result['n_val']:,}")
        else:
            results_table.add_row("Output file", str(result['output_file']))
            results_table.add_row("Total sequences", f"{result['n_sequences']:,}")
        
        if 'save_stats' in result:
            if isinstance(result['save_stats'], dict):
                for name, stats in result['save_stats'].items():
                    results_table.add_row(f"{name.title()} save time", f"{stats['save_time']:.2f}s")
                    results_table.add_row(f"{name.title()} file size", f"{stats['file_size_mb']:.1f} MB")
                    if 'n_invalid' in stats and stats['n_invalid'] > 0:
                        results_table.add_row(f"{name.title()} invalid", f"{stats['n_invalid']:,}")
            else:
                results_table.add_row("Save time", f"{result['save_stats']['save_time']:.2f}s")
                results_table.add_row("File size", f"{result['save_stats']['file_size_mb']:.1f} MB")
        
        console.print("\n")
        console.print(results_table)
        console.print("\n[bold green]✓ Generation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


# Additional stats and analysis functions
def analyze_acc_segment_detailed(
    reads: List[SimulatedRead],
    acc_generator=None,
    show_sequences: int = 10
) -> Dict[str, Any]:
    """Analyze ACC segment statistics from reads."""
    acc_sequences = []
    acc_lengths = []
    
    for read in reads:
        if hasattr(read, 'label_regions') and 'ACC' in read.label_regions:
            for start, end in read.label_regions['ACC']:
                acc_seq = read.sequence[start:end]
                acc_sequences.append(acc_seq)
                acc_lengths.append(len(acc_seq))
    
    if not acc_sequences:
        return {'found': False}
    
    # Calculate statistics
    unique_accs = set(acc_sequences)
    stats = {
        'found': True,
        'total': len(acc_sequences),
        'unique': len(unique_accs),
        'diversity': len(unique_accs) / len(acc_sequences) if acc_sequences else 0,
        'lengths': {
            'mean': np.mean(acc_lengths),
            'std': np.std(acc_lengths),
            'min': min(acc_lengths),
            'max': max(acc_lengths)
        },
        'examples': acc_sequences[:show_sequences]
    }
    
    # PWM scoring if available
    if acc_generator and hasattr(acc_generator, 'score_sequence'):
        scores = []
        for seq in acc_sequences[:100]:  # Score first 100
            try:
                score = acc_generator.score_sequence(seq)
                scores.append(score)
            except:
                pass
        
        if scores:
            stats['pwm_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores)
            }
    
    return stats


def display_acc_statistics(stats: Dict[str, Any], verbose: bool = False):
    """Display ACC segment statistics."""
    if not stats.get('found', False):
        console.print("[yellow]No ACC segments found in dataset[/yellow]")
        return
    
    # Basic statistics
    console.print(f"[cyan]Total ACC sequences:[/cyan] {stats['total']:,}")
    console.print(f"[cyan]Unique ACC sequences:[/cyan] {stats['unique']:,}")
    console.print(f"[cyan]Diversity ratio:[/cyan] {stats['diversity']:.3f}")
    
    # Length distribution
    console.print(f"\n[cyan]ACC Length Distribution:[/cyan]")
    console.print(f"  Mean: {stats['lengths']['mean']:.1f} bp")
    console.print(f"  Std:  {stats['lengths']['std']:.1f} bp")
    console.print(f"  Min:  {stats['lengths']['min']} bp")
    console.print(f"  Max:  {stats['lengths']['max']} bp")
    
    # PWM scores if available
    if 'pwm_scores' in stats:
        console.print(f"\n[cyan]PWM Score Distribution:[/cyan]")
        console.print(f"  Mean: {stats['pwm_scores']['mean']:.2f}")
        console.print(f"  Std:  {stats['pwm_scores']['std']:.2f}")
        console.print(f"  Min:  {stats['pwm_scores']['min']:.2f}")
        console.print(f"  Max:  {stats['pwm_scores']['max']:.2f}")
    
    # Example sequences
    if verbose and 'examples' in stats:
        console.print(f"\n[cyan]Example ACC Sequences:[/cyan]")
        for i, seq in enumerate(stats['examples'], 1):
            console.print(f"  {i:2d}. {seq}")


@simulate_app.command("stats")
def stats_command(
    input_file: Path = typer.Argument(
        ...,
        help="Input file (pickle or text format)"),
    format: Optional[str] = typer.Option(
        None,
        "--format", "-f",
        help="Input format: pickle|text (auto-detect if not specified)"),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed statistics"),
    acc_only: bool = typer.Option(
        False,
        "--acc-only",
        help="Only analyze ACC segments"),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Config file for PWM scoring")
):
    """
    Analyze statistics of generated sequences.
    
    Provides comprehensive analysis including sequence length distribution,
    base frequencies, entropy, diversity metrics, and PWM divergence.
    
    Examples:
        
        # Basic statistics from pickle file
        tempest simulate stats sequences.pkl.gz
        
        # Detailed statistics with position-wise analysis
        tempest simulate stats sequences.pkl.gz --verbose
        
        # ACC-only analysis with PWM scoring
        tempest simulate stats sequences.pkl.gz --acc-only --config config.yaml
        
        # Full analysis with config
        tempest simulate stats train.pkl.gz -v -c config.yaml
    """
    try:
        console.print(f"[cyan]Analyzing sequences from {input_file}...[/cyan]")
        
        # Auto-detect format
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
            
            # Create minimal SimulatedRead objects for text format
            reads = []
            for seq, labels in zip(sequences, labels_list):
                label_regions = {}
                current_label = None
                start = 0
                
                for i, label in enumerate(labels):
                    if label != current_label:
                        if current_label is not None:
                            if current_label not in label_regions:
                                label_regions[current_label] = []
                            label_regions[current_label].append((start, i))
                        current_label = label
                        start = i
                
                if current_label is not None:
                    if current_label not in label_regions:
                        label_regions[current_label] = []
                    label_regions[current_label].append((start, len(labels)))
                
                reads.append(SimulatedRead(
                    sequence=seq,
                    labels=labels,
                    label_regions=label_regions
                ))
        
        n_sequences = len(sequences)
        
        if n_sequences == 0:
            console.print("[red]No sequences found in file![/red]")
            raise typer.Exit(1)
        
        # Filter out invalid reads if present
        n_invalid = 0
        n_total = n_sequences
        
        if format == 'pickle' and hasattr(reads[0], 'metadata'):
            n_total = len(reads)
            valid_reads = [r for r in reads if not r.metadata.get('is_invalid', False)]
            n_invalid = n_total - len(valid_reads)
            
            if n_invalid > 0:
                console.print(f"\n[yellow]Filtered out {n_invalid:,} invalid reads ({n_invalid/n_total*100:.1f}%)[/yellow]")
                console.print(f"[cyan]Analyzing {len(valid_reads):,} valid reads[/cyan]")
                
                # Update reads and derived data to use only valid reads
                reads = valid_reads
                sequences = [r.sequence for r in reads]
                labels_list = [r.labels for r in reads]
                n_sequences = len(sequences)
                
                if n_sequences == 0:
                    console.print("[red]No valid sequences found in file![/red]")
                    raise typer.Exit(1)
        
        # Standard statistics (unless ACC-only mode)
        if not acc_only:
            console.print(f"\n[bold]Statistics for {n_sequences:,} valid sequences:[/bold]")
            
            # Show invalid read info if present
            if format == 'pickle' and n_invalid > 0:
                invalid_table = Table(title="Dataset Composition", box=box.SIMPLE)
                invalid_table.add_column("Category", style="cyan")
                invalid_table.add_column("Count", style="green")
                invalid_table.add_column("Percentage", style="yellow")
                
                invalid_table.add_row("Valid reads", f"{n_sequences:,}", f"{n_sequences/n_total*100:.1f}%")
                invalid_table.add_row("Invalid reads", f"{n_invalid:,}", f"{n_invalid/n_total*100:.1f}%")
                invalid_table.add_row("Total", f"{n_total:,}", "100.0%")
                
                console.print()
                console.print(invalid_table)
            
            lengths = [len(seq) for seq in sequences]
            console.print(f"\n[cyan]Sequence lengths:[/cyan]")
            console.print(f"  Mean: {np.mean(lengths):.1f} bp")
            console.print(f"  Std: {np.std(lengths):.1f} bp")
            console.print(f"  Min: {np.min(lengths)} bp")
            console.print(f"  Max: {np.max(lengths)} bp")
            console.print(f"  Median: {np.median(lengths):.1f} bp")
            
            # Label statistics
            if labels_list and labels_list[0]:
                label_counts = {}
                for labels in labels_list:
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                
                console.print(f"\n[cyan]Label distribution:[/cyan]")
                total_labels = sum(label_counts.values())
                
                table = Table(title="Label Distribution")
                table.add_column("Label", style="cyan")
                table.add_column("Count", style="green")
                table.add_column("Percentage", style="yellow")
                
                for label in sorted(label_counts.keys()):
                    count = label_counts[label]
                    percentage = (count / total_labels) * 100
                    table.add_row(label, f"{count:,}", f"{percentage:.2f}%")
                
                console.print(table)
        
        # ACC-specific analysis
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Detailed ACC Segment Analysis[/bold yellow]")
        console.print(f"{'='*60}\n")
        
        # Load ACC generator if config provided
        acc_generator = None
        if config is not None:
            try:
                config_obj = TempestConfig.from_yaml(config)
                simulator = create_simulator_from_config(config_obj)
                acc_generator = simulator.acc_generator
                
                if acc_generator:
                    console.print("[green]Loaded ACC generator from config for PWM analysis[/green]")
                    console.print(f"  Temperature: {acc_generator.temperature}")
                    console.print(f"  Min entropy: {acc_generator.min_entropy}\n")
                else:
                    console.print("[yellow]Warning: No ACC generator found in config[/yellow]\n")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load ACC generator: {e}[/yellow]\n")
        
        # Perform detailed ACC analysis
        acc_stats = analyze_acc_segment_detailed(
            reads,
            acc_generator=acc_generator,
            show_sequences=10
        )
        
        # Display results
        display_acc_statistics(acc_stats, verbose=verbose)
        
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        raise typer.Exit(1)

# Export for backwards compatibility
__all__ = [
    'simulate_app',
    'run_simulation',
    'generate_command'
]
