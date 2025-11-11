"""
Tempest demux command with sample-based demultiplexing.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import json
import yaml

from tempest.inference.sample_demultiplexer import demux_with_samples, SampleSheet

# Create the demux sub-application
demux_app = typer.Typer(help="Demultiplex FASTQ files with sample assignment")

console = Console()


def load_config(config_path: str):
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    
    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


@demux_app.command()
def samples(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained Tempest model"
    ),
    input_dir: Path = typer.Option(
        ...,
        "--input-dir", "-i",
        help="Directory containing FASTQ files to process"
    ),
    sample_sheet: Path = typer.Option(
        ...,
        "--sample-sheet", "-s",
        help="CSV file with sample_name, cbc, i5, i7 columns"
    ),
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Configuration file with demux settings"
    ),
    output_dir: Path = typer.Option(
        Path("./demux_samples"),
        "--output-dir", "-o",
        help="Output directory for per-sample FASTQ files"
    ),
    max_distance: int = typer.Option(
        2,
        "--max-distance",
        help="Maximum edit distance for barcode correction"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for model inference"
    ),
    confidence: float = typer.Option(
        0.85,
        "--confidence",
        help="Minimum confidence threshold"
    ),
    compress: bool = typer.Option(
        True,
        "--compress/--no-compress",
        help="Gzip output files"
    ),
    file_pattern: str = typer.Option(
        "*.fastq*",
        "--pattern",
        help="Pattern to match FASTQ files"
    )
):
    """
    Demultiplex FASTQ files and assign to samples based on barcodes.
    
    This command:
    1. Processes all FASTQ files in the input directory
    2. Uses the model to predict segment labels and validate architecture
    3. Extracts CBC, i5, i7 barcodes from valid reads
    4. Matches barcodes to samples in the sample sheet (with error correction)
    5. Outputs per-sample FASTQ files (optionally gzipped)
    
    Sample sheet format (CSV with header):
        sample_name,cbc,i5,i7
        Sample1,AAAAAA,ATATAGGA,ATTACTCG
        Sample2,CCCCCC,GGAGGATC,TCCGGAGA
    
    Example:
        tempest demux samples --model model.h5 \\
            --input-dir ./fastq_files \\
            --sample-sheet samples.csv \\
            --config config.yaml \\
            --output-dir ./demux_output
    """
    console.print(Panel.fit(
        "[bold blue]Sample-Based Demultiplexer[/bold blue]\n"
        "Assigning reads to samples based on extracted barcodes",
        border_style="blue"
    ))
    
    # Load and validate sample sheet
    console.print(f"\n[cyan]Loading sample sheet from {sample_sheet}...[/cyan]")
    
    try:
        sheet = SampleSheet(str(sample_sheet), max_distance)
        console.print(f"[green][/green] Loaded {len(sheet.samples)} samples")
        
        # Display sample summary
        _display_sample_summary(sheet)
        
    except Exception as e:
        console.print(f"[red]Error loading sample sheet: {e}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    console.print(f"\n[cyan]Loading configuration from {config}...[/cyan]")
    cfg = load_config(str(config))
    
    # Check for required demux settings
    if 'demux' not in cfg:
        console.print("[yellow]Warning: No demux section in config, using defaults[/yellow]")
    
    # Run demultiplexing
    console.print(f"\n[cyan]Processing FASTQ files in {input_dir}...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Demultiplexing...", start=False)
        
        try:
            results = demux_with_samples(
                model_path=str(model),
                input_dir=str(input_dir),
                sample_sheet_path=str(sample_sheet),
                config=cfg,
                output_dir=str(output_dir),
                max_edit_distance=max_distance,
                batch_size=batch_size,
                confidence_threshold=confidence,
                compress_output=compress,
                file_pattern=file_pattern
            )
            
            progress.update(task, completed=True)
            
        except Exception as e:
            console.print(f"\n[red]Error during demultiplexing: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    _display_demux_results(results, output_dir, compress)
    
    console.print(f"\n[green]Demultiplexing complete! Per-sample files in: {output_dir}[/green]")


@demux_app.command()
def validate(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model"
    ),
    input: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Input FASTQ file"
    ),
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Configuration file"
    ),
    output_dir: Path = typer.Option(
        Path("./validation"),
        "--output-dir", "-o",
        help="Output directory"
    ),
    save_valid: bool = typer.Option(
        True,
        "--save-valid/--no-save-valid",
        help="Save valid reads"
    ),
    save_invalid: bool = typer.Option(
        False,
        "--save-invalid/--no-save-invalid",
        help="Save invalid reads"
    )
):
    """
    Validate read architecture without sample assignment.
    
    Example:
        tempest demux validate --model model.h5 --input reads.fastq \\
            --config config.yaml
    """
    from tempest.inference.demux_validator import demux_with_validation
    
    console.print(Panel.fit(
        "[bold blue]Architecture Validator[/bold blue]",
        border_style="blue"
    ))
    
    cfg = load_config(str(config))
    
    console.print(f"[cyan]Processing {input}...[/cyan]")
    
    results = demux_with_validation(
        model_path=str(model),
        input_file=str(input),
        config=cfg,
        output_dir=str(output_dir),
        save_valid=save_valid,
        save_invalid=save_invalid
    )
    
    # Display validation results
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Total Reads", f"{results['total_reads']:,}")
    table.add_row("Valid", f"{results['valid']:,}")
    table.add_row("Invalid", f"{results['invalid']:,}")
    table.add_row("Valid Rate", f"{results.get('valid_rate', 0):.1%}")
    
    console.print(table)


@demux_app.command()
def check_sheet(
    sample_sheet: Path = typer.Argument(
        ...,
        help="Sample sheet CSV file to validate"
    )
):
    """
    Validate a sample sheet file.
    
    Checks for:
    - Required columns
    - Duplicate sample names
    - Duplicate barcode combinations
    - Barcode format issues
    
    Example:
        tempest demux check-sheet samples.csv
    """
    console.print(Panel.fit(
        "[bold blue]Sample Sheet Validator[/bold blue]",
        border_style="blue"
    ))
    
    try:
        sheet = SampleSheet(str(sample_sheet))
        
        # Check for issues
        issues = []
        
        # Check for duplicate sample names
        sample_names = [s.sample_name for s in sheet.samples]
        duplicates = [name for name in sample_names if sample_names.count(name) > 1]
        if duplicates:
            issues.append(f"Duplicate sample names: {set(duplicates)}")
        
        # Check for duplicate barcode combinations
        barcode_combos = [s.barcode_tuple() for s in sheet.samples]
        dup_combos = [combo for combo in barcode_combos if barcode_combos.count(combo) > 1]
        if dup_combos:
            issues.append(f"Duplicate barcode combinations: {len(set(dup_combos))}")
        
        # Check barcode lengths
        for s in sheet.samples:
            if len(s.cbc) not in [6, 8, 10, 16]:
                issues.append(f"Sample {s.sample_name}: unusual CBC length {len(s.cbc)}")
            if len(s.i5) != 8:
                issues.append(f"Sample {s.sample_name}: i5 should be 8bp, got {len(s.i5)}")
            if len(s.i7) != 8:
                issues.append(f"Sample {s.sample_name}: i7 should be 8bp, got {len(s.i7)}")
        
        # Display results
        if issues:
            console.print("[yellow]Issues found:[/yellow]")
            for issue in issues[:10]:  # Show first 10 issues
                console.print(f"  • {issue}")
            if len(issues) > 10:
                console.print(f"  ... and {len(issues) - 10} more")
        else:
            console.print(f"[green][/green] Sample sheet is valid!")
        
        # Display summary
        _display_sample_summary(sheet)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@demux_app.command()
def stats(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory containing demux results"
    )
):
    """
    Display statistics from sample-based demultiplexing.
    
    Example:
        tempest demux stats ./demux_output
    """
    console.print(Panel.fit(
        "[bold blue]Demultiplexing Statistics[/bold blue]",
        border_style="blue"
    ))
    
    # Load statistics file
    stats_file = results_dir / "demux_statistics.json"
    
    if not stats_file.exists():
        console.print(f"[red]Statistics file not found: {stats_file}[/red]")
        raise typer.Exit(1)
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Display overall statistics
    total = stats['total']
    
    overall_table = Table(title="Overall Statistics")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="magenta")
    
    overall_table.add_row("Total Reads", f"{total['reads']:,}")
    overall_table.add_row("Valid Reads", f"{total['valid']:,}")
    overall_table.add_row("Invalid Reads", f"{total['invalid']:,}")
    overall_table.add_row("Valid Rate", f"{total.get('valid_rate', 0):.1%}")
    overall_table.add_row("Assignment Rate", f"{total.get('assignment_rate', 0):.1%}")
    overall_table.add_row("", "")
    overall_table.add_row("Exact Matches", f"{total['exact_matches']:,}")
    overall_table.add_row("Corrected Matches", f"{total['corrected_matches']:,}")
    overall_table.add_row("No Matches", f"{total['no_matches']:,}")
    
    console.print(overall_table)
    
    # Display per-sample statistics
    if 'samples' in stats and stats['samples']:
        sample_table = Table(title="Per-Sample Statistics")
        sample_table.add_column("Sample", style="cyan")
        sample_table.add_column("Total", style="magenta")
        sample_table.add_column("Valid", style="green")
        sample_table.add_column("Invalid", style="red")
        sample_table.add_column("Exact", style="blue")
        sample_table.add_column("Corrected", style="yellow")
        
        # Sort samples by total reads
        sorted_samples = sorted(
            stats['samples'].items(),
            key=lambda x: x[1]['total_reads'],
            reverse=True
        )
        
        for sample_name, sample_stats in sorted_samples[:20]:  # Show top 20
            sample_table.add_row(
                sample_name[:30],  # Truncate long names
                f"{sample_stats['total_reads']:,}",
                f"{sample_stats['valid_reads']:,}",
                f"{sample_stats['invalid_reads']:,}",
                f"{sample_stats['exact_matches']:,}",
                f"{sample_stats['corrected_matches']:,}"
            )
        
        if len(sorted_samples) > 20:
            sample_table.add_row("...", "...", "...", "...", "...", "...")
        
        console.print("\n", sample_table)
    
    # Check for output files
    fastq_files = list(results_dir.glob("*.fastq*"))
    if fastq_files:
        console.print(f"\n[bold]Output Files:[/bold]")
        console.print(f"  Found {len(fastq_files)} sample files")
        
        # Show file sizes
        total_size = sum(f.stat().st_size for f in fastq_files)
        console.print(f"  Total size: {total_size / (1024**3):.2f} GB")


def _display_sample_summary(sheet: SampleSheet):
    """Display summary of loaded samples."""
    table = Table(title="Sample Summary", show_header=True)
    table.add_column("Sample", style="cyan")
    table.add_column("CBC", style="magenta")
    table.add_column("i5", style="green")
    table.add_column("i7", style="blue")
    
    # Show first few samples
    for i, sample in enumerate(sheet.samples[:5]):
        table.add_row(
            sample.sample_name,
            sample.cbc,
            sample.i5,
            sample.i7
        )
    
    if len(sheet.samples) > 5:
        table.add_row("...", "...", "...", "...")
        table.add_row(
            f"({len(sheet.samples)} total)",
            f"{len(sheet.valid_cbcs)} unique",
            f"{len(sheet.valid_i5s)} unique",
            f"{len(sheet.valid_i7s)} unique"
        )
    
    console.print(table)


def _display_demux_results(results: dict, output_dir: Path, compressed: bool):
    """Display demultiplexing results."""
    # Overall statistics
    total = results['total']
    
    table = Table(title="Demultiplexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Percentage", style="green")
    
    table.add_row(
        "Total Reads",
        f"{total['reads']:,}",
        "100.0%"
    )
    table.add_row(
        "Valid Architecture",
        f"{total['valid']:,}",
        f"{100*total.get('valid_rate', 0):.1f}%"
    )
    table.add_row(
        "Assigned to Samples",
        f"{total['exact_matches'] + total['corrected_matches']:,}",
        f"{100*total.get('assignment_rate', 0):.1f}%"
    )
    
    console.print("\n", table)
    
    # Sample distribution
    if 'samples' in results:
        # Get top samples by read count
        sorted_samples = sorted(
            [(name, stats['total_reads']) for name, stats in results['samples'].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        if sorted_samples:
            dist_table = Table(title="Top Samples by Read Count")
            dist_table.add_column("Sample", style="cyan")
            dist_table.add_column("Reads", style="magenta")
            
            for sample_name, count in sorted_samples[:10]:
                dist_table.add_row(sample_name, f"{count:,}")
            
            console.print("\n", dist_table)
    
    # Output information
    console.print(f"\n[bold]Output Location:[/bold]")
    console.print(f"  Directory: {output_dir}")
    console.print(f"  Format: {'Compressed (.fastq.gz)' if compressed else 'Uncompressed (.fastq)'}")
    console.print(f"  Statistics: {output_dir}/demux_statistics.json")


if __name__ == "__main__":
    demux_app()
