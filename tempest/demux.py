"""
Tempest demultiplexing commands using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from tempest.inference.demultiplexer import Demultiplexer
from tempest.utils import load_config

# Create the demux sub-application
demux_app = typer.Typer(help="Demultiplex FASTQ files using trained models")

console = Console()


@demux_app.command()
def fastq(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model",
        exists=True
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Input FASTQ file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        help="Input directory containing FASTQ files",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_dir: Path = typer.Option(
        Path("./demux_results"),
        "--output-dir", "-o",
        help="Output directory for results"
    ),
    whitelist_cbc: Optional[Path] = typer.Option(
        None,
        "--whitelist-cbc",
        help="CBC barcode whitelist file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    whitelist_i5: Optional[Path] = typer.Option(
        None,
        "--whitelist-i5",
        help="i5 barcode whitelist file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    whitelist_i7: Optional[Path] = typer.Option(
        None,
        "--whitelist-i7",
        help="i7 barcode whitelist file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    max_edit_distance: int = typer.Option(
        2,
        "--max-edit-distance",
        help="Maximum edit distance for barcode correction",
        min=0,
        max=5
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for model inference",
        min=1,
        max=512
    ),
    file_pattern: str = typer.Option(
        "*.fastq*",
        "--file-pattern",
        help="File pattern for directory processing"
    ),
    plot_metrics: bool = typer.Option(
        False,
        "--plot-metrics",
        help="Generate visualization plots"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file with model settings"
    )
):
    """
    Demultiplex FASTQ files using a trained model.
    
    Examples:
        # Single file demultiplexing
        tempest demux fastq --model model.h5 --input reads.fastq \\
            --whitelist-cbc cbc.txt --whitelist-i5 i5.txt --whitelist-i7 i7.txt
        
        # Directory processing
        tempest demux fastq --model model.h5 --input-dir ./fastq_files \\
            --output-dir ./demux_output --plot-metrics
        
        # With barcode correction
        tempest demux fastq --model model.h5 --input reads.fastq \\
            --max-edit-distance 3 --batch-size 64
    """
    if not input_file and not input_dir:
        console.print("[red]Error: Must specify either --input or --input-dir[/red]")
        raise typer.Exit(1)
    
    if input_file and input_dir:
        console.print("[red]Error: Cannot specify both --input and --input-dir[/red]")
        raise typer.Exit(1)
    
    console.print("[bold blue]TEMPEST Demultiplexer[/bold blue]")
    console.print("=" * 60)
    
    # Load configuration if provided
    cfg = None
    if config:
        cfg = load_config(str(config))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize demultiplexer
    demux = Demultiplexer(
        model_path=str(model),
        config=cfg,
        whitelist_cbc=str(whitelist_cbc) if whitelist_cbc else None,
        whitelist_i5=str(whitelist_i5) if whitelist_i5 else None,
        whitelist_i7=str(whitelist_i7) if whitelist_i7 else None,
        max_edit_distance=max_edit_distance,
        batch_size=batch_size
    )
    
    # Process files
    if input_file:
        # Single file processing
        console.print(f"Processing: {input_file}")
        results = _process_single_file(
            demux, input_file, output_dir
        )
        _display_results(results, plot_metrics, output_dir)
        
    else:
        # Directory processing
        fastq_files = list(Path(input_dir).glob(file_pattern))
        if not fastq_files:
            console.print(f"[red]No files matching pattern '{file_pattern}' found![/red]")
            raise typer.Exit(1)
        
        console.print(f"Found {len(fastq_files)} files to process")
        
        all_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(fastq_files))
            
            for fastq_file in fastq_files:
                progress.update(task, description=f"Processing {fastq_file.name}")
                results = _process_single_file(
                    demux, fastq_file, output_dir
                )
                all_results.append(results)
                progress.advance(task)
        
        # Aggregate and display results
        aggregated = _aggregate_results(all_results)
        _display_results(aggregated, plot_metrics, output_dir)
    
    console.print(f"\n[green]✓[/green] Demultiplexing complete! Results saved to: {output_dir}")


@demux_app.command()
def stats(
    results_dir: Path = typer.Option(
        ...,
        "--results-dir", "-r",
        help="Directory containing demux results",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save statistics to file"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        help="Show detailed statistics"
    )
):
    """
    Generate statistics from demultiplexing results.
    
    Examples:
        # Basic statistics
        tempest demux stats --results-dir ./demux_results
        
        # Detailed statistics saved to file
        tempest demux stats --results-dir ./demux_results --detailed --output stats.json
    """
    import json
    from collections import Counter
    
    console.print("[bold blue]Demultiplexing Statistics[/bold blue]")
    console.print("=" * 60)
    
    # Find all result files
    result_files = list(Path(results_dir).glob("*_results.json"))
    if not result_files:
        console.print("[red]No result files found![/red]")
        raise typer.Exit(1)
    
    # Load and aggregate statistics
    total_reads = 0
    total_passed = 0
    total_failed = 0
    barcode_counts = Counter()
    error_types = Counter()
    
    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)
            total_reads += data.get('total_reads', 0)
            total_passed += data.get('passed_reads', 0)
            total_failed += data.get('failed_reads', 0)
            
            if 'barcode_distribution' in data:
                barcode_counts.update(data['barcode_distribution'])
            if 'error_types' in data:
                error_types.update(data['error_types'])
    
    # Display statistics
    from rich.table import Table
    
    # Overall statistics
    table = Table(title="Overall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Percentage", style="yellow")
    
    table.add_row("Total Reads", f"{total_reads:,}", "-")
    table.add_row("Passed QC", f"{total_passed:,}", 
                  f"{100*total_passed/total_reads:.1f}%" if total_reads > 0 else "0%")
    table.add_row("Failed QC", f"{total_failed:,}", 
                  f"{100*total_failed/total_reads:.1f}%" if total_reads > 0 else "0%")
    
    console.print(table)
    
    if detailed:
        # Barcode distribution
        console.print("\n[bold]Top 10 Barcodes:[/bold]")
        barcode_table = Table()
        barcode_table.add_column("Barcode", style="cyan")
        barcode_table.add_column("Count", style="magenta")
        barcode_table.add_column("Percentage", style="yellow")
        
        for barcode, count in barcode_counts.most_common(10):
            pct = 100 * count / total_passed if total_passed > 0 else 0
            barcode_table.add_row(barcode, f"{count:,}", f"{pct:.2f}%")
        
        console.print(barcode_table)
        
        # Error distribution
        if error_types:
            console.print("\n[bold]Error Types:[/bold]")
            error_table = Table()
            error_table.add_column("Error Type", style="cyan")
            error_table.add_column("Count", style="magenta")
            
            for error_type, count in error_types.most_common():
                error_table.add_row(error_type, f"{count:,}")
            
            console.print(error_table)
    
    # Save if requested
    if output:
        stats_data = {
            'total_reads': total_reads,
            'passed_reads': total_passed,
            'failed_reads': total_failed,
            'pass_rate': total_passed / total_reads if total_reads > 0 else 0,
            'barcode_distribution': dict(barcode_counts),
            'error_distribution': dict(error_types)
        }
        
        with open(output, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        console.print(f"\n[green]✓[/green] Statistics saved to: {output}")


@demux_app.command()
def validate(
    model: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to trained model",
        exists=True
    ),
    test_fastq: Path = typer.Option(
        ...,
        "--test-fastq",
        help="Test FASTQ file with known barcodes",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    truth_file: Path = typer.Option(
        ...,
        "--truth-file",
        help="File with true barcode assignments",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save validation results"
    )
):
    """
    Validate demultiplexing accuracy against known truth.
    
    Examples:
        # Validate model accuracy
        tempest demux validate --model model.h5 --test-fastq test.fastq \\
            --truth-file truth.tsv --output validation.json
    """
    console.print("[bold blue]Validating Demultiplexing Accuracy[/bold blue]")
    console.print("=" * 60)
    
    # Load truth data
    console.print("Loading ground truth...")
    truth = {}
    with open(truth_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                truth[parts[0]] = parts[1]
    
    console.print(f"Loaded {len(truth)} truth assignments")
    
    # Initialize demultiplexer
    demux = Demultiplexer(model_path=str(model))
    
    # Process test file
    console.print("Running demultiplexing...")
    predictions = demux.process_fastq(str(test_fastq))
    
    # Compare predictions to truth
    correct = 0
    incorrect = 0
    confusion_matrix = {}
    
    for read_id, pred_barcode in predictions.items():
        if read_id in truth:
            true_barcode = truth[read_id]
            if pred_barcode == true_barcode:
                correct += 1
            else:
                incorrect += 1
                # Update confusion matrix
                key = f"{true_barcode}->{pred_barcode}"
                confusion_matrix[key] = confusion_matrix.get(key, 0) + 1
    
    # Calculate metrics
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0
    
    # Display results
    from rich.table import Table
    
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Reads", f"{total:,}")
    table.add_row("Correct", f"{correct:,}")
    table.add_row("Incorrect", f"{incorrect:,}")
    table.add_row("Accuracy", f"{accuracy:.4f}")
    
    console.print(table)
    
    # Show top misclassifications
    if confusion_matrix:
        console.print("\n[bold]Top Misclassifications:[/bold]")
        sorted_errors = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)[:10]
        
        error_table = Table()
        error_table.add_column("True → Predicted", style="cyan")
        error_table.add_column("Count", style="magenta")
        
        for error_type, count in sorted_errors:
            error_table.add_row(error_type, str(count))
        
        console.print(error_table)
    
    # Save if requested
    if output:
        import json
        validation_data = {
            'total_reads': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix
        }
        
        with open(output, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        console.print(f"\n[green]✓[/green] Validation results saved to: {output}")


def _process_single_file(demux, input_file: Path, output_dir: Path) -> dict:
    """Process a single FASTQ file."""
    # Create output filename
    output_prefix = input_file.stem.replace('.fastq', '').replace('.fq', '')
    
    # Run demultiplexing
    results = demux.process_fastq(
        input_file=str(input_file),
        output_dir=str(output_dir),
        output_prefix=output_prefix
    )
    
    return results


def _aggregate_results(results_list: List[dict]) -> dict:
    """Aggregate results from multiple files."""
    aggregated = {
        'total_reads': sum(r.get('total_reads', 0) for r in results_list),
        'passed_reads': sum(r.get('passed_reads', 0) for r in results_list),
        'failed_reads': sum(r.get('failed_reads', 0) for r in results_list),
        'files_processed': len(results_list)
    }
    
    # Aggregate barcode distributions
    from collections import Counter
    barcode_counts = Counter()
    for r in results_list:
        if 'barcode_distribution' in r:
            barcode_counts.update(r['barcode_distribution'])
    
    aggregated['barcode_distribution'] = dict(barcode_counts)
    
    return aggregated


def _display_results(results: dict, plot_metrics: bool, output_dir: Path):
    """Display demultiplexing results."""
    from rich.table import Table
    
    # Summary table
    table = Table(title="Demultiplexing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    total = results.get('total_reads', 0)
    passed = results.get('passed_reads', 0)
    failed = results.get('failed_reads', 0)
    
    table.add_row("Total Reads", f"{total:,}")
    table.add_row("Passed QC", f"{passed:,} ({100*passed/total:.1f}%)" if total > 0 else "0")
    table.add_row("Failed QC", f"{failed:,} ({100*failed/total:.1f}%)" if total > 0 else "0")
    
    if 'files_processed' in results:
        table.add_row("Files Processed", str(results['files_processed']))
    
    console.print(table)
    
    # Plot metrics if requested
    if plot_metrics and 'barcode_distribution' in results:
        import matplotlib.pyplot as plt
        
        # Barcode distribution plot
        barcodes = results['barcode_distribution']
        if barcodes:
            top_barcodes = sorted(barcodes.items(), key=lambda x: x[1], reverse=True)[:20]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(top_barcodes)), [c for _, c in top_barcodes])
            plt.xlabel('Barcode Rank')
            plt.ylabel('Read Count')
            plt.title('Top 20 Barcode Distribution')
            plt.yscale('log')
            
            plot_file = output_dir / 'barcode_distribution.png'
            plt.savefig(plot_file)
            plt.close()
            
            console.print(f"\n[green]✓[/green] Barcode distribution plot saved to: {plot_file}")
