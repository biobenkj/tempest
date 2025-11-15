#!/usr/bin/env python3
"""
Tempest CLI

Provides subcommands for simulation, training, evaluation, visualization,
demultiplexing, and ensemble model combination using the Typer framework.
"""
import os
import sys
import warnings

# Set default environment variables BEFORE any imports
# This ensures subcommands see these values when they import
if "TEMPEST_LOG_LEVEL" not in os.environ:
    os.environ["TEMPEST_LOG_LEVEL"] = "INFO"
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from tempest.utils.logging_utils import setup_rich_logging, console, status
setup_rich_logging(os.getenv("TEMPEST_LOG_LEVEL", "INFO"))

import typer
from pathlib import Path
from typing import Optional
from textwrap import dedent
import yaml
from rich.progress import Progress
from rich.table import Table
from rich.logging import RichHandler
from tempest.main import main as tempest_main
import logging

__version__ = "0.3.0"

# Main application
app = typer.Typer(
    name="tempest",
    help="TEMPEST: Advanced sequence annotation with length-constrained CRFs",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Import sub-applications AFTER setting default environment
from tempest.simulate import simulate_app
from tempest.evaluate import evaluate_app
from tempest.combine import combine_app
from tempest.demux import demux_app
from tempest.visualize import visualize_app

# Register subcommands
app.add_typer(simulate_app, name="simulate", help="Generate synthetic sequence data")
@app.command("train")
def train_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config YAML or JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    mode: str = typer.Option("standard", "--mode", "-m", help="Training mode: standard | hybrid | ensemble"),
    **kwargs,
):
    """
    Train Tempest models with various approaches (standard, hybrid, ensemble).
    """
    tempest_main("train", config, output=output_dir, subcommand=mode, **kwargs)
app.add_typer(evaluate_app, name="evaluate", help="Evaluate trained models")
app.add_typer(visualize_app, name="visualize", help="Visualize predictions, training history, and data distributions")
app.add_typer(combine_app, name="combine", help="Combine models using BMA/ensemble methods")
app.add_typer(demux_app, name="demux", help="Demultiplex FASTQ files with sample assignment")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for Tempest CLI with Rich integration.
    
    This function reconfigures logging to use the specified level,
    ensuring that global options like --debug propagate to all modules.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Reconfigure Rich logging (handles both initial setup and updates)
    setup_rich_logging(level)

    # Reconfigure all tempest module loggers to use the new level
    # This ensures subcommands that initialized loggers at import time get updated
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith('tempest'):
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(numeric_level)

    # Add a file handler if requested (only once)
    root_logger = logging.getLogger()
    if log_file and not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
        for h in root_logger.handlers
    ):
        fh = logging.FileHandler(log_file)
        fh.setLevel(numeric_level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


@app.callback()
def main_callback(
    debug: bool = typer.Option(
        False, 
        "--debug", 
        help="Enable debug mode with verbose output"
    ),
    log_level: str = typer.Option(
        "INFO", 
        "--log-level", 
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    log_file: Optional[Path] = typer.Option(
        None, 
        "--log-file", 
        help="Save logs to a file"
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit"
    ),
):
    """
    TEMPEST: Advanced sequence annotation toolkit for long-read RNA-seq
    
    TEMPEST (Transcript Element Mapping via Position-Encoded Sequence Tracking)
    is a tool that annotates long read RNA-seq read architecture using a deep
    learning approach that can enforce element length constraints. Multiple models
    can be trained and aggregated through Bayesian model averaging as well. Finally,
    models can be compared and evaluated prior to using for annotating and demultiplexing
    newly generated data.
    """
    if version:
        with status("Retrieving Tempest version..."):
            pass  # minimal delay keeps spinner clean
        console.print(f"[bold blue]Tempest[/bold blue] version {__version__}")
        raise typer.Exit()
    
    # Resolve final log level
    resolved_level = "DEBUG" if debug else log_level.upper()

    # Update environment variable so any late imports see the correct level
    os.environ["TEMPEST_LOG_LEVEL"] = resolved_level
    
    if debug:
        os.environ["TEMPEST_DEBUG"] = "1"

    # Reconfigure logging with user-specified level
    # This updates all existing loggers that were initialized with default settings
    setup_logging(
        level=resolved_level,
        log_file=str(log_file) if log_file else None,
    )
    
    if debug:
        console.print(f"[bold yellow]Debug mode active[/bold yellow] (TEMPEST_LOG_LEVEL={resolved_level})")


@app.command()
def info():
    """Display system and environment information."""
    import platform
    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 60)
    console.print(f"[cyan]Python:[/cyan] {sys.version.split()[0]}")
    console.print(f"[cyan]Platform:[/cyan] {platform.platform()}")
    console.print(f"[cyan]Tempest Version:[/cyan] {__version__}")

    # Use Rich status spinner for environment probing
    with status("Checking environment and hardware..."):
        # TensorFlow info
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            gpus = tf.config.list_physical_devices("GPU")
        except ImportError:
            tf_version = None
            gpus = None

        # NumPy info
        try:
            import numpy as np
            np_version = np.__version__
        except ImportError:
            np_version = None

        # Memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
        except ImportError:
            memory = None

    # Now display results after spinner completes
    if tf_version:
        console.print(f"[cyan]TensorFlow:[/cyan] {tf_version}")
        if gpus:
            console.print(f"[cyan]GPUs Available:[/cyan] {len(gpus)}")
            for i, gpu in enumerate(gpus):
                console.print(f"  GPU {i}: {gpu.name}")
        else:
            console.print("[yellow]No GPUs detected[/yellow]")
    else:
        console.print("[red]TensorFlow not installed[/red]")

    if np_version:
        console.print(f"[cyan]NumPy:[/cyan] {np_version}")
    else:
        console.print("[red]NumPy not installed[/red]")

    if memory:
        console.print(f"[cyan]System Memory:[/cyan] {memory.total / (1024**3):.1f} GB")
        console.print(f"[cyan]Available Memory:[/cyan] {memory.available / (1024**3):.1f} GB")

    console.print("")


@app.command()
def init(
    project_dir: Path = typer.Argument(
        Path("."), 
        help="Directory to initialize the Tempest project"
    ),
    with_examples: bool = typer.Option(
        False, 
        "--with-examples", 
        help="Include example configuration, data, and scripts"
    ),
):
    """
    Initialize a new Tempest project directory.
    
    Creates a project structure with directories for configs, data, models,
    results, logs, plots, and whitelists. If --with-examples is used, also
    creates example configuration and workflow scripts.
    """
    console.print(f"\n[bold blue]Initializing Tempest project in {project_dir}[/bold blue]")
    
    # Create directory structure
    dirs = [
        project_dir / "configs",
        project_dir / "data" / "raw",
        project_dir / "data" / "processed",
        project_dir / "data" / "simulated",
        project_dir / "models",
        project_dir / "results",
        project_dir / "logs",
        project_dir / "plots",
        project_dir / "whitelist",
        project_dir / "refs",
        project_dir / "tmp"
    ]
    
    with status("Creating directories..."):
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    console.log(f"[green]Created {len(dirs)} directories successfully[/green]")
    
    # Create minimal default configuration
    if with_examples:
        # Example configuration based on standard 11-segment architecture
        example_config = {
            "model": {
                "max_seq_len": 1500,
                "num_labels": 9,  # 9 segments without RP1/RP2
                "embedding_dim": 128,
                "lstm_units": 256,
                "lstm_layers": 2,
                "dropout": 0.3,
                "use_cnn": True,
                "use_bilstm": True,
                "batch_size": 32,
            },
            "simulation": {
                "num_sequences": 50000,
                "train_split": 0.8,
                "random_seed": 42,
                "sequence_order": [
                    "p7", "i7", "UMI", "ACC", "cDNA", "polyA", "CBC", "i5", "p5"
                ],
                "sequences": {
                    "p7": "CAAGCAGAAGACGGCATACGAGAT",
                    "p5": "GTGTAGATCTCGGTGGTCGCCGTATCATT",
                },
                "whitelist_files": {
                    "i7": "whitelist/udi_i7.txt",
                    "i5": "whitelist/udi_i5.txt",
                    "CBC": "whitelist/cbc.txt",
                },
                "pwm_files": {
                    "ACC": "whitelist/acc_pwm.txt",
                },
                "pwm": {
                    "pwm_file": "whitelist/acc_pwm.txt",
                    "temperature": 1.2,
                    "min_entropy": 0.1,
                    "diversity_boost": 1.0,
                    "pattern": "ACCSSV",
                },
                "segment_generation": {
                    "lengths": {
                        "p7": 24, "i7": 8, "UMI": 8, "ACC": 6,
                        "cDNA": 500, "polyA": 30, "CBC": 6, "i5": 8, "p5": 29
                    },
                    "generation_mode": {
                        "p7": "fixed", "i7": "whitelist", "UMI": "random",
                        "ACC": "pwm", "cDNA": "transcript", "polyA": "polya",
                        "CBC": "whitelist", "i5": "whitelist", "p5": "fixed"
                    },
                },
                "sequence_lengths": {
                    "p7": {"min": 24, "max": 24},
                    "i7": {"min": 8, "max": 8},
                    "UMI": {"min": 8, "max": 8},
                    "ACC": {"min": 6, "max": 6},
                    "cDNA": {"min": 200, "max": 1000},
                    "polyA": {"min": 10, "max": 50},
                    "CBC": {"min": 6, "max": 6},
                    "i5": {"min": 8, "max": 8},
                    "p5": {"min": 29, "max": 29},
                },
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping": {
                    "enabled": True,
                    "patience": 10,
                    "min_delta": 0.0001
                },
                "use_class_weights": True,
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "segment_accuracy", "confusion_matrix"]
            },
            "output": {
                "save_dir": "./tempest_output",
                "save_model": True,
                "save_predictions": True,
            },
        }
        
        # Save example configuration
        config_path = project_dir / "configs" / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(example_config, f, default_flow_style=False)
        console.print(f"[green]Created example configuration:[/green] {config_path}")
        
        # Create example workflow script
        workflow_script = project_dir / "run_workflow.sh"
        workflow_content = dedent(f"""\
            #!/bin/bash
            # Tempest workflow example
            
            echo "Step 1: Generate synthetic data"
            tempest simulate generate \\
                --config configs/config.yaml \\
                --output-dir data/processed \\
                --split \\
                --num-sequences 10000
            
            echo "Step 2: Train model"
            tempest train standard \\
                --config configs/config.yaml \\
                --train-data data/processed/train.pkl.gz \\
                --val-data data/processed/val.pkl.gz \\
                --output-dir models \\
                --epochs 20
            
            echo "Step 3: Evaluate model"
            tempest evaluate performance \\
                --model models/best_model.h5 \\
                --test-data data/processed/val.pkl.gz \\
                --output-dir results
            
            echo "Workflow complete!"
        """)
        workflow_script.write_text(workflow_content)
        workflow_script.chmod(0o755)
        console.print("[green]Created example workflow script[/green]")
        
        # Create example whitelist files
        i7_whitelist = project_dir / "whitelist" / "udi_i7.txt"
        i7_whitelist.write_text("ATTACTCG\nTCCGGAGA\nCGCTCATT\nGAGATTCC\n")
        console.print("[green][/green] Created i7 whitelist")
        
        i5_whitelist = project_dir / "whitelist" / "udi_i5.txt"
        i5_whitelist.write_text("TATAGCCT\nATATGAGA\nAGAGGATA\nTCTACTCT\n")
        console.print("[green][/green] Created i5 whitelist")
        
        cbc_whitelist = project_dir / "whitelist" / "cbc.txt"
        cbc_whitelist.write_text("AAAAAA\nAAAAAC\nAAAAAG\nAAAATA\n")
        console.print("[green][/green] Created CBC whitelist")
        
        acc_pwm = project_dir / "whitelist" / "acc_pwm.txt"
        acc_pwm.write_text("# ACC PWM Matrix\nA: 0.9 0.1 0.1 0.3 0.3 0.25\nC: 0.03 0.8 0.8 0.2 0.2 0.25\nG: 0.03 0.05 0.05 0.3 0.2 0.25\nT: 0.04 0.05 0.05 0.2 0.3 0.25\n")
        console.print("[green]Created ACC PWM file[/green] ")
    
    # Create README
    readme = project_dir / "README.md"
    readme_content = dedent(f"""\
        # Tempest Project - {project_dir.name}
        
        A Tempest project for sequence annotation with length-constrained CRFs.
        
        ## Project Structure
        ```
        {project_dir.name}/
        ├── configs/         # Configuration files
        ├── data/           # Data directory
        │   ├── raw/        # Raw sequence data
        │   ├── processed/  # Processed data
        │   └── simulated/  # Simulated data
        ├── models/         # Trained models
        ├── results/        # Evaluation results
        ├── logs/           # Training logs
        ├── plots/          # Visualizations
        └── whitelist/      # Barcode whitelists and PWMs
        ```
        
        ## Quick Start
        
        1. **Create Configuration:**
           Edit or create a configuration file in `configs/`
        
        2. **Generate Data:**
           ```bash
           tempest simulate generate --config configs/config.yaml --output-dir data/processed --split
           ```
        
        3. **Train Model:**
           ```bash
           tempest train standard --config configs/config.yaml \\
               --train-data data/processed/train.pkl.gz \\
               --val-data data/processed/val.pkl.gz \\
               --output-dir models
           ```
        
        4. **Evaluate:**
           ```bash
           tempest evaluate performance --model models/best_model.h5 \\
               --test-data data/processed/val.pkl.gz
           ```
        
        ## Documentation
        
        For more information, see the [Tempest documentation](https://github.com/tempest/docs).
    """)
    readme.write_text(readme_content)
    console.print("[green]Created README.md[/green]")
    
    # Display summary
    table = Table(title="Project Initialized", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Project Directory", str(project_dir))
    table.add_row("Configuration", "Created" if with_examples else "Ready for creation")
    table.add_row("Directories", "Created")
    if with_examples:
        table.add_row("Example Config", "Included")
        table.add_row("Workflow Script", "Generated")
        table.add_row("Whitelists", "Generated")
    
    console.print("\n", table)
    console.print("\n[bold green]Project initialized successfully![/bold green]")
    if with_examples:
        console.print(f"[dim]Next: cd {project_dir} && ./run_workflow.sh[/dim]\n")
    else:
        console.print(f"[dim]Next: Create a configuration file in {project_dir}/configs/[/dim]\n")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        if os.getenv("TEMPEST_DEBUG", "0") == "1":
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]Run with --debug for full traceback[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
