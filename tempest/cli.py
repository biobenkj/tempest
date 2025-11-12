#!/usr/bin/env python3
"""
Tempest CLI

Provides subcommands for simulation, training, evaluation, visualization,
demultiplexing, and ensemble model combination using the Typer framework.
"""
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import typer
import sys
import logging
from pathlib import Path
from typing import Optional
from textwrap import dedent
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tempest.main import main as tempest_main

__version__ = "0.3.0"
console = Console()

# Main application
app = typer.Typer(
    name="tempest",
    help="TEMPEST: Advanced sequence annotation with length-constrained CRFs",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Import sub-applications
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
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


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
        console.print(f"[bold blue]Tempest[/bold blue] version {__version__}")
        raise typer.Exit()
    
    setup_logging(
        level="DEBUG" if debug else log_level,
        log_file=str(log_file) if log_file else None,
    )
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["TEMPEST_DEBUG"] = "1"


@app.command()
def info():
    """Display system and environment information."""
    import platform
    
    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 60)
    console.print(f"[cyan]Python:[/cyan] {sys.version.split()[0]}")
    console.print(f"[cyan]Platform:[/cyan] {platform.platform()}")
    console.print(f"[cyan]Tempest Version:[/cyan] {__version__}")
    
    try:
        import tensorflow as tf
        console.print(f"[cyan]TensorFlow:[/cyan] {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            console.print(f"[cyan]GPUs Available:[/cyan] {len(gpus)}")
            for i, gpu in enumerate(gpus):
                console.print(f"  GPU {i}: {gpu.name}")
        else:
            console.print("[yellow]No GPUs detected[/yellow]")
    except ImportError:
        console.print("[red]TensorFlow not installed[/red]")
    
    try:
        import numpy as np
        console.print(f"[cyan]NumPy:[/cyan] {np.__version__}")
    except ImportError:
        console.print("[red]NumPy not installed[/red]")
    
    # Display memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        console.print(f"[cyan]System Memory:[/cyan] {memory.total / (1024**3):.1f} GB")
        console.print(f"[cyan]Available Memory:[/cyan] {memory.available / (1024**3):.1f} GB")
    except ImportError:
        pass
    
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
        project_dir / "models",
        project_dir / "results",
        project_dir / "logs",
        project_dir / "plots",
        project_dir / "whitelist",
    ]
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Creating directories...", total=len(dirs))
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            progress.advance(task)
    
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
        console.print(f"[green][/green] Created example configuration: {config_path}")
        
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
        console.print("[green][/green] Created example workflow script")
        
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
        console.print("[green][/green] Created ACC PWM file")
    
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
        │   └── processed/  # Processed/simulated data
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
    console.print("[green][/green] Created README.md")
    
    # Display summary
    table = Table(title="Project Initialized", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Project Directory", str(project_dir))
    table.add_row("Configuration", " Created" if with_examples else "Ready for creation")
    table.add_row("Directories", " Created")
    if with_examples:
        table.add_row("Example Config", " Included")
        table.add_row("Workflow Script", " Generated")
        table.add_row("Whitelists", " Generated")
    
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
