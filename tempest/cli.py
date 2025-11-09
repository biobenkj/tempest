#!/usr/bin/env python3
"""
Tempest CLI

Provides subcommands for simulation, training, evaluation, visualization,
demultiplexing, and ensemble model combination.
"""

import typer
import sys
import os
import logging
from pathlib import Path
from typing import Optional
from textwrap import dedent
import yaml
from rich.console import Console
from rich.progress import Progress

# import sub-applications
from tempest.simulate import simulate_app
from tempest.train import train_app
from tempest.evaluate import evaluate_app
from tempest.compare import compare_app
from tempest.combine import combine_app
from tempest.demux import demux_app

__version__ = "0.3.0"
console = Console()

# main
app = typer.Typer(
    name="tempest",
    help="TEMPEST: Advanced sequence annotation with length-constrained CRFs",
    add_completion=False,
    rich_markup_mode="rich",
)

# register subcommands
app.add_typer(simulate_app, name="simulate", help="Generate synthetic sequence data")
app.add_typer(train_app, name="train", help="Train Tempest models")
app.add_typer(evaluate_app, name="evaluate", help="Evaluate trained models")
app.add_typer(compare_app, name="compare", help="Compare multiple models")
app.add_typer(combine_app, name="combine", help="Combine models using ensemble methods")
app.add_typer(demux_app, name="demux", help="Demultiplex FASTQ files")

# logging and tf suppression utils
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


def suppress_tensorflow_logging():
    """Suppress TensorFlow verbose logging unless debug mode is enabled."""
    if "--debug" not in sys.argv and os.getenv("TEMPEST_DEBUG", "0") != "1":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ.setdefault("TF_DISABLE_PLUGIN_REGISTRATION", "1")
        os.environ.setdefault("TF_ENABLE_DEPRECATION_WARNINGS", "0")

        import warnings
        warnings.filterwarnings("ignore")

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").propagate = False

# cli metadata - global options
@app.callback()
def main_callback(
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode with verbose output"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Save logs to a file"
    ),
):
    """Global options for the Tempest CLI."""
    suppress_tensorflow_logging()
    setup_logging(
        level="DEBUG" if debug else log_level,
        log_file=str(log_file) if log_file else None,
    )

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["TEMPEST_DEBUG"] = "1"

# core commands
@app.command()
def version():
    """Show Tempest version information."""
    console.print(f"[bold blue]Tempest[/bold blue] version {__version__}")
    console.print("Long read RNA-seq sequence annotation toolkit")
    console.print("© Ben Johnson")


@app.command()
def info():
    """Display system and environment information."""
    import platform

    console.print("[bold blue]System Information[/bold blue]")
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

# init a new project
@app.command()
def init(
    project_dir: Path = typer.Argument(
        Path("."), help="Directory to initialize the Tempest project."
    ),
    with_examples: bool = typer.Option(
        False, "--with-examples", help="Include example data and scripts."
    ),
):
    """
    Initialize a new Tempest project directory and create a default
    configuration modeled after the probabilistic PWM architecture,
    excluding RP1 and RP2 sequences.
    """
    console.print(f"[bold blue]Initializing Tempest project in {project_dir}[/bold blue]")
    dirs = [
        project_dir / "configs",
        project_dir / "data",
        project_dir / "models",
        project_dir / "results",
        project_dir / "logs",
        project_dir / "plots",
    ]

    with Progress() as progress:
        task = progress.add_task("[cyan]Creating directories...", total=len(dirs))
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            progress.advance(task)

    # example config
    default_config = {
        "model": {
            "max_seq_len": 1500,
            "num_labels": 9,
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
                "UMI": "random",
                "cDNA": "transcript",
                "polyA": "polya",
            },
            "whitelist_files": {
                "i7": "whitelist/udi_i7.txt",
                "i5": "whitelist/udi_i5.txt",
                "CBC": "whitelist/cbc.txt",
            },
            "pwm_files": {"ACC": "whitelist/acc_pwm.txt"},
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
            "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.0001},
            "use_class_weights": True,
        },
        "evaluation": {"metrics": ["accuracy", "f1", "segment_accuracy"]},
        "output": {"save_dir": "./tempest_output", "save_model": True},
    }

    config_path = project_dir / "configs" / "default.yaml"
    yaml.safe_dump(default_config, config_path.open("w"))
    console.print(f"[green]✓[/green] Created default configuration: {config_path}")

    # optional examples
    if with_examples:
        ex_data = project_dir / "data" / "example_sequences.txt"
        ex_data.write_text("ATCGATCG\nGCTAGCTA\nTAGCTAGC\n")
        ex_script = project_dir / "train_example.sh"
        ex_script.write_text(dedent("""\
            #!/bin/bash
            tempest simulate generate --config configs/default.yaml --output-dir data
            tempest train standard --config configs/default.yaml --output-dir models
            tempest evaluate performance --model models/model.h5 --test-data data/test.txt
        """))
        ex_script.chmod(0o755)
        console.print("[green]✓[/green] Added example data and training script.")

    readme = project_dir / "README.md"
    readme.write_text(dedent(f"""\
        # {project_dir.name}

        A Tempest project initialized with probabilistic PWM configuration (RP1/RP2 removed).

        ## Structure
        {project_dir.name}/
        ├── configs/
        ├── data/
        ├── models/
        ├── results/
        ├── logs/
        └── plots/

        ## Next Steps
        1. Edit `configs/default.yaml` for your architecture.
        2. Generate data: `tempest simulate generate --config configs/default.yaml`
        3. Train a model: `tempest train standard --config configs/default.yaml`
    """))
    console.print("[bold green]Project initialized successfully![/bold green]")

# main entry point
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