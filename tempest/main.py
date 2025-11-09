"""
Tempest

This handles config loading, subcommand running, and is a general
execution entry point for all major modules (simulate, train, evaluate, etc.).
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging
import yaml
from tempest.config import TempestConfig

# import the runners
from tempest.simulate import run_simulation
from tempest.train import run_training
from tempest.evaluate import run_evaluation
from tempest.combine import run_combination
from tempest.compare import run_comparison
from tempest.demux import run_demultiplexing


# config parsing
def load_config(config_path: Union[str, Path]) -> TempestConfig:
    """
    Load a configuration file (YAML or JSON) into a TempestConfig dataclass.

    This function provides consistent validation and future-proof hooks
    for preprocessing (e.g., config overrides, defaults injection).
    """
    path = Path(config_path)
    logging.debug(f"Loading configuration from {path.resolve()}")

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix in [".yaml", ".yml"]:
        config = TempestConfig.from_yaml(path)
    elif path.suffix == ".json":
        config = TempestConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported configuration format: {path.suffix}")

    logging.info(f"Loaded configuration: {path.name}")
    return config


# runner
def run_pipeline(
    command: str,
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    extra_args: Optional[Dict[str, Any]] = None,
):
    """

    Parameters
    ----------
    command : str
        The name of the subcommand to run (simulate, train, evaluate,
        combine, compare, visualization, demux).
    config_path : str | Path
        Path to a YAML or JSON configuration file.
    output_dir : str | Path, optional
        Directory to store outputs, models, or logs.
    extra_args : dict, optional
        Additional arguments passed to the subcommand (simulate, train, evaluate,
        combine, compare, visualization, demux).
    """
    if extra_args is None:
        extra_args = {}

    config = load_config(config_path)
    logging.info(f"Executing TEMPEST pipeline: {command}")

    dispatch_map = {
        "simulate": run_simulation,
        "train": run_training,
        "evaluate": run_evaluation,
        "combine": run_combination,
        "compare": run_comparison,
        "demux": run_demultiplexing,
    }

    if command not in dispatch_map:
        raise ValueError(f"Unknown TEMPEST command: {command}")

    func = dispatch_map[command]
    logging.debug(f"Dispatching to function: {func.__module__}.{func.__name__}")

    return func(config, output_dir=output_dir, **extra_args)


# direct invocation
def main(
    command: str,
    config: Union[str, Path],
    output: Optional[str] = None,
    **kwargs,
):
    """
    Execute a TEMPEST command directly from Python.

    Example
    -------
    >>> from tempest.main import main
    >>> main("train", "configs/default.yaml", output="models/")
    """
    try:
        result = run_pipeline(command, config, output_dir=output, extra_args=kwargs)
        logging.info(f"TEMPEST {command} completed successfully.")
        return result
    except Exception as e:
        logging.exception(f"TEMPEST {command} failed: {e}")
        raise


if __name__ == "__main__":
    # Direct CLI bypass for quick tests
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tempest.main <command> <config_path> [output_dir]")
        sys.exit(1)

    cmd = sys.argv[1]
    cfg = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(cmd, cfg, output=out)