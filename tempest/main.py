"""
Tempest Main Entry Point

Enhanced for Typer nested subcommand compatibility.

This version preserves the full pipeline orchestration
(load_config, run_pipeline, etc.) while supporting nested
commands such as:
    tempest train standard
    tempest train hybrid
    tempest train ensemble
"""
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import logging
import pickle
import gzip

from tempest.config import TempestConfig, load_config as config_load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration and data utilities
# ---------------------------------------------------------------------
def load_config(config_path: Union[str, Path]) -> TempestConfig:
    """Load configuration file into TempestConfig."""
    return config_load_config(str(config_path))


def load_data(data_path: Union[str, Path], format: str = "auto") -> Union[List, Dict]:
    """Load training or inference data in pickle/text form."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if format == "auto":
        if path.suffix == ".pkl" or ".pkl.gz" in path.name:
            format = "pickle"
        elif path.suffix in [".txt", ".tsv"]:
            format = "text"
        else:
            with open(path, "rb") as f:
                header = f.read(4)
                if header[:2] == b"\x1f\x8b":  # gzip
                    format = "pickle"
                elif b"\x80" in header:
                    format = "pickle"
                else:
                    format = "text"

    logger.debug(f"Loading data from {path} as format: {format}")

    if format == "pickle":
        opener = gzip.open if path.suffix.endswith(".gz") or ".gz" in path.suffixes else open
        with opener(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data) if hasattr(data, '__len__') else 'unknown'} items from pickle file")
        return data

    elif format == "text":
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    data.append({"sequence": parts[0], "labels": parts[1].split()})
        logger.info(f"Loaded {len(data)} sequences from text file")
        return data

    raise ValueError(f"Unsupported data format: {format}")


# ---------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------
def run_pipeline(
    command: str,
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    subcommand: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
):
    """
    Execute a full Tempest pipeline command.

    Supports nested subcommands (e.g., train: standard, hybrid, ensemble).
    """
    extra_args = extra_args or {}
    config = load_config(config_path)

    logger.info(f"Executing TEMPEST pipeline: {command}{' ' + subcommand if subcommand else ''}")

    # Dynamic lazy imports
    if command == "train":
        # Training can have multiple variants
        module = __import__("tempest.train", fromlist=["run_training", "run_ensemble_training"])
        if subcommand in ("hybrid", "hybrid_training"):
            func = getattr(module, "run_hybrid_training", None)
        elif subcommand in ("ensemble", "bma", "ensemble_training"):
            func = getattr(module, "run_ensemble_training", None)
        else:
            func = getattr(module, "run_training", None)
    else:
        # Default dispatch map
        dispatch_map = {
            "simulate": lambda: __import__("tempest.simulate", fromlist=["run_simulation"]).run_simulation,
            "visualize": lambda: __import__("tempest.visualize", fromlist=["run_visualization"]).run_visualization,
        }
        if command not in dispatch_map:
            raise ValueError(f"Unknown TEMPEST command: {command}")
        func = dispatch_map[command]()

    if func is None:
        raise ValueError(f"No valid function found for {command} {subcommand or ''}")

    logger.debug(f"Dispatching to {func.__module__}.{func.__name__}")
    return func(config, output_dir=output_dir, **extra_args)


# ---------------------------------------------------------------------
# Entrypoint wrapper
# ---------------------------------------------------------------------
def main(
    command: str,
    config: Union[str, Path],
    output: Optional[str] = None,
    subcommand: Optional[str] = None,
    **kwargs,
):
    """Unified Python entrypoint for Typer or direct execution."""
    try:
        result = run_pipeline(command, config, output_dir=output, subcommand=subcommand, extra_args=kwargs)
        logger.info(f"TEMPEST {command} {subcommand or ''} completed successfully.")
        return result
    except Exception as e:
        logger.exception(f"TEMPEST {command} {subcommand or ''} failed: {e}")
        raise


# ---------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python -m tempest.main <command> <config_path> [subcommand] [output_dir]")
        sys.exit(1)

    cmd = sys.argv[1]
    cfg = sys.argv[2]
    subcmd = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    out = sys.argv[4] if len(sys.argv) > 4 else None

    main(cmd, cfg, output=out, subcommand=subcmd)