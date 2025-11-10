"""
Tempest Main Entry Point

This module serves as the bridge between the Typer CLI and the core functionality.
It's invoked by the CLI commands and handles the actual execution of pipelines.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import logging
import pickle
import gzip
from tempest.config import TempestConfig, load_config as config_load_config

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> TempestConfig:
    """
    Load a configuration file (YAML or JSON) into a TempestConfig dataclass.
    
    Delegates to the config module's load_config function for consistency.
    """
    return config_load_config(str(config_path))


def load_data(data_path: Union[str, Path], format: str = "auto") -> Union[List, Dict]:
    """
    Load training data from various formats including pickle and text.
    
    Args:
        data_path: Path to the data file
        format: Format of the data ('auto', 'pickle', 'text')
    
    Returns:
        Loaded data (format depends on file type)
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Auto-detect format based on extension
    if format == "auto":
        if path.suffix == ".pkl" or ".pkl.gz" in path.name:
            format = "pickle"
        elif path.suffix in [".txt", ".tsv"]:
            format = "text"
        else:
            # Try to detect by reading first few bytes
            try:
                with open(path, 'rb') as f:
                    header = f.read(4)
                    if header[:2] == b'\x1f\x8b':  # gzip magic number
                        format = "pickle"
                    elif b'\x80' in header:  # pickle protocol marker
                        format = "pickle"
                    else:
                        format = "text"
            except:
                format = "text"
    
    logger.debug(f"Loading data from {path} as format: {format}")
    
    if format == "pickle":
        if path.suffix == ".gz" or ".gz" in path.suffixes:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        logger.info(f"Loaded {len(data) if hasattr(data, '__len__') else 'unknown'} items from pickle file")
        return data
    
    elif format == "text":
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    data.append({
                        'sequence': parts[0],
                        'labels': parts[1].split() if len(parts) > 1 else []
                    })
        logger.info(f"Loaded {len(data)} sequences from text file")
        return data
    
    else:
        raise ValueError(f"Unsupported data format: {format}")


def run_pipeline(
    command: str,
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    extra_args: Optional[Dict[str, Any]] = None,
):
    """
    Main pipeline runner that dispatches to appropriate subcommand functions.
    
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
        Additional arguments passed to the subcommand.
    """
    if extra_args is None:
        extra_args = {}
    
    config = load_config(config_path)
    logger.info(f"Executing TEMPEST pipeline: {command}")
    
    # Import runners lazily to avoid circular dependencies
    dispatch_map = {
        "simulate": lambda: __import__('tempest.simulate', fromlist=['run_simulation']).run_simulation,
        "train": lambda: __import__('tempest.training', fromlist=['run_training']).run_training,
        "evaluate": lambda: __import__('tempest.evaluate', fromlist=['run_evaluation']).run_evaluation,
        "combine": lambda: __import__('tempest.combine', fromlist=['run_combination']).run_combination,
        "compare": lambda: __import__('tempest.compare', fromlist=['run_comparison']).run_comparison,
        "demux": lambda: __import__('tempest.demux', fromlist=['run_demultiplexing']).run_demultiplexing,
        "visualize": lambda: __import__('tempest.visualize', fromlist=['run_visualization']).run_visualization,
    }
    
    if command not in dispatch_map:
        raise ValueError(f"Unknown TEMPEST command: {command}")
    
    func = dispatch_map[command]()
    logger.debug(f"Dispatching to function: {func.__module__}.{func.__name__}")
    
    return func(config, output_dir=output_dir, **extra_args)


def main(
    command: str,
    config: Union[str, Path],
    output: Optional[str] = None,
    **kwargs,
):
    """
    Execute a TEMPEST command directly from Python.
    
    This function serves as the entry point when called from the Typer CLI
    or when used programmatically.
    
    Example
    -------
    >>> from tempest.main import main
    >>> main("train", "configs/default.yaml", output="models/")
    """
    try:
        result = run_pipeline(command, config, output_dir=output, extra_args=kwargs)
        logger.info(f"TEMPEST {command} completed successfully.")
        return result
    except Exception as e:
        logger.exception(f"TEMPEST {command} failed: {e}")
        raise


if __name__ == "__main__":
    # Direct CLI bypass for quick tests
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python -m tempest.main <command> <config_path> [output_dir]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    cfg = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(cmd, cfg, output=out)
