# tempest/utils/logging_utils.py
"""
Rich-based logging utilities for Tempest.

Ensures all modules share a consistent RichHandler setup
and suppresses TensorFlow C++ verbosity globally.
"""

import logging
import os
from rich.console import Console
from rich.logging import RichHandler

# Create one global console shared across modules
console = Console(log_path=False)

def status(msg: str):
    """Simple shared status context."""
    return console.status(f"[cyan]{msg}[/cyan]", spinner="dots")

def setup_rich_logging(level: str = "INFO") -> None:
    """Initialize RichHandler-based logging if not already active."""
    root_logger = logging.getLogger()

    # Only configure if not already using RichHandler
    if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(
                rich_tracebacks=True,
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
                keywords=["INFO", "WARNING", "ERROR"],
                )],
        )

        # Silence TensorFlow C++ backend chatter
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        root_logger.debug("Rich logging initialized with level %s", level)