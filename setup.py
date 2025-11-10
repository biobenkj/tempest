#!/usr/bin/env python
"""
Setup script for Tempest.

This file exists only for backward compatibility with environments
that do not yet support PEP 517/518 builds.
The authoritative configuration is defined in pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# ---------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text(encoding="utf-8")
    if (this_directory / "README.md").exists()
    else ""
)

# ---------------------------------------------------------------------
# Core runtime dependencies (harmonized with pyproject.toml)
# ---------------------------------------------------------------------
install_requires = [
    # Deep learning
    "tensorflow>=2.10.0,<2.16.0",
    "tensorflow-addons>=0.20.0",  # required for CRF layers
    "tf2crf>=0.1.33",

    # Scientific computing
    "numpy>=1.21.0,<2.0.0",
    "pandas>=2.0.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.2.0",

    # Bioinformatics
    "biopython>=1.80",

    # Visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "plotly>=5.18.0",

    # CLI framework (Typer-based)
    "typer>=0.9.0",
    "rich>=13.0.0",
    "shellingham>=1.5.4",
    "colorama>=0.4.6",

    # Configuration & data
    "pyyaml>=6.0",
    "pydantic>=2.0.0",

    # Utilities
    "tqdm>=4.60.0",
    "joblib>=1.2.0",
    "h5py>=3.8.0",
    "pickle-mixin>=1.0.2",
]

# ---------------------------------------------------------------------
# Optional dependency groups
# ---------------------------------------------------------------------
extras_require = {
    "dev": [
        "pytest>=7.0",
        "pytest-cov>=4.0",
        "pytest-mock>=3.10.0",
        "black>=22.0",
        "flake8>=5.0",
        "mypy>=0.990",
        "isort>=5.12.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=5.0",
        "sphinx-rtd-theme>=1.0",
        "sphinx-autodoc-typehints>=1.19",
        "myst-parser>=1.0.0",
    ],
    "viz": [
        "plotly>=5.0.0",
        "dash>=2.0.0",
    ],
}

# Combined convenience group
extras_require["all"] = sorted({pkg for group in extras_require.values() for pkg in group})

# ---------------------------------------------------------------------
# Setup configuration
# ---------------------------------------------------------------------
setup(
    name="tempest-bio",
    version="0.3.0",
    author="Ben Johnson",
    author_email="ben.johnson@vai.org",
    description="Long read RNA-seq sequence annotation using length-constrained CRFs and BMA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biobenkj/tempest",
    project_urls={
        "Bug Tracker": "https://github.com/biobenkj/tempest/issues",
        "Documentation": "https://github.com/biobenkj/tempest/tree/main/docs",
        "Source Code": "https://github.com/biobenkj/tempest",
    },
    packages=find_packages(exclude=["tests*", "examples*", "backup*", "docs*"]),
    package_data={"tempest": ["config/*.yaml", "data/*.yaml", "templates/*.yaml"]},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8,<3.12",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "tempest=tempest.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "bioinformatics",
        "deep learning",
        "sequence annotation",
        "CRF",
        "nanopore",
        "genomics",
        "machine learning",
        "CLI",
    ],
)