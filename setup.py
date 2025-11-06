"""
Setup script for Tempest package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
core_requirements = [
    "tensorflow>=2.10.0,<2.16.0",
    "tf2crf>=0.1.33",
    "numpy>=1.21.0,<2.0.0",
    "pandas>=2.0.0",
    "biopython>=1.80",
    "scikit-learn>=1.2.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "tqdm>=4.60.0",
]

setup(
    name="tempest-bio",
    version="0.2.0",
    author="Ben Johnson",
    author_email="ben.johnson@vai.org",
    description="Modular sequence annotation using length-constrained CRFs with PWM priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biobenkj/tempest",
    project_urls={
        "Bug Tracker": "https://github.com/biobenkj/tempest/issues",
        "Documentation": "https://github.com/biobenkj/tempest/tree/main/docs",
        "Source Code": "https://github.com/biobenkj/tempest",
    },
    packages=find_packages(exclude=['tests', 'examples', 'backup', 'docs']),
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
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        'tf-addons': [
            'tensorflow-addons>=0.20.0; python_version < "3.11"',
        ],
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=0.990',
        ],
        'docs': [
            'sphinx>=5.0',
            'sphinx-rtd-theme>=1.0',
            'sphinx-autodoc-typehints>=1.19',
        ],
    },
    include_package_data=True,
    package_data={
        'tempest': ['config/*.yaml'],
    },
    scripts=['bin/tempest'],
    keywords=[
        'bioinformatics',
        'deep learning',
        'sequence annotation',
        'CRF',
        'nanopore',
        'genomics',
        'machine learning',
    ],
)
