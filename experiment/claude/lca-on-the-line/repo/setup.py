#!/usr/bin/env python3
"""
LCA-on-the-Line: Setup Configuration
Paper: LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

This setup.py follows the method obligation:
"Declare lightweight core dependencies separately from optional heavy simulator/training dependencies when possible."

Package structure:
- Core: Lightweight analysis, metrics, hierarchy computation (numpy, scipy, networkx)
- Training: Heavy ML frameworks (torch, torchvision, timm)
- VLM: Vision-language model dependencies (CLIP, OpenCLIP)
- Visualization: Plotting and figure generation (matplotlib, seaborn)
- Dev: Development and testing tools (pytest, black, mypy)

reference_grounding: paperbench_ref_001 .github/workflows/prototype-tests-linux-gpu.yml
reference_grounding: paperbench_ref_006 hier_jax.py
reference_grounding: paperbench_ref_006 extract_clip.ipynb
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Core lightweight dependencies - required for basic functionality
# reference_grounding: paperbench_ref_006 hier_jax.py
# These dependencies enable LCA distance computation, hierarchy processing,
# and correlation analysis without requiring heavy ML frameworks
CORE_REQUIREMENTS = [
    "numpy>=1.23.0,<2.0.0",
    "scipy>=1.10.0",
    "pandas>=1.5.0",
    "networkx>=3.0",
    "nltk>=3.8",
    "scikit-learn>=1.2.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

# Heavy ML training dependencies - optional for evaluation-only usage
# reference_grounding: paperbench_ref_001 .github/workflows/prototype-tests-linux-gpu.yml
# Required for: Model loading (36 VMs), training soft-label models, fine-tuning
TRAINING_REQUIREMENTS = [
    "torch>=1.13.0,<3.0.0",
    "torchvision>=0.14.0,<1.0.0",
    "timm>=0.9.0",
    "transformers>=4.30.0",
]

# Vision-language model dependencies - optional for VLM evaluation
# reference_grounding: paperbench_ref_006 extract_clip.ipynb
# Required for: Loading 39 VLMs (CLIP, OpenCLIP variants)
VLM_REQUIREMENTS = [
    "open-clip-torch>=2.20.0",
    "ftfy>=6.1.0",
    "regex>=2023.0.0",
]

# Visualization dependencies - optional for figure generation
VISUALIZATION_REQUIREMENTS = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
]

# Development dependencies - optional for testing and code quality
DEV_REQUIREMENTS = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "types-PyYAML>=6.0.0",
    "types-tqdm>=4.65.0",
]

# All dependencies for full installation
ALL_REQUIREMENTS = (
    CORE_REQUIREMENTS
    + TRAINING_REQUIREMENTS
    + VLM_REQUIREMENTS
    + VISUALIZATION_REQUIREMENTS
    + DEV_REQUIREMENTS
)

setup(
    name="lca-on-the-line",
    version="1.0.0",
    author="LCA-on-the-Line Authors",
    description="Benchmarking Out-of-Distribution Generalization with Class Taxonomies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paperbench/lca-on-the-line",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "training": TRAINING_REQUIREMENTS,
        "vlm": VLM_REQUIREMENTS,
        "visualization": VISUALIZATION_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
        "all": ALL_REQUIREMENTS,
    },
    entry_points={
        "console_scripts": [
            "lca-benchmark=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "out-of-distribution",
        "generalization",
        "taxonomy",
        "hierarchical classification",
        "imagenet",
        "vision models",
        "vision-language models",
        "lca distance",
        "ood robustness",
        "model evaluation",
    ],
    license="MIT",
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.csv", "*.txt"],
    },
    zip_safe=False,
)