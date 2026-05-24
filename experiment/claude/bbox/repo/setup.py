# setup.py
# BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models
#
# Package setup for the BBox-Adapter reproduction codebase.
#
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# reference_grounding: paperbench_ref_005 toxigen/alice.py
# reference_grounding: paperbench_ref_006 readme.md
# reference_grounding: paperbench_ref_006 research/readme_exp.md
#
# Method obligations:
#   - Keep external simulator dependencies lazy and provide readiness checks;
#     code generation must not require running full training.
#   - Declare lightweight core dependencies separately from optional heavy
#     simulator/training dependencies when possible.
#   - Preserve dataset split ratios from paper (GSM8K, StrategyQA, TruthfulQA,
#     ScienceQA, ToxiGen).
#   - Maintain API compatibility for black-box LLMs (OpenAI, Azure, HuggingFace).
#   - Expose environment/task registry entries for QA benchmarks.
#
# Dataset/environment registry coverage:
#   Benchmarks: gsm8k, strategyqa, truthfulqa, scienceqa, toxigen
#   Models: gpt-3.5-turbo, davinci-002, Mixtral-8x7B-v0, BERT-based adapter
#
# Evidence bundle:
#   environment_setup <- paperbench_ref_006:research/readme_exp.md
#   environment_setup <- paperbench_ref_006:readme.md
#   environment_setup <- paperbench_ref_005:toxigen/alice.py
#   environment_setup <- paperbench_ref_002:src/models/qa/transformer_qa.py

from setuptools import setup, find_packages
import os

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Long description from README
# ---------------------------------------------------------------------------
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = (
        "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models. "
        "Energy-based online adaptation framework for black-box LLMs."
    )

# ---------------------------------------------------------------------------
# Core (lightweight) dependencies
# These must be satisfiable in a minimal code-only smoke environment.
# Heavy ML/training packages are listed under extras_require["training"].
# ---------------------------------------------------------------------------
INSTALL_REQUIRES = [
    # Configuration and serialization
    "pyyaml>=6.0",
    "omegaconf>=2.3.0",
    # HTTP / API clients for black-box LLM access
    "requests>=2.28.0",
    "httpx>=0.24.0",
    # CLI and environment utilities
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "tqdm>=4.65.0",
    # Data handling
    "numpy>=1.23.0",
    # Typing back-compat
    "typing_extensions>=4.5.0",
]

# ---------------------------------------------------------------------------
# Optional OpenAI / Azure API extras
# ---------------------------------------------------------------------------
OPENAI_REQUIRES = [
    "openai>=1.0.0",
    "tiktoken>=0.4.0",
    "azure-identity>=1.13.0",
    "azure-ai-textanalytics>=5.3.0",
]

# ---------------------------------------------------------------------------
# Optional HuggingFace / datasets extras
# (needed by dataset loaders, but lazy-imported at runtime)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------
DATASETS_REQUIRES = [
    "datasets>=2.14.0",
    "huggingface-hub>=0.16.0",
]

# ---------------------------------------------------------------------------
# Optional heavy training dependencies (energy model, NCE loss, adapter)
# These are imported lazily inside training/adapter code.
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# reference_grounding: paperbench_ref_005 toxigen/alice.py
# ---------------------------------------------------------------------------
TRAINING_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "sentencepiece>=0.1.99",
    "scikit-learn>=1.2.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
]

# ---------------------------------------------------------------------------
# Optional evaluation / metrics dependencies
# ---------------------------------------------------------------------------
EVAL_REQUIRES = [
    "sacrebleu>=2.3.0",
    "rouge-score>=0.1.2",
    "evaluate>=0.4.0",
]

# ---------------------------------------------------------------------------
# Optional toxigen / content-safety dependencies
# reference_grounding: paperbench_ref_005 toxigen/alice.py
# ---------------------------------------------------------------------------
TOXIGEN_REQUIRES = [
    "detoxify>=0.5.1",
]

# ---------------------------------------------------------------------------
# Optional plotting / reporting dependencies (fully lazy)
# ---------------------------------------------------------------------------
REPORTING_REQUIRES = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

# ---------------------------------------------------------------------------
# Dev / test extras
# ---------------------------------------------------------------------------
DEV_REQUIRES = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.3.0",
]

# ---------------------------------------------------------------------------
# Full "all" extras bundle
# ---------------------------------------------------------------------------
ALL_REQUIRES = (
    OPENAI_REQUIRES
    + DATASETS_REQUIRES
    + TRAINING_REQUIRES
    + EVAL_REQUIRES
    + TOXIGEN_REQUIRES
    + REPORTING_REQUIRES
)

setup(
    name="bbox-adapter",
    version=VERSION,
    author="BBox-Adapter Authors",
    author_email="",
    description=(
        "BBox-Adapter: Lightweight energy-based online adaptation for "
        "black-box large language models"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="MIT",
    python_requires=">=3.9",
    # ---------------------------------------------------------------------------
    # Package discovery
    # Finds: src/bbox_adapter, src/data, src/training, src/inference,
    #        src/evaluation, src/config, src/utils, src/methods, src/models,
    #        src/experiments, src/reporting, src/datasets
    # ---------------------------------------------------------------------------
    packages=find_packages(where="src") + find_packages(where="."),
    package_dir={
        # Root-level packages (scripts, tests, configs exposed as importable)
        "": ".",
    },
    # Also expose src/ sub-packages directly
    # setuptools find_packages will pick up src/* if src is on sys.path at runtime;
    # users should install with `pip install -e .` or add src to PYTHONPATH.
    include_package_data=True,
    package_data={
        "": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "*.md",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require={
        # Lightweight API-only mode (no training)
        "openai": OPENAI_REQUIRES,
        "datasets": DATASETS_REQUIRES,
        # Full training stack
        "training": TRAINING_REQUIRES,
        # Evaluation metrics
        "eval": EVAL_REQUIRES,
        # ToxiGen toxicity detection
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        "toxigen": TOXIGEN_REQUIRES,
        # Reporting / plotting (lazy)
        "reporting": REPORTING_REQUIRES,
        # Dev / test
        "dev": DEV_REQUIRES,
        # Everything
        "all": ALL_REQUIRES + DEV_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            # Main experiment runner
            "bbox-run=scripts.run_experiment:main",
            # Train adapter
            "bbox-train=scripts.train_adapter:main",
            # Evaluate adapter
            "bbox-eval=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "black-box LLM",
        "adaptation",
        "energy-based model",
        "NCE loss",
        "online learning",
        "QA benchmarks",
        "GSM8K",
        "StrategyQA",
        "TruthfulQA",
        "ScienceQA",
        "ToxiGen",
    ],
    zip_safe=False,
)