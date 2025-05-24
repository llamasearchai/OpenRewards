"""
Setup script for Reward Modeling Engineering Platform (RMEP)

A comprehensive, production-ready platform for training, evaluating, and deploying
reward models for reinforcement learning from human feedback (RLHF).
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]
    return []

# Core requirements
install_requires = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "accelerate>=0.24.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "datasets>=2.14.0",
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
    "jsonlines>=4.0.0",
    "pyyaml>=6.0.0",
    "click>=8.1.0",
    "rich>=13.7.0",
    "tqdm>=4.66.0",
    "sqlalchemy>=2.0.0",
    "redis>=5.0.0",
]

# Optional dependencies for different use cases
extras_require = {
    "full": read_requirements("requirements.txt"),
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
        "black>=23.10.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.1.0",
    ],
    "distributed": [
        "deepspeed>=0.12.0",
        "fairscale>=0.4.13",
        "ray[tune]>=2.8.0",
    ],
    "optimization": [
        "optuna>=3.4.0",
        "numba>=0.58.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
    ],
    "evaluation": [
        "rouge-score>=0.1.2",
        "bleu>=0.2.2",
        "bert-score>=0.3.13",
        "spacy>=3.7.0",
        "nltk>=3.8.0",
    ],
    "visualization": [
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "plotly>=5.17.0",
    ],
    "cloud": [
        "boto3>=1.34.0",
        "google-cloud-storage>=2.10.0",
    ],
    "production": [
        "psycopg2-binary>=2.9.0",
        "prometheus-client>=0.19.0",
        "structlog>=23.2.0",
        "bcrypt>=4.1.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.0",
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="reward-modeling-platform",
    version="1.0.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Production-ready platform for training and deploying reward models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenRewards",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Package data
    include_package_data=True,
    package_data={
        "reward_modeling": [
            "configs/*.yaml",
            "configs/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    
    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "rmep-train=reward_modeling.scripts.train:main",
            "rmep-evaluate=reward_modeling.scripts.evaluate:main",
            "rmep-serve=reward_modeling.scripts.serve:main",
            "rmep-config=reward_modeling.scripts.config:main",
            "rmep-data=reward_modeling.scripts.data_processing:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Framework :: FastAPI",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "machine learning", "deep learning", "nlp", "reward modeling", 
        "reinforcement learning", "rlhf", "preference learning", 
        "constitutional ai", "dpo", "transformers", "pytorch"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/llamasearchai/OpenRewards/issues",
        "Source": "https://github.com/llamasearchai/OpenRewards",
        "Documentation": "https://openrewards.readthedocs.io/",
        "Changelog": "https://github.com/llamasearchai/OpenRewards/blob/main/CHANGELOG.md",
    },
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
    
    # Test suite
    test_suite="tests",
    tests_require=extras_require["dev"],
) 