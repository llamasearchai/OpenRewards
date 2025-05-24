# OpenRewards: Advanced Reward Modeling Platform

Production-ready platform for AI alignment research and reward model development.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2+-61dafb.svg)](https://reactjs.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-000000.svg)](https://www.rust-lang.org)

## Overview

OpenRewards is a comprehensive platform for developing, training, and deploying reward models for AI alignment research. The platform implements modern machine learning techniques including Constitutional AI, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).

## Technical Features

### Model Architectures
- Transformer-based reward models with uncertainty quantification
- Constitutional AI models for value alignment
- Ensemble models for improved reliability
- Multi-objective models for complex preference learning
- Support for GPT, Claude, LLaMA, and custom model architectures

### Training Methods
- Bradley-Terry preference learning
- Direct Preference Optimization (DPO) implementation
- Constitutional AI training with safety constraints
- Reinforcement Learning from Human Feedback (RLHF)
- Distributed training with multi-GPU support

### Agent Integration
- DSPy framework integration for agent optimization
- Multi-agent collaboration systems
- Real-time response optimization using reward models
- Agent performance evaluation and improvement

### Evaluation Framework
- Standard metrics: accuracy, correlation, reward gap analysis
- Advanced metrics: calibration error, uncertainty quantification
- Benchmark integration capabilities
- Safety evaluation frameworks
- Statistical analysis tools

### Infrastructure
- FastAPI backend with async support and OpenAPI documentation
- React TypeScript frontend with Material-UI components
- Tauri desktop application for cross-platform deployment
- Rust modules for performance-critical operations
- PostgreSQL database with Redis caching
- Docker containerization support

### Monitoring
- Real-time system metrics (CPU, GPU, memory)
- Experiment tracking with MLflow and Weights & Biases
- Performance monitoring with alerting
- Resource optimization analysis
- Training progress visualization

## Installation

### Prerequisites
- Python 3.9 or higher
- Node.js 18+ with npm
- Rust 1.70+ with Cargo
- Docker and Docker Compose (recommended)
- NVIDIA GPU with CUDA support (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/llamasearchai/OpenRewards.git
   cd OpenRewards
   ```

2. Install Python dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   cd RewardModeling/python
   pip install -e ".[all]"
   ```

3. Start with Docker (recommended):
   ```bash
   cd RewardModeling
   docker-compose --profile development up -d
   ```

   Services will be available at:
   - Web UI: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Jupyter Lab: http://localhost:8888
   - MLflow: http://localhost:5000

## Usage

### Basic Reward Model Training

```python
from reward_modeling.models import TransformerRewardModel, RewardModelConfig
from reward_modeling.training import RewardModelTrainer, TrainingArguments
from reward_modeling.data import PreferenceDataset

# Load dataset
dataset = PreferenceDataset.from_file("data/preferences.json")
train_data, eval_data, _ = dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Configure model
config = RewardModelConfig(
    model_name_or_path="microsoft/DialoGPT-medium",
    use_uncertainty=True,
    dropout_rate=0.1
)
model = TransformerRewardModel(config)

# Set up training
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    use_wandb=True,
    evaluation_strategy="steps",
    eval_steps=500
)

# Train
trainer = RewardModelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data
)

results = trainer.train()
```

### Constitutional AI Training

```python
from reward_modeling.models import ConstitutionalRewardModel
from reward_modeling.training import ConstitutionalTrainer
from reward_modeling.data import ConstitutionalDataset

# Define principles
constitution = [
    "Be helpful and harmless",
    "Respect human autonomy",
    "Be honest and transparent",
    "Avoid harmful content"
]

# Load data and train
dataset = ConstitutionalDataset("data/constitutional_examples.json")
model = ConstitutionalRewardModel(config, constitution=constitution)

trainer = ConstitutionalTrainer(
    model=model,
    constitution=constitution,
    train_dataset=dataset,
    constitutional_weight=0.5
)

trainer.train()
```

### Command Line Training

```bash
# Train with synthetic data
cd RewardModeling/python
python scripts/train.py \
    --config-template quick \
    --output-dir ./experiments/model

# Train with custom data
python scripts/train.py \
    --train-data ./data/preferences.jsonl \
    --model-name microsoft/DialoGPT-medium \
    --training-type reward_modeling \
    --epochs 5 \
    --batch-size 16 \
    --output-dir ./experiments/custom_model
```

## Architecture

The platform follows a modular microservices architecture:

```
Frontend Layer          API Layer              Core Services
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web UI │    │  FastAPI Server │    │ Model Manager   │
│ Tauri Desktop  │ -> │  Authentication │ -> │ Training Engine │
│ API Docs       │    │  Rate Limiting  │    │ Agent Framework │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                      Storage Layer                     │
                ┌─────────────────────────────────────────┘
                │
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ PostgreSQL   │  │    Redis     │  │  S3/MinIO    │
        │   Database   │  │    Cache     │  │   Storage    │
        └──────────────┘  └──────────────┘  └──────────────┘
```

## Development

### Development Setup

```bash
git clone https://github.com/llamasearchai/OpenRewards.git
cd OpenRewards

# Install development dependencies
make install-dev

# Run quality checks
make lint test

# Start development servers
make run-api        # FastAPI backend
make run-frontend   # React development server
```

### Code Quality

The project maintains code quality through:
- Comprehensive testing with pytest (95%+ coverage target)
- Type checking with mypy
- Code formatting with Black and isort
- Linting with flake8
- Security scanning and dependency auditing

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines including:
- Development environment setup
- Code style requirements
- Testing procedures
- Pull request process

## Documentation Structure

- **API Documentation**: Complete REST API reference
- **Model Documentation**: Architecture details and implementation
- **Training Guide**: Advanced training techniques and configuration
- **Deployment Guide**: Production deployment procedures
- **Research Examples**: Academic use cases and methodologies

## Performance Characteristics

| Component | Training Speed | Inference Speed | Memory Usage |
|-----------|---------------|-----------------|--------------|
| Transformer (125M) | 45 samples/sec | 120 inferences/sec | 2.1 GB |
| Constitutional AI (125M) | 38 samples/sec | 95 inferences/sec | 2.8 GB |
| Ensemble (5×125M) | 12 samples/sec | 25 inferences/sec | 8.5 GB |

*Benchmarks measured on NVIDIA A100 GPU with batch size 16*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{openrewards2024,
  title={OpenRewards: Advanced Reward Modeling Platform for AI Alignment},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/llamasearchai/OpenRewards},
  license={MIT}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenRewards/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenRewards/discussions)
- **Email**: nikjois@llamasearch.ai

---

*OpenRewards is developed and maintained by Nik Jois. Contributions from the open source community are welcome.* 