"""
Comprehensive configuration management for the reward modeling platform.
Handles configuration loading, validation, environment variables, and merging.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
from enum import Enum
import argparse

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CONSTITUTIONAL = "constitutional"
    MULTI_OBJECTIVE = "multi_objective"

class TrainingType(Enum):
    """Supported training types."""
    REWARD_MODELING = "reward_modeling"
    DPO = "dpo"
    CONSTITUTIONAL = "constitutional"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "microsoft/DialoGPT-medium"
    model_type: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_function: str = "gelu"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Reward model specific
    num_objectives: int = 1
    use_uncertainty: bool = False
    uncertainty_method: str = "dropout"
    ensemble_size: int = 5
    
    # Constitutional AI specific
    num_principles: int = 10
    principle_weight: float = 1.0
    
    def __post_init__(self):
        # Validate model type
        if self.model_type not in [t.value for t in ModelType]:
            raise ValueError(f"Invalid model_type: {self.model_type}")

@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    data_format: str = "json"
    max_length: int = 512
    tokenizer_name: Optional[str] = None
    
    # Data processing options
    shuffle_train: bool = True
    train_split_ratio: float = 0.8
    eval_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    
    # Synthetic data options
    generate_synthetic: bool = False
    synthetic_samples: int = 1000
    
    # Multi-modal options
    image_dir: Optional[str] = None
    audio_dir: Optional[str] = None
    
    def __post_init__(self):
        # Validate split ratios
        total_ratio = self.train_split_ratio + self.eval_split_ratio + self.test_split_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # Training type specific
    training_type: str = "reward_modeling"
    beta: float = 0.1  # For DPO
    reference_model_path: Optional[str] = None  # For DPO
    constitution_path: Optional[str] = None  # For Constitutional AI
    objective_weights: List[float] = field(default_factory=lambda: [1.0])  # For multi-objective
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    deepspeed: Optional[str] = None
    
    def __post_init__(self):
        # Validate training type
        if self.training_type not in [t.value for t in TrainingType]:
            raise ValueError(f"Invalid training_type: {self.training_type}")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "reward_gap", "spearman_correlation"])
    compute_uncertainty: bool = False
    save_predictions: bool = False
    create_visualizations: bool = True
    
    # Benchmark evaluation
    run_anthropic_hh: bool = False
    run_openai_summarization: bool = False
    run_safety_eval: bool = False
    
    # Model comparison
    compare_models: bool = False
    comparison_models: List[str] = field(default_factory=list)

@dataclass
class AgentConfig:
    """Agent configuration."""
    enable_agents: bool = False
    agent_type: str = "dspy"
    reward_model_path: str = "./reward_model"
    max_iterations: int = 5
    reward_threshold: float = 0.5
    temperature: float = 0.7
    use_chain_of_thought: bool = True
    optimize_prompts: bool = True
    
    # Multi-agent settings
    enable_multi_agent: bool = False
    agent_names: List[str] = field(default_factory=list)

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_monitoring: bool = True
    collection_interval: float = 1.0
    use_wandb: bool = False
    wandb_project: str = "reward_modeling"
    wandb_run_name: Optional[str] = None
    
    # Alert thresholds
    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    gpu_memory_threshold: float = 95.0
    loss_spike_threshold: float = 2.0
    training_stall_threshold: float = 300.0
    
    # Export settings
    export_metrics: bool = True
    export_format: str = "json"
    create_plots: bool = True

@dataclass
class APIConfig:
    """API configuration."""
    enable_api: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # Authentication
    enable_auth: bool = False
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Database
    database_url: str = "sqlite:///./reward_modeling.db"
    redis_url: str = "redis://localhost:6379"

@dataclass
class UIConfig:
    """UI configuration."""
    enable_ui: bool = False
    ui_port: int = 3000
    api_base_url: str = "http://localhost:8000"
    theme: str = "dark"
    auto_refresh_interval: int = 5000  # milliseconds

@dataclass
class RewardModelingConfig:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # General settings
    experiment_name: str = "reward_modeling_experiment"
    experiment_description: str = ""
    tags: List[str] = field(default_factory=list)
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    def __post_init__(self):
        # Set tokenizer if not specified
        if self.data.tokenizer_name is None:
            self.data.tokenizer_name = self.model.model_name_or_path
        
        # Set agent reward model path if not specified
        if self.agents.reward_model_path == "./reward_model":
            self.agents.reward_model_path = self.training.output_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardModelingConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        agents_config = AgentConfig(**config_dict.get('agents', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        api_config = APIConfig(**config_dict.get('api', {}))
        ui_config = UIConfig(**config_dict.get('ui', {}))
        
        # Extract general settings
        general_keys = ['experiment_name', 'experiment_description', 'tags', 'seed', 'device']
        general_config = {k: config_dict.get(k) for k in general_keys if k in config_dict}
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            agents=agents_config,
            monitoring=monitoring_config,
            api=api_config,
            ui=ui_config,
            **general_config
        )
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'RewardModelingConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)

class ConfigManager:
    """Configuration manager with environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
        else:
            self.config = RewardModelingConfig()
    
    def load_config(self, path: str) -> None:
        """Load configuration from file."""
        self.config = RewardModelingConfig.from_file(path)
        self.config_path = path
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        self.config.save(save_path)
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Model config overrides
        if os.getenv('RM_MODEL_NAME'):
            self.config.model.model_name_or_path = os.getenv('RM_MODEL_NAME')
        if os.getenv('RM_MODEL_TYPE'):
            self.config.model.model_type = os.getenv('RM_MODEL_TYPE')
        
        # Training config overrides
        if os.getenv('RM_LEARNING_RATE'):
            self.config.training.learning_rate = float(os.getenv('RM_LEARNING_RATE'))
        if os.getenv('RM_BATCH_SIZE'):
            self.config.training.per_device_train_batch_size = int(os.getenv('RM_BATCH_SIZE'))
        if os.getenv('RM_EPOCHS'):
            self.config.training.num_train_epochs = int(os.getenv('RM_EPOCHS'))
        if os.getenv('RM_OUTPUT_DIR'):
            self.config.training.output_dir = os.getenv('RM_OUTPUT_DIR')
        
        # Data config overrides
        if os.getenv('RM_TRAIN_DATA'):
            self.config.data.train_data_path = os.getenv('RM_TRAIN_DATA')
        if os.getenv('RM_EVAL_DATA'):
            self.config.data.eval_data_path = os.getenv('RM_EVAL_DATA')
        
        # API config overrides
        if os.getenv('RM_API_HOST'):
            self.config.api.host = os.getenv('RM_API_HOST')
        if os.getenv('RM_API_PORT'):
            self.config.api.port = int(os.getenv('RM_API_PORT'))
        if os.getenv('RM_DATABASE_URL'):
            self.config.api.database_url = os.getenv('RM_DATABASE_URL')
        if os.getenv('RM_REDIS_URL'):
            self.config.api.redis_url = os.getenv('RM_REDIS_URL')
        
        # Monitoring config overrides
        if os.getenv('RM_WANDB_PROJECT'):
            self.config.monitoring.wandb_project = os.getenv('RM_WANDB_PROJECT')
        if os.getenv('RM_WANDB_API_KEY'):
            os.environ['WANDB_API_KEY'] = os.getenv('RM_WANDB_API_KEY')
        
        logger.info("Applied environment variable overrides")
    
    def merge_configs(self, other_config: Union[Dict[str, Any], RewardModelingConfig]) -> None:
        """Merge another configuration into this one."""
        if isinstance(other_config, RewardModelingConfig):
            other_dict = other_config.to_dict()
        else:
            other_dict = other_config
        
        # Recursive merge
        current_dict = self.config.to_dict()
        merged_dict = self._recursive_merge(current_dict, other_dict)
        
        self.config = RewardModelingConfig.from_dict(merged_dict)
        logger.info("Merged configurations")
    
    def _recursive_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate model configuration
        if not self.config.model.model_name_or_path:
            issues.append("Model name or path is required")
        
        # Validate data configuration
        if not self.config.data.train_data_path and not self.config.data.generate_synthetic:
            issues.append("Either train_data_path must be specified or generate_synthetic must be True")
        
        # Validate training configuration
        if self.config.training.per_device_train_batch_size <= 0:
            issues.append("Train batch size must be positive")
        
        if self.config.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        # Validate file paths
        if self.config.data.train_data_path and not Path(self.config.data.train_data_path).exists():
            issues.append(f"Train data file not found: {self.config.data.train_data_path}")
        
        if self.config.data.eval_data_path and not Path(self.config.data.eval_data_path).exists():
            issues.append(f"Eval data file not found: {self.config.data.eval_data_path}")
        
        # Validate agent configuration
        if self.config.agents.enable_agents and not self.config.agents.reward_model_path:
            issues.append("Agent reward model path is required when agents are enabled")
        
        return issues
    
    def get_config(self) -> RewardModelingConfig:
        """Get the current configuration."""
        return self.config

def create_default_config() -> RewardModelingConfig:
    """Create a default configuration."""
    return RewardModelingConfig()

def create_config_from_args(args: argparse.Namespace) -> RewardModelingConfig:
    """Create configuration from command line arguments."""
    config = RewardModelingConfig()
    
    # Map common arguments
    if hasattr(args, 'model_name') and args.model_name:
        config.model.model_name_or_path = args.model_name
    
    if hasattr(args, 'train_data') and args.train_data:
        config.data.train_data_path = args.train_data
    
    if hasattr(args, 'eval_data') and args.eval_data:
        config.data.eval_data_path = args.eval_data
    
    if hasattr(args, 'output_dir') and args.output_dir:
        config.training.output_dir = args.output_dir
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    
    if hasattr(args, 'epochs') and args.epochs:
        config.training.num_train_epochs = args.epochs
    
    if hasattr(args, 'seed') and args.seed:
        config.seed = args.seed
    
    return config

def setup_logging_from_config(config: RewardModelingConfig) -> None:
    """Setup logging based on configuration."""
    log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(config.training.output_dir) / "training.log")
        ]
    )

# Example configurations
def get_quick_test_config() -> RewardModelingConfig:
    """Get configuration for quick testing."""
    config = RewardModelingConfig()
    config.model.model_name_or_path = "microsoft/DialoGPT-small"
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 4
    config.training.eval_steps = 50
    config.training.save_steps = 50
    config.data.generate_synthetic = True
    config.data.synthetic_samples = 100
    config.monitoring.enable_monitoring = False
    return config

def get_production_config() -> RewardModelingConfig:
    """Get configuration for production training."""
    config = RewardModelingConfig()
    config.model.model_name_or_path = "microsoft/DialoGPT-large"
    config.training.num_train_epochs = 10
    config.training.per_device_train_batch_size = 16
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 1e-5
    config.training.warmup_ratio = 0.1
    config.training.weight_decay = 0.01
    config.training.fp16 = True
    config.monitoring.enable_monitoring = True
    config.monitoring.use_wandb = True
    config.evaluation.create_visualizations = True
    config.api.enable_api = True
    config.ui.enable_ui = True
    return config

# CLI argument parser
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Reward Modeling Platform")
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--config-template', choices=['quick', 'production'], 
                       help='Use a predefined configuration template')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, help='Model name or path')
    parser.add_argument('--model-type', choices=['transformer', 'ensemble', 'constitutional', 'multi_objective'],
                       help='Type of reward model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, help='Path to training data')
    parser.add_argument('--eval-data', type=str, help='Path to evaluation data')
    parser.add_argument('--data-format', choices=['json', 'jsonl', 'csv'], default='json',
                       help='Data format')
    
    # Training arguments
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--training-type', choices=['reward_modeling', 'dpo', 'constitutional', 'multi_objective'],
                       help='Type of training')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    
    # Action arguments
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--serve-api', action='store_true', help='Start API server')
    parser.add_argument('--serve-ui', action='store_true', help='Start UI server')
    
    return parser 