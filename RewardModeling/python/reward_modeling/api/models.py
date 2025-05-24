"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
class ModelType(str, Enum):
    reward_model = "reward_model"
    preference_model = "preference_model"
    dpo_model = "dpo_model"

class TrainingType(str, Enum):
    reward_modeling = "reward_modeling"
    dpo = "dpo"
    constitutional_ai = "constitutional_ai"

class AgentType(str, Enum):
    reward_guided = "reward_guided"
    constitutional = "constitutional"
    dspy_optimized = "dspy_optimized"

class ExperimentStatus(str, Enum):
    created = "created"
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

# Model Management Models
class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., description="Path to the model directory")
    model_type: ModelType = Field(..., description="Type of model to load")
    device: str = Field(default="cuda", description="Device to load model on")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Model path cannot be empty")
        return v

class ModelLoadResponse(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the loaded model")
    status: str = Field(..., description="Loading status")
    message: str = Field(..., description="Status message")

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    status: str
    device: str
    loaded_at: str

class PredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to get predictions for")
    max_length: int = Field(default=512, description="Maximum sequence length")
    batch_size: int = Field(default=16, description="Batch size for processing")

class PredictionResult(BaseModel):
    text: str
    reward: float
    confidence: float

class PredictionResponse(BaseModel):
    model_id: str
    predictions: List[PredictionResult]
    timestamp: str

# Training Models
class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Base model name or path")
    dataset_path: str = Field(..., description="Path to training dataset")
    training_type: TrainingType = Field(..., description="Type of training to perform")
    output_dir: str = Field(..., description="Output directory for model artifacts")
    
    # Training parameters
    num_epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=5e-5, gt=0, le=1e-2)
    warmup_steps: int = Field(default=0, ge=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    
    # Advanced configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional training configuration")

class TrainingResponse(BaseModel):
    experiment_id: str
    status: str
    message: str

class ExperimentStatus(BaseModel):
    experiment_id: str
    status: ExperimentStatus
    progress: float = Field(ge=0, le=1)
    metrics: Optional[Dict[str, float]] = None
    created_at: str
    config: Dict[str, Any]

# Evaluation Models
class EvaluationRequest(BaseModel):
    model_id: str = Field(..., description="ID of model to evaluate")
    dataset_path: str = Field(..., description="Path to evaluation dataset")
    metrics: List[str] = Field(default=["accuracy", "f1", "reward_gap"], description="Metrics to compute")
    batch_size: int = Field(default=32, ge=1, le=512)

class EvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    message: str

class EvaluationResult(BaseModel):
    evaluation_id: str
    model_id: str
    metrics: Dict[str, float]
    completed_at: str

# Dataset Models
class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    path: str
    sample_count: int
    dataset_type: str

class DatasetPreview(BaseModel):
    dataset_id: str
    total_samples: int
    preview_samples: List[Dict[str, Any]]

class PreferencePair(BaseModel):
    id: str
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict[str, str]] = None

# Agent Models
class AgentCreateRequest(BaseModel):
    agent_type: AgentType = Field(..., description="Type of agent to create")
    reward_model_path: str = Field(..., description="Path to reward model")
    task_module_config: Optional[Dict[str, Any]] = Field(default=None, description="Task module configuration")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

class AgentCreateResponse(BaseModel):
    agent_id: str
    agent_type: str
    status: str

class AgentRunRequest(BaseModel):
    inputs: Dict[str, Any] = Field(..., description="Input data for agent")
    max_iterations: int = Field(default=1, ge=1, le=10)

class AgentRunResponse(BaseModel):
    agent_id: str
    result: Any
    timestamp: str

# System Models
class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    active_models: int
    active_experiments: int

class ExperimentMetrics(BaseModel):
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    reward_gap: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    epoch: Optional[int] = None
    step: Optional[int] = None

# Advanced Models
class DistributedTrainingRequest(BaseModel):
    nodes: List[str] = Field(..., description="List of worker nodes")
    world_size: int = Field(..., ge=2, description="Total number of processes")
    backend: str = Field(default="nccl", description="Distributed backend")
    
class ConstitutionalAIRequest(BaseModel):
    constitution: List[str] = Field(..., description="List of constitutional principles")
    critique_model: Optional[str] = Field(None, description="Model for critiquing responses")
    revision_model: Optional[str] = Field(None, description="Model for revising responses")
    
class MultiModalRequest(BaseModel):
    text_inputs: List[str]
    image_inputs: Optional[List[str]] = None
    audio_inputs: Optional[List[str]] = None
    
class BenchmarkRequest(BaseModel):
    benchmark_name: str = Field(..., description="Name of benchmark to run")
    model_ids: List[str] = Field(..., description="Models to benchmark")
    config: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkResult(BaseModel):
    benchmark_name: str
    model_results: Dict[str, Dict[str, float]]
    completed_at: str

# Real-time Updates
class ExperimentUpdate(BaseModel):
    experiment_id: str
    update_type: str  # "metrics", "status", "log"
    data: Dict[str, Any]
    timestamp: str

class LogEntry(BaseModel):
    level: str
    message: str
    timestamp: str
    source: str

# Configuration Models
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "reward_modeling"
    username: str = "rmep"
    password: str

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

class ValidationError(BaseModel):
    field: str
    message: str
    value: Any 