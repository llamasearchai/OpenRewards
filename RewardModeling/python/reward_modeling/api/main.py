"""
FastAPI application for the Reward Modeling Engineering Platform.
Provides REST API endpoints for training, evaluation, and deployment.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import torch
import json
import tempfile
import os
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
import redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..models.reward_model import RewardModel
from ..training.trainer import RewardModelTrainer
from ..training.dpo_trainer import DirectPreferenceOptimizationTrainer, DPOTrainingArguments
from ..data.dataset import PreferenceDataset
from ..evaluation.metrics import RewardModelEvaluator
from ..agents.dspy_agent import RewardGuidedAgent
from ..utils.monitoring import MetricsCollector, ExperimentTracker
from ..utils.config import Config
from .models import *
from .dependencies import get_db, get_redis, verify_token, rate_limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "models": {},
    "experiments": {},
    "background_tasks": {},
    "metrics_collector": None,
    "redis_client": None,
    "db_engine": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Reward Modeling API...")
    
    # Initialize Redis
    app_state["redis_client"] = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True
    )
    
    # Initialize database
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://rmep:password@localhost/reward_modeling")
    app_state["db_engine"] = create_async_engine(database_url)
    
    # Initialize metrics collector
    app_state["metrics_collector"] = MetricsCollector()
    
    # Load any existing models
    await load_existing_models()
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    # Clean up resources
    if app_state["redis_client"]:
        app_state["redis_client"].close()
    if app_state["db_engine"]:
        await app_state["db_engine"].dispose()

# Initialize FastAPI app
app = FastAPI(
    title="Reward Modeling Engineering Platform API",
    description="Comprehensive API for reward modeling, preference learning, and LLM alignment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Health check
@app.get("/health", tags=["system"])
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

# Model Management Endpoints
@app.post("/models/load", response_model=ModelLoadResponse, tags=["models"])
async def load_model(
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Load a reward model for inference."""
    try:
        verify_token(credentials.credentials)
        
        model_id = str(uuid.uuid4())
        
        # Load model in background
        background_tasks.add_task(
            _load_model_background,
            model_id,
            request.model_path,
            request.model_type,
            request.device
        )
        
        return ModelLoadResponse(
            model_id=model_id,
            status="loading",
            message="Model loading started"
        )
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo], tags=["models"])
async def list_models(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """List all loaded models."""
    try:
        verify_token(credentials.credentials)
        return [
            ModelInfo(
                model_id=model_id,
                model_type=info["type"],
                status=info["status"],
                device=info["device"],
                loaded_at=info["loaded_at"]
            )
            for model_id, info in app_state["models"].items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/predict", response_model=PredictionResponse, tags=["models"])
async def predict(
    model_id: str,
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get reward predictions from a loaded model."""
    try:
        verify_token(credentials.credentials)
        
        if model_id not in app_state["models"]:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = app_state["models"][model_id]
        if model_info["status"] != "ready":
            raise HTTPException(status_code=400, detail="Model not ready")
        
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Process predictions
        predictions = []
        for text in request.texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=request.max_length,
                padding="max_length"
            ).to(model_info["device"])
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
                reward = outputs["rewards"].item()
                
            predictions.append(PredictionResult(
                text=text,
                reward=reward,
                confidence=abs(reward)  # Simple confidence measure
            ))
        
        return PredictionResponse(
            model_id=model_id,
            predictions=predictions,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Training Endpoints
@app.post("/training/start", response_model=TrainingResponse, tags=["training"])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Start a new training job."""
    try:
        verify_token(credentials.credentials)
        
        experiment_id = str(uuid.uuid4())
        
        # Validate dataset
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=400, detail="Dataset path not found")
        
        # Create experiment tracker
        tracker = ExperimentTracker(experiment_id)
        app_state["experiments"][experiment_id] = {
            "status": "starting",
            "tracker": tracker,
            "config": request.dict(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Start training in background
        background_tasks.add_task(
            _start_training_background,
            experiment_id,
            request
        )
        
        return TrainingResponse(
            experiment_id=experiment_id,
            status="starting",
            message="Training job started"
        )
        
    except Exception as e:
        logger.error(f"Training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/{experiment_id}", response_model=ExperimentStatus, tags=["training"])
async def get_experiment_status(
    experiment_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get experiment status and metrics."""
    try:
        verify_token(credentials.credentials)
        
        if experiment_id not in app_state["experiments"]:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        experiment = app_state["experiments"][experiment_id]
        tracker = experiment["tracker"]
        
        return ExperimentStatus(
            experiment_id=experiment_id,
            status=experiment["status"],
            progress=tracker.get_progress(),
            metrics=tracker.get_latest_metrics(),
            created_at=experiment["created_at"],
            config=experiment["config"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/{experiment_id}/logs", tags=["training"])
async def get_experiment_logs(
    experiment_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Stream experiment logs."""
    try:
        verify_token(credentials.credentials)
        
        if experiment_id not in app_state["experiments"]:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        def generate_logs():
            tracker = app_state["experiments"][experiment_id]["tracker"]
            for log_entry in tracker.get_logs():
                yield f"data: {json.dumps(log_entry)}\n\n"
        
        return StreamingResponse(
            generate_logs(),
            media_type="text/stream-events"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Evaluation Endpoints
@app.post("/evaluation/start", response_model=EvaluationResponse, tags=["evaluation"])
async def start_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Start model evaluation."""
    try:
        verify_token(credentials.credentials)
        
        eval_id = str(uuid.uuid4())
        
        # Validate model
        if request.model_id not in app_state["models"]:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Start evaluation in background
        background_tasks.add_task(
            _start_evaluation_background,
            eval_id,
            request
        )
        
        return EvaluationResponse(
            evaluation_id=eval_id,
            status="starting",
            message="Evaluation started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dataset Management Endpoints
@app.post("/datasets/upload", response_model=DatasetUploadResponse, tags=["datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = "preference",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Upload a new dataset."""
    try:
        verify_token(credentials.credentials)
        
        # Validate file type
        if not file.filename.endswith(('.jsonl', '.json', '.csv')):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save file
        dataset_id = str(uuid.uuid4())
        dataset_path = f"/tmp/datasets/{dataset_id}_{file.filename}"
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        with open(dataset_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate dataset format
        try:
            dataset = PreferenceDataset.from_file(dataset_path)
            sample_count = len(dataset)
        except Exception as e:
            os.remove(dataset_path)
            raise HTTPException(status_code=400, detail=f"Invalid dataset format: {str(e)}")
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            path=dataset_path,
            sample_count=sample_count,
            dataset_type=dataset_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview, tags=["datasets"])
async def preview_dataset(
    dataset_id: str,
    limit: int = 10,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Preview dataset samples."""
    try:
        verify_token(credentials.credentials)
        
        # Find dataset path (you'd implement proper storage)
        dataset_path = f"/tmp/datasets/{dataset_id}"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = PreferenceDataset.from_file(dataset_path)
        samples = [dataset[i] for i in range(min(limit, len(dataset)))]
        
        return DatasetPreview(
            dataset_id=dataset_id,
            total_samples=len(dataset),
            preview_samples=samples[:limit]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Agent Endpoints
@app.post("/agents/create", response_model=AgentCreateResponse, tags=["agents"])
async def create_agent(
    request: AgentCreateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Create a new reward-guided agent."""
    try:
        verify_token(credentials.credentials)
        
        agent_id = str(uuid.uuid4())
        
        # Create agent based on type
        if request.agent_type == "reward_guided":
            agent = RewardGuidedAgent(
                reward_model_path=request.reward_model_path,
                task_module=None,  # You'd implement task module creation
                num_candidates=request.config.get("num_candidates", 3)
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported agent type")
        
        # Store agent
        app_state["agents"] = app_state.get("agents", {})
        app_state["agents"][agent_id] = {
            "agent": agent,
            "type": request.agent_type,
            "config": request.config,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return AgentCreateResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status="ready"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/run", response_model=AgentRunResponse, tags=["agents"])
async def run_agent(
    agent_id: str,
    request: AgentRunRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Run an agent on input."""
    try:
        verify_token(credentials.credentials)
        
        if agent_id not in app_state.get("agents", {}):
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_info = app_state["agents"][agent_id]
        agent = agent_info["agent"]
        
        # Run agent
        result = agent(**request.inputs)
        
        return AgentRunResponse(
            agent_id=agent_id,
            result=result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Metrics and Monitoring Endpoints
@app.get("/metrics/system", tags=["metrics"])
async def get_system_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get system performance metrics."""
    try:
        verify_token(credentials.credentials)
        
        metrics = app_state["metrics_collector"].get_system_metrics()
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/experiments", tags=["metrics"])
async def get_experiment_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get aggregated experiment metrics."""
    try:
        verify_token(credentials.credentials)
        
        metrics = {}
        for exp_id, exp_info in app_state["experiments"].items():
            tracker = exp_info["tracker"]
            metrics[exp_id] = tracker.get_latest_metrics()
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _load_model_background(model_id: str, model_path: str, model_type: str, device: str):
    """Load model in background."""
    try:
        app_state["models"][model_id] = {"status": "loading"}
        
        if model_type == "reward_model":
            model = RewardModel.from_pretrained(model_path)
            tokenizer = None  # You'd load appropriate tokenizer
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.to(device)
        model.eval()
        
        app_state["models"][model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "type": model_type,
            "status": "ready",
            "device": device,
            "loaded_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Model {model_id} loaded successfully")
        
    except Exception as e:
        app_state["models"][model_id] = {"status": "error", "error": str(e)}
        logger.error(f"Failed to load model {model_id}: {str(e)}")

async def _start_training_background(experiment_id: str, request: TrainingRequest):
    """Start training in background."""
    try:
        experiment = app_state["experiments"][experiment_id]
        experiment["status"] = "running"
        tracker = experiment["tracker"]
        
        # Initialize dataset
        dataset = PreferenceDataset.from_file(request.dataset_path)
        
        # Initialize trainer based on type
        if request.training_type == "dpo":
            args = DPOTrainingArguments(
                output_dir=request.output_dir,
                num_train_epochs=request.num_epochs,
                per_device_train_batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                logging_steps=request.logging_steps,
                save_steps=request.save_steps,
                eval_steps=request.eval_steps,
                beta=request.config.get("beta", 0.1)
            )
            
            trainer = DirectPreferenceOptimizationTrainer(
                model=None,  # You'd initialize model here
                args=args,
                train_dataset=dataset,
                tokenizer=None  # You'd initialize tokenizer
            )
        else:
            trainer = RewardModelTrainer(
                model_name=request.model_name,
                dataset=dataset,
                output_dir=request.output_dir,
                num_epochs=request.num_epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate
            )
        
        # Start training with tracking
        trainer.train()
        
        experiment["status"] = "completed"
        logger.info(f"Training {experiment_id} completed successfully")
        
    except Exception as e:
        experiment["status"] = "failed"
        experiment["error"] = str(e)
        logger.error(f"Training {experiment_id} failed: {str(e)}")

async def _start_evaluation_background(eval_id: str, request: EvaluationRequest):
    """Start evaluation in background."""
    try:
        model_info = app_state["models"][request.model_id]
        model = model_info["model"]
        
        evaluator = RewardModelEvaluator(model)
        results = evaluator.evaluate(request.dataset_path, request.metrics)
        
        # Store results
        app_state["evaluations"] = app_state.get("evaluations", {})
        app_state["evaluations"][eval_id] = {
            "status": "completed",
            "results": results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Evaluation {eval_id} completed successfully")
        
    except Exception as e:
        app_state["evaluations"][eval_id] = {
            "status": "failed",
            "error": str(e)
        }
        logger.error(f"Evaluation {eval_id} failed: {str(e)}")

async def load_existing_models():
    """Load any existing models on startup."""
    # Implementation for loading persisted models
    pass

if __name__ == "__main__":
    uvicorn.run(
        "reward_modeling.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 