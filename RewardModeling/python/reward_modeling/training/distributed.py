"""
Distributed Training Utilities for Reward Modeling

This module provides utilities for distributed training of reward models across multiple GPUs
and nodes using PyTorch's native DistributedDataParallel (DDP) and DeepSpeed integration.
"""

import os
import json
import logging
import socket
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout_minutes: int = 30
    
    # DeepSpeed configuration
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = None
    zero_stage: int = 2
    gradient_accumulation_steps: int = 1
    
    # Data parallel configuration
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DistributedConfig":
        """Create from dictionary."""
        return cls(**config_dict)


class DistributedManager:
    """Manager for distributed training setup and coordination."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.process_group = None
        
    def initialize(self) -> bool:
        """
        Initialize distributed training environment.
        
        Returns:
            bool: True if successfully initialized, False otherwise
        """
        try:
            if self.is_initialized:
                logger.warning("Distributed training already initialized")
                return True
                
            # Set environment variables if not already set
            if "RANK" not in os.environ:
                os.environ["RANK"] = str(self.config.rank)
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = str(self.config.world_size)
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = str(self.config.local_rank)
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = self.config.master_addr
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = self.config.master_port
                
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.distributed.utils.timedelta(minutes=self.config.timeout_minutes)
            )
            
            # Set device for current process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                
            self.is_initialized = True
            logger.info(f"Initialized distributed training: rank={self.config.rank}, "
                       f"world_size={self.config.world_size}, backend={self.config.backend}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Cleaned up distributed training")
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process (rank 0)."""
        return self.config.rank == 0
    
    def get_rank(self) -> int:
        """Get current process rank."""
        return self.config.rank
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across all processes."""
        if self.is_initialized and self.config.world_size > 1:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather operation across all processes."""
        if not self.is_initialized or self.config.world_size == 1:
            return [tensor]
            
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all processes."""
        if self.is_initialized and self.config.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor


class DistributedDataParallelWrapper:
    """Wrapper for creating distributed data parallel models."""
    
    def __init__(self, manager: DistributedManager):
        self.manager = manager
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: PyTorch model to wrap
            
        Returns:
            Wrapped model for distributed training
        """
        if not self.manager.is_initialized or self.manager.config.world_size == 1:
            return model
            
        # Move model to appropriate device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.manager.config.local_rank}")
            model = model.to(device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.manager.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.manager.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.manager.config.find_unused_parameters,
            broadcast_buffers=self.manager.config.broadcast_buffers,
            bucket_cap_mb=self.manager.config.bucket_cap_mb
        )
        
        logger.info(f"Wrapped model with DDP on rank {self.manager.config.rank}")
        return ddp_model
    
    def create_dataloader(
        self, 
        dataset, 
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create distributed dataloader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size per process
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader with distributed sampler
        """
        if self.manager.is_initialized and self.manager.config.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.manager.config.world_size,
                rank=self.manager.config.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs
        )


class DeepSpeedManager:
    """Manager for DeepSpeed integration."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.engine = None
        
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not installed. Install with: pip install deepspeed")
    
    def create_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration."""
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto"
                }
            },
            "zero_optimization": {
                "stage": self.config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.config.zero_stage >= 2 else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.config.zero_stage >= 3 else "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            "fp16": {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "gradient_clipping": "auto",
            "wall_clock_breakdown": False
        }
        
        return ds_config
    
    def initialize_engine(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler = None,
        config_params: Dict[str, Any] = None
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Initialize DeepSpeed engine.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            lr_scheduler: Learning rate scheduler (optional)
            config_params: Additional configuration parameters
            
        Returns:
            Tuple of (engine, optimizer, dataloader, lr_scheduler)
        """
        # Load or create configuration
        if self.config.deepspeed_config_path and Path(self.config.deepspeed_config_path).exists():
            with open(self.config.deepspeed_config_path, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = self.create_config()
        
        # Update with provided parameters
        if config_params:
            ds_config.update(config_params)
        
        # Initialize DeepSpeed engine
        self.engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config
        )
        
        logger.info(f"Initialized DeepSpeed engine with ZeRO stage {self.config.zero_stage}")
        return self.engine, optimizer, None, lr_scheduler


def setup_distributed_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
    use_deepspeed: bool = False,
    **kwargs
) -> DistributedManager:
    """
    Convenience function to setup distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend
        use_deepspeed: Whether to use DeepSpeed
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized DistributedManager
    """
    config = DistributedConfig(
        rank=rank,
        world_size=world_size,
        local_rank=rank % torch.cuda.device_count() if torch.cuda.is_available() else 0,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        use_deepspeed=use_deepspeed,
        **kwargs
    )
    
    manager = DistributedManager(config)
    if manager.initialize():
        return manager
    else:
        raise RuntimeError("Failed to initialize distributed training")


def find_free_port() -> str:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)


def save_distributed_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    manager: DistributedManager,
    filename: Optional[str] = None
):
    """
    Save checkpoint in distributed training setting.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        manager: Distributed manager
        filename: Optional filename override
    """
    if not manager.is_main_process():
        return
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    # Extract model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'distributed_config': manager.config.to_dict()
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_distributed_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = "cpu"
) -> Dict[str, Any]:
    """
    Load checkpoint in distributed training setting.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        map_location: Device to map tensors to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'distributed_config': checkpoint.get('distributed_config', {})
    }


def get_device_memory_info() -> Dict[str, Any]:
    """Get memory information for current device."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "memory_info": "N/A"}
    
    device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    
    return {
        "device": f"cuda:{device}",
        "memory_allocated_mb": memory_allocated / 1024**2,
        "memory_reserved_mb": memory_reserved / 1024**2,
        "max_memory_allocated_mb": max_memory_allocated / 1024**2,
        "memory_allocated_gb": memory_allocated / 1024**3,
        "memory_reserved_gb": memory_reserved / 1024**3,
        "max_memory_allocated_gb": max_memory_allocated / 1024**3
    }


class DistributedMetrics:
    """Helper class for aggregating metrics across distributed processes."""
    
    def __init__(self, manager: DistributedManager):
        self.manager = manager
    
    def aggregate_scalar(self, value: float, operation: str = "mean") -> float:
        """
        Aggregate scalar value across all processes.
        
        Args:
            value: Scalar value to aggregate
            operation: Aggregation operation ("mean", "sum", "max", "min")
            
        Returns:
            Aggregated value
        """
        if not self.manager.is_initialized or self.manager.config.world_size == 1:
            return value
        
        tensor = torch.tensor(value, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        if operation == "sum":
            self.manager.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif operation == "mean":
            self.manager.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.manager.config.world_size
        elif operation == "max":
            self.manager.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif operation == "min":
            self.manager.all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return tensor.item()
    
    def gather_scalars(self, value: float) -> List[float]:
        """Gather scalar values from all processes."""
        if not self.manager.is_initialized or self.manager.config.world_size == 1:
            return [value]
        
        tensor = torch.tensor(value, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        gathered_tensors = self.manager.all_gather(tensor)
        return [t.item() for t in gathered_tensors] 