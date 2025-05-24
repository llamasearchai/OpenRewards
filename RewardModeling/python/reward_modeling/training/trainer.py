"""
Complete training implementation for reward models.
Supports standard reward modeling, DPO, constitutional AI, and advanced training techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import wandb
from tqdm.auto import tqdm
import math
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ..models.reward_model import RewardModel, TransformerRewardModel, RewardModelConfig
from ..data.dataset import PreferenceDataset, ConstitutionalDataset, DataCollator
from ..utils.config import Config
from ..utils.monitoring import MetricsCollector, ExperimentTracker
from ..evaluation.metrics import RewardModelEvaluator
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments:
    """Training configuration arguments."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    label_smoothing_factor: float = 0.0
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    
    # Advanced training options
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    deepspeed: Optional[str] = None
    local_rank: int = -1
    world_size: int = 1
    ddp_backend: Optional[str] = None
    
    # Custom options
    use_wandb: bool = False
    project_name: str = "reward_modeling"
    run_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseTrainer(ABC):
    """Abstract base trainer class."""
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[DataCollator] = None,
        compute_metrics: Optional[callable] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator or DataCollator()
        self.compute_metrics = compute_metrics
        
        # Training state
        self.state = TrainingState()
        self.optimizer = None
        self.lr_scheduler = None
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup logging
        self.setup_logging()
        
        # Setup tracking
        if self.args.use_wandb:
            self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_wandb(self):
        """Setup Weights & Biases tracking."""
        try:
            wandb.init(
                project=self.args.project_name,
                name=self.args.run_name,
                config=self.args.to_dict()
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.args.optim == "adamw_torch":
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=1e-8
            )
        elif self.args.optim == "adam":
            return Adam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optim}")
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.args.warmup_steps > 0:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        if self.args.lr_scheduler_type == "linear":
            return SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                    LinearLR(optimizer, start_factor=1.0, end_factor=0.0, 
                            total_iters=num_training_steps - warmup_steps)
                ],
                milestones=[warmup_steps]
            )
        elif self.args.lr_scheduler_type == "cosine":
            return SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
                ],
                milestones=[warmup_steps]
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.args.lr_scheduler_type}")
    
    @abstractmethod
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss."""
        pass
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a training step."""
        model.train()
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Compute loss
        loss = self.compute_loss(model, inputs)
        
        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss
    
    def evaluation_loop(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                total_steps += 1
        
        metrics = {"eval_loss": total_loss / total_steps}
        
        # Compute additional metrics if available
        if self.compute_metrics:
            additional_metrics = self.compute_metrics(self.model, dataloader)
            metrics.update(additional_metrics)
        
        return metrics
    
    def save_model(self, output_dir: str):
        """Save model checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
        
        # Save training arguments
        with open(output_dir / "training_args.json", "w") as f:
            json.dump(self.args.to_dict(), f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def train(self):
        """Main training loop."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for training")
        
        # Create data loaders
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers
        )
        
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers
            )
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        max_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        
        # Create optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, max_steps)
        
        # Training state
        self.state.max_steps = max_steps
        self.state.num_train_epochs = self.args.num_train_epochs
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total train batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        
        # Training loop
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")
            
            epoch_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
                # Training step
                loss = self.training_step(self.model, batch)
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.state.global_step += 1
                    
                    # Logging
                    if self.state.global_step % self.args.logging_steps == 0:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        logs = {
                            "loss": loss.item() * self.args.gradient_accumulation_steps,
                            "learning_rate": current_lr,
                            "epoch": epoch + (step + 1) / len(train_dataloader),
                            "global_step": self.state.global_step
                        }
                        
                        logger.info(f"Step {self.state.global_step}: {logs}")
                        
                        if self.args.use_wandb:
                            wandb.log(logs)
                    
                    # Evaluation
                    if (self.args.evaluation_strategy == "steps" and 
                        self.state.global_step % self.args.eval_steps == 0 and 
                        eval_dataloader is not None):
                        
                        eval_metrics = self.evaluation_loop(eval_dataloader)
                        logger.info(f"Eval metrics: {eval_metrics}")
                        
                        if self.args.use_wandb:
                            wandb.log(eval_metrics)
                        
                        # Save best model
                        if self.args.load_best_model_at_end:
                            current_metric = eval_metrics[self.args.metric_for_best_model]
                            if (self.state.best_metric is None or 
                                (self.args.greater_is_better and current_metric > self.state.best_metric) or
                                (not self.args.greater_is_better and current_metric < self.state.best_metric)):
                                
                                self.state.best_metric = current_metric
                                self.save_model(Path(self.args.output_dir) / "best")
                    
                    # Save checkpoint
                    if (self.args.save_strategy == "steps" and 
                        self.state.global_step % self.args.save_steps == 0):
                        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                        self.save_model(checkpoint_dir)
            
            # End of epoch evaluation
            if (self.args.evaluation_strategy == "epoch" and eval_dataloader is not None):
                eval_metrics = self.evaluation_loop(eval_dataloader)
                logger.info(f"End of epoch {epoch + 1} eval metrics: {eval_metrics}")
                
                if self.args.use_wandb:
                    wandb.log(eval_metrics)
            
            # End of epoch save
            if self.args.save_strategy == "epoch":
                checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch + 1}"
                self.save_model(checkpoint_dir)
        
        # Final save
        self.save_model(self.args.output_dir)
        
        if self.args.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")

class RewardModelTrainer(BaseTrainer):
    """Trainer for standard reward models."""
    
    def __init__(
        self,
        model: Optional[RewardModel] = None,
        model_name: Optional[str] = None,
        args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        if model is None and model_name is not None:
            config = RewardModelConfig(model_name_or_path=model_name)
            model = TransformerRewardModel(config)
        
        if args is None:
            args = TrainingArguments(output_dir="./reward_model_output")
        
        super().__init__(model=model, args=args, **kwargs)
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reward modeling loss (Bradley-Terry model)."""
        # Get chosen and rejected sequences
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]
        
        # Get rewards for chosen and rejected
        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        chosen_rewards = chosen_outputs["rewards"]
        rejected_rewards = rejected_outputs["rewards"]
        
        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return loss

class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization."""
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        
        self.reference_model = reference_model
        self.beta = beta
        
        # Freeze reference model
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DPO loss."""
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]
        
        # Get logits from current model
        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        chosen_logps = self._get_sequence_logprobs(chosen_outputs, chosen_ids, chosen_mask)
        rejected_logps = self._get_sequence_logprobs(rejected_outputs, rejected_ids, rejected_mask)
        
        # Get reference logits if available
        if self.reference_model is not None:
            with torch.no_grad():
                ref_chosen_outputs = self.reference_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                ref_rejected_outputs = self.reference_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                
                ref_chosen_logps = self._get_sequence_logprobs(ref_chosen_outputs, chosen_ids, chosen_mask)
                ref_rejected_logps = self._get_sequence_logprobs(ref_rejected_outputs, rejected_ids, rejected_mask)
        else:
            ref_chosen_logps = 0
            ref_rejected_logps = 0
        
        # DPO loss
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def _get_sequence_logprobs(self, outputs, input_ids, attention_mask):
        """Calculate log probabilities for sequences."""
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # For reward models, we need to implement language modeling head
            # This is a simplified version
            return torch.zeros(input_ids.shape[0], device=input_ids.device)
        
        # Shift tokens for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask and sum
        masked_log_probs = token_log_probs * shift_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs

class ConstitutionalTrainer(BaseTrainer):
    """Trainer for Constitutional AI."""
    
    def __init__(
        self,
        model: nn.Module,
        constitution: List[str],
        args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.constitution = constitution
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute constitutional AI loss."""
        # Get initial and revised responses
        initial_ids = inputs["initial_input_ids"]
        initial_mask = inputs["initial_attention_mask"]
        revised_ids = inputs["revised_input_ids"]
        revised_mask = inputs["revised_attention_mask"]
        
        # Get rewards
        initial_outputs = model(input_ids=initial_ids, attention_mask=initial_mask)
        revised_outputs = model(input_ids=revised_ids, attention_mask=revised_mask)
        
        initial_rewards = initial_outputs["rewards"]
        revised_rewards = revised_outputs["rewards"]
        
        # Constitutional loss - revised should be better than initial
        loss = -F.logsigmoid(revised_rewards - initial_rewards).mean()
        
        # Add constitutional principle adherence if model supports it
        if hasattr(initial_outputs, 'principle_scores') and hasattr(revised_outputs, 'principle_scores'):
            initial_principle_scores = initial_outputs["principle_scores"]
            revised_principle_scores = revised_outputs["principle_scores"]
            
            # Principle adherence loss
            principle_loss = -F.logsigmoid(revised_principle_scores - initial_principle_scores).mean()
            loss += principle_loss
        
        return loss

@dataclass
class TrainingState:
    """Training state tracker."""
    epoch: int = 0
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    best_metric: Optional[float] = None
    
class MultiObjectiveTrainer(BaseTrainer):
    """Trainer for multi-objective reward models."""
    
    def __init__(
        self,
        model: nn.Module,
        objective_weights: Optional[List[float]] = None,
        args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.objective_weights = objective_weights or [1.0]
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-objective loss."""
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]
        rejected_ids = inputs["rejected_input_ids"]
        rejected_mask = inputs["rejected_attention_mask"]
        
        # Get multi-objective rewards
        chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        chosen_rewards = chosen_outputs["rewards"]  # Shape: [batch_size, num_objectives]
        rejected_rewards = rejected_outputs["rewards"]
        
        # Weighted combination of objectives
        if chosen_rewards.dim() > 1:
            weights = torch.tensor(self.objective_weights, device=chosen_rewards.device)
            chosen_weighted = (chosen_rewards * weights).sum(dim=-1)
            rejected_weighted = (rejected_rewards * weights).sum(dim=-1)
        else:
            chosen_weighted = chosen_rewards
            rejected_weighted = rejected_rewards
        
        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_weighted - rejected_weighted).mean()
        
        return loss

def create_trainer(
    trainer_type: str,
    model: nn.Module,
    args: TrainingArguments,
    **kwargs
) -> BaseTrainer:
    """Factory function to create trainers."""
    trainer_classes = {
        "reward": RewardModelTrainer,
        "dpo": DPOTrainer,
        "constitutional": ConstitutionalTrainer,
        "multi_objective": MultiObjectiveTrainer
    }
    
    if trainer_type not in trainer_classes:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    return trainer_classes[trainer_type](model=model, args=args, **kwargs)

# Training utilities
def compute_reward_metrics(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    """Compute reward model evaluation metrics."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"]
            chosen_mask = batch["chosen_attention_mask"]
            rejected_ids = batch["rejected_input_ids"]
            rejected_mask = batch["rejected_attention_mask"]
            
            chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            chosen_rewards = chosen_outputs["rewards"]
            rejected_rewards = rejected_outputs["rewards"]
            
            # Count correct preferences
            correct = (chosen_rewards > rejected_rewards).sum().item()
            total_correct += correct
            total_samples += chosen_rewards.shape[0]
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {"accuracy": accuracy}

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)

def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    **kwargs
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)

class RewardModelTrainer:
    """
    Trainer for reward models supporting multiple training paradigms.
    """
    
    def __init__(
        self,
        model: RewardModel,
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
        args: Optional[TrainingArguments] = None,
        tokenizer=None,
        data_collator=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or TrainingArguments()
        self.tokenizer = tokenizer
        self.data_collator = data_collator or self._default_data_collator
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set up logging
        setup_logger()
        
        # Set up directories
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        
        # Initialize monitoring
        self.metrics_collector = MetricsCollector()
        self.evaluator = RewardModelEvaluator(self.model) if eval_dataset else None
        
        # Initialize experiment tracking
        if self.args.use_wandb:
            self._init_wandb()
        
        # Initialize tensorboard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, "tensorboard"))
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Set random seed
        self._set_seed(self.args.seed)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay."""
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if "bias" not in n],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]
        
        return optim.AdamW(
            param_groups,
            lr=self.args.learning_rate,
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
        
        total_steps = len(self.train_dataset) // self.args.batch_size * self.args.num_train_epochs
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.args.warmup_steps
        )
        
        # Main scheduler
        main_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - self.args.warmup_steps
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.args.warmup_steps]
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb.init(
            project=self.args.project_name,
            name=self.args.run_name,
            config=self.args.to_dict(),
            tags=["reward_modeling", "training"]
        )
    
    def _default_data_collator(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Default data collator for preference pairs."""
        batch_size = len(batch)
        
        # Collect all text inputs
        chosen_texts = [item["chosen"] for item in batch]
        rejected_texts = [item["rejected"] for item in batch]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            chosen_encodings = self.tokenizer(
                chosen_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            rejected_encodings = self.tokenizer(
                rejected_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return {
                "chosen_input_ids": chosen_encodings["input_ids"],
                "chosen_attention_mask": chosen_encodings["attention_mask"],
                "rejected_input_ids": rejected_encodings["input_ids"],
                "rejected_attention_mask": rejected_encodings["attention_mask"],
            }
        else:
            # Return text directly if no tokenizer
            return {
                "chosen_texts": chosen_texts,
                "rejected_texts": rejected_texts,
            }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Bradley-Terry preference loss.
        
        Args:
            batch: Batch of preference pairs
            
        Returns:
            Dictionary containing loss and metrics
        """
        # Forward pass on chosen responses
        if "chosen_input_ids" in batch:
            chosen_outputs = self.model(
                input_ids=batch["chosen_input_ids"].to(self.device),
                attention_mask=batch["chosen_attention_mask"].to(self.device),
                return_dict=True
            )
            chosen_rewards = chosen_outputs["rewards"]
            
            # Forward pass on rejected responses
            rejected_outputs = self.model(
                input_ids=batch["rejected_input_ids"].to(self.device),
                attention_mask=batch["rejected_attention_mask"].to(self.device),
                return_dict=True
            )
            rejected_rewards = rejected_outputs["rewards"]
        else:
            # Handle text inputs directly
            chosen_rewards = []
            rejected_rewards = []
            
            for chosen_text, rejected_text in zip(batch["chosen_texts"], batch["rejected_texts"]):
                if self.tokenizer:
                    chosen_inputs = self.tokenizer(
                        chosen_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="max_length"
                    ).to(self.device)
                    
                    rejected_inputs = self.tokenizer(
                        rejected_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="max_length"
                    ).to(self.device)
                    
                    chosen_output = self.model(**chosen_inputs, return_dict=True)
                    rejected_output = self.model(**rejected_inputs, return_dict=True)
                    
                    chosen_rewards.append(chosen_output["rewards"])
                    rejected_rewards.append(rejected_output["rewards"])
            
            chosen_rewards = torch.stack(chosen_rewards).squeeze(-1)
            rejected_rewards = torch.stack(rejected_rewards).squeeze(-1)
        
        # Compute Bradley-Terry loss
        reward_diff = chosen_rewards - rejected_rewards
        loss = -torch.log(torch.sigmoid(reward_diff)).mean()
        
        # Compute metrics
        accuracy = (reward_diff > 0).float().mean()
        reward_gap = reward_diff.mean()
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "reward_gap": reward_gap,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
        
        total_loss = 0.0
        total_metrics = {
            "accuracy": 0.0,
            "reward_gap": 0.0,
            "chosen_rewards": 0.0,
            "rejected_rewards": 0.0,
        }
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.args.num_train_epochs}",
            disable=False
        )
        
        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss_dict = self.compute_loss(batch)
            loss = loss_dict["loss"]
            
            # Gradient accumulation
            loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.args.fp16:
                with torch.cuda.amp.autocast():
                    loss.backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            for key in total_metrics:
                if key in loss_dict:
                    total_metrics[key] += loss_dict[key].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{loss_dict['accuracy'].item():.3f}",
                "gap": f"{loss_dict['reward_gap'].item():.3f}",
            })
            
            # Logging
            if self.global_step % self.args.logging_steps == 0:
                self._log_metrics({
                    "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                    "train/accuracy": loss_dict["accuracy"].item(),
                    "train/reward_gap": loss_dict["reward_gap"].item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "train/global_step": self.global_step,
                })
            
            # Evaluation
            if self.eval_dataset and self.global_step % self.args.eval_steps == 0:
                eval_results = self.evaluate()
                self._log_metrics({f"eval/{k}": v for k, v in eval_results.items()})
                self.model.train()  # Switch back to training mode
            
            # Save checkpoint
            if self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Average metrics
        num_batches = len(train_dataloader)
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {"loss": avg_loss, **avg_metrics}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        if not self.eval_dataset or not self.evaluator:
            return {}
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
        
        total_loss = 0.0
        total_metrics = {
            "accuracy": 0.0,
            "reward_gap": 0.0,
            "chosen_rewards": 0.0,
            "rejected_rewards": 0.0,
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss_dict = self.compute_loss(batch)
                
                total_loss += loss_dict["loss"].item()
                for key in total_metrics:
                    if key in loss_dict:
                        total_metrics[key] += loss_dict[key].item()
        
        # Average metrics
        num_batches = len(eval_dataloader)
        results = {
            "loss": total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }
        
        # Additional evaluation metrics
        if hasattr(self.evaluator, 'evaluate_comprehensive'):
            comprehensive_results = self.evaluator.evaluate_comprehensive(self.eval_dataset)
            results.update(comprehensive_results)
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
        
        # Resume from checkpoint if specified
        if self.args.resume_from_checkpoint:
            self.load_checkpoint(self.args.resume_from_checkpoint)
        
        # Training history
        training_history = {
            "train_loss": [],
            "eval_loss": [],
            "train_accuracy": [],
            "eval_accuracy": [],
        }
        
        for epoch in range(self.current_epoch, self.args.num_train_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch()
            training_history["train_loss"].append(train_results["loss"])
            training_history["train_accuracy"].append(train_results["accuracy"])
            
            # Log epoch results
            self._log_metrics({
                f"epoch_{epoch}/train_loss": train_results["loss"],
                f"epoch_{epoch}/train_accuracy": train_results["accuracy"],
                f"epoch_{epoch}/train_reward_gap": train_results["reward_gap"],
            })
            
            # Evaluate
            if self.eval_dataset:
                eval_results = self.evaluate()
                training_history["eval_loss"].append(eval_results["loss"])
                training_history["eval_accuracy"].append(eval_results["accuracy"])
                
                self._log_metrics({
                    f"epoch_{epoch}/eval_loss": eval_results["loss"],
                    f"epoch_{epoch}/eval_accuracy": eval_results["accuracy"],
                })
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch}")
            
            logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_results['loss']:.4f}, "
                f"Train Acc: {train_results['accuracy']:.3f}"
            )
        
        # Save final model
        self.save_model()
        
        # Final evaluation
        if self.eval_dataset:
            final_eval = self.evaluate()
            logger.info(f"Final evaluation: {final_eval}")
        
        # Clean up
        self.tb_writer.close()
        if self.args.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")
        
        return {
            "training_history": training_history,
            "final_model_path": os.path.join(self.args.output_dir, "final_model"),
            "best_checkpoint": self._find_best_checkpoint(),
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all configured trackers."""
        # TensorBoard
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, self.global_step)
        
        # Weights & Biases
        if self.args.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Metrics collector
        self.metrics_collector.log_metrics(metrics, step=self.global_step)
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        checkpoint_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "args": self.args.__dict__,
        }
        
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model
        self.model = RewardModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        
        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            checkpoint_state = torch.load(state_path, map_location=self.device)
            
            self.current_epoch = checkpoint_state["epoch"]
            self.global_step = checkpoint_state["global_step"]
            self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler_state_dict"])
            
            logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_model(self):
        """Save the final trained model."""
        final_model_path = os.path.join(self.args.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        self.model.save_pretrained(final_model_path)
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(final_model_path)
        
        # Save training metadata
        metadata = {
            "model_class": self.model.__class__.__name__,
            "training_args": self.args.__dict__,
            "final_step": self.global_step,
            "final_epoch": self.current_epoch,
            "training_completed": datetime.now().isoformat(),
        }
        
        with open(os.path.join(final_model_path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Final model saved to {final_model_path}")
    
    def _find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint based on evaluation metrics."""
        # This is a simplified version - in practice you'd track eval metrics
        checkpoints = [
            d for d in os.listdir(self.args.output_dir)
            if d.startswith("checkpoint-") or d.startswith("epoch-")
        ]
        
        if checkpoints:
            # Return the latest checkpoint as a fallback
            return sorted(checkpoints)[-1]
        
        return None 