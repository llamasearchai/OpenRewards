import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import wandb
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import PreTrainedModel, Trainer, TrainingArguments

from ..models.reward_model import RewardModel
from ..data.dataset import PreferenceDataset
from ..utils.logging import setup_logger
from ..utils.monitoring import log_metrics

logger = logging.getLogger(__name__)

@dataclass
class DPOTrainingArguments(TrainingArguments):
    """
    Arguments for Direct Preference Optimization training.
    """
    beta: float = field(default=0.1, metadata={"help": "Temperature parameter for DPO loss"})
    reference_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to reference model for KL penalty"}
    )
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    max_prompt_length: int = field(default=128, metadata={"help": "Maximum prompt length"})
    regularization_strength: float = field(default=0.001, metadata={"help": "KL regularization strength"})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "WandB project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "WandB run name"})

class DirectPreferenceOptimizationTrainer:
    """
    Trainer for Direct Preference Optimization (DPO) for LLM alignment.
    Implements the training loop for preference learning using DPO.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        reference_model: Optional[PreTrainedModel] = None,
        args: DPOTrainingArguments = None,
        train_dataset: Optional[PreferenceDataset] = None,
        eval_dataset: Optional[PreferenceDataset] = None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
    ):
        self.model = model
        self.args = args or DPOTrainingArguments("./output")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        
        # Initialize reference model if provided
        if reference_model is not None:
            self.reference_model = reference_model
        elif self.args.reference_model_path is not None:
            self.reference_model = model.__class__.from_pretrained(
                self.args.reference_model_path
            )
        else:
            # Clone the model as reference if not provided
            self.reference_model = model.__class__.from_pretrained(
                model.config._name_or_path
            )
        
        # Set reference model to evaluation mode
        self.reference_model.eval()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # Setup logger
        setup_logger()
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the DPO loss based on log probabilities from policy and reference models.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses from policy model
            policy_rejected_logps: Log probs of rejected responses from policy model
            reference_chosen_logps: Log probs of chosen responses from reference model
            reference_rejected_logps: Log probs of rejected responses from reference model
            
        Returns:
            DPO loss value
        """
        # Compute log ratios between policy and reference models
        chosen_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # Compute DPO loss
        logits = self.args.beta * (chosen_ratio - rejected_ratio)
        losses = -F.logsigmoid(logits)
        
        # Compute implicit rewards
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        # Add optional KL penalty
        if self.args.regularization_strength > 0:
            kl_penalty = self.args.regularization_strength * (
                torch.mean(torch.abs(chosen_ratio)) + 
                torch.mean(torch.abs(rejected_ratio))
            )
            losses = losses + kl_penalty
        
        return {
            "loss": losses.mean(),
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_gap": (chosen_rewards - rejected_rewards).mean(),
            "logits": logits.mean(),
        }

    def _get_batch_logps(
        self,
        model: PreTrainedModel,
        batch: Dict[str, torch.Tensor],
        inference: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get log probabilities for chosen and rejected responses.
        
        Args:
            model: Model to use for prediction
            batch: Batch of data containing input tensors
            inference: Whether to run in inference mode
            
        Returns:
            Tuple of log probabilities for chosen and rejected responses
        """
        with torch.inference_mode(inference):
            # Forward pass for chosen responses
            chosen_forward_outputs = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                return_dict=True
            )
            
            # Forward pass for rejected responses
            rejected_forward_outputs = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                return_dict=True
            )
            
            # Extract logits
            chosen_logits = chosen_forward_outputs.logits
            rejected_logits = rejected_forward_outputs.logits
            
            # Calculate log probabilities for token prediction
            chosen_logps = self._get_token_logps(
                chosen_logits, 
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"]
            )
            
            rejected_logps = self._get_token_logps(
                rejected_logits,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"]
            )
            
            return chosen_logps, rejected_logps
    
    def _get_token_logps(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute log probabilities for token predictions.
        
        Args:
            logits: Prediction logits from model
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Target labels
            
        Returns:
            Log probabilities for tokens
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probs at labels positions
        token_logps = torch.gather(log_probs[:, :-1], dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Create mask excluding padding and prompt tokens
        seq_lengths = attention_mask.sum(dim=1)
        response_mask = (attention_mask[:, 1:] == 1) & (labels[:, 1:] != -100)
        
        # Sum log probs for response tokens
        per_response_logps = (token_logps * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-9)
        
        return per_response_logps
    
    def train(self):
        """
        Train the model using DPO.
        """
        logger.info("Starting DPO training")
        
        # Initialize training
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        # Initialize progress tracking
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name)
        
        self.model.train()
        total_steps = self.args.num_train_epochs * len(train_dataloader)
        progress_bar = tqdm(range(total_steps), desc="Training")
        
        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0.0
            epoch_metrics = {
                "chosen_rewards": 0.0,
                "rejected_rewards": 0.0,
                "reward_gap": 0.0,
                "logits": 0.0,
            }
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                
                # Get log probabilities from policy model
                policy_chosen_logps, policy_rejected_logps = self._get_batch_logps(
                    self.model, batch
                )
                
                # Get log probabilities from reference model (no gradient tracking)
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps = self._get_batch_logps(
                        self.reference_model, batch, inference=True
                    )
                
                # Compute DPO loss
                loss_dict = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
                
                # Update model
                self.optimizer.zero_grad()
                loss_dict["loss"].backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                # Update progress
                progress_bar.update(1)
                epoch_loss += loss_dict["loss"].item()
                
                # Update metrics
                for k, v in loss_dict.items():
                    if k != "loss":
                        epoch_metrics[k] += v.item()
                
                # Log step metrics
                if self.args.use_wandb and step % self.args.logging_steps == 0:
                    log_metrics(
                        {"loss": loss_dict["loss"].item(), **{k: v.item() for k, v in loss_dict.items() if k != "loss"}},
                        step + epoch * len(train_dataloader)
                    )
            
            # Log epoch metrics
            avg_loss = epoch_loss / len(train_dataloader)
            avg_metrics = {k: v / len(train_dataloader) for k, v in epoch_metrics.items()}
            
            logger.info(f"Epoch {epoch+1}/{self.args.num_train_epochs}, Loss: {avg_loss:.4f}")
            
            if self.args.use_wandb:
                log_metrics({"epoch": epoch, "loss/epoch": avg_loss, **{f"metrics/{k}": v for k, v in avg_metrics.items()}})
            
            # Evaluate if needed
            if self.eval_dataset is not None and (epoch + 1) % self.args.eval_steps == 0:
                eval_results = self.evaluate()
                logger.info(f"Evaluation results: {eval_results}")
                
                if self.args.use_wandb:
                    log_metrics({f"eval/{k}": v for k, v in eval_results.items()})
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_steps == 0:
                self.save_model(f"{self.args.output_dir}/checkpoint-{epoch+1}")
        
        # Save final model
        self.save_model(f"{self.args.output_dir}/final-model")
        logger.info("Training completed")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation")
        
        # Initialize evaluation
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
        
        self.model.eval()
        all_metrics = {
            "eval_loss": 0.0,
            "eval_chosen_rewards": 0.0,
            "eval_rejected_rewards": 0.0,
            "eval_reward_gap": 0.0,
            "eval_accuracy": 0.0,
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                
                # Get log probabilities
                policy_chosen_logps, policy_rejected_logps = self._get_batch_logps(
                    self.model, batch, inference=True
                )
                reference_chosen_logps, reference_rejected_logps = self._get_batch_logps(
                    self.reference_model, batch, inference=True
                )
                
                # Compute DPO loss
                loss_dict = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
                
                # Calculate preference accuracy (how often model prefers chosen over rejected)
                batch_accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean().item()
                
                # Update metrics
                all_metrics["eval_loss"] += loss_dict["loss"].item()
                all_metrics["eval_chosen_rewards"] += loss_dict["chosen_rewards"].item()
                all_metrics["eval_rejected_rewards"] += loss_dict["rejected_rewards"].item()
                all_metrics["eval_reward_gap"] += loss_dict["reward_gap"].item()
                all_metrics["eval_accuracy"] += batch_accuracy
        
        # Average metrics
        all_metrics = {k: v / len(eval_dataloader) for k, v in all_metrics.items()}
        
        self.model.train()
        return all_metrics
    
    def save_model(self, output_dir: str):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Directory to save model to
        """
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)