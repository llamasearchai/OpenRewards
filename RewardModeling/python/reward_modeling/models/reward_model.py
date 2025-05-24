"""
Advanced Reward Model implementation for LLM alignment.
Supports multiple architectures, multi-modal inputs, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedModel
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RewardModelConfig:
    """Configuration for reward models."""
    model_name_or_path: str
    dropout_rate: float = 0.1
    use_cache: bool = True
    gradient_checkpointing: bool = False
    freeze_backbone: bool = False
    value_head_hidden_dim: Optional[int] = None
    use_layer_norm: bool = True
    activation_function: str = "tanh"
    uncertainty_estimation: bool = False
    multi_objective: bool = False
    num_objectives: int = 1
    use_attention_pooling: bool = False
    
class BaseRewardModel(PreTrainedModel, ABC):
    """Abstract base class for reward models."""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__(config)
        self.config = config
        
    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get reward scores for input sequences."""
        pass

class TransformerRewardModel(BaseRewardModel):
    """
    Transformer-based reward model with advanced features.
    Supports uncertainty estimation, multi-objective rewards, and attention pooling.
    """
    
    def __init__(self, config: RewardModelConfig):
        super().__init__(config)
        
        # Load pre-trained transformer
        self.transformer_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            config=self.transformer_config,
        )
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if config.gradient_checkpointing and not config.freeze_backbone:
            self.backbone.gradient_checkpointing_enable()

        # Value head configuration
        hidden_size = self.transformer_config.hidden_size
        value_head_hidden_dim = config.value_head_hidden_dim or hidden_size
        
        # Attention pooling layer
        if config.use_attention_pooling:
            self.attention_pooling = AttentionPooling(hidden_size)
        
        # Value head layers
        value_head_layers = []
        
        # Input layer
        value_head_layers.append(nn.Linear(hidden_size, value_head_hidden_dim))
        
        if config.use_layer_norm:
            value_head_layers.append(nn.LayerNorm(value_head_hidden_dim))
        
        # Activation function
        if config.activation_function == "tanh":
            value_head_layers.append(nn.Tanh())
        elif config.activation_function == "gelu":
            value_head_layers.append(nn.GELU())
        elif config.activation_function == "relu":
            value_head_layers.append(nn.ReLU())
        
        value_head_layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        output_dim = config.num_objectives if config.multi_objective else 1
        
        if config.uncertainty_estimation:
            # For uncertainty estimation, output both mean and log variance
            value_head_layers.append(nn.Linear(value_head_hidden_dim, output_dim * 2))
        else:
            value_head_layers.append(nn.Linear(value_head_hidden_dim, output_dim))
        
        self.value_head = nn.Sequential(*value_head_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize value head weights."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the reward model.
        
        Args:
            input_ids: Token ids of input sequences
            attention_mask: Mask to avoid attention on padding tokens
            token_type_ids: Token type ids for sequence pairs
            position_ids: Position ids for position embeddings
            return_dict: Whether to return a dictionary or tuple
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dict containing rewards and optional model outputs
        """
        # Get transformer outputs
        transformer_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states or self.training,
            return_dict=True,
            use_cache=self.config.use_cache and not self.training
        )
        
        last_hidden_state = transformer_outputs.last_hidden_state
        
        # Pool sequence representation
        if hasattr(self, 'attention_pooling'):
            # Use attention pooling
            pooled_output = self.attention_pooling(last_hidden_state, attention_mask)
        else:
            # Use last non-padding token
            if attention_mask is not None:
                # Find position of last non-padding token
                last_non_pad_indices = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                device = last_hidden_state.device
                
                # Index into hidden state to get last non-padding token representation
                pooled_output = last_hidden_state[
                    torch.arange(batch_size, device=device),
                    last_non_pad_indices
                ]
            else:
                # Default to last token if no attention mask
                pooled_output = last_hidden_state[:, -1]
        
        # Compute value head output
        value_output = self.value_head(pooled_output)
        
        # Process output based on configuration
        if self.config.uncertainty_estimation:
            # Split into mean and log variance
            if self.config.multi_objective:
                means = value_output[:, :self.config.num_objectives]
                log_vars = value_output[:, self.config.num_objectives:]
                rewards = means
                uncertainties = torch.exp(log_vars)
            else:
                means = value_output[:, 0]
                log_vars = value_output[:, 1]
                rewards = means
                uncertainties = torch.exp(log_vars)
        else:
            rewards = value_output.squeeze(-1) if not self.config.multi_objective else value_output
            uncertainties = None
        
        # Prepare output
        outputs = {
            "rewards": rewards,
            "pooled_output": pooled_output,
        }
        
        if uncertainties is not None:
            outputs["uncertainties"] = uncertainties
            
        if output_hidden_states:
            outputs["hidden_states"] = transformer_outputs.hidden_states
            outputs["last_hidden_state"] = last_hidden_state
        
        return outputs if return_dict else (rewards,)
    
    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get reward scores for input sequences."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs["rewards"]
    
    def save_pretrained(self, save_dir: str):
        """Save model to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save backbone
        self.backbone.save_pretrained(save_path / "backbone")
        
        # Save value head
        torch.save(self.value_head.state_dict(), save_path / "value_head.pt")
        
        # Save config
        config_dict = {
            "model_name_or_path": str(save_path / "backbone"),
            "dropout_rate": self.config.dropout_rate,
            "use_cache": self.config.use_cache,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "freeze_backbone": self.config.freeze_backbone,
            "value_head_hidden_dim": self.config.value_head_hidden_dim,
            "use_layer_norm": self.config.use_layer_norm,
            "activation_function": self.config.activation_function,
            "uncertainty_estimation": self.config.uncertainty_estimation,
            "multi_objective": self.config.multi_objective,
            "num_objectives": self.config.num_objectives,
            "use_attention_pooling": self.config.use_attention_pooling,
        }
        
        with open(save_path / "reward_model_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    @classmethod
    def from_pretrained(cls, load_dir: str, **kwargs):
        """Load model from directory."""
        load_path = Path(load_dir)
        
        # Load config
        config_path = load_path / "reward_model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config_dict.update(kwargs)  # Allow override
            config = RewardModelConfig(**config_dict)
        else:
            # Fallback to default config
            config = RewardModelConfig(
                model_name_or_path=str(load_path / "backbone"),
                **kwargs
            )
        
        # Create model
        model = cls(config)
        
        # Load value head weights
        value_head_path = load_path / "value_head.pt"
        if value_head_path.exists():
            model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
        
        logger.info(f"Model loaded from {load_dir}")
        return model

class AttentionPooling(nn.Module):
    """Attention-based pooling layer for sequence representations."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to sequence.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Compute attention weights
        attention_weights = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Softmax
        attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, seq_len]
        
        # Weighted sum
        pooled_output = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, hidden_size]
        
        return pooled_output

class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for improved reliability."""
    
    def __init__(self, models: List[BaseRewardModel], aggregation: str = "mean"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        all_outputs = []
        
        for model in self.models:
            outputs = model(**kwargs)
            all_outputs.append(outputs)
        
        # Aggregate rewards
        all_rewards = torch.stack([outputs["rewards"] for outputs in all_outputs], dim=0)
        
        if self.aggregation == "mean":
            ensemble_rewards = torch.mean(all_rewards, dim=0)
        elif self.aggregation == "median":
            ensemble_rewards = torch.median(all_rewards, dim=0)[0]
        elif self.aggregation == "max":
            ensemble_rewards = torch.max(all_rewards, dim=0)[0]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        
        # Compute uncertainty as variance across models
        reward_variance = torch.var(all_rewards, dim=0)
        
        return {
            "rewards": ensemble_rewards,
            "uncertainties": reward_variance,
            "individual_rewards": all_rewards,
        }
    
    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get ensemble reward scores."""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            return outputs["rewards"]

class ConstitutionalRewardModel(TransformerRewardModel):
    """
    Reward model that incorporates constitutional AI principles.
    Evaluates responses based on adherence to specified principles.
    """
    
    def __init__(self, config: RewardModelConfig, constitution: List[str]):
        super().__init__(config)
        self.constitution = constitution
        
        # Add constitutional principle embeddings
        self.principle_embeddings = nn.Embedding(
            len(constitution), 
            self.transformer_config.hidden_size
        )
        
        # Constitutional attention layer
        self.constitutional_attention = nn.MultiheadAttention(
            self.transformer_config.hidden_size,
            num_heads=8,
            dropout=config.dropout_rate
        )
        
        # Modified value head for constitutional scoring
        self.constitutional_head = nn.Sequential(
            nn.Linear(self.transformer_config.hidden_size * 2, self.transformer_config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.transformer_config.hidden_size, len(constitution) + 1)  # +1 for general reward
        )
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with constitutional evaluation."""
        # Get base transformer outputs
        transformer_outputs = self.backbone(**kwargs)
        last_hidden_state = transformer_outputs.last_hidden_state
        
        # Pool sequence representation
        if kwargs.get("attention_mask") is not None:
            attention_mask = kwargs["attention_mask"]
            last_non_pad_indices = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            device = last_hidden_state.device
            
            pooled_output = last_hidden_state[
                torch.arange(batch_size, device=device),
                last_non_pad_indices
            ]
        else:
            pooled_output = last_hidden_state[:, -1]
        
        # Get principle embeddings
        principle_indices = torch.arange(len(self.constitution), device=pooled_output.device)
        principle_embeds = self.principle_embeddings(principle_indices)  # [num_principles, hidden_size]
        
        # Apply constitutional attention
        pooled_output_expanded = pooled_output.unsqueeze(1)  # [batch_size, 1, hidden_size]
        principle_embeds_expanded = principle_embeds.unsqueeze(0).expand(
            pooled_output.shape[0], -1, -1
        )  # [batch_size, num_principles, hidden_size]
        
        attended_output, attention_weights = self.constitutional_attention(
            pooled_output_expanded.transpose(0, 1),  # [1, batch_size, hidden_size]
            principle_embeds_expanded.transpose(0, 1),  # [num_principles, batch_size, hidden_size]
            principle_embeds_expanded.transpose(0, 1)
        )
        
        attended_output = attended_output.transpose(0, 1).squeeze(1)  # [batch_size, hidden_size]
        
        # Combine original and constitutional representations
        combined_representation = torch.cat([pooled_output, attended_output], dim=-1)
        
        # Get constitutional scores
        constitutional_scores = self.constitutional_head(combined_representation)
        
        # Split into general reward and principle scores
        general_reward = constitutional_scores[:, 0]
        principle_scores = constitutional_scores[:, 1:]
        
        return {
            "rewards": general_reward,
            "principle_scores": principle_scores,
            "attention_weights": attention_weights.squeeze(0),  # [batch_size, num_principles]
            "pooled_output": pooled_output,
            "constitutional_representation": attended_output,
        }

# Convenience class that maintains backward compatibility
RewardModel = TransformerRewardModel