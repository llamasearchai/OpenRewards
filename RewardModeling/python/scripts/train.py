#!/usr/bin/env python3
"""
Comprehensive training script for reward models.
Supports multiple training paradigms, evaluation, and deployment.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer
import wandb

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward_modeling.models.reward_model import RewardModel
from reward_modeling.data.dataset import (
    PreferenceDataset, 
    create_synthetic_dataset,
    load_huggingface_dataset
)
from reward_modeling.training.trainer import RewardModelTrainer, TrainingArguments
from reward_modeling.training.dpo_trainer import (
    DirectPreferenceOptimizationTrainer, 
    DPOTrainingArguments
)
from reward_modeling.evaluation.metrics import RewardModelEvaluator, ModelComparison
from reward_modeling.utils.logging import setup_logger
from reward_modeling.utils.config import Config

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate reward models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-name", 
        type=str, 
        default="distilbert-base-uncased",
        help="Pretrained model name or path"
    )
    model_group.add_argument(
        "--dropout-rate", 
        type=float, 
        default=0.1,
        help="Dropout rate for the value head"
    )
    model_group.add_argument(
        "--gradient-checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    model_group.add_argument(
        "--freeze-backbone", 
        action="store_true",
        help="Freeze the backbone model during training"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--train-data", 
        type=str,
        help="Path to training data file (JSON/JSONL/CSV)"
    )
    data_group.add_argument(
        "--eval-data", 
        type=str,
        help="Path to evaluation data file (JSON/JSONL/CSV)"
    )
    data_group.add_argument(
        "--test-data", 
        type=str,
        help="Path to test data file (JSON/JSONL/CSV)"
    )
    data_group.add_argument(
        "--dataset-name", 
        type=str,
        help="Hugging Face dataset name (alternative to file paths)"
    )
    data_group.add_argument(
        "--dataset-subset", 
        type=str,
        help="Subset of the Hugging Face dataset"
    )
    data_group.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum number of samples to use"
    )
    data_group.add_argument(
        "--create-synthetic", 
        action="store_true",
        help="Create synthetic dataset for testing"
    )
    data_group.add_argument(
        "--synthetic-samples", 
        type=int, 
        default=1000,
        help="Number of synthetic samples to create"
    )
    data_group.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    data_group.add_argument(
        "--train-split", 
        type=float, 
        default=0.8,
        help="Fraction of data to use for training"
    )
    data_group.add_argument(
        "--val-split", 
        type=float, 
        default=0.1,
        help="Fraction of data to use for validation"
    )
    
    # Training arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--training-type", 
        type=str, 
        choices=["reward_modeling", "dpo"], 
        default="reward_modeling",
        help="Type of training to perform"
    )
    training_group.add_argument(
        "--output-dir", 
        type=str, 
        default="./output",
        help="Output directory for model and logs"
    )
    training_group.add_argument(
        "--num-epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Training batch size"
    )
    training_group.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=32,
        help="Evaluation batch size"
    )
    training_group.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    training_group.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.01,
        help="Weight decay"
    )
    training_group.add_argument(
        "--warmup-steps", 
        type=int, 
        default=100,
        help="Number of warmup steps"
    )
    training_group.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=1,
        help="Number of gradient accumulation steps"
    )
    training_group.add_argument(
        "--max-grad-norm", 
        type=float, 
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    training_group.add_argument(
        "--fp16", 
        action="store_true",
        help="Use mixed precision training"
    )
    training_group.add_argument(
        "--dataloader-num-workers", 
        type=int, 
        default=4,
        help="Number of dataloader workers"
    )
    
    # DPO specific arguments
    dpo_group = parser.add_argument_group("DPO Configuration")
    dpo_group.add_argument(
        "--beta", 
        type=float, 
        default=0.1,
        help="DPO temperature parameter"
    )
    dpo_group.add_argument(
        "--reference-model-path", 
        type=str,
        help="Path to reference model for DPO"
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--eval-only", 
        action="store_true",
        help="Only run evaluation, skip training"
    )
    eval_group.add_argument(
        "--model-path", 
        type=str,
        help="Path to trained model for evaluation"
    )
    eval_group.add_argument(
        "--compute-uncertainty", 
        action="store_true",
        help="Compute uncertainty estimates during evaluation"
    )
    eval_group.add_argument(
        "--save-predictions", 
        action="store_true",
        help="Save individual predictions"
    )
    
    # Logging and monitoring
    logging_group = parser.add_argument_group("Logging and Monitoring")
    logging_group.add_argument(
        "--logging-steps", 
        type=int, 
        default=10,
        help="Log every N steps"
    )
    logging_group.add_argument(
        "--eval-steps", 
        type=int, 
        default=500,
        help="Evaluate every N steps"
    )
    logging_group.add_argument(
        "--save-steps", 
        type=int, 
        default=500,
        help="Save checkpoint every N steps"
    )
    logging_group.add_argument(
        "--use-wandb", 
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    logging_group.add_argument(
        "--wandb-project", 
        type=str, 
        default="reward_modeling",
        help="Weights & Biases project name"
    )
    logging_group.add_argument(
        "--wandb-run-name", 
        type=str,
        help="Weights & Biases run name"
    )
    logging_group.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    # System arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )
    system_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    system_group.add_argument(
        "--resume-from-checkpoint", 
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file (YAML/JSON)"
    )
    parser.add_argument(
        "--save-config", 
        type=str,
        help="Save current configuration to file"
    )
    
    return parser.parse_args()

def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    else:
        device = device_arg
        logger.info(f"Using specified device: {device}")
    
    return device

def load_datasets(args) -> tuple:
    """Load training, validation, and test datasets."""
    if args.create_synthetic:
        logger.info(f"Creating synthetic dataset with {args.synthetic_samples} samples")
        full_dataset = create_synthetic_dataset(
            num_samples=args.synthetic_samples,
            random_seed=args.seed
        )
        
        # Split into train/val/test
        train_dataset, val_dataset, test_dataset = full_dataset.split(
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=1 - args.train_split - args.val_split,
            random_seed=args.seed
        )
        
    elif args.dataset_name:
        logger.info(f"Loading dataset from Hugging Face: {args.dataset_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        full_dataset = load_huggingface_dataset(
            dataset_name=args.dataset_name,
            subset=args.dataset_subset,
            tokenizer=tokenizer,
            max_samples=args.max_samples
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = full_dataset.split(
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=1 - args.train_split - args.val_split,
            random_seed=args.seed
        )
        
    else:
        # Load from files
        train_dataset = None
        val_dataset = None
        test_dataset = None
        
        if args.train_data:
            logger.info(f"Loading training data from {args.train_data}")
            train_dataset = PreferenceDataset.from_file(
                args.train_data,
                max_length=args.max_length
            )
        
        if args.eval_data:
            logger.info(f"Loading evaluation data from {args.eval_data}")
            val_dataset = PreferenceDataset.from_file(
                args.eval_data,
                max_length=args.max_length
            )
        
        if args.test_data:
            logger.info(f"Loading test data from {args.test_data}")
            test_dataset = PreferenceDataset.from_file(
                args.test_data,
                max_length=args.max_length
            )
        
        # If only train data provided, split it
        if train_dataset and not val_dataset and not test_dataset:
            logger.info("Splitting training data into train/val/test")
            train_dataset, val_dataset, test_dataset = train_dataset.split(
                train_ratio=args.train_split,
                val_ratio=args.val_split,
                test_ratio=1 - args.train_split - args.val_split,
                random_seed=args.seed
            )
    
    # Log dataset sizes
    if train_dataset:
        logger.info(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def create_model(args) -> RewardModel:
    """Create and return the reward model."""
    logger.info(f"Creating reward model: {args.model_name}")
    
    model = RewardModel(
        model_name_or_path=args.model_name,
        dropout_rate=args.dropout_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_backbone=args.freeze_backbone
    )
    
    return model

def create_training_args(args) -> TrainingArguments:
    """Create training arguments."""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    return training_args

def train_model(model, train_dataset, val_dataset, args) -> Dict[str, Any]:
    """Train the reward model."""
    logger.info("Starting model training...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create training arguments
    training_args = create_training_args(args)
    
    if args.training_type == "dpo":
        # DPO Training
        dpo_args = DPOTrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            fp16=args.fp16,
            beta=args.beta,
            reference_model_path=args.reference_model_path,
            max_length=args.max_length
        )
        
        trainer = DirectPreferenceOptimizationTrainer(
            model=model,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )
    else:
        # Standard reward modeling
        trainer = RewardModelTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            tokenizer=tokenizer
        )
    
    # Train the model
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    return results

def evaluate_model(model_path: str, test_dataset, args) -> Dict[str, Any]:
    """Evaluate a trained model."""
    logger.info(f"Loading model for evaluation: {model_path}")
    
    # Load model and tokenizer
    model = RewardModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create evaluator
    evaluator = RewardModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=setup_device(args.device),
        batch_size=args.eval_batch_size,
        compute_uncertainty=args.compute_uncertainty
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(
        test_dataset,
        return_predictions=args.save_predictions
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    results.save(results_path)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Print key metrics
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {results.accuracy:.4f}")
    logger.info(f"  Reward Gap: {results.reward_gap:.4f}")
    logger.info(f"  Spearman Correlation: {results.spearman_rho:.4f}")
    logger.info(f"  Kendall Tau: {results.kendall_tau:.4f}")
    logger.info(f"  Calibration Error: {results.calibration_error:.4f}")
    
    return results.to_dict()

def save_configuration(args, save_path: str):
    """Save the current configuration to a file."""
    config_dict = vars(args)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {save_path}")

def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration from a file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config

def main():
    """Main training and evaluation pipeline."""
    args = parse_args()
    
    # Load configuration file if provided
    if args.config:
        config = load_configuration(args.config)
        # Update args with config values (command line takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Setup logging
    setup_logger(level=args.log_level)
    
    # Save configuration if requested
    if args.save_config:
        save_configuration(args, args.save_config)
        return
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run configuration
    config_path = os.path.join(args.output_dir, "config.json")
    save_configuration(args, config_path)
    
    try:
        # Load datasets
        train_dataset, val_dataset, test_dataset = load_datasets(args)
        
        if args.eval_only:
            # Evaluation only mode
            if not args.model_path:
                raise ValueError("--model-path required for evaluation-only mode")
            if not test_dataset:
                raise ValueError("Test dataset required for evaluation")
            
            evaluate_model(args.model_path, test_dataset, args)
            
        else:
            # Training mode
            if not train_dataset:
                raise ValueError("Training dataset required for training mode")
            
            # Create model
            model = create_model(args)
            
            # Train model
            training_results = train_model(model, train_dataset, val_dataset, args)
            
            # Evaluate on test set if available
            if test_dataset:
                logger.info("Running final evaluation on test set...")
                model_path = training_results["final_model_path"]
                eval_results = evaluate_model(model_path, test_dataset, args)
                
                # Save final results
                final_results = {
                    "training_results": training_results,
                    "evaluation_results": eval_results
                }
                
                results_path = os.path.join(args.output_dir, "final_results.json")
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2)
                
                logger.info(f"Final results saved to {results_path}")
            
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 