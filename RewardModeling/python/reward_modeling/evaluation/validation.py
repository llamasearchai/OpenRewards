"""
Validation Utilities for Reward Modeling

This module provides comprehensive validation functionality for reward models,
including cross-validation, hold-out validation, temporal validation, and
advanced validation techniques for preference learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    train_test_split, GroupKFold
)
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from ..data.dataset import PreferenceDataset, PreferencePair
from ..models.reward_model import RewardModel
from ..evaluation.metrics import RewardModelEvaluator, EvaluationResults
from ..training.trainer import RewardModelTrainer, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation procedures."""
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "kfold"  # kfold, stratified, group, timeseries
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Hold-out validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    stratify: bool = False
    
    # Temporal validation settings
    temporal_split_date: Optional[str] = None
    temporal_window_days: int = 30
    temporal_step_days: int = 7
    
    # Bootstrap validation settings
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    
    # Evaluation settings
    metrics: List[str] = None
    save_predictions: bool = True
    save_models: bool = False
    
    # Early stopping for validation
    early_stopping_patience: int = 5
    early_stopping_metric: str = "accuracy"
    early_stopping_mode: str = "max"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", "auc", 
                "ranking_loss", "kendall_tau", "spearman_rho"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ValidationConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ValidationResults:
    """Results from validation procedures."""
    
    validation_type: str
    fold_results: List[EvaluationResults]
    aggregated_metrics: Dict[str, float]
    aggregated_std: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Additional metadata
    config: ValidationConfig
    model_info: Dict[str, Any]
    validation_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validation_type": self.validation_type,
            "fold_results": [result.to_dict() for result in self.fold_results],
            "aggregated_metrics": self.aggregated_metrics,
            "aggregated_std": self.aggregated_std,
            "confidence_intervals": self.confidence_intervals,
            "config": self.config.to_dict(),
            "model_info": self.model_info,
            "validation_time": self.validation_time,
            "timestamp": self.timestamp.isoformat()
        }
    
    def save(self, filepath: str):
        """Save results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved validation results to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ValidationResults":
        """Load results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        config = ValidationConfig.from_dict(data["config"])
        fold_results = [EvaluationResults.from_dict(result) for result in data["fold_results"]]
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            validation_type=data["validation_type"],
            fold_results=fold_results,
            aggregated_metrics=data["aggregated_metrics"],
            aggregated_std=data["aggregated_std"],
            confidence_intervals=data["confidence_intervals"],
            config=config,
            model_info=data["model_info"],
            validation_time=data["validation_time"],
            timestamp=timestamp
        )


class BaseValidator:
    """Base class for validation strategies."""
    
    def __init__(
        self,
        config: ValidationConfig,
        evaluator: RewardModelEvaluator,
        device: str = "auto"
    ):
        self.config = config
        self.evaluator = evaluator
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initialized validator with device: {self.device}")
    
    def validate(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> ValidationResults:
        """
        Perform validation. To be implemented by subclasses.
        
        Args:
            model_class: Class of model to validate
            dataset: Dataset to validate on
            training_args: Training arguments
            **kwargs: Additional arguments
            
        Returns:
            ValidationResults object
        """
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _train_model(
        self,
        model_class: type,
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> RewardModel:
        """Train a model on given datasets."""
        # Initialize model
        model = model_class(**kwargs.get("model_kwargs", {}))
        model = model.to(self.device)
        
        # Initialize trainer
        trainer = RewardModelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=None  # Use our evaluator instead
        )
        
        # Train model
        trainer.train()
        
        return model
    
    def _aggregate_results(
        self,
        fold_results: List[EvaluationResults],
        validation_type: str,
        model_info: Dict[str, Any],
        validation_time: float
    ) -> ValidationResults:
        """Aggregate results from multiple folds."""
        # Extract metrics from all folds
        metrics_by_fold = defaultdict(list)
        for result in fold_results:
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics_by_fold[metric].append(value)
        
        # Compute aggregated statistics
        aggregated_metrics = {}
        aggregated_std = {}
        confidence_intervals = {}
        
        for metric, values in metrics_by_fold.items():
            if len(values) > 0:
                aggregated_metrics[metric] = np.mean(values)
                aggregated_std[metric] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                # Compute confidence intervals
                if len(values) > 1:
                    confidence_level = self.config.bootstrap_confidence
                    alpha = 1 - confidence_level
                    ci = stats.t.interval(
                        confidence_level,
                        len(values) - 1,
                        loc=np.mean(values),
                        scale=stats.sem(values)
                    )
                    confidence_intervals[metric] = ci
                else:
                    confidence_intervals[metric] = (values[0], values[0])
        
        return ValidationResults(
            validation_type=validation_type,
            fold_results=fold_results,
            aggregated_metrics=aggregated_metrics,
            aggregated_std=aggregated_std,
            confidence_intervals=confidence_intervals,
            config=self.config,
            model_info=model_info,
            validation_time=validation_time,
            timestamp=datetime.now()
        )


class CrossValidator(BaseValidator):
    """Cross-validation implementation."""
    
    def validate(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> ValidationResults:
        """Perform cross-validation."""
        start_time = datetime.now()
        
        # Prepare cross-validation splitter
        cv_splitter = self._get_cv_splitter(dataset)
        
        fold_results = []
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter):
            logger.info(f"Training fold {fold_idx + 1}/{self.config.cv_folds}")
            
            # Create train/validation datasets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            # Train model
            model = self._train_model(
                model_class, train_dataset, val_dataset, training_args, **kwargs
            )
            
            # Evaluate model
            eval_results = self.evaluator.evaluate(model, val_dataset)
            eval_results.fold_id = fold_idx
            fold_results.append(eval_results)
            
            # Clean up model if not saving
            if not self.config.save_models:
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        model_info = {
            "model_class": model_class.__name__,
            "total_samples": len(dataset),
            "cv_folds": self.config.cv_folds
        }
        
        return self._aggregate_results(
            fold_results, "cross_validation", model_info, validation_time
        )
    
    def _get_cv_splitter(self, dataset: Dataset):
        """Get cross-validation splitter based on configuration."""
        n_samples = len(dataset)
        
        if self.config.cv_strategy == "kfold":
            return KFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            ).split(range(n_samples))
        
        elif self.config.cv_strategy == "stratified":
            # Extract labels for stratification
            labels = self._extract_labels(dataset)
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            ).split(range(n_samples), labels)
        
        elif self.config.cv_strategy == "group":
            # Extract groups for group-based splitting
            groups = self._extract_groups(dataset)
            return GroupKFold(n_splits=self.config.cv_folds).split(range(n_samples), groups=groups)
        
        elif self.config.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.config.cv_folds).split(range(n_samples))
        
        else:
            raise ValueError(f"Unknown CV strategy: {self.config.cv_strategy}")
    
    def _extract_labels(self, dataset: Dataset) -> List[int]:
        """Extract labels for stratified cross-validation."""
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, PreferencePair):
                # For preference pairs, use preference as label
                labels.append(1 if item.preference == "chosen" else 0)
            elif isinstance(item, dict) and "label" in item:
                labels.append(item["label"])
            else:
                # Default: binary random labels
                labels.append(i % 2)
        return labels
    
    def _extract_groups(self, dataset: Dataset) -> List[str]:
        """Extract groups for group-based cross-validation."""
        groups = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, PreferencePair) and hasattr(item, "group"):
                groups.append(item.group)
            elif isinstance(item, dict) and "group" in item:
                groups.append(item["group"])
            else:
                # Default: assign groups based on index
                groups.append(str(i // (len(dataset) // self.config.cv_folds)))
        return groups


class HoldOutValidator(BaseValidator):
    """Hold-out validation implementation."""
    
    def validate(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> ValidationResults:
        """Perform hold-out validation."""
        start_time = datetime.now()
        
        # Split dataset
        train_dataset, val_test_dataset = self._split_dataset(
            dataset, 1 - self.config.validation_split - self.config.test_split
        )
        
        val_dataset, test_dataset = self._split_dataset(
            val_test_dataset, 
            self.config.validation_split / (self.config.validation_split + self.config.test_split)
        )
        
        logger.info(f"Dataset splits - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Train model
        model = self._train_model(
            model_class, train_dataset, val_dataset, training_args, **kwargs
        )
        
        # Evaluate on validation set
        val_results = self.evaluator.evaluate(model, val_dataset)
        val_results.split_type = "validation"
        
        # Evaluate on test set
        test_results = self.evaluator.evaluate(model, test_dataset)
        test_results.split_type = "test"
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        fold_results = [val_results, test_results]
        model_info = {
            "model_class": model_class.__name__,
            "total_samples": len(dataset),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
        }
        
        return self._aggregate_results(
            fold_results, "holdout_validation", model_info, validation_time
        )
    
    def _split_dataset(self, dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and test portions."""
        indices = list(range(len(dataset)))
        
        if self.config.stratify:
            labels = self._extract_labels_for_stratification(dataset)
            train_indices, test_indices = train_test_split(
                indices,
                train_size=train_ratio,
                stratify=labels,
                random_state=self.config.cv_random_state
            )
        else:
            train_indices, test_indices = train_test_split(
                indices,
                train_size=train_ratio,
                random_state=self.config.cv_random_state
            )
        
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        return train_dataset, test_dataset
    
    def _extract_labels_for_stratification(self, dataset: Dataset) -> List[int]:
        """Extract labels for stratified splitting."""
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, PreferencePair):
                labels.append(1 if item.preference == "chosen" else 0)
            elif isinstance(item, dict) and "label" in item:
                labels.append(item["label"])
            else:
                labels.append(i % 2)
        return labels


class TemporalValidator(BaseValidator):
    """Temporal validation for time-series data."""
    
    def validate(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> ValidationResults:
        """Perform temporal validation."""
        start_time = datetime.now()
        
        # Extract timestamps and sort dataset
        timestamps = self._extract_timestamps(dataset)
        sorted_indices = np.argsort(timestamps)
        sorted_dataset = Subset(dataset, sorted_indices)
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        
        # Create temporal splits
        splits = self._create_temporal_splits(sorted_timestamps)
        
        fold_results = []
        for fold_idx, (train_end, val_start, val_end) in enumerate(splits):
            logger.info(f"Training temporal fold {fold_idx + 1}/{len(splits)}")
            
            # Create train/validation datasets
            train_indices = [i for i, ts in enumerate(sorted_timestamps) if ts <= train_end]
            val_indices = [i for i, ts in enumerate(sorted_timestamps) 
                          if val_start <= ts <= val_end]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                logger.warning(f"Skipping fold {fold_idx} due to empty splits")
                continue
            
            train_dataset = Subset(sorted_dataset, train_indices)
            val_dataset = Subset(sorted_dataset, val_indices)
            
            # Train model
            model = self._train_model(
                model_class, train_dataset, val_dataset, training_args, **kwargs
            )
            
            # Evaluate model
            eval_results = self.evaluator.evaluate(model, val_dataset)
            eval_results.fold_id = fold_idx
            eval_results.temporal_info = {
                "train_end": train_end.isoformat(),
                "val_start": val_start.isoformat(),
                "val_end": val_end.isoformat()
            }
            fold_results.append(eval_results)
            
            # Clean up model if not saving
            if not self.config.save_models:
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        model_info = {
            "model_class": model_class.__name__,
            "total_samples": len(dataset),
            "temporal_folds": len(splits),
            "temporal_window_days": self.config.temporal_window_days,
            "temporal_step_days": self.config.temporal_step_days
        }
        
        return self._aggregate_results(
            fold_results, "temporal_validation", model_info, validation_time
        )
    
    def _extract_timestamps(self, dataset: Dataset) -> List[datetime]:
        """Extract timestamps from dataset."""
        timestamps = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, PreferencePair) and hasattr(item, "timestamp"):
                timestamps.append(item.timestamp)
            elif isinstance(item, dict) and "timestamp" in item:
                timestamps.append(datetime.fromisoformat(item["timestamp"]))
            else:
                # Default: assign timestamps based on index
                base_time = datetime.now() - timedelta(days=len(dataset))
                timestamps.append(base_time + timedelta(days=i))
        return timestamps
    
    def _create_temporal_splits(self, timestamps: List[datetime]) -> List[Tuple[datetime, datetime, datetime]]:
        """Create temporal train/validation splits."""
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        splits = []
        current_time = min_time + timedelta(days=self.config.temporal_window_days)
        
        while current_time + timedelta(days=self.config.temporal_window_days) <= max_time:
            train_end = current_time
            val_start = current_time
            val_end = current_time + timedelta(days=self.config.temporal_window_days)
            
            splits.append((train_end, val_start, val_end))
            current_time += timedelta(days=self.config.temporal_step_days)
        
        return splits


class BootstrapValidator(BaseValidator):
    """Bootstrap validation implementation."""
    
    def validate(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        **kwargs
    ) -> ValidationResults:
        """Perform bootstrap validation."""
        start_time = datetime.now()
        
        fold_results = []
        n_samples = len(dataset)
        
        for bootstrap_idx in range(self.config.bootstrap_samples):
            if bootstrap_idx % 100 == 0:
                logger.info(f"Bootstrap sample {bootstrap_idx + 1}/{self.config.bootstrap_samples}")
            
            # Create bootstrap sample
            train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            val_indices = list(set(range(n_samples)) - set(train_indices))
            
            if len(val_indices) == 0:
                continue
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            # Train model with reduced training to speed up bootstrap
            bootstrap_training_args = TrainingArguments(
                num_train_epochs=max(1, training_args.num_train_epochs // 2),
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                learning_rate=training_args.learning_rate,
                output_dir=training_args.output_dir,
                logging_steps=training_args.logging_steps * 10,  # Reduce logging
                save_steps=training_args.save_steps * 10,  # Reduce saving
                eval_steps=training_args.eval_steps * 10 if training_args.eval_steps else None
            )
            
            # Train model
            model = self._train_model(
                model_class, train_dataset, val_dataset, bootstrap_training_args, **kwargs
            )
            
            # Evaluate model
            eval_results = self.evaluator.evaluate(model, val_dataset)
            eval_results.fold_id = bootstrap_idx
            fold_results.append(eval_results)
            
            # Clean up model
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Aggregate results
        model_info = {
            "model_class": model_class.__name__,
            "total_samples": len(dataset),
            "bootstrap_samples": self.config.bootstrap_samples
        }
        
        return self._aggregate_results(
            fold_results, "bootstrap_validation", model_info, validation_time
        )


class ValidationSuite:
    """Comprehensive validation suite combining multiple validation strategies."""
    
    def __init__(
        self,
        config: ValidationConfig,
        evaluator: RewardModelEvaluator,
        device: str = "auto"
    ):
        self.config = config
        self.evaluator = evaluator
        self.device = device
        
        # Initialize validators
        self.validators = {
            "cross_validation": CrossValidator(config, evaluator, device),
            "holdout_validation": HoldOutValidator(config, evaluator, device),
            "temporal_validation": TemporalValidator(config, evaluator, device),
            "bootstrap_validation": BootstrapValidator(config, evaluator, device)
        }
    
    def run_validation_suite(
        self,
        model_class: type,
        dataset: Dataset,
        training_args: TrainingArguments,
        validation_types: List[str] = None,
        **kwargs
    ) -> Dict[str, ValidationResults]:
        """
        Run comprehensive validation suite.
        
        Args:
            model_class: Class of model to validate
            dataset: Dataset to validate on
            training_args: Training arguments
            validation_types: List of validation types to run
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping validation type to results
        """
        if validation_types is None:
            validation_types = ["cross_validation", "holdout_validation"]
        
        results = {}
        
        for val_type in validation_types:
            if val_type not in self.validators:
                logger.warning(f"Unknown validation type: {val_type}")
                continue
            
            logger.info(f"Running {val_type}")
            try:
                results[val_type] = self.validators[val_type].validate(
                    model_class, dataset, training_args, **kwargs
                )
                logger.info(f"Completed {val_type}")
            except Exception as e:
                logger.error(f"Failed to run {val_type}: {e}")
                continue
        
        return results
    
    def save_suite_results(self, results: Dict[str, ValidationResults], output_dir: str):
        """Save all validation results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for val_type, result in results.items():
            filepath = output_dir / f"{val_type}_results.json"
            result.save(filepath)
        
        # Create summary report
        self._create_summary_report(results, output_dir / "validation_summary.json")
        
        logger.info(f"Saved validation suite results to {output_dir}")
    
    def _create_summary_report(self, results: Dict[str, ValidationResults], filepath: str):
        """Create a summary report of all validation results."""
        summary = {
            "validation_suite_summary": {
                "timestamp": datetime.now().isoformat(),
                "validation_types": list(results.keys()),
                "config": self.config.to_dict()
            },
            "results_summary": {}
        }
        
        for val_type, result in results.items():
            summary["results_summary"][val_type] = {
                "aggregated_metrics": result.aggregated_metrics,
                "confidence_intervals": result.confidence_intervals,
                "num_folds": len(result.fold_results),
                "validation_time": result.validation_time
            }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def create_validator(
    validation_type: str,
    config: ValidationConfig,
    evaluator: RewardModelEvaluator,
    device: str = "auto"
) -> BaseValidator:
    """
    Factory function to create validators.
    
    Args:
        validation_type: Type of validator to create
        config: Validation configuration
        evaluator: Model evaluator
        device: Device to use for training/evaluation
        
    Returns:
        Validator instance
    """
    validators = {
        "cross_validation": CrossValidator,
        "holdout_validation": HoldOutValidator,
        "temporal_validation": TemporalValidator,
        "bootstrap_validation": BootstrapValidator
    }
    
    if validation_type not in validators:
        raise ValueError(f"Unknown validation type: {validation_type}")
    
    return validators[validation_type](config, evaluator, device)


def run_quick_validation(
    model_class: type,
    dataset: Dataset,
    training_args: TrainingArguments,
    validation_split: float = 0.2,
    device: str = "auto",
    **kwargs
) -> ValidationResults:
    """
    Run a quick hold-out validation for rapid prototyping.
    
    Args:
        model_class: Class of model to validate
        dataset: Dataset to validate on
        training_args: Training arguments
        validation_split: Validation split ratio
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Validation results
    """
    config = ValidationConfig(validation_split=validation_split, cv_folds=1)
    evaluator = RewardModelEvaluator(device=device)
    validator = HoldOutValidator(config, evaluator, device)
    
    return validator.validate(model_class, dataset, training_args, **kwargs) 