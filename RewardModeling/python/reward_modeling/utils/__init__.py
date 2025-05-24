"""
Utility modules for the Reward Modeling Platform.

This package provides comprehensive utilities for configuration management,
logging, monitoring, data processing, and performance optimization.
"""

from .config import (
    RewardModelingConfig,
    TrainingConfig,
    EvaluationConfig,
    DataConfig,
    ModelConfig,
    load_config,
    save_config,
    get_default_config,
    validate_config,
    merge_configs
)

from .logging import (
    setup_logger,
    get_logger,
    LoggingConfig,
    StructuredFormatter,
    PerformanceLogger,
    LogContext,
    log_function_call,
    setup_training_logger,
    log_model_info,
    performance_logger
)

from .monitoring import (
    MetricsCollector,
    SystemMonitor,
    GPUMonitor,
    TrainingMonitor,
    HealthCheck,
    AlertManager,
    DashboardServer,
    setup_monitoring,
    log_system_info,
    monitor_training
)

# Utility functions for common operations
from .data_utils import (
    load_dataset,
    save_dataset,
    validate_dataset,
    compute_dataset_statistics,
    split_dataset,
    augment_dataset,
    clean_text,
    tokenize_text,
    pad_sequences
)

from .model_utils import (
    count_parameters,
    freeze_parameters,
    unfreeze_parameters,
    get_model_size,
    load_model_checkpoint,
    save_model_checkpoint,
    verify_model_integrity,
    optimize_model_for_inference
)

from .training_utils import (
    set_seed,
    get_device,
    move_to_device,
    gradient_clipping,
    learning_rate_schedule,
    early_stopping,
    checkpoint_manager,
    training_progress
)

from .evaluation_utils import (
    compute_metrics,
    statistical_significance_test,
    bootstrap_confidence_intervals,
    cross_validation_scores,
    model_comparison,
    generate_evaluation_report
)

__all__ = [
    # Config utilities
    "RewardModelingConfig",
    "TrainingConfig", 
    "EvaluationConfig",
    "DataConfig",
    "ModelConfig",
    "load_config",
    "save_config",
    "get_default_config",
    "validate_config",
    "merge_configs",
    
    # Logging utilities
    "setup_logger",
    "get_logger",
    "LoggingConfig",
    "StructuredFormatter",
    "PerformanceLogger",
    "LogContext",
    "log_function_call",
    "setup_training_logger",
    "log_model_info",
    "performance_logger",
    
    # Monitoring utilities
    "MetricsCollector",
    "SystemMonitor",
    "GPUMonitor",
    "TrainingMonitor",
    "HealthCheck",
    "AlertManager",
    "DashboardServer",
    "setup_monitoring",
    "log_system_info",
    "monitor_training",
    
    # Data utilities
    "load_dataset",
    "save_dataset",
    "validate_dataset",
    "compute_dataset_statistics",
    "split_dataset",
    "augment_dataset",
    "clean_text",
    "tokenize_text",
    "pad_sequences",
    
    # Model utilities
    "count_parameters",
    "freeze_parameters",
    "unfreeze_parameters",
    "get_model_size",
    "load_model_checkpoint",
    "save_model_checkpoint",
    "verify_model_integrity",
    "optimize_model_for_inference",
    
    # Training utilities
    "set_seed",
    "get_device",
    "move_to_device",
    "gradient_clipping",
    "learning_rate_schedule",
    "early_stopping",
    "checkpoint_manager",
    "training_progress",
    
    # Evaluation utilities
    "compute_metrics",
    "statistical_significance_test",
    "bootstrap_confidence_intervals",
    "cross_validation_scores",
    "model_comparison",
    "generate_evaluation_report"
] 