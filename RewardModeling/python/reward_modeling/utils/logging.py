"""
Enhanced Logging Utilities for Reward Modeling Platform

This module provides comprehensive logging functionality including structured logging,
performance tracking, log filtering, and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
import os
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import threading
from contextlib import contextmanager
from functools import wraps

# Configure logger levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add thread information
        log_data['thread_id'] = record.thread
        log_data['thread_name'] = record.threadName
        
        # Add process information
        log_data['process_id'] = record.process
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger_name: str = "reward_modeling.performance"):
        self.logger = logging.getLogger(logger_name)
        self._timers = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        with self._lock:
            self._timers[operation] = time.time()
    
    def end_timer(self, operation: str, extra_data: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and log the duration."""
        with self._lock:
            if operation not in self._timers:
                self.logger.warning(f"Timer for operation '{operation}' was not started")
                return 0.0
            
            duration = time.time() - self._timers.pop(operation)
            
            log_data = {
                'operation': operation,
                'duration': duration,
                'duration_ms': duration * 1000
            }
            
            if extra_data:
                log_data.update(extra_data)
            
            self.logger.info(f"Operation completed: {operation}", extra={'extra_fields': log_data})
            return duration
    
    @contextmanager
    def timer(self, operation: str, extra_data: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations."""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, extra_data)


class LoggingConfig:
    """Configuration class for logging setup."""
    
    def __init__(
        self,
        level: Union[str, int] = logging.INFO,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        structured_logging: bool = False,
        console_logging: bool = True,
        log_to_file: bool = True,
        enable_performance_logging: bool = True,
        filter_modules: Optional[List[str]] = None
    ):
        self.level = level if isinstance(level, int) else LOG_LEVELS.get(level.upper(), logging.INFO)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_file = log_file or "reward_modeling.log"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.structured_logging = structured_logging
        self.console_logging = console_logging
        self.log_to_file = log_to_file
        self.enable_performance_logging = enable_performance_logging
        self.filter_modules = filter_modules or []


def setup_logger(
    name: str = "reward_modeling",
    config: Optional[LoggingConfig] = None,
    **kwargs
) -> logging.Logger:
    """
    Set up comprehensive logging for the reward modeling platform.
    
    Args:
        name: Logger name
        config: LoggingConfig instance or None to use default
        **kwargs: Additional configuration options
    
    Returns:
        Configured logger instance
    """
    if config is None:
        config = LoggingConfig(**kwargs)
    
    logger = logging.getLogger(name)
    logger.setLevel(config.level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    if config.structured_logging:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
    
    # Console handler
    if config.console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(config.level)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.log_to_file:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = config.log_dir / config.log_file
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(config.level)
        logger.addHandler(file_handler)
    
    # Add module filters if specified
    if config.filter_modules:
        for module_name in config.filter_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.WARNING)  # Filter noisy modules
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "reward_modeling") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured data to logs."""
    
    def __init__(self, logger: logging.Logger, **context_data):
        self.logger = logger
        self.context_data = context_data
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            if not hasattr(record, 'extra_fields'):
                record.extra_fields = {}
            record.extra_fields.update(self.context_data)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """Decorator to log function calls with parameters and execution time."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log function entry
            logger.log(
                level,
                f"Entering function: {func.__name__}",
                extra={
                    'extra_fields': {
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful exit
                logger.log(
                    level,
                    f"Exiting function: {func.__name__}",
                    extra={
                        'extra_fields': {
                            'function': func.__name__,
                            'module': func.__module__,
                            'duration': duration,
                            'status': 'success'
                        }
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log exception
                logger.error(
                    f"Function {func.__name__} raised exception: {str(e)}",
                    extra={
                        'extra_fields': {
                            'function': func.__name__,
                            'module': func.__module__,
                            'duration': duration,
                            'status': 'error',
                            'exception_type': type(e).__name__
                        }
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def setup_training_logger(
    output_dir: str,
    experiment_name: str = "experiment",
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """Set up specialized logger for training experiments."""
    log_dir = Path(output_dir) / "logs"
    log_file = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    config = LoggingConfig(
        level=level,
        log_dir=str(log_dir),
        log_file=log_file,
        structured_logging=True,
        enable_performance_logging=True
    )
    
    return setup_logger(f"reward_modeling.training.{experiment_name}", config)


def log_model_info(
    logger: logging.Logger,
    model,
    dataset_size: Optional[int] = None,
    training_args: Optional[Dict[str, Any]] = None
):
    """Log model and training information."""
    model_info = {
        'model_class': model.__class__.__name__,
        'model_name': getattr(model, 'name_or_path', 'unknown'),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    if dataset_size is not None:
        model_info['dataset_size'] = dataset_size
    
    if training_args is not None:
        model_info['training_args'] = training_args
    
    logger.info(
        "Model information logged",
        extra={'extra_fields': model_info}
    )


# Global performance logger instance
performance_logger = PerformanceLogger()


# Example usage and main function
if __name__ == '__main__':
    # Example of setting up comprehensive logging
    config = LoggingConfig(
        level=logging.DEBUG,
        log_dir="./logs",
        structured_logging=True,
        enable_performance_logging=True
    )
    
    logger = setup_logger(config=config)
    
    # Example usage
    logger.info("Logging system initialized")
    
    # Example with context
    with LogContext(logger, experiment_id="exp_001", model_type="reward_model"):
        logger.info("Training started")
        
        # Example with performance logging
        with performance_logger.timer("model_training", {"batch_size": 32}):
            time.sleep(0.1)  # Simulate work
            logger.info("Model training step completed")
    
    # Example with function decorator
    @log_function_call(logger)
    def example_function(x, y=10):
        return x + y
    
    result = example_function(5, y=15)
    logger.info(f"Function result: {result}")
    
    logger.info("Example completed") 