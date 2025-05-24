"""
Model Utilities for Reward Modeling Platform

This module provides comprehensive utilities for model management, optimization,
and manipulation including parameter counting, checkpointing, and inference optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import logging
import json
import hashlib
import time
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_parameters(model: nn.Module, module_names: Optional[List[str]] = None) -> None:
    """
    Freeze model parameters to prevent updates during training.
    
    Args:
        model: PyTorch model
        module_names: List of module names to freeze. If None, freeze all parameters
    """
    if module_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")
    else:
        # Freeze specific modules
        frozen_count = 0
        for name, module in model.named_modules():
            if any(target_name in name for target_name in module_names):
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += param.numel()
        logger.info(f"Froze {frozen_count} parameters in modules: {module_names}")


def unfreeze_parameters(model: nn.Module, module_names: Optional[List[str]] = None) -> None:
    """
    Unfreeze model parameters to allow updates during training.
    
    Args:
        model: PyTorch model
        module_names: List of module names to unfreeze. If None, unfreeze all parameters
    """
    if module_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters")
    else:
        # Unfreeze specific modules
        unfrozen_count = 0
        for name, module in model.named_modules():
            if any(target_name in name for target_name in module_names):
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += param.numel()
        logger.info(f"Unfroze {unfrozen_count} parameters in modules: {module_names}")


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive model size information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary containing model size information
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Calculate memory usage
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_memory = param_size + buffer_size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'param_memory_mb': param_size / (1024 * 1024),
        'buffer_memory_mb': buffer_size / (1024 * 1024),
        'total_memory_mb': total_memory / (1024 * 1024),
        'model_architecture': str(model.__class__.__name__)
    }


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: Optional[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with error handling and validation.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        map_location: Device to map loaded tensors to
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        else:
            state_dict = checkpoint
            metadata = {}
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        logger.info("Model checkpoint loaded successfully")
        
        return {
            'metadata': metadata,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'checkpoint_path': str(checkpoint_path)
        }
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with comprehensive information.
    
    Args:
        model: PyTorch model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch number
        global_step: Global training step
        loss: Current loss value
        metrics: Training/validation metrics
        metadata: Additional metadata to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': time.time(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_class'] = optimizer.__class__.__name__
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint['scheduler_class'] = scheduler.__class__.__name__
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if global_step is not None:
        checkpoint['global_step'] = global_step
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Add model size information
    checkpoint['model_size'] = get_model_size(model)
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise


def verify_model_integrity(model: nn.Module) -> Dict[str, Any]:
    """
    Verify model integrity and detect potential issues.
    
    Args:
        model: PyTorch model to verify
    
    Returns:
        Dictionary containing verification results
    """
    verification_report = {
        'model_valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check for NaN or Inf parameters
    nan_params = []
    inf_params = []
    zero_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
            verification_report['model_valid'] = False
        
        if torch.isinf(param).any():
            inf_params.append(name)
            verification_report['model_valid'] = False
        
        if torch.all(param == 0):
            zero_params.append(name)
    
    if nan_params:
        verification_report['issues'].append(f"NaN parameters found: {nan_params}")
    
    if inf_params:
        verification_report['issues'].append(f"Inf parameters found: {inf_params}")
    
    if zero_params:
        verification_report['warnings'].append(f"Zero parameters found: {zero_params}")
    
    # Check gradient flow
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad_params.append(name)
    
    if no_grad_params and len(no_grad_params) < len(list(model.parameters())):
        verification_report['warnings'].append(f"Parameters without gradients: {no_grad_params}")
    
    # Compute parameter statistics
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats[name] = {
            'shape': list(param.shape),
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item(),
            'requires_grad': param.requires_grad
        }
    
    verification_report['statistics'] = param_stats
    
    logger.info(f"Model verification completed. Valid: {verification_report['model_valid']}")
    
    return verification_report


def optimize_model_for_inference(
    model: nn.Module,
    optimization_level: str = "moderate",
    quantization: bool = False,
    jit_compile: bool = False
) -> nn.Module:
    """
    Optimize model for inference by applying various optimizations.
    
    Args:
        model: PyTorch model to optimize
        optimization_level: Level of optimization ("light", "moderate", "aggressive")
        quantization: Whether to apply quantization
        jit_compile: Whether to compile with TorchScript
    
    Returns:
        Optimized model
    """
    optimized_model = model
    
    # Set to evaluation mode
    optimized_model.eval()
    
    # Disable gradient computation
    for param in optimized_model.parameters():
        param.requires_grad = False
    
    if optimization_level in ["moderate", "aggressive"]:
        # Fuse certain operations for better performance
        try:
            # This is a placeholder - actual fusion depends on model architecture
            if hasattr(torch.nn.utils, 'fuse_conv_bn_eval'):
                optimized_model = torch.nn.utils.fuse_conv_bn_eval(optimized_model)
            logger.info("Applied operator fusion optimizations")
        except Exception as e:
            logger.warning(f"Could not apply fusion optimizations: {str(e)}")
    
    if quantization and optimization_level == "aggressive":
        try:
            # Dynamic quantization
            optimized_model = torch.quantization.quantize_dynamic(
                optimized_model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
        except Exception as e:
            logger.warning(f"Could not apply quantization: {str(e)}")
    
    if jit_compile:
        try:
            # Create example input for tracing (this would need to be model-specific)
            # This is a placeholder - actual input shape depends on model
            example_input = torch.randn(1, 512)  # Adjust based on model requirements
            optimized_model = torch.jit.trace(optimized_model, example_input)
            logger.info("Applied TorchScript compilation")
        except Exception as e:
            logger.warning(f"Could not apply JIT compilation: {str(e)}")
    
    # Calculate memory savings
    original_size = get_model_size(model)
    optimized_size = get_model_size(optimized_model)
    
    logger.info(f"Model optimization completed. "
               f"Memory reduction: {original_size['total_memory_mb']:.2f}MB -> "
               f"{optimized_size['total_memory_mb']:.2f}MB")
    
    return optimized_model


def compute_model_hash(model: nn.Module) -> str:
    """
    Compute hash of model parameters for integrity checking.
    
    Args:
        model: PyTorch model
    
    Returns:
        SHA256 hash of model parameters
    """
    hash_object = hashlib.sha256()
    
    for param in model.parameters():
        hash_object.update(param.data.cpu().numpy().tobytes())
    
    return hash_object.hexdigest()


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    Compare two models for differences in architecture and parameters.
    
    Args:
        model1: First model to compare
        model2: Second model to compare
    
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        'architecture_match': False,
        'parameter_match': False,
        'size_comparison': {},
        'differences': []
    }
    
    # Compare architectures
    if model1.__class__ == model2.__class__:
        comparison['architecture_match'] = True
    else:
        comparison['differences'].append(
            f"Architecture mismatch: {model1.__class__.__name__} vs {model2.__class__.__name__}"
        )
    
    # Compare model sizes
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    comparison['size_comparison'] = {
        'model1': size1,
        'model2': size2,
        'parameter_diff': size1['total_parameters'] - size2['total_parameters']
    }
    
    # Compare parameter names and shapes
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    if set(params1.keys()) != set(params2.keys()):
        comparison['differences'].append("Parameter names don't match")
    else:
        # Check parameter shapes and values
        shape_mismatches = []
        value_mismatches = []
        
        for name in params1.keys():
            if params1[name].shape != params2[name].shape:
                shape_mismatches.append(
                    f"{name}: {params1[name].shape} vs {params2[name].shape}"
                )
            else:
                # Compare parameter values
                if not torch.allclose(params1[name], params2[name], rtol=1e-5, atol=1e-8):
                    value_mismatches.append(name)
        
        if shape_mismatches:
            comparison['differences'].extend(shape_mismatches)
        
        if value_mismatches:
            comparison['differences'].append(f"Value mismatches in: {value_mismatches}")
        else:
            comparison['parameter_match'] = True
    
    return comparison


def get_layer_sizes(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about each layer in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary containing layer information
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            module_buffers = sum(b.numel() for b in module.buffers())
            
            layer_info[name] = {
                'type': module.__class__.__name__,
                'parameters': module_params,
                'buffers': module_buffers,
                'total_elements': module_params + module_buffers,
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            
            # Add layer-specific information
            if hasattr(module, 'weight') and module.weight is not None:
                layer_info[name]['weight_shape'] = list(module.weight.shape)
            
            if hasattr(module, 'bias') and module.bias is not None:
                layer_info[name]['bias_shape'] = list(module.bias.shape)
    
    return layer_info


def prune_model(
    model: nn.Module,
    pruning_ratio: float = 0.2,
    pruning_method: str = "magnitude"
) -> nn.Module:
    """
    Prune model parameters to reduce size and computational requirements.
    
    Args:
        model: PyTorch model to prune
        pruning_ratio: Fraction of parameters to prune
        pruning_method: Pruning method ("magnitude", "random")
    
    Returns:
        Pruned model
    """
    try:
        import torch.nn.utils.prune as prune
    except ImportError:
        logger.error("Pruning requires torch.nn.utils.prune (PyTorch >= 1.4)")
        return model
    
    parameters_to_prune = []
    
    # Collect linear and convolutional layers for pruning
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            parameters_to_prune.append((module, 'weight'))
    
    if not parameters_to_prune:
        logger.warning("No prunable layers found in model")
        return model
    
    # Apply pruning
    if pruning_method == "magnitude":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
    elif pruning_method == "random":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=pruning_ratio
        )
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    original_params = count_parameters(model)
    remaining_params = count_parameters(model)
    
    logger.info(f"Model pruned: {original_params} -> {remaining_params} parameters "
               f"({(1 - remaining_params/original_params)*100:.1f}% reduction)")
    
    return model 