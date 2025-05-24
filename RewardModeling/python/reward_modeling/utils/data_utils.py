"""
Data Processing Utilities for Reward Modeling Platform

This module provides comprehensive utilities for data loading, processing,
validation, and augmentation for reward modeling datasets.
"""

import json
import jsonlines
import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from collections import defaultdict, Counter
import random
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from ..data.dataset import PreferencePair, PreferenceDataset

logger = logging.getLogger(__name__)


def load_dataset(
    file_path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Load dataset from various file formats.
    
    Args:
        file_path: Path to the dataset file
        format: File format ('json', 'jsonl', 'csv', 'parquet'). Auto-detected if None
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        List of data samples as dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Auto-detect format from extension
    if format is None:
        format = file_path.suffix.lower().lstrip('.')
    
    logger.info(f"Loading dataset from {file_path} with format: {format}")
    
    try:
        if format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        
        elif format == 'jsonl':
            data = []
            with jsonlines.open(file_path, mode='r') as reader:
                for item in reader:
                    data.append(item)
            return data
        
        elif format == 'csv':
            df = pd.read_csv(file_path, **kwargs)
            return df.to_dict('records')
        
        elif format == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {format}")
    
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {str(e)}")
        raise


def save_dataset(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save dataset to various file formats.
    
    Args:
        data: List of data samples as dictionaries
        file_path: Path to save the dataset
        format: File format ('json', 'jsonl', 'csv', 'parquet'). Auto-detected if None
        **kwargs: Additional arguments for specific savers
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from extension
    if format is None:
        format = file_path.suffix.lower().lstrip('.')
    
    logger.info(f"Saving dataset to {file_path} with format: {format}")
    
    try:
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'jsonl':
            with jsonlines.open(file_path, mode='w') as writer:
                for item in data:
                    writer.write(item)
        
        elif format == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, **kwargs)
        
        elif format == 'parquet':
            df = pd.DataFrame(data)
            df.to_parquet(file_path, index=False, **kwargs)
        
        else:
            raise ValueError(f"Unsupported file format: {format}")
    
    except Exception as e:
        logger.error(f"Error saving dataset to {file_path}: {str(e)}")
        raise


def validate_dataset(
    data: List[Dict[str, Any]],
    required_fields: List[str],
    validation_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate dataset structure and content.
    
    Args:
        data: List of data samples
        required_fields: List of required field names
        validation_rules: Additional validation rules
    
    Returns:
        Validation report with statistics and errors
    """
    validation_report = {
        'total_samples': len(data),
        'valid_samples': 0,
        'invalid_samples': 0,
        'errors': [],
        'warnings': [],
        'field_statistics': {}
    }
    
    if not data:
        validation_report['errors'].append("Dataset is empty")
        return validation_report
    
    # Check required fields
    for i, sample in enumerate(data):
        sample_valid = True
        
        for field in required_fields:
            if field not in sample:
                validation_report['errors'].append(
                    f"Sample {i}: Missing required field '{field}'"
                )
                sample_valid = False
            elif sample[field] is None or sample[field] == "":
                validation_report['warnings'].append(
                    f"Sample {i}: Empty value for required field '{field}'"
                )
        
        if sample_valid:
            validation_report['valid_samples'] += 1
        else:
            validation_report['invalid_samples'] += 1
    
    # Compute field statistics
    all_fields = set()
    for sample in data:
        all_fields.update(sample.keys())
    
    for field in all_fields:
        field_stats = {
            'present_count': 0,
            'missing_count': 0,
            'null_count': 0,
            'empty_count': 0
        }
        
        for sample in data:
            if field in sample:
                field_stats['present_count'] += 1
                if sample[field] is None:
                    field_stats['null_count'] += 1
                elif isinstance(sample[field], str) and sample[field].strip() == "":
                    field_stats['empty_count'] += 1
            else:
                field_stats['missing_count'] += 1
        
        validation_report['field_statistics'][field] = field_stats
    
    # Apply additional validation rules
    if validation_rules:
        apply_validation_rules(data, validation_rules, validation_report)
    
    validation_report['validation_passed'] = len(validation_report['errors']) == 0
    
    logger.info(f"Dataset validation completed: {validation_report['valid_samples']}/{validation_report['total_samples']} valid samples")
    
    return validation_report


def apply_validation_rules(
    data: List[Dict[str, Any]],
    rules: Dict[str, Any],
    report: Dict[str, Any]
) -> None:
    """Apply custom validation rules to dataset."""
    
    # Minimum samples rule
    if 'min_samples' in rules:
        min_samples = rules['min_samples']
        if len(data) < min_samples:
            report['errors'].append(f"Dataset has {len(data)} samples, minimum required: {min_samples}")
    
    # Text length rules
    if 'text_length' in rules:
        text_rules = rules['text_length']
        for field in text_rules.get('fields', []):
            min_length = text_rules.get('min_length', 0)
            max_length = text_rules.get('max_length', float('inf'))
            
            for i, sample in enumerate(data):
                if field in sample and isinstance(sample[field], str):
                    text_length = len(sample[field])
                    if text_length < min_length:
                        report['warnings'].append(
                            f"Sample {i}: Field '{field}' too short ({text_length} < {min_length})"
                        )
                    elif text_length > max_length:
                        report['warnings'].append(
                            f"Sample {i}: Field '{field}' too long ({text_length} > {max_length})"
                        )


def compute_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for the dataset.
    
    Args:
        data: List of data samples
    
    Returns:
        Dictionary containing dataset statistics
    """
    if not data:
        return {'error': 'Empty dataset'}
    
    stats = {
        'total_samples': len(data),
        'field_statistics': {},
        'text_statistics': {},
        'general_statistics': {}
    }
    
    # Collect all fields
    all_fields = set()
    for sample in data:
        all_fields.update(sample.keys())
    
    # Compute field-specific statistics
    for field in all_fields:
        field_values = [sample.get(field) for sample in data if field in sample]
        field_stats = {
            'count': len(field_values),
            'null_count': sum(1 for v in field_values if v is None),
            'type_distribution': Counter(type(v).__name__ for v in field_values if v is not None)
        }
        
        # Text field statistics
        text_values = [v for v in field_values if isinstance(v, str)]
        if text_values:
            lengths = [len(v) for v in text_values]
            words = [len(v.split()) for v in text_values]
            
            field_stats.update({
                'text_count': len(text_values),
                'avg_length': np.mean(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_words': np.mean(words),
                'min_words': min(words),
                'max_words': max(words)
            })
        
        # Numeric field statistics
        numeric_values = [v for v in field_values if isinstance(v, (int, float))]
        if numeric_values:
            field_stats.update({
                'numeric_count': len(numeric_values),
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'median': np.median(numeric_values)
            })
        
        stats['field_statistics'][field] = field_stats
    
    # Overall text statistics
    all_texts = []
    for sample in data:
        for value in sample.values():
            if isinstance(value, str):
                all_texts.append(value)
    
    if all_texts:
        all_lengths = [len(text) for text in all_texts]
        all_words = [len(text.split()) for text in all_texts]
        
        stats['text_statistics'] = {
            'total_texts': len(all_texts),
            'avg_length': np.mean(all_lengths),
            'avg_words': np.mean(all_words),
            'vocabulary_size': len(set(' '.join(all_texts).split()))
        }
    
    logger.info(f"Computed statistics for dataset with {stats['total_samples']} samples")
    
    return stats


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify_field: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
        stratify_field: Field to use for stratified splitting
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Sum of ratios must equal 1.0")
    
    if len(data) == 0:
        return [], [], []
    
    # Prepare stratification labels if specified
    stratify = None
    if stratify_field:
        stratify = [sample.get(stratify_field) for sample in data]
        if any(label is None for label in stratify):
            logger.warning(f"Some samples missing stratification field '{stratify_field}', using random split")
            stratify = None
    
    # First split: separate test set
    if test_ratio > 0:
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify
        )
    else:
        train_val_data, test_data = data, []
    
    # Second split: separate train and validation from remaining data
    if val_ratio > 0 and len(train_val_data) > 1:
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        
        # Update stratification for remaining data
        stratify_remaining = None
        if stratify_field and stratify:
            test_indices = set(data.index(sample) for sample in test_data)
            stratify_remaining = [
                sample.get(stratify_field) for i, sample in enumerate(data)
                if i not in test_indices
            ]
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=stratify_remaining
        )
    else:
        train_data, val_data = train_val_data, []
    
    logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def clean_text(text: str, remove_special_chars: bool = True, normalize_whitespace: bool = True) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Input text to clean
        remove_special_chars: Whether to remove special characters
        normalize_whitespace: Whether to normalize whitespace
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove or replace problematic characters
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove special characters if requested
    if remove_special_chars:
        # Keep alphanumeric, basic punctuation, and whitespace
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', ' ', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def tokenize_text(
    texts: Union[str, List[str]],
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
    **tokenizer_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Tokenize text using HuggingFace tokenizers.
    
    Args:
        texts: Text(s) to tokenize
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        **tokenizer_kwargs: Additional tokenizer arguments
    
    Returns:
        Dictionary containing tokenized outputs
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
        **tokenizer_kwargs
    )
    
    return encoded


def pad_sequences(
    sequences: List[List[int]],
    max_length: Optional[int] = None,
    padding_value: int = 0,
    truncate_from: str = "right"
) -> List[List[int]]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences to pad
        max_length: Maximum length (uses longest sequence if None)
        padding_value: Value to use for padding
        truncate_from: Where to truncate if too long ("left" or "right")
    
    Returns:
        List of padded sequences
    """
    if not sequences:
        return []
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    
    for seq in sequences:
        # Truncate if necessary
        if len(seq) > max_length:
            if truncate_from == "left":
                seq = seq[-max_length:]
            else:  # truncate_from == "right"
                seq = seq[:max_length]
        
        # Pad if necessary
        if len(seq) < max_length:
            padding_length = max_length - len(seq)
            seq = seq + [padding_value] * padding_length
        
        padded_sequences.append(seq)
    
    return padded_sequences


def augment_dataset(
    data: List[Dict[str, Any]],
    augmentation_strategies: List[str],
    augmentation_ratio: float = 0.5,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Augment dataset using various strategies.
    
    Args:
        data: Original dataset
        augmentation_strategies: List of strategies to apply
        augmentation_ratio: Ratio of original data to augment
        **kwargs: Additional arguments for augmentation strategies
    
    Returns:
        Augmented dataset
    """
    augmented_data = data.copy()
    n_to_augment = int(len(data) * augmentation_ratio)
    
    if n_to_augment == 0:
        return augmented_data
    
    # Select samples to augment
    samples_to_augment = random.sample(data, min(n_to_augment, len(data)))
    
    for strategy in augmentation_strategies:
        if strategy == "paraphrase":
            augmented_data.extend(_paraphrase_augmentation(samples_to_augment, **kwargs))
        elif strategy == "backtranslation":
            augmented_data.extend(_backtranslation_augmentation(samples_to_augment, **kwargs))
        elif strategy == "synonym_replacement":
            augmented_data.extend(_synonym_replacement_augmentation(samples_to_augment, **kwargs))
        elif strategy == "random_insertion":
            augmented_data.extend(_random_insertion_augmentation(samples_to_augment, **kwargs))
        elif strategy == "random_swap":
            augmented_data.extend(_random_swap_augmentation(samples_to_augment, **kwargs))
        elif strategy == "random_deletion":
            augmented_data.extend(_random_deletion_augmentation(samples_to_augment, **kwargs))
        else:
            logger.warning(f"Unknown augmentation strategy: {strategy}")
    
    logger.info(f"Dataset augmented from {len(data)} to {len(augmented_data)} samples")
    
    return augmented_data


def _paraphrase_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Simple paraphrase augmentation (placeholder implementation)."""
    # This is a placeholder - in practice, you'd use a paraphrasing model
    augmented = []
    text_fields = kwargs.get('text_fields', ['prompt', 'chosen', 'rejected'])
    
    for sample in samples:
        new_sample = sample.copy()
        new_sample['id'] = f"{sample.get('id', 'unknown')}_paraphrase"
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                # Simple word order shuffling as placeholder
                words = sample[field].split()
                if len(words) > 3:
                    # Randomly swap adjacent words
                    for _ in range(len(words) // 4):
                        i = random.randint(0, len(words) - 2)
                        words[i], words[i + 1] = words[i + 1], words[i]
                    new_sample[field] = ' '.join(words)
        
        augmented.append(new_sample)
    
    return augmented


def _synonym_replacement_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Simple synonym replacement augmentation."""
    # Placeholder implementation
    augmented = []
    text_fields = kwargs.get('text_fields', ['prompt', 'chosen', 'rejected'])
    replacement_rate = kwargs.get('replacement_rate', 0.1)
    
    # Simple word replacements (in practice, use WordNet or similar)
    simple_replacements = {
        'good': 'excellent',
        'bad': 'poor',
        'big': 'large',
        'small': 'tiny',
        'fast': 'quick',
        'slow': 'sluggish'
    }
    
    for sample in samples:
        new_sample = sample.copy()
        new_sample['id'] = f"{sample.get('id', 'unknown')}_synonym"
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                words = sample[field].split()
                n_replace = max(1, int(len(words) * replacement_rate))
                
                for _ in range(n_replace):
                    if words:
                        idx = random.randint(0, len(words) - 1)
                        word = words[idx].lower()
                        if word in simple_replacements:
                            words[idx] = simple_replacements[word]
                
                new_sample[field] = ' '.join(words)
        
        augmented.append(new_sample)
    
    return augmented


def _random_swap_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Random word swap augmentation."""
    augmented = []
    text_fields = kwargs.get('text_fields', ['prompt', 'chosen', 'rejected'])
    swap_rate = kwargs.get('swap_rate', 0.1)
    
    for sample in samples:
        new_sample = sample.copy()
        new_sample['id'] = f"{sample.get('id', 'unknown')}_swap"
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                words = sample[field].split()
                n_swaps = max(1, int(len(words) * swap_rate))
                
                for _ in range(n_swaps):
                    if len(words) > 1:
                        i, j = random.sample(range(len(words)), 2)
                        words[i], words[j] = words[j], words[i]
                
                new_sample[field] = ' '.join(words)
        
        augmented.append(new_sample)
    
    return augmented


def _random_deletion_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Random word deletion augmentation."""
    augmented = []
    text_fields = kwargs.get('text_fields', ['prompt', 'chosen', 'rejected'])
    deletion_rate = kwargs.get('deletion_rate', 0.1)
    
    for sample in samples:
        new_sample = sample.copy()
        new_sample['id'] = f"{sample.get('id', 'unknown')}_deletion"
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                words = sample[field].split()
                n_delete = max(0, int(len(words) * deletion_rate))
                
                if n_delete > 0 and len(words) > n_delete:
                    indices_to_delete = random.sample(range(len(words)), n_delete)
                    words = [word for i, word in enumerate(words) if i not in indices_to_delete]
                
                new_sample[field] = ' '.join(words)
        
        augmented.append(new_sample)
    
    return augmented


def _random_insertion_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Random word insertion augmentation."""
    augmented = []
    text_fields = kwargs.get('text_fields', ['prompt', 'chosen', 'rejected'])
    insertion_rate = kwargs.get('insertion_rate', 0.1)
    
    # Common words for insertion
    common_words = ['the', 'and', 'or', 'but', 'so', 'very', 'really', 'quite']
    
    for sample in samples:
        new_sample = sample.copy()
        new_sample['id'] = f"{sample.get('id', 'unknown')}_insertion"
        
        for field in text_fields:
            if field in sample and isinstance(sample[field], str):
                words = sample[field].split()
                n_insert = max(0, int(len(words) * insertion_rate))
                
                for _ in range(n_insert):
                    if words:
                        insert_pos = random.randint(0, len(words))
                        insert_word = random.choice(common_words)
                        words.insert(insert_pos, insert_word)
                
                new_sample[field] = ' '.join(words)
        
        augmented.append(new_sample)
    
    return augmented


def _backtranslation_augmentation(samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Backtranslation augmentation (placeholder implementation)."""
    # This would require translation models in practice
    logger.warning("Backtranslation augmentation not implemented - using identity transformation")
    return [
        {**sample, 'id': f"{sample.get('id', 'unknown')}_backtrans"}
        for sample in samples
    ] 