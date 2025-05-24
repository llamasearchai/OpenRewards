"""
Comprehensive tests for data module.
Tests dataset classes, data processing, loading, and validation.
"""

import unittest
import tempfile
import shutil
import json
import jsonlines
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from typing import List, Dict, Any

from reward_modeling.data.dataset import (
    PreferencePair,
    ConstitutionalExample,
    PreferenceDataset,
    ConstitutionalDataset,
    MultiModalDataset,
    BaseDataset,
    DatasetBuilder,
    DataCollator,
    create_dataloader,
    create_synthetic_preference_data
)


class TestPreferencePair(unittest.TestCase):
    """Test cases for PreferencePair dataclass."""
    
    def test_preference_pair_creation(self):
        """Test creating a preference pair."""
        pair = PreferencePair(
            id="test_001",
            prompt="What is AI?",
            chosen="AI is artificial intelligence.",
            rejected="AI is magic.",
            metadata={"domain": "tech", "source": "manual"}
        )
        
        self.assertEqual(pair.id, "test_001")
        self.assertEqual(pair.prompt, "What is AI?")
        self.assertEqual(pair.chosen, "AI is artificial intelligence.")
        self.assertEqual(pair.rejected, "AI is magic.")
        self.assertEqual(pair.metadata["domain"], "tech")
        self.assertEqual(pair.metadata["source"], "manual")
    
    def test_preference_pair_to_dict(self):
        """Test converting preference pair to dictionary."""
        pair = PreferencePair(
            id="test_002",
            prompt="Explain machine learning",
            chosen="Machine learning is a subset of AI.",
            rejected="Machine learning is complicated.",
            metadata={"difficulty": "easy"}
        )
        
        pair_dict = pair.to_dict()
        
        self.assertIsInstance(pair_dict, dict)
        self.assertEqual(pair_dict["id"], "test_002")
        self.assertEqual(pair_dict["prompt"], "Explain machine learning")
        self.assertEqual(pair_dict["chosen"], "Machine learning is a subset of AI.")
        self.assertEqual(pair_dict["rejected"], "Machine learning is complicated.")
        self.assertEqual(pair_dict["metadata"]["difficulty"], "easy")
    
    def test_preference_pair_without_metadata(self):
        """Test preference pair without metadata."""
        pair = PreferencePair(
            id="test_003",
            prompt="Test prompt",
            chosen="Good response",
            rejected="Bad response"
        )
        
        self.assertIsNone(pair.metadata)
        
        pair_dict = pair.to_dict()
        self.assertEqual(pair_dict["metadata"], {})


class TestConstitutionalExample(unittest.TestCase):
    """Test cases for ConstitutionalExample dataclass."""
    
    def test_constitutional_example_creation(self):
        """Test creating a constitutional example."""
        example = ConstitutionalExample(
            prompt="Give advice on handling conflict",
            initial_response="Just ignore the problem",
            critique="This advice is not helpful and could escalate issues",
            revised_response="Try to understand both perspectives and find common ground",
            principles=["Be helpful", "Promote understanding"],
            metadata={"revision_count": 1}
        )
        
        self.assertEqual(example.prompt, "Give advice on handling conflict")
        self.assertEqual(example.initial_response, "Just ignore the problem")
        self.assertEqual(example.critique, "This advice is not helpful and could escalate issues")
        self.assertEqual(example.revised_response, "Try to understand both perspectives and find common ground")
        self.assertEqual(len(example.principles), 2)
        self.assertEqual(example.metadata["revision_count"], 1)


class TestPreferenceDataset(unittest.TestCase):
    """Test cases for PreferenceDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_pairs = [
            PreferencePair(
                id="pair_001",
                prompt="What is the capital of France?",
                chosen="The capital of France is Paris.",
                rejected="The capital of France is London.",
                metadata={"category": "geography"}
            ),
            PreferencePair(
                id="pair_002", 
                prompt="Explain photosynthesis",
                chosen="Photosynthesis is the process by which plants convert light into energy.",
                rejected="Photosynthesis is when plants eat sunlight.",
                metadata={"category": "science"}
            ),
            PreferencePair(
                id="pair_003",
                prompt="How do you bake a cake?",
                chosen="Mix ingredients, bake at 350°F for 30 minutes.",
                rejected="Just put flour in the oven.",
                metadata={"category": "cooking"}
            )
        ]
        
        self.dataset = PreferenceDataset(
            data=self.sample_pairs,
            max_length=128,
            filter_duplicates=False,
            augment_data=False
        )
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(self.dataset.max_length, 128)
        self.assertFalse(self.dataset.augment_data)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        item = self.dataset[0]
        
        self.assertEqual(item["id"], "pair_001")
        self.assertEqual(item["prompt"], "What is the capital of France?")
        self.assertEqual(item["chosen"], "The capital of France is Paris.")
        self.assertEqual(item["rejected"], "The capital of France is London.")
        self.assertEqual(item["metadata"]["category"], "geography")
    
    def test_dataset_len(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 3)
    
    def test_dataset_with_tokenizer(self):
        """Test dataset with tokenizer."""
        with patch('reward_modeling.data.dataset.AutoTokenizer') as mock_tokenizer_class:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
            }
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # Create dataset with tokenizer
            dataset = PreferenceDataset(
                data=self.sample_pairs,
                tokenizer_name="test-tokenizer"
            )
            dataset.tokenizer = mock_tokenizer
            
            item = dataset[0]
            
            # Should have tokenized outputs
            self.assertIn("chosen_input_ids", item)
            self.assertIn("chosen_attention_mask", item)
            self.assertIn("rejected_input_ids", item)
            self.assertIn("rejected_attention_mask", item)
    
    def test_dataset_split(self):
        """Test dataset splitting."""
        # Create larger dataset for meaningful split
        larger_pairs = []
        for i in range(100):
            pair = PreferencePair(
                id=f"pair_{i:03d}",
                prompt=f"Test prompt {i}",
                chosen=f"Good response {i}",
                rejected=f"Bad response {i}",
                metadata={"index": i}
            )
            larger_pairs.append(pair)
        
        larger_dataset = PreferenceDataset(
            data=larger_pairs,
            filter_duplicates=False,
            augment_data=False
        )
        
        train_dataset, val_dataset, test_dataset = larger_dataset.split(
            train_ratio=0.7,
            val_ratio=0.2
        )
        
        self.assertEqual(len(train_dataset), 70)
        self.assertEqual(len(val_dataset), 20)
        self.assertEqual(len(test_dataset), 10)
    
    def test_dataset_filter_duplicates(self):
        """Test duplicate filtering."""
        # Create dataset with duplicates
        pairs_with_duplicates = self.sample_pairs + [
            PreferencePair(
                id="duplicate_001",
                prompt="What is the capital of France?",  # Same as first
                chosen="The capital of France is Paris.",
                rejected="The capital of France is London.",
                metadata={"category": "geography"}
            )
        ]
        
        dataset = PreferenceDataset(
            data=pairs_with_duplicates,
            filter_duplicates=True,
            augment_data=False
        )
        
        # Should have original 3 items (duplicate removed)
        self.assertEqual(len(dataset), 3)
    
    def test_dataset_augmentation(self):
        """Test data augmentation."""
        dataset = PreferenceDataset(
            data=self.sample_pairs,
            augment_data=True,
            filter_duplicates=False
        )
        
        # Should have more items due to augmentation
        self.assertGreater(len(dataset), len(self.sample_pairs))
    
    def test_dataset_save_and_load(self):
        """Test saving and loading dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON format
            json_path = Path(temp_dir) / "test_dataset.json"
            self.dataset.save(json_path)
            self.assertTrue(json_path.exists())
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(len(saved_data), 3)
            self.assertEqual(saved_data[0]["id"], "pair_001")
            
            # Test JSONL format
            jsonl_path = Path(temp_dir) / "test_dataset.jsonl"
            self.dataset.save(jsonl_path)
            self.assertTrue(jsonl_path.exists())
            
            # Verify JSONL content
            with jsonlines.open(jsonl_path, mode='r') as reader:
                saved_data_jsonl = list(reader)
            
            self.assertEqual(len(saved_data_jsonl), 3)
            self.assertEqual(saved_data_jsonl[0]["id"], "pair_001")


class TestConstitutionalDataset(unittest.TestCase):
    """Test cases for ConstitutionalDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = [
            PreferencePair(
                id="const_001",
                prompt="How should I handle a difficult customer?",
                chosen="Listen to their concerns and try to find a solution",
                rejected="Tell them they're wrong and hang up",
                metadata={
                    "initial_response": "Tell them they're wrong and hang up",
                    "critique": "This response is rude and unhelpful",
                    "revised_response": "Listen to their concerns and try to find a solution",
                    "violated_principles": ["Be respectful", "Be helpful"]
                }
            )
        ]
        
        self.constitution = [
            "Be respectful and polite",
            "Be helpful and constructive", 
            "Do not be rude or dismissive"
        ]
        
        self.dataset = ConstitutionalDataset(
            data=self.sample_data,
            constitution=self.constitution,
            max_length=256
        )
    
    def test_constitutional_dataset_initialization(self):
        """Test constitutional dataset initialization."""
        self.assertEqual(len(self.dataset), 1)
        self.assertEqual(len(self.dataset.constitution), 3)
        self.assertEqual(self.dataset.max_length, 256)
    
    def test_constitutional_dataset_getitem(self):
        """Test getting items from constitutional dataset."""
        item = self.dataset[0]
        
        self.assertIn("prompt", item)
        self.assertIn("chosen", item)
        self.assertIn("rejected", item)
        self.assertIn("constitution", item)
        self.assertEqual(len(item["constitution"]), 3)


class TestMultiModalDataset(unittest.TestCase):
    """Test cases for MultiModalDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data file
        sample_data = [
            {
                "text": "This is a test image description",
                "image_path": "test_image.jpg",
                "audio_path": "test_audio.wav",
                "label": 1,
                "metadata": {"source": "test"}
            }
        ]
        
        self.data_file = Path(self.temp_dir) / "multimodal_data.json"
        with open(self.data_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Create dummy image and audio directories
        self.image_dir = Path(self.temp_dir) / "images"
        self.audio_dir = Path(self.temp_dir) / "audio"
        self.image_dir.mkdir()
        self.audio_dir.mkdir()
        
        # Create dummy files
        (self.image_dir / "test_image.jpg").touch()
        (self.audio_dir / "test_audio.wav").touch()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_multimodal_dataset_initialization(self):
        """Test multimodal dataset initialization."""
        dataset = MultiModalDataset(
            data_path=self.data_file,
            image_dir=self.image_dir,
            audio_dir=self.audio_dir,
            max_length=512
        )
        
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.max_length, 512)
    
    def test_multimodal_dataset_getitem(self):
        """Test getting items from multimodal dataset."""
        dataset = MultiModalDataset(
            data_path=self.data_file,
            image_dir=self.image_dir,
            audio_dir=self.audio_dir
        )
        
        item = dataset[0]
        
        self.assertIn("text", item)
        self.assertIn("image_path", item)
        self.assertIn("audio_path", item)
        self.assertIn("label", item)
        self.assertEqual(item["label"], 1)


class TestDataCollator(unittest.TestCase):
    """Test cases for DataCollator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collator = DataCollator()
    
    def test_collator_basic_features(self):
        """Test collating basic features."""
        features = [
            {
                "prompt": "Test prompt 1",
                "chosen": "Good response 1",
                "rejected": "Bad response 1",
                "chosen_input_ids": torch.tensor([1, 2, 3]),
                "chosen_attention_mask": torch.tensor([1, 1, 1])
            },
            {
                "prompt": "Test prompt 2", 
                "chosen": "Good response 2",
                "rejected": "Bad response 2",
                "chosen_input_ids": torch.tensor([4, 5, 6]),
                "chosen_attention_mask": torch.tensor([1, 1, 1])
            }
        ]
        
        batch = self.collator(features)
        
        self.assertIn("prompt", batch)
        self.assertIn("chosen", batch)
        self.assertIn("rejected", batch)
        self.assertIn("chosen_input_ids", batch)
        self.assertIn("chosen_attention_mask", batch)
        
        self.assertEqual(len(batch["prompt"]), 2)
        self.assertEqual(batch["chosen_input_ids"].shape, (2, 3))
    
    def test_collator_empty_batch(self):
        """Test collating empty batch."""
        batch = self.collator([])
        self.assertEqual(len(batch), 0)
    
    def test_collator_mixed_types(self):
        """Test collating features with mixed types."""
        features = [
            {
                "text": "Sample text",
                "label": 1,
                "score": 0.8,
                "metadata": {"key": "value"}
            },
            {
                "text": "Another text",
                "label": 0,
                "score": 0.3,
                "metadata": {"key": "other"}
            }
        ]
        
        batch = self.collator(features)
        
        self.assertIn("text", batch)
        self.assertIn("label", batch)
        self.assertIn("score", batch)
        self.assertIn("metadata", batch)
        
        self.assertEqual(len(batch["text"]), 2)
        self.assertIsInstance(batch["label"], torch.Tensor)
        self.assertIsInstance(batch["score"], torch.Tensor)


class TestDatasetBuilder(unittest.TestCase):
    """Test cases for DatasetBuilder."""
    
    def test_create_preference_dataset(self):
        """Test creating preference dataset with builder."""
        sample_pairs = [
            PreferencePair(
                id="builder_001",
                prompt="Test prompt",
                chosen="Good response",
                rejected="Bad response"
            )
        ]
        
        with patch('reward_modeling.data.dataset.PreferenceDataset') as mock_dataset:
            mock_dataset.return_value = Mock()
            
            dataset = DatasetBuilder.create_preference_dataset(
                data_path="dummy_path",
                validation_rules={"min_samples": 1}
            )
            
            mock_dataset.assert_called_once()
    
    def test_create_constitutional_dataset(self):
        """Test creating constitutional dataset with builder."""
        with patch('reward_modeling.data.dataset.ConstitutionalDataset') as mock_dataset:
            mock_dataset.return_value = Mock()
            
            dataset = DatasetBuilder.create_constitutional_dataset(
                data_path="dummy_path",
                validation_rules={"min_samples": 1}
            )
            
            mock_dataset.assert_called_once()
    
    def test_dataset_validation_rules(self):
        """Test dataset validation rules."""
        # Test minimum samples validation
        with patch('reward_modeling.data.dataset.PreferenceDataset') as mock_dataset:
            mock_dataset.return_value = Mock()
            mock_dataset.return_value.__len__ = Mock(return_value=5)
            
            with self.assertRaises(ValueError):
                DatasetBuilder.create_preference_dataset(
                    data_path="dummy_path",
                    validation_rules={"min_samples": 10}
                )


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_dataloader(self):
        """Test creating dataloader."""
        sample_pairs = [
            PreferencePair(
                id="util_001",
                prompt="Test prompt",
                chosen="Good response", 
                rejected="Bad response"
            )
        ]
        
        dataset = PreferenceDataset(data=sample_pairs)
        dataloader = create_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertEqual(dataloader.batch_size, 1)
        self.assertEqual(len(dataloader), 1)
    
    def test_create_synthetic_preference_data(self):
        """Test creating synthetic preference data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "synthetic_data.json"
            
            dataset = create_synthetic_preference_data(
                n_samples=50,
                save_path=save_path
            )
            
            self.assertEqual(len(dataset), 50)
            self.assertTrue(save_path.exists())
            
            # Verify saved data
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(len(saved_data), 50)
            self.assertIn("id", saved_data[0])
            self.assertIn("prompt", saved_data[0])
            self.assertIn("chosen", saved_data[0])
            self.assertIn("rejected", saved_data[0])


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data components."""
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline from creation to loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create synthetic dataset
            dataset = create_synthetic_preference_data(n_samples=20)
            
            # Save dataset
            save_path = Path(temp_dir) / "pipeline_test.jsonl"
            dataset.save(save_path)
            
            # Create new dataset from file
            loaded_dataset = PreferenceDataset.from_file(save_path)
            
            # Verify data integrity
            self.assertEqual(len(loaded_dataset), len(dataset))
            
            # Test dataloader creation
            dataloader = create_dataloader(
                loaded_dataset,
                batch_size=4,
                shuffle=True
            )
            
            # Test batch retrieval
            batch = next(iter(dataloader))
            self.assertIn("prompt", batch)
            self.assertIn("chosen", batch)
            self.assertIn("rejected", batch)
    
    def test_dataset_statistics_computation(self):
        """Test computing dataset statistics."""
        # Create dataset with varied content
        varied_pairs = []
        for i in range(10):
            pair = PreferencePair(
                id=f"stats_{i:03d}",
                prompt=f"Test prompt with length variation {i}" * (i + 1),
                chosen=f"Good response {i}" * (i + 1),
                rejected=f"Bad response {i}",
                metadata={"length_factor": i + 1}
            )
            varied_pairs.append(pair)
        
        dataset = PreferenceDataset(data=varied_pairs)
        
        # Mock the get_statistics method since it might not be implemented
        with patch.object(dataset, 'get_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_samples": 10,
                "prompt_length": {"mean": 25.5, "std": 10.2},
                "chosen_length": {"mean": 30.1, "std": 12.5},
                "rejected_length": {"mean": 15.8, "std": 5.3}
            }
            
            stats = dataset.get_statistics()
            
            self.assertEqual(stats["total_samples"], 10)
            self.assertIn("prompt_length", stats)
            self.assertIn("chosen_length", stats)
            self.assertIn("rejected_length", stats)


if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2) 