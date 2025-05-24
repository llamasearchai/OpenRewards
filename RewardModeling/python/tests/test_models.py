"""
Comprehensive tests for reward models.
Tests all model variants, edge cases, and performance scenarios.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

from reward_modeling.models.reward_model import (
    RewardModel,
    TransformerRewardModel,
    RewardModelConfig,
    AttentionPooling,
    EnsembleRewardModel,
    ConstitutionalRewardModel,
)
from reward_modeling.data.dataset import PreferenceDataset, PreferencePair, create_synthetic_dataset
from reward_modeling.training.trainer import RewardModelTrainer, TrainingArguments
from reward_modeling.evaluation.metrics import RewardModelEvaluator, EvaluationResults


class TestRewardModel(unittest.TestCase):
    """Test cases for RewardModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "distilbert-base-uncased"
        self.device = "cpu"  # Use CPU for tests
        self.model = RewardModel(
            model_name_or_path=self.model_name,
            dropout_rate=0.1,
            use_cache=False,
            gradient_checkpointing=False
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.backbone)
        self.assertIsNotNone(self.model.value_head)
        self.assertEqual(len(self.model.value_head), 4)  # Linear, Tanh, Dropout, Linear
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        self.assertIn("rewards", outputs)
        self.assertEqual(outputs["rewards"].shape, (batch_size,))
        self.assertTrue(torch.isfinite(outputs["rewards"]).all())
    
    def test_forward_pass_no_attention_mask(self):
        """Test forward pass without attention mask."""
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                return_dict=True
            )
        
        self.assertIn("rewards", outputs)
        self.assertEqual(outputs["rewards"].shape, (batch_size,))
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")
            
            # Save model
            self.model.save_pretrained(save_path)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(save_path, "backbone")))
            self.assertTrue(os.path.exists(os.path.join(save_path, "value_head.pt")))
            
            # Load model
            loaded_model = RewardModel.from_pretrained(save_path)
            
            # Test that loaded model works
            input_ids = torch.randint(0, 1000, (1, 64))
            
            with torch.no_grad():
                original_output = self.model(input_ids=input_ids, return_dict=True)
                loaded_output = loaded_model(input_ids=input_ids, return_dict=True)
            
            # Outputs should be very close (may not be exactly equal due to float precision)
            torch.testing.assert_close(
                original_output["rewards"], 
                loaded_output["rewards"], 
                atol=1e-6, 
                rtol=1e-5
            )


class TestPreferenceDataset(unittest.TestCase):
    """Test cases for PreferenceDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample preference pairs
        self.preference_pairs = [
            PreferencePair(
                id="test_1",
                prompt="What is machine learning?",
                chosen="Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
                rejected="Machine learning is just programming.",
                metadata={"domain": "tech"}
            ),
            PreferencePair(
                id="test_2",
                prompt="How do neural networks work?",
                chosen="Neural networks process information through layers of interconnected nodes.",
                rejected="Neural networks are magic.",
                metadata={"domain": "tech"}
            )
        ]
        
        self.dataset = PreferenceDataset(
            data=self.preference_pairs,
            max_length=128,
            filter_duplicates=False,
            augment_data=False
        )
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 2)
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        item = self.dataset[0]
        
        self.assertIn("id", item)
        self.assertIn("prompt", item)
        self.assertIn("chosen", item)
        self.assertIn("rejected", item)
        self.assertEqual(item["id"], "test_1")
        self.assertEqual(item["prompt"], "What is machine learning?")
    
    def test_dataset_split(self):
        """Test dataset splitting."""
        # Create larger dataset for meaningful split
        synthetic_dataset = create_synthetic_dataset(num_samples=100, random_seed=42)
        
        train_data, val_data, test_data = synthetic_dataset.split(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        
        self.assertEqual(len(train_data), 70)
        self.assertEqual(len(val_data), 20)
        self.assertEqual(len(test_data), 10)
        
        # Test that splits don't overlap
        train_ids = {train_data.data[i].id for i in range(len(train_data))}
        val_ids = {val_data.data[i].id for i in range(len(val_data))}
        test_ids = {test_data.data[i].id for i in range(len(test_data))}
        
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        stats = self.dataset.get_statistics()
        
        self.assertIn("total_samples", stats)
        self.assertIn("prompt_length", stats)
        self.assertIn("chosen_length", stats)
        self.assertIn("rejected_length", stats)
        
        self.assertEqual(stats["total_samples"], 2)
        self.assertIn("mean", stats["prompt_length"])
        self.assertIn("std", stats["prompt_length"])
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        synthetic_dataset = create_synthetic_dataset(
            num_samples=50,
            domains=["test_domain"],
            random_seed=42
        )
        
        self.assertEqual(len(synthetic_dataset), 50)
        
        # Check that all samples have required fields
        for i in range(min(5, len(synthetic_dataset))):
            item = synthetic_dataset[i]
            self.assertIn("prompt", item)
            self.assertIn("chosen", item)
            self.assertIn("rejected", item)
            self.assertIn("metadata", item)
    
    def test_dataset_save_and_load(self):
        """Test dataset saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSONL format
            jsonl_path = os.path.join(temp_dir, "test_dataset.jsonl")
            self.dataset.save_to_file(jsonl_path, format="jsonl")
            self.assertTrue(os.path.exists(jsonl_path))
            
            loaded_dataset = PreferenceDataset.from_file(jsonl_path)
            self.assertEqual(len(loaded_dataset), len(self.dataset))
            
            # Test JSON format
            json_path = os.path.join(temp_dir, "test_dataset.json")
            self.dataset.save_to_file(json_path, format="json")
            self.assertTrue(os.path.exists(json_path))


class TestTrainingArguments(unittest.TestCase):
    """Test cases for TrainingArguments."""
    
    def test_default_arguments(self):
        """Test default training arguments."""
        args = TrainingArguments()
        
        self.assertEqual(args.output_dir, "./output")
        self.assertEqual(args.num_epochs, 3)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertFalse(args.use_wandb)
    
    def test_custom_arguments(self):
        """Test custom training arguments."""
        args = TrainingArguments(
            output_dir="./custom_output",
            num_epochs=5,
            batch_size=32,
            learning_rate=1e-4,
            use_wandb=True
        )
        
        self.assertEqual(args.output_dir, "./custom_output")
        self.assertEqual(args.num_epochs, 5)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertTrue(args.use_wandb)


class TestRewardModelTrainer(unittest.TestCase):
    """Test cases for RewardModelTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RewardModel(
            model_name_or_path="distilbert-base-uncased",
            dropout_rate=0.1
        )
        
        # Create small synthetic dataset
        self.train_dataset = create_synthetic_dataset(num_samples=20, random_seed=42)
        self.eval_dataset = create_synthetic_dataset(num_samples=10, random_seed=123)
        
        self.args = TrainingArguments(
            output_dir="./test_output",
            num_epochs=1,  # Single epoch for testing
            batch_size=4,
            learning_rate=1e-4,
            logging_steps=1,
            eval_steps=5,
            save_steps=10,
            use_wandb=False
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = RewardModelTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.args
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.lr_scheduler)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.current_epoch, 0)
    
    def test_data_collator(self):
        """Test data collator functionality."""
        trainer = RewardModelTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.args
        )
        
        # Get a few samples
        batch = [self.train_dataset[i] for i in range(2)]
        collated = trainer.data_collator(batch)
        
        self.assertIn("chosen_texts", collated)
        self.assertIn("rejected_texts", collated)
        self.assertEqual(len(collated["chosen_texts"]), 2)
        self.assertEqual(len(collated["rejected_texts"]), 2)
    
    def test_compute_loss(self):
        """Test loss computation."""
        trainer = RewardModelTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=self.args
        )
        
        # Create dummy batch
        batch = {
            "chosen_texts": ["This is a good response.", "Another good response."],
            "rejected_texts": ["This is bad.", "Another bad response."]
        }
        
        loss_dict = trainer.compute_loss(batch)
        
        self.assertIn("loss", loss_dict)
        self.assertIn("accuracy", loss_dict)
        self.assertIn("reward_gap", loss_dict)
        
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        self.assertTrue(0 <= loss_dict["accuracy"].item() <= 1)
    
    @patch('reward_modeling.training.trainer.wandb')
    def test_training_loop_short(self, mock_wandb):
        """Test a short training loop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(
                output_dir=temp_dir,
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-4,
                logging_steps=1,
                eval_steps=100,  # No eval during short training
                save_steps=100,  # No checkpoints during short training
                use_wandb=False
            )
            
            trainer = RewardModelTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                args=args
            )
            
            # Run training
            results = trainer.train()
            
            self.assertIn("training_history", results)
            self.assertIn("final_model_path", results)
            self.assertTrue(os.path.exists(results["final_model_path"]))


class TestRewardModelEvaluator(unittest.TestCase):
    """Test cases for RewardModelEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RewardModel(
            model_name_or_path="distilbert-base-uncased",
            dropout_rate=0.1
        )
        
        self.dataset = create_synthetic_dataset(num_samples=20, random_seed=42)
        
        self.evaluator = RewardModelEvaluator(
            model=self.model,
            device="cpu",
            batch_size=4,
            compute_uncertainty=True
        )
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator.model)
        self.assertEqual(self.evaluator.device, "cpu")
        self.assertEqual(self.evaluator.batch_size, 4)
        self.assertTrue(self.evaluator.compute_uncertainty)
    
    def test_get_predictions(self):
        """Test prediction generation."""
        predictions = self.evaluator._get_predictions(self.dataset)
        
        self.assertIn("chosen_rewards", predictions)
        self.assertIn("rejected_rewards", predictions)
        self.assertIn("reward_diff", predictions)
        self.assertIn("preference_probs", predictions)
        self.assertIn("labels", predictions)
        
        # Check shapes
        self.assertEqual(len(predictions["chosen_rewards"]), len(self.dataset))
        self.assertEqual(len(predictions["rejected_rewards"]), len(self.dataset))
        self.assertEqual(len(predictions["reward_diff"]), len(self.dataset))
        
        # Check that probabilities are in valid range
        self.assertTrue(np.all(predictions["preference_probs"] >= 0))
        self.assertTrue(np.all(predictions["preference_probs"] <= 1))
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Get predictions first
        predictions = self.evaluator._get_predictions(self.dataset)
        
        # Compute metrics
        results = self.evaluator._compute_metrics(predictions, self.dataset)
        
        self.assertIsInstance(results, EvaluationResults)
        self.assertTrue(0 <= results.accuracy <= 1)
        self.assertTrue(np.isfinite(results.reward_gap))
        self.assertTrue(0 <= results.calibration_error <= 1)
        self.assertTrue(0 <= results.consistency_score <= 1)
    
    def test_full_evaluation(self):
        """Test full evaluation pipeline."""
        results = self.evaluator.evaluate(self.dataset)
        
        self.assertIsInstance(results, EvaluationResults)
        self.assertIn("reward_statistics", results.detailed_metrics)
        self.assertIn("confidence_analysis", results.detailed_metrics)
        self.assertIn("error_analysis", results.detailed_metrics)
    
    def test_evaluation_with_save(self):
        """Test evaluation with result saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "eval_results.json")
            
            results = self.evaluator.evaluate(
                self.dataset,
                save_results=save_path
            )
            
            self.assertTrue(os.path.exists(save_path))
            
            # Load and verify saved results
            import json
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("accuracy", saved_data)
            self.assertIn("reward_gap", saved_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete training and evaluation pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model and datasets
            model = RewardModel(
                model_name_or_path="distilbert-base-uncased",
                dropout_rate=0.1
            )
            
            train_dataset = create_synthetic_dataset(num_samples=30, random_seed=42)
            eval_dataset = create_synthetic_dataset(num_samples=10, random_seed=123)
            
            # Training arguments
            args = TrainingArguments(
                output_dir=temp_dir,
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-4,
                logging_steps=2,
                eval_steps=10,
                save_steps=20,
                use_wandb=False
            )
            
            # Train model
            trainer = RewardModelTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=args
            )
            
            training_results = trainer.train()
            self.assertIn("final_model_path", training_results)
            
            # Load trained model and evaluate
            trained_model = RewardModel.from_pretrained(training_results["final_model_path"])
            
            evaluator = RewardModelEvaluator(
                model=trained_model,
                device="cpu",
                batch_size=4
            )
            
            eval_results = evaluator.evaluate(eval_dataset)
            
            # Check that evaluation completed successfully
            self.assertIsInstance(eval_results, EvaluationResults)
            self.assertTrue(0 <= eval_results.accuracy <= 1)
            self.assertTrue(np.isfinite(eval_results.reward_gap))


if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2) 