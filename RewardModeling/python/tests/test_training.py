"""
Comprehensive tests for training module.
Tests all trainer classes, training arguments, and training utilities.
"""

import unittest
import tempfile
import shutil
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any

from reward_modeling.training.trainer import (
    TrainingArguments,
    BaseTrainer,
    RewardModelTrainer,
    DPOTrainer,
    ConstitutionalTrainer,
    MultiObjectiveTrainer,
    TrainingState,
    create_trainer,
    compute_reward_metrics,
    load_checkpoint,
    save_checkpoint
)
from reward_modeling.training.dpo_trainer import (
    DirectPreferenceOptimizationTrainer,
    DPOTrainingArguments
)
from reward_modeling.training.distributed import (
    DistributedConfig,
    DistributedManager,
    DistributedDataParallelWrapper,
    setup_distributed_training,
    find_free_port
)
from reward_modeling.models.reward_model import RewardModel
from reward_modeling.data.dataset import PreferenceDataset, PreferencePair, create_synthetic_preference_data


class TestTrainingArguments(unittest.TestCase):
    """Test cases for TrainingArguments."""
    
    def test_default_arguments(self):
        """Test default training arguments initialization."""
        args = TrainingArguments(output_dir="./test_output")
        
        self.assertEqual(args.output_dir, "./test_output")
        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.per_device_train_batch_size, 16)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertFalse(args.fp16)
        self.assertFalse(args.use_wandb)
    
    def test_custom_arguments(self):
        """Test custom training arguments."""
        args = TrainingArguments(
            output_dir="./custom_output",
            num_train_epochs=5,
            per_device_train_batch_size=32,
            learning_rate=1e-4,
            weight_decay=0.05,
            fp16=True,
            use_wandb=True,
            project_name="test_project"
        )
        
        self.assertEqual(args.output_dir, "./custom_output")
        self.assertEqual(args.num_train_epochs, 5)
        self.assertEqual(args.per_device_train_batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertTrue(args.fp16)
        self.assertTrue(args.use_wandb)
        self.assertEqual(args.project_name, "test_project")
    
    def test_training_arguments_to_dict(self):
        """Test converting training arguments to dictionary."""
        args = TrainingArguments(
            output_dir="./test",
            num_train_epochs=2,
            learning_rate=1e-3
        )
        
        args_dict = args.to_dict()
        
        self.assertIsInstance(args_dict, dict)
        self.assertEqual(args_dict["output_dir"], "./test")
        self.assertEqual(args_dict["num_train_epochs"], 2)
        self.assertEqual(args_dict["learning_rate"], 1e-3)


class TestTrainingState(unittest.TestCase):
    """Test cases for TrainingState."""
    
    def test_training_state_initialization(self):
        """Test training state initialization."""
        state = TrainingState()
        
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.global_step, 0)
        self.assertEqual(state.max_steps, 0)
        self.assertEqual(state.num_train_epochs, 0)
        self.assertIsNone(state.best_metric)
    
    def test_training_state_updates(self):
        """Test updating training state."""
        state = TrainingState()
        
        state.epoch = 1
        state.global_step = 100
        state.best_metric = 0.85
        
        self.assertEqual(state.epoch, 1)
        self.assertEqual(state.global_step, 100)
        self.assertEqual(state.best_metric, 0.85)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        # Simple mock forward pass
        batch_size = input_ids.shape[0]
        rewards = torch.randn(batch_size)
        
        if return_dict:
            return {"rewards": rewards}
        return rewards


class TestBaseTrainer(unittest.TestCase):
    """Test cases for BaseTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-4
        )
        
        # Create a mock dataset
        self.train_data = [
            {"chosen_input_ids": torch.tensor([1, 2, 3]), "rejected_input_ids": torch.tensor([4, 5, 6])},
            {"chosen_input_ids": torch.tensor([7, 8, 9]), "rejected_input_ids": torch.tensor([10, 11, 12])}
        ]
    
    def test_base_trainer_initialization(self):
        """Test base trainer initialization."""
        trainer = BaseTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_data
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.args)
        self.assertEqual(len(trainer.train_dataset), 2)
        self.assertIsNotNone(trainer.state)
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        trainer = BaseTrainer(
            model=self.model,
            args=self.args
        )
        
        optimizer = trainer.create_optimizer()
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)  # Regular params and bias params
    
    def test_create_scheduler(self):
        """Test learning rate scheduler creation."""
        trainer = BaseTrainer(
            model=self.model,
            args=self.args
        )
        
        optimizer = trainer.create_optimizer()
        scheduler = trainer.create_scheduler(optimizer, num_training_steps=100)
        
        self.assertIsNotNone(scheduler)
    
    @patch('reward_modeling.training.trainer.wandb')
    def test_setup_wandb(self, mock_wandb):
        """Test wandb setup."""
        args = TrainingArguments(
            output_dir="./test",
            use_wandb=True,
            project_name="test_project"
        )
        
        trainer = BaseTrainer(model=self.model, args=args)
        trainer.setup_wandb()
        
        mock_wandb.init.assert_called_once()


class TestRewardModelTrainer(unittest.TestCase):
    """Test cases for RewardModelTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model
        self.model = MockModel()
        
        # Create training arguments
        self.args = TrainingArguments(
            output_dir=self.temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            use_wandb=False
        )
        
        # Create synthetic dataset
        self.train_dataset = create_synthetic_preference_data(n_samples=10)
        self.eval_dataset = create_synthetic_preference_data(n_samples=5)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_reward_model_trainer_initialization(self):
        """Test reward model trainer initialization."""
        trainer = RewardModelTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.lr_scheduler)
        self.assertEqual(trainer.global_step, 0)
    
    def test_compute_loss(self):
        """Test loss computation."""
        trainer = RewardModelTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset
        )
        
        # Create mock batch
        batch = {
            "chosen_texts": ["Good response"],
            "rejected_texts": ["Bad response"]
        }
        
        loss_dict = trainer.compute_loss(batch)
        
        self.assertIn("loss", loss_dict)
        self.assertIn("accuracy", loss_dict)
        self.assertIn("reward_gap", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
    
    @patch('reward_modeling.training.trainer.wandb')
    def test_training_loop_short(self, mock_wandb):
        """Test a short training loop."""
        # Use smaller dataset for faster testing
        small_train_dataset = create_synthetic_preference_data(n_samples=4)
        
        trainer = RewardModelTrainer(
            model=self.model,
            args=self.args,
            train_dataset=small_train_dataset
        )
        
        # Mock the train method to avoid full training
        with patch.object(trainer, 'train_epoch') as mock_train_epoch:
            mock_train_epoch.return_value = {
                "loss": 0.5,
                "accuracy": 0.7,
                "reward_gap": 0.3
            }
            
            with patch.object(trainer, 'evaluate') as mock_evaluate:
                mock_evaluate.return_value = {
                    "loss": 0.4,
                    "accuracy": 0.8
                }
                
                results = trainer.train()
                
                self.assertIn("training_history", results)
                self.assertIn("final_model_path", results)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        trainer = RewardModelTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset
        )
        
        # Save checkpoint
        checkpoint_name = "test_checkpoint"
        trainer.save_checkpoint(checkpoint_name)
        
        checkpoint_path = os.path.join(self.temp_dir, checkpoint_name)
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Check that training state was saved
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        self.assertTrue(os.path.exists(training_state_path))


class TestDPOTrainer(unittest.TestCase):
    """Test cases for DPOTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.reference_model = MockModel()
        
        self.args = TrainingArguments(
            output_dir="./test_dpo",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
    
    def test_dpo_trainer_initialization(self):
        """Test DPO trainer initialization."""
        trainer = DPOTrainer(
            model=self.model,
            reference_model=self.reference_model,
            beta=0.1,
            args=self.args
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.reference_model)
        self.assertEqual(trainer.beta, 0.1)
        
        # Reference model should be frozen
        for param in trainer.reference_model.parameters():
            self.assertFalse(param.requires_grad)
    
    def test_dpo_compute_loss(self):
        """Test DPO loss computation."""
        trainer = DPOTrainer(
            model=self.model,
            reference_model=self.reference_model,
            beta=0.1,
            args=self.args
        )
        
        # Mock inputs
        inputs = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10)
        }
        
        loss = trainer.compute_loss(trainer.model, inputs)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))


class TestConstitutionalTrainer(unittest.TestCase):
    """Test cases for ConstitutionalTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.constitution = [
            "Be helpful and harmless",
            "Be honest and transparent",
            "Respect human values"
        ]
        
        self.args = TrainingArguments(
            output_dir="./test_constitutional",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
    
    def test_constitutional_trainer_initialization(self):
        """Test constitutional trainer initialization."""
        trainer = ConstitutionalTrainer(
            model=self.model,
            constitution=self.constitution,
            args=self.args
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertEqual(len(trainer.constitution), 3)
    
    def test_constitutional_compute_loss(self):
        """Test constitutional loss computation."""
        trainer = ConstitutionalTrainer(
            model=self.model,
            constitution=self.constitution,
            args=self.args
        )
        
        # Mock inputs
        inputs = {
            "initial_input_ids": torch.randint(0, 100, (2, 10)),
            "initial_attention_mask": torch.ones(2, 10),
            "revised_input_ids": torch.randint(0, 100, (2, 10)),
            "revised_attention_mask": torch.ones(2, 10)
        }
        
        loss = trainer.compute_loss(trainer.model, inputs)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))


class TestMultiObjectiveTrainer(unittest.TestCase):
    """Test cases for MultiObjectiveTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model that returns multi-objective rewards
        class MultiObjectiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 3)  # 3 objectives
            
            def forward(self, input_ids, attention_mask=None, return_dict=True):
                batch_size = input_ids.shape[0]
                rewards = torch.randn(batch_size, 3)  # Multi-objective
                
                if return_dict:
                    return {"rewards": rewards}
                return rewards
        
        self.model = MultiObjectiveModel()
        self.objective_weights = [0.5, 0.3, 0.2]
        
        self.args = TrainingArguments(
            output_dir="./test_multi_objective",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
    
    def test_multi_objective_trainer_initialization(self):
        """Test multi-objective trainer initialization."""
        trainer = MultiObjectiveTrainer(
            model=self.model,
            objective_weights=self.objective_weights,
            args=self.args
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertEqual(len(trainer.objective_weights), 3)
    
    def test_multi_objective_compute_loss(self):
        """Test multi-objective loss computation."""
        trainer = MultiObjectiveTrainer(
            model=self.model,
            objective_weights=self.objective_weights,
            args=self.args
        )
        
        # Mock inputs
        inputs = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10)
        }
        
        loss = trainer.compute_loss(trainer.model, inputs)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))


class TestDPOTrainingArguments(unittest.TestCase):
    """Test cases for DPOTrainingArguments."""
    
    def test_dpo_arguments_initialization(self):
        """Test DPO training arguments initialization."""
        args = DPOTrainingArguments(
            output_dir="./test_dpo",
            beta=0.2,
            reference_model_path="./reference_model",
            max_length=1024
        )
        
        self.assertEqual(args.output_dir, "./test_dpo")
        self.assertEqual(args.beta, 0.2)
        self.assertEqual(args.reference_model_path, "./reference_model")
        self.assertEqual(args.max_length, 1024)


class TestDirectPreferenceOptimizationTrainer(unittest.TestCase):
    """Test cases for DirectPreferenceOptimizationTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.model = MockModel()
        
        self.args = DPOTrainingArguments(
            output_dir=self.temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            beta=0.1,
            max_length=128
        )
        
        self.train_dataset = create_synthetic_preference_data(n_samples=5)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_dpo_trainer_initialization(self):
        """Test DPO trainer initialization."""
        with patch('transformers.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            trainer = DirectPreferenceOptimizationTrainer(
                model=self.model,
                args=self.args,
                train_dataset=self.train_dataset,
                tokenizer=mock_tokenizer
            )
            
            self.assertIsNotNone(trainer.model)
            self.assertEqual(trainer.args.beta, 0.1)


class TestTrainerFactory(unittest.TestCase):
    """Test cases for trainer factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.args = TrainingArguments(output_dir="./test")
    
    def test_create_reward_trainer(self):
        """Test creating reward trainer."""
        trainer = create_trainer(
            trainer_type="reward",
            model=self.model,
            args=self.args
        )
        
        self.assertIsInstance(trainer, RewardModelTrainer)
    
    def test_create_dpo_trainer(self):
        """Test creating DPO trainer."""
        trainer = create_trainer(
            trainer_type="dpo",
            model=self.model,
            args=self.args,
            beta=0.1
        )
        
        self.assertIsInstance(trainer, DPOTrainer)
    
    def test_create_constitutional_trainer(self):
        """Test creating constitutional trainer."""
        constitution = ["Be helpful", "Be harmless"]
        
        trainer = create_trainer(
            trainer_type="constitutional",
            model=self.model,
            args=self.args,
            constitution=constitution
        )
        
        self.assertIsInstance(trainer, ConstitutionalTrainer)
    
    def test_create_multi_objective_trainer(self):
        """Test creating multi-objective trainer."""
        trainer = create_trainer(
            trainer_type="multi_objective",
            model=self.model,
            args=self.args,
            objective_weights=[0.6, 0.4]
        )
        
        self.assertIsInstance(trainer, MultiObjectiveTrainer)
    
    def test_unknown_trainer_type(self):
        """Test creating trainer with unknown type."""
        with self.assertRaises(ValueError):
            create_trainer(
                trainer_type="unknown",
                model=self.model,
                args=self.args
            )


class TestTrainingUtilities(unittest.TestCase):
    """Test cases for training utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            global_step=100,
            loss=0.5
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["global_step"], 100)
        self.assertEqual(checkpoint["loss"], 0.5)
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        
        # Save checkpoint first
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=2,
            global_step=200
        )
        
        # Create new model and optimizer
        new_model = MockModel()
        new_optimizer = torch.optim.AdamW(new_model.parameters())
        
        # Load checkpoint
        epoch, global_step = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        self.assertEqual(epoch, 2)
        self.assertEqual(global_step, 200)
    
    def test_compute_reward_metrics(self):
        """Test computing reward metrics."""
        # Create mock dataloader
        dataset = [
            {
                "chosen_input_ids": torch.randint(0, 100, (1, 10)),
                "chosen_attention_mask": torch.ones(1, 10),
                "rejected_input_ids": torch.randint(0, 100, (1, 10)),
                "rejected_attention_mask": torch.ones(1, 10)
            }
            for _ in range(5)
        ]
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        metrics = compute_reward_metrics(self.model, dataloader)
        
        self.assertIn("accuracy", metrics)
        self.assertTrue(0 <= metrics["accuracy"] <= 1)


class TestDistributedTraining(unittest.TestCase):
    """Test cases for distributed training utilities."""
    
    def test_distributed_config(self):
        """Test distributed configuration."""
        config = DistributedConfig(
            world_size=2,
            rank=0,
            backend="gloo",
            use_deepspeed=False
        )
        
        self.assertEqual(config.world_size, 2)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.backend, "gloo")
        self.assertFalse(config.use_deepspeed)
        
        # Test serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # Test deserialization
        new_config = DistributedConfig.from_dict(config_dict)
        self.assertEqual(new_config.world_size, 2)
        self.assertEqual(new_config.rank, 0)
    
    def test_distributed_manager_initialization(self):
        """Test distributed manager initialization."""
        config = DistributedConfig(world_size=1, rank=0)  # Single process
        manager = DistributedManager(config)
        
        self.assertEqual(manager.config.world_size, 1)
        self.assertEqual(manager.config.rank, 0)
        self.assertFalse(manager.is_initialized)
    
    def test_find_free_port(self):
        """Test finding free port."""
        port = find_free_port()
        
        self.assertIsInstance(port, str)
        self.assertTrue(port.isdigit())
        port_num = int(port)
        self.assertTrue(1024 <= port_num <= 65535)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('reward_modeling.training.trainer.wandb')
    def test_end_to_end_training_pipeline(self, mock_wandb):
        """Test complete training pipeline."""
        # Create model and dataset
        model = MockModel()
        train_dataset = create_synthetic_preference_data(n_samples=8)
        eval_dataset = create_synthetic_preference_data(n_samples=4)
        
        # Training arguments
        args = TrainingArguments(
            output_dir=self.temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            logging_steps=1,
            eval_steps=2,
            save_steps=10,
            use_wandb=False
        )
        
        # Create trainer
        trainer = RewardModelTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Mock training methods to speed up test
        with patch.object(trainer, 'train_epoch') as mock_train_epoch:
            mock_train_epoch.return_value = {
                "loss": 0.6,
                "accuracy": 0.75,
                "reward_gap": 0.4
            }
            
            with patch.object(trainer, 'evaluate') as mock_evaluate:
                mock_evaluate.return_value = {
                    "loss": 0.5,
                    "accuracy": 0.8
                }
                
                # Run training
                results = trainer.train()
                
                # Verify results
                self.assertIn("training_history", results)
                self.assertIn("final_model_path", results)
                
                # Check that final model directory exists
                final_model_path = results["final_model_path"]
                self.assertTrue(os.path.exists(final_model_path))


if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2) 