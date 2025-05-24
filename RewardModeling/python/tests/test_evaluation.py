"""
Comprehensive tests for evaluation module.
Tests evaluation metrics, model comparison, and validation utilities.
"""

import unittest
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from typing import Dict, Any, List

from reward_modeling.evaluation.metrics import (
    EvaluationResults,
    RewardModelEvaluator,
    ModelComparison,
    compute_benchmark_metrics,
    _compute_ndcg,
    _compute_map,
    _compute_mrr,
    _compute_precision_at_k
)
from reward_modeling.evaluation.validation import (
    ValidationConfig,
    ValidationResults,
    BaseValidator,
    CrossValidator,
    HoldOutValidator,
    TemporalValidator,
    BootstrapValidator,
    ValidationSuite,
    create_validator,
    run_quick_validation
)
from reward_modeling.data.dataset import PreferenceDataset, PreferencePair, create_synthetic_preference_data
from reward_modeling.models.reward_model import RewardModel


class MockRewardModel(nn.Module):
    """Mock reward model for testing."""
    
    def __init__(self, deterministic=False):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.deterministic = deterministic
    
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        batch_size = input_ids.shape[0]
        
        if self.deterministic:
            # Deterministic rewards for consistent testing
            rewards = torch.sum(input_ids, dim=1).float() * 0.01
        else:
            rewards = torch.randn(batch_size)
        
        if return_dict:
            return {"rewards": rewards}
        return rewards
    
    def to(self, device):
        return self
    
    def eval(self):
        return self


class TestEvaluationResults(unittest.TestCase):
    """Test cases for EvaluationResults."""
    
    def test_evaluation_results_creation(self):
        """Test creating evaluation results."""
        results = EvaluationResults(
            accuracy=0.85,
            preference_correlation=0.72,
            reward_gap=0.45,
            kendall_tau=0.68,
            spearman_rho=0.78,
            calibration_error=0.12,
            consistency_score=0.91,
            robustness_score=0.76,
            uncertainty_quality=0.83,
            detailed_metrics={"extra_metric": 0.95}
        )
        
        self.assertEqual(results.accuracy, 0.85)
        self.assertEqual(results.preference_correlation, 0.72)
        self.assertEqual(results.reward_gap, 0.45)
        self.assertEqual(results.kendall_tau, 0.68)
        self.assertEqual(results.spearman_rho, 0.78)
        self.assertEqual(results.calibration_error, 0.12)
        self.assertEqual(results.consistency_score, 0.91)
        self.assertEqual(results.robustness_score, 0.76)
        self.assertEqual(results.uncertainty_quality, 0.83)
        self.assertEqual(results.detailed_metrics["extra_metric"], 0.95)
    
    def test_evaluation_results_to_dict(self):
        """Test converting evaluation results to dictionary."""
        results = EvaluationResults(
            accuracy=0.80,
            preference_correlation=0.65,
            reward_gap=0.35,
            kendall_tau=0.70,
            spearman_rho=0.75,
            calibration_error=0.15,
            consistency_score=0.88,
            robustness_score=0.72,
            uncertainty_quality=0.80
        )
        
        results_dict = results.to_dict()
        
        self.assertIsInstance(results_dict, dict)
        self.assertEqual(results_dict["accuracy"], 0.80)
        self.assertEqual(results_dict["kendall_tau"], 0.70)
        self.assertIn("detailed_metrics", results_dict)
    
    def test_evaluation_results_save_and_load(self):
        """Test saving and loading evaluation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = EvaluationResults(
                accuracy=0.90,
                preference_correlation=0.85,
                reward_gap=0.50,
                kendall_tau=0.80,
                spearman_rho=0.85,
                calibration_error=0.10,
                consistency_score=0.95,
                robustness_score=0.85,
                uncertainty_quality=0.90
            )
            
            save_path = Path(temp_dir) / "test_results.json"
            results.save(save_path)
            
            self.assertTrue(save_path.exists())
            
            # Verify saved content
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data["accuracy"], 0.90)
            self.assertEqual(saved_data["spearman_rho"], 0.85)


class TestRewardModelEvaluator(unittest.TestCase):
    """Test cases for RewardModelEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockRewardModel(deterministic=True)
        self.dataset = create_synthetic_preference_data(n_samples=20)
        
        self.evaluator = RewardModelEvaluator(
            model=self.model,
            device="cpu",
            batch_size=4,
            compute_uncertainty=True
        )
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.device, "cpu")
        self.assertEqual(self.evaluator.batch_size, 4)
        self.assertTrue(self.evaluator.compute_uncertainty)
        self.assertIsNotNone(self.evaluator.model)
    
    def test_get_predictions(self):
        """Test getting model predictions."""
        predictions = self.evaluator._get_predictions(self.dataset)
        
        # Check all required keys are present
        expected_keys = [
            "chosen_rewards", "rejected_rewards", "reward_diff", 
            "preference_probs", "labels"
        ]
        for key in expected_keys:
            self.assertIn(key, predictions)
        
        # Check shapes
        n_samples = len(self.dataset)
        self.assertEqual(len(predictions["chosen_rewards"]), n_samples)
        self.assertEqual(len(predictions["rejected_rewards"]), n_samples)
        self.assertEqual(len(predictions["reward_diff"]), n_samples)
        
        # Check value ranges
        self.assertTrue(np.all(predictions["preference_probs"] >= 0))
        self.assertTrue(np.all(predictions["preference_probs"] <= 1))
        self.assertTrue(np.all(predictions["labels"] == 1))  # All chosen should be preferred
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        predictions = self.evaluator._get_predictions(self.dataset)
        results = self.evaluator._compute_metrics(predictions, self.dataset)
        
        self.assertIsInstance(results, EvaluationResults)
        
        # Check that all metrics are within reasonable ranges
        self.assertTrue(0 <= results.accuracy <= 1)
        self.assertTrue(0 <= results.calibration_error <= 1)
        self.assertTrue(0 <= results.consistency_score <= 1)
        self.assertTrue(np.isfinite(results.reward_gap))
    
    def test_compute_calibration_error(self):
        """Test calibration error computation."""
        # Create mock predictions with known probabilities
        predictions = {
            "preference_probs": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            "labels": np.array([0, 0, 1, 1, 1])  # Simulate some correct/incorrect predictions
        }
        
        calibration_error = self.evaluator._compute_calibration_error(predictions)
        
        self.assertIsInstance(calibration_error, float)
        self.assertTrue(0 <= calibration_error <= 1)
    
    def test_compute_consistency_score(self):
        """Test consistency score computation."""
        predictions = {
            "chosen_rewards": np.array([1.0, 1.5, 2.0, 1.2, 1.8]),
            "rejected_rewards": np.array([0.5, 0.8, 1.0, 0.6, 0.9])
        }
        
        consistency_score = self.evaluator._compute_consistency_score(predictions, self.dataset)
        
        self.assertIsInstance(consistency_score, float)
        self.assertTrue(0 <= consistency_score <= 1)
    
    def test_compute_uncertainty_quality(self):
        """Test uncertainty quality computation."""
        predictions = {
            "chosen_rewards": np.array([1.0, 0.5, 2.0, 1.5]),
            "rejected_rewards": np.array([0.8, 0.7, 1.5, 1.2]),
            "chosen_uncertainties": np.array([0.2, 0.8, 0.1, 0.3]),
            "rejected_uncertainties": np.array([0.3, 0.6, 0.2, 0.4])
        }
        
        uncertainty_quality = self.evaluator._compute_uncertainty_quality(predictions)
        
        self.assertIsInstance(uncertainty_quality, float)
        self.assertTrue(0 <= uncertainty_quality <= 1)
    
    def test_full_evaluation(self):
        """Test full evaluation pipeline."""
        results = self.evaluator.evaluate(self.dataset)
        
        self.assertIsInstance(results, EvaluationResults)
        
        # Check that detailed metrics are computed
        self.assertIn("reward_statistics", results.detailed_metrics)
        self.assertIn("confidence_analysis", results.detailed_metrics)
        self.assertIn("error_analysis", results.detailed_metrics)
        
        # Check reward statistics
        reward_stats = results.detailed_metrics["reward_statistics"]
        self.assertIn("chosen_mean", reward_stats)
        self.assertIn("rejected_mean", reward_stats)
        self.assertIn("chosen_std", reward_stats)
        self.assertIn("rejected_std", reward_stats)
    
    def test_evaluation_with_save(self):
        """Test evaluation with saving results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "eval_results.json"
            
            results = self.evaluator.evaluate(
                self.dataset,
                save_results=str(save_path)
            )
            
            self.assertTrue(save_path.exists())
            
            # Verify saved results can be loaded
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("accuracy", saved_data)
            self.assertIn("detailed_metrics", saved_data)


class TestModelComparison(unittest.TestCase):
    """Test cases for ModelComparison."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = {
            "model_a": MockRewardModel(deterministic=True),
            "model_b": MockRewardModel(deterministic=False)
        }
        
        self.dataset = create_synthetic_preference_data(n_samples=15)
        
        self.comparison = ModelComparison(self.models)
    
    def test_model_comparison_initialization(self):
        """Test model comparison initialization."""
        self.assertEqual(len(self.comparison.models), 2)
        self.assertEqual(len(self.comparison.evaluators), 2)
        self.assertIn("model_a", self.comparison.models)
        self.assertIn("model_b", self.comparison.models)
    
    def test_compare_on_dataset(self):
        """Test comparing models on dataset."""
        results = self.comparison.compare_on_dataset(self.dataset)
        
        self.assertEqual(len(results), 2)
        self.assertIn("model_a", results)
        self.assertIn("model_b", results)
        
        # Check that each result is an EvaluationResults object
        for model_name, result in results.items():
            self.assertIsInstance(result, EvaluationResults)
    
    def test_create_comparison_summary(self):
        """Test creating comparison summary."""
        # Mock evaluation results
        mock_results = {
            "model_a": EvaluationResults(
                accuracy=0.85, reward_gap=0.5, spearman_rho=0.8,
                kendall_tau=0.7, calibration_error=0.1,
                preference_correlation=0.75, consistency_score=0.9,
                robustness_score=0.8, uncertainty_quality=0.85
            ),
            "model_b": EvaluationResults(
                accuracy=0.80, reward_gap=0.4, spearman_rho=0.75,
                kendall_tau=0.65, calibration_error=0.15,
                preference_correlation=0.70, consistency_score=0.85,
                robustness_score=0.75, uncertainty_quality=0.80
            )
        }
        
        summary = self.comparison._create_comparison_summary(mock_results)
        
        self.assertIn("model_rankings", summary)
        self.assertIn("metric_summary", summary)
        self.assertIn("best_model", summary)
        
        # Check that rankings are computed
        self.assertIn("accuracy", summary["model_rankings"])
        self.assertEqual(len(summary["model_rankings"]["accuracy"]), 2)
        
        # Check that best model is identified
        self.assertIn("name", summary["best_model"])
        self.assertIn("average_rank", summary["best_model"])
    
    def test_save_comparison(self):
        """Test saving comparison results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock results
            mock_results = {
                "model_a": EvaluationResults(
                    accuracy=0.85, reward_gap=0.5, spearman_rho=0.8,
                    kendall_tau=0.7, calibration_error=0.1,
                    preference_correlation=0.75, consistency_score=0.9,
                    robustness_score=0.8, uncertainty_quality=0.85
                )
            }
            
            mock_summary = {
                "best_model": {"name": "model_a", "average_rank": 0.5},
                "model_rankings": {"accuracy": ["model_a"]},
                "metric_summary": {"accuracy": {"best": "model_a"}}
            }
            
            save_path = Path(temp_dir) / "comparison_results.json"
            
            self.comparison._save_comparison(mock_results, mock_summary, str(save_path))
            
            self.assertTrue(save_path.exists())
            
            # Verify saved content
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("summary", saved_data)
            self.assertIn("individual_results", saved_data)
            self.assertIn("metadata", saved_data)


class TestBenchmarkMetrics(unittest.TestCase):
    """Test cases for benchmark metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample ranking data
        self.true_rankings = [
            [0, 1, 2],  # First query: item 0 is most relevant
            [1, 0, 2],  # Second query: item 1 is most relevant
            [0, 2, 1]   # Third query: item 0 is most relevant
        ]
        
        self.predicted_scores = [
            [0.9, 0.7, 0.3],  # Predictions for first query
            [0.6, 0.8, 0.2],  # Predictions for second query  
            [0.95, 0.4, 0.85] # Predictions for third query
        ]
    
    def test_compute_ndcg(self):
        """Test NDCG computation."""
        ndcg = _compute_ndcg(self.true_rankings, self.predicted_scores, k=3)
        
        self.assertIsInstance(ndcg, float)
        self.assertTrue(0 <= ndcg <= 1)
    
    def test_compute_map(self):
        """Test MAP computation."""
        map_score = _compute_map(self.true_rankings, self.predicted_scores)
        
        self.assertIsInstance(map_score, float)
        self.assertTrue(0 <= map_score <= 1)
    
    def test_compute_mrr(self):
        """Test MRR computation."""
        mrr = _compute_mrr(self.true_rankings, self.predicted_scores)
        
        self.assertIsInstance(mrr, float)
        self.assertTrue(0 <= mrr <= 1)
    
    def test_compute_precision_at_k(self):
        """Test Precision@K computation."""
        precision_at_1 = _compute_precision_at_k(self.true_rankings, self.predicted_scores, k=1)
        precision_at_2 = _compute_precision_at_k(self.true_rankings, self.predicted_scores, k=2)
        
        self.assertIsInstance(precision_at_1, float)
        self.assertIsInstance(precision_at_2, float)
        self.assertTrue(0 <= precision_at_1 <= 1)
        self.assertTrue(0 <= precision_at_2 <= 1)
    
    def test_compute_benchmark_metrics(self):
        """Test computing all benchmark metrics."""
        metrics = compute_benchmark_metrics(
            self.true_rankings,
            self.predicted_scores,
            metrics=["ndcg", "map", "mrr", "precision_at_k"]
        )
        
        self.assertIn("ndcg", metrics)
        self.assertIn("map", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("precision_at_1", metrics)
        self.assertIn("precision_at_3", metrics)
        
        # Check all values are in valid ranges
        for metric, value in metrics.items():
            self.assertTrue(0 <= value <= 1)


class TestValidationConfig(unittest.TestCase):
    """Test cases for ValidationConfig."""
    
    def test_validation_config_defaults(self):
        """Test validation config with defaults."""
        config = ValidationConfig()
        
        self.assertEqual(config.cv_folds, 5)
        self.assertEqual(config.cv_strategy, "kfold")
        self.assertTrue(config.cv_shuffle)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.bootstrap_samples, 1000)
        self.assertIsNotNone(config.metrics)
    
    def test_validation_config_custom(self):
        """Test validation config with custom values."""
        config = ValidationConfig(
            cv_folds=10,
            cv_strategy="stratified",
            validation_split=0.15,
            bootstrap_samples=500,
            metrics=["accuracy", "f1"]
        )
        
        self.assertEqual(config.cv_folds, 10)
        self.assertEqual(config.cv_strategy, "stratified")
        self.assertEqual(config.validation_split, 0.15)
        self.assertEqual(config.bootstrap_samples, 500)
        self.assertEqual(len(config.metrics), 2)
    
    def test_validation_config_serialization(self):
        """Test validation config serialization."""
        config = ValidationConfig(cv_folds=8, validation_split=0.25)
        
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        new_config = ValidationConfig.from_dict(config_dict)
        self.assertEqual(new_config.cv_folds, 8)
        self.assertEqual(new_config.validation_split, 0.25)


class TestValidationResults(unittest.TestCase):
    """Test cases for ValidationResults."""
    
    def test_validation_results_creation(self):
        """Test creating validation results."""
        from datetime import datetime
        
        # Mock fold results
        fold_results = [
            EvaluationResults(
                accuracy=0.85, reward_gap=0.5, spearman_rho=0.8,
                kendall_tau=0.7, calibration_error=0.1,
                preference_correlation=0.75, consistency_score=0.9,
                robustness_score=0.8, uncertainty_quality=0.85
            )
        ]
        
        config = ValidationConfig()
        
        results = ValidationResults(
            validation_type="cross_validation",
            fold_results=fold_results,
            aggregated_metrics={"accuracy": 0.85},
            aggregated_std={"accuracy": 0.05},
            confidence_intervals={"accuracy": (0.80, 0.90)},
            config=config,
            model_info={"model_class": "TestModel"},
            validation_time=120.5,
            timestamp=datetime.now()
        )
        
        self.assertEqual(results.validation_type, "cross_validation")
        self.assertEqual(len(results.fold_results), 1)
        self.assertEqual(results.aggregated_metrics["accuracy"], 0.85)
    
    def test_validation_results_save_and_load(self):
        """Test saving and loading validation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            from datetime import datetime
            
            # Create test results
            fold_results = [
                EvaluationResults(
                    accuracy=0.85, reward_gap=0.5, spearman_rho=0.8,
                    kendall_tau=0.7, calibration_error=0.1,
                    preference_correlation=0.75, consistency_score=0.9,
                    robustness_score=0.8, uncertainty_quality=0.85
                )
            ]
            
            config = ValidationConfig()
            
            results = ValidationResults(
                validation_type="holdout_validation",
                fold_results=fold_results,
                aggregated_metrics={"accuracy": 0.85},
                aggregated_std={"accuracy": 0.05},
                confidence_intervals={"accuracy": (0.80, 0.90)},
                config=config,
                model_info={"model_class": "TestModel"},
                validation_time=120.5,
                timestamp=datetime.now()
            )
            
            save_path = Path(temp_dir) / "validation_results.json"
            results.save(str(save_path))
            
            self.assertTrue(save_path.exists())


class TestValidators(unittest.TestCase):
    """Test cases for validation classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(cv_folds=3, bootstrap_samples=10)  # Small values for testing
        self.evaluator = RewardModelEvaluator(MockRewardModel(), device="cpu")
        self.dataset = create_synthetic_preference_data(n_samples=20)
    
    def test_create_validator(self):
        """Test validator factory function."""
        # Test cross validation
        cv_validator = create_validator("cross_validation", self.config, self.evaluator)
        self.assertIsInstance(cv_validator, CrossValidator)
        
        # Test holdout validation
        holdout_validator = create_validator("holdout_validation", self.config, self.evaluator)
        self.assertIsInstance(holdout_validator, HoldOutValidator)
        
        # Test temporal validation
        temporal_validator = create_validator("temporal_validation", self.config, self.evaluator)
        self.assertIsInstance(temporal_validator, TemporalValidator)
        
        # Test bootstrap validation
        bootstrap_validator = create_validator("bootstrap_validation", self.config, self.evaluator)
        self.assertIsInstance(bootstrap_validator, BootstrapValidator)
        
        # Test unknown validator type
        with self.assertRaises(ValueError):
            create_validator("unknown_validation", self.config, self.evaluator)
    
    def test_cross_validator_initialization(self):
        """Test cross validator initialization."""
        validator = CrossValidator(self.config, self.evaluator)
        
        self.assertEqual(validator.config.cv_folds, 3)
        self.assertIsNotNone(validator.evaluator)
    
    def test_holdout_validator_initialization(self):
        """Test holdout validator initialization."""
        validator = HoldOutValidator(self.config, self.evaluator)
        
        self.assertEqual(validator.config.validation_split, 0.2)
        self.assertIsNotNone(validator.evaluator)
    
    def test_temporal_validator_initialization(self):
        """Test temporal validator initialization."""
        validator = TemporalValidator(self.config, self.evaluator)
        
        self.assertEqual(validator.config.temporal_window_days, 30)
        self.assertIsNotNone(validator.evaluator)
    
    def test_bootstrap_validator_initialization(self):
        """Test bootstrap validator initialization."""
        validator = BootstrapValidator(self.config, self.evaluator)
        
        self.assertEqual(validator.config.bootstrap_samples, 10)
        self.assertIsNotNone(validator.evaluator)


class TestValidationSuite(unittest.TestCase):
    """Test cases for ValidationSuite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(cv_folds=2, bootstrap_samples=5)  # Small values for testing
        self.evaluator = RewardModelEvaluator(MockRewardModel(), device="cpu")
        self.suite = ValidationSuite(self.config, self.evaluator)
        self.dataset = create_synthetic_preference_data(n_samples=10)
    
    def test_validation_suite_initialization(self):
        """Test validation suite initialization."""
        self.assertIn("cross_validation", self.suite.validators)
        self.assertIn("holdout_validation", self.suite.validators)
        self.assertIn("temporal_validation", self.suite.validators)
        self.assertIn("bootstrap_validation", self.suite.validators)
    
    def test_run_validation_suite(self):
        """Test running validation suite."""
        from reward_modeling.training.trainer import TrainingArguments
        
        # Mock model class
        class MockModelClass:
            def __init__(self, **kwargs):
                pass
        
        # Mock training arguments
        training_args = TrainingArguments(output_dir="./test")
        
        # Mock the validate method to avoid actual training
        with patch.object(CrossValidator, 'validate') as mock_cv_validate:
            with patch.object(HoldOutValidator, 'validate') as mock_holdout_validate:
                # Mock return values
                mock_results = ValidationResults(
                    validation_type="test",
                    fold_results=[],
                    aggregated_metrics={"accuracy": 0.8},
                    aggregated_std={"accuracy": 0.1},
                    confidence_intervals={"accuracy": (0.7, 0.9)},
                    config=self.config,
                    model_info={},
                    validation_time=60.0,
                    timestamp=__import__('datetime').datetime.now()
                )
                
                mock_cv_validate.return_value = mock_results
                mock_holdout_validate.return_value = mock_results
                
                # Run validation suite
                results = self.suite.run_validation_suite(
                    MockModelClass,
                    self.dataset,
                    training_args,
                    validation_types=["cross_validation", "holdout_validation"]
                )
                
                self.assertIn("cross_validation", results)
                self.assertIn("holdout_validation", results)


class TestQuickValidation(unittest.TestCase):
    """Test cases for quick validation utility."""
    
    def test_run_quick_validation(self):
        """Test running quick validation."""
        from reward_modeling.training.trainer import TrainingArguments
        
        # Mock model class
        class MockModelClass:
            def __init__(self, **kwargs):
                pass
        
        dataset = create_synthetic_preference_data(n_samples=8)
        training_args = TrainingArguments(output_dir="./test")
        
        # Mock the validator to avoid actual training
        with patch('reward_modeling.evaluation.validation.HoldOutValidator') as MockValidator:
            mock_validator_instance = Mock()
            mock_results = ValidationResults(
                validation_type="holdout_validation",
                fold_results=[],
                aggregated_metrics={"accuracy": 0.85},
                aggregated_std={"accuracy": 0.05},
                confidence_intervals={"accuracy": (0.80, 0.90)},
                config=ValidationConfig(),
                model_info={},
                validation_time=30.0,
                timestamp=__import__('datetime').datetime.now()
            )
            mock_validator_instance.validate.return_value = mock_results
            MockValidator.return_value = mock_validator_instance
            
            results = run_quick_validation(
                MockModelClass,
                dataset,
                training_args,
                validation_split=0.3
            )
            
            self.assertIsInstance(results, ValidationResults)


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create model and dataset
        model = MockRewardModel(deterministic=True)
        dataset = create_synthetic_preference_data(n_samples=25)
        
        # Create evaluator
        evaluator = RewardModelEvaluator(
            model=model,
            device="cpu",
            batch_size=5,
            compute_uncertainty=True
        )
        
        # Run evaluation
        results = evaluator.evaluate(dataset)
        
        # Verify results
        self.assertIsInstance(results, EvaluationResults)
        self.assertTrue(0 <= results.accuracy <= 1)
        self.assertIn("reward_statistics", results.detailed_metrics)
        
        # Test model comparison
        models = {
            "model_1": MockRewardModel(deterministic=True),
            "model_2": MockRewardModel(deterministic=False)
        }
        
        comparison = ModelComparison(models)
        comparison_results = comparison.compare_on_dataset(dataset)
        
        self.assertEqual(len(comparison_results), 2)
        self.assertIn("model_1", comparison_results)
        self.assertIn("model_2", comparison_results)
    
    def test_evaluation_with_various_metrics(self):
        """Test evaluation with different metric configurations."""
        model = MockRewardModel(deterministic=True)
        dataset = create_synthetic_preference_data(n_samples=15)
        
        evaluator = RewardModelEvaluator(model, device="cpu", compute_uncertainty=False)
        
        # Test basic evaluation
        results = evaluator.evaluate(dataset)
        self.assertIsInstance(results, EvaluationResults)
        
        # Test evaluation with predictions saved
        results_with_preds = evaluator.evaluate(dataset, return_predictions=True)
        self.assertIn("predictions", results_with_preds.detailed_metrics)


if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2) 