"""
Comprehensive evaluation metrics and analysis tools for reward models.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd

from ..data.dataset import PreferenceDataset
from ..models.reward_model import RewardModel

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    accuracy: float
    preference_correlation: float
    reward_gap: float
    kendall_tau: float
    spearman_rho: float
    calibration_error: float
    consistency_score: float
    robustness_score: float
    uncertainty_quality: float
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "preference_correlation": self.preference_correlation,
            "reward_gap": self.reward_gap,
            "kendall_tau": self.kendall_tau,
            "spearman_rho": self.spearman_rho,
            "calibration_error": self.calibration_error,
            "consistency_score": self.consistency_score,
            "robustness_score": self.robustness_score,
            "uncertainty_quality": self.uncertainty_quality,
            "detailed_metrics": self.detailed_metrics
        }
    
    def save(self, path: Union[str, Path]):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class RewardModelEvaluator:
    """
    Comprehensive evaluator for reward models with advanced metrics.
    """
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer=None,
        device: str = None,
        batch_size: int = 32,
        compute_uncertainty: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.compute_uncertainty = compute_uncertainty
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        dataset: PreferenceDataset,
        return_predictions: bool = False,
        save_results: Optional[str] = None
    ) -> EvaluationResults:
        """
        Comprehensive evaluation of the reward model.
        
        Args:
            dataset: Evaluation dataset
            return_predictions: Whether to return individual predictions
            save_results: Path to save detailed results
            
        Returns:
            EvaluationResults object with all metrics
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Get model predictions
        predictions = self._get_predictions(dataset)
        
        # Compute all metrics
        results = self._compute_metrics(predictions, dataset)
        
        # Add detailed analysis
        if return_predictions:
            results.detailed_metrics["predictions"] = predictions
        
        # Compute additional advanced metrics
        results.detailed_metrics.update(self._compute_advanced_metrics(predictions))
        
        # Save results if path provided
        if save_results:
            results.save(save_results)
            logger.info(f"Evaluation results saved to {save_results}")
        
        logger.info(f"Evaluation completed. Accuracy: {results.accuracy:.3f}")
        return results
    
    def _get_predictions(self, dataset: PreferenceDataset) -> Dict[str, np.ndarray]:
        """Get model predictions for the entire dataset."""
        chosen_rewards = []
        rejected_rewards = []
        chosen_uncertainties = []
        rejected_uncertainties = []
        preference_probs = []
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions for chosen responses
                if "chosen_input_ids" in batch:
                    chosen_outputs = self.model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                        return_dict=True
                    )
                    chosen_batch_rewards = chosen_outputs["rewards"].cpu().numpy()
                    
                    # Get predictions for rejected responses
                    rejected_outputs = self.model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                        return_dict=True
                    )
                    rejected_batch_rewards = rejected_outputs["rewards"].cpu().numpy()
                else:
                    # Handle text inputs
                    chosen_batch_rewards = []
                    rejected_batch_rewards = []
                    
                    for chosen_text, rejected_text in zip(batch["chosen"], batch["rejected"]):
                        if self.tokenizer:
                            # Tokenize and predict
                            chosen_inputs = self.tokenizer(
                                chosen_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding="max_length"
                            ).to(self.device)
                            
                            rejected_inputs = self.tokenizer(
                                rejected_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding="max_length"
                            ).to(self.device)
                            
                            chosen_output = self.model(**chosen_inputs, return_dict=True)
                            rejected_output = self.model(**rejected_inputs, return_dict=True)
                            
                            chosen_batch_rewards.append(chosen_output["rewards"].item())
                            rejected_batch_rewards.append(rejected_output["rewards"].item())
                    
                    chosen_batch_rewards = np.array(chosen_batch_rewards)
                    rejected_batch_rewards = np.array(rejected_batch_rewards)
                
                chosen_rewards.extend(chosen_batch_rewards)
                rejected_rewards.extend(rejected_batch_rewards)
                
                # Compute preference probabilities
                reward_diff = chosen_batch_rewards - rejected_batch_rewards
                probs = expit(reward_diff)  # Sigmoid function
                preference_probs.extend(probs)
                
                # Compute uncertainties if enabled
                if self.compute_uncertainty:
                    # Simple uncertainty estimation using reward magnitude
                    chosen_unc = np.abs(chosen_batch_rewards)
                    rejected_unc = np.abs(rejected_batch_rewards)
                    chosen_uncertainties.extend(chosen_unc)
                    rejected_uncertainties.extend(rejected_unc)
        
        predictions = {
            "chosen_rewards": np.array(chosen_rewards),
            "rejected_rewards": np.array(rejected_rewards),
            "reward_diff": np.array(chosen_rewards) - np.array(rejected_rewards),
            "preference_probs": np.array(preference_probs),
            "labels": np.ones(len(chosen_rewards)),  # Chosen is always preferred
        }
        
        if self.compute_uncertainty:
            predictions.update({
                "chosen_uncertainties": np.array(chosen_uncertainties),
                "rejected_uncertainties": np.array(rejected_uncertainties),
            })
        
        return predictions
    
    def _compute_metrics(self, predictions: Dict[str, np.ndarray], dataset: PreferenceDataset) -> EvaluationResults:
        """Compute comprehensive evaluation metrics."""
        
        # Basic preference accuracy
        correct_preferences = (predictions["reward_diff"] > 0).astype(int)
        accuracy = np.mean(correct_preferences)
        
        # Reward gap (average difference between chosen and rejected)
        reward_gap = np.mean(predictions["reward_diff"])
        
        # Correlation metrics
        # For correlation, we need some ground truth ranking
        # Here we use the implicit ranking from preferences
        chosen_rewards = predictions["chosen_rewards"]
        rejected_rewards = predictions["rejected_rewards"]
        
        # Create ranking data for correlation analysis
        all_rewards = np.concatenate([chosen_rewards, rejected_rewards])
        # True ranks: chosen responses should have higher ranks
        true_ranks = np.concatenate([
            np.ones(len(chosen_rewards)),  # Chosen = 1 (higher preference)
            np.zeros(len(rejected_rewards))  # Rejected = 0 (lower preference)
        ])
        
        # Spearman correlation
        spearman_rho, _ = stats.spearmanr(all_rewards, true_ranks)
        
        # Kendall tau
        kendall_tau, _ = stats.kendalltau(all_rewards, true_ranks)
        
        # Preference correlation (how well reward differences correlate with preferences)
        preference_correlation = np.corrcoef(predictions["reward_diff"], predictions["labels"])[0, 1]
        
        # Calibration error
        calibration_error = self._compute_calibration_error(predictions)
        
        # Consistency score (how consistent are predictions across similar inputs)
        consistency_score = self._compute_consistency_score(predictions, dataset)
        
        # Robustness score (placeholder - would need adversarial examples)
        robustness_score = 0.5  # Default value
        
        # Uncertainty quality (if uncertainties are computed)
        uncertainty_quality = self._compute_uncertainty_quality(predictions) if self.compute_uncertainty else 0.0
        
        return EvaluationResults(
            accuracy=accuracy,
            preference_correlation=preference_correlation if not np.isnan(preference_correlation) else 0.0,
            reward_gap=reward_gap,
            kendall_tau=kendall_tau if not np.isnan(kendall_tau) else 0.0,
            spearman_rho=spearman_rho if not np.isnan(spearman_rho) else 0.0,
            calibration_error=calibration_error,
            consistency_score=consistency_score,
            robustness_score=robustness_score,
            uncertainty_quality=uncertainty_quality
        )
    
    def _compute_calibration_error(self, predictions: Dict[str, np.ndarray]) -> float:
        """Compute calibration error of preference predictions."""
        probs = predictions["preference_probs"]
        labels = predictions["labels"]
        
        try:
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, probs, n_bins=10, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = labels[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        except:
            # Return 0 if calibration computation fails
            return 0.0
    
    def _compute_consistency_score(
        self, 
        predictions: Dict[str, np.ndarray], 
        dataset: PreferenceDataset
    ) -> float:
        """Compute consistency score based on prediction variance for similar inputs."""
        # This is a simplified consistency measure
        # In practice, you'd want to identify similar inputs and measure prediction variance
        
        # For now, compute the inverse of reward prediction variance as a proxy
        chosen_var = np.var(predictions["chosen_rewards"])
        rejected_var = np.var(predictions["rejected_rewards"])
        
        # Normalize to [0, 1] range
        max_var = max(chosen_var, rejected_var, 1e-8)
        consistency = 1.0 - min(chosen_var + rejected_var, max_var) / max_var
        
        return consistency
    
    def _compute_uncertainty_quality(self, predictions: Dict[str, np.ndarray]) -> float:
        """Compute quality of uncertainty estimates."""
        if "chosen_uncertainties" not in predictions:
            return 0.0
        
        # Measure how well uncertainties correlate with prediction errors
        chosen_rewards = predictions["chosen_rewards"]
        rejected_rewards = predictions["rejected_rewards"]
        chosen_uncertainties = predictions["chosen_uncertainties"]
        rejected_uncertainties = predictions["rejected_uncertainties"]
        
        # For preference tasks, errors occur when chosen_reward < rejected_reward
        errors = (chosen_rewards < rejected_rewards).astype(float)
        
        # Average uncertainty for each pair
        avg_uncertainties = (chosen_uncertainties + rejected_uncertainties) / 2
        
        # Correlation between uncertainty and errors
        if len(np.unique(errors)) > 1 and len(np.unique(avg_uncertainties)) > 1:
            uncertainty_correlation = np.abs(np.corrcoef(avg_uncertainties, errors)[0, 1])
            return uncertainty_correlation if not np.isnan(uncertainty_correlation) else 0.0
        else:
            return 0.0
    
    def _compute_advanced_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute additional advanced metrics."""
        metrics = {}
        
        # Reward distribution statistics
        metrics["reward_statistics"] = {
            "chosen_mean": float(np.mean(predictions["chosen_rewards"])),
            "chosen_std": float(np.std(predictions["chosen_rewards"])),
            "rejected_mean": float(np.mean(predictions["rejected_rewards"])),
            "rejected_std": float(np.std(predictions["rejected_rewards"])),
            "reward_gap_std": float(np.std(predictions["reward_diff"])),
        }
        
        # Prediction confidence analysis
        probs = predictions["preference_probs"]
        metrics["confidence_analysis"] = {
            "mean_confidence": float(np.mean(probs)),
            "confidence_std": float(np.std(probs)),
            "high_confidence_ratio": float(np.mean(probs > 0.8)),
            "low_confidence_ratio": float(np.mean(probs < 0.6)),
        }
        
        # Error analysis
        errors = (predictions["reward_diff"] < 0).astype(int)
        metrics["error_analysis"] = {
            "error_rate": float(np.mean(errors)),
            "large_error_rate": float(np.mean(predictions["reward_diff"] < -1.0)),
            "margin_analysis": {
                "close_calls": float(np.mean(np.abs(predictions["reward_diff"]) < 0.5)),
                "confident_correct": float(np.mean((predictions["reward_diff"] > 1.0) & (errors == 0))),
            }
        }
        
        return metrics
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        if not batch:
            return {}
        
        # Handle different types of inputs
        result = {}
        
        # Text fields
        for key in ["prompt", "chosen", "rejected"]:
            if key in batch[0]:
                result[key] = [item[key] for item in batch]
        
        # Tensor fields
        tensor_keys = ["chosen_input_ids", "chosen_attention_mask", 
                      "rejected_input_ids", "rejected_attention_mask"]
        
        for key in tensor_keys:
            if key in batch[0]:
                result[key] = torch.stack([item[key] for item in batch])
        
        return result
    
    def evaluate_comprehensive(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Extended evaluation method for compatibility."""
        results = self.evaluate(dataset)
        return results.to_dict()

class ModelComparison:
    """
    Tool for comparing multiple reward models.
    """
    
    def __init__(self, models: Dict[str, RewardModel], tokenizer=None):
        self.models = models
        self.tokenizer = tokenizer
        self.evaluators = {
            name: RewardModelEvaluator(model, tokenizer)
            for name, model in models.items()
        }
    
    def compare_on_dataset(
        self,
        dataset: PreferenceDataset,
        save_path: Optional[str] = None
    ) -> Dict[str, EvaluationResults]:
        """
        Compare all models on the given dataset.
        
        Args:
            dataset: Dataset for comparison
            save_path: Optional path to save comparison results
            
        Returns:
            Dictionary mapping model names to evaluation results
        """
        logger.info(f"Comparing {len(self.models)} models on dataset with {len(dataset)} samples")
        
        results = {}
        for name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating model: {name}")
            results[name] = evaluator.evaluate(dataset)
        
        # Create comparison summary
        summary = self._create_comparison_summary(results)
        
        if save_path:
            self._save_comparison(results, summary, save_path)
        
        return results
    
    def _create_comparison_summary(
        self,
        results: Dict[str, EvaluationResults]
    ) -> Dict[str, Any]:
        """Create a summary of model comparison."""
        metrics = ["accuracy", "reward_gap", "spearman_rho", "kendall_tau", "calibration_error"]
        
        summary = {
            "model_rankings": {},
            "metric_summary": {},
            "best_model": {},
        }
        
        # Rank models by each metric
        for metric in metrics:
            metric_values = {name: getattr(result, metric) for name, result in results.items()}
            
            # Sort by metric (higher is better for most metrics, except calibration_error)
            reverse = metric != "calibration_error"
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse)
            
            summary["model_rankings"][metric] = [name for name, _ in sorted_models]
            summary["metric_summary"][metric] = {
                "values": metric_values,
                "best": sorted_models[0][0],
                "worst": sorted_models[-1][0],
            }
        
        # Overall best model (simple average ranking)
        model_scores = defaultdict(float)
        for metric, ranking in summary["model_rankings"].items():
            for i, model_name in enumerate(ranking):
                model_scores[model_name] += i  # Lower rank is better
        
        best_model_name = min(model_scores.items(), key=lambda x: x[1])[0]
        summary["best_model"] = {
            "name": best_model_name,
            "average_rank": model_scores[best_model_name] / len(metrics)
        }
        
        return summary
    
    def _save_comparison(
        self,
        results: Dict[str, EvaluationResults],
        summary: Dict[str, Any],
        save_path: str
    ):
        """Save comparison results to file."""
        comparison_data = {
            "summary": summary,
            "individual_results": {name: result.to_dict() for name, result in results.items()},
            "metadata": {
                "num_models": len(self.models),
                "model_names": list(self.models.keys()),
            }
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Model comparison results saved to {save_path}")

def compute_benchmark_metrics(
    true_rankings: List[List[int]],
    predicted_scores: List[List[float]],
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute benchmark metrics for ranking tasks.
    
    Args:
        true_rankings: List of true rankings for each query
        predicted_scores: List of predicted scores for each query
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ["ndcg", "map", "mrr", "precision_at_k"]
    
    results = {}
    
    for metric in metrics:
        if metric == "ndcg":
            results["ndcg"] = _compute_ndcg(true_rankings, predicted_scores)
        elif metric == "map":
            results["map"] = _compute_map(true_rankings, predicted_scores)
        elif metric == "mrr":
            results["mrr"] = _compute_mrr(true_rankings, predicted_scores)
        elif metric == "precision_at_k":
            for k in [1, 3, 5, 10]:
                results[f"precision_at_{k}"] = _compute_precision_at_k(
                    true_rankings, predicted_scores, k
                )
    
    return results

def _compute_ndcg(true_rankings: List[List[int]], predicted_scores: List[List[float]], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    ndcg_scores = []
    
    for true_rank, pred_scores in zip(true_rankings, predicted_scores):
        # Sort by predicted scores
        sorted_indices = np.argsort(pred_scores)[::-1][:k]
        
        # Compute DCG
        dcg = 0.0
        for i, idx in enumerate(sorted_indices):
            relevance = len(true_rank) - true_rank[idx] if idx < len(true_rank) else 0
            dcg += relevance / np.log2(i + 2)
        
        # Compute IDCG (ideal DCG)
        ideal_relevances = sorted([len(true_rank) - r for r in true_rank], reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

def _compute_map(true_rankings: List[List[int]], predicted_scores: List[List[float]]) -> float:
    """Compute Mean Average Precision."""
    ap_scores = []
    
    for true_rank, pred_scores in zip(true_rankings, predicted_scores):
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        # Compute Average Precision
        relevant_count = 0
        precision_sum = 0.0
        
        for i, idx in enumerate(sorted_indices):
            if idx < len(true_rank) and true_rank[idx] == 0:  # Assuming 0 is the most relevant
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / max(relevant_count, 1)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def _compute_mrr(true_rankings: List[List[int]], predicted_scores: List[List[float]]) -> float:
    """Compute Mean Reciprocal Rank."""
    rr_scores = []
    
    for true_rank, pred_scores in zip(true_rankings, predicted_scores):
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        # Find rank of first relevant item
        for i, idx in enumerate(sorted_indices):
            if idx < len(true_rank) and true_rank[idx] == 0:  # First relevant item
                rr_scores.append(1.0 / (i + 1))
                break
        else:
            rr_scores.append(0.0)
    
    return np.mean(rr_scores)

def _compute_precision_at_k(
    true_rankings: List[List[int]], 
    predicted_scores: List[List[float]], 
    k: int
) -> float:
    """Compute Precision at K."""
    precision_scores = []
    
    for true_rank, pred_scores in zip(true_rankings, predicted_scores):
        sorted_indices = np.argsort(pred_scores)[::-1][:k]
        
        # Count relevant items in top k
        relevant_count = sum(
            1 for idx in sorted_indices 
            if idx < len(true_rank) and true_rank[idx] == 0
        )
        
        precision = relevant_count / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores) 