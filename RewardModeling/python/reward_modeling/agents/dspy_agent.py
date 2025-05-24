"""
DSPy agent integration with reward model guidance.
Implements intelligent agents that use reward models to optimize their behavior.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import time
from abc import ABC, abstractmethod

try:
    import dspy
    from dspy import Signature, InputField, OutputField, ChainOfThought, Retrieve
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Create dummy classes for when DSPy is not available
    class Signature:
        pass
    class InputField:
        def __init__(self, *args, **kwargs):
            pass
    class OutputField:
        def __init__(self, *args, **kwargs):
            pass
    class ChainOfThought:
        def __init__(self, *args, **kwargs):
            pass
    class Retrieve:
        def __init__(self, *args, **kwargs):
            pass

from ..models.reward_model import BaseRewardModel
from ..data.dataset import PreferenceDataset

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for reward-guided agents."""
    model_name: str
    reward_model_path: str
    max_iterations: int = 5
    reward_threshold: float = 0.5
    temperature: float = 0.7
    top_k: int = 3
    use_chain_of_thought: bool = True
    optimize_prompts: bool = True
    cache_responses: bool = True

class BaseRewardGuidedAgent(ABC):
    """Abstract base class for reward-guided agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.reward_model = None
        self.response_cache = {}
        self.performance_history = []
        
        if config.reward_model_path:
            self._load_reward_model(config.reward_model_path)
    
    def _load_reward_model(self, model_path: str):
        """Load reward model for guidance."""
        try:
            from ..models.reward_model import TransformerRewardModel
            self.reward_model = TransformerRewardModel.from_pretrained(model_path)
            self.reward_model.eval()
            logger.info(f"Loaded reward model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            self.reward_model = None
    
    def evaluate_response(self, prompt: str, response: str) -> float:
        """Evaluate response quality using reward model."""
        if self.reward_model is None:
            return 0.0
        
        try:
            # Create input for reward model
            input_text = f"{prompt} {response}"
            
            # Tokenize and get reward
            with torch.no_grad():
                # This would need proper tokenization in real implementation
                reward = np.random.random()  # Placeholder
                
            return reward
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return 0.0
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response to prompt."""
        pass
    
    @abstractmethod
    def optimize_response(self, prompt: str, initial_response: str) -> str:
        """Optimize response using reward guidance."""
        pass

class RewardGuidedDSPyAgent(BaseRewardGuidedAgent):
    """DSPy agent with reward model guidance."""
    
    def __init__(self, config: AgentConfig, task_signature: Optional[type] = None):
        super().__init__(config)
        
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for RewardGuidedDSPyAgent. Install with: pip install dspy-ai")
        
        self.task_signature = task_signature or self._create_default_signature()
        self.predictor = self._create_predictor()
        self.optimizer = None
        
        if config.optimize_prompts:
            self._setup_optimizer()
    
    def _create_default_signature(self):
        """Create default signature for general tasks."""
        class GeneralTask(Signature):
            """Answer questions thoughtfully and helpfully."""
            question = InputField(desc="The question or prompt to answer")
            answer = OutputField(desc="A helpful and accurate answer")
        
        return GeneralTask
    
    def _create_predictor(self):
        """Create DSPy predictor."""
        if self.config.use_chain_of_thought:
            return ChainOfThought(self.task_signature)
        else:
            return dspy.Predict(self.task_signature)
    
    def _setup_optimizer(self):
        """Setup DSPy optimizer for automatic prompt optimization."""
        try:
            # Setup optimizer with reward-based evaluation
            self.optimizer = dspy.BootstrapFewShot(
                metric=self._reward_based_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=16
            )
        except Exception as e:
            logger.warning(f"Could not setup optimizer: {e}")
            self.optimizer = None
    
    def _reward_based_metric(self, example, pred, trace=None):
        """Custom metric based on reward model evaluation."""
        if hasattr(example, 'question') and hasattr(pred, 'answer'):
            reward = self.evaluate_response(example.question, pred.answer)
            return reward > self.config.reward_threshold
        return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using DSPy predictor."""
        try:
            # Check cache first
            cache_key = f"{prompt}_{hash(str(kwargs))}"
            if self.config.cache_responses and cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Generate response
            if hasattr(self.predictor, '__call__'):
                result = self.predictor(question=prompt, **kwargs)
                response = result.answer if hasattr(result, 'answer') else str(result)
            else:
                response = "I apologize, but I cannot generate a response at this time."
            
            # Cache response
            if self.config.cache_responses:
                self.response_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def optimize_response(self, prompt: str, initial_response: str) -> str:
        """Optimize response using reward guidance and iterative refinement."""
        best_response = initial_response
        best_reward = self.evaluate_response(prompt, initial_response)
        
        for iteration in range(self.config.max_iterations):
            # Generate alternative responses
            candidates = self._generate_candidates(prompt, best_response, iteration)
            
            # Evaluate candidates
            for candidate in candidates:
                reward = self.evaluate_response(prompt, candidate)
                
                if reward > best_reward:
                    best_reward = reward
                    best_response = candidate
                    
                    # Stop if we reach threshold
                    if reward >= self.config.reward_threshold:
                        break
            
            # Log progress
            logger.debug(f"Iteration {iteration + 1}: Best reward = {best_reward:.4f}")
            
            # Early stopping if reward is good enough
            if best_reward >= self.config.reward_threshold:
                break
        
        # Record performance
        self.performance_history.append({
            "prompt": prompt,
            "initial_reward": self.evaluate_response(prompt, initial_response),
            "final_reward": best_reward,
            "iterations": iteration + 1,
            "timestamp": time.time()
        })
        
        return best_response
    
    def _generate_candidates(self, prompt: str, current_response: str, iteration: int) -> List[str]:
        """Generate candidate responses for optimization."""
        candidates = []
        
        # Strategy 1: Modify current response
        modification_prompts = [
            f"Improve this response: {current_response}",
            f"Make this response more helpful: {current_response}",
            f"Clarify and enhance this response: {current_response}"
        ]
        
        for mod_prompt in modification_prompts:
            try:
                candidate = self.generate_response(mod_prompt)
                if candidate and candidate != current_response:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Error generating candidate: {e}")
        
        # Strategy 2: Generate fresh responses with slight variations
        for i in range(2):
            try:
                varied_prompt = f"{prompt} (Please provide a high-quality response.)"
                candidate = self.generate_response(varied_prompt)
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Error generating varied candidate: {e}")
        
        return candidates[:self.config.top_k]
    
    def train_on_examples(self, examples: List[Dict[str, str]]):
        """Train/optimize the agent on examples."""
        if self.optimizer is None:
            logger.warning("No optimizer available for training")
            return
        
        try:
            # Convert examples to DSPy format
            dspy_examples = []
            for ex in examples:
                if "question" in ex and "answer" in ex:
                    dspy_examples.append(
                        dspy.Example(question=ex["question"], answer=ex["answer"])
                    )
            
            if dspy_examples:
                # Optimize predictor
                optimized_predictor = self.optimizer.compile(
                    self.predictor,
                    trainset=dspy_examples
                )
                self.predictor = optimized_predictor
                logger.info(f"Optimized agent on {len(dspy_examples)} examples")
        
        except Exception as e:
            logger.error(f"Error training agent: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        initial_rewards = [h["initial_reward"] for h in self.performance_history]
        final_rewards = [h["final_reward"] for h in self.performance_history]
        improvements = [f - i for f, i in zip(final_rewards, initial_rewards)]
        iterations = [h["iterations"] for h in self.performance_history]
        
        return {
            "avg_initial_reward": np.mean(initial_rewards),
            "avg_final_reward": np.mean(final_rewards),
            "avg_improvement": np.mean(improvements),
            "avg_iterations": np.mean(iterations),
            "success_rate": np.mean([r >= self.config.reward_threshold for r in final_rewards]),
            "total_queries": len(self.performance_history)
        }

class MultiAgentRewardSystem:
    """System for managing multiple reward-guided agents."""
    
    def __init__(self):
        self.agents = {}
        self.collaboration_history = []
    
    def add_agent(self, name: str, agent: BaseRewardGuidedAgent):
        """Add an agent to the system."""
        self.agents[name] = agent
        logger.info(f"Added agent '{name}' to multi-agent system")
    
    def remove_agent(self, name: str):
        """Remove an agent from the system."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Removed agent '{name}' from multi-agent system")
    
    def collaborative_response(self, prompt: str, agent_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate collaborative response using multiple agents."""
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        responses = {}
        rewards = {}
        
        # Get responses from each agent
        for name in agent_names:
            if name in self.agents:
                try:
                    response = self.agents[name].generate_response(prompt)
                    reward = self.agents[name].evaluate_response(prompt, response)
                    
                    responses[name] = response
                    rewards[name] = reward
                    
                except Exception as e:
                    logger.error(f"Error getting response from agent '{name}': {e}")
        
        # Find best response
        if rewards:
            best_agent = max(rewards, key=rewards.get)
            best_response = responses[best_agent]
            
            # Record collaboration
            self.collaboration_history.append({
                "prompt": prompt,
                "responses": responses,
                "rewards": rewards,
                "best_agent": best_agent,
                "best_reward": rewards[best_agent],
                "timestamp": time.time()
            })
            
            return {
                "best_response": best_response,
                "best_agent": best_agent,
                "all_responses": responses,
                "all_rewards": rewards
            }
        
        return {"error": "No valid responses generated"}
    
    def consensus_response(self, prompt: str, agent_names: Optional[List[str]] = None) -> str:
        """Generate consensus response by combining agent outputs."""
        collaboration_result = self.collaborative_response(prompt, agent_names)
        
        if "error" in collaboration_result:
            return "Unable to generate consensus response."
        
        responses = collaboration_result["all_responses"]
        
        # Simple consensus: use the response with highest reward
        # In practice, you might want more sophisticated consensus mechanisms
        return collaboration_result["best_response"]
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        if not self.collaboration_history:
            return {}
        
        agent_wins = {}
        for collab in self.collaboration_history:
            winner = collab["best_agent"]
            agent_wins[winner] = agent_wins.get(winner, 0) + 1
        
        total_collaborations = len(self.collaboration_history)
        avg_reward = np.mean([c["best_reward"] for c in self.collaboration_history])
        
        return {
            "total_collaborations": total_collaborations,
            "average_best_reward": avg_reward,
            "agent_win_rates": {
                name: wins / total_collaborations 
                for name, wins in agent_wins.items()
            },
            "agent_participation": {
                name: sum(1 for c in self.collaboration_history if name in c["responses"])
                for name in self.agents.keys()
            }
        }

class RewardModelTrainer:
    """Trainer for improving reward models using agent interactions."""
    
    def __init__(self, reward_model: BaseRewardModel):
        self.reward_model = reward_model
        self.interaction_data = []
    
    def collect_interaction(self, prompt: str, response: str, human_feedback: float):
        """Collect interaction data for training."""
        self.interaction_data.append({
            "prompt": prompt,
            "response": response,
            "human_feedback": human_feedback,
            "model_prediction": self._get_model_prediction(prompt, response),
            "timestamp": time.time()
        })
    
    def _get_model_prediction(self, prompt: str, response: str) -> float:
        """Get current model prediction for comparison."""
        try:
            # This would need proper implementation with tokenization
            return 0.5  # Placeholder
        except Exception:
            return 0.0
    
    def create_training_dataset(self) -> PreferenceDataset:
        """Create training dataset from collected interactions."""
        # This would convert interaction data to preference pairs
        # For now, return a placeholder
        logger.warning("Training dataset creation not fully implemented")
        return None
    
    def get_feedback_statistics(self) -> Dict[str, float]:
        """Get statistics about collected feedback."""
        if not self.interaction_data:
            return {}
        
        human_scores = [d["human_feedback"] for d in self.interaction_data]
        model_scores = [d["model_prediction"] for d in self.interaction_data]
        
        return {
            "total_interactions": len(self.interaction_data),
            "avg_human_score": np.mean(human_scores),
            "avg_model_score": np.mean(model_scores),
            "correlation": np.corrcoef(human_scores, model_scores)[0, 1] if len(human_scores) > 1 else 0.0,
            "disagreement_rate": np.mean([abs(h - m) > 0.5 for h, m in zip(human_scores, model_scores)])
        }

# Factory functions and utilities

def create_reward_guided_agent(
    agent_type: str = "dspy",
    config: Optional[AgentConfig] = None,
    **kwargs
) -> BaseRewardGuidedAgent:
    """Factory function to create reward-guided agents."""
    if config is None:
        config = AgentConfig(
            model_name="gpt-3.5-turbo",
            reward_model_path="./reward_model"
        )
    
    if agent_type == "dspy":
        return RewardGuidedDSPyAgent(config, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def load_agent_from_config(config_path: str) -> BaseRewardGuidedAgent:
    """Load agent from configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = AgentConfig(**config_dict["agent_config"])
    agent_type = config_dict.get("agent_type", "dspy")
    
    return create_reward_guided_agent(agent_type, config)

def evaluate_agent_performance(
    agent: BaseRewardGuidedAgent,
    test_prompts: List[str],
    ground_truth: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate agent performance on test prompts."""
    results = []
    
    for i, prompt in enumerate(test_prompts):
        try:
            response = agent.generate_response(prompt)
            reward = agent.evaluate_response(prompt, response)
            
            result = {
                "prompt": prompt,
                "response": response,
                "reward": reward,
                "timestamp": time.time()
            }
            
            if ground_truth and i < len(ground_truth):
                # Compare with ground truth if available
                gt_reward = agent.evaluate_response(prompt, ground_truth[i])
                result["ground_truth_reward"] = gt_reward
                result["reward_difference"] = reward - gt_reward
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating prompt '{prompt}': {e}")
    
    # Calculate statistics
    rewards = [r["reward"] for r in results]
    stats = {
        "total_prompts": len(test_prompts),
        "successful_responses": len(results),
        "average_reward": np.mean(rewards) if rewards else 0.0,
        "reward_std": np.std(rewards) if rewards else 0.0,
        "min_reward": np.min(rewards) if rewards else 0.0,
        "max_reward": np.max(rewards) if rewards else 0.0
    }
    
    if ground_truth:
        gt_diffs = [r.get("reward_difference", 0) for r in results if "reward_difference" in r]
        if gt_diffs:
            stats["avg_reward_vs_gt"] = np.mean(gt_diffs)
            stats["outperformed_gt_rate"] = np.mean([d > 0 for d in gt_diffs])
    
    return stats

# Example usage and demonstrations
def demo_reward_guided_agent():
    """Demonstrate reward-guided agent capabilities."""
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, skipping demo")
        return
    
    # Create agent configuration
    config = AgentConfig(
        model_name="gpt-3.5-turbo",
        reward_model_path="./reward_model",
        max_iterations=3,
        reward_threshold=0.7
    )
    
    # Create agent
    agent = create_reward_guided_agent("dspy", config)
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How can I improve my communication skills?"
    ]
    
    # Generate and optimize responses
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        
        # Generate initial response
        initial_response = agent.generate_response(prompt)
        logger.info(f"Initial response: {initial_response}")
        
        # Optimize response
        optimized_response = agent.optimize_response(prompt, initial_response)
        logger.info(f"Optimized response: {optimized_response}")
    
    # Show performance statistics
    stats = agent.get_performance_stats()
    logger.info(f"\nPerformance statistics: {stats}")

if __name__ == "__main__":
    demo_reward_guided_agent()