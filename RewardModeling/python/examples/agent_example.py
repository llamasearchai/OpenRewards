#!/usr/bin/env python
"""
Example script demonstrating how to use BaseAgent implementations 
with both DSPy and LangChain.
"""

import argparse
import logging
from pathlib import Path
import dspy_ai as dspy
from transformers import AutoTokenizer

from reward_modeling.utils import setup_logger, load_config
from reward_modeling.agents import BaseAgent, LangChainAgent, RewardGuidedAgent, OptimizedRewardAgent
from reward_modeling.models import RewardModel


def main():
    parser = argparse.ArgumentParser(description="Agent example script")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--agent_type", 
        type=str, 
        default="langchain",
        choices=["langchain", "dspy", "dspy_optimized"],
        help="Type of agent to use"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="What is reward modeling and why is it useful?",
        help="Input text to generate a response for"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    
    # Initialize agent based on type
    logger.info(f"Initializing {args.agent_type} agent")
    
    if args.agent_type == "langchain":
        agent = LangChainAgent(
            model_name_or_path=config.model.model_name_or_path,
            reward_model_path=config.agent.reward_model_path if config.agent else None,
            temperature=config.agent.temperature if config.agent else 0.7,
            max_tokens=config.agent.max_tokens if config.agent else 512,
            device=config.device
        )
    
    elif args.agent_type == "dspy":
        # Initialize DSPy
        dspy.settings.configure(lm=dspy.OpenAI(model=config.model.model_name_or_path))
        
        # Create a simple DSPy module for the task
        class SimpleResponseModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate = dspy.ChainOfThought(
                    dspy.Signature(
                        input=dspy.InputField(desc="Input text to respond to"),
                        response=dspy.OutputField(desc="Detailed response to the input")
                    )
                )
            
            def forward(self, input):
                return self.generate(input=input)
        
        # Create the DSPy task module
        task_module = SimpleResponseModule()
        
        # Initialize the reward-guided agent
        agent = RewardGuidedAgent(
            model_name_or_path=config.model.model_name_or_path,
            reward_model_path=config.agent.reward_model_path if config.agent else None,
            task_module=task_module,
            tokenizer=tokenizer,
            num_candidates=config.agent.num_candidates if config.agent else 3,
            device=config.device
        )
    
    elif args.agent_type == "dspy_optimized":
        # Initialize DSPy
        dspy.settings.configure(lm=dspy.OpenAI(model=config.model.model_name_or_path))
        
        # Create a simple DSPy module for the task
        class SimpleResponseModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate = dspy.ChainOfThought(
                    dspy.Signature(
                        input=dspy.InputField(desc="Input text to respond to"),
                        response=dspy.OutputField(desc="Detailed response to the input")
                    )
                )
            
            def forward(self, input):
                return self.generate(input=input)
        
        # Create the DSPy task module
        task_module = SimpleResponseModule()
        
        # Create some simple examples for optimization
        class Example:
            def __init__(self, input, response):
                self.input = input
                self.response = response
        
        examples = [
            Example(
                "What is reward modeling?",
                "Reward modeling is a technique used in AI alignment where a model is trained to predict human preferences."
            ),
            Example(
                "How does DPO work?",
                "Direct Preference Optimization (DPO) is a technique that directly optimizes a policy to align with human preferences without using a separate reward model."
            )
        ]
        
        # Initialize the optimized reward agent
        agent = OptimizedRewardAgent(
            model_name_or_path=config.model.model_name_or_path,
            reward_model_path=config.agent.reward_model_path if config.agent else None,
            task_module=task_module,
            tokenizer=tokenizer,
            train_examples=examples,
            device=config.device
        )
        
        # Optimize the agent if reward model is available
        if config.agent and config.agent.reward_model_path:
            logger.info("Optimizing agent...")
            agent.optimize(num_iterations=3)  # Limited iterations for example
    
    # Generate response
    logger.info(f"Generating response for input: {args.input}")
    result = agent.generate_with_reward(args.input)
    
    # Print results
    print("\n===== AGENT RESPONSE =====")
    print(f"Input: {args.input}")
    print(f"Response: {result['response']}")
    if result.get('reward') is not None:
        print(f"Reward Score: {result['reward']:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    main() 