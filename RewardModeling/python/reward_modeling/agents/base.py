from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the reward modeling system.
    Provides common interface and functionality for different agent implementations.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            model_name_or_path: Path to the model or model name
            device: Device to run the model on ('cuda', 'cpu', etc.)
            **kwargs: Additional arguments for specific implementations
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._setup(**kwargs)
    
    @abstractmethod
    def _setup(self, **kwargs) -> None:
        """
        Set up the agent with the given parameters.
        To be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """
        Generate a response for the given input.
        
        Args:
            input_text: Input text/prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def generate_with_reward(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response with associated reward score.
        
        Args:
            input_text: Input text/prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing response text and reward score
        """
        pass
    
    @abstractmethod
    def batch_generate(self, input_texts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple inputs.
        
        Args:
            input_texts: List of input texts/prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated response texts
        """
        pass
    
    def __call__(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str], Dict[str, Any]]:
        """
        Call the agent to generate a response.
        
        Args:
            input_text: Input text/prompt or list of inputs
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response(s) or dictionary with response and reward
        """
        if isinstance(input_text, list):
            return self.batch_generate(input_text, **kwargs)
        elif kwargs.get("with_reward", False):
            return self.generate_with_reward(input_text, **kwargs)
        else:
            return self.generate(input_text, **kwargs) 