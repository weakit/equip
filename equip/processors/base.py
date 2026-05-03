"""Base classes for preprocessing and postprocessing."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class Preprocessor(ABC):
    """Abstract base class for input preprocessing."""
    
    def __init__(self, **config):
        """Initialize preprocessor with configuration."""
        self.config = config
        
    @abstractmethod
    def process(self, queries: List[str], **generation_kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess input queries before model generation.
        
        Args:
            queries: List of input queries
            **generation_kwargs: Generation parameters
            
        Returns:
            Tuple of (processed_inputs, updated_generation_kwargs)
        """
        pass


class Postprocessor(ABC):
    """Abstract base class for output postprocessing."""
    
    def __init__(self, **config):
        """Initialize postprocessor with configuration."""
        self.config = config
        
    @abstractmethod
    def process(self, outputs: List[Any], original_queries: List[str]) -> Union[List[str], List[Tuple[str, str]]]:
        """Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            original_queries: Original input queries
            
        Returns:
            Processed responses. Can be either:
            - List[str]: Simple response texts
            - List[Tuple[str, str]]: (response, reasoning) pairs
        """
        pass
