"""Base classes for query generators."""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Generator(ABC):
    """Abstract base class for query generators."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.is_loaded = False
        logger.debug(f"Generator {model_name} initialized with kwargs: {kwargs}")
    
    def _merge_kwargs(self, **function_kwargs):
        """Merge stored initialization kwargs with function call kwargs.
        
        Function call kwargs take precedence over initialization kwargs.
        
        Args:
            **function_kwargs: Kwargs passed to the function call
            
        Returns:
            dict: Merged kwargs with function call kwargs taking precedence
        """
        merged = self.kwargs.copy()
        merged.update(function_kwargs)
        return merged

    @abstractmethod
    async def load(self):
        """Load the model and initialize resources (async)."""
        pass

    @abstractmethod
    async def unload(self):
        """Unload the model and free resources (async)."""
        pass

    @abstractmethod
    async def generate(
        self, queries: List[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str, Optional[str]]]]:
        """Generate responses for a list of queries (async).

        Args:
            queries: List of query strings
            **generation_kwargs: Model-specific generation parameters

        Returns:
            List of responses. Each response can be either:
            - str: Simple response text (for models without reasoning)
            - Tuple[str, Optional[str]]: (response_text, reasoning_trace) for models with reasoning
        """
        pass
