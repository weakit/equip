"""Base classes for entailment models."""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class EntailmentResult(BaseModel):
    """Result of an entailment check."""
    reasoning: str
    entailment: str  # agree, disagree, neutral, error
    unsure: bool


class EntailmentModel(ABC):
    """Abstract base class for entailment models."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.is_loaded = False
        logger.debug(f"EntailmentModel {model_name} initialized")
    
    def _merge_kwargs(self, **function_kwargs):
        """Merge stored initialization kwargs with function call kwargs."""
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
    async def check_entailment(
        self, claims: List[str], responses: List[str], **kwargs
    ) -> List[EntailmentResult]:
        """Check entailment between claims and responses (async).

        Args:
            claims: List of claim strings
            responses: List of response strings
            **kwargs: Model-specific parameters

        Returns:
            List of EntailmentResult objects
        """
        pass
