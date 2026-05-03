"""Evaluator modules for entailment checking."""

from .base import EntailmentModel, EntailmentResult
from .vllm import VLLMEntailmentModel

__all__ = ['EntailmentModel', 'EntailmentResult', 'VLLMEntailmentModel']
