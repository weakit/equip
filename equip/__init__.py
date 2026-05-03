"""UPHILL v2 - Modern evaluation framework for LLM benchmarks."""

__version__ = "2.0.0"

from .models import Query, Generation, Evaluation, AggregatedMetrics
from .loaders import DatasetLoader, UphillLoader, FoolMeTwiceLoader, SciFactLoader, get_dataset_loader
from .storage import Storage
from .config import ModelConfig
from .utils import setup_logging, generate_id

__all__ = [
    "Query",
    "Generation",
    "Evaluation",
    "AggregatedMetrics",
    "DatasetLoader",
    "UphillLoader",
    "FoolMeTwiceLoader",
    "SciFactLoader",
    "get_dataset_loader",
    "Storage",
    "ModelConfig",
    "setup_logging",
    "generate_id",
]
