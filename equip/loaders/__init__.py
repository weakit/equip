"""Dataset loaders for different benchmark formats."""

from .base import DatasetLoader
from .uphill import UphillLoader
from .foolmetwice import FoolMeTwiceLoader
from .scifact import SciFactLoader


# Registry for easy access
DATASET_LOADERS = {
    "uphill": UphillLoader,
    "foolmetwice": FoolMeTwiceLoader,
    "scifact": SciFactLoader,
}


def get_dataset_loader(dataset_name: str) -> DatasetLoader:
    """Get a dataset loader by name."""
    loader_class = DATASET_LOADERS.get(dataset_name.lower())
    if loader_class is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {', '.join(DATASET_LOADERS.keys())}"
        )
    return loader_class()


__all__ = ["DatasetLoader", "UphillLoader", "FoolMeTwiceLoader", "SciFactLoader", "get_dataset_loader"]
