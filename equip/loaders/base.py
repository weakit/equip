"""Base classes for dataset loaders."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import Query
from ..utils import get_data_dir

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_dir = get_data_dir() / dataset_name
    
    @abstractmethod
    def load(self) -> List[Query]:
        """Load the dataset and return a list of Query objects."""
        pass
