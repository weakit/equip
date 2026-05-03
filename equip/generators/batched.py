"""Base class for batched asynchronous generators."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from ..models import Query, Generation
from ..storage import Storage


class BatchedGenerator(ABC):
    """Base class for batched asynchronous generators.
    
    Unlike regular Generators which return results immediately,
    BatchedGenerators manage long-running batch jobs across multiple runs.
    
    Each run should:
    1. Consolidate completed batches and save Generation objects
    2. Enqueue new batches for pending work
    3. Return status information
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # Extract and store common batch parameters
        self.concurrent_batches = kwargs.pop('concurrent_batches', 5)
        self.max_batch_requests = kwargs.pop('max_batch_requests', None)  # Can be None if not used
        self.kwargs = kwargs
        self.is_loaded = False
    
    def _save_generations_with_dedup(self, storage: Storage, new_generations: List[Generation]) -> None:
        """Save new generations with deduplication against existing ones.
        
        Loads existing generations, deduplicates by gen_id (newer override older),
        and saves the complete deduplicated list atomically.
        
        Args:
            storage: Storage instance for loading/saving
            new_generations: List of new Generation objects to save
        """
        if not new_generations:
            return
        
        # Load existing generations
        existing_gens = storage.load_generations()
        
        # Deduplicate by gen_id (newer generations override older ones)
        gen_dict = {g.gen_id: g for g in existing_gens}
        for g in new_generations:
            gen_dict[g.gen_id] = g
        
        # Save deduplicated list atomically
        deduplicated = list(gen_dict.values())
        storage.save_generations(deduplicated)
    
    @abstractmethod
    async def load(self):
        """Load the client/resources."""
        pass
    
    @abstractmethod
    async def unload(self):
        """Unload and cleanup."""
        pass
    
    @abstractmethod
    async def consolidate_batches(self, storage: Storage) -> Dict[str, Any]:
        """Consolidate completed batches and save Generation objects.
        
        Args:
            storage: Storage instance for saving generations
            
        Returns:
            Status dict with keys:
            - completed: Number of generations saved this run
            - batches_completed: Number of batches completed
            - batches_failed: Number of batches that failed
        """
        pass
    
    @abstractmethod
    async def enqueue_batches(
        self,
        work_queue: List[Tuple[Query, int]],
        storage: Storage,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Enqueue new batches for pending work.
        
        Args:
            work_queue: List of (Query, n_samples_needed) tuples
            storage: Storage instance
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Status dict with keys:
            - newly_enqueued: Number of new requests enqueued
            - batches_active: Number of active batch jobs
            - pending_requests: Total pending requests
        """
        pass
