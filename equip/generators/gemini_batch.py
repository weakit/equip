"""
Gemini Batch API generator implementation.

Supports both thinking_budget (explicit token control) and thinking_level (LOW/HIGH) modes.
Uses the Gemini File API to upload batch input files and monitor batch jobs.
"""

import logging
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
from functools import cached_property

from equip.models import Generation, Query
from equip.storage import Storage
from equip.utils import generate_id
from .batched import BatchedGenerator

logger = logging.getLogger(__name__)


class GeminiBatchGenerator(BatchedGenerator):
    """
    Generator for Gemini Batch API with extended thinking support.
    
    Supports:
    - thinking_budget: Explicit token control (e.g., 2048)
    - thinking_level: HIGH/LOW for Gemini 3+ models
    - Thought extraction from candidate.content.parts where part.thought == True
    """
    
    def __init__(
        self,
        model_name: str,
        storage_base_dir: Path | None = None,
        max_batch_requests: int = 10000,
        concurrent_batches: int = 5,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 8192,
    ):
        super().__init__(model_name, max_batch_requests=max_batch_requests, concurrent_batches=concurrent_batches)
        self.storage_base_dir = storage_base_dir
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Client and types will be initialized in load()
        self.client = None
        self.types = None
        
        logger.info(
            f"Initialized GeminiBatchGenerator: model={model_name}, "
            f"thinking_budget={thinking_budget}, thinking_level={thinking_level}, "
            f"max_batch_requests={max_batch_requests}, concurrent_batches={concurrent_batches}"
        )
    
    async def load(self):
        """Load the Gemini client."""
        if self.is_loaded:
            return
        
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client()
            self.types = types
        except ImportError as e:
            raise ImportError(
                "GeminiBatchGenerator requires google-genai package. "
                "Install with: pip install google-genai"
            ) from e
        
        self.is_loaded = True
        logger.info(f"Loaded Gemini Batch API client for model: {self.model_name}")
    
    async def unload(self):
        """Unload the Gemini client."""
        if not self.is_loaded:
            return
        
        if self.client is not None:
            del self.client
            self.client = None
        
        self.is_loaded = False
        logger.info(f"Unloaded Gemini Batch API client")
    
    def _get_batch_dir(self, storage: Storage) -> Path:
        """Get batch directory for a storage instance."""
        batch_dir = storage.base_dir / "batches"
        batch_dir.mkdir(exist_ok=True)
        return batch_dir
    
    def _get_inputs_dir(self, storage: Storage) -> Path:
        """Get inputs directory for a storage instance."""
        inputs_dir = self._get_batch_dir(storage) / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        return inputs_dir
    
    def _get_metadata_file(self, storage: Storage) -> Path:
        """Get batch metadata file path for a storage instance."""
        return self._get_batch_dir(storage) / "batch_metadata.jsonl"

    
    @cached_property
    def _generation_config(self) -> dict[str, Any]:
        """Build generation config for batch requests."""
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if self.thinking_level is not None or self.thinking_budget is not None:
            # Convert to dict for JSONL serialization
            config["thinking_config"] = {
                "include_thoughts": True
            }
            if self.thinking_level is not None:
                config["thinking_config"]["thinking_level"] = self.thinking_level
            elif self.thinking_budget is not None:
                config["thinking_config"]["thinking_budget"] = self.thinking_budget
        
        return config
    
    def _create_batch_input_line(self, query_text: str, custom_id: str) -> dict[str, Any]:
        """Create a single batch input line in Gemini format."""
        return {
            "key": custom_id,
            "request": {
                "contents": [
                    {
                        "parts": [{"text": query_text}],
                        "role": "user"
                    }
                ],
                "generation_config": self._generation_config
            }
        }
    
    def _extract_content_and_thoughts(self, candidate: Any) -> tuple[str, str | None]:
        """
        Extract response content and thoughts from a candidate.
        
        Gemini responses have parts where:
        - part.thought == True: Contains thought summary
        - part.thought == False: Contains regular content
        
        Handles both dict (from file) and object (from inline) formats.
        """
        content_parts = []
        thought_parts = []
        
        # Get content - handle both dict and object formats
        if isinstance(candidate, dict):
            content = candidate.get('content', {})
            parts = content.get('parts', [])
        else:
            content = candidate.content
            parts = content.parts
        
        # Process parts
        for part in parts:
            # Handle both dict and object formats
            if isinstance(part, dict):
                is_thought = part.get('thought', False)
                text = part.get('text', '')
            else:
                is_thought = hasattr(part, 'thought') and part.thought
                text = part.text
            
            if is_thought:
                # This is a thought summary
                thought_parts.append(text)
            else:
                # This is regular content
                content_parts.append(text)
        
        response_text = "\n".join(content_parts).strip()
        reasoning_trace = "\n".join(thought_parts).strip() or None
        
        return response_text, reasoning_trace
    
    def _load_batch_metadata(self, storage: Storage) -> list[dict[str, Any]]:
        """Load batch metadata from tracking file."""
        metadata_file = self._get_metadata_file(storage)
        if not metadata_file.exists():
            return []
        
        batches = []
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    batches.append(json.loads(line))
        return batches
    
    def _save_batch_metadata(self, storage: Storage, batches: list[dict[str, Any]]):
        """Save batch metadata to tracking file."""
        metadata_file = self._get_metadata_file(storage)
        with open(metadata_file, 'w') as f:
            for batch in batches:
                f.write(json.dumps(batch) + '\n')
    
    def _add_batch_metadata(self, storage: Storage, batch_info: dict[str, Any]):
        """Add a new batch to metadata tracking."""
        batches = self._load_batch_metadata(storage)
        batches.append(batch_info)
        self._save_batch_metadata(storage, batches)
    
    def _update_batch_status(self, storage: Storage, batch_id: str, updates: dict[str, Any]):
        """Update a batch's status in metadata."""
        batches = self._load_batch_metadata(storage)
        for batch in batches:
            if batch['batch_id'] == batch_id:
                batch.update(updates)
                break
        self._save_batch_metadata(storage, batches)
    
    async def consolidate_batches(self, storage: Storage) -> dict[str, Any]:
        """
        Check status of active batches and consolidate completed ones.
        
        Args:
            storage: Storage instance for saving generations
        
        Returns:
            Dict with keys:
            - completed: Number of generations consolidated
            - batches_completed: Number of batches completed
            - batches_failed: Number of batches failed
        """
        logger.info("Consolidating Gemini batches...")
        
        batches = self._load_batch_metadata(storage)
        completed_count = 0
        batches_completed = 0
        batches_failed = 0
        
        for batch in batches:
            if batch.get('status') != 'active':
                continue
            
            batch_id = batch['batch_id']
            batch_name = batch['batch_name']
            
            try:
                # Check batch status
                logger.info(f"Checking batch {batch_id} status...")
                batch_job = self.client.batches.get(name=batch_name)
                
                state = batch_job.state.name  # PENDING, RUNNING, SUCCEEDED, FAILED, etc.
                state = state.removeprefix("JOB_STATE_")

                if state in ['PENDING', 'RUNNING']:
                    logger.info(f"Batch {batch_id} is {state}, skipping...")
                    continue
                
                elif state == 'SUCCEEDED':
                    logger.info(f"Batch {batch_id} succeeded, processing results...")
                    
                    # Load custom_id metadata
                    metadata_path = Path(batch['metadata_file'])
                    with open(metadata_path, 'r') as f:
                        custom_id_map = json.load(f)
                    
                    # Initialize generations list for this batch
                    generations = []
                    
                    # Get results - they might be inline or in a file
                    if hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                        # Results are in a file - download and parse JSONL
                        logger.info(f"Downloading results from file: {batch_job.dest.file_name}")
                        file_content_bytes = self.client.files.download(file=batch_job.dest.file_name)
                        file_content = file_content_bytes.decode('utf-8')
                        
                        results = []
                        
                        for line in file_content.splitlines():
                            if line.strip():
                                results.append(json.loads(line))
                    
                    elif hasattr(batch_job.dest, 'inlined_responses'):
                        # Results are inline
                        results = batch_job.dest.inlined_responses
                    
                    else:
                        logger.error(f"Batch {batch_id}: No results found")
                        self._update_batch_status(storage, batch_id, {'status': 'failed', 'error': 'no_results'})
                        batches_failed += 1
                        continue
                    
                    # Process each result
                    for result in results:
                        custom_id = result['key']
                        metadata = custom_id_map.get(custom_id)
                        
                        if not metadata:
                            logger.warning(f"No metadata found for custom_id: {custom_id}")
                            continue
                        
                        # Extract response
                        if 'response' in result and result['response']:
                            response = result['response']
                            
                            # Get first candidate
                            if response.get('candidates'):
                                candidate = response['candidates'][0]
                                
                                # Extract content and thoughts
                                response_text, reasoning_trace = self._extract_content_and_thoughts(candidate)
                                
                                # Create generation object
                                generation = Generation(
                                    gen_id=generate_id(),
                                    query_id=metadata['query_id'],
                                    dataset=metadata['dataset'],
                                    presupposition_level=metadata.get('presupposition_level'),
                                    sample_idx=metadata.get('sample_idx'),
                                    gen_model=metadata['gen_model'],
                                    response=response_text,
                                    reasoning_trace=reasoning_trace
                                )
                                
                                generations.append(generation)
                                completed_count += 1
                            else:
                                logger.warning(f"No candidates in response for {custom_id}")
                        
                        elif 'error' in result:
                            # Log error
                            error = result['error']
                            logger.error(f"Error for {custom_id}: {error}")
                            
                            # Save error to file
                            error_file = self._get_batch_dir(storage) / f"error_{batch_id}.jsonl"
                            with open(error_file, 'a') as f:
                                f.write(json.dumps({
                                    'custom_id': custom_id,
                                    'metadata': metadata,
                                    'error': error
                                }) + '\n')
                    
                    # Save all generations with deduplication
                    if generations:
                        self._save_generations_with_dedup(storage, generations)
                        logger.info(f"Saved {len(generations)} generations from batch {batch_id}")
                    
                    # Mark batch as consolidated
                    self._update_batch_status(storage, batch_id, {'status': 'completed'})
                    batches_completed += 1
                    logger.info(f"Batch {batch_id} completed: {completed_count} generations")
                
                elif state in ['FAILED', 'CANCELLED', 'EXPIRED']:
                    logger.error(f"Batch {batch_id} {state}")
                    self._update_batch_status(storage, batch_id, {'status': 'failed', 'error': state.lower()})
                    batches_failed += 1
                
                else:
                    logger.error(f"Batch {batch_id} in unexpected state: {state}")
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
                self._update_batch_status(storage, batch_id, {'status': 'failed', 'error': str(e)})
                batches_failed += 1
        
        logger.info(
            f"Consolidation complete: {completed_count} generations, "
            f"{batches_completed} batches completed, {batches_failed} batches failed"
        )
        
        return {
            'completed': completed_count,
            'batches_completed': batches_completed,
            'batches_failed': batches_failed
        }
    
    async def enqueue_batches(
        self,
        work_queue: list[tuple[Query, int]],
        storage: Storage,
        **generation_kwargs
    ) -> dict[str, Any]:
        """Enqueue new batch jobs from pending queries.
        
        Args:
            work_queue: List of (Query, n_samples_needed) tuples
            storage: Storage instance
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dict with keys:
            - newly_enqueued: Number of requests enqueued
            - batches_active: Number of active batches
            - pending_requests: Number of requests still pending
        """
        if not work_queue:
            return {
                'newly_enqueued': 0,
                'batches_active': 0,
                'pending_requests': 0
            }
        
        # Load existing batches to check what's already pending
        batches = self._load_batch_metadata(storage)
        
        # Count requests in active batches
        pending_in_batches = {}  # (query_id, presupposition_level, sample_idx) -> True
        for batch in batches:
            if batch.get('status') == 'active':
                # Load metadata file to see what's in this batch
                metadata_file = Path(batch['metadata_file'])
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        custom_id_map = json.load(f)
                    
                    for custom_id, metadata in custom_id_map.items():
                        key = (
                            metadata['query_id'],
                            metadata.get('presupposition_level'),
                            metadata.get('sample_idx', 0)
                        )
                        pending_in_batches[key] = True
        
        # Expand work_queue into individual queries, excluding those already pending
        queries = []
        skipped = 0
        for query, n_samples in work_queue:
            for sample_idx in range(n_samples):
                key = (query.id, query.presupposition_level, sample_idx)
                if key in pending_in_batches:
                    skipped += 1
                    continue
                
                # Create a query copy with sample_idx
                query_copy = Query(
                    id=query.id,
                    claim=query.claim,
                    veracity=query.veracity,
                    presupposition_level=query.presupposition_level,
                    query_text=query.query_text,
                    metadata={**query.metadata, 'sample_idx': sample_idx}
                )
                queries.append(query_copy)
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} requests already in active batches")
        
        logger.info(f"Enqueueing batches for {len(queries)} requests (from {len(work_queue)} queries)...")
        
        # Count active batches
        active_batches = sum(1 for b in batches if b.get('status') == 'active')
        
        newly_enqueued = 0
        
        # Process queries in chunks
        remaining_queries = queries[:]
        inputs_dir = self._get_inputs_dir(storage)
        
        while remaining_queries and active_batches < self.concurrent_batches:
            # Take a batch worth of queries
            batch_queries = remaining_queries[:self.max_batch_requests]
            remaining_queries = remaining_queries[self.max_batch_requests:]
            
            # Create batch ID and files
            batch_id = str(uuid4())
            timestamp = int(time.time())
            input_file = inputs_dir / f"batch_{timestamp}_{batch_id}.jsonl"
            metadata_file = inputs_dir / f"batch_{timestamp}_{batch_id}_metadata.json"
            
            # Create custom_id map and batch input
            custom_id_map = {}
            batch_lines = []
            
            for query in batch_queries:
                # Generate stable UUID for custom_id
                custom_id = str(uuid4())
                
                # Store metadata
                custom_id_map[custom_id] = {
                    'query_id': query.id,
                    'dataset': storage.dataset_name,
                    'presupposition_level': query.presupposition_level,
                    'sample_idx': query.metadata.get('sample_idx', 0),
                    'gen_model': self.model_name
                }
                
                # Create batch line
                batch_line = self._create_batch_input_line(query.query_text, custom_id)
                batch_lines.append(batch_line)
            
            # Write batch input file
            with open(input_file, 'w') as f:
                for line in batch_lines:
                    f.write(json.dumps(line) + '\n')
            
            # Write metadata file
            with open(metadata_file, 'w') as f:
                json.dump(custom_id_map, f, indent=2)
            
            try:
                # Upload file to Gemini
                logger.info(f"Uploading batch file: {input_file.name}")
                uploaded_file = self.client.files.upload(
                    file=str(input_file),
                    config=self.types.UploadFileConfig(
                        display_name=input_file.name,
                        mime_type='jsonl'
                    )
                )
                
                # Create batch job
                logger.info(f"Creating batch job for {len(batch_queries)} requests...")
                batch_job = self.client.batches.create(
                    model=self.model_name,
                    src=uploaded_file.name,
                    config={
                        'display_name': f'batch-{batch_id[:8]}'
                    }
                )
                
                batch_name = batch_job.name
                
                # Save batch metadata
                batch_info = {
                    'batch_id': batch_id,
                    'batch_name': batch_name,
                    'input_file': str(input_file),
                    'metadata_file': str(metadata_file),
                    'status': 'active',
                    'request_count': len(batch_queries),
                    'created_at': timestamp
                }
                self._add_batch_metadata(storage, batch_info)
                
                newly_enqueued += len(batch_queries)
                active_batches += 1
                
                logger.info(
                    f"Batch {batch_id} created: {len(batch_queries)} requests, "
                    f"active_batches={active_batches}"
                )
            
            except Exception as e:
                logger.error(f"Failed to create batch: {e}", exc_info=True)
                # Continue to next batch
                continue
        
        pending_requests = len(remaining_queries)
        
        logger.info(
            f"Enqueue complete: {newly_enqueued} enqueued, "
            f"{active_batches} active batches, {pending_requests} pending"
        )
        
        return {
            'newly_enqueued': newly_enqueued,
            'batches_active': active_batches,
            'pending_requests': pending_requests
        }
