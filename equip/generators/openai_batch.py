"""OpenAI Batch API Generator with 50% cost savings."""

import os
import json
import uuid

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

from .batched import BatchedGenerator
from ..utils import generate_id
from ..models import Query, Generation
from ..storage import Storage


# Reasoning models that use the Responses API
REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "gpt-5"}


class OpenAIBatchGenerator(BatchedGenerator):
    """Generator using OpenAI Batch API with 50% cost savings.

    Supports both standard Chat Completions and Responses API for reasoning models.

    Args:
        model_name: OpenAI model name (e.g., "gpt-4", "o1-preview")
        max_batch_queries: Maximum queries per batch (default: 50000)
        concurrent_batches: Maximum concurrent batches (default: 5)
        reasoning_effort: For reasoning models (low/medium/high, default: medium)
        temperature: Temperature for non-reasoning models (default: 1.0)
        max_tokens: Max completion tokens (default: 2048)
        **kwargs: Additional OpenAI parameters
    """

    def __init__(
        self,
        model_name: str,
        max_batch_queries: int = 50000,
        concurrent_batches: int = 5,
        reasoning_effort: str = "medium",
        temperature: float = 1.0,
        max_tokens: int = 8192,
        **kwargs,
    ):
        super().__init__(model_name, concurrent_batches=concurrent_batches, max_batch_requests=max_batch_queries, **kwargs)
        self.max_batch_queries = max_batch_queries  # Keep for backward compatibility
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client: Optional[AsyncOpenAI] = None

    async def load(self):
        """Initialize the AsyncOpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.is_loaded = True
        print(f"OpenAI Batch API client loaded for model: {self.model_name}")

    async def unload(self):
        """Close the client."""
        if self.client:
            await self.client.close()
            self.client = None
        self.is_loaded = False

    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model (o1/o3/gpt-5)."""
        model_lower = self.model_name.lower()
        return any(rm in model_lower for rm in REASONING_MODELS)

    def _get_batch_dir(self, storage: Storage) -> Path:
        """Get the batch directory path."""
        batch_dir = Path(storage.base_dir) / "batches"
        batch_dir.mkdir(exist_ok=True)
        return batch_dir

    def _get_metadata_file(self, storage: Storage) -> Path:
        """Get the batch metadata file path."""
        return self._get_batch_dir(storage) / "batch_metadata.jsonl"

    def _get_batch_input_dir(self, storage: Storage) -> Path:
        """Get the batch inputs directory."""
        input_dir = self._get_batch_dir(storage) / "inputs"
        input_dir.mkdir(exist_ok=True)
        return input_dir

    def _load_batch_metadata(self, storage: Storage) -> List[Dict[str, Any]]:
        """Load batch metadata from file."""
        metadata_file = self._get_metadata_file(storage)
        if not metadata_file.exists():
            return []

        batches = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batches.append(json.loads(line))
        return batches

    def _save_batch_metadata(self, storage: Storage, batch_record: Dict[str, Any]):
        """Append a batch metadata record."""
        metadata_file = self._get_metadata_file(storage)
        with open(metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(batch_record) + "\n")

    def _update_batch_metadata(self, storage: Storage, batches: List[Dict[str, Any]]):
        """Overwrite batch metadata file."""
        metadata_file = self._get_metadata_file(storage)
        with open(metadata_file, "w", encoding="utf-8") as f:
            for batch in batches:
                f.write(json.dumps(batch) + "\n")

    async def _create_batch_input_file(
        self, work_queue: List[Tuple[Query, int]], storage: Storage, **generation_kwargs
    ) -> Tuple[Path, int]:
        """Create a JSONL batch input file with rich metadata.

        Returns:
            (input_file_path, request_count)
        """
        batch_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = self._get_batch_input_dir(storage) / f"batch_{timestamp}_{batch_id}.jsonl"
        metadata_file = (
            self._get_batch_input_dir(storage) / f"batch_{timestamp}_{batch_id}_metadata.json"
        )

        is_reasoning = self._is_reasoning_model()
        request_count = 0
        metadata_map = {}  # custom_id -> metadata (stored locally)

        with open(input_file, "w", encoding="utf-8") as f:
            for query, n_samples in work_queue[: self.max_batch_queries]:
                for sample_idx in range(n_samples):
                    custom_id = str(uuid.uuid4())

                    # Rich metadata for tracking (stored locally, not sent to OpenAI)
                    metadata = {
                        "query_id": query.id,
                        "dataset": storage.dataset_name,
                        "presupposition_level": query.presupposition_level,
                        "sample_idx": sample_idx,
                        "gen_model": self.model_name,
                        "timestamp": timestamp,
                        "batch_id": batch_id,
                    }
                    metadata_map[custom_id] = metadata

                    # Build request based on API type
                    if is_reasoning:
                        # Responses API for o1/o3/gpt-5
                        request = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/responses",
                            "body": {
                                "model": self.model_name,
                                "input": [{"role": "user", "content": query.query_text}],
                                "reasoning": {"effort": self.reasoning_effort},
                                "max_output_tokens": self.max_tokens,
                            },
                        }
                    else:
                        # Chat Completions API for standard models
                        request = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model_name,
                                "messages": [{"role": "user", "content": query.query_text}],
                                "temperature": self.temperature,
                                "max_tokens": self.max_tokens,
                            },
                        }

                    f.write(json.dumps(request) + "\n")
                    request_count += 1

                    if request_count >= self.max_batch_queries:
                        break

                if request_count >= self.max_batch_queries:
                    break

        # Save metadata mapping locally
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_map, f, indent=2)

        print(f"Created batch input file: {input_file.name} ({request_count} requests)")
        return input_file, request_count

    async def _upload_batch(self, input_file: Path) -> str:
        """Upload batch file and create batch job.

        Returns:
            batch_id
        """
        # Upload the file
        with open(input_file, "rb") as f:
            file_obj = await self.client.files.create(file=f, purpose="batch")

        # Create the batch
        batch = await self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses" if self._is_reasoning_model() else "/v1/chat/completions",
            completion_window="24h",
        )

        print(f"Uploaded batch {batch.id} (file: {file_obj.id})")
        return batch.id

    async def _consolidate_completed_batches(self, storage: Storage) -> Tuple[int, int, int]:
        """Check batch statuses and consolidate completed ones.

        Returns:
            (generations_saved, batches_completed, batches_failed)
        """
        batches = self._load_batch_metadata(storage)
        updated_batches = []
        generations_saved = 0
        batches_completed = 0
        batches_failed = 0

        for batch_record in batches:
            if batch_record["status"] in ["completed", "failed", "consolidated"]:
                updated_batches.append(batch_record)
                continue

            # Check batch status
            batch_id = batch_record["batch_id"]
            
            try:
                batch = await self.client.batches.retrieve(batch_id)
                
                batch_record["status"] = batch.status

                if batch.status == "completed":
                    # Download and process results
                    print(f"Consolidating completed batch {batch_id}...")

                    # Read metadata from local file
                    input_file = Path(batch_record["input_file"])
                    metadata_file = input_file.parent / f"{input_file.stem}_metadata.json"

                    if not metadata_file.exists():
                        print(f"WARNING: Metadata file not found: {metadata_file}")
                        metadata_map = {}
                    else:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata_map = json.load(f)  # custom_id -> metadata

                    # Download output file
                    output_content = await self.client.files.content(batch.output_file_id)
                    output_text = output_content.text

                    # Parse results and create Generation objects
                    generations = []
                    for line in output_text.strip().split("\n"):
                        if not line.strip():
                            continue

                        result = json.loads(line)
                        custom_id = result["custom_id"]

                        if custom_id not in metadata_map:
                            print(f"WARNING: No metadata found for custom_id {custom_id}")
                            continue

                        metadata = metadata_map[custom_id]
                        response_obj = result.get("response", {}).get("body", {})

                        # Extract response text based on API type
                        try:
                            if response_obj.get("output"):
                                # Responses API
                                outputs = response_obj["output"]
                                reasoning_outputs = [x for x in outputs if x["type"] == "reasoning"]
                                text_outputs = [x for x in outputs if x["type"] == "message"]

                                reasoning_trace = (
                                    "\n".join(
                                        [
                                            "\n".join(
                                                s["text"]
                                                for s in x["summary"]
                                                if s["type"] == "summary_text"
                                            )
                                            for x in reasoning_outputs
                                        ]
                                    ).strip()
                                    or None
                                )

                                response_text = "\n".join(
                                    [
                                        "\n".join(
                                            s["text"]
                                            for s in x["content"]
                                            if s["type"] == "output_text"
                                        )
                                        for x in text_outputs
                                    ]
                                ).strip()
                            else:
                                # Chat Completions API
                                reasoning_trace = None
                                response_text = response_obj["choices"][0]["message"]["content"]

                        except Exception as e:
                            print(f"WARNING: Unknown response format for {custom_id}: {e}")
                            print(f"Response object: {response_obj}")
                            continue

                        # Create Generation object
                        generation = Generation(
                            gen_id=generate_id(),
                            query_id=metadata["query_id"],
                            dataset=metadata["dataset"],
                            presupposition_level=metadata["presupposition_level"],
                            gen_model=metadata["gen_model"],
                            response=response_text,
                            reasoning_trace=reasoning_trace,
                        )
                        generations.append(generation)

                    # Save all generations with deduplication
                    self._save_generations_with_dedup(storage, generations)
                    generations_saved += len(generations)
                    batches_completed += 1

                    batch_record["status"] = "consolidated"
                    batch_record["completed_at"] = datetime.now().isoformat()
                    batch_record["generations_saved"] = len(generations)

                    print(f"Saved {len(generations)} generations from batch {batch_id}")

                    # Immediate cleanup
                    print(f"Cleaning up batch {batch_id}...")
                    try:
                        # Delete input and output files
                        await self.client.files.delete(batch.input_file_id)
                        if batch.output_file_id:
                            await self.client.files.delete(batch.output_file_id)
                        if batch.error_file_id:
                            await self.client.files.delete(batch.error_file_id)
                        print(f"Deleted files for batch {batch_id}")
                    except Exception as e:
                        print(f"Warning: Failed to delete files for batch {batch_id}: {e}")

                elif batch.status == "failed":
                    print(f"Batch {batch_id} failed!")
                    batches_failed += 1

                    # Print batch details for debugging
                    print(f"\nBatch Details:")
                    print(f"  Status: {batch.status}")
                    print(f"  Request counts: {batch.request_counts}")
                    if hasattr(batch, "errors") and batch.errors:
                        print(f"  Errors: {batch.errors}")

                    # Download and print error file if available
                    if batch.error_file_id:
                        try:
                            error_content = await self.client.files.content(batch.error_file_id)
                            error_file = self._get_batch_dir(storage) / f"error_{batch_id}.jsonl"
                            with open(error_file, "w", encoding="utf-8") as f:
                                f.write(error_content.text)
                            print(f"\nError details saved to: {error_file}")
                            print(f"Error file contents:")
                            print("-" * 80)
                            print(error_content.text)
                            print("-" * 80)
                        except Exception as e:
                            print(f"Could not download error file: {e}")

                    # Try to download output file if available (partial results)
                    if batch.output_file_id:
                        try:
                            output_content = await self.client.files.content(batch.output_file_id)
                            output_file = self._get_batch_dir(storage) / f"output_{batch_id}.jsonl"
                            with open(output_file, "w", encoding="utf-8") as f:
                                f.write(output_content.text)
                            print(f"\nPartial output saved to: {output_file}")
                            print(f"Partial output (first 1000 chars):")
                            print("-" * 80)
                            print(output_content.text[:1000])
                            print("-" * 80)
                        except Exception as e:
                            print(f"Could not download output file: {e}")

            except Exception as e:
                print(f"Error checking batch {batch_id}: {e}")

            updated_batches.append(batch_record)

        # Update metadata file
        self._update_batch_metadata(storage, updated_batches)

        return generations_saved, batches_completed, batches_failed

    async def consolidate_batches(self, storage: Storage) -> Dict[str, Any]:
        """Consolidate completed batches and save Generation objects.
        
        Args:
            storage: Storage instance for saving generations
            
        Returns:
            Status dict with completed/batches_completed/batches_failed
        """
        if not self.is_loaded:
            await self.load()
        
        generations_saved, batches_completed, batches_failed = (
            await self._consolidate_completed_batches(storage)
        )
        
        return {
            "completed": generations_saved,
            "batches_completed": batches_completed,
            "batches_failed": batches_failed,
        }

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
            Status dict with newly_enqueued/batches_active/pending_requests
        """
        if not self.is_loaded:
            await self.load()
        
        # Check active batches and load pending requests
        batches = self._load_batch_metadata(storage)
        active_batches = [
            b for b in batches if b["status"] in ["validating", "in_progress", "finalizing"]
        ]
        
        # Count requests in active batches to avoid duplicates
        pending_in_batches = {}  # (query_id, presupposition_level, sample_idx) -> True
        for batch in active_batches:
            # Load metadata file to see what's in this batch
            input_file = Path(batch['input_file'])
            metadata_file = input_file.parent / f"{input_file.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_map = json.load(f)
                
                for _, metadata in metadata_map.items():
                    key = (
                        metadata['query_id'],
                        metadata.get('presupposition_level'),
                        metadata.get('sample_idx', 0)
                    )
                    pending_in_batches[key] = True
        
        # Filter work queue to exclude already-pending requests
        filtered_work_queue = []
        skipped = 0
        for query, n_samples in work_queue:
            needed_samples = 0
            for sample_idx in range(n_samples):
                key = (query.id, query.presupposition_level, sample_idx)
                if key not in pending_in_batches:
                    needed_samples += 1
                else:
                    skipped += 1
            
            if needed_samples > 0:
                filtered_work_queue.append((query, needed_samples))
        
        if skipped > 0:
            print(f"Skipped {skipped} requests already in active batches")
        
        # Enqueue new batches if we have capacity
        newly_enqueued = 0
        if len(active_batches) < self.concurrent_batches and filtered_work_queue:
            available_slots = self.concurrent_batches - len(active_batches)
            print(f"Enqueueing up to {available_slots} new batch(es)...")
            
            remaining_work = filtered_work_queue.copy()
            for _ in range(available_slots):
                if not remaining_work:
                    break
                
                # Create batch input file
                input_file, request_count = await self._create_batch_input_file(
                    remaining_work, storage, **generation_kwargs
                )
                
                # Upload batch
                batch_id = await self._upload_batch(input_file)
                
                # Record metadata
                batch_record = {
                    "batch_id": batch_id,
                    "status": "validating",
                    "input_file": str(input_file),
                    "request_count": request_count,
                    "created_at": datetime.now().isoformat(),
                }
                self._save_batch_metadata(storage, batch_record)
                active_batches.append(batch_record)  # Update active list
                
                newly_enqueued += request_count
                
                # Remove processed items from remaining work
                processed = 0
                new_remaining = []
                for query, n_samples in remaining_work:
                    if processed >= request_count:
                        new_remaining.append((query, n_samples))
                    elif processed + n_samples <= request_count:
                        processed += n_samples
                    else:
                        # Partial sample
                        taken = request_count - processed
                        new_remaining.append((query, n_samples - taken))
                        processed = request_count
                
                remaining_work = new_remaining
        
        # Calculate updated pending requests (active batches + remaining work)
        pending_requests = sum(b["request_count"] for b in active_batches)
        
        return {
            "newly_enqueued": newly_enqueued,
            "batches_active": len(active_batches),
            "pending_requests": pending_requests,
        }