"""Generate command - Generate responses for queries."""

import asyncio
import logging
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from ..config import ModelConfig
from ..loaders import get_dataset_loader
from ..storage import Storage
from ..workers import GeneratorWorker
from ..generators import BatchedGenerator

logger = logging.getLogger(__name__)


async def generate(
    model: str,
    dataset: str,
    n_samples: int = 1,
    run_prefix: str = "exp-1",
    config_path: Optional[str] = None,
    **kwargs,
):
    """Generate responses for a dataset.
    
    Args:
        model: Generator model name
        dataset: Dataset name
        n_samples: Number of samples to generate per query
        run_prefix: Run identifier
        config_path: Path to models.yaml
        **kwargs: Additional parameters to override model config
    """
    logger.info(f"Starting generation: model={model}, dataset={dataset}, n_samples={n_samples}")
    
    # Load configuration
    config = ModelConfig(config_path)
    
    # Load dataset
    loader = get_dataset_loader(dataset)
    queries = loader.load()
    logger.info(f"Loaded {len(queries)} queries")
    
    # Initialize storage
    storage = Storage(run_prefix, dataset, model)
    
    # Create generator
    generator = config.create_generator(model, storage_base_dir=storage.base_dir, **kwargs)
    
    # Total generations expected across all queries
    total_expected = len(queries) * n_samples
    
    # Check if this is a batched generator
    if isinstance(generator, BatchedGenerator):
        # Batched mode: consolidate first, then build work queue
        await _generate_batched(
            generator=generator,
            queries=queries,
            storage=storage,
            n_samples=n_samples,
            total_expected=total_expected,
            **kwargs
        )
    else:
        # Immediate mode: build work queue once, then process
        existing_counts = storage.get_generation_count_by_query()
        logger.info(f"Found {len(existing_counts)} query-level combinations with existing generations")
        
        work_queue_items = []
        total_needed = 0
        total_existing = 0
        
        for query in queries:
            key = (query.id, query.presupposition_level)
            existing_count = existing_counts.get(key, 0)
            needed = max(0, n_samples - existing_count)
            
            total_existing += existing_count
            
            if needed > 0:
                work_queue_items.append((query, needed))
                total_needed += needed
        
        if total_needed == 0:
            logger.info("All queries already have the required number of generations")
            print(f"✓ All queries already have {n_samples} generation(s)")
            return
        
        logger.info(f"Need to generate {total_needed} new responses for {len(work_queue_items)} queries")
        logger.info(f"Already have {total_existing} generations, targeting {total_expected} total")
        print(f"Generating {total_needed} responses for {len(work_queue_items)} queries...")
        print(f"Progress: {total_existing}/{total_expected} already complete")
        
        await _generate_immediate(
            generator=generator,
            work_queue=work_queue_items,
            storage=storage,
            model=model,
            dataset=dataset,
            total_needed=total_needed,
            total_existing=total_existing,
            total_expected=total_expected,
            **kwargs
        )


async def _generate_batched(
    generator: BatchedGenerator,
    queries,
    storage: Storage,
    n_samples: int,
    total_expected: int,
    **kwargs
):
    """Handle batched generation (e.g., OpenAI Batch API).
    
    Batched generators manage long-running jobs across multiple runs.
    For each run:
    1. Consolidate any completed batches
    2. Rebuild work queue based on current state
    3. Enqueue new batches if capacity allows
    """
    logger.info("Using batched generation mode")
    
    try:
        # Load generator
        await generator.load()
        
        # Step 1: Consolidate completed batches first
        print("\n=== Consolidating completed batches ===")
        status = await generator.consolidate_batches(storage)
        
        completed = status.get("completed", 0)
        batches_completed = status.get("batches_completed", 0)
        batches_failed = status.get("batches_failed", 0)
        
        if batches_completed > 0:
            print(f"Consolidated {batches_completed} batch(es), saved {completed} generations")
        
        if batches_failed > 0:
            print(f"⚠ {batches_failed} batch(es) failed")
        
        # Step 2: Check what's actually needed now (after consolidation)
        existing_counts = storage.get_generation_count_by_query()
        work_queue = []
        total_needed = 0
        total_existing = 0
        
        for query in queries:
            key = (query.id, query.presupposition_level)
            existing_count = existing_counts.get(key, 0)
            needed = max(0, n_samples - existing_count)
            
            total_existing += existing_count
            
            if needed > 0:
                work_queue.append((query, needed))
                total_needed += needed
        
        print(f"\n=== Current Status ===")
        print(f"Progress: {total_existing}/{total_expected} ({total_existing * 100 / total_expected:.1f}%)")
        
        if total_needed == 0:
            print(f"✓ All generations complete!")
            return
        
        print(f"Still need: {total_needed} generations for {len(work_queue)} queries")
        
        # Step 3: Enqueue new batches
        enqueue_status = await generator.enqueue_batches(work_queue, storage, **kwargs)
        
        newly_enqueued = enqueue_status.get("newly_enqueued", 0)
        batches_active = enqueue_status.get("batches_active", 0)
        pending_requests = enqueue_status.get("pending_requests", 0)
        
        print(f"\n=== Batch Queue ===")
        print(f"Active batches: {batches_active}")
        print(f"Pending requests: {pending_requests}")
        print(f"Newly enqueued: {newly_enqueued}")
        
        if batches_completed > 0:
            print(f"\n✓ Saved to: {storage.generations_file}")
        
        if pending_requests > 0 or newly_enqueued > 0:
            print(f"\nℹ Run this command again to check for completed batches")
        
    finally:
        await generator.unload()


async def _generate_immediate(
    generator,
    work_queue,
    storage: Storage,
    model: str,
    dataset: str,
    total_needed: int,
    total_existing: int,
    total_expected: int,
    **kwargs
):
    """Handle immediate generation with worker."""
    logger.info("Using immediate generation mode")
    
    # Create queues
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    # Populate input queue
    for item in work_queue:
        await input_queue.put(item)
    
    # Create and start worker (pass model short name, not path)
    worker = GeneratorWorker(generator, input_queue, output_queue, model, dataset)
    worker_task = asyncio.create_task(worker.run())
    
    # Collect results with progress bar
    all_generations = []
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            # Progress shows total expected, starting from existing count
            task = progress.add_task("Generating...", total=total_expected, completed=total_existing)
            
            while len(all_generations) < total_needed:
                # Check if worker task has completed or failed
                if worker_task.done():
                    try:
                        worker_task.result()  # This will raise if there was an exception
                        # Worker completed successfully
                        logger.info("Worker completed")
                        break
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                        print(f"\n✗ Generation failed: {e}")
                        raise
                
                # Wait for a batch of results
                try:
                    generations = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                    
                    if generations:
                        all_generations.extend(generations)
                        
                        # Save incrementally: append only the NEW batch
                        # Load existing, append new batch, deduplicate by gen_id
                        existing_gens = storage.load_generations()
                        
                        # Deduplicate by gen_id (newer generations override older ones)
                        gen_dict = {g.gen_id: g for g in existing_gens}
                        for g in generations:
                            gen_dict[g.gen_id] = g
                        
                        deduplicated = list(gen_dict.values())
                        storage.save_generations(deduplicated)
                        
                        # Update progress: existing + newly generated
                        progress.update(task, completed=total_existing + len(all_generations))
                    
                except asyncio.TimeoutError:
                    # Timeout is OK, just check worker status and continue
                    continue
            
    finally:
        # Stop worker
        worker.running = False
        await input_queue.join()
        
        # Cancel worker task
        if not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        else:
            # Check if it failed
            try:
                worker_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Worker task had exception: {e}")
                # Don't re-raise here as we've already handled it
    
    logger.info(f"Generation complete. Generated {len(all_generations)} new responses")
    print(f"✓ Generated {len(all_generations)} responses")
    print(f"  Saved to: {storage.generations_file}")

