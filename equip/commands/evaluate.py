"""Evaluate command - Evaluate generated responses."""

import asyncio
import logging
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from ..config import ModelConfig
from ..loaders import get_dataset_loader
from ..storage import Storage
from ..workers import EvaluatorWorker

logger = logging.getLogger(__name__)


async def evaluate(
    generator_model: str,
    evaluator_model: Optional[str],
    dataset: str,
    run_prefix: str = "exp-1",
    config_path: Optional[str] = None,
    max_retries: int = 10,
    **kwargs,
):
    """Evaluate generated responses with automatic retry of failed evaluations.
    
    Args:
        generator_model: Generator model name
        evaluator_model: Evaluator model name (if None, use first evaluator from config)
        dataset: Dataset name
        run_prefix: Run identifier
        config_path: Path to models.yaml
        max_retries: Maximum number of retry attempts for failed evaluations (default: 10)
        **kwargs: Additional parameters to override evaluator config
    """
    # Load configuration
    config = ModelConfig(config_path)
    
    # If evaluator_model is None, pick the first evaluator
    if evaluator_model is None:
        evaluators = config.list_evaluators()
        if not evaluators:
            logger.error("No evaluator models found in config")
            print("✗ No evaluator models configured")
            return
        evaluator_model = evaluators[0]
        logger.info(f"No evaluator specified, using first evaluator: {evaluator_model}")
        print(f"Using evaluator: {evaluator_model}")
    
    logger.info(
        f"Starting evaluation: generator={generator_model}, "
        f"evaluator={evaluator_model}, dataset={dataset}"
    )
    
    # Load dataset (need it for claims)
    loader = get_dataset_loader(dataset)
    queries = loader.load()
    
    # Create query lookup by (id, presupposition_level)
    query_lookup = {(q.id, q.presupposition_level): q for q in queries}
    logger.info(f"Loaded {len(queries)} queries")
    
    # Initialize storage
    storage = Storage(run_prefix, dataset, generator_model)
    
    # Load generations
    generations = storage.load_generations()
    if not generations:
        logger.error("No generations found to evaluate")
        print(f"✗ No generations found for {generator_model} on {dataset}")
        print(f"  Run generate command first")
        return
    
    logger.info(f"Found {len(generations)} generations")
    
    total_evaluations = len(generations)
    
    # Retry loop for failed evaluations
    for retry_attempt in range(max_retries + 1):
        # Load existing evaluations (excluding errors to allow retry)
        successfully_evaluated_ids = storage.get_successfully_evaluated_generation_ids(evaluator_model)
        
        # Find unevaluated generations (including previously errored ones)
        unevaluated = [g for g in generations if g.gen_id not in successfully_evaluated_ids]
        
        total_existing = len(successfully_evaluated_ids)
        
        if not unevaluated:
            logger.info("All generations successfully evaluated")
            print(f"✓ All generations successfully evaluated by {evaluator_model}")
            
            # Print final statistics
            if retry_attempt > 0:
                print(f"  Completed after {retry_attempt} retry attempt(s)")
            
            return
        
        # Determine if this is initial run or retry
        if retry_attempt == 0:
            logger.info(f"Need to evaluate {len(unevaluated)} generations")
            logger.info(f"Already have {total_existing} successful evaluations, targeting {total_evaluations} total")
            print(f"Evaluating {len(unevaluated)} responses...")
            print(f"Progress: {total_existing}/{total_evaluations} successfully completed")
        else:
            # This is a retry attempt
            logger.info(f"Retry attempt {retry_attempt}/{max_retries}: Re-evaluating {len(unevaluated)} failed responses")
            print(f"\n⟳ Retry {retry_attempt}/{max_retries}: Re-evaluating {len(unevaluated)} failed responses...")
        
        # Run evaluation for this attempt
        success = await _run_evaluation_pass(
            unevaluated=unevaluated,
            query_lookup=query_lookup,
            storage=storage,
            config=config,
            evaluator_model=evaluator_model,
            total_evaluations=total_evaluations,
            total_existing=total_existing,
            **kwargs
        )
        
        if not success:
            logger.error("Evaluation pass failed")
            print(f"\n✗ Evaluation pass failed")
            return
    
    # Max retries exhausted - report final status
    final_successful = storage.get_successfully_evaluated_generation_ids(evaluator_model)
    final_errors = total_evaluations - len(final_successful)
    
    logger.warning(f"Max retries ({max_retries}) exhausted. {final_errors} evaluations still have errors")
    print(f"\n⚠ Max retries ({max_retries}) exhausted")
    print(f"  Successfully evaluated: {len(final_successful)}/{total_evaluations}")
    print(f"  Still have errors: {final_errors}")
    
    eval_file = storage.get_evaluation_file(evaluator_model)
    print(f"  Saved to: {eval_file}")


async def _run_evaluation_pass(
    unevaluated,
    query_lookup,
    storage,
    config,
    evaluator_model,
    total_evaluations,
    total_existing,
    **kwargs
):
    """Run a single evaluation pass.
    
    Args:
        unevaluated: List of unevaluated Generation objects
        query_lookup: Dict mapping (query_id, presupposition_level) to Query
        storage: Storage instance
        config: ModelConfig instance
        evaluator_model: Evaluator model name
        total_evaluations: Total number of evaluations expected
        total_existing: Number of existing successful evaluations
        **kwargs: Additional evaluator parameters
        
    Returns:
        bool: True if pass completed successfully, False if failed
    """
    """Run a single evaluation pass.
    
    Args:
        unevaluated: List of unevaluated Generation objects
        query_lookup: Dict mapping (query_id, presupposition_level) to Query
        storage: Storage instance
        config: ModelConfig instance
        evaluator_model: Evaluator model name
        total_evaluations: Total number of evaluations expected
        total_existing: Number of existing successful evaluations
        **kwargs: Additional evaluator parameters
        
    Returns:
        bool: True if pass completed successfully, False if failed
    """
    # Prepare work items (Generation, claim)
    work_items = []
    for gen in unevaluated:
        key = (gen.query_id, gen.presupposition_level)
        query = query_lookup.get(key)
        
        if query is None:
            logger.warning(f"Could not find query for generation {gen.gen_id}")
            continue
        
        work_items.append((gen, query.claim))
    
    if not work_items:
        logger.error("No valid work items after lookup")
        print("✗ Could not match generations to queries")
        return False
    
    # Create evaluator
    evaluator = config.create_evaluator(evaluator_model, **kwargs)
    
    # Create queues
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    # Populate input queue
    for item in work_items:
        await input_queue.put(item)
    
    # Create and start worker (pass model short name, not path)
    worker = EvaluatorWorker(evaluator, input_queue, output_queue, evaluator_model)
    worker_task = asyncio.create_task(worker.run())
    
    # Collect results with progress bar
    all_evaluations = []
    
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
            task = progress.add_task("Evaluating...", total=total_evaluations, completed=total_existing)
            
            while len(all_evaluations) < len(work_items):
                # Check if worker task has failed or completed
                if worker_task.done():
                    try:
                        worker_task.result()  # This will raise if there was an exception
                        # Worker completed successfully
                        logger.info("Worker completed")
                        break
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                        print(f"\n✗ Evaluation failed: {e}")
                        return False
                
                # Wait for a batch of results
                try:
                    evaluations = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                    
                    if evaluations:
                        all_evaluations.extend(evaluations)
                        
                        # Save incrementally: append only the NEW batch
                        # Load existing, append new batch, deduplicate by gen_id (keep most recent)
                        existing_evals = storage.load_evaluations(evaluator_model)
                        
                        # Combine and deduplicate: newer evaluations override older ones for same gen_id
                        eval_dict = {e.gen_id: e for e in existing_evals}
                        for e in evaluations:
                            eval_dict[e.gen_id] = e  # Override if gen_id exists
                        
                        deduplicated = list(eval_dict.values())
                        storage.save_evaluations(evaluator_model, deduplicated)
                        
                        # Update progress: existing + newly evaluated
                        progress.update(task, completed=total_existing + len(all_evaluations))
                    
                except asyncio.TimeoutError:
                    # Timeout is OK, just check worker status and continue
                    continue
            
    finally:
        # Stop worker
        worker.stop()
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
                return False
    
    eval_file = storage.get_evaluation_file(evaluator_model)
    logger.info(f"Evaluation pass complete. Evaluated {len(all_evaluations)} responses")
    
    return True
