"""Async workers for queue-based generation and evaluation."""

import asyncio
import logging
from typing import List, Tuple, Optional
from datetime import datetime

from .models import Query, Generation, Evaluation
from .generators import Generator
from .evaluators import EntailmentModel
from .utils import generate_id

logger = logging.getLogger(__name__)


class GeneratorWorker:
    """Async worker for generating responses using a queue-based approach."""
    
    def __init__(
        self,
        generator: Generator,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        model_name: str,
        dataset_name: str,
        batch_size: Optional[int] = None,
    ):
        """Initialize the generator worker.
        
        Args:
            generator: The generator instance to use
            input_queue: Queue to pull (Query, n_samples_needed) tuples from
            output_queue: Queue to push Generation objects to
            model_name: Short model name (e.g., "gpt-oss-20b-medium")
            dataset_name: Dataset name (e.g., "uphill")
            batch_size: Batch size for generation (if None, uses generator's default)
        """
        self.generator = generator
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size or getattr(generator, 'batch_size', 32)
        self.running = False
    
    async def run(self):
        """Run the worker loop."""
        self.running = True
        logger.info(f"Generator worker starting with batch_size={self.batch_size}")
        
        try:
            # Load the generator
            await self.generator.load()
            
            while self.running:
                # Pull a batch of work from the queue
                batch = await self._pull_batch()
                
                if not batch:
                    # No more work, check if we should exit
                    if self.input_queue.empty():
                        logger.info("Queue empty and no batch pulled, exiting")
                        break
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
                    continue
                
                # Generate responses
                await self._process_batch(batch)
                
        except asyncio.CancelledError:
            logger.info("Generator worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in generator worker: {e}", exc_info=True)
            raise
        finally:
            # Unload the generator
            await self.generator.unload()
            logger.info("Generator worker stopped")
    
    async def _pull_batch(self) -> List[Tuple[Query, int]]:
        """Pull a batch of queries from the input queue.
        
        Returns:
            List of (Query, n_samples_needed) tuples
        """
        batch = []
        
        try:
            # Try to get batch_size items, but don't wait forever
            for _ in range(self.batch_size):
                try:
                    item = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    batch.append(item)
                    self.input_queue.task_done()
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            logger.error(f"Error pulling batch: {e}")
        
        return batch
    
    async def _process_batch(self, batch: List[Tuple[Query, int]]):
        """Process a batch of queries and generate responses.
        
        Args:
            batch: List of (Query, n_samples_needed) tuples
        """
        # Expand batch to handle multiple samples per query
        expanded_queries = []
        query_refs = []  # Track which query each sample belongs to
        
        for query, n_samples in batch:
            for _ in range(n_samples):
                expanded_queries.append(query.query_text)
                query_refs.append(query)
        
        if not expanded_queries:
            return
        
        logger.debug(f"Generating {len(expanded_queries)} responses")
        
        try:
            # Generate responses
            raw_responses = await self.generator.generate(expanded_queries)
            
            # Create Generation objects
            generations = []
            for response, query in zip(raw_responses, query_refs):
                # Skip None responses (errors/empty responses from generator)
                if response is None:
                    logger.warning(f"Skipping None response for query {query.id}")
                    continue
                
                # Handle both string and tuple responses
                if isinstance(response, tuple):
                    response_text, reasoning_trace = response
                else:
                    response_text = response
                    reasoning_trace = None
                
                # Skip empty responses
                if not response_text or (isinstance(response_text, str) and not response_text.strip()):
                    logger.warning(f"Skipping empty response for query {query.id}")
                    continue
                
                generation = Generation(
                    gen_id=generate_id(),
                    query_id=query.id,
                    dataset=self.dataset_name,
                    presupposition_level=query.presupposition_level,
                    gen_model=self.model_name,
                    response=response_text,
                    reasoning_trace=reasoning_trace,
                    timestamp=datetime.utcnow(),
                )
                generations.append(generation)
            
            # Push to output queue
            await self.output_queue.put(generations)
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}", exc_info=True)
            raise
    
    def stop(self):
        """Signal the worker to stop."""
        self.running = False


class EvaluatorWorker:
    """Async worker for evaluating responses using a queue-based approach."""
    
    def __init__(
        self,
        evaluator: EntailmentModel,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        model_name: str,
        batch_size: Optional[int] = None,
    ):
        """Initialize the evaluator worker.
        
        Args:
            evaluator: The evaluator instance to use
            input_queue: Queue to pull (Generation, claim) tuples from
            output_queue: Queue to push Evaluation objects to
            model_name: Short model name (e.g., "gpt-oss-20b-entailment")
            batch_size: Batch size for evaluation (if None, uses evaluator's default)
        """
        self.evaluator = evaluator
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.batch_size = batch_size or getattr(evaluator, 'batch_size', 64)
        self.running = False
    
    async def run(self):
        """Run the worker loop."""
        self.running = True
        logger.info(f"Evaluator worker starting with batch_size={self.batch_size}")
        
        try:
            # Load the evaluator
            await self.evaluator.load()
            
            while self.running:
                # Pull a batch of work from the queue
                batch = await self._pull_batch()
                
                if not batch:
                    # No more work, wait a bit before checking again
                    await asyncio.sleep(0.1)
                    continue
                
                # Evaluate responses
                await self._process_batch(batch)
                
        except asyncio.CancelledError:
            logger.info("Evaluator worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in evaluator worker: {e}", exc_info=True)
            # Push error to output queue so main thread knows we failed
            try:
                await self.output_queue.put([])
            except:
                pass
            raise
        finally:
            # Unload the evaluator
            await self.evaluator.unload()
            logger.info("Evaluator worker stopped")
    
    async def _pull_batch(self) -> List[Tuple[Generation, str]]:
        """Pull a batch of generations from the input queue.
        
        Returns:
            List of (Generation, claim) tuples
        """
        batch = []
        
        try:
            for _ in range(self.batch_size):
                try:
                    item = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    batch.append(item)
                    self.input_queue.task_done()
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            logger.error(f"Error pulling batch: {e}")
        
        return batch
    
    async def _process_batch(self, batch: List[Tuple[Generation, str]]):
        """Process a batch of generations and evaluate them.
        
        Args:
            batch: List of (Generation, claim) tuples
        """
        if not batch:
            return
        
        logger.debug(f"Evaluating {len(batch)} responses")
        
        try:
            # Extract claims and responses
            generations = [gen for gen, _ in batch]
            claims = [claim for _, claim in batch]
            responses = [gen.response for gen in generations]
            
            # Evaluate
            results = await self.evaluator.check_entailment(claims, responses)
            
            # Create Evaluation objects
            evaluations = []
            for generation, result in zip(generations, results):
                evaluation = Evaluation(
                    eval_id=generate_id(),
                    gen_id=generation.gen_id,
                    eval_model=self.model_name,
                    entailment=result.entailment,
                    reasoning=result.reasoning,
                    unsure=result.unsure,
                    timestamp=datetime.utcnow(),
                )
                evaluations.append(evaluation)
            
            # Push to output queue
            await self.output_queue.put(evaluations)
            
        except Exception as e:
            logger.error(f"Error evaluating responses: {e}", exc_info=True)
            # Push empty batch with error
            await self.output_queue.put([])
    
    def stop(self):
        """Signal the worker to stop."""
        self.running = False
