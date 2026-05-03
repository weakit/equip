"""VLLM Online generator using OpenAI-compatible API."""

import asyncio
import logging
from typing import List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import Generator
from ..processors import Preprocessor, Postprocessor, StandardPreprocessor, StandardPostprocessor

logger = logging.getLogger(__name__)


class VLLMOnlineGenerator(Generator):
    """Generator using vLLM server with OpenAI-compatible API."""
    
    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        api_key: Optional[str] = "EMPTY",
        base_url: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
        postprocessor: Optional[Postprocessor] = None,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.port = port
        self.api_key = api_key
        self.base_url = base_url or f"http://localhost:{port}/v1"
        self.batch_size = kwargs.get('batch_size', 16)
        
        # Use standard processors by default
        self.preprocessor = preprocessor or StandardPreprocessor()
        self.postprocessor = postprocessor or StandardPostprocessor()
        
        self.client = None
    
    async def load(self):
        """Load the OpenAI client."""
        if self.is_loaded:
            return
        
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "VLLMOnlineGenerator requires openai package. "
                "Install with: pip install openai"
            ) from e
        
        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.is_loaded = True
        logger.info(f"Loaded vLLM Online client for model: {self.model_path} at {self.base_url}")
    
    async def unload(self):
        """Unload the OpenAI client."""
        if not self.is_loaded:
            return
        
        if self.client is not None:
            del self.client
            self.client = None
        
        self.is_loaded = False
        logger.info(f"Unloaded vLLM Online client for model: {self.model_path}")
    
    def _generate_single(
        self,
        query: str,
        system_prompt: Optional[str],
        **generation_kwargs
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """Generate a single response using vLLM OpenAI-compatible API.
        
        Args:
            query: Query text
            system_prompt: System instruction
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Either response text or (response, thinking_trace) tuple
        """
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            # Extract generation parameters
            completion_kwargs = {
                "model": self.model_path,
                "messages": messages,
                "temperature": generation_kwargs.get('temperature', 1.0),
                "max_tokens": generation_kwargs.get('max_tokens', 8192),
            }
            
            # Add optional parameters if present
            if 'stop' in generation_kwargs and generation_kwargs['stop']:
                completion_kwargs['stop'] = generation_kwargs['stop']
            
            if 'seed' in generation_kwargs:
                completion_kwargs['seed'] = generation_kwargs['seed']
            
            # Call the API
            response = self.client.chat.completions.create(**completion_kwargs)
            
            # print(response.choices[0].message)

            # Extract content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                # print(response.choices[0].message)
                
                if content:
                    content = content.strip()
                    
                    # Check if this is a thinking model response
                    # Thinking models often have <think> tags or similar markers
                    thinking_trace = None
                    if '<think>' in content and '</think>' in content:
                        # Extract thinking trace
                        think_start = content.find('<think>')
                        think_end = content.find('</think>') + len('</think>')
                        thinking_trace = content[think_start+7:think_end-8].strip()
                        # Remove thinking from content
                        content = content[:think_start] + content[think_end:]
                        content = content.strip()
                    
                    # Return tuple if we have thinking, otherwise just content
                    if thinking_trace:
                        return (content, thinking_trace)
                    else:
                        return content
            
            # Handle empty responses
            logger.warning(f"Empty response for query: {query[:50]}...")
            return None
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if hasattr(e, 'message'):
                logger.error(f"Error details: {e.message}")
            return None
    
    async def generate(
        self, queries: List[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str, Optional[str]]]]:
        """Generate responses using vLLM OpenAI-compatible API with batch processing.
        
        Args:
            queries: List of query strings
            **generation_kwargs: Generation parameters including:
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens
                - system_prompt: System instruction
                - batch_size: Number of concurrent requests
                - stop: Stop sequences
                - seed: Random seed
                
        Returns:
            List of responses (strings or tuples with thinking traces)
        """
        if not self.is_loaded or self.client is None:
            raise RuntimeError("Client not loaded. Call load() before generate()")
        
        # Merge stored kwargs with function call kwargs
        merged_kwargs = self._merge_kwargs(**generation_kwargs)
        
        # Extract parameters
        system_prompt = merged_kwargs.get('system_prompt')
        max_workers = merged_kwargs.get('batch_size', self.batch_size)
        
        logger.debug(
            f"Generating {len(queries)} responses with "
            f"batch_size={max_workers}"
        )
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            self._generate_batch,
            queries,
            system_prompt,
            max_workers,
            merged_kwargs
        )
        
        return responses
    
    def _generate_batch(
        self,
        queries: List[str],
        system_prompt: Optional[str],
        max_workers: int,
        merged_kwargs: dict
    ) -> List[Union[str, Tuple[str, Optional[str]]]]:
        """Generate batch of responses in thread pool.
        
        Args:
            queries: List of queries
            system_prompt: System instruction
            max_workers: Number of concurrent workers
            merged_kwargs: Merged generation kwargs
            
        Returns:
            List of responses in same order as queries
        """
        # Remove parameters that are passed explicitly to avoid conflicts
        gen_kwargs = merged_kwargs.copy()
        gen_kwargs.pop('system_prompt', None)
        gen_kwargs.pop('batch_size', None)
        
        # If only one query or batch_size is 1, use sequential processing
        if len(queries) == 1 or max_workers == 1:
            responses = []
            for query in queries:
                response = self._generate_single(
                    query, system_prompt, **gen_kwargs
                )
                responses.append(response)
            return responses
        
        # Use ThreadPoolExecutor for parallel processing
        responses = [None] * len(queries)  # Pre-allocate to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._generate_single,
                    query,
                    system_prompt,
                    **gen_kwargs
                ): idx
                for idx, query in enumerate(queries)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    responses[idx] = result
                except Exception as e:
                    logger.error(f"Batch processing error for query {idx}: {e}")
                    responses[idx] = None
        
        return responses
