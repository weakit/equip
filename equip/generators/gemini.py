"""Gemini API generator using Google GenAI SDK."""

import asyncio
import logging
from typing import List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import Generator

logger = logging.getLogger(__name__)


class GeminiGenerator(Generator):
    """Generator using Google Gemini API with async support."""
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path  # For Gemini, this is the model name
        self.api_key = api_key
        self.thinking_budget = kwargs.get('thinking_budget')
        self.batch_size = kwargs.get('batch_size', 16)
        self.client = None
        self.genai = None
        self.types = None
    
    async def load(self):
        """Load the Gemini client."""
        if self.is_loaded:
            return
        
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError as e:
            raise ImportError(
                "GeminiGenerator requires google-genai package. "
                "Install with: pip install google-genai"
            ) from e
        
        # Initialize Gemini client
        client_kwargs = {}
        if self.api_key:
            client_kwargs['api_key'] = self.api_key
        
        # Try to load from environment if no API key provided
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        self.client = self.genai.Client(**client_kwargs)
        self.is_loaded = True
        logger.info(f"Loaded Gemini client for model: {self.model_path}")
    
    async def unload(self):
        """Unload the Gemini client."""
        if not self.is_loaded:
            return
        
        if self.client is not None:
            del self.client
            self.client = None
        
        self.is_loaded = False
        logger.info(f"Unloaded Gemini client for model: {self.model_path}")
    
    def _generate_single(
        self,
        query: str,
        thinking_budget: Optional[int],
        system_prompt: Optional[str],
        **generation_kwargs
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """Generate a single response using Gemini API.
        
        Args:
            query: Query text
            thinking_budget: Thinking budget (-1 for dynamic, None to disable, or positive int)
            system_prompt: System instruction
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Either response text or (response, thinking_trace) tuple
        """
        try:
            # Prepare the configuration
            config_kwargs = {
                'temperature': generation_kwargs.get('temperature', 1.0),
                'max_output_tokens': generation_kwargs.get('max_tokens', 8192),
            }
            
            # Add system instruction if provided
            if system_prompt:
                config_kwargs['system_instruction'] = system_prompt
            
            # Configure thinking if budget is specified
            if thinking_budget is not None:
                thinking_config = self.types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True
                )
                config_kwargs['thinking_config'] = thinking_config
            
            config = self.types.GenerateContentConfig(**config_kwargs)
            
            response = self.client.models.generate_content(
                model=self.model_path,
                contents=query,
                config=config
            )
            
            # Extract content and thoughts
            content = ""
            thoughts = ""
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            if hasattr(part, 'thought') and part.thought:
                                # This is a thought summary
                                thoughts += part.text
                            else:
                                # This is regular content
                                content += part.text
                elif hasattr(response, 'text') and response.text:
                    # Fallback for simpler response structure
                    content = response.text
            
            content = content.strip()
            thoughts = thoughts.strip()
            
            # Handle empty responses - return None to indicate failure
            if not content and not thoughts:
                logger.warning(f"Empty response for query: {query[:50]}...")
                return None
            
            # Return tuple if we have thoughts, otherwise just content
            if thoughts:
                return (content, thoughts)
            else:
                return content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if hasattr(e, 'message'):
                logger.error(f"Error details: {e.message}")
            return None
    
    async def generate(
        self, queries: List[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str, Optional[str]]]]:
        """Generate responses using Gemini API with batch processing support.
        
        Args:
            queries: List of query strings
            **generation_kwargs: Generation parameters including:
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens
                - thinking_budget: Thinking budget for reasoning
                - system_prompt: System instruction
                - batch_size: Number of concurrent requests
                
        Returns:
            List of responses (strings or tuples with thinking traces)
        """
        if not self.is_loaded or self.client is None:
            raise RuntimeError("Client not loaded. Call load() before generate()")
        
        # Merge stored kwargs with function call kwargs
        merged_kwargs = self._merge_kwargs(**generation_kwargs)
        
        # Extract parameters
        thinking_budget = merged_kwargs.get('thinking_budget', self.thinking_budget)
        system_prompt = merged_kwargs.get('system_prompt')
        max_workers = merged_kwargs.get('batch_size', self.batch_size)
        
        logger.debug(
            f"Generating {len(queries)} responses with "
            f"thinking_budget={thinking_budget}, batch_size={max_workers}"
        )
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            self._generate_batch,
            queries,
            thinking_budget,
            system_prompt,
            max_workers,
            merged_kwargs
        )
        
        return responses
    
    def _generate_batch(
        self,
        queries: List[str],
        thinking_budget: Optional[int],
        system_prompt: Optional[str],
        max_workers: int,
        merged_kwargs: dict
    ) -> List[Union[str, Tuple[str, Optional[str]]]]:
        """Generate batch of responses in thread pool.
        
        Args:
            queries: List of queries
            thinking_budget: Thinking budget
            system_prompt: System instruction
            max_workers: Number of concurrent workers
            merged_kwargs: Merged generation kwargs
            
        Returns:
            List of responses in same order as queries
        """
        # Remove parameters that are passed explicitly to avoid conflicts
        gen_kwargs = merged_kwargs.copy()
        gen_kwargs.pop('thinking_budget', None)
        gen_kwargs.pop('system_prompt', None)
        gen_kwargs.pop('batch_size', None)
        
        # If only one query or batch_size is 1, use sequential processing
        if len(queries) == 1 or max_workers == 1:
            responses = []
            for query in queries:
                response = self._generate_single(
                    query, thinking_budget, system_prompt, **gen_kwargs
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
                    thinking_budget,
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
                    print(e)
                    responses[idx] = None
        
        return responses
