"""VLLM generator with configurable preprocessing and postprocessing."""

import asyncio
import logging
from typing import List, Union, Tuple, Optional
import random
from .base import Generator
from ..processors import Preprocessor, Postprocessor, StandardPreprocessor, StandardPostprocessor

logger = logging.getLogger(__name__)


class VLLMGenerator(Generator):
    """Unified VLLM generator with configurable pre/post processors."""

    def __init__(
        self,
        model_path: str,
        preprocessor: Optional[Preprocessor] = None,
        postprocessor: Optional[Postprocessor] = None,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.batch_size = kwargs.get("batch_size", 128)

        # Use standard processors by default
        self.preprocessor = preprocessor or StandardPreprocessor()
        self.postprocessor = postprocessor or StandardPostprocessor()
        self.llm = None

    async def load(self):
        """Load the model and initialize resources."""
        if self.is_loaded:
            return

        try:
            from vllm import LLM
            import torch
        except ImportError as e:
            raise ImportError("VLLMGenerator requires vllm package") from e

        logger.info(f"Loading vLLM model: {self.model_path}")

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self.is_loaded = True
        logger.info(f"vLLM model loaded successfully: {self.model_path}")

    def _load_model(self):
        """Synchronous model loading (called in thread pool)."""
        from vllm import LLM

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.kwargs.get("tensor_parallel_size", 1),
            data_parallel_size=self.kwargs.get("data_parallel_size", 1),
            pipeline_parallel_size=self.kwargs.get("pipeline_parallel_size", 1),
            enable_prefix_caching=self.kwargs.get("enable_prefix_caching", True),
            gpu_memory_utilization=self.kwargs.get("gpu_memory_utilization", 0.9),
            enable_expert_parallel=self.kwargs.get("enable_expert_parallel", False),
        )

    async def unload(self):
        """Unload the model and free resources."""
        if not self.is_loaded:
            return

        if self.llm is not None:
            logger.info(f"Unloading vLLM model: {self.model_path}")

            # Run cleanup in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._unload_model)

            self.llm = None

        self.is_loaded = False
        logger.info(f"vLLM model unloaded: {self.model_path}")

    def _unload_model(self):
        """Synchronous model unloading (called in thread pool)."""
        import torch
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        torch.cuda.empty_cache()

    async def generate(
        self, queries: List[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str, Optional[str]]]]:
        """Generate responses using vLLM with configurable processors."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Merge stored kwargs with function call kwargs
        merged_kwargs = self._merge_kwargs(**generation_kwargs)

        logger.debug(f"Generating {len(queries)} responses with merged_kwargs: {merged_kwargs}")

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(None, self._generate_sync, queries, merged_kwargs)

        return responses

    def _generate_sync(self, queries: List[str], merged_kwargs: dict):
        """Synchronous generation (called in thread pool)."""
        from vllm import SamplingParams

        # Add model_path to kwargs for preprocessor (e.g., Qwen needs it for tokenizer)
        merged_kwargs["model_path"] = self.model_path

        # Preprocess inputs
        processed_inputs, updated_kwargs = self.preprocessor.process(queries, **merged_kwargs)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=updated_kwargs.get("temperature", 1.0),
            max_tokens=updated_kwargs.get("max_tokens", 8192),
            stop=updated_kwargs.get("stop", None),
            stop_token_ids=updated_kwargs.get("stop_token_ids", None),
            seed=updated_kwargs.get("seed", random.randint(0, 2**32 - 1)),
        )

        # Generate responses
        if isinstance(processed_inputs[0], list) and isinstance(processed_inputs[0][0], dict):
            # Standard chat format
            outputs = self.llm.chat(
                processed_inputs, sampling_params=sampling_params, use_tqdm=True
            )
        else:
            # Tokenized format
            outputs = self.llm.generate(
                prompts=[{"prompt_token_ids": inp} for inp in processed_inputs],
                sampling_params=sampling_params,
                use_tqdm=True,
            )

        # Pass model_path to postprocessor if it supports it (e.g., QwenPostprocessor)
        if hasattr(self.postprocessor, "load_tokenizer"):
            self.postprocessor.load_tokenizer(self.model_path)

        # Postprocess outputs
        responses = self.postprocessor.process(outputs, queries)

        return responses
