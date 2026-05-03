"""SGLang generator with configurable preprocessing and postprocessing."""

import asyncio
import logging
from typing import List, Union, Tuple, Optional
from .base import Generator
from ..processors import Preprocessor, Postprocessor, StandardPreprocessor, StandardPostprocessor

logger = logging.getLogger(__name__)


class SGLangGenerator(Generator):
    """Unified SGLang generator with configurable pre/post processors."""

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
        self.engine = None
        self.tokenizer = None

    async def load(self):
        """Load the model and initialize resources."""
        if self.is_loaded:
            return

        try:
            from sglang import Engine
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("SGLangGenerator requires sglang package") from e

        logger.info(f"Loading SGLang model: {self.model_path}")
        
        # Load engine and tokenizer
        await self._load_model()
        
        self.is_loaded = True
        logger.info(f"SGLang model loaded successfully: {self.model_path}")
    
    async def _load_model(self):
        """Load the engine asynchronously."""
        from sglang import Engine
        from transformers import AutoTokenizer
        
        # Initialize engine with async support
        self.engine = Engine(
            model_path=self.model_path,
            tp_size=self.kwargs.get("tensor_parallel_size", 1),
            dp_size=self.kwargs.get("data_parallel_size", 1),
            enable_p2p_check=self.kwargs.get("enable_p2p_check", False),
            log_level=self.kwargs.get("log_level", "warning"),
            mem_fraction_static=self.kwargs.get("mem_fraction_static", 0.85)
        )
        
        # Load tokenizer for chat template processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    async def unload(self):
        """Unload the model and free resources."""
        if not self.is_loaded:
            return

        if self.engine is not None:
            logger.info(f"Unloading SGLang model: {self.model_path}")
            
            # Shutdown engine
            self.engine.shutdown()
            self.engine = None
            self.tokenizer = None

        self.is_loaded = False
        logger.info(f"SGLang model unloaded: {self.model_path}")

    async def generate(
        self, queries: List[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str, Optional[str]]]]:
        """Generate responses using SGLang with configurable processors."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Merge stored kwargs with function call kwargs
        merged_kwargs = self._merge_kwargs(**generation_kwargs)

        logger.debug(f"Generating {len(queries)} responses with merged_kwargs: {merged_kwargs}")

        # Preprocess inputs
        processed_inputs, updated_kwargs = self.preprocessor.process(
            queries, **merged_kwargs
        )

        # Generate responses
        responses = await self._generate_async(processed_inputs, updated_kwargs)
        
        return responses
    
    async def _generate_async(self, processed_inputs: List, updated_kwargs: dict):
        """Asynchronous generation using SGLang."""
        
        # Set up sampling parameters as a dict (not a class)
        sampling_params = {
            "temperature": updated_kwargs.get("temperature", 1.0),
            "max_new_tokens": updated_kwargs.get("max_tokens", 8192),
        }
        
        # Add optional parameters if provided
        if "stop" in updated_kwargs and updated_kwargs["stop"]:
            sampling_params["stop"] = updated_kwargs["stop"]
        if "stop_token_ids" in updated_kwargs and updated_kwargs["stop_token_ids"]:
            sampling_params["stop_token_ids"] = updated_kwargs["stop_token_ids"]
        
        # Check if separate reasoning is enabled
        enable_reasoning = updated_kwargs.get("enable_reasoning", False)
        if enable_reasoning:
            sampling_params["enable_reasoning"] = True
        
        # Handle different input formats from preprocessor
        prompts = None
        input_ids = None
        
        if isinstance(processed_inputs[0], list):
            if isinstance(processed_inputs[0][0], dict):
                # Chat messages format - tokenize using chat template
                prompts = []
                for messages in processed_inputs:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(prompt)
            elif isinstance(processed_inputs[0][0], int):
                # Already tokenized - use input_ids directly
                input_ids = processed_inputs
        else:
            # Plain text prompts
            prompts = processed_inputs

        # Generate responses using async batch interface
        outputs = await self.engine.async_generate(
            prompt=prompts,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )

        # Process outputs - SGLang returns dict with 'text' key
        responses = []
        for output in outputs:
            if enable_reasoning and isinstance(output, dict) and "meta_info" in output and "reasoning_content" in output["meta_info"]:
                # Extract reasoning trace
                reasoning = output["meta_info"]["reasoning_content"]
                response_text = output["text"]
                responses.append((response_text, reasoning))
            else:
                # Output is a dict with 'text' key
                responses.append(output["text"] if isinstance(output, dict) else output)

        # Postprocess outputs using the original queries
        # Reconstruct original queries if needed
        if isinstance(processed_inputs[0], list):
            original_queries = [str(inp) for inp in processed_inputs]
        else:
            original_queries = processed_inputs
        
        final_responses = self.postprocessor.process(responses, original_queries)

        return final_responses
