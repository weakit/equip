"""Qwen3 preprocessing and postprocessing with thinking mode support."""

from typing import List, Dict, Any, Optional, Tuple
from .base import Preprocessor, Postprocessor


class QwenPreprocessor(Preprocessor):
    """Qwen3 preprocessor with thinking mode control.

    Supports switching between thinking and non-thinking modes using the
    enable_thinking parameter in apply_chat_template.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.tokenizer = None

    def load_tokenizer(self, model_path: str):
        """Load tokenizer for the model."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def process(self, queries: List[str], **generation_kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess queries into Qwen3 chat format with thinking mode control.

        Args:
            queries: List of query strings
            **generation_kwargs: Generation parameters, including:
                - system_prompt: System prompt (default: "You are a helpful assistant.")
                - enable_thinking: Enable thinking mode (default: from config or True)
                - model_path: Path to model for tokenizer loading

        Returns:
            Tuple of (tokenized_inputs, updated_kwargs)
        """
        # Get parameters
        system_prompt = generation_kwargs.get("system_prompt", "You are a helpful assistant.")
        enable_thinking = generation_kwargs.get(
            "enable_thinking", self.config.get("enable_thinking", True)
        )
        model_path = generation_kwargs.get("model_path")

        # Load tokenizer if needed
        if model_path:
            self.load_tokenizer(model_path)

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Pass model_path in generation_kwargs.")

        # Prepare chat messages
        tokenized_inputs = []

        for query in queries:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": query})

            # Apply chat template with enable_thinking parameter
            # Tokenize directly instead of rendering to string
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors=None,  # Return list, not tensor
            )

            tokenized_inputs.append(tokens)

        # Update generation kwargs
        updated_kwargs = generation_kwargs.copy()

        # Set temperature based on thinking mode (as recommended in docs)
        if "temperature" not in updated_kwargs:
            if enable_thinking:
                updated_kwargs["temperature"] = 0.6
            else:
                updated_kwargs["temperature"] = 0.7

        # Set top_p based on thinking mode
        if "top_p" not in updated_kwargs:
            if enable_thinking:
                updated_kwargs["top_p"] = 0.95
            else:
                updated_kwargs["top_p"] = 0.8

        # Set top_k if not specified (same for both modes)
        if "top_k" not in updated_kwargs:
            updated_kwargs["top_k"] = 20

        return tokenized_inputs, updated_kwargs


class QwenPostprocessor(Postprocessor):
    """Qwen3 postprocessor with thinking content extraction.

    Extracts thinking content from <think>...</think> tags and separates it
    from the final response.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.tokenizer = None
        # Special token ID for </think> is 151668 according to docs
        self.think_end_token_id = 151668

    def load_tokenizer(self, model_path: str):
        """Load tokenizer for the model."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def process(
        self, outputs: List[Any], original_queries: List[str]
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract text and thinking content from Qwen3 outputs.

        Uses the method from Qwen3 documentation: find </think> token (151668)
        in output_ids and split thinking/response content based on that.

        Args:
            outputs: vLLM output objects
            original_queries: Original query strings (unused but required by interface)

        Returns:
            List of (response_text, thinking_trace) tuples.
            thinking_trace is None if no thinking content was generated.
        """
        results = []

        for output in outputs:
            # Get output token IDs
            output_ids = output.outputs[0].token_ids

            # Find the </think> token (151668) using rindex method from docs
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(self.think_end_token_id)
            except ValueError:
                # No </think> token found - no thinking content
                index = 0

            if index > 0:
                # Split at </think> token
                thinking_ids = output_ids[:index]
                response_ids = output_ids[index:]

                # Load tokenizer if needed (using model from vLLM output if available)
                if self.tokenizer is None:
                    # Try to extract model path from output or use a default
                    # For now, we'll decode without tokenizer as fallback
                    full_text = output.outputs[0].text
                    if "</think>" in full_text:
                        parts = full_text.split("</think>", 1)
                        thinking_content = parts[0].replace("<think>", "").strip("\n")
                        response_content = parts[1].strip("\n")
                    else:
                        thinking_content = ""
                        response_content = full_text.strip("\n")
                else:
                    # Decode thinking and response content separately
                    thinking_content = self.tokenizer.decode(
                        thinking_ids, skip_special_tokens=True
                    ).strip("\n")
                    response_content = self.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    ).strip("\n")

                thinking_content = thinking_content.strip("</think>").strip("<think>").strip()

                results.append((response_content, thinking_content))
            else:
                # No thinking content
                response_content = output.outputs[0].text.strip("\n")
                results.append((response_content, None))

        return results
