"""Splitter postprocessor for models that output reasoning and response in one text."""

from typing import List, Any, Optional, Union, Tuple
from .base import Postprocessor


class SplitterPostprocessor(Postprocessor):
    """Postprocessor that splits model output into thinking trace and response.
    
    Splits at a configurable end tag and cleans start tags from the thinking trace.
    """

    def __init__(
        self,
        end_tag: str = "</think>",
        start_tag: Optional[str] = "<think>",
        order: str = "thinking-first",
        **config
    ):
        """Initialize the splitter postprocessor.

        Args:
            end_tag: The end tag to split on (default: "</think>")
            start_tag: The start tag to clean from thinking trace (default: "<think>")
            order: Order of content - "thinking-first" or "response-first" (default: "thinking-first")
            **config: Additional configuration passed to base class
        """
        super().__init__(**config)
        self.end_tag = end_tag
        self.start_tag = start_tag
        self.order = order.lower()

        if self.order not in ["thinking-first", "response-first"]:
            raise ValueError(
                f"Invalid order '{order}'. Must be 'thinking-first' or 'response-first'"
            )

    def process(
        self, outputs: List[Any], original_queries: List[str]
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract and split text from model outputs.

        Args:
            outputs: List of model outputs (vLLM outputs or raw strings)
            original_queries: Original query strings (unused but required by interface)

        Returns:
            List of (response_text, thinking_trace) tuples.
            thinking_trace is None if no thinking content was generated.
        """
        results = []

        for output in outputs:
            # Extract raw text from output - handle different formats
            if isinstance(output, str):
                # Direct string output
                raw_text = output
            elif hasattr(output, "outputs") and len(output.outputs) > 0:
                # vLLM output format
                raw_text = output.outputs[0].text
            elif hasattr(output, "choices") and len(output.choices) > 0:
                # OpenAI API response format
                raw_text = output.choices[0].message.content
            elif hasattr(output, "text"):
                # Generic text attribute
                raw_text = output.text
            else:
                # Fallback: convert to string
                raw_text = str(output)

            # Check if end tag is present
            if self.end_tag in raw_text:
                # Split on the end tag (only first occurrence)
                parts = raw_text.split(self.end_tag, 1)

                if self.order == "thinking-first":
                    # Format: <start_tag>thinking<end_tag>response
                    thinking_content = parts[0]
                    response_content = parts[1] if len(parts) > 1 else ""
                else:
                    # Format: response<end_tag>thinking
                    response_content = parts[0]
                    thinking_content = parts[1] if len(parts) > 1 else ""

                # Clean start tag from thinking content if specified
                if self.start_tag and thinking_content:
                    thinking_content = thinking_content.replace(self.start_tag, "")

                # Strip whitespace
                thinking_content = thinking_content.strip()
                response_content = response_content.strip()

                results.append((response_content, thinking_content))
            else:
                # No end tag found - treat entire text as response
                response_content = raw_text.strip()
                results.append((response_content, None))

        return results
