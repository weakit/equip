"""Harmony preprocessing and postprocessing for OpenAI GPT-OSS models."""

from typing import List, Dict, Any, Optional, Tuple
from .base import Preprocessor, Postprocessor

try:
    from openai_harmony import (
        ReasoningEffort,
        load_harmony_encoding,
        HarmonyEncodingName,
        Conversation,
        Message,
        Role,
        SystemContent,
        TextContent
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False


class HarmonyPreprocessor(Preprocessor):
    """Harmony preprocessor for OpenAI GPT-OSS models with reasoning levels."""

    def __init__(self, harmony_encoding_name: str = "HARMONY_GPT_OSS", **config):
        super().__init__(harmony_encoding_name=harmony_encoding_name, **config)
        self.harmony_encoding_name = harmony_encoding_name
        self.encoding = None
        
        if not HARMONY_AVAILABLE:
            raise ImportError("openai-harmony package is required for HarmonyPreprocessor")

    def load_encoding(self):
        """Load harmony encoding if not already loaded."""
        if self.encoding is None:
            encoding_name = getattr(HarmonyEncodingName, self.harmony_encoding_name)
            self.encoding = load_harmony_encoding(encoding_name)

    def process(
        self, queries: List[str], **generation_kwargs
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess queries into harmony-encoded conversations."""
        self.load_encoding()

        reasoning_level = self.config.get("reasoning_level", "off")
        instructions = generation_kwargs.get("system_prompt")

        # Determine reasoning settings based on level
        effort_map = {
            "high": ReasoningEffort.HIGH,
            "medium": ReasoningEffort.MEDIUM,
            "low": ReasoningEffort.LOW,
        }

        reasoning_effort = effort_map.get(reasoning_level.lower(), False)

        # Prepare harmony conversations
        conversations = self._prepare_conversations(
            queries, instructions, reasoning_effort
        )

        # Update generation kwargs with harmony-specific settings
        updated_kwargs = generation_kwargs.copy()
        updated_kwargs["disable_reasoning"] = not reasoning_effort
        updated_kwargs["stop_token_ids"] = self.encoding.stop_tokens_for_assistant_actions()

        return conversations, updated_kwargs

    def _prepare_conversations(
        self,
        queries: List[str],
        system_instructions: Optional[str] = None,
        reasoning_effort: bool | ReasoningEffort = False,
    ) -> List[List[int]]:
        """Prepare conversations for vLLM generation."""

        conversations = []

        for query in queries:
            messages = []

            # Add system message with reasoning effort
            if reasoning_effort:
                messages.append(
                    Message.from_role_and_content(
                        Role.SYSTEM,
                        SystemContent().new().with_reasoning_effort(reasoning_effort),
                    )
                )
            else:
                messages.append(
                    Message.from_role_and_content(
                        Role.SYSTEM,
                        SystemContent()
                        .new()
                        .with_reasoning_effort(ReasoningEffort.LOW),
                    )
                )

            # Add system instructions if present
            if system_instructions:
                messages.append(
                    Message.from_role_and_content(
                        Role.DEVELOPER,
                        system_instructions,
                    )
                )

            # Add user query
            messages.append(Message.from_role_and_content(Role.USER, query))

            # Prefill empty reasoning trace if reasoning is off
            if not reasoning_effort:
                messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, "").with_channel(
                        "analysis"
                    )
                )

            conversation = Conversation.from_messages(messages)
            conversation_tokens = self.encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
            conversations.append(conversation_tokens)

        return conversations


class HarmonyPostprocessor(Postprocessor):
    """Harmony postprocessor for OpenAI GPT-OSS models with reasoning extraction."""

    def __init__(self, harmony_encoding_name: str = "HARMONY_GPT_OSS", **config):
        super().__init__(harmony_encoding_name=harmony_encoding_name, **config)
        self.harmony_encoding_name = harmony_encoding_name
        self.encoding = None
        
        if not HARMONY_AVAILABLE:
            raise ImportError("openai-harmony package is required for HarmonyPostprocessor")

    def load_encoding(self):
        """Load harmony encoding if not already loaded."""
        if self.encoding is None:
            encoding_name = getattr(HarmonyEncodingName, self.harmony_encoding_name)
            self.encoding = load_harmony_encoding(encoding_name)

    def process(
        self, outputs: List[Any], original_queries: List[str]
    ) -> List[Tuple[str, str]]:
        """Decode harmony outputs and extract reasoning if present."""
        self.load_encoding()

        responses = []

        for output in outputs:
            reasoning_trace = ""
            assistant_response = ""

            messages = self.encoding.parse_messages_from_completion_tokens(
                output.outputs[0].token_ids, Role.ASSISTANT
            )

            for message in messages:
                full_text_content = ""

                for content in message.content:
                    if not isinstance(content, TextContent):
                        continue
                    full_text_content += content.text + "\n"
                
                if message.channel == "analysis":
                    reasoning_trace += full_text_content
                if message.channel == "final":
                    assistant_response += full_text_content
            
            responses.append((assistant_response.strip(), reasoning_trace.strip()))
        
        return responses
