"""Standard preprocessing and postprocessing for regular chat models."""

from typing import List, Dict, Any, Tuple
from .base import Preprocessor, Postprocessor


class StandardPreprocessor(Preprocessor):
    """Standard preprocessor for regular chat models."""
    
    def process(self, queries: List[str], **generation_kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess queries into standard chat format."""
        system_prompt = generation_kwargs.get('system_prompt', "You are a helpful assistant.")
        
        # Prepare chat messages
        messages_batch = []
        
        for query in queries:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": query})
            
            messages_batch.append(messages)
            
        return messages_batch, generation_kwargs


class StandardPostprocessor(Postprocessor):
    """Standard postprocessor for regular chat models."""
    
    def process(self, outputs: List[Any], original_queries: List[str]) -> List[str]:
        """Extract text from standard model outputs."""
        responses = []
        
        for output in outputs:
            responses.append(output.outputs[0].text)
        
        return responses
