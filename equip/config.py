"""Configuration loader for model specifications."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .processors import (
    HarmonyPreprocessor,
    HarmonyPostprocessor,
    StandardPreprocessor,
    StandardPostprocessor,
    QwenPreprocessor,
    QwenPostprocessor,
    SplitterPostprocessor,
)
from .generators import Generator, VLLMGenerator, VLLMOnlineGenerator, SGLangGenerator, GeminiGenerator, OpenAIBatchGenerator, GeminiBatchGenerator
from .evaluators import EntailmentModel, VLLMEntailmentModel
from .utils import get_project_root

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration loader and factory for models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with path to configuration file."""
        if config_path is None:
            config_path = get_project_root() / "models.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in self.config['models']:
            available = ', '.join(self.config['models'].keys())
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {available}"
            )

        return self.config['models'][model_name]
    
    def list_models(self) -> list:
        """List all available model names."""
        return list(self.config['models'].keys())
    
    def list_generators(self) -> list:
        """List all generator models."""
        return [
            name for name, config in self.config['models'].items() 
            if config.get('type') == 'generator'
        ]
    
    def list_evaluators(self) -> list:
        """List all evaluator models."""
        return [
            name for name, config in self.config['models'].items() 
            if config.get('type') == 'evaluator'
        ]
    
    def _create_preprocessor(self, preprocessor_config: Dict[str, Any]) -> Any:
        """Create preprocessor instance from configuration."""
        preprocessor_type = preprocessor_config.get('type', 'standard')
        config = preprocessor_config.get('config', {})
        
        if preprocessor_type == 'harmony':
            return HarmonyPreprocessor(**config)
        elif preprocessor_type == 'qwen':
            return QwenPreprocessor(**config)
        elif preprocessor_type == 'standard':
            return StandardPreprocessor(**config)
        else:
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")
    
    def _create_postprocessor(self, postprocessor_config: Dict[str, Any]) -> Any:
        """Create postprocessor instance from configuration."""
        postprocessor_type = postprocessor_config.get('type', 'standard')
        config = postprocessor_config.get('config', {})
        
        if postprocessor_type == 'harmony':
            return HarmonyPostprocessor(**config)
        elif postprocessor_type == 'qwen':
            return QwenPostprocessor(**config)
        elif postprocessor_type == 'splitter':
            return SplitterPostprocessor(**config)
        elif postprocessor_type == 'standard':
            return StandardPostprocessor(**config)
        else:
            raise ValueError(f"Unknown postprocessor type: {postprocessor_type}")
    
    def create_generator(self, model_name: str, storage_base_dir=None, **override_kwargs) -> Generator:
        """Create a generator instance from configuration.
        
        Args:
            model_name: Name of the model from configuration
            storage_base_dir: Base directory for storage (used by batch generators)
            **override_kwargs: Override parameters from config
        """
        model_config = self.get_model_config(model_name)
        
        if model_config.get('type') != 'generator':
            raise ValueError(f"Model '{model_name}' is not a generator")
        
        backend = model_config['backend']
        parameters = model_config.get('parameters', {})
        
        # Override with any provided kwargs
        parameters.update(override_kwargs)
        
        if backend == 'vllm':
            # Create processors if specified
            preprocessor = None
            postprocessor = None
            
            if 'preprocessor' in model_config:
                preprocessor = self._create_preprocessor(model_config['preprocessor'])
            if 'postprocessor' in model_config:
                postprocessor = self._create_postprocessor(model_config['postprocessor'])
            
            return VLLMGenerator(
                model_path=model_config['model_path'],
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                **parameters
            )
        elif backend == 'vllm-online':
            # Create processors if specified
            preprocessor = None
            postprocessor = None
            
            if 'preprocessor' in model_config:
                preprocessor = self._create_preprocessor(model_config['preprocessor'])
            if 'postprocessor' in model_config:
                postprocessor = self._create_postprocessor(model_config['postprocessor'])
            
            # Extract vllm-online specific parameters
            port = parameters.pop('port', 8000)
            api_key = parameters.pop('api_key', 'EMPTY')
            base_url = parameters.pop('base_url', None)
            
            return VLLMOnlineGenerator(
                model_path=model_config['model_path'],
                port=port,
                api_key=api_key,
                base_url=base_url,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                **parameters
            )
        elif backend == 'sglang':
            # Create processors if specified
            preprocessor = None
            postprocessor = None
            
            if 'preprocessor' in model_config:
                preprocessor = self._create_preprocessor(model_config['preprocessor'])
            if 'postprocessor' in model_config:
                postprocessor = self._create_postprocessor(model_config['postprocessor'])
            
            return SGLangGenerator(
                model_path=model_config['model_path'],
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                **parameters
            )
        elif backend == 'gemini':
            # Gemini doesn't use preprocessor/postprocessor pattern
            # It handles everything internally via API
            return GeminiGenerator(
                model_path=model_config['model_path'],
                api_key=model_config.get('api_key'),
                **parameters
            )
        elif backend == 'openai_batch':
            # OpenAI Batch API generator
            # Pass storage_base_dir if provided
            if storage_base_dir:
                parameters['storage_base_dir'] = storage_base_dir
            
            return OpenAIBatchGenerator(
                model_name=model_config['model_path'],
                api_key=model_config.get('api_key'),
                base_url=model_config.get('base_url'),
                **parameters
            )
        elif backend == 'gemini_batch':
            # Gemini Batch API generator
            # Pass storage_base_dir if provided
            if storage_base_dir:
                parameters['storage_base_dir'] = storage_base_dir
            
            # Extract generation parameters
            gen_params = {
                'model_name': model_config['model_path'],
                'storage_base_dir': parameters.get('storage_base_dir'),
                'max_batch_requests': parameters.get('max_batch_requests', 10000),
                'concurrent_batches': parameters.get('concurrent_batches', 5),
                'temperature': parameters.get('temperature', 1.0),
                'max_tokens': parameters.get('max_tokens', 8192),
            }
            
            # Handle thinking configuration
            if 'thinking_level' in parameters:
                gen_params['thinking_level'] = parameters['thinking_level']
            elif 'thinking_budget' in parameters:
                gen_params['thinking_budget'] = parameters['thinking_budget']
            
            return GeminiBatchGenerator(**gen_params)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def create_evaluator(self, model_name: str, **override_kwargs) -> EntailmentModel:
        """Create an evaluator instance from configuration."""
        model_config = self.get_model_config(model_name)
        
        if model_config.get('type') != 'evaluator':
            raise ValueError(f"Model '{model_name}' is not an evaluator")
        
        backend = model_config['backend']
        parameters = model_config.get('parameters', {})
        
        # Override with any provided kwargs
        parameters.update(override_kwargs)
        
        if backend == 'vllm':
            # Create processors if specified
            preprocessor = None
            postprocessor = None
            
            if 'preprocessor' in model_config:
                preprocessor = self._create_preprocessor(model_config['preprocessor'])
            if 'postprocessor' in model_config:
                postprocessor = self._create_postprocessor(model_config['postprocessor'])
            
            return VLLMEntailmentModel(
                model_path=model_config['model_path'],
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                **parameters
            )
        else:
            raise ValueError(f"Backend '{backend}' not supported for evaluators")
