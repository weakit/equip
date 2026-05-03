"""Sample generation logic for human annotation."""

import logging
import random
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from ..utils import generate_id
from ..explorer.data_loader import ExplorerDataLoader, ExplorerData, GenerationWithEval
from .annotation_models import ResponseSample, SampleSet

logger = logging.getLogger(__name__)


class AnnotationSampler:
    """Generates sample sets for human annotation."""
    
    def __init__(self, data_loader: Optional[ExplorerDataLoader] = None):
        """Initialize the sampler.
        
        Args:
            data_loader: ExplorerDataLoader instance. If None, creates a new one.
        """
        self.data_loader = data_loader or ExplorerDataLoader()
    
    def create_sample_set(
        self,
        prefix: str,
        dataset: str,
        models: List[str],
        evaluator: Optional[str],
        total_samples: int,
        set_name: str,
        description: str = "",
        random_seed: Optional[int] = None
    ) -> Tuple[SampleSet, List[ResponseSample]]:
        """Create a new sample set for annotation.
        
        Args:
            prefix: Result prefix (e.g., 'exp-1')
            dataset: Dataset name
            models: List of model names to sample from
            evaluator: Optional evaluator name
            total_samples: Total number of samples to generate
            set_name: Human-readable name for the set
            description: Optional description
            random_seed: Optional random seed for reproducibility
        
        Returns:
            Tuple of (SampleSet, List[ResponseSample])
        
        Raises:
            ValueError: If insufficient data or invalid parameters
        """
        if total_samples <= 0:
            raise ValueError("total_samples must be positive")
        
        if total_samples % 2 != 0:
            raise ValueError("total_samples must be even (split between true/false)")
        
        if not models:
            raise ValueError("At least one model must be specified")
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Load data for all models
        all_gen_with_evals: List[Tuple[str, GenerationWithEval, str, str]] = []  # (model, gen_with_eval, claim, veracity)
        
        for model in models:
            logger.info(f"Loading data for model: {model}")
            data = self.data_loader.load_data(prefix, dataset, model, evaluator)
            
            # Extract all generations with their metadata
            for base_query_id in data.get_base_query_ids():
                metadata = data.get_query_metadata(base_query_id)
                if not metadata:
                    continue
                
                # Get all levels for this query
                all_levels = data.get_all_levels_for_base(base_query_id)
                
                # Collect all generations from all levels
                for level, gens_at_level in all_levels.items():
                    for gen_with_eval in gens_at_level:
                        all_gen_with_evals.append((
                            model,
                            gen_with_eval,
                            metadata.claim,
                            metadata.normalized_veracity
                        ))
        
        logger.info(f"Total available generations: {len(all_gen_with_evals)}")
        
        # Separate by veracity
        true_samples = [(m, g, c, v) for m, g, c, v in all_gen_with_evals if v == "true"]
        false_samples = [(m, g, c, v) for m, g, c, v in all_gen_with_evals if v == "false"]
        
        logger.info(f"True samples: {len(true_samples)}, False samples: {len(false_samples)}")
        
        # Calculate samples per veracity
        samples_per_veracity = total_samples // 2
        
        # Check if we have enough data
        if len(true_samples) < samples_per_veracity:
            raise ValueError(
                f"Insufficient true samples: need {samples_per_veracity}, have {len(true_samples)}"
            )
        
        if len(false_samples) < samples_per_veracity:
            raise ValueError(
                f"Insufficient false samples: need {samples_per_veracity}, have {len(false_samples)}"
            )
        
        # Calculate samples per model per veracity
        samples_per_model_per_veracity = samples_per_veracity // len(models)
        remainder = samples_per_veracity % len(models)
        
        logger.info(f"Sampling {samples_per_model_per_veracity} per model per veracity (remainder: {remainder})")
        
        # Sample evenly across models for each veracity
        selected_samples = []
        
        for veracity_label, sample_pool in [("true", true_samples), ("false", false_samples)]:
            # Group by model
            by_model = defaultdict(list)
            for model, gen_with_eval, claim, veracity in sample_pool:
                by_model[model].append((model, gen_with_eval, claim, veracity))
            
            # Sample from each model
            for i, model in enumerate(models):
                model_pool = by_model[model]
                
                # Add one extra sample to some models to handle remainder
                num_to_sample = samples_per_model_per_veracity
                if i < remainder:
                    num_to_sample += 1
                
                if len(model_pool) < num_to_sample:
                    raise ValueError(
                        f"Insufficient {veracity_label} samples for model {model}: "
                        f"need {num_to_sample}, have {len(model_pool)}"
                    )
                
                # Random sample
                sampled = random.sample(model_pool, num_to_sample)
                selected_samples.extend(sampled)
        
        logger.info(f"Selected {len(selected_samples)} samples before shuffling")
        
        # Shuffle all samples together
        random.shuffle(selected_samples)
        
        # Create ResponseSample objects
        response_samples = []
        sample_ids = []
        
        for model, gen_with_eval, claim, veracity in selected_samples:
            sample_id = generate_id()
            
            response_sample = ResponseSample(
                sample_id=sample_id,
                gen_id=gen_with_eval.gen_id,
                query_id=gen_with_eval.query_id,
                dataset=dataset,
                model=model,
                presupposition_level=gen_with_eval.presupposition_level,
                claim=claim,
                claim_veracity=veracity,
                response=gen_with_eval.response,
                reasoning_trace=gen_with_eval.reasoning_trace,
                llm_entailment=gen_with_eval.entailment,
                llm_reasoning=gen_with_eval.eval_reasoning
            )
            
            response_samples.append(response_sample)
            sample_ids.append(sample_id)
        
        # Create SampleSet
        set_id = generate_id()
        sample_set = SampleSet(
            set_id=set_id,
            set_name=set_name,
            description=description,
            prefix=prefix,
            dataset=dataset,
            models=models,
            evaluator=evaluator,
            total_samples=total_samples,
            samples_per_veracity=samples_per_veracity,
            samples_per_model=samples_per_veracity // len(models),
            sample_ids=sample_ids
        )
        
        logger.info(f"Created sample set '{set_name}' with {len(response_samples)} samples")
        
        return sample_set, response_samples
