"""Storage system for generations and evaluations."""

import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

from .models import Generation, Evaluation
from .utils import get_results_dir, ensure_dir

logger = logging.getLogger(__name__)


class Storage:
    """Handles saving and loading of generations and evaluations."""
    
    def __init__(self, run_prefix: str, dataset_name: str, generator_model: str):
        """Initialize storage for a specific run/dataset/model combination.
        
        Args:
            run_prefix: Run identifier (e.g., "exp-1", "baseline")
            dataset_name: Dataset name (e.g., "uphill")
            generator_model: Generator model name
        """
        self.run_prefix = run_prefix
        self.dataset_name = dataset_name
        self.generator_model = generator_model
        
        # Construct paths
        base_dir = get_results_dir() / run_prefix / dataset_name / generator_model
        self.base_dir = ensure_dir(base_dir)
        self.generations_file = self.base_dir / "generations.jsonl"
        self.plots_dir = ensure_dir(self.base_dir / "plots")
    
    def get_evaluation_file(self, evaluator_model: str) -> Path:
        """Get the evaluation file path for a specific evaluator."""
        # Sanitize model name for filename
        safe_name = evaluator_model.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"evaluation_{safe_name}.jsonl"
    
    def save_generations(self, generations: List[Generation]) -> None:
        """Save generations to JSONL atomically."""
        if not generations:
            logger.warning("No generations to save")
            return
        
        # Convert to JSON lines
        lines = [json.dumps(g.model_dump(), default=str) + "\n" for g in generations]
        
        # Atomic write using temporary file
        self._atomic_write_jsonl(lines, self.generations_file)
        logger.info(f"Saved {len(generations)} generations to {self.generations_file}")
    
    def load_generations(self) -> List[Generation]:
        """Load all generations from JSONL."""
        if not self.generations_file.exists():
            logger.info(f"No existing generations found at {self.generations_file}")
            return []
        
        generations = []
        with open(self.generations_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    generations.append(Generation(**data))
        
        logger.info(f"Loaded {len(generations)} generations from {self.generations_file}")
        return generations
    
    def get_generation_count_by_query(self) -> Dict[tuple, int]:
        """Get count of existing generations per (query_id, presupposition_level).
        
        Returns:
            Dict mapping (query_id, presupposition_level) -> count
        """
        generations = self.load_generations()
        counts = {}
        for gen in generations:
            key = (gen.query_id, gen.presupposition_level)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def save_evaluations(self, evaluator_model: str, evaluations: List[Evaluation]) -> None:
        """Save evaluations to JSONL atomically."""
        if not evaluations:
            logger.warning("No evaluations to save")
            return
        
        eval_file = self.get_evaluation_file(evaluator_model)
        
        # Convert to JSON lines
        lines = [json.dumps(e.model_dump(), default=str) + "\n" for e in evaluations]
        
        # Atomic write using temporary file
        self._atomic_write_jsonl(lines, eval_file)
        logger.info(f"Saved {len(evaluations)} evaluations to {eval_file}")
    
    def load_evaluations(self, evaluator_model: str) -> List[Evaluation]:
        """Load evaluations for a specific evaluator from JSONL."""
        eval_file = self.get_evaluation_file(evaluator_model)
        
        if not eval_file.exists():
            logger.info(f"No existing evaluations found at {eval_file}")
            return []
        
        evaluations = []
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    evaluations.append(Evaluation(**data))
        
        logger.info(f"Loaded {len(evaluations)} evaluations from {eval_file}")
        return evaluations
    
    def get_evaluated_generation_ids(self, evaluator_model: str) -> set:
        """Get set of gen_ids that have been evaluated by a specific evaluator.
        
        Note: This returns ALL evaluated gen_ids, including errors.
        Use get_successfully_evaluated_generation_ids() to exclude errors.
        """
        evaluations = self.load_evaluations(evaluator_model)
        return {e.gen_id for e in evaluations}
    
    def get_successfully_evaluated_generation_ids(self, evaluator_model: str) -> set:
        """Get set of gen_ids that have been successfully evaluated (no errors).
        
        This excludes evaluations where entailment == 'error', allowing them to be retried.
        """
        evaluations = self.load_evaluations(evaluator_model)
        return {e.gen_id for e in evaluations if e.entailment != "error"}
    
    def _atomic_write_jsonl(self, lines: List[str], file_path: Path) -> None:
        """Write lines to JSONL atomically using temporary file + rename.
        
        This prevents corruption if the process is interrupted during writing.
        """
        temp_dir = file_path.parent
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=temp_dir,
            prefix=f".{file_path.name}.tmp.",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.writelines(lines)
            temp_path = Path(temp_file.name)
        
        # Atomic replace
        temp_path.replace(file_path)
