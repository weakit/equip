"""Data loader for the explorer - loads and organizes generations and evaluations."""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from ..models import Generation, Evaluation
from ..utils import get_results_dir, get_data_dir

logger = logging.getLogger(__name__)


def normalize_veracity(veracity: str) -> str:
    """Normalize veracity labels to standard format: true, false, or neutral.
    
    Args:
        veracity: Raw veracity label from dataset
    
    Returns:
        Normalized veracity: 'true', 'false', or 'neutral'
    """
    veracity_lower = veracity.lower().strip()
    
    # Map SUPPORTS variations to true
    if veracity_lower in ['supports', 'support', 'true', 'yes']:
        return 'true'
    
    # Map REFUTES variations to false
    if veracity_lower in ['refutes', 'refute', 'false', 'no']:
        return 'false'
    
    # Map mixture/neutral variations
    if veracity_lower in ['mixture', 'neutral', 'not enough info', 'nei']:
        return 'neutral'
    
    # Default to neutral for unknown
    return 'neutral'


@dataclass
class QueryMetadata:
    """Metadata about a query (claim, veracity, query text per level)."""
    claim: str
    veracity: str  # e.g., "SUPPORTS", "REFUTES", "true", "false" - raw from dataset
    normalized_veracity: str = "neutral"  # Normalized to: 'true', 'false', or 'neutral'
    category: Optional[str] = None
    # Map: presupposition_level -> query text
    queries_by_level: Dict[int, str] = field(default_factory=dict)
    
    def is_claim_true(self) -> bool:
        """Check if the claim is true based on veracity."""
        return self.normalized_veracity == "true"


@dataclass
class GenerationWithEval:
    """A generation paired with its evaluation (if available)."""
    generation: Generation
    evaluation: Optional[Evaluation] = None
    
    @property
    def gen_id(self) -> str:
        return self.generation.gen_id
    
    @property
    def query_id(self) -> str:
        return self.generation.query_id
    
    @property
    def presupposition_level(self) -> int:
        return self.generation.presupposition_level
    
    @property
    def response(self) -> str:
        return self.generation.response
    
    @property
    def reasoning_trace(self) -> Optional[str]:
        return self.generation.reasoning_trace
    
    @property
    def entailment(self) -> Optional[str]:
        return self.evaluation.entailment if self.evaluation else None
    
    @property
    def eval_reasoning(self) -> Optional[str]:
        return self.evaluation.reasoning if self.evaluation else None
    
    @property
    def unsure(self) -> Optional[bool]:
        return self.evaluation.unsure if self.evaluation else None


@dataclass
class ExplorerData:
    """Container for all explorer data."""
    prefix: str
    dataset: str
    model: str
    evaluator: Optional[str]
    
    # Map: base_query_id -> presupposition_level -> list of GenerationWithEval
    data: Dict[str, Dict[int, List[GenerationWithEval]]]
    
    # Sorted list of base query IDs (computed at load time)
    base_query_ids: List[str] = field(default_factory=list)
    
    # Total generation count
    total_generations: int = 0
    
    # Query metadata (claim, veracity, query text per level)
    query_metadata: Dict[str, QueryMetadata] = field(default_factory=dict)
    
    def get_base_query_ids(self) -> List[str]:
        """Get sorted list of base query IDs."""
        return self.base_query_ids
    
    def get_all_levels_for_base(self, base_query_id: str) -> Dict[int, List[GenerationWithEval]]:
        """Get all presupposition levels and their generations for a base query."""
        return self.data.get(base_query_id, {})
    
    def get_total_base_queries(self) -> int:
        """Get total number of unique base queries."""
        return len(self.base_query_ids)
    
    def get_total_generations(self) -> int:
        """Get total number of generations across all queries."""
        return self.total_generations
    
    def get_query_metadata(self, base_query_id: str) -> Optional[QueryMetadata]:
        """Get metadata for a base query."""
        return self.query_metadata.get(base_query_id)
    
    def calculate_generation_set_resistance(self, base_query_id: str, gen_index: int = 0) -> Optional[int]:
        """Calculate resistance for a specific generation set.
        
        Resistance is the first level where the generation response is factually inaccurate.
        - Accurate = (claim true AND generation agrees) OR (claim false AND generation disagrees)
        - Inaccurate = opposite of accurate
        
        Args:
            base_query_id: Base query ID
            gen_index: Which generation in the set (default 0)
        
        Returns:
            Resistance level (0-5), or None if cannot determine
            - 0: Level 0 is inaccurate
            - 1: Level 0 accurate, level 1 inaccurate
            - ...
            - 5: All levels 0-4 are accurate
        """
        metadata = self.query_metadata.get(base_query_id)
        if not metadata:
            return None
        
        claim_is_true = metadata.is_claim_true()
        all_levels = self.data.get(base_query_id, {})
        
        # Check levels 0-4
        for level in range(5):
            gens_at_level = all_levels.get(level, [])
            if gen_index >= len(gens_at_level):
                return None  # Generation doesn't exist at this level
            
            gen_with_eval = gens_at_level[gen_index]
            if not gen_with_eval.evaluation:
                return None  # No evaluation available
            
            entailment = gen_with_eval.entailment
            if not entailment or entailment == "error":
                return None  # Cannot determine accuracy
            
            # Check if generation is accurate
            is_accurate = self._is_generation_accurate(claim_is_true, entailment)
            
            if not is_accurate:
                return level  # First inaccurate level
        
        # All levels 0-4 are accurate
        return 5
    
    def _is_generation_accurate(self, claim_is_true: bool, entailment: str) -> bool:
        """Check if a generation is factually accurate.
        
        Args:
            claim_is_true: Whether the claim is true
            entailment: Generation entailment ("agree", "disagree", "neutral")
        
        Returns:
            True if generation is accurate, False otherwise
        """
        if entailment == "neutral":
            # Neutral responses could be considered accurate or not - let's say accurate
            return True
        elif entailment == "agree":
            # Agrees with claim - accurate if claim is true
            return claim_is_true
        elif entailment == "disagree":
            # Disagrees with claim - accurate if claim is false
            return not claim_is_true
        else:
            return False
    
    def get_all_resistances_for_query(self, base_query_id: str) -> List[Optional[int]]:
        """Get resistance values for all generation sets of a query.
        
        Returns:
            List of resistance values, one per generation set
        """
        all_levels = self.data.get(base_query_id, {})
        gens_at_level_0 = all_levels.get(0, [])
        num_generations = len(gens_at_level_0)
        
        resistances = []
        for gen_index in range(num_generations):
            resistance = self.calculate_generation_set_resistance(base_query_id, gen_index)
            resistances.append(resistance)
        
        return resistances
    
    def get_resistance_for_query(self, base_query_id: str) -> Optional[int]:
        """Get resistance level for a base query (checks if ANY generation set has the pattern).
        
        For filtering: a query matches resistance R if at least one generation set has resistance R.
        """
        # Check first generation set for simplicity (can be extended to check all)
        return self.calculate_generation_set_resistance(base_query_id, gen_index=0)
    
    def get_available_resistance_levels(self) -> List[int]:
        """Get list of available resistance levels in the dataset."""
        resistance_set = set()
        for base_query_id in self.base_query_ids:
            resistance = self.get_resistance_for_query(base_query_id)
            if resistance is not None:
                resistance_set.add(resistance)
        return sorted(resistance_set)
    
    def filter_by_resistance(self, resistance_level: Optional[int]) -> List[str]:
        """Filter base query IDs by resistance level.
        
        Args:
            resistance_level: Target resistance level (0-4), or None for no filtering
        
        Returns:
            Filtered list of base query IDs
        """
        if resistance_level is None:
            return self.base_query_ids
        
        filtered = []
        for base_query_id in self.base_query_ids:
            query_resistance = self.get_resistance_for_query(base_query_id)
            if query_resistance == resistance_level:
                filtered.append(base_query_id)
        return filtered
    
    def get_available_veracity_values(self) -> List[str]:
        """Get list of available claim veracity values in the dataset."""
        veracity_set = set()
        for base_query_id in self.base_query_ids:
            metadata = self.query_metadata.get(base_query_id)
            if metadata and metadata.normalized_veracity:
                veracity_set.add(metadata.normalized_veracity)
        return sorted(veracity_set)
    
    def filter_by_veracity(self, veracity: Optional[str]) -> List[str]:
        """Filter base query IDs by claim veracity.
        
        Args:
            veracity: Target veracity value (normalized: true/false/neutral), or None for no filtering
        
        Returns:
            Filtered list of base query IDs
        """
        if veracity is None:
            return self.base_query_ids
        
        filtered = []
        for base_query_id in self.base_query_ids:
            metadata = self.query_metadata.get(base_query_id)
            if metadata and metadata.normalized_veracity == veracity:
                filtered.append(base_query_id)
        return filtered
    
    def filter_queries(self, resistance_level: Optional[int] = None, veracity: Optional[str] = None) -> List[str]:
        """Filter base query IDs by resistance level and/or veracity.
        
        Args:
            resistance_level: Target resistance level (0-4), or None for no filtering
            veracity: Target veracity value, or None for no filtering
        
        Returns:
            Filtered list of base query IDs
        """
        # Start with all query IDs
        filtered = self.base_query_ids[:]
        
        # Apply resistance filter
        if resistance_level is not None:
            filtered = [qid for qid in filtered if self.get_resistance_for_query(qid) == resistance_level]
        
        # Apply veracity filter
        if veracity is not None:
            filtered = [qid for qid in filtered 
                       if self.query_metadata.get(qid) and self.query_metadata[qid].normalized_veracity == veracity]
        
        return filtered


class ExplorerDataLoader:
    """Loads and organizes data for the explorer UI."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize the data loader.
        
        Args:
            results_dir: Path to results directory. If None, uses default from utils.
        """
        self.results_dir = results_dir or get_results_dir()
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        logger.info(f"Initialized ExplorerDataLoader with results_dir: {self.results_dir}")
    
    def get_available_prefixes(self) -> List[str]:
        """Get list of available result prefixes (e.g., 'exp-1', 'final-1')."""
        prefixes = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                prefixes.append(item.name)
        return sorted(prefixes)
    
    def get_available_datasets(self, prefix: str) -> List[str]:
        """Get list of available datasets for a given prefix."""
        prefix_dir = self.results_dir / prefix
        if not prefix_dir.exists():
            return []
        
        datasets = []
        for item in prefix_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                datasets.append(item.name)
        return sorted(datasets)
    
    def get_available_models(self, prefix: str, dataset: str) -> List[str]:
        """Get list of available models for a given prefix and dataset."""
        dataset_dir = self.results_dir / prefix / dataset
        if not dataset_dir.exists():
            return []
        
        models = []
        for item in dataset_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Only include if it has a generations.jsonl file
                if (item / "generations.jsonl").exists():
                    models.append(item.name)
        return sorted(models)
    
    def get_available_evaluators(self, prefix: str, dataset: str, model: str) -> List[str]:
        """Get list of available evaluators for a given prefix, dataset, and model."""
        model_dir = self.results_dir / prefix / dataset / model
        if not model_dir.exists():
            return []
        
        evaluators = []
        for item in model_dir.iterdir():
            if item.is_file() and item.name.startswith("evaluation_") and item.name.endswith(".jsonl"):
                # Extract evaluator name from filename
                # Format: evaluation_{evaluator_name}.jsonl
                evaluator_name = item.name[11:-6]  # Remove "evaluation_" and ".jsonl"
                evaluators.append(evaluator_name)
        
        return sorted(evaluators)
    
    def load_data(
        self,
        prefix: str,
        dataset: str,
        model: str,
        evaluator: Optional[str] = None
    ) -> ExplorerData:
        """Load all generations and evaluations for the specified configuration.
        
        Args:
            prefix: Result prefix (e.g., 'exp-1')
            dataset: Dataset name (e.g., 'foolmetwice')
            model: Model name (e.g., 'gpt-oss-20b-medium')
            evaluator: Optional evaluator name to load evaluations
        
        Returns:
            ExplorerData containing all loaded data
        """
        model_dir = self.results_dir / prefix / dataset / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load generations
        generations_file = model_dir / "generations.jsonl"
        if not generations_file.exists():
            raise FileNotFoundError(f"Generations file not found: {generations_file}")
        
        logger.info(f"Loading generations from {generations_file}")
        generations = self._load_generations(generations_file)
        logger.info(f"Loaded {len(generations)} generations")
        
        # Load evaluations if evaluator specified
        evaluations_map = {}
        if evaluator:
            eval_file = model_dir / f"evaluation_{evaluator}.jsonl"
            if eval_file.exists():
                logger.info(f"Loading evaluations from {eval_file}")
                evaluations = self._load_evaluations(eval_file)
                evaluations_map = {e.gen_id: e for e in evaluations}
                logger.info(f"Loaded {len(evaluations)} evaluations")
            else:
                logger.warning(f"Evaluation file not found: {eval_file}")
        
        # Organize data by BASE query_id and presupposition_level
        # This groups all levels under the same base query for efficient access
        data: Dict[str, Dict[int, List[GenerationWithEval]]] = defaultdict(lambda: defaultdict(list))
        total_gens = 0
        
        for gen in generations:
            evaluation = evaluations_map.get(gen.gen_id)
            gen_with_eval = GenerationWithEval(generation=gen, evaluation=evaluation)
            
            # Extract base query ID (remove _N suffix)
            base_query_id = self._extract_base_query(gen.query_id)
            presup_level = gen.presupposition_level
            
            data[base_query_id][presup_level].append(gen_with_eval)
            total_gens += 1
        
        # Convert defaultdict to regular dict and get sorted base query IDs
        data = {qid: dict(presup_dict) for qid, presup_dict in data.items()}
        base_query_ids = sorted(data.keys())
        
        logger.info(f"Organized into {len(base_query_ids)} base queries")
        
        # Load query metadata (claim, veracity, query text per level)
        query_metadata = self._load_query_metadata(dataset, base_query_ids)
        logger.info(f"Loaded metadata for {len(query_metadata)} queries")
        
        return ExplorerData(
            prefix=prefix,
            dataset=dataset,
            model=model,
            evaluator=evaluator,
            data=data,
            base_query_ids=base_query_ids,
            total_generations=total_gens,
            query_metadata=query_metadata
        )
    
    @staticmethod
    def _extract_base_query(query_id: str) -> str:
        """Extract base query ID by removing presupposition level suffix."""
        match = re.match(r'(.+)_(\d+)$', query_id)
        if match:
            return match.group(1)
        return query_id
    
    def _load_generations(self, file_path: Path) -> List[Generation]:
        """Load generations from JSONL file."""
        generations = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    generations.append(Generation(**data))
        return generations
    
    def _load_evaluations(self, file_path: Path) -> List[Evaluation]:
        """Load evaluations from JSONL file."""
        evaluations = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    evaluations.append(Evaluation(**data))
        return evaluations
    
    def _load_query_metadata(self, dataset: str, base_query_ids: List[str]) -> Dict[str, QueryMetadata]:
        """Load query metadata (claim, veracity, query text per level) from data files.
        
        Supports both JSONL (foolme2) and CSV (uphill) formats.
        """
        data_dir = get_data_dir()
        metadata: Dict[str, QueryMetadata] = {}
        
        # Map dataset names (results might use different names than data folders)
        dataset_map = {
            "foolmetwice": "foolme2",
            "foolme2": "foolme2",
            "uphill": "uphill",
            "scifact": "scifact"
        }
        
        # Get the actual folder name
        actual_dataset = dataset_map.get(dataset, dataset)
        
        # Try JSONL format first (foolme2 style)
        jsonl_path = data_dir / actual_dataset / "queries.jsonl"
        if jsonl_path.exists():
            return self._load_query_metadata_jsonl(jsonl_path, base_query_ids)
        
        # Try CSV format (uphill style)
        csv_path = data_dir / actual_dataset / "queries.csv"
        if csv_path.exists():
            return self._load_query_metadata_csv(csv_path, base_query_ids)
        
        logger.warning(f"No query metadata file found for dataset: {dataset} (tried {actual_dataset})")
        return metadata
    
    def _load_query_metadata_jsonl(self, file_path: Path, base_query_ids: List[str]) -> Dict[str, QueryMetadata]:
        """Load query metadata from JSONL file (foolme2 format)."""
        # Build a set of base query IDs for faster lookup
        base_ids_set = set(base_query_ids)
        
        # Temporary storage: base_id -> {claim, veracity, category, queries_by_level}
        temp_data: Dict[str, dict] = {}
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                query_id = data.get("id", "")
                
                # Extract base query ID
                base_id = self._extract_base_query(query_id)
                
                if base_id not in base_ids_set:
                    continue
                
                level = data.get("presupposition_level", 0)
                
                if base_id not in temp_data:
                    temp_data[base_id] = {
                        "claim": data.get("claim", ""),
                        "veracity": data.get("label", ""),  # SUPPORTS, REFUTES
                        "category": data.get("category", ""),
                        "queries_by_level": {}
                    }
                
                temp_data[base_id]["queries_by_level"][level] = data.get("query", "")
        
        # Convert to QueryMetadata objects
        result = {}
        for base_id, info in temp_data.items():
            result[base_id] = QueryMetadata(
                claim=info["claim"],
                veracity=info["veracity"],
                normalized_veracity=normalize_veracity(info["veracity"]),
                category=info["category"],
                queries_by_level=info["queries_by_level"]
            )
        
        return result
    
    def _load_query_metadata_csv(self, file_path: Path, base_query_ids: List[str]) -> Dict[str, QueryMetadata]:
        """Load query metadata from CSV file (uphill format)."""
        base_ids_set = set(base_query_ids)
        temp_data: Dict[str, dict] = {}
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                claim_id = row.get("claim_id", "")
                
                if claim_id not in base_ids_set:
                    continue
                
                level = int(row.get("presupposition_level", 0)) if row.get("presupposition_level", "").isdigit() else 0
                
                if claim_id not in temp_data:
                    temp_data[claim_id] = {
                        "claim": row.get("claim", ""),
                        "veracity": row.get("claim_veracity", ""),  # true, false, mixture
                        "category": row.get("source_db", ""),
                        "queries_by_level": {}
                    }
                
                temp_data[claim_id]["queries_by_level"][level] = row.get("query_with_presupposition", "")
        
        # Convert to QueryMetadata objects
        result = {}
        for base_id, info in temp_data.items():
            result[base_id] = QueryMetadata(
                claim=info["claim"],
                veracity=info["veracity"],
                normalized_veracity=normalize_veracity(info["veracity"]),
                category=info["category"],
                queries_by_level=info["queries_by_level"]
            )
        
        return result
