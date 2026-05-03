"""Data models for the evaluation framework."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Query(BaseModel):
    """A query to be evaluated."""
    
    id: str = Field(..., description="Unique query identifier (e.g., claim_id)")
    claim: str = Field(..., description="The claim being evaluated")
    veracity: Literal["true", "false", "mixture", "fabricated"] = Field(..., description="Ground truth veracity")
    presupposition_level: int = Field(..., ge=0, le=4, description="Presupposition level (0-4)")
    query_text: str = Field(..., description="The actual query text to send to the model")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    
    def get_normalized_veracity(self) -> Literal["true", "false", "mixture"]:
        """Get normalized veracity (fabricated -> false)."""
        return "false" if self.veracity == "fabricated" else self.veracity
    
    class Config:
        frozen = True  # Immutable


class Generation(BaseModel):
    """A generated response from a model."""
    
    gen_id: str = Field(..., description="Unique generation identifier (nanoid)")
    query_id: str = Field(..., description="Reference to Query.id")
    dataset: str = Field(..., description="Dataset name")
    presupposition_level: int = Field(..., ge=0, le=4, description="Presupposition level")
    gen_model: str = Field(..., description="Model name that generated this")
    response: str = Field(..., description="Generated response text")
    reasoning_trace: Optional[str] = Field(None, description="Reasoning trace if available")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    
    class Config:
        frozen = True


class Evaluation(BaseModel):
    """An evaluation of a generated response."""
    
    eval_id: str = Field(..., description="Unique evaluation identifier (nanoid)")
    gen_id: str = Field(..., description="Reference to Generation.gen_id")
    eval_model: str = Field(..., description="Model name that performed evaluation")
    entailment: Literal["agree", "disagree", "neutral", "error"] = Field(
        ..., description="Entailment judgment"
    )
    reasoning: str = Field(..., description="Reasoning for the judgment")
    unsure: bool = Field(..., description="Whether the evaluator was unsure")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    
    class Config:
        frozen = True


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across multiple generations."""
    
    query_id: str
    presupposition_level: int
    n_generations: int
    
    # Per-veracity metrics
    agree_rate: float
    disagree_rate: float
    neutral_rate: float
    error_rate: float
    unsure_rate: float
    
    # Consistency
    is_consistent: bool  # All generations have same entailment
    most_common_entailment: Optional[str]
    
    class Config:
        frozen = True
