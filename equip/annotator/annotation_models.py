"""Data models for human annotation system."""

from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class ResponseSample(BaseModel):
    """A sampled response for annotation."""
    
    sample_id: str = Field(..., description="Unique identifier for this sample")
    gen_id: str = Field(..., description="Reference to the original Generation.gen_id")
    query_id: str = Field(..., description="Reference to Query.id")
    dataset: str = Field(..., description="Dataset name")
    model: str = Field(..., description="Model that generated the response")
    presupposition_level: int = Field(..., description="Presupposition level (0-4)")
    
    # Core content for annotation
    claim: str = Field(..., description="The claim being evaluated")
    claim_veracity: str = Field(..., description="Ground truth veracity (true/false/neutral)")
    response: str = Field(..., description="Generated response text to be annotated")
    
    # Optional metadata
    reasoning_trace: Optional[str] = Field(None, description="Reasoning trace if available")
    
    # LLM judge's evaluation (for comparison after annotation)
    llm_entailment: Optional[str] = Field(None, description="LLM judge's entailment judgment")
    llm_reasoning: Optional[str] = Field(None, description="LLM judge's reasoning")
    
    class Config:
        frozen = True


class HumanAnnotation(BaseModel):
    """A human annotation of a response sample."""
    
    annotation_id: str = Field(..., description="Unique annotation identifier")
    sample_id: str = Field(..., description="Reference to ResponseSample.sample_id")
    sample_set_id: str = Field(..., description="Reference to the sample set")
    annotator_name: str = Field(..., description="Annotator's identifier/name")
    
    # Annotation choice
    entailment: Literal["agree", "disagree", "neutral"] = Field(
        ..., description="Human judgment of entailment"
    )
    
    # Optional notes
    notes: Optional[str] = Field(None, description="Optional notes from annotator")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Annotation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        frozen = False  # Allow updates for editing annotations


class SampleSet(BaseModel):
    """A set of samples for annotation."""
    
    set_id: str = Field(..., description="Unique identifier for this sample set")
    set_name: str = Field(..., description="Human-readable name for the set")
    description: str = Field(default="", description="Description of this sample set")
    
    # Configuration
    prefix: str = Field(..., description="Result prefix (e.g., 'exp-1')")
    dataset: str = Field(..., description="Dataset name")
    models: List[str] = Field(..., description="Models included in sampling")
    evaluator: Optional[str] = Field(None, description="Evaluator used (optional)")
    
    # Sampling parameters
    total_samples: int = Field(..., description="Total number of samples")
    samples_per_veracity: int = Field(..., description="Samples per veracity category")
    samples_per_model: int = Field(..., description="Samples per model")
    
    # Sample IDs (in shuffled order for presentation)
    sample_ids: List[str] = Field(..., description="List of sample IDs in this set")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        frozen = True


class AnnotationProgress(BaseModel):
    """Progress tracking for an annotator on a sample set."""
    
    annotator_name: str
    sample_set_id: str
    total_samples: int
    annotated_count: int
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.annotated_count / self.total_samples) * 100.0
    
    def __str__(self) -> str:
        """String representation for display."""
        return f"{self.annotated_count}/{self.total_samples} ({self.percentage:.1f}%)"
