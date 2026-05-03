"""Storage system for human annotations."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from ..utils import get_project_root, ensure_dir
from .annotation_models import ResponseSample, HumanAnnotation, SampleSet, AnnotationProgress

logger = logging.getLogger(__name__)


class AnnotationStorage:
    """Handles saving and loading of annotation data."""
    
    def __init__(self, annotations_dir: Optional[Path] = None):
        """Initialize annotation storage.
        
        Args:
            annotations_dir: Directory for storing annotations. If None, uses default.
        """
        if annotations_dir is None:
            annotations_dir = get_project_root() / "annotations"
        
        self.annotations_dir = ensure_dir(annotations_dir)
        self.sample_sets_dir = ensure_dir(self.annotations_dir / "sample_sets")
        self.annotations_file = self.annotations_dir / "annotations.jsonl"
        
        logger.info(f"Initialized AnnotationStorage at {self.annotations_dir}")
    
    def save_sample_set(
        self,
        sample_set: SampleSet,
        response_samples: List[ResponseSample]
    ) -> None:
        """Save a sample set and its samples.
        
        Args:
            sample_set: The SampleSet to save
            response_samples: List of ResponseSample objects in the set
        """
        set_dir = ensure_dir(self.sample_sets_dir / sample_set.set_id)
        
        # Save sample set metadata
        metadata_file = set_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(sample_set.model_dump(), f, indent=2, default=str)
        
        # Save response samples
        samples_file = set_dir / "samples.jsonl"
        with open(samples_file, "w", encoding="utf-8") as f:
            for sample in response_samples:
                f.write(json.dumps(sample.model_dump(), default=str) + "\n")
        
        logger.info(f"Saved sample set '{sample_set.set_name}' with {len(response_samples)} samples")
    
    def load_sample_set(self, set_id: str) -> Optional[SampleSet]:
        """Load a sample set by ID.
        
        Args:
            set_id: The sample set ID to load
        
        Returns:
            SampleSet or None if not found
        """
        set_dir = self.sample_sets_dir / set_id
        metadata_file = set_dir / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Sample set not found: {set_id}")
            return None
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return SampleSet(**data)
    
    def load_sample_set_samples(self, set_id: str) -> List[ResponseSample]:
        """Load all samples for a sample set.
        
        Args:
            set_id: The sample set ID
        
        Returns:
            List of ResponseSample objects
        """
        set_dir = self.sample_sets_dir / set_id
        samples_file = set_dir / "samples.jsonl"
        
        if not samples_file.exists():
            logger.warning(f"Samples file not found for set: {set_id}")
            return []
        
        samples = []
        with open(samples_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(ResponseSample(**data))
        
        return samples
    
    def list_sample_sets(self) -> List[SampleSet]:
        """List all available sample sets.
        
        Returns:
            List of SampleSet objects, sorted by creation time (newest first)
        """
        sample_sets = []
        
        for set_dir in self.sample_sets_dir.iterdir():
            if set_dir.is_dir():
                sample_set = self.load_sample_set(set_dir.name)
                if sample_set:
                    sample_sets.append(sample_set)
        
        # Sort by creation time (newest first)
        sample_sets.sort(key=lambda s: s.created_at, reverse=True)
        
        return sample_sets
    
    def save_annotation(self, annotation: HumanAnnotation) -> None:
        """Save or update a single annotation.
        
        This appends to the annotations file. Use load_annotations() to get the latest state.
        
        Args:
            annotation: The HumanAnnotation to save
        """
        # Update timestamp
        annotation.updated_at = datetime.utcnow()
        
        # Append to annotations file
        with open(self.annotations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(annotation.model_dump(), default=str) + "\n")
        
        logger.debug(f"Saved annotation {annotation.annotation_id}")
    
    def load_annotations(
        self,
        sample_set_id: Optional[str] = None,
        annotator_name: Optional[str] = None
    ) -> List[HumanAnnotation]:
        """Load annotations, optionally filtered.
        
        Note: Returns the LATEST version of each annotation (by annotation_id).
        
        Args:
            sample_set_id: Optional filter by sample set ID
            annotator_name: Optional filter by annotator name
        
        Returns:
            List of HumanAnnotation objects (latest versions only)
        """
        if not self.annotations_file.exists():
            return []
        
        # Load all annotations and keep track of latest by annotation_id
        latest_annotations: Dict[str, HumanAnnotation] = {}
        
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    annotation = HumanAnnotation(**data)
                    
                    # Apply filters
                    if sample_set_id and annotation.sample_set_id != sample_set_id:
                        continue
                    if annotator_name and annotation.annotator_name != annotator_name:
                        continue
                    
                    # Keep the latest version (by updated_at)
                    ann_id = annotation.annotation_id
                    if ann_id not in latest_annotations or \
                       annotation.updated_at > latest_annotations[ann_id].updated_at:
                        latest_annotations[ann_id] = annotation
        
        return list(latest_annotations.values())
    
    def get_annotation_for_sample(
        self,
        sample_id: str,
        sample_set_id: str,
        annotator_name: str
    ) -> Optional[HumanAnnotation]:
        """Get the annotation for a specific sample by a specific annotator.
        
        Args:
            sample_id: The sample ID
            sample_set_id: The sample set ID
            annotator_name: The annotator name
        
        Returns:
            HumanAnnotation or None if not found
        """
        annotations = self.load_annotations(
            sample_set_id=sample_set_id,
            annotator_name=annotator_name
        )
        
        for annotation in annotations:
            if annotation.sample_id == sample_id:
                return annotation
        
        return None
    
    def get_annotator_progress(
        self,
        sample_set_id: str,
        annotator_name: str
    ) -> AnnotationProgress:
        """Get annotation progress for an annotator on a sample set.
        
        Args:
            sample_set_id: The sample set ID
            annotator_name: The annotator name
        
        Returns:
            AnnotationProgress object
        """
        sample_set = self.load_sample_set(sample_set_id)
        if not sample_set:
            return AnnotationProgress(
                annotator_name=annotator_name,
                sample_set_id=sample_set_id,
                total_samples=0,
                annotated_count=0
            )
        
        annotations = self.load_annotations(
            sample_set_id=sample_set_id,
            annotator_name=annotator_name
        )
        
        return AnnotationProgress(
            annotator_name=annotator_name,
            sample_set_id=sample_set_id,
            total_samples=sample_set.total_samples,
            annotated_count=len(annotations)
        )
    
    def list_annotators_for_set(self, sample_set_id: str) -> List[str]:
        """List all annotators who have annotated a sample set.
        
        Args:
            sample_set_id: The sample set ID
        
        Returns:
            List of annotator names
        """
        annotations = self.load_annotations(sample_set_id=sample_set_id)
        annotators = set(a.annotator_name for a in annotations)
        return sorted(annotators)
    
    def export_annotations_for_analysis(
        self,
        sample_set_id: str,
        output_file: Path
    ) -> None:
        """Export annotations for a sample set in a format suitable for analysis.
        
        Creates a JSON file with samples and all their annotations.
        
        Args:
            sample_set_id: The sample set ID
            output_file: Path to output JSON file
        """
        sample_set = self.load_sample_set(sample_set_id)
        if not sample_set:
            raise ValueError(f"Sample set not found: {sample_set_id}")
        
        samples = self.load_sample_set_samples(sample_set_id)
        annotations = self.load_annotations(sample_set_id=sample_set_id)
        
        # Organize annotations by sample_id
        annotations_by_sample: Dict[str, List[HumanAnnotation]] = {}
        for annotation in annotations:
            sample_id = annotation.sample_id
            if sample_id not in annotations_by_sample:
                annotations_by_sample[sample_id] = []
            annotations_by_sample[sample_id].append(annotation)
        
        # Build export data
        export_data = {
            "sample_set": sample_set.model_dump(),
            "samples": []
        }
        
        for sample in samples:
            sample_data = sample.model_dump()
            sample_data["annotations"] = [
                a.model_dump() for a in annotations_by_sample.get(sample.sample_id, [])
            ]
            export_data["samples"].append(sample_data)
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported annotations to {output_file}")
