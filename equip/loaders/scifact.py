"""SciFact dataset loader."""

import json
import logging
from pathlib import Path
from typing import List

from ..models import Query
from .base import DatasetLoader

logger = logging.getLogger(__name__)


class SciFactLoader(DatasetLoader):
    """Loader for SciFact dataset."""
    
    def __init__(self):
        super().__init__("scifact")
        self.queries_file = self.data_dir / "queries.jsonl"
    
    def load(self) -> List[Query]:
        """Load SciFact queries from JSONL file.
        
        Returns:
            List of Query objects
        """
        if not self.queries_file.exists():
            raise FileNotFoundError(f"SciFact data not found at {self.queries_file}")
        
        logger.info(f"Loading SciFact dataset from {self.queries_file}")
        queries = []
        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                # Map label to veracity
                label = data["label"]
                if label == "SUPPORT":
                    veracity = "true"
                elif label == "CONTRADICT":
                    veracity = "false"
                else:
                    # Skip other labels if any
                    continue
                
                query = Query(
                    id=data["id"],
                    claim=data["claim"],
                    veracity=veracity,
                    presupposition_level=data["presupposition_level"],
                    query_text=data["query"],
                    metadata={
                        "original_id": data.get("original_id"),
                        "subset": data.get("subset"),
                        "evidence": data.get("evidence", {}),
                        "cited_doc_ids": data.get("cited_doc_ids", []),
                    },
                )
                queries.append(query)
        
        logger.info(f"Loaded {len(queries)} queries from SciFact")
        return queries
