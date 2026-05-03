"""FoolMeTwice dataset loader."""

import json
import logging
from pathlib import Path
from typing import List

from ..models import Query
from .base import DatasetLoader

logger = logging.getLogger(__name__)


class FoolMeTwiceLoader(DatasetLoader):
    """Loader for FoolMeTwice dataset."""
    
    def __init__(self):
        super().__init__("foolme2")
        self.queries_file = self.data_dir / "queries.jsonl"
    
    def load(self) -> List[Query]:
        """Load FoolMeTwice queries from JSONL file.
        
        Returns:
            List of Query objects
        """
        if not self.queries_file.exists():
            raise FileNotFoundError(f"FoolMeTwice data not found at {self.queries_file}")
        
        logger.info(f"Loading FoolMeTwice dataset from {self.queries_file}")
        queries = []
        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                # Map label to veracity
                label = data["label"]
                if label == "SUPPORTS":
                    veracity = "true"
                elif label == "REFUTES":
                    veracity = "false"
                else:
                    # Skip NOT ENOUGH INFO or other labels
                    continue
                
                query = Query(
                    id=data["id"],
                    claim=data["claim"],
                    veracity=veracity,
                    presupposition_level=data["presupposition_level"],
                    query_text=data["query"],
                    metadata={
                        "category": data.get("category"),
                        "wikipedia_page": data.get("wikipedia_page"),
                        "evidence": data.get("evidence", []),
                    },
                )
                queries.append(query)
        
        logger.info(f"Loaded {len(queries)} queries from FoolMeTwice")
        return queries
