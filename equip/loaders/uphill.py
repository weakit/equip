"""UPHILL dataset loader."""

import logging
from typing import List
import pandas as pd

from ..models import Query
from .base import DatasetLoader

logger = logging.getLogger(__name__)


class UphillLoader(DatasetLoader):
    """Loader for the UPHILL dataset."""
    
    # Mapping from string presupposition levels to integers
    PRESUPPOSITION_MAP = {
        "Neutral": 0,
        "Mild Presupposition": 1,
        "Unequivocal Presupposition": 2,
        "Writing Request": 3,
        "Writing Demand": 4,
    }
    
    def __init__(self):
        super().__init__("uphill")
        self.queries_file = self.data_dir / "queries.csv"
    
    def load(self) -> List[Query]:
        """Load UPHILL dataset from queries.csv."""
        if not self.queries_file.exists():
            raise FileNotFoundError(f"UPHILL dataset not found at {self.queries_file}")
        
        logger.info(f"Loading UPHILL dataset from {self.queries_file}")
        df = pd.read_csv(self.queries_file)
        
        # Rename first column to ID if needed
        if df.columns[0] != "ID":
            df = df.rename(columns={df.columns[0]: "ID"})
        
        queries = []
        for _, row in df.iterrows():
            # Convert presupposition level to integer
            presup_str = row["presupposition_level"]
            presup_int = self.PRESUPPOSITION_MAP.get(presup_str)
            
            if presup_int is None:
                logger.warning(f"Unknown presupposition level: {presup_str}, skipping row")
                continue
            
            query = Query(
                id=row["claim_id"],
                claim=row["claim"],
                veracity=row["claim_veracity"],
                presupposition_level=presup_int,
                query_text=row["query_with_presupposition"],
                metadata={
                    "source_db": row.get("source_db", ""),
                    "date_published": row.get("date_published", ""),
                    "veracity_explanation": row.get("veracity_explanation", ""),
                    "subjects": row.get("subjects", ""),
                }
            )
            queries.append(query)
        
        logger.info(f"Loaded {len(queries)} queries from UPHILL dataset")
        return queries
