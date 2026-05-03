"""Main entry point for the LLM Judge Annotation Tool."""

import argparse
import logging
import sys
from pathlib import Path

from .ui import AnnotatorUI
from .storage import AnnotationStorage
from ..explorer.data_loader import ExplorerDataLoader
from ..utils import get_results_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the annotator."""
    parser = argparse.ArgumentParser(
        description="LLM Judge Annotation Tool - Annotate model responses for judge evaluation"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory (default: auto-detect from project)"
    )
    
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Path to annotations directory (default: project_root/annotations)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the server on (default: 7861)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server hostname (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Initialize components
        results_dir = args.results_dir or get_results_dir()
        
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            sys.exit(1)
        
        logger.info(f"Using results directory: {results_dir}")
        
        data_loader = ExplorerDataLoader(results_dir)
        storage = AnnotationStorage(args.annotations_dir)
        
        logger.info(f"Using annotations directory: {storage.annotations_dir}")
        
        # Create and launch UI
        ui = AnnotatorUI(storage=storage, data_loader=data_loader)
        
        logger.info(f"Launching annotation tool on {args.server_name}:{args.port}")
        
        ui.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share
        )
        
    except KeyboardInterrupt:
        logger.info("Annotation tool stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error in annotation tool")
        sys.exit(1)


if __name__ == "__main__":
    main()
