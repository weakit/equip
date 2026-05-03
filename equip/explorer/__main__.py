"""Main entry point for the UPHILL Response Explorer."""

import argparse
import logging
import sys
from pathlib import Path

from .ui import ExplorerUI
from .data_loader import ExplorerDataLoader
from ..utils import get_results_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the explorer."""
    parser = argparse.ArgumentParser(
        description="UPHILL Response Explorer - Browse and analyze evaluation results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory (default: auto-detect from project)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
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
        help="Server name/IP to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize data loader
    try:
        results_dir = args.results_dir or get_results_dir()
        logger.info(f"Using results directory: {results_dir}")
        
        data_loader = ExplorerDataLoader(results_dir=results_dir)
        
        # Check if we have any data
        prefixes = data_loader.get_available_prefixes()
        if not prefixes:
            logger.error(f"No result prefixes found in {results_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(prefixes)} result prefixes: {', '.join(prefixes)}")
        
    except Exception as e:
        logger.exception("Failed to initialize data loader")
        sys.exit(1)
    
    # Create and launch UI
    try:
        ui = ExplorerUI(data_loader=data_loader)
        
        logger.info(f"Launching explorer on {args.server_name}:{args.port}")
        if args.share:
            logger.info("Creating public share link...")
        
        ui.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    except Exception as e:
        logger.exception("Failed to launch UI")
        sys.exit(1)


if __name__ == "__main__":
    main()
