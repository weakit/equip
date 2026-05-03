"""Main CLI entry point."""

import asyncio
import argparse
import sys
from dotenv import load_dotenv

from .utils import setup_logging
from .commands import generate, evaluate, analyze
from .config import ModelConfig


load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UPHILL v2 - Modern evaluation framework for LLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global options
    parser.add_argument(
        "--config",
        default="models.yaml",
        help="Path to models.yaml (default: models.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (default: stdout only)",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate responses for queries",
    )
    gen_parser.add_argument(
        "--model",
        required=True,
        help="Generator model name",
    )
    gen_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., 'uphill')",
    )
    gen_parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per query (default: 1)",
    )
    gen_parser.add_argument(
        "--run-prefix",
        default="exp-1",
        help="Run identifier (default: 'exp-1')",
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate generated responses",
    )
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Generator model name",
    )
    eval_parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Evaluator model name (default: first evaluator from config)",
    )
    eval_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., 'uphill')",
    )
    eval_parser.add_argument(
        "--run-prefix",
        default="exp-1",
        help="Run identifier (default: 'exp-1')",
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze evaluation results",
    )
    analyze_parser.add_argument(
        "--model",
        required=True,
        help="Generator model name",
    )
    analyze_parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Evaluator model name (default: first evaluator from config)",
    )
    analyze_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., 'uphill')",
    )
    analyze_parser.add_argument(
        "--run-prefix",
        default="exp-1",
        help="Run identifier (default: 'exp-1')",
    )
    analyze_parser.add_argument(
        "--per-generation",
        action="store_true",
        help="Show per-generation results",
    )
    analyze_parser.add_argument(
        "--aggregated",
        action="store_true",
        default=True,
        help="Show aggregated results (default: True)",
    )
    
    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available models",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Handle list-models command
    if args.command == "list-models":
        config = ModelConfig(args.config)
        print("\nAvailable models:")
        print("\nGenerators:")
        for model in config.list_generators():
            print(f"  - {model}")
        print("\nEvaluators:")
        for model in config.list_evaluators():
            print(f"  - {model}")
        return
    
    # Execute commands
    try:
        if args.command == "generate":
            asyncio.run(
                generate(
                    model=args.model,
                    dataset=args.dataset,
                    n_samples=args.n_samples,
                    run_prefix=args.run_prefix,
                    config_path=args.config,
                )
            )
        
        elif args.command == "evaluate":
            asyncio.run(
                evaluate(
                    generator_model=args.model,
                    evaluator_model=args.evaluator_model,
                    dataset=args.dataset,
                    run_prefix=args.run_prefix,
                    config_path=args.config,
                )
            )
        
        elif args.command == "analyze":
            analyze(
                generator_model=args.model,
                dataset=args.dataset,
                evaluator_model=args.evaluator_model,
                run_prefix=args.run_prefix,
                per_generation=args.per_generation,
                aggregated=args.aggregated,
                config_path=args.config,
            )
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
