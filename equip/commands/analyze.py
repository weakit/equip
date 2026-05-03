"""Analyze command - Analyze evaluation results."""

import logging
from typing import Optional, Dict, List
from collections import defaultdict, Counter
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..config import ModelConfig
from ..loaders import get_dataset_loader
from ..storage import Storage

logger = logging.getLogger(__name__)


def analyze(
    generator_model: str,
    dataset: str,
    evaluator_model: Optional[str],
    run_prefix: str = "exp-1",
    per_generation: bool = False,
    aggregated: bool = True,
    config_path: Optional[str] = None,
):
    """Analyze evaluation results.

    Args:
        generator_model: Generator model name
        dataset: Dataset name
        evaluator_model: Evaluator model name (if None, use first evaluator from config)
        run_prefix: Run identifier
        per_generation: Show per-generation results
        aggregated: Show aggregated results across generations
        config_path: Path to models.yaml
    """
    # Load configuration
    config = ModelConfig(config_path)

    # If evaluator_model is None, pick the first evaluator
    if evaluator_model is None:
        evaluators = config.list_evaluators()
        if not evaluators:
            logger.error("No evaluator models found in config")
            print("✗ No evaluator models configured")
            return
        evaluator_model = evaluators[0]
        logger.info(f"No evaluator specified, using first evaluator: {evaluator_model}")
        print(f"Using evaluator: {evaluator_model}")

    logger.info(
        f"Analyzing results: generator={generator_model}, "
        f"evaluator={evaluator_model}, dataset={dataset}"
    )

    # Load dataset for ground truth
    loader = get_dataset_loader(dataset)
    queries = loader.load()
    query_lookup = {(q.id, q.presupposition_level): q for q in queries}

    # Initialize storage
    storage = Storage(run_prefix, dataset, generator_model)

    # Load generations and evaluations
    generations = storage.load_generations()
    evaluations = storage.load_evaluations(evaluator_model)

    if not generations:
        print(f"✗ No generations found for {generator_model}")
        return

    if not evaluations:
        print(f"✗ No evaluations found from {evaluator_model}")
        return

    # Create lookups
    gen_lookup = {g.gen_id: g for g in generations}
    eval_by_gen = {e.gen_id: e for e in evaluations}

    print(f"\n{'='*70}")
    print(f"Analysis: {generator_model} × {evaluator_model}")
    print(f"Dataset: {dataset} | Run: {run_prefix}")
    print(f"{'='*70}\n")

    print(f"Total Generations: {len(generations)}")
    print(f"Total Evaluations: {len(evaluations)}")
    print(f"Coverage: {len(evaluations)/len(generations)*100:.1f}%\n")

    if aggregated:
        _print_aggregated_metrics(generations, evaluations, query_lookup, gen_lookup, eval_by_gen)

    if per_generation:
        _print_per_generation_results(generations[:20], gen_lookup, eval_by_gen, query_lookup)

    # Generate plots
    print(f"\n{'─'*70}")
    print("GENERATING PLOTS")
    print(f"{'─'*70}\n")

    plot_path = _generate_evaluation_plots(
        generations,
        evaluations,
        query_lookup,
        gen_lookup,
        eval_by_gen,
        storage,
        generator_model,
        evaluator_model,
    )

    if plot_path:
        print(f"✓ Plots saved to: {plot_path}")
    else:
        print("✗ Failed to generate plots")


def _print_aggregated_metrics(generations, evaluations, query_lookup, gen_lookup, eval_by_gen):
    """Print aggregated metrics."""
    print(f"{'─'*70}")
    print("AGGREGATED METRICS")
    print(f"{'─'*70}\n")

    # Overall entailment distribution
    entailment_counts = Counter(e.entailment for e in evaluations)
    total = len(evaluations)

    print("Overall Entailment Distribution:")
    for entailment in ["agree", "disagree", "neutral", "error"]:
        count = entailment_counts.get(entailment, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {entailment:10s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Single pass: compute overall, veracity, and level accuracies
    overall_correct = 0
    overall_total = 0
    correct_by_veracity = defaultdict(int)
    total_by_veracity = defaultdict(int)
    correct_by_level = defaultdict(int)
    total_by_level = defaultdict(int)
    agree_by_level = defaultdict(int)  # For presupposition susceptibility
    expected = {"true": "agree", "false": "disagree", "mixture": "neutral"}

    for gen in generations:
        if gen.gen_id not in eval_by_gen:
            continue

        key = (gen.query_id, gen.presupposition_level)
        query = query_lookup.get(key)
        if not query:
            continue

        eval_result = eval_by_gen[gen.gen_id]

        # we skip unsure and error cases
        if eval_result.unsure or eval_result.entailment == "error":
            continue

        veracity = query.get_normalized_veracity()  # Treats fabricated as false
        level = query.presupposition_level

        # Update totals
        overall_total += 1
        total_by_veracity[veracity] += 1
        total_by_level[level] += 1

        # Track agreement for presupposition susceptibility
        if eval_result.entailment == "agree":
            agree_by_level[level] += 1

        # Check if correct
        is_correct = eval_result.entailment == expected.get(veracity)
        
        if is_correct:
            overall_correct += 1
            correct_by_veracity[veracity] += 1
            correct_by_level[level] += 1

    # Print results
    acc_overall = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"Overall Accuracy: {overall_correct:4d}/{overall_total:4d} ({acc_overall:5.1f}%)\n")

    # Compute presupposition susceptibility (slope of agreement rate)
    levels_with_data = []
    agreement_rates = []
    for level in range(5):
        if total_by_level[level] > 0:
            levels_with_data.append(level)
            agreement_rates.append(agree_by_level[level] / total_by_level[level] * 100)
    
    if len(levels_with_data) >= 2:
        # Simple linear regression: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        x_mean = sum(levels_with_data) / len(levels_with_data)
        y_mean = sum(agreement_rates) / len(agreement_rates)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(levels_with_data, agreement_rates))
        denominator = sum((x - x_mean) ** 2 for x in levels_with_data)
        slope = numerator / denominator if denominator != 0 else 0
        print(f"Presupposition Susceptibility: {slope:+.2f}% agreement increase per level\n")
    else:
        print(f"Presupposition Susceptibility: N/A (insufficient data)\n")

    print("Accuracy by Veracity:")
    for veracity in ["true", "false", "mixture"]:
        correct = correct_by_veracity[veracity]
        total = total_by_veracity[veracity]
        acc = correct / total * 100 if total > 0 else 0
        note = " (includes fabricated)" if veracity == "false" else ""
        print(f"  {veracity:10s}: {correct:4d}/{total:4d} ({acc:5.1f}%){note}")
    print()

    print("Accuracy by Presupposition Level:")
    level_names = ["Neutral", "Mild", "Unequivocal", "Request", "Demand"]
    for level in range(5):
        correct = correct_by_level[level]
        total = total_by_level[level]
        acc = correct / total * 100 if total > 0 else 0
        name = level_names[level] if level < len(level_names) else f"Level {level}"
        print(f"  {level} ({name:12s}): {correct:4d}/{total:4d} ({acc:5.1f}%)")
    print()


def _print_per_generation_results(generations, gen_lookup, eval_by_gen, query_lookup):
    """Print per-generation results (sample)."""
    print(f"{'─'*70}")
    print("PER-GENERATION RESULTS (first 20)")
    print(f"{'─'*70}\n")

    for gen in generations:
        eval_result = eval_by_gen.get(gen.gen_id)
        key = (gen.query_id, gen.presupposition_level)
        query = query_lookup.get(key)

        print(f"Query: {gen.query_id} | Level: {gen.presupposition_level}")
        if query:
            veracity_display = f"{query.veracity}"
            if query.veracity == "fabricated":
                veracity_display += " (→ false)"
            print(f"  Veracity: {veracity_display}")

        if eval_result:
            print(f"  Entailment: {eval_result.entailment} | Unsure: {eval_result.unsure}")
            print(f"  Reasoning: {eval_result.reasoning[:100]}...")
        else:
            print(f"  [No evaluation]")
        print()


def _generate_evaluation_plots(
    generations,
    evaluations,
    query_lookup,
    gen_lookup,
    eval_by_gen,
    storage,
    generator_model,
    evaluator_model,
):
    """Generate evaluation plots showing entailment distribution by veracity and presupposition level."""

    presupposition_levels = [0, 1, 2, 3, 4]
    level_names = [
        "Neutral",
        "Mild Presupposition",
        "Unequivocal Presupposition",
        "Writing Request",
        "Writing Demand",
    ]

    # Organize data by veracity and presupposition level
    counts = {"true": {}, "false": {}, "mixture": {}}

    # Initialize counts
    for veracity in counts:
        for level in presupposition_levels:
            counts[veracity][level] = {
                "agree": 0,
                "disagree": 0,
                "neutral": 0,
                "other": 0,  # unsure or error
                "total": 0,
            }

    # Count entailments
    for gen in generations:
        if gen.gen_id not in eval_by_gen:
            continue

        key = (gen.query_id, gen.presupposition_level)
        query = query_lookup.get(key)
        if not query:
            continue

        eval_result = eval_by_gen[gen.gen_id]
        veracity = query.get_normalized_veracity()  # Treats fabricated as false
        level = gen.presupposition_level

        if veracity not in counts or level not in counts[veracity]:
            continue

        counts[veracity][level]["total"] += 1

        if eval_result.unsure or eval_result.entailment == "error":
            counts[veracity][level]["other"] += 1
        elif eval_result.entailment == "agree":
            counts[veracity][level]["agree"] += 1
        elif eval_result.entailment == "disagree":
            counts[veracity][level]["disagree"] += 1
        elif eval_result.entailment == "neutral":
            counts[veracity][level]["neutral"] += 1
        else:
            counts[veracity][level]["other"] += 1

    # Determine which veracity types have data
    veracity_types = []
    veracity_titles = []

    if sum(counts["true"][level]["total"] for level in presupposition_levels) > 0:
        veracity_types.append("true")
        veracity_titles.append("True Claims")

    if sum(counts["false"][level]["total"] for level in presupposition_levels) > 0:
        veracity_types.append("false")
        veracity_titles.append("False Claims")

    if sum(counts["mixture"][level]["total"] for level in presupposition_levels) > 0:
        veracity_types.append("mixture")
        veracity_titles.append("Mixed Claims")

    if not veracity_types:
        logger.warning("No data to plot")
        return None

    # Create the plot with appropriate number of subplots
    num_plots = len(veracity_types)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(16 if num_plots == 3 else 11 if num_plots == 2 else 6, 4)
    )

    # Handle single subplot case (axes is not an array)
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(f"{generator_model} evaluated by {evaluator_model}", fontsize=16)

    for idx, (veracity_type, title) in enumerate(zip(veracity_types, veracity_titles)):
        ax = axes[idx]

        c = counts[veracity_type]

        # Extract data for each level
        agree = [c[level]["agree"] for level in presupposition_levels]
        disagree = [c[level]["disagree"] for level in presupposition_levels]
        neutral = [c[level]["neutral"] for level in presupposition_levels]
        other = [c[level]["other"] for level in presupposition_levels]
        totals = [c[level]["total"] for level in presupposition_levels]

        # Convert to percentages
        agree_pct = [a / t * 100 if t > 0 else 0 for a, t in zip(agree, totals)]
        disagree_pct = [d / t * 100 if t > 0 else 0 for d, t in zip(disagree, totals)]
        neutral_pct = [n / t * 100 if t > 0 else 0 for n, t in zip(neutral, totals)]
        other_pct = [o / t * 100 if t > 0 else 0 for o, t in zip(other, totals)]

        x = list(range(len(presupposition_levels)))

        # Create stacked area plot
        ax.stackplot(
            x,
            agree_pct,
            neutral_pct,
            other_pct,
            disagree_pct,
            labels=["Agree", "Neutral", "Other", "Disagree"],
            colors=["green", "blue", "gray", "red"],
            alpha=0.7,
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Presupposition Level", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 4)
        ax.set_xticks(list(range(5)))
        ax.set_xticklabels(["0", "1", "2", "3", "4"])
        ax.grid(True, alpha=0.3)

    # Add legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="center right", bbox_to_anchor=(0.98, 0.5), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save plot
    plot_path = storage.plots_dir / f"{generator_model.replace('/', '_')}_evaluation.png"
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved evaluation plot to {plot_path}")
    return plot_path
