"""Computes and visualizes metrics.

Usage (examples):
  python -m equip.metrics --models gpt-5-mini-minimal,gpt-oss-20b-medium --prefix final-1
  python -m equip.metrics --models qwen3-8b-thinking --evaluator gpt-5-mini-minimal --prefix exp-1

"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import numpy as np
import pandas as pd
from scipy.stats import bootstrap as _scipy_bootstrap, ttest_rel

# Import from package
from equip.config import ModelConfig
from equip.loaders import get_dataset_loader
from equip.storage import Storage

logger = logging.getLogger(__name__)


VERACITY_TRUE = "true"
VERACITY_FALSE = "false"
VERACITY_MIXTURE = "mixture"
LEVELS = [0, 1, 2, 3, 4]
LEVEL_NAMES = [
    "Neutral",
    "Mild",
    "Unequivocal",
    "Request",
    "Demand",
]

def _bootstrap_ci(
    data_array: np.ndarray,
    n_resamples: int = 9999,
    rng: int = 0,
) -> Tuple[float, float]:
    """Return (low, high) 95% CI for an array of proportions, expressed as percentages.

    Uses a percentile bootstrap over ``n_resamples`` resamples of the data array.
    """
    if len(data_array) == 0:
        return (0.0, 0.0)
    pct = np.mean(data_array)
    if np.all(data_array == data_array[0]):
        return (pct * 100, pct * 100)
    res = _scipy_bootstrap(
        (data_array,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=0.95,
        batch=1000,
        method="BCa",
        random_state=rng,
    )

    return (res.confidence_interval.low * 100, res.confidence_interval.high * 100)


MODEL_FRIENDLY_NAMES = {
    "gpt-oss-20b-off": "GPT-OSS 20B (off)",
    "gpt-oss-20b-medium": "GPT-OSS 20B (med.)",
    "gemini-2.5-flash-no-thinking": "Gemini 2.5 Flash (off)",
    "gemini-2.5-flash-thinking": "Gemini 2.5 Flash (on)",
}


def list_available_datasets() -> List[str]:
    """Return the dataset names present in the repository.

    This script focuses on these datasets; update if new datasets are added.
    """
    return ["foolmetwice", "scifact", "uphill"]


def pick_evaluator(config_path: Optional[str], evaluator_model: Optional[str]) -> str:
    """Pick evaluator from config if not provided."""
    if evaluator_model:
        return evaluator_model
    config = ModelConfig(config_path)
    evaluators = config.list_evaluators()
    if not evaluators:
        raise RuntimeError("No evaluator models found in config")
    logger.info(f"No evaluator specified, using first evaluator: {evaluators[0]}")
    return evaluators[0]


def aggregate_counts_for_model_group(
    models: List[str],
    evaluator_model: str,
    run_prefix: str,
    datasets: List[str],
) -> Dict[str, Dict[str, int]]:
    """Aggregate entailment counts across all levels for a group of models.

    'other' (unsure/error) evaluations are excluded at the per-instance level
    in aggregate_counts_for_model, so totals here reflect only valid instances.

    Returns structure:
        {
          'true': { 'agree': float, 'disagree': float, 'neutral': float, 'total': int },
          'false': { 'agree': float, 'disagree': float, 'neutral': float, 'total': int },
        }
    """
    # Initialize counts
    counts = {
        VERACITY_TRUE: {"agree": 0, "disagree": 0, "neutral": 0, "total": 0},
        VERACITY_FALSE: {"agree": 0, "disagree": 0, "neutral": 0, "total": 0},
    }

    for model in models:
        model_counts = aggregate_counts_for_model(model, evaluator_model, run_prefix, datasets)
        
        # Aggregate across all levels, excluding 'other'
        for veracity in [VERACITY_TRUE, VERACITY_FALSE]:
            for lvl in LEVELS:
                counts[veracity]["agree"] += model_counts[veracity][lvl]["agree"]
                counts[veracity]["disagree"] += model_counts[veracity][lvl]["disagree"]
                counts[veracity]["neutral"] += model_counts[veracity][lvl]["neutral"]
                counts[veracity]["total"] += model_counts[veracity][lvl]["total"]

    return counts


def plot_overview_comparison(
    nr_counts: Dict[str, Dict[str, int]],
    r_counts: Dict[str, Dict[str, int]],
    run_prefix: str,
    evaluator_model: str,
    output_path: Path,
):
    """Plot horizontal stacked bar chart comparing NR vs R models.
    
    Creates a 2-row figure (True top, False bottom) with horizontal bars showing
    Agree/Neutral/Disagree distribution for non-reasoning (NR) and reasoning (R) models.
    """
    fig, axes = plt.subplots(2, 1, figsize=(4, 5))

    categories = ['NR', 'R']
    
    for row, (veracity, title) in enumerate([(VERACITY_TRUE, 'True Claims'), (VERACITY_FALSE, 'False Claims')]):
        ax = axes[row]
        
        # Get counts for both groups
        counts_list = [nr_counts, r_counts]
        
        # Calculate percentages for each group
        agree_pcts = []
        neutral_pcts = []
        disagree_pcts = []
        
        for counts in counts_list:
            total = counts[veracity]["total"]
            if total > 0:
                agree_pcts.append(counts[veracity]["agree"] / total * 100)
                neutral_pcts.append(counts[veracity]["neutral"] / total * 100)
                disagree_pcts.append(counts[veracity]["disagree"] / total * 100)
            else:
                agree_pcts.append(0)
                neutral_pcts.append(0)
                disagree_pcts.append(0)
        
        # Create horizontal stacked bars (NR on top = position 1, R on bottom = position 0)
        # Use larger height to make bars wider
        y_pos = [1, 0]
        
        # Plot bars (agree, then neutral, then disagree) without edges
        ax.barh(y_pos, agree_pcts, height=0.7, color='green', alpha=0.7, edgecolor='none', 
                label='Agree' if row == 0 else '')
        ax.barh(y_pos, neutral_pcts, left=agree_pcts, height=0.7, color='blue', alpha=0.7, edgecolor='none',
                label='Neutral' if row == 0 else '')
        ax.barh(y_pos, disagree_pcts, left=[a+n for a, n in zip(agree_pcts, neutral_pcts)], 
                height=0.7, color='red', alpha=0.7, edgecolor='none', label='Disagree' if row == 0 else '')
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 100])
        ax.set_xticklabels(['0%', '100%'])
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0, pad=10)
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Title above each subplot
        ax.set_title(title, fontsize=14, pad=10)
    
    # Legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=11,
                        frameon=False, handlelength=0.9, handleheight=1, handletextpad=0.5, columnspacing=1.0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved overview plot to {output_path}")
    return output_path


def compute_model_metrics(
    models: List[str],
    evaluator_model: str,
    run_prefix: str,
    datasets: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy and decisiveness for each model.
    
    Accuracy: proportion of correct responses (agree for true, disagree for false)
    Decisiveness: proportion of non-neutral responses (agree + disagree) / total
    
    Returns:
        Dict mapping model name to {'accuracy': float, 'decisiveness': float}
    """
    results = {}
    
    for model in models:
        counts = aggregate_counts_for_model(model, evaluator_model, run_prefix, datasets)
        
        total_correct = 0
        total_all = 0
        total_decisive = 0  # agree + disagree
        
        for veracity in [VERACITY_TRUE, VERACITY_FALSE, VERACITY_MIXTURE]:
            for lvl in LEVELS:
                c = counts[veracity][lvl]
                
                # Count correct responses
                if veracity == VERACITY_TRUE:
                    total_correct += c["agree"]
                elif veracity == VERACITY_FALSE:
                    total_correct += c["disagree"]
                else:  # mixture
                    total_correct += c["neutral"]
                
                # Count all responses (excluding 'other' which are errors/unsure)
                total_all += c["agree"] + c["disagree"] + c["neutral"]
                
                # Count decisive responses (non-neutral)
                total_decisive += c["agree"] + c["disagree"]
        
        accuracy = (total_correct / total_all * 100) if total_all > 0 else 0.0
        decisiveness = (total_decisive / total_all * 100) if total_all > 0 else 0.0
        
        results[model] = {
            "accuracy": accuracy,
            "decisiveness": decisiveness,
        }
    
    return results


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    run_prefix: str,
    evaluator_model: str,
    output_path: Path,
):
    """Plot grouped horizontal bar chart showing accuracy and decisiveness per model.
    
    Models are stacked vertically with two bars each (accuracy and decisiveness).
    """
    models = list(reversed(list(metrics.keys())))
    n_models = len(models)
    
    if n_models == 0:
        logger.warning("No models to plot.")
        return None
    
    # Prepare data
    accuracies = [metrics[m]["accuracy"] for m in models]
    decisiveness = [metrics[m]["decisiveness"] for m in models]
    
    # Create figure with more vertical space
    fig, ax = plt.subplots(figsize=(4, n_models * 0.8))
    
    # Y positions for bars
    y_pos = range(n_models)
    bar_height = 0.3
    
    # Create grouped horizontal bars
    ax.barh([y + bar_height/2 for y in y_pos], accuracies, height=bar_height, 
            color='#afdc8f', alpha=0.8, edgecolor='none', label='Accuracy')
    ax.barh([y - bar_height/2 for y in y_pos], decisiveness, height=bar_height,
            color='#4394e5', alpha=0.8, edgecolor='none', label='Decisiveness')
    
    # Add model names above each bar group
    for i, model in enumerate(models):
        ax.text(1.75, i + 0.375, model, ha='left', va='bottom', fontsize=10)    
    # Styling
    ax.set_yticks([])  # Remove y-ticks
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Percentage', fontsize=12)
    ax.tick_params(axis='x', length=0)
    
    # Remove all spines except add lines at left and right
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add vertical lines at 0 and 100
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=100, color='black', linewidth=0.8)
    
    # Add legend below, horizontal
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11,
             handlelength=0.9, handleheight=1, handletextpad=0.5, columnspacing=1.0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved metrics plot to {output_path}")
    return output_path


def aggregate_counts_for_model(
    generator_model: str,
    evaluator_model: str,
    run_prefix: str,
    datasets: List[str],
) -> Dict[str, Dict[int, Dict[str, any]]]:
    """Aggregate per-instance entailment counts per level for True, False, and Mixture.

    For each instance, proportions are computed excluding 'other' (unsure/error)
    evaluations from both numerator and denominator. Instances where all evaluations
    are 'other' are skipped entirely and not counted in 'total'.

    Returns structure:
        {
          'true': { level: { 'agree': float, 'disagree': float, 'neutral': float,
                             'total': int, 'agree_arr': ndarray, 'disagree_arr': ndarray,
                             'neutral_arr': ndarray } },
          ...
        }
    """
    # Collect items first: instances[(veracity, level)][query_id] = { 'agree': 0, ... }
    items = {
        VERACITY_TRUE: {lvl: defaultdict(lambda: {"agree": 0, "disagree": 0, "neutral": 0, "other": 0, "total": 0}) for lvl in LEVELS},
        VERACITY_FALSE: {lvl: defaultdict(lambda: {"agree": 0, "disagree": 0, "neutral": 0, "other": 0, "total": 0}) for lvl in LEVELS},
        VERACITY_MIXTURE: {lvl: defaultdict(lambda: {"agree": 0, "disagree": 0, "neutral": 0, "other": 0, "total": 0}) for lvl in LEVELS},
    }

    for dataset in datasets:
        loader = get_dataset_loader(dataset)
        queries = loader.load()
        query_lookup = {(q.id, q.presupposition_level): q for q in queries}

        storage = Storage(run_prefix, dataset, generator_model)
        generations = storage.load_generations()
        evaluations = storage.load_evaluations(evaluator_model)

        if not generations or not evaluations:
            logger.warning(f"Skipping dataset '{dataset}' for model '{generator_model}' due to missing data.")
            continue

        eval_by_gen = {e.gen_id: e for e in evaluations}

        for gen in generations:
            eval_result = eval_by_gen.get(gen.gen_id)
            if not eval_result: continue
            query = query_lookup.get((gen.query_id, gen.presupposition_level))
            if not query: continue
            veracity = query.get_normalized_veracity()
            level = gen.presupposition_level
            if level not in items[VERACITY_TRUE]: continue

            c = items[veracity][level][gen.query_id]
            c["total"] += 1
            if getattr(eval_result, "unsure", False) or getattr(eval_result, "entailment", None) == "error":
                c["other"] += 1
            elif eval_result.entailment == "agree":
                c["agree"] += 1
            elif eval_result.entailment == "disagree":
                c["disagree"] += 1
            elif eval_result.entailment == "neutral":
                c["neutral"] += 1
            else:
                c["other"] += 1

    counts = {
        VERACITY_TRUE: {lvl: {"agree": 0.0, "disagree": 0.0, "neutral": 0.0, "total": 0, "agree_arr": [], "disagree_arr": [], "neutral_arr": []} for lvl in LEVELS},
        VERACITY_FALSE: {lvl: {"agree": 0.0, "disagree": 0.0, "neutral": 0.0, "total": 0, "agree_arr": [], "disagree_arr": [], "neutral_arr": []} for lvl in LEVELS},
        VERACITY_MIXTURE: {lvl: {"agree": 0.0, "disagree": 0.0, "neutral": 0.0, "total": 0, "agree_arr": [], "disagree_arr": [], "neutral_arr": []} for lvl in LEVELS},
    }

    for veracity in [VERACITY_TRUE, VERACITY_FALSE, VERACITY_MIXTURE]:
        for lvl in LEVELS:
            level_items = items[veracity][lvl]
            for q_id, c in level_items.items():
                valid = c["agree"] + c["disagree"] + c["neutral"]
                if valid == 0:
                    continue
                a_prop = c["agree"] / valid
                d_prop = c["disagree"] / valid
                n_prop = c["neutral"] / valid

                counts[veracity][lvl]["total"] += 1
                counts[veracity][lvl]["agree"] += a_prop
                counts[veracity][lvl]["disagree"] += d_prop
                counts[veracity][lvl]["neutral"] += n_prop

                counts[veracity][lvl]["agree_arr"].append(a_prop)
                counts[veracity][lvl]["disagree_arr"].append(d_prop)
                counts[veracity][lvl]["neutral_arr"].append(n_prop)

            counts[veracity][lvl]["agree_arr"] = np.array(counts[veracity][lvl]["agree_arr"])
            counts[veracity][lvl]["disagree_arr"] = np.array(counts[veracity][lvl]["disagree_arr"])
            counts[veracity][lvl]["neutral_arr"] = np.array(counts[veracity][lvl]["neutral_arr"])

    return counts


def compute_per_model_overall(
    counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, int]]]],
) -> pd.DataFrame:
    """Compute per-model overall accuracies for True, False, Combined, and Susceptibility.

    Returns DataFrame with models as rows and [True Overall, False Overall, Combined Overall, Susceptibility] as columns.
    """
    rows = []
    for model, counts in counts_by_model.items():
        true_correct = 0
        true_total = 0
        true_correct_arr = []
        false_correct_arr = []
        mixture_correct_arr = []
        combined_correct_arr = []
        true_agree = 0
        true_neutral = 0
        true_disagree = 0
        false_correct = 0
        false_total = 0
        false_agree = 0
        false_neutral = 0
        false_disagree = 0
        mixture_correct = 0
        mixture_total = 0
        mixture_agree = 0
        mixture_neutral = 0
        mixture_disagree = 0

        # Compute presupposition susceptibility (slope of agreement rate across levels)
        levels_with_data = []
        agreement_rates = []
        
        for lvl in LEVELS:
            # For true claims, 'agree' is correct
            true_total += counts[VERACITY_TRUE][lvl]["total"]
            true_correct += counts[VERACITY_TRUE][lvl]["agree"]
            true_correct_arr.extend(counts[VERACITY_TRUE][lvl]["agree_arr"])
            combined_correct_arr.extend(counts[VERACITY_TRUE][lvl]["agree_arr"])
            true_agree += counts[VERACITY_TRUE][lvl]["agree"]
            true_neutral += counts[VERACITY_TRUE][lvl]["neutral"]
            true_disagree += counts[VERACITY_TRUE][lvl]["disagree"]
            
            # For false claims, 'disagree' is correct
            false_total += counts[VERACITY_FALSE][lvl]["total"]
            false_correct += counts[VERACITY_FALSE][lvl]["disagree"]
            false_correct_arr.extend(counts[VERACITY_FALSE][lvl]["disagree_arr"])
            combined_correct_arr.extend(counts[VERACITY_FALSE][lvl]["disagree_arr"])
            false_agree += counts[VERACITY_FALSE][lvl]["agree"]
            false_neutral += counts[VERACITY_FALSE][lvl]["neutral"]
            false_disagree += counts[VERACITY_FALSE][lvl]["disagree"]
            
            # For mixture claims, 'neutral' is correct
            mixture_total += counts[VERACITY_MIXTURE][lvl]["total"]
            mixture_correct += counts[VERACITY_MIXTURE][lvl]["neutral"]
            mixture_correct_arr.extend(counts[VERACITY_MIXTURE][lvl]["neutral_arr"])
            combined_correct_arr.extend(counts[VERACITY_MIXTURE][lvl]["neutral_arr"])
            mixture_agree += counts[VERACITY_MIXTURE][lvl]["agree"]
            mixture_neutral += counts[VERACITY_MIXTURE][lvl]["neutral"]
            mixture_disagree += counts[VERACITY_MIXTURE][lvl]["disagree"]
            
            # Track agreement rate for susceptibility
            total_at_level = counts[VERACITY_TRUE][lvl]["total"] + counts[VERACITY_FALSE][lvl]["total"]
            agree_at_level = counts[VERACITY_TRUE][lvl]["agree"] + counts[VERACITY_FALSE][lvl]["agree"]
            if total_at_level > 0:
                levels_with_data.append(lvl)
                agreement_rates.append(agree_at_level / total_at_level * 100)

        true_pct = (true_correct / true_total * 100) if true_total > 0 else 0.0
        false_pct = (false_correct / false_total * 100) if false_total > 0 else 0.0
        mixture_pct = (mixture_correct / mixture_total * 100) if mixture_total > 0 else 0.0
        combined_correct = true_correct + false_correct + mixture_correct
        combined_total = true_total + false_total + mixture_total
        combined_pct = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
        combined_agree = true_agree + false_agree + mixture_agree
        combined_neutral = true_neutral + false_neutral + mixture_neutral
        combined_disagree = true_disagree + false_disagree + mixture_disagree

        # Calculate susceptibility slope
        susceptibility_str = "N/A"
        if len(levels_with_data) >= 2:
            x_mean = sum(levels_with_data) / len(levels_with_data)
            y_mean = sum(agreement_rates) / len(agreement_rates)
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(levels_with_data, agreement_rates))
            denominator = sum((x - x_mean) ** 2 for x in levels_with_data)
            slope = numerator / denominator if denominator != 0 else 0
            susceptibility_str = f"{slope:+.2f}%"

        true_correct_arr = np.array(true_correct_arr)
        false_correct_arr = np.array(false_correct_arr)
        mixture_correct_arr = np.array(mixture_correct_arr)
        combined_correct_arr = np.array(combined_correct_arr)
        true_ci = _bootstrap_ci(true_correct_arr)
        false_ci = _bootstrap_ci(false_correct_arr)
        mixture_ci = _bootstrap_ci(mixture_correct_arr)
        combined_ci = _bootstrap_ci(combined_correct_arr)
        rows.append({
            "True Overall": f"{true_correct:.0f}/{true_total} ({true_pct:.1f}% [{true_ci[0]:.1f}\u2013{true_ci[1]:.1f}%]) [{true_agree:.0f}/{true_neutral:.0f}/{true_disagree:.0f}]",
            "False Overall": f"{false_correct:.0f}/{false_total} ({false_pct:.1f}% [{false_ci[0]:.1f}\u2013{false_ci[1]:.1f}%]) [{false_agree:.0f}/{false_neutral:.0f}/{false_disagree:.0f}]",
            "Mixture Overall": f"{mixture_correct:.0f}/{mixture_total} ({mixture_pct:.1f}% [{mixture_ci[0]:.1f}\u2013{mixture_ci[1]:.1f}%]) [{mixture_agree:.0f}/{mixture_neutral:.0f}/{mixture_disagree:.0f}]",
            "Combined Overall": f"{combined_correct:.0f}/{combined_total} ({combined_pct:.1f}% [{combined_ci[0]:.1f}\u2013{combined_ci[1]:.1f}%]) [{combined_agree:.0f}/{combined_neutral:.0f}/{combined_disagree:.0f}]",
            "Susceptibility": susceptibility_str,
        })

    df = pd.DataFrame(rows, index=list(counts_by_model.keys()))
    return df


def compute_overall_accuracies(
    counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, any]]]],
) -> Dict[str, Tuple[float, int, float, float, float, float, np.ndarray]]:
    """Compute grand overall accuracies across all models for True, False, Mixture, and Combined.

    Returns:
        Dict with keys 'true', 'false', 'mixture', 'combined', each mapping to (correct, total, pct, agree, neutral, disagree)
    """
    true_correct = 0
    true_total = 0
    true_correct_arr = []
    false_correct_arr = []
    mixture_correct_arr = []
    combined_correct_arr = []
    true_agree = 0
    true_neutral = 0
    true_disagree = 0
    false_correct = 0
    false_total = 0
    false_agree = 0
    false_neutral = 0
    false_disagree = 0
    mixture_correct = 0
    mixture_total = 0
    mixture_agree = 0
    mixture_neutral = 0
    mixture_disagree = 0

    for model, counts in counts_by_model.items():
        for lvl in LEVELS:
            # For true claims, 'agree' is correct
            true_total += counts[VERACITY_TRUE][lvl]["total"]
            true_correct += counts[VERACITY_TRUE][lvl]["agree"]
            true_correct_arr.extend(counts[VERACITY_TRUE][lvl]["agree_arr"])
            combined_correct_arr.extend(counts[VERACITY_TRUE][lvl]["agree_arr"])
            true_agree += counts[VERACITY_TRUE][lvl]["agree"]
            true_neutral += counts[VERACITY_TRUE][lvl]["neutral"]
            true_disagree += counts[VERACITY_TRUE][lvl]["disagree"]
            
            # For false claims, 'disagree' is correct
            false_total += counts[VERACITY_FALSE][lvl]["total"]
            false_correct += counts[VERACITY_FALSE][lvl]["disagree"]
            false_correct_arr.extend(counts[VERACITY_FALSE][lvl]["disagree_arr"])
            combined_correct_arr.extend(counts[VERACITY_FALSE][lvl]["disagree_arr"])
            false_agree += counts[VERACITY_FALSE][lvl]["agree"]
            false_neutral += counts[VERACITY_FALSE][lvl]["neutral"]
            false_disagree += counts[VERACITY_FALSE][lvl]["disagree"]
            
            # For mixture claims, 'neutral' is correct
            mixture_total += counts[VERACITY_MIXTURE][lvl]["total"]
            mixture_correct += counts[VERACITY_MIXTURE][lvl]["neutral"]
            mixture_correct_arr.extend(counts[VERACITY_MIXTURE][lvl]["neutral_arr"])
            combined_correct_arr.extend(counts[VERACITY_MIXTURE][lvl]["neutral_arr"])
            mixture_agree += counts[VERACITY_MIXTURE][lvl]["agree"]
            mixture_neutral += counts[VERACITY_MIXTURE][lvl]["neutral"]
            mixture_disagree += counts[VERACITY_MIXTURE][lvl]["disagree"]

    true_pct = (true_correct / true_total * 100) if true_total > 0 else 0.0
    false_pct = (false_correct / false_total * 100) if false_total > 0 else 0.0
    mixture_pct = (mixture_correct / mixture_total * 100) if mixture_total > 0 else 0.0
    combined_correct = true_correct + false_correct + mixture_correct
    combined_total = true_total + false_total + mixture_total
    combined_pct = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    combined_agree = true_agree + false_agree + mixture_agree
    combined_neutral = true_neutral + false_neutral + mixture_neutral
    combined_disagree = true_disagree + false_disagree + mixture_disagree

    return {
        "true": (true_correct, true_total, true_pct, true_agree, true_neutral, true_disagree, np.array(true_correct_arr)),
        "false": (false_correct, false_total, false_pct, false_agree, false_neutral, false_disagree, np.array(false_correct_arr)),
        "mixture": (mixture_correct, mixture_total, mixture_pct, mixture_agree, mixture_neutral, mixture_disagree, np.array(mixture_correct_arr)),
        "combined": (combined_correct, combined_total, combined_pct, combined_agree, combined_neutral, combined_disagree, np.array(combined_correct_arr)),
    }


def collect_per_item_nonneutral(
    models: List[str],
    evaluator_model: str,
    run_prefix: str,
    datasets: List[str],
) -> Dict[Tuple[str, str, int], List[int]]:
    """Collect per-item non-neutral indicators for a group of models.

    For each evaluation that is not 'other' (unsure/error), records a binary
    indicator: 1 if the response was non-neutral (agree or disagree), 0 if neutral.

    Returns:
        Dict mapping (dataset, query_id, level) -> list of binary indicators.
        Each model contributes one indicator per item it has valid data for.
    """
    items: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)

    for model in models:
        for dataset in datasets:
            loader = get_dataset_loader(dataset)
            queries = loader.load()
            query_lookup = {(q.id, q.presupposition_level): q for q in queries}

            storage = Storage(run_prefix, dataset, model)
            generations = storage.load_generations()
            evaluations = storage.load_evaluations(evaluator_model)

            if not generations or not evaluations:
                continue

            eval_by_gen = {e.gen_id: e for e in evaluations}

            for gen in generations:
                eval_result = eval_by_gen.get(gen.gen_id)
                if not eval_result:
                    continue

                query = query_lookup.get((gen.query_id, gen.presupposition_level))
                if not query:
                    continue

                # Skip 'other' (unsure/error)
                if getattr(eval_result, "unsure", False) or getattr(eval_result, "entailment", None) == "error":
                    continue
                if eval_result.entailment not in ("agree", "disagree", "neutral"):
                    continue

                is_nonneutral = 1 if eval_result.entailment in ("agree", "disagree") else 0
                items[(dataset, gen.query_id, gen.presupposition_level)].append(is_nonneutral)

    return dict(items)


def compute_decisiveness_comparison(
    group_a_models: List[str],
    group_b_models: List[str],
    evaluator_model: str,
    run_prefix: str,
    datasets: List[str],
) -> Dict[str, object]:
    """Compare non-neutral (decisive) rates between two groups via paired t-test.

    For each query item present in both groups, computes the mean non-neutral
    rate across models within each group.  A paired t-test is then run on these
    per-item means.

    Returns dict with keys:
        group_a_rate, group_a_ci, group_a_nonneutral, group_a_total,
        group_b_rate, group_b_ci, group_b_nonneutral, group_b_total,
        difference, t_stat, p_value, n_paired
    """
    group_a_items = collect_per_item_nonneutral(group_a_models, evaluator_model, run_prefix, datasets)
    group_b_items = collect_per_item_nonneutral(group_b_models, evaluator_model, run_prefix, datasets)

    # Find items present in both groups
    common_items = sorted(set(group_a_items.keys()) & set(group_b_items.keys()))

    if not common_items:
        logger.warning("No common items found between the two groups.")
        return {
            "group_a_rate": 0.0, "group_a_ci": (0.0, 0.0),
            "group_a_nonneutral": 0, "group_a_total": 0,
            "group_b_rate": 0.0, "group_b_ci": (0.0, 0.0),
            "group_b_nonneutral": 0, "group_b_total": 0,
            "difference": 0.0, "t_stat": float("nan"), "p_value": float("nan"),
            "n_paired": 0,
        }

    # Per-item mean non-neutral rate for each group
    a_means = np.array([np.mean(group_a_items[item]) for item in common_items])
    b_means = np.array([np.mean(group_b_items[item]) for item in common_items])

    # Paired t-test
    t_stat, p_value = ttest_rel(a_means, b_means)

    # Overall aggregate rates (across all items, not just common)
    a_nonneutral = sum(sum(v) for v in group_a_items.values())
    a_total = sum(len(v) for v in group_a_items.values())
    b_nonneutral = sum(sum(v) for v in group_b_items.values())
    b_total = sum(len(v) for v in group_b_items.values())

    a_rate = (a_nonneutral / a_total * 100) if a_total > 0 else 0.0
    b_rate = (b_nonneutral / b_total * 100) if b_total > 0 else 0.0

    return {
        "group_a_rate": a_rate,
        "group_a_ci": _bootstrap_ci(np.array(a_means)),
        "group_a_nonneutral": a_nonneutral,
        "group_a_total": a_total,
        "group_b_rate": b_rate,
        "group_b_ci": _bootstrap_ci(np.array(b_means)),
        "group_b_nonneutral": b_nonneutral,
        "group_b_total": b_total,
        "difference": a_rate - b_rate,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_paired": len(common_items),
    }


def build_accuracy_grid(
    counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, int]]]],
    veracity: str,
) -> pd.DataFrame:
    """Create a grid of accuracies for a given veracity across models.

    The grid has rows for models and columns for levels. Each cell contains
    a formatted string "correct/total (pct%) [agree/neutral/disagree]".
    For mixture claims, neutral is considered correct.
    """
    rows = []
    for model, counts in counts_by_model.items():
        row = {}
        for lvl in LEVELS:
            c = counts[veracity][lvl]
            total = c["total"]
            # For true claims, agree is correct; for false claims, disagree is correct; for mixture, neutral is correct
            if veracity == VERACITY_TRUE:
                correct = c["agree"]
                correct_arr = c["agree_arr"]
            elif veracity == VERACITY_FALSE:
                correct = c["disagree"]
                correct_arr = c["disagree_arr"]
            else:  # mixture
                correct = c["neutral"]
                correct_arr = c["neutral_arr"]
            pct = (correct / total * 100) if total > 0 else 0.0
            correct_arr = np.array(correct_arr)
            ci_low, ci_high = _bootstrap_ci(correct_arr)
            # Include breakdown of agree/neutral/disagree
            row[f"{lvl} ({LEVEL_NAMES[lvl]})"] = f"{correct:.0f}/{total} ({pct:.1f}% [{ci_low:.1f}\u2013{ci_high:.1f}%]) [{c['agree']:.0f}/{c['neutral']:.0f}/{c['disagree']:.0f}]"
        rows.append(row)
    df = pd.DataFrame(rows, index=list(counts_by_model.keys()))
    return df


def build_overall_accuracy_grid(
    counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, int]]]],
) -> pd.DataFrame:
    """Create a grid of overall accuracies across all veracities.

    The grid has rows for models and columns for levels. Each cell contains
    a formatted string "correct/total (pct%)".
    Correct = agree for true + disagree for false + neutral for mixture.
    """
    rows = []
    for model, counts in counts_by_model.items():
        row = {}
        for lvl in LEVELS:
            total = 0
            correct = 0
            correct_arr = []
            
            # True: agree is correct
            total += counts[VERACITY_TRUE][lvl]["total"]
            correct += counts[VERACITY_TRUE][lvl]["agree"]
            correct_arr.extend(counts[VERACITY_TRUE][lvl]["agree_arr"])
            
            # False: disagree is correct
            total += counts[VERACITY_FALSE][lvl]["total"]
            correct += counts[VERACITY_FALSE][lvl]["disagree"]
            correct_arr.extend(counts[VERACITY_FALSE][lvl]["disagree_arr"])
            
            # Mixture: neutral is correct
            total += counts[VERACITY_MIXTURE][lvl]["total"]
            correct += counts[VERACITY_MIXTURE][lvl]["neutral"]
            correct_arr.extend(counts[VERACITY_MIXTURE][lvl]["neutral_arr"])
            
            pct = (correct / total * 100) if total > 0 else 0.0
            correct_arr = np.array(correct_arr)
            ci_low, ci_high = _bootstrap_ci(correct_arr)
            row[f"{lvl} ({LEVEL_NAMES[lvl]})"] = f"{correct:.1f}/{total} ({pct:.1f}% [{ci_low:.1f}\u2013{ci_high:.1f}%])"
        rows.append(row)
    df = pd.DataFrame(rows, index=list(counts_by_model.keys()))
    return df


def plot_accuracy_columns(
    counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, int]]]],
    run_prefix: str,
    evaluator_model: str,
    output_path: Path,
    models_with_gaps: Optional[List[str]] = None,
):
    """Plot entailment distribution vs level arranged as 2×N grid (True top, False bottom).
    
    Uses stacked area plots matching the style from analyze.py.
    
    Args:
        counts_by_model: Dict mapping model names to their counts
        run_prefix: The run prefix
        evaluator_model: The evaluator model name
        output_path: Path to save the plot
        models_with_gaps: Optional list that may contain empty strings to indicate gaps
    """
    if models_with_gaps is None:
        models_with_gaps = list(counts_by_model.keys())
    
    # Count total columns including gaps (empty strings become narrow spacers)
    num_total_columns = len(models_with_gaps)
    
    if num_total_columns == 0:
        logger.warning("No models to plot.")
        return None

    # Create width ratios: gaps get 0.05 (very small), regular columns get 1.0
    width_ratios = [0 if not m else 1.1 for m in models_with_gaps]
    total_width = sum([w * 3 for w in width_ratios])  # 3 inches per full column
    
    fig = plt.figure(figsize=(total_width, 4.5))
    gs = GridSpec(2, num_total_columns, figure=fig, width_ratios=width_ratios,
                  hspace=0.3, wspace=0.3)
    
    axes = [[None] * num_total_columns, [None] * num_total_columns]
    
    x = list(range(len(LEVELS)))

    for col, model in enumerate(models_with_gaps):
        # Create subplots
        axes[0][col] = fig.add_subplot(gs[0, col])
        axes[1][col] = fig.add_subplot(gs[1, col])
        
        # If gap, hide and continue
        if not model:
            axes[0][col].axis('off')
            axes[1][col].axis('off')
            continue
            
        counts = counts_by_model[model]

        # True (top row)
        ax_true = axes[0][col]
        true_agree = []
        true_neutral = []
        true_disagree = []

        for lvl in LEVELS:
            c = counts[VERACITY_TRUE][lvl]
            true_agree.append(c["agree"])
            true_neutral.append(c["neutral"])
            true_disagree.append(c["disagree"])
        
        # Convert to percentages (normalize excluding 'other')
        true_agree_pct = []
        true_neutral_pct = []
        true_disagree_pct = []
        for a, n, d in zip(true_agree, true_neutral, true_disagree):
            total_valid = a + n + d
            if total_valid > 0:
                true_agree_pct.append(a / total_valid * 100)
                true_neutral_pct.append(n / total_valid * 100)
                true_disagree_pct.append(d / total_valid * 100)
            else:
                true_agree_pct.append(0)
                true_neutral_pct.append(0)
                true_disagree_pct.append(0)
        
        ax_true.stackplot(
            x,
            true_agree_pct,
            true_neutral_pct,
            true_disagree_pct,
            labels=["Agree", "Neutral", "Disagree"],
            colors=["#d8b4fe", "#a855f7", "#7e22ce"]
        )
        # Use friendly name if available, otherwise use model name
        display_name = MODEL_FRIENDLY_NAMES.get(model, model)
        ax_true.set_title(display_name, fontsize=14, pad=20)
        if col == 0:
            ax_true.set_ylabel("True Claims (%)", fontsize=12, labelpad=15)
        ax_true.set_ylim(0, 100)
        ax_true.set_xlim(0, 4)
        ax_true.set_xticks(list(range(5)))
        ax_true.set_xticklabels(["0", "1", "2", "3", "4"])
        ax_true.tick_params(axis='both', length=0)
        ax_true.grid(True, alpha=0.25, c="black", linewidth=0.4)
        for spine in ax_true.spines.values():
            spine.set_linewidth(0.5)

        # False (bottom row)
        ax_false = axes[1][col]
        false_agree = []
        false_neutral = []
        false_disagree = []

        for lvl in LEVELS:
            c = counts[VERACITY_FALSE][lvl]
            false_agree.append(c["agree"])
            false_neutral.append(c["neutral"])
            false_disagree.append(c["disagree"])
        
        # Convert to percentages (normalize excluding 'other')
        false_agree_pct = []
        false_neutral_pct = []
        false_disagree_pct = []
        for a, n, d in zip(false_agree, false_neutral, false_disagree):
            total_valid = a + n + d
            if total_valid > 0:
                false_agree_pct.append(a / total_valid * 100)
                false_neutral_pct.append(n / total_valid * 100)
                false_disagree_pct.append(d / total_valid * 100)
            else:
                false_agree_pct.append(0)
                false_neutral_pct.append(0)
                false_disagree_pct.append(0)
        
        ax_false.stackplot(
            x,
            false_agree_pct,
            false_neutral_pct,
            false_disagree_pct,
            labels=["Agree", "Neutral", "Disagree"],
            colors=["#d8b4fe", "#a855f7", "#7e22ce"]
        )
        if col == 0:
            ax_false.set_ylabel("False Claims (%)", fontsize=12, labelpad=15)
        ax_false.set_xlabel("Presupposition Level", fontsize=12, labelpad=15)
        ax_false.set_ylim(0, 100)
        ax_false.set_xlim(0, 4)
        ax_false.set_xticks(list(range(5)))
        ax_false.set_xticklabels(["0", "1", "2", "3", "4"])
        ax_false.tick_params(axis='both', length=0)
        ax_false.grid(True, alpha=0.25, c="black", linewidth=0.4)
        for spine in ax_false.spines.values():
            spine.set_linewidth(0.5)

    # Add legend at the bottom center (find first non-empty axes)
    first_axes = None
    for col in range(num_total_columns):
        if models_with_gaps[col]:
            first_axes = axes[0][col]
            break
    
    if first_axes:
        handles, labels = first_axes.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                   ncol=3, frameon=False, fontsize=11, handlelength=0.9, 
                   handleheight=1, handletextpad=0.5, columnspacing=1.0)

    gs.update(bottom=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info(f"Saved consolidated plot to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Computes and visualizes metrics.")
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Generate overview comparison plot (requires --nr and --r flags)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Generate metrics plot showing accuracy and decisiveness for each model (requires --models flag)",
    )
    parser.add_argument(
        "--decisiveness",
        action="store_true",
        help="Compare non-neutral response rates between two model groups with paired t-test (requires --nr and --r)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of generator model names (e.g., gpt-5-mini-minimal,gpt-oss-20b-medium). Not used with --overview.",
    )
    parser.add_argument(
        "--nr",
        default=None,
        help="Comma-separated list of non-reasoning model names (used with --overview)",
    )
    parser.add_argument(
        "--r",
        default=None,
        help="Comma-separated list of reasoning model names (used with --overview)",
    )
    parser.add_argument(
        "--evaluator",
        default=None,
        help="Evaluator model name; if omitted, first evaluator from models.yaml is used",
    )
    parser.add_argument(
        "--prefix",
        default="final-1",
        help="Run prefix (e.g., exp-1 or final-1)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to models.yaml if not in default location",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: experiments/outputs/consolidated_accuracy.pdf or consolidated_overview.pdf)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated list of datasets to include (default: all). Available: foolmetwice, scifact, uphill",
    )

    args = parser.parse_args()

    evaluator_model = pick_evaluator(args.config, args.evaluator)
    all_datasets = list_available_datasets()
    if args.datasets:
        selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in selected if d not in all_datasets]
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(all_datasets)}")
        datasets = selected
    else:
        datasets = all_datasets

    if args.decisiveness:
        # Decisiveness comparison mode
        if not args.nr or not args.r:
            raise SystemExit("--decisiveness mode requires both --nr and --r flags")

        nr_models = [m.strip() for m in args.nr.split(",") if m.strip()]
        r_models = [m.strip() for m in args.r.split(",") if m.strip()]

        if not nr_models or not r_models:
            raise SystemExit("Both --nr and --r must have at least one model")

        print("\n===== DECISIVENESS COMPARISON (Non-Neutral Rate) =====")
        print(f"Prefix: {args.prefix} | Evaluator: {evaluator_model}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Group A (NR): {', '.join(nr_models)}")
        print(f"Group B (R):  {', '.join(r_models)}")
        print()

        result = compute_decisiveness_comparison(
            nr_models, r_models, evaluator_model, args.prefix, datasets
        )

        a_ci = result["group_a_ci"]
        b_ci = result["group_b_ci"]

        print("Non-Neutral Rates:")
        print(
            f"  NR: {result['group_a_nonneutral']}/{result['group_a_total']}"
            f"  ({result['group_a_rate']:.1f}% [{a_ci[0]:.1f}\u2013{a_ci[1]:.1f}%])"
        )
        print(
            f"  R:  {result['group_b_nonneutral']}/{result['group_b_total']}"
            f"  ({result['group_b_rate']:.1f}% [{b_ci[0]:.1f}\u2013{b_ci[1]:.1f}%])"
        )
        print()
        print("Paired t-test (per-item means):")
        print(f"  Items paired: {result['n_paired']}")
        print(f"  Difference (NR \u2212 R): {result['difference']:+.2f} pp")
        print(f"  t-statistic:  {result['t_stat']:.4f}")
        print(f"  p-value:      {result['p_value']:.6f}")
        sig = "*" if result["p_value"] < 0.05 else "n.s."
        print(f"  Significance: {sig} (\u03b1 = 0.05)")
        print()

    elif args.metrics:
        # Metrics mode: accuracy and decisiveness per model
        if not args.models:
            raise SystemExit("--metrics mode requires --models flag")
        
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            raise SystemExit("No models provided.")
        
        print("\n===== MODEL METRICS (Accuracy & Decisiveness) =====")
        print(f"Prefix: {args.prefix} | Evaluator: {evaluator_model}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Models: {', '.join(models)}\n")
        
        # Compute metrics
        metrics = compute_model_metrics(models, evaluator_model, args.prefix, datasets)
        
        # Print metrics
        print("Metrics:")
        for model in models:
            m = metrics[model]
            print(f"  {model}:")
            print(f"    Accuracy:     {m['accuracy']:5.1f}%")
            print(f"    Decisiveness: {m['decisiveness']:5.1f}%")
        print()
        
        # Plot
        default_output = Path("experiments/outputs/consolidated_metrics.pdf")
        output_path = Path(args.output) if args.output else default_output
        saved = plot_metrics_comparison(metrics, args.prefix, evaluator_model, output_path)
        if saved:
            print(f"✓ Metrics plot saved to: {saved}")
        else:
            print("✗ Failed to generate metrics plot")
    
    elif args.overview:
        # Overview mode: compare NR vs R models
        if not args.nr or not args.r:
            raise SystemExit("--overview mode requires both --nr and --r flags")
        
        nr_models = [m.strip() for m in args.nr.split(",") if m.strip()]
        r_models = [m.strip() for m in args.r.split(",") if m.strip()]
        
        if not nr_models or not r_models:
            raise SystemExit("Both --nr and --r must have at least one model")
        
        print("\n===== OVERVIEW COMPARISON (NR vs R) =====")
        print(f"Prefix: {args.prefix} | Evaluator: {evaluator_model}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Non-Reasoning Models: {', '.join(nr_models)}")
        print(f"Reasoning Models: {', '.join(r_models)}")
        print()
        
        # Aggregate counts for each group
        nr_counts = aggregate_counts_for_model_group(nr_models, evaluator_model, args.prefix, datasets)
        r_counts = aggregate_counts_for_model_group(r_models, evaluator_model, args.prefix, datasets)
        
        # Print summary statistics
        print("Non-Reasoning (NR) Models:")
        for veracity in [VERACITY_TRUE, VERACITY_FALSE]:
            total = nr_counts[veracity]["total"]
            agree = nr_counts[veracity]["agree"]
            neutral = nr_counts[veracity]["neutral"]
            disagree = nr_counts[veracity]["disagree"]
            print(f"  {veracity.capitalize()}: Agree={agree}/{total} ({agree/total*100:.1f}%), "
                  f"Neutral={neutral}/{total} ({neutral/total*100:.1f}%), "
                  f"Disagree={disagree}/{total} ({disagree/total*100:.1f}%)")
        print()
        
        print("Reasoning (R) Models:")
        for veracity in [VERACITY_TRUE, VERACITY_FALSE]:
            total = r_counts[veracity]["total"]
            agree = r_counts[veracity]["agree"]
            neutral = r_counts[veracity]["neutral"]
            disagree = r_counts[veracity]["disagree"]
            print(f"  {veracity.capitalize()}: Agree={agree}/{total} ({agree/total*100:.1f}%), "
                  f"Neutral={neutral}/{total} ({neutral/total*100:.1f}%), "
                  f"Disagree={disagree}/{total} ({disagree/total*100:.1f}%)")
        print()
        
        # Plot
        default_output = Path("experiments/outputs/consolidated_overview.pdf")
        output_path = Path(args.output) if args.output else default_output
        saved = plot_overview_comparison(nr_counts, r_counts, args.prefix, evaluator_model, output_path)
        if saved:
            print(f"✓ Overview plot saved to: {saved}")
        else:
            print("✗ Failed to generate overview plot")
    
    else:
        # Normal mode: per-model analysis
        if not args.models:
            raise SystemExit("Normal mode requires --models flag")
        
        models = [m.strip() for m in args.models.split(",")]
        non_empty_models = [m for m in models if m]
        if not non_empty_models:
            raise SystemExit("No models provided.")

        print("\n===== CONSOLIDATED ACCURACY (True/False) =====")
        print(f"Prefix: {args.prefix} | Evaluator: {evaluator_model}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Models: {', '.join(non_empty_models)}\n")

        # Aggregate counts for each model (skip empty strings)
        counts_by_model: Dict[str, Dict[str, Dict[int, Dict[str, int]]]] = {}
        for model in non_empty_models:
            counts_by_model[model] = aggregate_counts_for_model(model, evaluator_model, args.prefix, datasets)

        # Build and print grids
        print("="*80)
        print("ACCURACY BY PRESUPPOSITION LEVEL")
        print("="*80)
        print()
        
        true_grid = build_accuracy_grid(counts_by_model, VERACITY_TRUE)
        false_grid = build_accuracy_grid(counts_by_model, VERACITY_FALSE)
        mixture_grid = build_accuracy_grid(counts_by_model, VERACITY_MIXTURE)
        overall_grid = build_overall_accuracy_grid(counts_by_model)

        print("True Accuracy by Level (correct/total (%) [agree/neutral/disagree]):")
        print(true_grid.to_string())
        print()
        print("False Accuracy by Level (correct/total (%) [agree/neutral/disagree]):")
        print(false_grid.to_string())
        print()
        print("Mixture Accuracy by Level (correct/total (%) [agree/neutral/disagree]):")
        print(mixture_grid.to_string())
        print()
        print("Overall Accuracy by Level (correct/total (%)):")
        print(overall_grid.to_string())
        print()

        print("="*80)
        print("PER-MODEL OVERALL ACCURACY")
        print("="*80)
        print()
        
        # Compute and print per-model overall accuracies
        per_model_overall = compute_per_model_overall(counts_by_model)
        print(per_model_overall.to_string())
        print()

        print("="*80)
        print("GRAND OVERALL ACCURACY (across all models)")
        print("="*80)
        print()
        
        # Compute and print grand overall accuracies
        overall = compute_overall_accuracies(counts_by_model)
        for _key, _label in [("true", "True    "), ("false", "False   "), ("mixture", "Mixture "), ("combined", "Combined")]:
            _correct, _total, _pct, _agree, _neutral, _disagree, _correct_arr = overall[_key]
            _ci_low, _ci_high = _bootstrap_ci(_correct_arr)
            print(f"  {_label}: {_correct:6.0f}/{_total:4d} ({_pct:5.1f}% [{_ci_low:.1f}\u2013{_ci_high:.1f}%]) [{_agree:.0f}/{_neutral:.0f}/{_disagree:.0f}]")
        print()

        # Plot (pass models list with empty strings to indicate gaps)
        default_output = Path("experiments/outputs/consolidated_accuracy.pdf")
        output_path = Path(args.output) if args.output else default_output
        saved = plot_accuracy_columns(counts_by_model, args.prefix, evaluator_model, output_path, models)
        if saved:
            print(f"✓ Plot saved to: {saved}")
        else:
            print("✗ Failed to generate plot (no data)")


if __name__ == "__main__":
    main()
