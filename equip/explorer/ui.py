"""Gradio UI for the response explorer."""

import logging
from typing import Optional, List, Tuple, Any
import gradio as gr

from .data_loader import ExplorerDataLoader, ExplorerData, GenerationWithEval, QueryMetadata

logger = logging.getLogger(__name__)

# Custom CSS and JavaScript for Inter font and styled response box
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.infobox {
    background: linear-gradient(135deg, var(--background-fill-secondary) 0%, var(--background-fill-primary) 100%);
    border: 1px solid var(--border-color-primary);
    border-left: 4px solid var(--color-accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.infobox-claim {
    font-size: 1.1em;
    font-weight: 500;
    margin-bottom: 8px;
    line-height: 1.5;
}

.infobox-veracity {
    font-size: 0.95em;
    color: var(--body-text-color-subdued);
}

.veracity-true, .veracity-supports {
    color: #22c55e;
    font-weight: 600;
}

.veracity-false, .veracity-refutes {
    color: #ef4444;
    font-weight: 600;
}

.veracity-mixture {
    color: #f59e0b;
    font-weight: 600;
}

.response-box {
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    white-space: pre-wrap;
    line-height: 1.6;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.response-preview {
    position: relative;
}

.response-preview details {
    margin-top: 8px;
}

.response-preview summary {
    cursor: pointer;
    user-select: none;
    font-weight: 500;
    color: var(--color-accent);
    padding: 8px 16px;
    background-color: var(--button-secondary-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    list-style: none;
    text-align: center;
    transition: all 0.2s ease;
    display: inline-block;
    min-width: 120px;
}

.response-preview summary::-webkit-details-marker {
    display: none;
}

.response-preview summary::after {
    content: ' ▼';
    font-size: 0.8em;
    margin-left: 6px;
    transition: transform 0.2s ease;
    display: inline-block;
}

.response-preview details[open] summary::after {
    content: ' ▲';
}

.response-preview summary:hover {
    background-color: var(--button-secondary-background-fill-hover);
    border-color: var(--color-accent);
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.response-preview summary:active {
    transform: translateY(0);
}

.response-content-preview {
    max-height: 120px;
    overflow: hidden;
    position: relative;
    margin-bottom: 8px;
}

.response-content-preview::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(to bottom, transparent, var(--background-fill-secondary));
    pointer-events: none;
}

.response-content-full {
    padding-top: 8px;
    border-top: 1px solid var(--border-color-primary);
    margin-top: 8px;
}

.level-header {
    font-size: 1.15em;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--body-text-color);
}

.level-section {
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border-color-primary);
}

.level-section:last-child {
    border-bottom: none;
}

.eval-section {
    margin-top: 16px;
}

.eval-section details {
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 12px;
    margin-top: 8px;
    background-color: var(--background-fill-primary);
}

.eval-section details[open] {
    background-color: var(--background-fill-secondary);
}

.eval-section summary {
    cursor: pointer;
    user-select: none;
    font-weight: 500;
    padding: 4px 0;
    list-style: none;
}

.eval-section summary::-webkit-details-marker {
    display: none;
}

.eval-section summary:hover {
    opacity: 0.8;
}

.eval-content {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color-primary);
    line-height: 1.6;
}

.reasoning-trace {
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 12px;
    margin-top: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
    line-height: 1.5;
    max-height: 400px;
    overflow-y: auto;
    color: var(--body-text-color-subdued);
}

.gen-id-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    background-color: var(--background-fill-secondary);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border-color-primary);
}
"""


class ExplorerUI:
    """Gradio UI for exploring evaluation results."""
    
    def __init__(self, data_loader: Optional[ExplorerDataLoader] = None):
        """Initialize the UI.
        
        Args:
            data_loader: ExplorerDataLoader instance. If None, creates a new one.
        """
        self.data_loader = data_loader or ExplorerDataLoader()
        self.current_data: Optional[ExplorerData] = None
        
        # Get initial available options
        self.available_prefixes = self.data_loader.get_available_prefixes()
        if not self.available_prefixes:
            raise ValueError("No result prefixes found in results directory")
        
        logger.info(f"Initialized ExplorerUI with {len(self.available_prefixes)} prefixes")
    
    def create_ui(self) -> gr.Blocks:
        """Create and return the Gradio UI."""
        with gr.Blocks(title="UPHILL Response Explorer", css=CUSTOM_CSS) as demo:
            gr.Markdown("# 🔍 UPHILL Response Explorer")
            gr.Markdown("Explore model generations and evaluations across different presupposition levels.")
            
            # Top navigation bar with selectors
            with gr.Row():
                prefix_selector = gr.Dropdown(
                    choices=self.available_prefixes,
                    value=self.available_prefixes[0] if self.available_prefixes else None,
                    label="Result Prefix",
                    interactive=True
                )
                
                dataset_selector = gr.Dropdown(
                    choices=[],
                    label="Dataset",
                    interactive=True
                )
                
                model_selector = gr.Dropdown(
                    choices=[],
                    label="Model",
                    interactive=True
                )
                
                evaluator_selector = gr.Dropdown(
                    choices=[],
                    label="Evaluator (Optional)",
                    interactive=True,
                    allow_custom_value=False
                )
            
            # Load button and status
            with gr.Row():
                load_btn = gr.Button("Load Data", variant="primary", size="lg")
                status_text = gr.Textbox(
                    label="Status",
                    value="Select configuration and click Load Data",
                    interactive=False
                )
            
            # Filters
            with gr.Row():
                resistance_filter = gr.Dropdown(
                    choices=[],
                    label="Filter by Resistance (Optional)",
                    value=None,
                    interactive=False,
                    info="First level where query becomes factually inaccurate"
                )
                
                veracity_filter = gr.Dropdown(
                    choices=[],
                    label="Filter by Claim Veracity (Optional)",
                    value=None,
                    interactive=False,
                    info="Veracity of the claim (e.g., SUPPORTS, REFUTES)"
                )
            
            with gr.Row():
                filter_info = gr.Markdown("", visible=False)
            
            # Query navigation
            with gr.Row():
                query_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    label="Query Index",
                    interactive=False
                )
                
                query_info = gr.Textbox(
                    label="Base Query ID",
                    value="",
                    interactive=False
                )
            
            # Generation navigation
            gr.Markdown("## Generations")
            
            gen_slider = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Generation #",
                interactive=False
            )
            
            # Infobox for claim and veracity
            infobox_display = gr.Markdown("", visible=False)
            
            # Single generation display
            gen_display = gr.Markdown("*Select and load data to view generations*")
            
            # Hidden state to store loaded data info
            data_state = gr.State(value=None)
            gen_count_state = gr.State(value=0)  # Store number of generations per level
            filtered_query_ids_state = gr.State(value=[])  # Store filtered query IDs
            
            # Event handlers
            def update_datasets(prefix):
                """Update available datasets when prefix changes."""
                if not prefix:
                    return gr.update(choices=[], value=None)
                datasets = self.data_loader.get_available_datasets(prefix)
                return gr.update(
                    choices=datasets,
                    value=datasets[0] if datasets else None
                )
            
            def update_models(prefix, dataset):
                """Update available models when prefix or dataset changes."""
                if not prefix or not dataset:
                    return gr.update(choices=[], value=None)
                models = self.data_loader.get_available_models(prefix, dataset)
                return gr.update(
                    choices=models,
                    value=models[0] if models else None
                )
            
            def update_evaluators(prefix, dataset, model):
                """Update available evaluators when model changes."""
                if not prefix or not dataset or not model:
                    return gr.update(choices=[], value=None)
                evaluators = self.data_loader.get_available_evaluators(prefix, dataset, model)
                # Add None option for no evaluator
                return gr.update(
                    choices=["None"] + evaluators,
                    value=evaluators[0] if evaluators else "None"
                )
            
            def load_data(prefix, dataset, model, evaluator):
                """Load data based on selected configuration."""
                try:
                    # Handle "None" evaluator
                    eval_param = None if evaluator == "None" else evaluator
                    
                    self.current_data = self.data_loader.load_data(
                        prefix=prefix,
                        dataset=dataset,
                        model=model,
                        evaluator=eval_param
                    )
                    
                    base_query_ids = self.current_data.get_base_query_ids()
                    total_gens = self.current_data.get_total_generations()
                    
                    status_msg = (
                        f"✅ Loaded {len(base_query_ids)} base queries, {total_gens} generations\n"
                        f"Config: {prefix}/{dataset}/{model}"
                    )
                    if eval_param:
                        status_msg += f"\nEvaluator: {eval_param}"
                    
                    # Update slider
                    slider_update = gr.update(
                        minimum=0,
                        maximum=max(0, len(base_query_ids) - 1),
                        value=0,
                        interactive=True
                    )
                    
                    # Get first query info
                    first_query_id = base_query_ids[0] if base_query_ids else ""
                    
                    # Get generation count for first query at level 0
                    gen_count = 0
                    if first_query_id:
                        all_levels = self.current_data.get_all_levels_for_base(first_query_id)
                        gens_at_level = all_levels.get(0, [])
                        gen_count = len(gens_at_level)
                    
                    # Store data info in state
                    data_info = {
                        "prefix": prefix,
                        "dataset": dataset,
                        "model": model,
                        "evaluator": eval_param
                    }
                    
                    # Gen slider update
                    gen_slider_update = gr.update(
                        minimum=0,
                        maximum=max(0, gen_count - 1),
                        value=0,
                        interactive=gen_count > 1
                    )
                    
                    # Get available resistance levels
                    available_resistance = self.current_data.get_available_resistance_levels()
                    resistance_choices = ["All"] + [f"Resistance {r}" for r in available_resistance]
                    resistance_filter_update = gr.update(
                        choices=resistance_choices,
                        value="All",
                        interactive=len(available_resistance) > 0
                    )
                    
                    # Get available veracity values
                    available_veracity = self.current_data.get_available_veracity_values()
                    veracity_choices = ["All"] + available_veracity
                    veracity_filter_update = gr.update(
                        choices=veracity_choices,
                        value="All",
                        interactive=len(available_veracity) > 0
                    )
                    
                    return (
                        status_msg,
                        slider_update,
                        first_query_id,
                        data_info,
                        gen_slider_update,
                        gen_count,
                        resistance_filter_update,
                        veracity_filter_update,
                        base_query_ids,
                        gr.update(visible=False, value="")
                    )
                    
                except Exception as e:
                    logger.exception("Error loading data")
                    return (
                        f"❌ Error: {str(e)}",
                        gr.update(),
                        "",
                        None,
                        gr.update(),
                        0,
                        gr.update(),
                        gr.update(),
                        [],
                        gr.update(visible=False, value="")
                    )
            
            def apply_filters(resistance_value, veracity_value, data_info):
                """Apply resistance and veracity filters to query list."""
                if self.current_data is None or data_info is None:
                    return gr.update(), [], gr.update(visible=False, value=""), "", gr.update(), 0
                
                # Parse resistance value
                resistance_level = None
                if resistance_value and resistance_value != "All":
                    # Extract number from "Resistance X"
                    try:
                        resistance_level = int(resistance_value.split()[-1])
                    except:
                        resistance_level = None
                
                # Parse veracity value
                veracity = None
                if veracity_value and veracity_value != "All":
                    veracity = veracity_value
                
                # Get filtered query IDs using combined filter
                filtered_ids = self.current_data.filter_queries(
                    resistance_level=resistance_level,
                    veracity=veracity
                )
                
                if not filtered_ids:
                    # Build error message
                    filter_parts = []
                    if resistance_level is not None:
                        filter_parts.append(f"resistance {resistance_level}")
                    if veracity is not None:
                        filter_parts.append(f"veracity '{veracity}'")
                    filter_desc = " and ".join(filter_parts) if filter_parts else "these filters"
                    
                    return (
                        gr.update(minimum=0, maximum=0, value=0, interactive=False),
                        [],
                        gr.update(visible=True, value=f"⚠️ No queries found with {filter_desc}"),
                        "",
                        gr.update(),
                        0
                    )
                
                # Update query slider
                slider_update = gr.update(
                    minimum=0,
                    maximum=max(0, len(filtered_ids) - 1),
                    value=0,
                    interactive=True
                )
                
                # Get first query info
                first_query_id = filtered_ids[0]
                all_levels = self.current_data.get_all_levels_for_base(first_query_id)
                gens_at_level = all_levels.get(0, [])
                gen_count = len(gens_at_level)
                
                # Build info message
                filter_parts = []
                if resistance_level is not None:
                    filter_parts.append(f"resistance {resistance_level}")
                if veracity is not None:
                    filter_parts.append(f"veracity '{veracity}'")
                
                if filter_parts:
                    filter_desc = " and ".join(filter_parts)
                    info_msg = f"📊 Showing {len(filtered_ids)} queries with {filter_desc}"
                else:
                    info_msg = f"📊 Showing all {len(filtered_ids)} queries"
                
                return (
                    slider_update,
                    filtered_ids,
                    gr.update(visible=True, value=info_msg),
                    first_query_id,
                    gr.update(minimum=0, maximum=max(0, gen_count - 1), value=0, interactive=gen_count > 1),
                    gen_count
                )
            
            def display_query(query_index, data_info, filtered_query_ids):
                """Display generation for a query at current level."""
                if self.current_data is None or data_info is None:
                    return "*No data loaded*", gr.update(), 0
                
                # Use filtered query IDs if available
                base_query_ids = filtered_query_ids if filtered_query_ids else self.current_data.get_base_query_ids()
                if query_index >= len(base_query_ids):
                    return "*Invalid query index*", gr.update(), 0
                
                base_query_id = base_query_ids[int(query_index)]
                
                # Get gens at level 0 by default
                all_levels = self.current_data.get_all_levels_for_base(base_query_id)
                gens_at_level = all_levels.get(0, [])
                gen_count = len(gens_at_level)
                
                # Update gen slider
                gen_slider_update = gr.update(
                    minimum=0,
                    maximum=max(0, gen_count - 1),
                    value=0,
                    interactive=gen_count > 1
                )
                
                return base_query_id, gen_slider_update, gen_count
            
            def display_generation(query_index, gen_index, data_info, gen_count, filtered_query_ids):
                """Display all levels for a single generation."""
                if self.current_data is None or data_info is None:
                    return "*No data loaded*", gr.update(), gen_count, gr.update(visible=False, value="")
                
                # Use filtered query IDs if available
                base_query_ids = filtered_query_ids if filtered_query_ids else self.current_data.get_base_query_ids()
                if query_index >= len(base_query_ids):
                    return "*Invalid query index*", gr.update(), gen_count, gr.update(visible=False, value="")
                
                base_query_id = base_query_ids[int(query_index)]
                all_levels = self.current_data.get_all_levels_for_base(base_query_id)
                
                # Get query metadata
                metadata = self.current_data.get_query_metadata(base_query_id)
                
                # Get gen count from level 0 (assuming same count across levels)
                gens_at_level_0 = all_levels.get(0, [])
                new_gen_count = len(gens_at_level_0)
                
                # Adjust gen_index if it exceeds available gens
                actual_gen_index = min(int(gen_index), max(0, new_gen_count - 1))
                
                # Update gen slider if gen count changed
                gen_slider_update = gr.update(
                    minimum=0,
                    maximum=max(0, new_gen_count - 1),
                    value=actual_gen_index,
                    interactive=new_gen_count > 1
                )
                
                if new_gen_count == 0:
                    return "*No generations available*", gen_slider_update, new_gen_count, gr.update(visible=False, value="")
                
                # Build infobox
                infobox = self._build_infobox(metadata)
                infobox_update = gr.update(visible=bool(metadata), value=infobox)
                
                # Build display for all levels at this generation index
                display = self._build_all_levels_display(all_levels, actual_gen_index, new_gen_count, metadata)
                
                return display, gen_slider_update, new_gen_count, infobox_update
            
            # Wire up event handlers - chain them properly
            prefix_selector.change(
                fn=update_datasets,
                inputs=[prefix_selector],
                outputs=[dataset_selector]
            ).then(
                fn=update_models,
                inputs=[prefix_selector, dataset_selector],
                outputs=[model_selector]
            ).then(
                fn=update_evaluators,
                inputs=[prefix_selector, dataset_selector, model_selector],
                outputs=[evaluator_selector]
            )
            
            dataset_selector.change(
                fn=update_models,
                inputs=[prefix_selector, dataset_selector],
                outputs=[model_selector]
            ).then(
                fn=update_evaluators,
                inputs=[prefix_selector, dataset_selector, model_selector],
                outputs=[evaluator_selector]
            )
            
            model_selector.change(
                fn=update_evaluators,
                inputs=[prefix_selector, dataset_selector, model_selector],
                outputs=[evaluator_selector]
            )
            
            load_btn.click(
                fn=load_data,
                inputs=[prefix_selector, dataset_selector, model_selector, evaluator_selector],
                outputs=[status_text, query_slider, query_info, data_state, gen_slider, gen_count_state, resistance_filter, veracity_filter, filtered_query_ids_state, filter_info]
            ).then(
                fn=display_generation,
                inputs=[query_slider, gen_slider, data_state, gen_count_state, filtered_query_ids_state],
                outputs=[gen_display, gen_slider, gen_count_state, infobox_display]
            )
            
            # Filter changes - both resistance and veracity
            resistance_filter.change(
                fn=apply_filters,
                inputs=[resistance_filter, veracity_filter, data_state],
                outputs=[query_slider, filtered_query_ids_state, filter_info, query_info, gen_slider, gen_count_state]
            ).then(
                fn=display_generation,
                inputs=[query_slider, gen_slider, data_state, gen_count_state, filtered_query_ids_state],
                outputs=[gen_display, gen_slider, gen_count_state, infobox_display]
            )
            
            veracity_filter.change(
                fn=apply_filters,
                inputs=[resistance_filter, veracity_filter, data_state],
                outputs=[query_slider, filtered_query_ids_state, filter_info, query_info, gen_slider, gen_count_state]
            ).then(
                fn=display_generation,
                inputs=[query_slider, gen_slider, data_state, gen_count_state, filtered_query_ids_state],
                outputs=[gen_display, gen_slider, gen_count_state, infobox_display]
            )
            
            # Update display when query slider changes
            query_slider.change(
                fn=display_query,
                inputs=[query_slider, data_state, filtered_query_ids_state],
                outputs=[query_info, gen_slider, gen_count_state]
            ).then(
                fn=display_generation,
                inputs=[query_slider, gen_slider, data_state, gen_count_state, filtered_query_ids_state],
                outputs=[gen_display, gen_slider, gen_count_state, infobox_display]
            )
            
            # Update display when generation slider changes
            gen_slider.change(
                fn=display_generation,
                inputs=[query_slider, gen_slider, data_state, gen_count_state, filtered_query_ids_state],
                outputs=[gen_display, gen_slider, gen_count_state, infobox_display]
            )
            
            # Initialize first dataset/model/evaluator on load
            demo.load(
                fn=update_datasets,
                inputs=[prefix_selector],
                outputs=[dataset_selector]
            ).then(
                fn=update_models,
                inputs=[prefix_selector, dataset_selector],
                outputs=[model_selector]
            ).then(
                fn=update_evaluators,
                inputs=[prefix_selector, dataset_selector, model_selector],
                outputs=[evaluator_selector]
            )
        
        return demo
    
    def _build_all_levels_display(self, all_levels: dict, gen_index: int, total_gens: int, metadata: Optional[QueryMetadata]) -> str:
        """Build markdown display for all levels of a single generation."""
        markdown_parts = []
        
        # Header with generation info
        if total_gens > 1:
            markdown_parts.append(f"### Generation {gen_index + 1}/{total_gens}")
            markdown_parts.append("")
        
        # Display each level
        for level in range(5):
            gens_at_level = all_levels.get(level, [])
            
            if gen_index < len(gens_at_level):
                gen_with_eval = gens_at_level[gen_index]
                markdown_parts.append(self._build_level_display(gen_with_eval, level, metadata))
            else:
                markdown_parts.append(f'<div class="level-section"><span class="level-header">Level {level}</span> · <em>No generation</em></div>')
        
        return "\n".join(markdown_parts)
    
    def _build_level_display(self, gen_with_eval: GenerationWithEval, level: int, metadata: Optional[QueryMetadata]) -> str:
        """Build markdown display for a single level."""
        markdown_parts = []
        
        # Start level section
        markdown_parts.append('<div class="level-section">')
        
        # Header with level and gen ID on same line
        markdown_parts.append(f'<div style="margin-bottom: 12px;">')
        markdown_parts.append(f'<span class="level-header">Level {level}</span> <span class="gen-id-badge">{gen_with_eval.gen_id}</span>')
        markdown_parts.append(f'</div>')
        
        # Query text for this level (no truncation)
        if metadata and level in metadata.queries_by_level:
            query_text = metadata.queries_by_level[level]
            query_escaped = self._escape_html(query_text)
            markdown_parts.append(f'<div class="query-box">📋 <strong>Query:</strong> {query_escaped}</div>')
        
        # Response with read more/less functionality
        markdown_parts.append(self._format_response_with_read_more(gen_with_eval.response))
        
        # Evaluation section wrapper
        markdown_parts.append('<div class="eval-section">')
        
        # Reasoning trace if available
        if gen_with_eval.reasoning_trace:
            markdown_parts.append('<details>')
            markdown_parts.append('<summary>📝 <strong>Reasoning Trace</strong></summary>')
            markdown_parts.append(f'<div class="reasoning-trace">{self._escape_html(gen_with_eval.reasoning_trace)}</div>')
            markdown_parts.append('</details>')
        
        # Evaluation if available - closed by default
        if gen_with_eval.evaluation:
            entailment = gen_with_eval.entailment
            unsure_text = " ⚠️ (unsure)" if gen_with_eval.unsure else ""
            
            entailment_emoji = {
                "agree": "✅",
                "disagree": "❌",
                "neutral": "➖",
                "error": "⚠️"
            }.get(entailment, "❓")
            
            markdown_parts.append('<details>')
            markdown_parts.append(
                f'<summary>{entailment_emoji} <strong>Evaluation: {entailment.upper()}{unsure_text}</strong></summary>'
            )
            markdown_parts.append(f'<div class="eval-content">{self._escape_html(gen_with_eval.eval_reasoning)}</div>')
            markdown_parts.append('</details>')
        else:
            markdown_parts.append('<p style="color: var(--body-text-color-subdued); font-style: italic;">No evaluation available</p>')
        
        markdown_parts.append('</div>')  # Close eval-section
        markdown_parts.append('</div>')  # Close level-section
        
        return "\n".join(markdown_parts)
    
    def _build_infobox(self, metadata: Optional[QueryMetadata]) -> str:
        """Build the infobox showing claim and veracity."""
        if not metadata:
            return ""
        
        # Escape claim text
        claim_escaped = self._escape_html(metadata.claim)
        
        # Determine veracity CSS class
        veracity_lower = metadata.veracity.lower()
        veracity_class = "veracity-" + veracity_lower.replace(" ", "-")
        
        # Build infobox HTML
        parts = [
            '<div class="infobox">',
            f'<div class="infobox-claim"><strong>Claim:</strong> {claim_escaped}</div>',
            f'<div class="infobox-veracity"><strong>Veracity:</strong> <span class="{veracity_class}">{metadata.veracity.upper()}</span></div>',
            '</div>'
        ]
        
        return "\n".join(parts)
    
    def _format_response_with_read_more(self, response_text: str, preview_lines: int = 4) -> str:
        """Format response text with read more/less functionality."""
        lines = response_text.split('\n')
        
        if len(lines) <= preview_lines:
            # Short response, no need for read more
            return f'<div class="response-box">\n\n{response_text}\n\n</div>'
        
        # Create preview (first few lines)
        preview_text = '\n'.join(lines[:preview_lines])
        
        # Create response with read more using details/summary
        return f'''<div class="response-box">
<div class="response-preview">
<div class="response-content-preview">

{preview_text}

</div>
<details>
<summary>Read more</summary>
<div class="response-content-full">

{response_text}

</div>
</details>
</div>
</div>'''
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML entities in text."""
        import html
        return html.escape(text)
    
    def launch(self, **kwargs):
        """Launch the Gradio UI.
        
        Args:
            **kwargs: Arguments to pass to gr.Blocks.launch()
        """
        theme = gr.themes.Ocean(
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono")
        )

        demo = self.create_ui()
        demo.launch(theme=theme, **kwargs)
