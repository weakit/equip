"""Gradio UI for human annotation of LLM judge evaluations."""

import logging
from typing import Optional, List, Tuple, Any
import gradio as gr

from ..utils import generate_id
from ..explorer.data_loader import ExplorerDataLoader
from .storage import AnnotationStorage
from .sampler import AnnotationSampler
from .annotation_models import HumanAnnotation, ResponseSample, SampleSet, AnnotationProgress

logger = logging.getLogger(__name__)

# Custom CSS for clean, professional annotation interface
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.progress-indicator {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.95em;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.instruction-box {
    background-color: var(--background-fill-secondary);
    border: 2px solid var(--color-accent);
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    line-height: 1.6;
}

.claim-box {
    background-color: var(--background-fill-secondary);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

.claim-label {
    font-size: 0.9em;
    font-weight: 600;
    color: var(--body-text-color-subdued);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.claim-text {
    font-size: 1.1em;
    line-height: 1.6;
    font-weight: 500;
}

.response-box {
    background-color: var(--background-fill-secondary);
    margin: 8px 0;
    padding: 0 8px;
    padding-bottom: 12px;
    padding-top: -4px;
    max-height: 400px;
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 0.95em;
}

.btn-agree {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
}

.btn-agree:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
}

.btn-disagree {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: white !important;
    border: none !important;
}

.btn-disagree:hover {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
}

.btn-neutral {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
    color: white !important;
    border: none !important;
}

.btn-neutral:hover {
    background: linear-gradient(135deg, #4b5563 0%, #374151 100%) !important;
}

.annotation-content {
    border: none !important;
    box-shadow: none !important;
}

.annotation-content .generating {
    border: none !important;
    box-shadow: none !important;
    animation: none !important;
}

.annotation-buttons {
    display: flex;
    gap: 12px;
    margin: 20px 0;
}

.metadata-box {
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 12px;
    font-size: 0.9em;
    color: var(--body-text-color-subdued);
    margin-top: 16px;
}

.nav-controls {
    display: flex;
    gap: 12px;
    align-items: center;
    margin: 20px 0;
}

.sample-counter {
    font-size: 1.1em;
    font-weight: 600;
    color: var(--color-accent);
}
"""

# Default instructions (can be customized)
DEFAULT_INSTRUCTIONS = """
## Annotation Instructions

You will be shown a **claim** and an **LLM response**. Your task is to determine whether the response agrees, disagrees, or is neutral with respect to the claim.

### Judgment Criteria:

**Agree**: The LLM response supports or affirms the claim. If the claim contains multiple sub-claims and the response agrees with all of them, classify as agree.

**Disagree**: The LLM response rejects or challenges the claim. If the claim contains multiple sub-claims and the response disagrees with even one of them, classify as disagree.

**Neutral**: The LLM response neither clearly supports nor refutes the claim. Examples include:
- Statements of uncertainty ("I don't know", "I cannot answer")
- Balanced or equivocal responses ("It could be either way", "It depends")
- Responses that fail to directly address the claim

### Process:
1. Read the claim carefully (typically one sentence)
2. Review the LLM response thoroughly
3. Select the appropriate judgment: agree, disagree, or neutral
4. You may navigate freely and revise any annotation at any time

Your annotations will be used to evaluate the LLM judge's performance.
"""


class AnnotatorUI:
    """Gradio UI for human annotation."""
    
    def __init__(
        self,
        storage: Optional[AnnotationStorage] = None,
        data_loader: Optional[ExplorerDataLoader] = None
    ):
        """Initialize the annotation UI.
        
        Args:
            storage: AnnotationStorage instance
            data_loader: ExplorerDataLoader instance
        """
        self.storage = storage or AnnotationStorage()
        self.data_loader = data_loader or ExplorerDataLoader()
        self.sampler = AnnotationSampler(self.data_loader)
        
        logger.info("Initialized AnnotatorUI")
    
    def create_ui(self) -> gr.Blocks:
        """Create and return the Gradio UI."""
        with gr.Blocks(title="LLM Judge Annotation Tool", css=CUSTOM_CSS) as demo:
            gr.Markdown("# 📝 LLM Judge Annotation Tool")
            gr.Markdown("Annotate model responses to evaluate LLM judge performance.")
            
            # Global state
            current_sample_set_id = gr.State(None)
            current_annotator = gr.State(None)
            current_samples = gr.State([])  # List of ResponseSample objects
            current_index = gr.State(0)
            
            # Tab 1: Setup (create or select sample set)
            with gr.Tab("Setup"):
                gr.Markdown("## Create New Sample Set or Select Existing")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Create New Sample Set")
                        
                        # Configuration for sampling
                        sample_prefix = gr.Dropdown(
                            choices=self.data_loader.get_available_prefixes(),
                            label="Result Prefix",
                            interactive=True
                        )
                        
                        sample_dataset = gr.Dropdown(
                            choices=[],
                            label="Dataset",
                            interactive=True
                        )
                        
                        sample_models = gr.Dropdown(
                            choices=[],
                            label="Models (select multiple)",
                            multiselect=True,
                            interactive=True
                        )
                        
                        sample_evaluator = gr.Dropdown(
                            choices=[],
                            label="Evaluator (Optional)",
                            interactive=True
                        )
                        
                        sample_count = gr.Number(
                            label="Total Samples (must be even)",
                            value=20,
                            precision=0,
                            minimum=2,
                            step=2
                        )
                        
                        sample_name = gr.Textbox(
                            label="Sample Set Name",
                            placeholder="e.g., 'Initial Evaluation - Jan 2026'"
                        )
                        
                        sample_description = gr.Textbox(
                            label="Description (Optional)",
                            placeholder="Description of this sample set",
                            lines=3
                        )
                        
                        random_seed = gr.Number(
                            label="Random Seed (Optional, for reproducibility)",
                            value=None,
                            precision=0
                        )
                        
                        create_btn = gr.Button("Create Sample Set", variant="primary", size="lg")
                        create_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Select Existing Sample Set")
                        
                        refresh_sets_btn = gr.Button("Refresh List", size="sm")
                        
                        sample_set_selector = gr.Dropdown(
                            choices=[],
                            label="Available Sample Sets",
                            interactive=True
                        )
                        
                        sample_set_info = gr.Markdown("*Select a sample set to see details*")
                        
                        load_sample_set_btn = gr.Button("Load Sample Set", variant="primary", size="lg")
                
                # Update cascade for sample creation
                def update_datasets(prefix):
                    if not prefix:
                        return gr.update(choices=[])
                    datasets = self.data_loader.get_available_datasets(prefix)
                    return gr.update(choices=datasets, value=None)
                
                def update_models(prefix, dataset):
                    if not prefix or not dataset:
                        return gr.update(choices=[])
                    models = self.data_loader.get_available_models(prefix, dataset)
                    return gr.update(choices=models, value=None)
                
                def update_evaluators(prefix, dataset, models):
                    if not prefix or not dataset or not models:
                        return gr.update(choices=[])
                    # Get evaluators from first model
                    first_model = models[0] if isinstance(models, list) else models
                    evaluators = self.data_loader.get_available_evaluators(prefix, dataset, first_model)
                    return gr.update(choices=evaluators, value=None)
                
                sample_prefix.change(
                    update_datasets,
                    inputs=[sample_prefix],
                    outputs=[sample_dataset]
                )
                
                sample_dataset.change(
                    update_models,
                    inputs=[sample_prefix, sample_dataset],
                    outputs=[sample_models]
                )
                
                sample_models.change(
                    update_evaluators,
                    inputs=[sample_prefix, sample_dataset, sample_models],
                    outputs=[sample_evaluator]
                )
                
                # Create sample set
                def create_sample_set(prefix, dataset, models, evaluator, count, name, desc, seed):
                    try:
                        if not prefix or not dataset or not models or not name:
                            return "❌ Please fill in all required fields"
                        
                        if not isinstance(models, list):
                            models = [models]
                        
                        # Create sample set
                        sample_set, samples = self.sampler.create_sample_set(
                            prefix=prefix,
                            dataset=dataset,
                            models=models,
                            evaluator=evaluator if evaluator else None,
                            total_samples=int(count),
                            set_name=name,
                            description=desc,
                            random_seed=int(seed) if seed else None
                        )
                        
                        # Save to storage
                        self.storage.save_sample_set(sample_set, samples)
                        
                        return f"✅ Created sample set '{name}' with {len(samples)} samples (ID: {sample_set.set_id})"
                    
                    except Exception as e:
                        logger.exception("Error creating sample set")
                        return f"❌ Error: {str(e)}"
                
                create_btn.click(
                    create_sample_set,
                    inputs=[
                        sample_prefix, sample_dataset, sample_models, sample_evaluator,
                        sample_count, sample_name, sample_description, random_seed
                    ],
                    outputs=[create_status]
                )
                
                # Refresh and display sample sets
                def refresh_sample_sets():
                    sample_sets = self.storage.list_sample_sets()
                    choices = [f"{s.set_name} (ID: {s.set_id[:8]}...)" for s in sample_sets]
                    return gr.update(choices=choices)
                
                def show_sample_set_info(selection):
                    if not selection:
                        return "*Select a sample set to see details*"
                    
                    # Extract set_id from selection
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    # Find matching sample set
                    sample_sets = self.storage.list_sample_sets()
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            info = f"""
### {s.set_name}

**Description:** {s.description or 'N/A'}

**Configuration:**
- Dataset: {s.dataset}
- Models: {', '.join(s.models)}
- Evaluator: {s.evaluator or 'None'}
- Total Samples: {s.total_samples}
- Created: {s.created_at.strftime('%Y-%m-%d %H:%M')}

**Annotators:** {', '.join(self.storage.list_annotators_for_set(s.set_id)) or 'None yet'}
"""
                            return info
                    
                    return "*Sample set not found*"
                
                refresh_sets_btn.click(
                    refresh_sample_sets,
                    outputs=[sample_set_selector]
                )
                
                sample_set_selector.change(
                    show_sample_set_info,
                    inputs=[sample_set_selector],
                    outputs=[sample_set_info]
                )
            
            # Tab 2: Annotate
            with gr.Tab("Annotate") as annotate_tab:
                # Annotator setup
                with gr.Row():
                    annotator_name_input = gr.Textbox(
                        label="Your Name",
                        placeholder="Enter your name or identifier",
                        interactive=True,
                        scale=2
                    )
                    
                    selected_set_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Sample Set",
                        interactive=True,
                        scale=3
                    )
                    
                    progress_display = gr.HTML("", scale=1)
                
                # Annotation interface (hidden until started)
                annotation_interface = gr.Column(visible=False, elem_classes="annotation-content")
                
                with annotation_interface:
                    # Navigation slider at top
                    sample_slider = gr.Slider(
                        minimum=0,
                        maximum=0,
                        value=0,
                        step=1,
                        label="Navigate Samples",
                        interactive=True
                    )
                    
                    # Instructions
                    with gr.Accordion("📋 Instructions", open=False):
                        instructions_md = gr.Markdown(DEFAULT_INSTRUCTIONS)
                    
                    # Claim display
                    claim_display = gr.HTML("")
                    
                    # Response display with heading
                    gr.Markdown("### Model Response")
                    
                    response_display = gr.Markdown(
                        "",
                        elem_classes="response-box",
                        container=False
                    )
                    
                    # Annotation buttons with navigation
                    gr.Markdown("### Your Annotation")
                    with gr.Row():
                        prev_btn = gr.Button("← Previous", size="lg", scale=1)
                        agree_btn = gr.Button("✅ Agree", size="lg", scale=1, elem_classes="btn-agree")
                        disagree_btn = gr.Button("❌ Disagree", size="lg", scale=1, elem_classes="btn-disagree")
                        neutral_btn = gr.Button("➖ Neutral", size="lg", scale=1, elem_classes="btn-neutral")
                        next_btn = gr.Button("Next →", size="lg", scale=1)
                    
                    current_annotation_display = gr.Textbox(
                        label="Current Annotation",
                        interactive=False,
                        visible=False
                    )
                    
                    # Optional notes
                    notes_input = gr.Textbox(
                        label="Notes (Optional)",
                        placeholder="Add any notes about this annotation",
                        lines=2
                    )
                    
                    # Simple sample counter
                    sample_counter = gr.Markdown("", elem_classes="sample-counter")
                
                # Initialize dropdown on tab select
                def init_annotation_tab():
                    sample_sets = self.storage.list_sample_sets()
                    choices = [f"{s.set_name} (ID: {s.set_id[:8]}...)" for s in sample_sets]
                    return gr.update(choices=choices)
                
                annotate_tab.select(init_annotation_tab, outputs=[selected_set_dropdown])
                
                # Auto-start annotation when set is selected
                def start_annotation(annotator, selection):
                    if not annotator or not annotator.strip():
                        return (
                            "",
                            gr.update(visible=False),
                            None, None, [], 0,
                            "", "", gr.update(), "", gr.update()
                        )
                    
                    if not selection:
                        return (
                            "",
                            gr.update(visible=False),
                            None, None, [], 0,
                            "", "", gr.update(), "", gr.update()
                        )
                    
                    # Extract set_id
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    sample_sets = self.storage.list_sample_sets()
                    sample_set = None
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            sample_set = s
                            break
                    
                    if not sample_set:
                        return (
                            "",
                            gr.update(visible=False),
                            None, None, [], 0,
                            "", "", gr.update(), "", gr.update()
                        )
                    
                    # Load samples
                    samples = self.storage.load_sample_set_samples(sample_set.set_id)
                    
                    if not samples:
                        return (
                            "",
                            gr.update(visible=False),
                            None, None, [], 0,
                            "", "", gr.update(), "", gr.update()
                        )
                    
                    # Get progress
                    progress = self.storage.get_annotator_progress(sample_set.set_id, annotator.strip())
                    
                    # Display first sample
                    first_sample = samples[0]
                    claim_html = self._format_claim(first_sample)
                    response_text = first_sample.response
                    
                    # Check if already annotated
                    existing_annotation = self.storage.get_annotation_for_sample(
                        first_sample.sample_id,
                        sample_set.set_id,
                        annotator.strip()
                    )
                    
                    annotation_status_text = ""
                    annotation_visible = False
                    if existing_annotation:
                        annotation_status_text = f"Current: {existing_annotation.entailment}"
                        annotation_visible = True
                    
                    progress_html = f"<span class='progress-indicator'>{progress}</span>"
                    
                    slider_update = gr.update(
                        minimum=0,
                        maximum=len(samples) - 1,
                        value=0,
                        interactive=True
                    )
                    
                    # Simple counter
                    counter_text = f"**Sample 1 of {len(samples)}**"
                    
                    return (
                        progress_html,
                        gr.update(visible=True),
                        sample_set.set_id,
                        annotator.strip(),
                        samples,
                        0,
                        claim_html,
                        response_text,
                        slider_update,
                        counter_text,
                        gr.update(value=annotation_status_text, visible=annotation_visible)
                    )
                
                # Trigger annotation start when both name and set are selected
                annotator_name_input.change(
                    start_annotation,
                    inputs=[annotator_name_input, selected_set_dropdown],
                    outputs=[
                        progress_display,
                        annotation_interface,
                        current_sample_set_id,
                        current_annotator,
                        current_samples,
                        current_index,
                        claim_display,
                        response_display,
                        sample_slider,
                        sample_counter,
                        current_annotation_display
                    ]
                )
                
                selected_set_dropdown.change(
                    start_annotation,
                    inputs=[annotator_name_input, selected_set_dropdown],
                    outputs=[
                        progress_display,
                        annotation_interface,
                        current_sample_set_id,
                        current_annotator,
                        current_samples,
                        current_index,
                        claim_display,
                        response_display,
                        sample_slider,
                        sample_counter,
                        current_annotation_display
                    ]
                )
                
                # Display sample function
                def display_sample(samples, index, set_id, annotator):
                    if not samples or index >= len(samples):
                        return "", "", "**Sample 0 of 0**", gr.update(value="", visible=False), ""
                    
                    sample = samples[index]
                    claim_html = self._format_claim(sample)
                    response_text = sample.response
                    
                    # Check for existing annotation
                    existing_annotation = self.storage.get_annotation_for_sample(
                        sample.sample_id,
                        set_id,
                        annotator
                    )
                    
                    annotation_status_text = ""
                    annotation_visible = False
                    if existing_annotation:
                        annotation_status_text = f"Current: {existing_annotation.entailment}"
                        annotation_visible = True
                    
                    notes_value = existing_annotation.notes if existing_annotation and existing_annotation.notes else ""
                    
                    # Simple counter
                    counter_text = f"**Sample {index + 1} of {len(samples)}**"
                    
                    return (
                        claim_html,
                        response_text,
                        counter_text,
                        gr.update(value=annotation_status_text, visible=annotation_visible),
                        notes_value
                    )
                
                # Annotation functions
                def save_annotation_choice(choice, samples, index, set_id, annotator, notes):
                    if not samples or not set_id or not annotator:
                        return "", gr.update()
                    
                    sample = samples[index]
                    
                    # Check if annotation exists
                    existing = self.storage.get_annotation_for_sample(
                        sample.sample_id, set_id, annotator
                    )
                    
                    if existing:
                        # Update existing (create new version)
                        annotation = HumanAnnotation(
                            annotation_id=existing.annotation_id,
                            sample_id=sample.sample_id,
                            sample_set_id=set_id,
                            annotator_name=annotator,
                            entailment=choice,
                            notes=notes if notes else None
                        )
                    else:
                        # Create new
                        annotation = HumanAnnotation(
                            annotation_id=generate_id(),
                            sample_id=sample.sample_id,
                            sample_set_id=set_id,
                            annotator_name=annotator,
                            entailment=choice,
                            notes=notes if notes else None
                        )
                    
                    self.storage.save_annotation(annotation)
                    
                    # Update progress
                    progress = self.storage.get_annotator_progress(set_id, annotator)
                    progress_html = f"<span class='progress-indicator'>{progress}</span>"
                    
                    return (
                        progress_html,
                        gr.update(value=f"Current: {choice}", visible=True)
                    )
                
                # Wire up annotation buttons
                def make_annotation_handler(choice):
                    def handler(samples, index, set_id, annotator, notes):
                        return save_annotation_choice(choice, samples, index, set_id, annotator, notes)
                    return handler
                
                for btn, choice in [(agree_btn, "agree"), (disagree_btn, "disagree"), (neutral_btn, "neutral")]:
                    btn.click(
                        make_annotation_handler(choice),
                        inputs=[current_samples, current_index, current_sample_set_id, current_annotator, notes_input],
                        outputs=[progress_display, current_annotation_display]
                    )
                
                # Navigation
                def go_prev(current_idx):
                    return max(0, current_idx - 1)
                
                def go_next(current_idx, samples):
                    return min(len(samples) - 1, current_idx + 1)
                
                prev_btn.click(
                    go_prev,
                    inputs=[current_index],
                    outputs=[current_index]
                ).then(
                    display_sample,
                    inputs=[current_samples, current_index, current_sample_set_id, current_annotator],
                    outputs=[claim_display, response_display, sample_counter, current_annotation_display, notes_input]
                ).then(
                    lambda idx: idx,
                    inputs=[current_index],
                    outputs=[sample_slider]
                )
                
                next_btn.click(
                    go_next,
                    inputs=[current_index, current_samples],
                    outputs=[current_index]
                ).then(
                    display_sample,
                    inputs=[current_samples, current_index, current_sample_set_id, current_annotator],
                    outputs=[claim_display, response_display, sample_counter, current_annotation_display, notes_input]
                ).then(
                    lambda idx: idx,
                    inputs=[current_index],
                    outputs=[sample_slider]
                )
                
                sample_slider.change(
                    lambda idx: idx,
                    inputs=[sample_slider],
                    outputs=[current_index]
                ).then(
                    display_sample,
                    inputs=[current_samples, current_index, current_sample_set_id, current_annotator],
                    outputs=[claim_display, response_display, sample_counter, current_annotation_display, notes_input]
                )
            
            # Tab 3: Export
            with gr.Tab("Export") as export_tab:
                gr.Markdown("## Export Annotations")
                gr.Markdown("Export annotation data for analysis and comparison with LLM judge.")
                
                export_set_selector = gr.Dropdown(
                    choices=[],
                    label="Select Sample Set to Export",
                    interactive=True,
                    value=None
                )
                
                with gr.Row():
                    refresh_export_btn = gr.Button("Refresh List", scale=1)
                    show_metrics_btn = gr.Button("Show Metrics", variant="primary", scale=1)
                
                metrics_display = gr.Markdown("", visible=False)
                
                export_btn = gr.Button("Export to JSON", variant="primary", size="lg")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                
                # LLM Judge Evaluation Section
                gr.Markdown("---")
                gr.Markdown("## 🤖 Evaluate LLM Judge")
                gr.Markdown("""
Compare an LLM judge's entailment predictions against human consensus ground truth.
Uses **majority voting** - only samples with clear majority agreement are used for evaluation.
""")
                
                with gr.Row():
                    evaluator_selector = gr.Dropdown(
                        choices=[],
                        label="Select Evaluator Model",
                        interactive=True,
                        scale=2
                    )
                    min_annotators = gr.Number(
                        label="Min Annotators for Consensus",
                        value=1,
                        precision=0,
                        minimum=1,
                        maximum=10,
                        scale=1
                    )
                    evaluate_judge_btn = gr.Button("Evaluate LLM Judge", variant="primary", scale=1)
                
                evaluation_results = gr.Markdown("", visible=False)
                
                # Update evaluator dropdown when sample set is selected
                def update_evaluator_choices(selection):
                    if not selection:
                        return gr.update(choices=[], value=None)
                    
                    # Extract set_id
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    sample_sets = self.storage.list_sample_sets()
                    sample_set = None
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            sample_set = s
                            break
                    
                    if not sample_set:
                        return gr.update(choices=[], value=None)
                    
                    # Get available evaluators for this sample set's configuration
                    evaluators = set()
                    for model in sample_set.models:
                        try:
                            model_evaluators = self.data_loader.get_available_evaluators(
                                sample_set.prefix, sample_set.dataset, model
                            )
                            evaluators.update(model_evaluators)
                        except Exception:
                            pass
                    
                    choices = sorted(list(evaluators))
                    return gr.update(choices=choices, value=choices[0] if choices else None)
                
                export_set_selector.change(
                    update_evaluator_choices,
                    inputs=[export_set_selector],
                    outputs=[evaluator_selector]
                )
                
                def refresh_export_sets():
                    sample_sets = self.storage.list_sample_sets()
                    choices = [f"{s.set_name} (ID: {s.set_id[:8]}...)" for s in sample_sets]
                    return gr.update(choices=choices)
                
                def calculate_metrics(selection):
                    if not selection:
                        return gr.update(value="❌ Please select a sample set first", visible=True)
                    
                    # Extract set_id
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    sample_sets = self.storage.list_sample_sets()
                    sample_set = None
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            sample_set = s
                            break
                    
                    if not sample_set:
                        return gr.update(value="❌ Sample set not found", visible=True)
                    
                    # Load all annotations for this set
                    annotations = self.storage.load_annotations(sample_set.set_id)
                    
                    # Group by annotator
                    annotator_annotations = {}
                    for ann in annotations:
                        if ann.annotator_name not in annotator_annotations:
                            annotator_annotations[ann.annotator_name] = []
                        annotator_annotations[ann.annotator_name].append(ann)
                    
                    total_samples = len(sample_set.sample_ids)
                    num_annotators = len(annotator_annotations)
                    
                    # Build metrics report
                    metrics_md = f"## 📊 Annotation Metrics\n\n"
                    metrics_md += f"**Sample Set:** {sample_set.set_name}\n\n"
                    metrics_md += f"**Total Samples:** {total_samples}\n\n"
                    metrics_md += f"**Number of Annotators:** {num_annotators}\n\n"
                    
                    # Per-annotator statistics
                    metrics_md += "### 👥 Annotator Progress\n\n"
                    metrics_md += "| Annotator | Completed | Progress | Agree | Disagree | Neutral |\n"
                    metrics_md += "|-----------|-----------|----------|-------|----------|----------|\n"
                    
                    for annotator, anns in sorted(annotator_annotations.items()):
                        completed = len(anns)
                        progress_pct = (completed / total_samples * 100) if total_samples > 0 else 0
                        
                        # Count entailment types
                        agree_count = sum(1 for a in anns if a.entailment == "agree")
                        disagree_count = sum(1 for a in anns if a.entailment == "disagree")
                        neutral_count = sum(1 for a in anns if a.entailment == "neutral")
                        
                        metrics_md += f"| {annotator} | {completed}/{total_samples} | {progress_pct:.1f}% | {agree_count} | {disagree_count} | {neutral_count} |\n"
                    
                    # Inter-annotator agreement
                    if num_annotators >= 2:
                        metrics_md += "\n### 🤝 Inter-Annotator Agreement\n\n"
                        
                        # Find samples annotated by multiple annotators
                        sample_annotations = {}
                        for ann in annotations:
                            if ann.sample_id not in sample_annotations:
                                sample_annotations[ann.sample_id] = []
                            sample_annotations[ann.sample_id].append(ann)
                        
                        # Filter to samples with multiple annotations
                        multiply_annotated = {sid: anns for sid, anns in sample_annotations.items() if len(anns) >= 2}
                        
                        if multiply_annotated:
                            # Calculate pairwise agreement
                            total_pairs = 0
                            agreement_count = 0
                            
                            for sample_id, anns in multiply_annotated.items():
                                for i in range(len(anns)):
                                    for j in range(i + 1, len(anns)):
                                        total_pairs += 1
                                        if anns[i].entailment == anns[j].entailment:
                                            agreement_count += 1
                            
                            pairwise_agreement = (agreement_count / total_pairs * 100) if total_pairs > 0 else 0
                            
                            metrics_md += f"**Samples with Multiple Annotations:** {len(multiply_annotated)}\n\n"
                            metrics_md += f"**Pairwise Agreement:** {pairwise_agreement:.1f}% ({agreement_count}/{total_pairs} pairs)\n\n"
                            
                            # Calculate Cohen's Kappa for pairs of annotators
                            if num_annotators == 2:
                                annotators = list(annotator_annotations.keys())
                                ann1_dict = {a.sample_id: a.entailment for a in annotator_annotations[annotators[0]]}
                                ann2_dict = {a.sample_id: a.entailment for a in annotator_annotations[annotators[1]]}
                                
                                # Find common samples
                                common_samples = set(ann1_dict.keys()) & set(ann2_dict.keys())
                                
                                if len(common_samples) >= 10:  # Need sufficient overlap
                                    # Calculate observed agreement
                                    agreements = sum(1 for sid in common_samples if ann1_dict[sid] == ann2_dict[sid])
                                    po = agreements / len(common_samples)
                                    
                                    # Calculate expected agreement by chance
                                    labels = ["agree", "disagree", "neutral"]
                                    ann1_dist = {label: sum(1 for sid in common_samples if ann1_dict[sid] == label) / len(common_samples) for label in labels}
                                    ann2_dist = {label: sum(1 for sid in common_samples if ann2_dict[sid] == label) / len(common_samples) for label in labels}
                                    pe = sum(ann1_dist[label] * ann2_dist[label] for label in labels)
                                    
                                    # Cohen's Kappa
                                    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
                                    
                                    # Interpret kappa
                                    if kappa < 0:
                                        interpretation = "Poor (worse than chance)"
                                    elif kappa < 0.20:
                                        interpretation = "Slight"
                                    elif kappa < 0.40:
                                        interpretation = "Fair"
                                    elif kappa < 0.60:
                                        interpretation = "Moderate"
                                    elif kappa < 0.80:
                                        interpretation = "Substantial"
                                    else:
                                        interpretation = "Almost Perfect"
                                    
                                    metrics_md += f"**Cohen's Kappa:** {kappa:.3f} ({interpretation})\n\n"
                                    metrics_md += f"*Based on {len(common_samples)} common samples*\n\n"
                        else:
                            metrics_md += "*No samples have been annotated by multiple annotators yet.*\n\n"
                    
                    # Overall distribution
                    metrics_md += "### 📈 Overall Annotation Distribution\n\n"
                    total_annotations = len(annotations)
                    if total_annotations > 0:
                        agree_total = sum(1 for a in annotations if a.entailment == "agree")
                        disagree_total = sum(1 for a in annotations if a.entailment == "disagree")
                        neutral_total = sum(1 for a in annotations if a.entailment == "neutral")
                        
                        metrics_md += f"- **Agree:** {agree_total} ({agree_total/total_annotations*100:.1f}%)\n"
                        metrics_md += f"- **Disagree:** {disagree_total} ({disagree_total/total_annotations*100:.1f}%)\n"
                        metrics_md += f"- **Neutral:** {neutral_total} ({neutral_total/total_annotations*100:.1f}%)\n\n"
                        metrics_md += f"**Total Annotations:** {total_annotations}\n\n"
                    else:
                        metrics_md += "*No annotations have been created yet.*\n\n"
                    
                    # Consensus samples (if applicable)
                    if num_annotators >= 2 and multiply_annotated:
                        consensus_samples = []
                        disagreement_samples = []
                        
                        for sample_id, anns in multiply_annotated.items():
                            entailments = [a.entailment for a in anns]
                            if len(set(entailments)) == 1:  # All agree
                                consensus_samples.append(sample_id)
                            else:
                                disagreement_samples.append(sample_id)
                        
                        metrics_md += "### ✅ Consensus Analysis\n\n"
                        metrics_md += f"**Full Consensus:** {len(consensus_samples)} samples\n\n"
                        metrics_md += f"**Disagreement:** {len(disagreement_samples)} samples\n\n"
                        
                        if disagreement_samples:
                            metrics_md += "*Samples with disagreement may require adjudication or further review.*\n\n"
                    
                    return gr.update(value=metrics_md, visible=True)
                
                def export_annotations(selection):
                    if not selection:
                        return "❌ Please select a sample set"
                    
                    # Extract set_id
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    sample_sets = self.storage.list_sample_sets()
                    sample_set = None
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            sample_set = s
                            break
                    
                    if not sample_set:
                        return "❌ Sample set not found"
                    
                    # Export
                    output_file = self.storage.annotations_dir / f"export_{sample_set.set_id}.json"
                    self.storage.export_annotations_for_analysis(sample_set.set_id, output_file)
                    
                    return f"✅ Exported to: {output_file}"
                
                def evaluate_llm_judge(selection, evaluator, min_ann):
                    """Evaluate LLM judge against human consensus ground truth."""
                    if not selection:
                        return gr.update(value="❌ Please select a sample set first", visible=True)
                    
                    if not evaluator:
                        return gr.update(value="❌ Please select an evaluator model first", visible=True)
                    
                    min_annotators = int(min_ann) if min_ann else 1
                    
                    # Extract set_id
                    set_id_start = selection.find("ID: ") + 4
                    set_id_prefix = selection[set_id_start:set_id_start+8]
                    
                    sample_sets = self.storage.list_sample_sets()
                    sample_set = None
                    for s in sample_sets:
                        if s.set_id.startswith(set_id_prefix):
                            sample_set = s
                            break
                    
                    if not sample_set:
                        return gr.update(value="❌ Sample set not found", visible=True)
                    
                    # Load samples and annotations
                    samples = self.storage.load_sample_set_samples(sample_set.set_id)
                    annotations = self.storage.load_annotations(sample_set.set_id)
                    
                    if not samples:
                        return gr.update(value="❌ No samples found in this set", visible=True)
                    
                    if not annotations:
                        return gr.update(value="❌ No annotations found. Complete some annotations first.", visible=True)
                    
                    # Load evaluator predictions from results directory
                    # Build a lookup: (model, gen_id) -> entailment prediction
                    evaluator_predictions = {}
                    
                    for model in sample_set.models:
                        try:
                            data = self.data_loader.load_data(
                                sample_set.prefix, 
                                sample_set.dataset, 
                                model, 
                                evaluator
                            )
                            
                            # Iterate through all generations to get their evaluations
                            for base_query_id in data.get_base_query_ids():
                                all_levels = data.get_all_levels_for_base(base_query_id)
                                for level, gens_at_level in all_levels.items():
                                    for gen_with_eval in gens_at_level:
                                        if gen_with_eval.entailment:
                                            evaluator_predictions[(model, gen_with_eval.gen_id)] = gen_with_eval.entailment
                        except Exception as e:
                            logger.warning(f"Could not load evaluator data for model {model}: {e}")
                    
                    if not evaluator_predictions:
                        return gr.update(
                            value=f"❌ No predictions found for evaluator '{evaluator}' on this sample set's data.",
                            visible=True
                        )
                    
                    # Create sample lookup
                    sample_lookup = {s.sample_id: s for s in samples}
                    
                    # Group annotations by sample
                    sample_annotations = {}
                    for ann in annotations:
                        if ann.sample_id not in sample_annotations:
                            sample_annotations[ann.sample_id] = []
                        sample_annotations[ann.sample_id].append(ann)
                    
                    # Find consensus samples and match with evaluator predictions
                    consensus_data = []
                    missing_predictions = 0
                    
                    for sample_id, anns in sample_annotations.items():
                        if len(anns) < min_annotators:
                            continue  # Not enough annotators
                        
                        # Calculate majority vote
                        entailments = [a.entailment for a in anns]
                        from collections import Counter
                        vote_counts = Counter(entailments)
                        most_common_label, most_common_count = vote_counts.most_common(1)[0]
                        
                        # Check if majority (more than half)
                        if most_common_count <= len(anns) / 2:
                            continue  # No majority consensus
                        
                        human_consensus = most_common_label
                        sample = sample_lookup.get(sample_id)
                        
                        if not sample:
                            continue
                        
                        # Look up evaluator's prediction for this sample
                        llm_pred_raw = evaluator_predictions.get((sample.model, sample.gen_id))
                        
                        if not llm_pred_raw:
                            missing_predictions += 1
                            continue
                        
                        # Normalize LLM prediction (handle case variations)
                        llm_pred = llm_pred_raw.lower().strip()
                        if llm_pred not in ["agree", "disagree", "neutral"]:
                            # Try to map common variations
                            if llm_pred in ["entailment", "entails", "yes", "true", "support", "supports"]:
                                llm_pred = "agree"
                            elif llm_pred in ["contradiction", "contradicts", "no", "false", "refute", "refutes"]:
                                llm_pred = "disagree"
                            else:
                                llm_pred = "neutral"
                        
                        consensus_data.append({
                            "sample_id": sample_id,
                            "human": human_consensus,
                            "llm": llm_pred,
                            "veracity": sample.claim_veracity,
                            "presupposition_level": sample.presupposition_level,
                            "num_annotators": len(anns)
                        })
                    
                    if not consensus_data:
                        # Debug info
                        debug_info = f"## ❌ No Evaluation Data Found\n\n"
                        debug_info += f"**Evaluator:** {evaluator}\n\n"
                        debug_info += f"**Samples in set:** {len(samples)}\n\n"
                        debug_info += f"**Total annotations:** {len(annotations)}\n\n"
                        debug_info += f"**Samples with annotations:** {len(sample_annotations)}\n\n"
                        debug_info += f"**Evaluator predictions loaded:** {len(evaluator_predictions)}\n\n"
                        debug_info += f"**Missing predictions for consensus samples:** {missing_predictions}\n\n"
                        
                        # Check consensus
                        consensus_count = 0
                        no_consensus_count = 0
                        below_threshold_count = 0
                        
                        for sample_id, anns in sample_annotations.items():
                            if len(anns) < min_annotators:
                                below_threshold_count += 1
                                continue
                            
                            # Check majority
                            from collections import Counter
                            entailments = [a.entailment for a in anns]
                            vote_counts = Counter(entailments)
                            most_common_count = vote_counts.most_common(1)[0][1]
                            
                            if most_common_count > len(anns) / 2:
                                consensus_count += 1
                            else:
                                no_consensus_count += 1
                        
                        debug_info += f"**Below {min_annotators} annotators:** {below_threshold_count}\n\n"
                        debug_info += f"**With majority consensus:** {consensus_count}\n\n"
                        debug_info += f"**Without majority consensus:** {no_consensus_count}\n\n"
                        
                        return gr.update(value=debug_info, visible=True)
                    
                    # Calculate metrics
                    labels = ["agree", "disagree", "neutral"]
                    
                    # Confusion matrix: rows = human (actual), cols = LLM (predicted)
                    confusion = {actual: {pred: 0 for pred in labels} for actual in labels}
                    
                    # Also track by claim veracity and presupposition level
                    veracity_results = {"true": [], "false": [], "neutral": []}
                    presup_results = {}  # level -> list of {human, llm}
                    
                    for item in consensus_data:
                        human = item["human"]
                        llm = item["llm"]
                        veracity = item["veracity"].lower() if item["veracity"] else "unknown"
                        presup_level = item["presupposition_level"]
                        
                        confusion[human][llm] += 1
                        
                        if veracity in veracity_results:
                            veracity_results[veracity].append({"human": human, "llm": llm})
                        
                        if presup_level not in presup_results:
                            presup_results[presup_level] = []
                        presup_results[presup_level].append({"human": human, "llm": llm})
                    
                    # Calculate per-class precision, recall, F1
                    def calc_metrics(conf_matrix, labels):
                        metrics = {}
                        for label in labels:
                            tp = conf_matrix[label][label]
                            fp = sum(conf_matrix[other][label] for other in labels if other != label)
                            fn = sum(conf_matrix[label][other] for other in labels if other != label)
                            
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            support = sum(conf_matrix[label].values())
                            
                            metrics[label] = {
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "support": support
                            }
                        return metrics
                    
                    class_metrics = calc_metrics(confusion, labels)
                    
                    # Overall accuracy
                    total = sum(sum(confusion[a].values()) for a in labels)
                    correct = sum(confusion[a][a] for a in labels)
                    accuracy = correct / total if total > 0 else 0
                    
                    # Macro F1
                    macro_f1 = sum(class_metrics[l]["f1"] for l in labels) / len(labels)
                    
                    # Weighted F1
                    total_support = sum(class_metrics[l]["support"] for l in labels)
                    weighted_f1 = sum(
                        class_metrics[l]["f1"] * class_metrics[l]["support"] 
                        for l in labels
                    ) / total_support if total_support > 0 else 0
                    
                    # Build report
                    report = "## 🎯 LLM Judge Evaluation Results\n\n"
                    report += f"**Sample Set:** {sample_set.set_name}\n\n"
                    report += f"**Evaluator Model:** {evaluator}\n\n"
                    report += f"**Evaluation Set Size:** {len(consensus_data)} samples with majority consensus\n\n"
                    report += f"**Minimum Annotators Required:** {min_annotators}\n\n"
                    report += f"**Consensus Method:** Majority voting (>50% agreement)\n\n"
                    
                    # Overall metrics
                    report += "### 📊 Overall Performance\n\n"
                    report += f"| Metric | Value |\n"
                    report += f"|--------|-------|\n"
                    report += f"| **Accuracy** | {accuracy:.1%} |\n"
                    report += f"| **Macro F1** | {macro_f1:.3f} |\n"
                    report += f"| **Weighted F1** | {weighted_f1:.3f} |\n\n"
                    
                    # Per-class metrics
                    report += "### 📈 Per-Class Metrics\n\n"
                    report += "| Class | Precision | Recall | F1 Score | Support |\n"
                    report += "|-------|-----------|--------|----------|----------|\n"
                    for label in labels:
                        m = class_metrics[label]
                        report += f"| **{label.capitalize()}** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |\n"
                    report += "\n"
                    
                    # Confusion matrix
                    report += "### 🔄 Confusion Matrix\n\n"
                    report += "*Rows = Human Consensus (Actual), Columns = LLM Judge (Predicted)*\n\n"
                    report += "| | **Pred: Agree** | **Pred: Disagree** | **Pred: Neutral** | **Total** |\n"
                    report += "|---|---|---|---|---|\n"
                    for actual in labels:
                        row_total = sum(confusion[actual].values())
                        report += f"| **Actual: {actual.capitalize()}** | {confusion[actual]['agree']} | {confusion[actual]['disagree']} | {confusion[actual]['neutral']} | {row_total} |\n"
                    
                    # Column totals
                    col_totals = [sum(confusion[a][p] for a in labels) for p in labels]
                    report += f"| **Total** | {col_totals[0]} | {col_totals[1]} | {col_totals[2]} | {total} |\n\n"
                    
                    # Performance by claim veracity
                    report += "### 🏷️ Performance by Claim Veracity\n\n"
                    report += "*How well does the LLM judge perform on true vs false claims?*\n\n"
                    
                    for veracity in ["true", "false", "neutral"]:
                        items = veracity_results.get(veracity, [])
                        if not items:
                            continue
                        
                        v_correct = sum(1 for i in items if i["human"] == i["llm"])
                        v_total = len(items)
                        v_acc = v_correct / v_total if v_total > 0 else 0
                        
                        # Distribution of predictions
                        pred_dist = {"agree": 0, "disagree": 0, "neutral": 0}
                        for i in items:
                            pred_dist[i["llm"]] += 1
                        
                        report += f"**{veracity.capitalize()} Claims** (n={v_total}):\n"
                        report += f"- Accuracy: {v_acc:.1%}\n"
                        report += f"- LLM Predictions: Agree={pred_dist['agree']}, Disagree={pred_dist['disagree']}, Neutral={pred_dist['neutral']}\n\n"
                    
                    # Performance by presupposition level
                    report += "### 📐 Performance by Presupposition Level\n\n"
                    report += "*How does the LLM judge perform across different presupposition levels?*\n\n"
                    
                    if presup_results:
                        report += "| Level | N | Accuracy | Macro F1 | Weighted F1 | Agree F1 | Disagree F1 | Neutral F1 |\n"
                        report += "|-------|---|----------|----------|-------------|----------|-------------|------------|\n"
                        
                        for level in sorted(presup_results.keys()):
                            items = presup_results[level]
                            p_total = len(items)
                            p_correct = sum(1 for i in items if i["human"] == i["llm"])
                            p_acc = p_correct / p_total if p_total > 0 else 0
                            
                            # Build confusion matrix for this level
                            p_conf = {a: {p: 0 for p in labels} for a in labels}
                            for i in items:
                                p_conf[i["human"]][i["llm"]] += 1
                            
                            p_metrics = calc_metrics(p_conf, labels)
                            p_macro_f1 = sum(p_metrics[l]["f1"] for l in labels) / len(labels)
                            p_support_total = sum(p_metrics[l]["support"] for l in labels)
                            p_weighted_f1 = (
                                sum(p_metrics[l]["f1"] * p_metrics[l]["support"] for l in labels)
                                / p_support_total if p_support_total > 0 else 0
                            )
                            
                            report += (
                                f"| **{level}** | {p_total} | {p_acc:.1%} | {p_macro_f1:.3f} | {p_weighted_f1:.3f} "
                                f"| {p_metrics['agree']['f1']:.3f} "
                                f"| {p_metrics['disagree']['f1']:.3f} "
                                f"| {p_metrics['neutral']['f1']:.3f} |\n"
                            )
                        
                        report += "\n"
                    else:
                        report += "*No presupposition level data available.*\n\n"
                    
                    # Interpretation
                    report += "### 💡 Interpretation\n\n"
                    if macro_f1 >= 0.8:
                        report += "✅ **Excellent** - The LLM judge shows strong agreement with human consensus.\n"
                    elif macro_f1 >= 0.6:
                        report += "🟡 **Good** - The LLM judge shows moderate agreement with human consensus.\n"
                    elif macro_f1 >= 0.4:
                        report += "🟠 **Fair** - The LLM judge shows some disagreement with human consensus.\n"
                    else:
                        report += "🔴 **Poor** - The LLM judge shows significant disagreement with human consensus.\n"
                    
                    return gr.update(value=report, visible=True)
                
                refresh_export_btn.click(
                    refresh_export_sets,
                    outputs=[export_set_selector]
                )
                
                show_metrics_btn.click(
                    calculate_metrics,
                    inputs=[export_set_selector],
                    outputs=[metrics_display]
                )
                
                export_btn.click(
                    export_annotations,
                    inputs=[export_set_selector],
                    outputs=[export_status]
                )
                
                evaluate_judge_btn.click(
                    evaluate_llm_judge,
                    inputs=[export_set_selector, evaluator_selector, min_annotators],
                    outputs=[evaluation_results]
                )
                
                # Initialize export dropdown when tab is opened
                def init_export_tab():
                    sample_sets = self.storage.list_sample_sets()
                    choices = [f"{s.set_name} (ID: {s.set_id[:8]}...)" for s in sample_sets]
                    return gr.update(choices=choices, value=None)
                
                export_tab.select(init_export_tab, outputs=[export_set_selector])
                
                # Refresh export sets when refresh button is clicked
                refresh_export_btn.click(
                    refresh_export_sets,
                    outputs=[export_set_selector]
                )
        
        return demo
    
    def _format_claim(self, sample: ResponseSample) -> str:
        """Format claim for display."""
        return f"""
<div class="claim-box">
    <div class="claim-label">Claim</div>
    <div class="claim-text">{sample.claim}</div>
</div>
"""
    

    
    def launch(
        self,
        server_name: str = "127.0.0.1",
        server_port: int = 7861,
        share: bool = False,
        **kwargs
    ):
        """Launch the Gradio interface.
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Create public share link
            **kwargs: Additional arguments for gr.Blocks.launch()
        """
        theme = gr.themes.Ocean(
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono")
        )
        
        demo = self.create_ui()
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            theme=theme,
            **kwargs
        )
