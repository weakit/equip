# LLM Judge Annotation Tool

A human annotation interface for evaluating LLM judge performance by having humans annotate model responses and comparing them with the LLM judge's evaluations.

## Overview

This tool allows you to:
1. **Create sample sets**: Sample responses evenly across models and claim veracities
2. **Annotate responses**: Multiple annotators can independently judge whether model responses agree/disagree/neutral with claims
3. **Track progress**: Real-time progress tracking with the ability to review and change annotations
4. **Export data**: Export annotations for inter-annotator agreement analysis and comparison with LLM judge

## Quick Start

### Launch the annotation tool

```bash
python -m equip.annotator
```

Or with custom settings:

```bash
python -m equip.annotator --port 7861 --share
```

### Command-line Options

- `--results-dir PATH`: Path to results directory (default: auto-detect)
- `--annotations-dir PATH`: Path to annotations directory (default: project_root/annotations)
- `--port PORT`: Server port (default: 7861)
- `--share`: Create a public Gradio share link
- `--server-name HOST`: Server hostname (default: 127.0.0.1)
- `--debug`: Enable debug logging

## Workflow

### 1. Setup (Create or Select Sample Set)

#### Create New Sample Set
1. Select result prefix (e.g., "exp-1", "final-1")
2. Select dataset (e.g., "foolmetwice", "uphill")
3. Select one or more models to sample from
4. Optionally select evaluator to include LLM judge comparisons
5. Specify total samples (must be even - will be split X/2 true, X/2 false)
6. Give your sample set a name
7. Click "Create Sample Set"

**Sampling Logic:**
- Samples are split evenly between true and false claims
- Within each veracity, samples are distributed evenly across selected models
- All samples are shuffled before storage to prevent bias
- Samples include responses from all presupposition levels

#### Select Existing Sample Set
- View previously created sample sets
- See configuration and which annotators have worked on each set
- Multiple annotators can annotate the same set independently

### 2. Annotate

1. **Enter your name**: This identifies your annotations
2. **Select a sample set**: Choose which set to annotate
3. **Start annotating**: Click "Start Annotating"

#### Annotation Interface

**What you see:**
- Progress counter at the top (e.g., "15/20 (75.0%)")
- The **claim** with its veracity badge (TRUE/FALSE/NEUTRAL)
- The **model response** to annotate
- Your current annotation (if already annotated)

**How to annotate:**
- Click **✅ Agree** if the response supports/affirms the claim
- Click **❌ Disagree** if the response contradicts/refutes the claim
- Click **➖ Neutral** if the response is neutral or doesn't clearly agree/disagree

**Optional notes:**
- Add notes for any sample if you want to document your reasoning

**Navigation:**
- Use **← Previous** and **Next →** buttons to move between samples
- Use the **slider** to jump to any sample
- Your progress is saved automatically after each annotation
- You can freely navigate and change any annotation at any time

### 3. Export

Export annotation data for analysis:
1. Select a sample set
2. Click "Export to JSON"
3. Find the exported file in `annotations/export_{set_id}.json`

**Export format:**
```json
{
  "sample_set": { ... },
  "samples": [
    {
      "sample_id": "...",
      "claim": "...",
      "response": "...",
      "llm_entailment": "agree",
      "annotations": [
        {
          "annotator_name": "Alice",
          "entailment": "agree",
          "notes": "..."
        },
        {
          "annotator_name": "Bob",
          "entailment": "disagree",
          "notes": "..."
        }
      ]
    }
  ]
}
```

## File Structure

```
annotations/
├── sample_sets/
│   ├── {set_id_1}/
│   │   ├── metadata.json        # Sample set configuration
│   │   └── samples.jsonl        # Response samples
│   └── {set_id_2}/
│       └── ...
├── annotations.jsonl            # All annotations (append-only log)
└── export_{set_id}.json        # Exported data for analysis
```

## Data Models

### ResponseSample
- `sample_id`: Unique identifier
- `gen_id`: Reference to original generation
- `claim`: The claim being evaluated
- `claim_veracity`: Ground truth (true/false/neutral)
- `response`: Model's response text
- `model`: Model name
- `llm_entailment`: LLM judge's judgment (for comparison)

### HumanAnnotation
- `annotation_id`: Unique identifier
- `sample_id`: Reference to sample
- `annotator_name`: Annotator identifier
- `entailment`: Human judgment (agree/disagree/neutral)
- `notes`: Optional notes
- `timestamp`: When annotated
- `updated_at`: When last updated

### SampleSet
- `set_id`: Unique identifier
- `set_name`: Human-readable name
- `models`: List of models sampled from
- `total_samples`: Total number of samples
- `sample_ids`: Shuffled list of sample IDs

## Multi-Annotator Support

- Multiple annotators can independently annotate the same sample set
- Each annotator's progress is tracked separately
- Annotations are stored with annotator names
- Export includes all annotations for inter-annotator agreement analysis

## Tips for Annotators

1. **Read carefully**: Make sure you understand both the claim and the response
2. **Be consistent**: Try to apply the same criteria across all annotations
3. **When in doubt**: Choose "neutral" rather than guessing
4. **Use notes**: Document any uncertainty or reasoning for complex cases
5. **Take breaks**: Annotation requires focus - take breaks to maintain quality
6. **Review**: Use navigation to review and adjust previous annotations if needed

## Analysis Workflow

After collecting annotations:
1. Export the sample set
2. Calculate inter-annotator agreement (e.g., Cohen's Kappa, Fleiss' Kappa)
3. Compare human consensus with LLM judge evaluations
4. Identify cases where LLM judge disagrees with human consensus
5. Analyze patterns in disagreements

## Technical Details

### Sampling Algorithm
1. Load all generations from selected models
2. Separate by claim veracity (true/false)
3. Calculate samples per model per veracity
4. Randomly sample from each model's pool
5. Shuffle all samples together

### Storage
- Sample sets stored as JSON + JSONL for easy loading
- Annotations stored in append-only JSONL (keeps full history)
- Latest annotation version determined by `updated_at` timestamp
- Supports concurrent annotators (separate sessions)

### Progress Tracking
- Calculated on-the-fly from annotations file
- Counts unique sample_ids annotated by each annotator
- Updates in real-time as annotations are saved

## Troubleshooting

**No models appear in dropdown**
- Ensure the results directory contains generation files
- Check that `generations.jsonl` exists in model directories

**Sample creation fails**
- Ensure total_samples is even
- Check sufficient data exists for selected models
- Verify enough samples for each veracity category

**Annotations not saving**
- Check file permissions in annotations directory
- Look for errors in console/logs
- Ensure disk space available

**Progress not updating**
- Refresh the page
- Check that annotation was actually saved
- Verify sample_set_id matches

## Development

### Adding Custom Instructions

Edit the `DEFAULT_INSTRUCTIONS` in [ui.py](equip/annotator/ui.py) to customize annotation guidelines.

### Extending Data Export

Modify `export_annotations_for_analysis()` in [storage.py](equip/annotator/storage.py) to change export format.

### Custom Analysis

Load exported JSON and analyze with your preferred tools:
```python
import json

with open("annotations/export_{set_id}.json") as f:
    data = json.load(f)

# Your analysis code here
```

## Related Tools

- **`python -m equip.explorer`**: Browse and analyze evaluation results
- **Main evaluation framework**: Generate and evaluate model responses
