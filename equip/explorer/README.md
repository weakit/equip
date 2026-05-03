# UPHILL Response Explorer

A Gradio-based web interface for exploring and analyzing model generations and evaluations from the UPHILL evaluation framework.

## Features

- **Interactive browsing**: Navigate through queries using an intuitive slider interface
- **Multi-level view**: View all presupposition levels (0-4) for each query simultaneously
- **Evaluation details**: Inspect entailment judgments and reasoning in collapsible accordions
- **Multiple generations**: Handle and display multiple generations per query-presupposition pair
- **Flexible configuration**: Select different result prefixes, datasets, models, and evaluators

## Quick Start

### Launch the Explorer

```bash
# From the equip directory
python -m equip.explorer

# Or using the module directly
python -m equip.explorer

# With custom port
python -m equip.explorer --port 8080

# Create a public share link
python -m equip.explorer --share

# Use custom results directory
python -m equip.explorer --results-dir /path/to/results
```

### Usage

1. **Select Configuration**: Use the dropdowns at the top to choose:
   - Result prefix (e.g., `exp-1`, `final-1`)
   - Dataset (e.g., `foolmetwice`, `uphill`)
   - Model (e.g., `gpt-oss-20b-medium`)
   - Evaluator (optional, e.g., `gpt-oss-20b-entailment`)

2. **Load Data**: Click the "Load Data" button to load the selected configuration

3. **Navigate Queries**: Use the slider to browse through different queries

4. **View Results**: For each query, see:
   - All 5 presupposition levels (0-4)
   - Model responses for each level
   - Evaluation results (entailment + reasoning) if evaluator selected
   - Reasoning traces (expandable)
   - Multiple generations if available (numbered Generation 1/N, 2/N, etc.)

## Architecture

```
explorer/
├── __init__.py          # Module exports
├── __main__.py          # CLI entry point
├── data_loader.py       # Data loading and organization
├── ui.py                # Gradio UI components
└── README.md            # This file
```

### Key Components

#### `ExplorerDataLoader`
Handles loading and organizing data from the results directory:
- Scans available prefixes, datasets, and models
- Loads generations and evaluations from JSONL files
- Organizes data by query_id and presupposition_level
- Joins generations with their evaluations

#### `ExplorerUI`
Creates and manages the Gradio interface:
- Dynamic dropdowns that update based on available data
- Query navigation with slider
- Presupposition level displays with accordions
- Evaluation visualization with color-coded entailments

#### `GenerationWithEval`
Data class that pairs a generation with its optional evaluation:
- Provides convenient access to both generation and evaluation data
- Handles missing evaluations gracefully

## Data Format

The explorer expects the following directory structure:

```
results/
├── {prefix}/              # e.g., exp-1, final-1
│   └── {dataset}/         # e.g., foolmetwice, uphill
│       └── {model}/       # e.g., gpt-oss-20b-medium
│           ├── generations.jsonl
│           └── evaluation_{evaluator}.jsonl
```

### Generations Format (JSONL)
```json
{
  "gen_id": "unique_id",
  "query_id": "query_identifier",
  "dataset": "dataset_name",
  "presupposition_level": 0,
  "gen_model": "model_name",
  "response": "Generated response text...",
  "reasoning_trace": "Optional reasoning trace...",
  "timestamp": "2025-11-12T15:06:28.122277"
}
```

### Evaluations Format (JSONL)
```json
{
  "eval_id": "unique_id",
  "gen_id": "generation_id_reference",
  "eval_model": "evaluator_model_name",
  "entailment": "agree",
  "reasoning": "Evaluation reasoning...",
  "unsure": false,
  "timestamp": "2025-11-13T07:51:46.780810"
}
```

## Command-Line Options

```
usage: explorer.py [-h] [--results-dir RESULTS_DIR] [--port PORT] [--share]
                   [--server-name SERVER_NAME] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --results-dir RESULTS_DIR
                        Path to results directory (default: auto-detect)
  --port PORT           Port to run the server on (default: 7860)
  --share               Create a public share link
  --server-name SERVER_NAME
                        Server name/IP to bind to (default: 127.0.0.1)
  --debug               Enable debug logging
```

## Development

### Adding New Features

The modular design makes it easy to extend:

1. **Data Loader**: Add new data sources or formats in `data_loader.py`
2. **UI Components**: Add new visualizations or interactions in `ui.py`
3. **Entry Point**: Add new CLI options in `__main__.py`

### Best Practices

- Keep data loading separate from UI logic
- Use Pydantic models for type safety
- Handle missing data gracefully (not all queries may have all presupposition levels)
- Log important operations for debugging

## Troubleshooting

### No data showing up
- Check that your results directory exists and has the expected structure
- Verify generations.jsonl files exist in model directories
- Check the status text after clicking "Load Data"

### Evaluations not displaying
- Make sure you selected an evaluator from the dropdown
- Verify that evaluation files exist with the format `evaluation_{evaluator}.jsonl`
- Some generations may not have evaluations yet

### UI not updating
- Check browser console for errors
- Try refreshing the page
- Check server logs with `--debug` flag

## Examples

### Basic usage
```bash
python -m equip.explorer
```

### Production deployment
```bash
python -m equip.explorer --server-name 0.0.0.0 --port 8080
```

### Quick sharing
```bash
python -m equip.explorer --share
```

### Custom results location
```bash
python -m equip.explorer --results-dir ~/my-results/uphill-runs
```
