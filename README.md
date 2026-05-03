# Evaluating Reasoning Models for Queries with Presuppositions 

## Setup

Requires Python 3.10+. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

To install optional dev dependencies:

```bash
uv sync --extra dev
```

API keys go in a `.env` file in the project root. Which keys you need depends on the backends you're using (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

---

## Pipeline

The core workflow is three steps: **generate → evaluate → analyze**.

### 1. Generate

Runs a model against a dataset and saves responses to `results/`.

```bash
uv run python -m equip.main generate \
  --model gpt-oss-20b-medium \
  --dataset uphill \
  --n-samples 3 \
  --run-prefix exp-1
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Model name from `models.yaml` |
| `--dataset` | required | `uphill`, `foolmetwice`, or `scifact` |
| `--n-samples` | `1` | Generations per query |
| `--run-prefix` | `exp-1` | Groups results under `results/{prefix}/` |

`--run-prefix` is the label for a given experimental run. All three pipeline steps must use the same prefix — it determines where results are read from and written to.

### 2. Evaluate

Runs an entailment judge over the saved generations.

```bash
uv run python -m equip.main evaluate \
  --model gpt-oss-20b-medium \
  --evaluator-model gpt-oss-20b-entailment \
  --dataset uphill \
  --run-prefix exp-1
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Generator model (must match generate step) |
| `--evaluator-model` | first evaluator in config | Evaluator model name from `models.yaml` |
| `--dataset` | required | Must match generate step |
| `--run-prefix` | `exp-1` | Must match generate step |

### 3. Analyze

Computes final metrics.

```bash
uv run python -m equip.main analyze \
  --model gpt-oss-20b-medium \
  --evaluator-model gpt-oss-20b-entailment \
  --dataset uphill \
  --run-prefix exp-1
```

---

## Models

Available models are defined in `models.yaml`. See [models.md](models.md) for how to configure or add models.

To list all currently configured models:

```bash
uv run python -m equip.main list-models
```

To use a different config file:

```bash
uv run python -m equip.main --config path/to/models.yaml generate ...
```

---

## Other Commands

### Explorer

A Gradio UI for browsing results interactively. Reads from `results/` by default.

```bash
uv run python -m equip.explorer
```

```bash
uv run python -m equip.explorer \
  --results-dir ./results \
  --port 7860 \
  --share
```

`--share` creates a public Gradio link. `--server-name 0.0.0.0` to bind to all interfaces.

### Annotator

A Gradio UI for human annotation of model responses. Useful for validating the LLM judge.

```bash
uv run python -m equip.annotator
```

```bash
uv run python -m equip.annotator \
  --results-dir ./results \
  --annotations-dir ./annotations \
  --port 7861 \
  --share
```

Annotations are saved to `annotations/` and can be used to compute inter-annotator agreement against the LLM judge.

---

## Logging

All commands accept `--log-level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`) and `--log-file` for writing logs to disk.

```bash
uv run python -m equip.main generate \
  --model gpt-oss-20b-medium \
  --dataset uphill \
  --log-level DEBUG \
  --log-file run.log
```
