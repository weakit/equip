# Evaluating Reasoning Models for Queries with Presuppositions 


## Data

Three datasets live under `data/`, each containing queries derived from factual claims at five presupposition levels (0–4). Level 0 is a neutral question with no embedded assumption; each subsequent level encodes the claim more assertively into the query phrasing.

| Dataset | Path | Format | `--dataset` value |
|---|---|---|---|
| UPHILL | `data/uphill/queries.csv` | CSV | `uphill` |
| FoolMeTwice | `data/foolme2/queries.jsonl` | JSONL | `foolmetwice` |
| SciFact | `data/scifact/queries.jsonl` | JSONL | `scifact` |

Each entry has a `claim`, a `label` (whether the claim is true/false depending on the dataset), a `presupposition_level`, and the corresponding `query` text to be sent to the model.

We also provide our raw results and outputs [here](https://drive.google.com/drive/folders/1dMNW053JNn5GzodNlWkPZ8ylMnzV_QhU?usp=sharing).

---

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

---

## Citation

If you find our codebase or dataset beneficial, please cite our work:

```bibtex
@inproceedings{sathyanathan26acl,
    title = "Evaluating Reasoning Models for Queries with Presuppositions",
    author = {Sathyanathan, Rose 
      and Vasisht, Kinshuk 
      and Pruthi, Danish},
    booktitle = "Annual Meeting of the Association for Computational Linguistics (ACL) Findings",
    year = "2026",
    month = jul,
    address = "San Diego, USA"
}
```

Please also make sure to credit the creators of UPHILL, FoolMeTwice and SciFact, which our dataset builds up on:

```bibtex
@inproceedings{kaur-etal-2024-evaluating,
    title = "Evaluating Large Language Models for Health-related Queries with Presuppositions",
    author = {Kaur, Navreet and Choudhury, Monojit  and Pruthi, Danish},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.850/",
    doi = "10.18653/v1/2024.findings-acl.850",
    pages = "14308--14331"
}
 
@inproceedings{eisenschlos-etal-2021-fool,
    title = "Fool Me Twice: Entailment from {W}ikipedia Gamification",
    author = {Eisenschlos, Julian Martin  and Dhingra, Bhuwan  and Bulian, Jannis  and B{\"o}rschinger, Benjamin  and Boyd-Graber, Jordan},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.32",
    pages = "352--365"
}
 
@inproceedings{wadden-etal-2020-fact,
    title = "Fact or Fiction: Verifying Scientific Claims",
    author = {Wadden, David  and Lin, Shanchuan  and Lo, Kyle  and Wang, Lucy Lu  and van Zuylen, Madeleine  and Cohan, Arman  and Hajishirzi, Hannaneh},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.609",
    doi = "10.18653/v1/2020.emnlp-main.609",
    pages = "7534--7550",
}
```

## Contact

If you have any questions, please email roshans [at] iisc [dot] ac [dot] in. Thanks!
