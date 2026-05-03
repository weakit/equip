# Model Configuration

Models are defined in `models.yaml`. Each entry is a named model that can be passed to `--model` or `--evaluator-model` on the CLI.

## Structure

```yaml
models:
  my-model-name:
    backend: vllm         # execution engine
    type: generator       # generator or evaluator
    model_path: "..."     # HuggingFace ID or local path
    preprocessor: ...     # optional
    postprocessor: ...    # optional
    parameters: ...       # backend-specific
```

---

## Fields

### `backend`

| Value | Description |
|---|---|
| `vllm` | Local vLLM inference |
| `vllm_online` | Remote vLLM endpoint |
| `gemini` | Google Gemini API (streaming) |
| `gemini_batch` | Google Gemini batch API |
| `openai_batch` | OpenAI batch API |

### `type`

Either `generator` or `evaluator`. Generators are used with `generate`, evaluators with `evaluate`.

### `model_path`

HuggingFace hub ID (e.g. `"Qwen/Qwen3-8b"`) or an absolute local path to model weights.

---

## Preprocessors

Preprocessors format the input query before it reaches the model.

| Type | Use case |
|---|---|
| `standard` | Identity pass-through (default if omitted) |
| `qwen` | Adds Qwen thinking/no-thinking prompt formatting |
| `harmony` | Encodes claims using OpenAI Harmony special tokens |

**Qwen config options:**

```yaml
preprocessor:
  type: qwen
  config:
    enable_thinking: true   # or false to suppress reasoning
```

**Harmony config options:**

```yaml
preprocessor:
  type: harmony
  config:
    harmony_encoding_name: "HARMONY_GPT_OSS"
    reasoning_level: "medium"   # low, medium, high
```

---

## Postprocessors

Postprocessors extract the final response (and optional reasoning trace) from raw model output.

| Type | Use case |
|---|---|
| `standard` | Identity pass-through (default if omitted) |
| `qwen` | Splits `<think>...</think>` reasoning from the response |
| `harmony` | Extracts entailment label from Harmony-encoded output |
| `splitter` | Splits output on a delimiter |

Postprocessor type should match the preprocessor type used.

---

## Parameters

Parameters are passed directly to the backend. Common ones:

| Parameter | Backends | Notes |
|---|---|---|
| `batch_size` | all | Concurrent requests |
| `temperature` | vllm, sglang | Sampling temperature |
| `max_tokens` | vllm, sglang | Max output tokens |
| `tensor_parallel_size` | vllm, sglang | GPUs for tensor parallelism |
| `data_parallel_size` | vllm | Data parallelism |
| `pipeline_parallel_size` | vllm | Pipeline parallelism |
| `gpu_memory_utilization` | vllm | Fraction of GPU memory to use (0–1) |
| `system_prompt` | vllm | Override system prompt (set to `null` to disable) |
| `reasoning_effort` | openai_batch | `low`, `medium`, or `high` |
| `thinking_budget` | gemini, gemini_batch | Token budget for thinking (0 = disabled) |

---

## Examples

### Local model via vLLM

```yaml
my-llama:
  backend: vllm
  type: generator
  model_path: "meta-llama/Llama-3-8B-Instruct"
  parameters:
    tensor_parallel_size: 2
    batch_size: 512
    temperature: 0.8
    max_tokens: 4096
```

### Qwen with thinking

```yaml
qwen3-32b-thinking:
  backend: vllm
  type: generator
  model_path: "Qwen/Qwen3-32b"
  preprocessor:
    type: qwen
    config:
      enable_thinking: true
  postprocessor:
    type: qwen
  parameters:
    tensor_parallel_size: 4
    batch_size: 1024
```

### OpenAI batch

```yaml
gpt-5-mini-medium:
  backend: openai_batch
  type: generator
  model_path: "gpt-5-mini"
  parameters:
    reasoning_effort: "medium"
    batch_size: 512
```

### Gemini with thinking

```yaml
gemini-2.5-flash-thinking:
  backend: gemini
  type: generator
  model_path: "gemini-2.5-flash"
  parameters:
    thinking_budget: 2048
    batch_size: 100
```

### Evaluator (entailment judge)

```yaml
my-entailment-judge:
  backend: vllm
  type: evaluator
  model_path: "openai/gpt-oss-20b"
  preprocessor:
    type: harmony
    config:
      harmony_encoding_name: "HARMONY_GPT_OSS"
      reasoning_level: "medium"
  postprocessor:
    type: harmony
    config:
      harmony_encoding_name: "HARMONY_GPT_OSS"
  parameters:
    tensor_parallel_size: 1
    batch_size: 2048
    temperature: 1
    max_tokens: 16384
```
