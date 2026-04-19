# mlx-spec
Speculative Decoding with MLX on Apple Silicon.

- Reproduce basic speculative decoding algoratihm in `inference.py::_speculative_decode`.
- Domain-Specific (Python coding)  draft model training.
- Benchmarks


Studied whether on-policy KD of a domain-specific draft model increases accepted token length.
Distilled `Qwen3-0.6B` into a Python-coding draft model on MLX-LM using target-resampled training data.
Achieving 1.33x speedup vs. 1.27x without resampling.



## Setup

1. Install [`uv`](https://docs.astral.sh/uv/)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create `venv`
```bash
uv venv --python 3.11
```

3. Install dependencies
```bash
uv sync
```

## Quick start

```bash
uv run python -m benchmark
```

## Benchmark Results

1. Target Model `Qwen3-4B` / Draft Model `Qwen3-0.6B` (on Apple M2 Base)

```
==============================
Benchmarking on Non-coding Workload
Speedup Report:
Model                               | Decoding TPS | Avg Accepted Len | Avg Speed Up
Baseline                            | 10.746       | -                | 1.00
Speculate with Non-distilled Draft  | 11.127       | 1.59             | 1.04
Speculate with Distilled Draft      | 11.046       | 1.52             | 1.03

==============================
Benchmarking on Coding Workload
Speedup Report:
Model                               | Decoding TPS | Avg Accepted Len | Avg Speed Up
Baseline                            | 9.383        | -                | 1.00
Speculate with Non-distilled Draft  | 10.924       | 1.91             | 1.16
Speculate with Distilled Draft      | 11.191       | 1.97             | 1.19
```

2. Target Model `Qwen3-1.7B` ,`Qwen3-4B`, `Qwen3-8B` / Draft Model `Qwen3-0.6B` (on Apple M2 Pro)

![Decoding speedup over baseline and average acceptance length (τ ) on Qwen3 draft models with thinking mode disabled and a maximum of 512 output tokens.png](./Table%202:%20Decoding%20speedup%20over%20baseline%20and%20average%20acceptance%20length%20(τ%20)%20on%20Qwen3%20draft%20models%20with%20thinking%20mode%20disabled%20and%20a%20maximum%20of%20512%20output%20tokens.png)
