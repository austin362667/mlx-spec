# mlx-spec
Speculative Decoding in MLX


## Quick start

```bash
uv run python -m benchmark
```


## Benchmarks

1. Target Model `Qwen3-4B` / Draft Model `Qwen3-0.6B` (on Apple M2)

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
