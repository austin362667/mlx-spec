# Distill a draft model that mimics the output distribution of a target model
# via full-weight fine-tuning with KL-divergence loss.

# Fuse the adoapters and submit to huggingface repo.
# ```
# uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B-MLX-bf16 --upload-repo austin362667/Qwen3-0.6B-MLX-bf16-python-5k-alpaca-resampled-Qwen-4B
# ```