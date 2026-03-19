from __future__ import annotations
import argparse
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from datasets import load_dataset

import mlx.core as mx
import mlx.optimizers as optim

from mlx_lm import load
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.callbacks import get_reporting_callbacks
# Distill a draft model that mimics the output distribution of a target model
# via full-weight fine-tuning with KL-divergence loss.

# Fuse the adoapters and submit to huggingface repo.
# ```
# uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B-MLX-bf16 --upload-repo austin362667/Qwen3-0.6B-MLX-bf16-python-5k-alpaca-resampled-Qwen-4B
# ```

# Based on mlx-lm lora trainer code: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-4B-MLX-bf16" #
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-0.6B-MLX-bf16" # or "Qwen/Qwen3-0.6B-MLX-bf16"
DEFAULT_DATASET = "Austin362667/Qwen3-0.6B-MLX-bf16-python-5k-alpaca-resampled-Qwen-4B"
# 5k resampled from iamtarun/python_code_instructions_18k_alpaca using Qwen3-4B
# Dataset regeneration: https://docs.sglang.io/SpecForge/basic_usage/data_preparation.html#regenerate-datasets

def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    x_max = mx.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))

def _safe_strip(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _row_to_messages(row: dict) -> Optional[Tuple[List[dict], List[dict]]]:
    instruction = _safe_strip(row.get("instruction"))
    inp = _safe_strip(row.get("input"))
    output = _safe_strip(
        row.get("output")
        or row.get("response")
        or row.get("completion")
        or row.get("answer")
    )
    prompt = _safe_strip(row.get("prompt"))

    if prompt:
        user_text = prompt
    else:
        if not instruction:
            return None
        user_text = instruction
        if inp:
            user_text += f"\n\nInput:\n{inp}"

    if not output:
        return None

    full_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]
    prompt_messages = [
        {"role": "user", "content": user_text},
    ]
    return full_messages, prompt_messages


def _tokenize_example(tokenizer, row: dict) -> Optional[Tuple[np.ndarray, int]]:
    converted = _row_to_messages(row)
    if converted is None:
        return None

    full_messages, prompt_messages = converted

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(prompt_text)

    if len(full_ids) <= len(prompt_ids):
        return None

    return np.array(full_ids, dtype=np.int32), len(prompt_ids)


def build_tokenized_splits(
    dataset_name: str,
    tokenizer,
    seed: int,
    eval_ratio: float,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
):
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)

    split = raw.train_test_split(test_size=eval_ratio, seed=seed)
    train_raw = split["train"]
    eval_raw = split["test"]

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    def convert(ds):
        out = []
        skipped = 0
        for row in ds:
            item = _tokenize_example(tokenizer, row)
            if item is None:
                skipped += 1
                continue
            out.append(item)
        return out, skipped

    train_data, train_skipped = convert(train_raw)
    eval_data, eval_skipped = convert(eval_raw)

    if len(train_data) == 0:
        raise RuntimeError("No valid training samples after preprocessing.")
    if len(eval_data) == 0:
        raise RuntimeError("No valid eval samples after preprocessing.")

    print(
        f"Prepared dataset: train={len(train_data)} (skipped={train_skipped}), "
        f"eval={len(eval_data)} (skipped={eval_skipped})"
    )
    return train_data, eval_data


@dataclass
class DistillContext:
    teacher_model: object
    temperature: float

# ref kl_div_loss impl in mlx-lm: https://github.com/ml-explore/mlx-lm/blob/f8019f77694fa484f8c6cd95b37b00ac5d9ec634/mlx_lm/tuner/losses.py#L377
def make_kl_loss(ctx: DistillContext):
    teacher_model = ctx.teacher_model
    T = float(ctx.temperature)

    def kl_loss(student_model, batch, lengths):
        """
        Signature matches mlx_lm.tuner.trainer.train(loss=...)
        batch   : [B, S]
        lengths : [B, 2] where lengths[:, 0] is prompt offset,
                  and lengths[:, 1] is actual sequence length
        """
        inputs = batch[:, :-1]
        # shape: [B, S-1, V]
        student_logits = student_model(inputs).astype(mx.float32)
        teacher_logits = teacher_model(inputs).astype(mx.float32)

        student_log_probs = _log_softmax(student_logits / T, axis=-1)
        teacher_log_probs = _log_softmax(teacher_logits / T, axis=-1)
        teacher_probs = mx.exp(teacher_log_probs)

        # KL per token: sum_v p_t(v) * (log p_t(v) - log p_s(v))
        token_kl = mx.sum(
            teacher_probs * (teacher_log_probs - student_log_probs),
            axis=-1,
        )  # [B, S-1]

        steps = mx.arange(1, token_kl.shape[1] + 1)  # 1..S-1
        mask = mx.logical_and(
            steps >= lengths[:, 0:1],
            steps <= lengths[:, 1:],
        )
        mask = mask.astype(mx.float32)

        ntoks = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
        loss = mx.sum(token_kl * mask) / ntoks

        loss = loss * (T ** 2) # distillation scaling
        return loss.astype(mx.float32), ntoks

    return kl_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

    parser.add_argument("--iters", type=int, default=5000) # number of training iterations (batches), not epochs
    parser.add_argument("--batch-size", type=int, default=2) # because we are contrained in single GPU memroy A100-80G * 1 (not distributed trainig cluster)  
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    # other ways of seq-length setting: PyTorch speculator training [blog](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/),
    # they have a two-phase approach to training a speculator.
    # in phase 1, they train on small batches with long sequence lengths (4k tokens)
    # and use the standard causal LM approach for training.
    # In phase 2, they use large batches with short sequence lengths (256 tokens) generated from the base model.
    # In this training phase, they tune the heads to match the output of the base model.
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=10)
    parser.add_argument("--steps-per-save", type=int, default=10)
    parser.add_argument("--grad-checkpoint", action="store_true", default=True)
    # enable gradient checkpointing to save memory, at the cost of extra computation durinf backwared pass 
    parser.add_argument("--max-train-samples", type=int, default=10)
    parser.add_argument("--max-eval-samples", type=int, default=100)

    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help="Services to report logs to ('wandb', 'swanlab', or 'wandb,swanlab').",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project name for logging. Defaults to the name of the root directory.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)


    print(f"Loading student: {args.student_model}")
    student_model, student_tokenizer = load(args.student_model)

    print(f"Loading teacher: {args.teacher_model}")
    teacher_model, teacher_tokenizer = load(args.teacher_model)

    # same tokenizer family / vocab alignment.
    # distillation on logits requires that.
    if getattr(student_tokenizer, "eos_token_id", None) != getattr(
        teacher_tokenizer, "eos_token_id", None
    ):
        print(
            "[WARN] Teacher/student EOS token ids differ. "
            "Make sure these models share the same tokenizer/vocab."
        )

    teacher_model.eval()

    train_data, eval_data = build_tokenized_splits(
        dataset_name=args.dataset,
        tokenizer=student_tokenizer,
        seed=args.seed,
        eval_ratio=args.eval_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    train_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=-1,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accum,
    )

    distill_ctx = DistillContext(
        teacher_model=teacher_model,
        temperature=args.temperature,
    )
    loss_fn = make_kl_loss(distill_ctx)

    training_callback = get_reporting_callbacks(
        "wandb", # args.report_to,
        project_name="spec-de",
        log_dir="adapters",
        config=vars(args),
    )

    print("Starting full-model distillation...")
    train(
        model=student_model,
        optimizer=optimizer,
        train_dataset=train_data,
        val_dataset=eval_data,
        args=train_args,
        loss=loss_fn,
        training_callback=training_callback,

    )

    print("Done. Saving full student model weights...")

    os.makedirs("adapters", exist_ok=True)

    weight_path = os.path.join("adapters", "adapters.safetensors")
    meta_path = os.path.join("adapters", "adapter_config.json")

    full_weights = {
        str(k): v for k, v in tree_flatten(student_model.parameters())
    }

    mx.save_safetensors(weight_path, full_weights)

    meta = {
        "base_model": args.student_model,
        "teacher_model": args.teacher_model,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "fine_tune_type": "full",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved weights to {weight_path}")
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()