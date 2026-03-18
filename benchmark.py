from inference import decode, speculative_decode, _speculative_decode
from mlx_lm import load


DRAFT_MODEL_SIZE = "0.6B"
TARGET_MODEL_SIZE = "1.7B"
# full-weights fine-tuned
DISTILLED_DRAFT_MODEL_NAME  = f"Austin362667/Qwen3-{DRAFT_MODEL_SIZE}-MLX-bf16-python-18k-alpaca"
# not fine-tuned
DRAFT_MODEL_NAME  = f"Qwen/Qwen3-{DRAFT_MODEL_SIZE}-MLX-bf16"
# the model we want to speed up
TARGET_MODEL_NAME = f"Qwen/Qwen3-{TARGET_MODEL_SIZE}-MLX-bf16"

MAX_NEW_TOKENS = 64
K = 3

VERBOSE = False

# A batch of prompts
non_coding_prompts = [
    "MLX is super fast, don't you think so?",
    "Write a news about Einstein.",
    "Why is the sky blue?",
    "How tall is Mt Everest?",
    "Write a poem about the ocean.",
    "What is the meaning of life?",
    "Who is Tommy Shelby?",
    "I love you, do you?",
    "Name some of best ideas in Computer Architecture!",
    "Rewrite email in Mandarin: Attached my resume. Let me know if you have any questions. Thank you."
]

coding_prompts = [
    "Generate Python code of Dijkstra algorithm.",
    "Write a Python function to compute the Fibonacci sequence.",
    "Generate Python code to implement a binary search tree.",
    "Write a Python script to scrape data from a website.",
    "Python codebase for a simple web server using Flask.",
    "Design a efficient Python function to perform matrix multiplication.",
    "Generate Python code to read and write CSV files.",
    "Write a Python script to analyze a dataset using Pandas.",
    "Give me Python code that doing self-attention.",
    "Write a Python function to implement the quicksort algorithm."
]


def bench_baseline(workload, model, tokenizer, max_new_tokens, verbose=False):
    # history
    prefill_tps_list = []
    decode_tps_list = []
    for prompt in workload:
        response_text, num_prefills, prefill_tps, num_decodes, decode_tps = decode(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        prefill_tps_list.append(prefill_tps)
        decode_tps_list.append(decode_tps)
        if verbose:
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response_text}")
            print(f"Prefill Tokens: {num_prefills} | Prefill Tokens-per-Second: {prefill_tps:.3f} | Decode Tokens: {num_decodes} | Decode Tokens-per-Second: {decode_tps:.3f}")

    return sum(prefill_tps_list) / len(prefill_tps_list), sum(decode_tps_list) / len(decode_tps_list)


def bench_speculative_decoding(workload, target_model, draft_model, tokenizer, k, max_new_tokens, verbose=False):
    # history
    prefill_tps_list = []
    decode_tps_list = []
    avg_accepted_len_list = []
    
    for prompt in workload:
        response_text, num_prefills, prefill_tps, num_decodes, decode_tps, avg_accepted_len = speculative_decode(target_model, draft_model, tokenizer, k=k, prompt=prompt, max_new_tokens=max_new_tokens)
        prefill_tps_list.append(prefill_tps)
        decode_tps_list.append(decode_tps)
        avg_accepted_len_list.append(avg_accepted_len)
        if verbose:
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response_text}")
            print(f"Prefill Tokens: {num_prefills} | Prefill Tokens-per-Second: {prefill_tps:.3f} | Decode Tokens: {num_decodes} | Decode Tokens-per-Second: {decode_tps:.3f} | Average accepted length: {avg_accepted_len:.2f} Tokens")

    return sum(prefill_tps_list) / len(prefill_tps_list), sum(decode_tps_list) / len(decode_tps_list), sum(avg_accepted_len_list) / len(avg_accepted_len_list)


if __name__ == "__main__":
    target_model, tokenizer = load(TARGET_MODEL_NAME)
    draft_model, _ = load(DRAFT_MODEL_NAME)
    distilled_draft_model, _ = load(DISTILLED_DRAFT_MODEL_NAME)

    print("=" * 30)
    print("Benchmarking on Non-coding Workload")
    # avg across whole workload eval dataset    
    _, avg_decode_tps = bench_baseline(non_coding_prompts, target_model, tokenizer, MAX_NEW_TOKENS, VERBOSE)
    _, avg_sd_decode_tps, avg_sd_accept_len = bench_speculative_decoding(non_coding_prompts, target_model, draft_model, tokenizer, K, MAX_NEW_TOKENS, VERBOSE)
    _, avg_sd_distill_decode_tps, avg_sd_distill_accept_len = bench_speculative_decoding(non_coding_prompts, target_model, distilled_draft_model, tokenizer, K, MAX_NEW_TOKENS, VERBOSE)
    print("Speedup Report:")
    print(f"{'Model':<70} | {'Decoding TPS':<30} | {'Avg Accepted Len':<30} | {'Avg Speed Up':<30}")
    print(f"{'Baseline':<70} | {avg_decode_tps:<30.3f} | {'-':<30} | {avg_decode_tps/avg_decode_tps:.2f}")
    print(f"{'Speculate with Non-distilled Draft':<70} | {avg_sd_decode_tps:<30.3f} | {avg_sd_accept_len:<30.2f} | {avg_sd_decode_tps/avg_decode_tps:.2f}")
    print(f"{'Speculate with Distilled Draft':<70} | {avg_sd_distill_decode_tps:<30.3f} | {avg_sd_distill_accept_len:<30.2f} | {avg_sd_distill_decode_tps/avg_decode_tps:.2f}")

    print("\n" + "=" * 30)
    print("Benchmarking on Coding Workload")
    # avg across whole workload eval dataset    
    _, avg_decode_tps = bench_baseline(coding_prompts, target_model, tokenizer, MAX_NEW_TOKENS, VERBOSE)
    _, avg_sd_decode_tps, avg_sd_accept_len = bench_speculative_decoding(coding_prompts, target_model, draft_model, tokenizer, K, MAX_NEW_TOKENS, VERBOSE)
    _, avg_sd_distill_decode_tps, avg_sd_distill_accept_len = bench_speculative_decoding(coding_prompts, target_model, distilled_draft_model, tokenizer, K, MAX_NEW_TOKENS, VERBOSE)
    print("Speedup Report:")
    print(f"{'Model':<70} | {'Decoding TPS':<30} | {'Avg Accepted Len':<30} | {'Avg Speed Up':<30}")
    print(f"{'Baseline':<70} | {avg_decode_tps:<30.3f} | {'-':<30} | {avg_decode_tps/avg_decode_tps:.2f}")
    print(f"{'Speculate with Non-distilled Draft':<70} | {avg_sd_decode_tps:<30.3f} | {avg_sd_accept_len:<30.2f} | {avg_sd_decode_tps/avg_decode_tps:.2f}")
    print(f"{'Speculate with Distilled Draft':<70} | {avg_sd_distill_decode_tps:<30.3f} | {avg_sd_distill_accept_len:<30.2f} | {avg_sd_distill_decode_tps/avg_decode_tps:.2f}")

