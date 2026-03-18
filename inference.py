import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


# Regular autoregressive decoding.
def decode(model, tokenizer, prompt="", max_new_tokens=64):
    conversations = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(
        conversations, add_generation_prompt=True,
        enable_thinking=False,
    )

    text = ""
    for response in stream_generate(model, tokenizer, prompt_tokens, max_new_tokens):
        text += response.text

    if len(text) == 0:
        print("No text generated for this prompt")
        return
    
    # return response texts, number of prefilled tokens & their tps, number of generated tokens & their tps
    return text, response.prompt_tokens, response.prompt_tps, response.generation_tokens, response.generation_tps


# Speculative decoding.
def speculative_decode(target_model, draft_model, tokenizer, k=3, prompt="", max_new_tokens=64):
    conversations = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(
        conversations, add_generation_prompt=True,
        enable_thinking=False,
    )

    accepted_len = 0
    accepted_len_list = []

    text = ""
    for response in stream_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_new_tokens,
        draft_model=draft_model,
        num_draft_tokens=k
    ):
        text += response.text

        if response.from_draft: # a boolean
            accepted_len += 1
        else:
            accepted_len_list.append(accepted_len)
            accepted_len = 0
    
    if len(text) == 0:
        print("No text generated for this prompt")
        return

    avg_accepted_len = sum(accepted_len_list) / len(accepted_len_list) if accepted_len_list else 0.0
    
    # return response texts, number of prefilled tokens & their tps, number of generated tokens & their tps and average accepted length of draft tokens
    # `GenerationResponse`: https://github.com/ml-explore/mlx-lm/blob/564281f79328df07c4997b3a6ca00bd929381287/mlx_lm/generate.py#L266
    return text, response.prompt_tokens, response.prompt_tps, response.generation_tokens, response.generation_tps, avg_accepted_len

# Same speculative decoding without using `mlx_lm` built-in `generate`.
def _speculative_decode(target_model, draft_model, tokenizer, k=3, prompt="", max_new_tokens=64):
    conversations = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(
        conversations, add_generation_prompt=True,
        enable_thinking=False,
    )
    draft_kv_cache = make_prompt_cache(draft_model)
    target_kv_cache = make_prompt_cache(target_model)

    accepted_len = 0
    accepted_len_list = []
    generated_tokens = []

    t_start = time.perf_counter()

    # 1. prefill the whole prompt (prompt length `n`) for both models
    draft_out  = draft_model(mx.array([prompt_tokens]), cache=draft_kv_cache)
    target_out = target_model(mx.array([prompt_tokens]), cache=target_kv_cache)
    mx.async_eval(draft_out, target_out)
    
    prefill_elapsed = time.perf_counter() - t_start
    t_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        k = min(k, max_new_tokens - len(generated_tokens)) 
        
        draft_q_list = []
        draft_probs_list = []
        draft_token_list = []

        # 2. decode the next token right after the prefilled prompt (1st within k drafted tokens) in draft model
        draft_logit = draft_out[0, -1] # B, S 
        draft_probs = mx.softmax(draft_logit, axis=-1)
        # sampling from draft distribution
        # mlx_lm default sampler is `mlx.core.argmax`
        draft_token = mx.random.categorical(mx.log(draft_probs)).item()
        
        draft_q_list.append(draft_probs[draft_token].item())
        draft_probs_list.append(draft_probs)
        draft_token_list.append(draft_token)

        while len(draft_token_list) < k:
            # 3. decode the next token right after the prefilled prompt + alreafy drafted tokens (2nd~kth within k drafted tokens)
            draft_out = draft_model(mx.array([[draft_token]]), cache=draft_kv_cache)
            draft_logit = draft_out[0, -1] # B, S
            draft_probs = mx.softmax(draft_logit, axis=-1)
            # sampling from draft distribution
            draft_token = mx.random.categorical(mx.log(draft_probs)).item()
            draft_q_list.append(draft_probs[draft_token].item())
            draft_probs_list.append(draft_probs)
            draft_token_list.append(draft_token)

        # 3. verify (prefilled prompt + k drafted tokens) in target model
        verify_out = target_model(mx.array([draft_token_list]), cache=target_kv_cache)
        mx.async_eval(verify_out)

        eos_reached = False
        rejected = False

        for i in range(k):
            target_logit = target_out[0, -1] if i == 0 else verify_out[0, i - 1]
            target_probs = mx.softmax(target_logit, axis=-1)
            
            # rejection sampling with p (target prob) and q (draft prob)
            p = target_probs[draft_token_list[i]].item()
            q = draft_q_list[i]

            if mx.random.uniform().item() < min(1.0, p / q):
                # Accept
                generated_tokens.append(draft_token_list[i])
                accepted_len += 1 # https://github.com/ml-explore/mlx-lm/blob/564281f79328df07c4997b3a6ca00bd929381287/mlx_lm/generate.py#L617

                if draft_token_list[i] == tokenizer.eos_token_id:
                    eos_reached = True
                    break
            else:
                # Reject
                
                # trim reject cache in-place
                # layer.offset = max(0, layer.offset - n)
                # https://github.com/ml-explore/mlx-lm/blob/47be7150a6e6069ede2dc7e6cfa8b23a520b9804/mlx_lm/models/cache.py#L212
                trim_prompt_cache(draft_kv_cache, k - 1 - i)
                trim_prompt_cache(target_kv_cache, k - i)

                # resample from residual distribution 
                # after applying softmax, we get and treat it probility distribution afterawads.
                # subtracting two logits before applying softmax gives us identical results too.
                q_probs   = draft_probs_list[i]
                residual = mx.maximum(target_probs - q_probs, 0) # clamp
                residual = residual / residual.sum() # norm (residual is non-negative)
                corrected_token = mx.random.categorical(mx.log(residual)).item()
                generated_tokens.append(corrected_token)

                # advance both caches with the resampled token (update KV caches as well)
                draft_out = draft_model(mx.array([[corrected_token]]), cache=draft_kv_cache)
                target_out = target_model(mx.array([[corrected_token]]), cache=target_kv_cache)
                
                if corrected_token == tokenizer.eos_token_id:
                    eos_reached = True

                rejected = True
                break

        if not rejected:
            # all k accepted (get a bonus token from last target logit)
            bonus_probs = mx.softmax(verify_out[0, k - 1], axis=-1)
            bonus_token = mx.random.categorical(mx.log(bonus_probs)).item()
            generated_tokens.append(bonus_token)
            accepted_len += 1 # To match mlx_lm behavior: https://github.com/ml-explore/mlx-lm/blob/564281f79328df07c4997b3a6ca00bd929381287/mlx_lm/generate.py#L622
            # advance both caches with bonus token
            draft_out = draft_model(mx.array([[bonus_token]]), cache=draft_kv_cache)
            target_out = target_model(mx.array([[bonus_token]]), cache=target_kv_cache)

        accepted_len_list.append(accepted_len)
        accepted_len = 0

        if eos_reached:
            break

    decode_elapsed = time.perf_counter() - t_start
    text = tokenizer.decode(generated_tokens)
    prefill_tps = len(prompt_tokens) / prefill_elapsed if prefill_elapsed > 0 else 0
    decode_tps = len(generated_tokens) / decode_elapsed if decode_elapsed > 0 else 0
    avg_accepted_len = sum(accepted_len_list) / len(accepted_len_list) if accepted_len_list else 0

    return text, len(prompt_tokens), prefill_tps, len(generated_tokens), decode_tps, avg_accepted_len

