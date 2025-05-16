import torch
import math

MAX_LEN = 100000

MAX_TRAIN_SET_SIZE = 10000

MAX_VALID_SET_SIZE = 1000

def compute_len_reward_linear(history_len, seq_len, is_corr, prompt_idx, consider_readability: bool, tolerance_ratio: float, w_lr: float):
    if history_len < MAX_LEN and history_len > 0:
        # h_i is not Null
        if is_corr:
            if seq_len < history_len:
                ratio_len = seq_len / history_len
                if consider_readability:
                    raise NotImplementedError
                else:
                    r = 1.0 - ratio_len
            elif seq_len >= history_len and seq_len < (1+tolerance_ratio) * history_len:
                r = 0
            else:
                unconstrained_r = -1.0 * ((seq_len - (1+tolerance_ratio) * history_len) / history_len)
                # Any response that is correct should have a higher combined reward than that of any response that is incorrect
                r = max(-0.7, unconstrained_r)
        else:
            if seq_len < (1+tolerance_ratio) * history_len:
                r = 0
            else:
                unconstrained_r = -1.0 * ((seq_len - (1+tolerance_ratio) * history_len) / history_len)
                r = max(-1.0, unconstrained_r)
    else:
        # h_i is Null
        r = 0
        
    return r * w_lr

def compute_len_reward(history_len, seq_len, is_corr, prompt_idx, consider_readability: bool, tolerance_ratio: float, w_lr: float):
    if history_len < MAX_LEN and history_len > 0:
        # h_i is not Null
        if seq_len < 2 * history_len:
            r = math.cos((seq_len/history_len) * (math.pi/2))
        else:
            r = math.cos(math.pi) # Which is -1

        if is_corr:
            r = max(-0.7, r)
        else:
            r = min(0, r)
    else:
        # h_i is Null
        r = 0
        
    return r * w_lr



def check_dataset_len_and_init_len_dict(train_dataset_len, eval_dataset_len, device, to_long=True, fill_value=MAX_LEN):
    assert train_dataset_len <= MAX_TRAIN_SET_SIZE and eval_dataset_len <= MAX_VALID_SET_SIZE
    if to_long:
        return torch.full(size=(MAX_TRAIN_SET_SIZE+MAX_VALID_SET_SIZE,), fill_value=fill_value, dtype=torch.long).to(device=device)
    else:
        return torch.full(size=(MAX_TRAIN_SET_SIZE+MAX_VALID_SET_SIZE,), fill_value=fill_value, device=device)


# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
def zipngram(tokens: list[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])

def compute_repetition_penalty_reward(resp_tokens: list[int], ngram_size: int, max_penalty: float, only_start: bool = False) -> float:
    """
    reward function the penalizes repetitions
    ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    Args:
        resp_tokens: Token ids of a single response
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")


    if max_penalty == 0.0 or len(resp_tokens) < ngram_size:
        return 0.0
    
    # Find repeated n-grams and their positions
    
    repeated_positions = []
    ngrams = set()

    for start_idx, ng in enumerate(zipngram(resp_tokens, ngram_size)):
        if ng in ngrams:
            repeated_positions.append(start_idx)
        ngrams.add(ng)

    # Calculate word-level penalties
    word_penalties = [0.0] * len(resp_tokens)
    curr_end_idx = -1

    for start_idx in repeated_positions:
        if not only_start or start_idx > curr_end_idx:
            # Apply penalty to each token in the repeated n-gram
            for i in range(start_idx, start_idx + ngram_size):
                word_penalties[i] = max_penalty
        # Changed from curr_end_idx = start_idx + ngram_size to curr_end_idx = start_idx + ngram_size - 1
        curr_end_idx = start_idx + ngram_size - 1

    # Average the word-level penalties for the final reward
    reward = sum(word_penalties) / len(word_penalties) if word_penalties else 0.0


    return reward


def compute_repetition_penalty_reward_1(resp_tokens: list[int], ngram_size: int, max_penalty: float) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0.0:
        return 0.0

    ngrams = set()
    total = 0
    for ng in zipngram(resp_tokens, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty
