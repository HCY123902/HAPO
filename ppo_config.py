from dataclasses import dataclass
from typing import Dict, Literal, Optional
from trl import PPOConfig


@dataclass
class PPOWtihLengthRewardConfig(PPOConfig):
    w_lr: float = 1.0
    type_lr: Literal["cosine", "linear"] = "cosine"
    mode: Literal["min", "mean"] = "min"
    rep_ngram_size: int = 3
    rep_penalty: float = 0.0
    max_prompt_length: int = 512