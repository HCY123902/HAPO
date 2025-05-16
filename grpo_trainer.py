# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers.trainer_utils import PredictionOutput

from trl import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

from evaluation.math_utils import is_correct
from utils import *

import json

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb



history_tracker = None




def combined_reward_func(completions, ground_truth, **kwargs):
    global history_tracker
    combined_rewards = []
    
    completions_tokens = history_tracker.tokenizer(completions, add_special_tokens=False)["input_ids"]

    curr_batch_proc_corr_lens = {}
    for prompt_idx in kwargs["prompt_idx"]:
        curr_batch_proc_corr_lens[prompt_idx] = []

    gathered_len_dict = history_tracker.gather_len_dict()

    for resp, resp_tokens, gt, prompt_idx in zip(completions, completions_tokens, ground_truth, kwargs["prompt_idx"]):
        r_corr = int(is_correct(resp, gt, use_math_verify=history_tracker.use_math_verify))
        
        seq_len = len(resp_tokens)
        
        r_len = history_tracker.compute_len_reward(seq_len, r_corr>=1.0, prompt_idx, gathered_len_dict)

        r_rep = history_tracker.compute_rep_reward(resp_tokens)

        combined_rewards.append(r_corr+r_len+r_rep)

        # Only includes the part of history that the current process accesses
        if r_corr >= 1:
            curr_batch_proc_corr_lens[prompt_idx].append(seq_len)

    for prompt_idx in kwargs["prompt_idx"]:
        if len(curr_batch_proc_corr_lens[prompt_idx]) == 0:
            continue
        if history_tracker.mode == "min":
            history_tracker.lens_dict[prompt_idx] = min(min(curr_batch_proc_corr_lens[prompt_idx]), history_tracker.lens_dict[prompt_idx].item())
        elif history_tracker.mode == "mean":
            curr_batch_mean = sum(curr_batch_proc_corr_lens[prompt_idx])/len(curr_batch_proc_corr_lens[prompt_idx])
            if history_tracker.count_dict[prompt_idx].item() <= 0:
                assert history_tracker.lens_dict[prompt_idx].item() == MAX_LEN
                history_tracker.lens_dict[prompt_idx] = curr_batch_mean
            else:
                assert history_tracker.lens_dict[prompt_idx].item() < MAX_LEN
                ratio = len(curr_batch_proc_corr_lens[prompt_idx]) / (history_tracker.count_dict[prompt_idx].item() + len(curr_batch_proc_corr_lens[prompt_idx]))
                history_tracker.lens_dict[prompt_idx] = ratio * curr_batch_mean + (1-ratio) * history_tracker.lens_dict[prompt_idx].item()
            history_tracker.count_dict[prompt_idx] = history_tracker.count_dict[prompt_idx] + len(curr_batch_proc_corr_lens[prompt_idx])

    return combined_rewards



class HistroyTracker:
    def __init__(self, accelerator, tokenizer, train_dataset_len, eval_dataset_len, w_lr=1.0, type_lr="cosine", rep_ngram_size=3, rep_penalty=0.0, mode="min"):
        self.accelerator = accelerator
        self.tokenizer = tokenizer

        self.lens_dict = check_dataset_len_and_init_len_dict(train_dataset_len, eval_dataset_len, self.accelerator.device, to_long=False)

        self.tolerance_ratio = 0.1

        self.consider_readability = False
        
        self.w_lr = w_lr
        self.accelerator.print("Setting w_lr to {}".format(self.w_lr))

        self.compute_lr = None
        if type_lr=="cosine":
            self.compute_lr = compute_len_reward
        elif type_lr=="linear":
            self.compute_lr = compute_len_reward_linear
        else:
            raise Exception("{} is not a valid type for length reward".format(type_lr))

        self.rep_ngram_size = rep_ngram_size
        self.rep_penalty = rep_penalty

        self.accelerator.print("Setting rep_ngram_size to {} and rep_penalty to {}".format(self.rep_ngram_size, self.rep_penalty))

        self.mode = mode
        if self.mode == "mean":
            self.count_dict = check_dataset_len_and_init_len_dict(train_dataset_len, eval_dataset_len, self.accelerator.device, to_long=False, fill_value=0)
        self.accelerator.print("Setting mode to {}".format(self.mode))
    
        self.use_math_verify = True
        self.accelerator.print("Setting use_math_verify to {}".format(self.use_math_verify))

    def gather_len_dict(self):
        gathered_len_dict = self.accelerator.gather(self.lens_dict)
        assert gathered_len_dict.size(0) == (MAX_TRAIN_SET_SIZE+MAX_VALID_SET_SIZE) * self.accelerator.num_processes
        gathered_len_dict = gathered_len_dict.reshape(self.accelerator.num_processes, MAX_TRAIN_SET_SIZE+MAX_VALID_SET_SIZE)

        if self.mode == "min":
            gathered_len_dict = gathered_len_dict.min(dim=0)[0]
        elif self.mode == "mean":
            gathered_count_dict = self.accelerator.gather(self.count_dict).reshape(self.accelerator.num_processes, MAX_TRAIN_SET_SIZE+MAX_VALID_SET_SIZE)
            total_count = gathered_count_dict.sum(dim=0, keepdim=True)
            total_count = torch.maximum(torch.ones_like(total_count), total_count) # Prevent division by 0 error when the total count is 0
            gathered_ratio = gathered_count_dict / total_count # total_count is automatically broadcasted along dimension 0
            
            gathered_len_dict = gathered_len_dict * gathered_ratio # If the total count is 0 for a prompt_idx, then the gathered_len_dict will have a value of 0 in that entry
            gathered_len_dict = gathered_len_dict.sum(dim=0)

        return gathered_len_dict
    
    def compute_len_reward(self, seq_len, is_corr, prompt_idx, gathered_len_dict):
        history_len = gathered_len_dict[prompt_idx].item()

        r = self.compute_lr(
            history_len=history_len, 
            seq_len=seq_len, 
            is_corr=is_corr, 
            prompt_idx=prompt_idx, 
            consider_readability=self.consider_readability, 
            tolerance_ratio=self.tolerance_ratio, 
            w_lr=self.w_lr
        )

        return r



    def compute_rep_reward(self, resp_tokens):
        return compute_repetition_penalty_reward(resp_tokens=resp_tokens, ngram_size=self.rep_ngram_size, max_penalty=self.rep_penalty, only_start=False)



class GRPOTrainerWithLengthReward(GRPOTrainer):
    def init_history_tracker(self, w_lr=1.0, type_lr="cosine", rep_ngram_size=3, rep_penalty=0.0, mode="min"):
        train_dataset_len = len(self.train_dataset)
        eval_dataset_len = len(self.eval_dataset)
        self.accelerator.print("Train dataset length: {}".format(train_dataset_len))
        global history_tracker
        history_tracker = HistroyTracker(self.accelerator, self.tokenizer, train_dataset_len, eval_dataset_len, w_lr=w_lr, type_lr=type_lr, rep_ngram_size=rep_ngram_size, rep_penalty=rep_penalty, mode=mode)
        self.eval_s = 1 if self.args.eval_steps is None else self.args.eval_steps
        if self.args.eval_on_start:
            self.curr_eval_step = 0
        else:
            self.curr_eval_step = self.eval_s
        while os.path.exists(os.path.join(self.args.output_dir, "history_lens_dict_step_{}.json".format(self.curr_eval_step))):
            self.curr_eval_step = self.curr_eval_step + self.eval_s
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        global history_tracker

        res = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        gathered_len_dict = history_tracker.gather_len_dict()
        if self.accelerator.is_main_process:
            with open(os.path.join(self.args.output_dir, "history_lens_dict_step_{}.json".format(self.curr_eval_step)), "w") as tgt_json:
                json.dump({i: gathered_len_dict[i].item() for i in range(gathered_len_dict.size(0))}, tgt_json, indent=4)
        self.curr_eval_step = self.curr_eval_step + self.eval_s
        return res
    
    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[list[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        global history_tracker

        res = super().perdict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        gathered_len_dict = history_tracker.gather_len_dict()
        if self.accelerator.is_main_process:
            with open(os.path.join(self.args.output_dir, "history_lens_dict_step_{}.json".format("predict")), "w") as tgt_json:
                json.dump({i: gathered_len_dict[i].item() for i in range(gathered_len_dict.size(0))}, tgt_json, indent=4)
        return res