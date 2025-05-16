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

import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    # PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from ppo_trainer import PPOTrainerWithLengthReward
from ppo_config import PPOWtihLengthRewardConfig

from prompting_utils import *

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass
from utils import MAX_TRAIN_SET_SIZE

@dataclass
class DataCollatorWithLengthReward:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tensor_feature_columns = ["input_ids", "lengths"]
        tensor_features = [
            {
                key: item[key]  for key in tensor_feature_columns
            } for item in features
        ]
         
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            tensor_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        non_tensor_batch = {}
        for key in features[0].keys():
            if key not in tensor_feature_columns:
                non_tensor_batch[key] = [item[key] for item in features]
        batch.update(non_tensor_batch)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

# Check modifitcations in https://github.com/huggingface/trl/pull/2560/files to incorporate rule based reward

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOWtihLengthRewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    data_files = {
        "train": "datasets/train_samples_dsr_2000.json",
        "valid": "datasets/valid_samples_dsr_500.json",
    }

    dataset = load_dataset("json", data_files=data_files)

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def prepare_dataset(dataset, tokenizer, is_eval):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element, idx, max_prompt_length: int = 512, is_eval: bool = False):
            input_ids = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": element["question"]
                    }
                ],
                padding=False,
                add_generation_prompt=True,
            )

            if len(input_ids) > 512:
                input_ids = input_ids[-max_prompt_length :]

            return {
                "input_ids": input_ids, 
                "lengths": len(input_ids), 
                "ground_truth": element["ground_truth"], 
                "level": element["level"], 
                "type": element["type"],
                "prompt_idx": idx if not is_eval else idx + MAX_TRAIN_SET_SIZE
            }

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
            with_indices=True,
            fn_kwargs={
                "max_prompt_length": training_args.max_prompt_length,
                "is_eval": is_eval
            }
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer, is_eval=False)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer, is_eval=True)
        # # filtering
        # train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        # if eval_dataset is not None:
        #     eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"


    data_collator = DataCollatorWithLengthReward(tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainerWithLengthReward(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.init_lens_dict(w_lr=training_args.w_lr, type_lr=training_args.type_lr, rep_ngram_size=training_args.rep_ngram_size, rep_penalty=training_args.rep_penalty, mode=training_args.mode)
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()