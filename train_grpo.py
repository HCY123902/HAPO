#!/usr/bin/env python
# coding: utf-8
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "False"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import subprocess
import torch.utils.cpp_extension

import argparse
from datasets import load_dataset, load_from_disk
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
from trl import GRPOConfig
import re
import wandb
import torch
from trl.trainer.callbacks import LogCompletionsCallback
from peft import LoraConfig, get_peft_model

from transformers import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint

import numpy as np
from grpo_trainer import combined_reward_func, GRPOTrainerWithLengthReward

from prompting_utils import *
from utils import MAX_TRAIN_SET_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description="RL Training Configuration")

    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to training dataset JSON file")
    parser.add_argument("--val_dataset", type=str, required=True,
                        help="Path to validation dataset JSON file")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pretrained model name or path")
    parser.add_argument("--deepspeed_path", type=str, required=True,
                        help="Path to deepspeed config file")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                        help="Wandb run name")

    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for training results")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per device batch size")
    parser.add_argument("--max_prompt_length", type=int, default=256,
                        help="Maximum prompt sequence length")
    parser.add_argument("--max_completion_length", type=int, default=800,
                        help="Maximum completion sequence length")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Number of steps between model checkpoints")
    parser.add_argument("--restore", type=bool, default=False,
                        help="Restore from checkpoint or not")

    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of distributed workers")
    parser.add_argument("--gpus_per_worker", type=int, default=1,
                        help="GPUs per worker")
    parser.add_argument("--cpus_per_worker", type=int, default=64,
                        help="CPUs per worker")

    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")


    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="The number of gradient accumulation steps")

    parser.add_argument("--w_lr", type=float, default=1.0,
                        help="Weight for the length reward")

    parser.add_argument("--type_lr", type=str, choices=["linear", "cosine"], default="cosine",
                        help="Type of length reward")

    parser.add_argument("--mode", type=str, choices=["min", "mean"], default="min",
                    help="Histroy tracking mode")

    parser.add_argument("--rep_ngram_size", type=int, default=3,
                        help="Size of ngram for the repetition reward")

    parser.add_argument("--rep_penalty", type=float, default=0.0,
                        help="Penalty for each token in a repeated ngram, which is summed and averaged across the length of the response")

    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt")

    return parser.parse_args()


def main():
    args = parse_args()

    wandb.login()

    training_dataset = load_dataset(
        "json",
        data_files=args.train_dataset,
        split="train"
    )

    val_dataset = load_dataset(
        "json",
        data_files=args.val_dataset,
        split="train"
    )

    temp_tokenizer = AutoTokenizer.from_pretrained(args.model_name)    
    

    def add_prefix(example, idx, is_eval: bool = False):
        if isinstance(example["question"], list):
            question_str = " ".join(str(item) for item in example["question"])
        else:
            question_str = example["question"]

        prompt = temp_tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question_str
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "ground_truth": example["ground_truth"],
            "prompt_idx": idx if not is_eval else idx + MAX_TRAIN_SET_SIZE,
        }


    training_dataset = training_dataset.map(
        add_prefix, 
        remove_columns=training_dataset.column_names, 
        with_indices=True,
        fn_kwargs={
            "is_eval": False,
        }
    )

    val_dataset = val_dataset.map(
        add_prefix, 
        remove_columns=val_dataset.column_names, 
        with_indices=True, 
        fn_kwargs={
            "is_eval": True,
        }
    )

    resume_ckpt = None
    enable_resume = False
    # Training on large datasets takes long time and are preempted more often
    if len(training_dataset) >= 1000:
        enable_resume = True
        resume_ckpt = get_last_checkpoint(args.output_dir)


    def train_func(config):        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True
        )

        
        if args.lora_r > 0:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
            model = get_peft_model(model, peft_config)



        
        training_args = GRPOConfig(
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.batch_size,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_accumulation_steps=4,
            bf16=True,
            deepspeed=args.deepspeed_path,
            report_to='wandb',
            fp16_full_eval=True,
            bf16_full_eval=False,
            run_name=os.path.basename(args.output_dir),
            ddp_find_unused_parameters=False,
            log_completions=True,
            use_vllm=True,
            vllm_device="cuda:0",
            vllm_gpu_memory_utilization=0.4,
            save_only_model=True if not enable_resume else False,
            save_total_limit=args.num_epochs,
            temperature=args.temperature,
        )

        
        trainer = GRPOTrainerWithLengthReward(
            model=model,
            args=training_args,
            reward_funcs=combined_reward_func,
            train_dataset=training_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,

        )

        trainer.init_history_tracker(w_lr=args.w_lr, type_lr=args.type_lr, rep_ngram_size=args.rep_ngram_size, rep_penalty=args.rep_penalty, mode=args.mode)

        if resume_ckpt is not None:
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()



    train_func(config=None)




if __name__ == "__main__":
    main()
