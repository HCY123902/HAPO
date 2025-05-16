# HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization

This repostiory contains the code and dataset for our paper: HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization.

## Set up

1. Run `nvcc -V` and make sure your system CUDA version is 12.1 or newer.

1. Clone this repository, and then create a new conda environment with the required packages

```
conda create -n hapo python=3.11
conda activate hapo
cd HAPO
pip install -r requirements.txt
```

1. Sign in to HuggingFace and Weight and Biases with your respective tokens
```
huggingface-cli login
wandb init
```

## Datasets

Our training and validation set is in the `datasets` folder. We use `train_samples_dsr_2000.json` for training and `valid_samples_dsr_500.json` for validation in our main experiments. The examples are sampled from [`DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset).

Each example in the dataset has a question in the `"question"` key, answer in the `"ground_truth"` key, and optionally complete solution in the `"solution"` key.

## Training

To train with GRPO, run `bash train_grpo.sh`. To train with PPO, run `bash train_ppo.sh`.

### Additional Training Arguments

`w_lr`: Weight of the length reward. Defaults to `1.0`. A higher value means more emphasis of the length reard. The weight needs to be in the range [0, 1], to ensure that any correct response has a higher combined reward than any incorrect response.
`type_lr`: The type of length reward function. Defaults to `"cosine"`. The other option `"linear"` produces a picece wise linear function for length reward computation.
`mode`: The update mode, also denoted as the aggregate function in the paper. Defaults to `"min"`. `"mean"` makes `h_i` keep track of the mean length of historical correct responses, instead of the minimum length of them.
`rep_ngram_size`: Size of ngram for the repetition reward. Defaults to `3`. A higher value means only penalizing repeated ngrams that are larger than or equal to the specified size.
`rep_penalty`: Maximum penalty for the repetition reward. Defaults to `0.0`, meaning no penalty. The penalty needs to be a non positive number. A smaller value means more emphasis on the reptition reward.

## Evaluation

We use [`lighteval`](https://github.com/huggingface/lighteval) to evaluate the trained models. The evaluation scripts are provided in the `evaluation` folder.

1. Run `bash evaluate_{benchmark}.sh` to evaluate the model, where `{benchmark}` can be `gsm8k` (GSM8k), `math_500` (MATH500), `ae_2024` (AIME2024), `gpqa` (GPQA), or `lcb` (LiveCodeBench).

1. Since `lighteval` only computes accuracy, we provide additional scripts to compute response lengths. Run `python compute_length.py --path {path_to_parquet}` where `{path_to_parquet}` points to the `parquet` file produced by `lighteval` for a specific evaluation run. 

1. Additionally, we notice that `lighteval` produces an accuracy of 0 for GSM8K benchmark, and therefore we provide another script to fix this. Run `python verify.py --path {path_to_parquet}` to comput the correct GSM8K accuracy and length of responses.

## Model Checkpoints

The trained model checkpoints will be released soon.