import pandas as pd
import polars as pl

from transformers import AutoTokenizer

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the parquet file produced by lighteval for a specific evaluation run.")
args = parser.parse_args()
path = args.path

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

lens = []

if "lcb:codegeneration" in path:
    df_parquet = pl.read_parquet(path)
    for i in range(len(df_parquet)):
        assert len(df_parquet[i]["predictions"]) == 1
        for pred in df_parquet[i]["predictions"][0]:
            lens.append(len(tokenizer.encode(pred, add_special_tokens=False)))
else:
    df_parquet = pd.read_parquet(path)
    for index, row in df_parquet.iterrows():
        for pred in row["predictions"]:
            lens.append(len(tokenizer.encode(pred, add_special_tokens=False)))


print("Response length: {}\n\n".format(path, sum(lens)/len(lens)))