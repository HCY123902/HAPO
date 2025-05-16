import json
from math_verify import parse, verify, math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from math_utils import is_correct, remove_boxed, last_boxed_only_string

from transformers import AutoTokenizer

import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the parquet file produced by lighteval for a specific evaluation run.")
args = parser.parse_args()
path = args.path

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


max_model_len = 32768

show_per_level_results = False


if path.endswith(".parquet"):
    df_parquet = pd.read_parquet(path)

    samples = [{"solution": row["gold"][0], "all_generated_responses": row["predictions"], "level": ""} for index, row in df_parquet.iterrows()]

elif path.endswith(".json"):
    with open(path, "r") as src_json:
        samples = json.load(src_json)

else:
    print("Cannot open {}".format(path))

if not show_per_level_results:
    hit = []
    skipped = 0

    lens = []

    for i, s in enumerate(samples):
        for resp in s["all_generated_responses"]:
            lens.append(len(tokenizer.encode(resp, add_special_tokens=False)))
            try:
                answer = parse(resp)
                gold = parse(s["solution"])
                is_eq = int(verify(gold, answer))
            except Exception as e:
                skipped = skipped + 1
                is_eq = 0
            
            if is_eq == 1:
                hit.append(1)
            else:
                hit.append(0)

    print("{} skipped".format(skipped))
    print("Accuracy: {}".format(sum(hit)/len(hit)))

    print("Response length: {}\n\n".format(sum(lens)/len(lens)))


else:

    res_map = {}

    for s in samples:
        curr_hits = []
        curr_lens = []

        if s["level"] != "":
            diff_level = int(s["level"].split()[-1]) if isinstance(s["level"], str) else int(s["level"])
        else:
            diff_level = 1

        if diff_level not in res_map:
            res_map[diff_level] = {"hits": 0, "lens": 0, "solution_lens": 0, "num_examples": 0, "excess_count": 0, "total_count": 0, "max_sample_length": 0, "skipped_count": 0}
        
        res_map[diff_level]["total_count"] = res_map[diff_level]["total_count"] + len(s["all_generated_responses"])
        
        for resp in s["all_generated_responses"]:
            try:
                answer = parse(resp)
                gold = parse(s["solution"])
                is_eq = int(verify(gold, answer))
            except Exception as e:
                res_map[diff_level]["skipped_count"] = res_map[diff_level]["skipped_count"] + 1
                is_eq = 0

            if is_eq==1:
                curr_hits.append(1)
            else:
                curr_hits.append(0)
            
            
            resp_len = len(tokenizer.encode(resp, add_special_tokens=False))
            curr_lens.append(resp_len)

            curr_sample_length = len(tokenizer.encode(s["text_prompt"], add_special_tokens=False)) + resp_len
            
            if curr_sample_length >= max_model_len:
                res_map[diff_level]["excess_count"] = res_map[diff_level]["excess_count"] + 1
            res_map[diff_level]["max_sample_length"] = max(res_map[diff_level]["max_sample_length"], curr_sample_length)

            # print("answer: {}; prediction: {}; is_eq: {}".format(answer, prediction, is_eq))
        s["is_correct"] = curr_hits
        s["resp_lens"] = curr_lens

        res_map[diff_level]["hits"] = res_map[diff_level]["hits"] + sum(curr_hits) / len(curr_hits)
        res_map[diff_level]["lens"] = res_map[diff_level]["lens"] + sum(curr_lens) / len(curr_lens)
        res_map[diff_level]["solution_lens"] = res_map[diff_level]["solution_lens"] + len(tokenizer.encode(s["solution"], add_special_tokens=False))
        res_map[diff_level]["num_examples"] = res_map[diff_level]["num_examples"] + 1 

    

    for dl in sorted(list(res_map.keys())):
        res = res_map[dl]
        print("Difficulty level: {}\nAccuracy: {}\nAvg len: {}\nAvg solution len: {}\nNum examples: {}\nExcess count: {}\nSkipped count: {}\nTotal count: {}\nMaximum sample length: {}\n\n".format(
                dl,
                res["hits"]/res["num_examples"],
                res["lens"]/res["num_examples"],
                res["solution_lens"]/res["num_examples"],
                res["num_examples"],
                res["excess_count"],
                res["skipped_count"],
                res["total_count"],
                res["max_sample_length"],
            )
        )
    total_res_map = {
        key: sum([res_map[dl][key] for dl in res_map]) for key in res_map[1]
    }

    print("Combined: Accuracy: {}\nAvg len: {}\nAvg solution len: {}\nNum examples: {}\nExcess count: {}\nSkipped count: {}\nTotal count: {}\nMaximum sample length: {}\n\n".format(
            total_res_map["hits"]/total_res_map["num_examples"],
            total_res_map["lens"]/total_res_map["num_examples"],
            total_res_map["solution_lens"]/total_res_map["num_examples"],
            total_res_map["num_examples"],
            total_res_map["excess_count"],
            total_res_map["skipped_count"],
            total_res_map["total_count"],
            max([res_map[dl]["max_sample_length"] for dl in res_map]),
        )
    )
    
