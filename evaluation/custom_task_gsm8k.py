# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Task to evaluate LLMs on the training set of the Kaggle AIMO competition: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import lighteval.tasks.default_prompts as prompt

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)

from lighteval.metrics.metrics_sample import (
    MajAtK,
    PassAtK,
)

from lighteval.metrics.normalizations import (
    gsm8k_normalizer,
)

from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)

import numpy as np

pass_at_1_4n_gsm8k = SampleLevelMetric(
    metric_name="gsm8k_pass@1:4_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=4,
        strip_strings=True, normalize_pred=gsm8k_normalizer, normalize_gold=gsm8k_normalizer
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.MATH,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

gsm8k_custom = LightevalTaskConfig(
    name="gsm8k",
    suite=["custom"],
    prompt_function=prompt.gsm8k,
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[Metrics.quasi_exact_match_gsm8k, Metrics.maj_at_8_gsm8k, pass_at_1_4n_gsm8k],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [gsm8k_custom]