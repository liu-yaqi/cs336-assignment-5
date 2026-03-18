"""
Baseline evaluation utilities for MATH dataset.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import format_r1_zero_prompt, evaluate_vllm, load_math_dataset


QWEN_MATH_BASE_PATH = Path(__file__).parent / "Qwen2.5-Math-1.5B"
DATA_PATH = Path(__file__).parent / "data/math/val.jsonl"
OUTPUT_PATH = Path(__file__).parent / "eval_baseline_results.jsonl"


def load_and_format_prompts(data_path: str) -> Tuple[List[str], List[str]]:
    """
    Load MATH dataset and format prompts for evaluation.

    Args:
        data_path: Path to the MATH validation JSONL file
    Returns:
        Tuple of (formatted prompts, ground truth answers)
    """
    examples = load_math_dataset(data_path)
    prompts = [format_r1_zero_prompt(ex["problem"]) for ex in examples]
    ground_truths = [ex["expected_answer"] for ex in examples]
    return prompts, ground_truths


def main():
    # Load and format prompts
    prompts, ground_truths = load_and_format_prompts(DATA_PATH)

    # Build model and sampling params
    vllm_model = LLM(QWEN_MATH_BASE_PATH)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Evaluate model
    metrics = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=ground_truths,
        output_path=OUTPUT_PATH
    )
    print(metrics)


if __name__ == "__main__":
    main()