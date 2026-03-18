#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

python autodl-tmp/cs336-assignment-5/cs336_alignment/sft.py --wandb_run_name sft_qwen_math_1.5b-0318 > logs/sft_qwen_math_1.5b-0318.log 2>&1

python autodl-tmp/cs336-assignment-5/cs336_alignment/ei.py --wandb_run_name ei_qwen_math_1.5b-0318 > logs/ei_qwen_math_1.5b-0318.log 2>&1

python autodl-tmp/cs336-assignment-5/cs336_alignment/grpo.py --wandb_run_name grpo_qwen_math_1.5b-0318 > logs/grpo_qwen_math_1.5b-0318.log 2>&1


cp -r /root/autodl-tmp/qwen-math-1.5b/Qwen/Qwen2.5-Math-1.5B/* ./models/Qwen2.5-Math-1.5B/