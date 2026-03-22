import json
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple, Dict, Optional

from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams


# Load r1_zero prompt template
PROMPT_DIR = Path(__file__).parent / "prompts"
with open(PROMPT_DIR / "r1_zero.prompt", "r") as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read().strip()


def load_math_dataset(data_path: str) -> list[dict[str, Any]]:
    """
    Load MATH dataset from JSONL file.

    Args:
        data_path: Path to the MATH validation JSONL file

    Returns:
        List of MATH examples
    """
    # examples = []
    # with open(data_path, "r") as f:
    #     for line in f:
    #         examples.append(json.loads(line))
    # return examples
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def format_r1_zero_prompt(problem: str) -> str:
    """
    Format a MATH problem with R1 zero-shot prompting.

    Uses the prompt template from cs336_alignment/prompts/r1_zero.prompt.

    Args:
        problem: The MATH problem text

    Returns:
        Formatted prompt string
    """
    return R1_ZERO_PROMPT_TEMPLATE.format(question=problem)


def load_math_dataset_and_format(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r") as f:
        examples = json.load(f)

    formatted_examples = []
    for sample in examples:
        sample['prompt'] = format_r1_zero_prompt(sample['problem'])
        formatted_examples.append(sample)
    return formatted_examples


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # 批量分词。若设置了 max_seq_len，先在 tokenizer 侧做基础截断，避免极端长样本。
    use_truncation = max_seq_len is not None
    prompt_enc = tokenizer(
        prompt_strs,
        padding=False,
        truncation=use_truncation,
        max_length=max_seq_len,
    )
    output_enc = tokenizer(
        output_strs,
        padding=False,
        truncation=use_truncation,
        max_length=max_seq_len,
    )

    # 将每个样本的 token IDs 转为 1D 张量
    prompt_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompt_enc["input_ids"]]
    output_ids = [torch.tensor(ids, dtype=torch.long) for ids in output_enc["input_ids"]]

    # 计算最大总长度（prompt + output）
    max_len = 0
    for p, o in zip(prompt_ids, output_ids):
        total_len = len(p) + len(o)
        if total_len > max_len:
            max_len = total_len

    input_ids_list = []
    labels_list = []
    response_mask_list = []

    for p, o in zip(prompt_ids, output_ids):
        # 拼接 prompt 和 output
        token_ids = torch.cat([p, o])                # (total_len,)

        # 构建 response mask：输出部分为 1，其余为 0
        response_mask = torch.cat([torch.zeros_like(p, dtype=torch.bool), 
                                   torch.ones_like(o, dtype=torch.bool)], dim=0)
        
        # 填充到 max_len
        pad_len = max_len - token_ids.size(0)
        if pad_len > 0:
            token_ids = torch.nn.functional.pad(token_ids, (0, pad_len), value=tokenizer.pad_token_id)
            response_mask = torch.nn.functional.pad(response_mask, (0, pad_len), value=False)

        # 生成 input_ids（去掉最后一个 token）和 labels（去掉第一个 token）
        input_ids = token_ids[:-1]      # (max_len-1,)
        labels = token_ids[1:]           # (max_len-1,)
        response_mask_sliced = response_mask[1:]  # (max_len-1,)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        response_mask_list.append(response_mask_sliced)

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),        # (batch_size, max_len-1)
        "labels": torch.stack(labels_list, dim=0),              # (batch_size, max_len-1)
        "response_mask": torch.stack(response_mask_list, dim=0), # (batch_size, max_len-1)
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # logits shape: (batch_size, seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1) # batch_size, seq_len

    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
    only_return_mean_response_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions.
    """

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Get log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)

    # Gather log_probs at label positions
    # labels shape: (batch_size, seq_len)
    # We need to gather along the vocab dimension
    label_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1) # (batch_size, seq_len)

    result = {"log_probs": label_log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # Apply mask to tensor
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor)) # (batch_size, seq_len)

    if dim is None:
        # Sum over all dimensions
        return torch.sum(masked_tensor) / normalize_constant
    else:
        return torch.sum(masked_tensor, dim=dim) / normalize_constant


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    vllm_set_random_seed(seed)
    # single gpu setup: patch get_world_size to return 1, and patch the profiling function to avoid the memory footprint assertion error
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,

            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm:LLM):
    policy = _unwrap_policy_model(policy)
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def _unwrap_policy_model(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the original model and prefixes state_dict keys with _orig_mod.
    # vLLM expects the original key names, so always sync/save with the unwrapped module.
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    return_output_results: bool = False,
    output_path: str = None,

) -> dict[str, Any]:
    """
    评估语言模型在一组提示词上的性能，计算评估指标并将结果序列化到磁盘。

    Args:
        vllm_model: vLLM 模型实例
        reward_fn: 奖励函数，用于评估模型输出
        prompts: 输入提示词列表
        eval_sampling_params: 采样参数
        ground_truths: 真实答案列表
        output_path: 输出路径，可选
    Returns:
        包含评估指标的字典
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    correct = 0
    format_correct = 0
    correct_length = 0
    incorrect_length = 0

    if return_output_results:
        results = []

    for i, (output, ground_truth) in enumerate(zip(outputs, ground_truths)):
        out_text = output.outputs[0].text if output.outputs else ""
        gt = str(ground_truth).strip() if ground_truth is not None else ""
        reward = reward_fn(out_text, gt, fast=True)

        is_correct = reward.get("reward", 0.0) >= 0.5
        has_format = reward.get("format_reward", 0.0) >= 0.5

        if is_correct:
            correct += 1
            correct_length += len(out_text)
        else:
            incorrect_length += len(out_text)
        if has_format:
            format_correct += 1

        if return_output_results:
            result = {
                "prompt": prompts[i],
                "model_output": out_text,
                "ground_truth": gt,
                "format_reward": reward.get("format_reward", 0.0),
                "answer_reward": reward.get("answer_reward", 0.0),
                "reward": reward.get("reward", 0.0),
                "response_length": len(out_text),
            }
            results.append(result)

    total = len(prompts)
    accuracy = correct / total if total > 0 else 0
    format_rate = format_correct / total if total > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "format_rate": format_rate,
        "format_correct": format_correct,
        "total": total,
        "avg_length": (correct_length + incorrect_length) / total if total > 0 else 0,
        "avg_correct_length": correct_length/max(correct, 1),
        "avg_incorrect_length": incorrect_length/max(total - correct, 1)
    }

    # Write results to disk if output_path provided
    if output_path is not None:
        with open(output_path, "w") as fout:
            if return_output_results:
                for result in results:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.write(json.dumps({"metrics": metrics}, ensure_ascii=False) + "\n")

    if return_output_results:
        return metrics, results
    else:
        return metrics


def get_eval_example_count(
    grpo_step: int,
    n_grpo_steps: int,
    final_n_eval_examples: int,
    first_n_eval_examples: int = 1024
) -> int:
    # Use a smaller eval set in early/mid training to save time, then switch
    # to the full configured eval size in the final 20% of training.
    warmup_eval_examples = min(first_n_eval_examples, final_n_eval_examples)
    ramp_start_step = max(1, int(0.9 * n_grpo_steps))
    if grpo_step < ramp_start_step:
        return warmup_eval_examples
    return final_n_eval_examples


# ----------------------------- Logging Utility ---------------------------
from datetime import datetime
import os
import time
class Log:
    def __init__(self, log_path, time_key=True):
        self.path = log_path
        if time_key:
            self.path = self.path.replace(
                ".", "{}.".format(time.strftime("_%Y%m%d%H%M%S", time.localtime(time.time())))
            )
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            file=open(self.path, "a+"),
        )
        print("log path:", self.path)
        print("****************开始记录*********************", file=open(self.path, "a+"))

    def __call__(self, *content):
        t1 = time.strftime("%H:%M:%S", time.localtime(time.time()))
        print(*content)
        print(t1, content, file=open(self.path, "a+"))

    def clean(self):
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            file=open(self.path, "w"),
        )
        print("****************开始记录*********************", file=open(self.path, "a+"))


def init_log_and_output_dir(output_dir, model_name):
    now = datetime.now()
    current_time_name = now.strftime("%m%d-%H%M%S") + "-" + model_name

    if not os.path.exists(f"{output_dir}/{current_time_name}"):
        os.makedirs(f"{output_dir}/{current_time_name}")
    output_dir = f"{output_dir}/{current_time_name}"
    log = Log(f"{output_dir}/logs.txt")
    return log, output_dir
