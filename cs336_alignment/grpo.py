"""GRPO training loop for GSM8K/MATH-style reasoning."""

from __future__ import annotations

import random
import numpy as np
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, List, Literal, Optional

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from cs336_alignment.config_utils import (
    apply_cli_overrides_to_dataclass,
    load_dataclass_config_from_yaml,
)

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_helper import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean
)
from cs336_alignment.utils import (
    init_vllm, 
    tokenize_prompt_and_output, 
    get_response_log_probs,
    evaluate_vllm, 
    load_policy_into_vllm_instance,
    load_math_dataset_and_format,
    init_log_and_output_dir,
    get_eval_example_count)


import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9"


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "/root/autodl-tmp/qwen-math-1.5b/Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA_PATH = str(REPO_ROOT / "data" / "math" / "train.jsonl")
DEFAULT_TEST_DATA_PATH = str(REPO_ROOT / "data" / "math" / "val.jsonl")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "logs" / "grpo_checkpoints")


@dataclass(slots=True)
class GRPOConfig:
    train_data_path: str = DEFAULT_TRAIN_DATA_PATH
    test_data_path: str = DEFAULT_TEST_DATA_PATH
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    device_train: str = "cuda:0"
    device_vllm: str = "cuda:1"
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    micro_old_log_prob_batch_size: int = 4
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip"
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    seed: int = 69
    eval_every: int = 5
    n_eval_examples: int = 2048
    n_first_eval_examples: int = 1024
    top_p: float = 1.0
    cliprange: float = 0.2
    wandb_project: str = "cs336-grpo"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    use_torch_compile: bool = True
    norm_type: Literal["constant", "mean", "normalize"] = "mean"
    norm_constant: float = 1.0

    @property
    def micro_train_batch_size(self) -> int:
        return self.train_batch_size // self.gradient_accumulation_steps

    @property
    def n_prompts_per_rollout_batch(self) -> int:
        return self.rollout_batch_size // self.group_size

    @property
    def n_microbatches_per_rollout_batch(self) -> int:
        return self.rollout_batch_size // self.micro_train_batch_size
    
    @property
    def num_train_steps_per_rollout(self) -> int:
        return self.rollout_batch_size // self.train_batch_size * self.epochs_per_rollout_batch

    def validate(self) -> None:
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        assert self.micro_train_batch_size > 0, "micro_train_batch_size must be positive"
        assert self.rollout_batch_size % self.micro_train_batch_size == 0, (
            "rollout_batch_size must be divisible by micro_train_batch_size"
        )
        if self.loss_type == "grpo_clip" and self.epochs_per_rollout_batch == 1:
            typer.echo(
                "Warning: grpo_clip is typically most useful in the off-policy setting with multiple epochs."
            )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_wandb_sweep_overrides(config: GRPOConfig) -> GRPOConfig:
    config_field_names = {f.name for f in fields(GRPOConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


def load_config_from_yaml(config_path: str) -> GRPOConfig:
    return load_dataclass_config_from_yaml(config_path, GRPOConfig)


def sample_question_batch(
    data: list[dict[str, str]],
    n_prompts_per_rollout_batch: int,
) -> tuple[list[str], list[str]]:
    ## todo
    batch_size = min(n_prompts_per_rollout_batch, len(data))
    indices = random.sample(range(len(data)), batch_size)
    batch_prompts = [data[index]["prompt"] for index in indices]
    batch_ground_truths = [data[index]["expected_answer"] for index in indices]
    return batch_prompts, batch_ground_truths


def build_rollout_batch(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    group_size: int,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    sampling_temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    outputs = vllm_model.generate(prompts, sampling_params)

    repeated_prompts: list[str] = []
    rollout_responses: list[str] = []
    repeated_ground_truths: list[str] = []

    for prompt, output_group, ground_truth in zip(prompts, outputs, ground_truths):
        for completion in output_group.outputs:
            repeated_prompts.append(prompt)
            rollout_responses.append(completion.text.strip())
            repeated_ground_truths.append(ground_truth)
    return repeated_prompts, rollout_responses, repeated_ground_truths


def compute_old_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device_train: str,
    micro_batch_size: int,
) -> torch.Tensor:
    old_log_prob_chunks: list[torch.Tensor] = []
    batch_size = input_ids.shape[0]
    model.eval()
    with torch.no_grad():
        for start in range(0, batch_size, micro_batch_size):
            end = start + micro_batch_size
            old_log_probs = get_response_log_probs(
                model=model,
                input_ids=input_ids[start:end].to(device_train),
                labels=labels[start:end].to(device_train),
                return_token_entropy=False,
            )["log_probs"]
            old_log_prob_chunks.append(old_log_probs) # todo:没有移出gpu
    torch.cuda.empty_cache()
    model.train()
    return torch.cat(old_log_prob_chunks, dim=0)


def iter_microbatches(
    batch_tensors: dict[str, torch.Tensor],
    micro_batch_size: int,
):
    rollout_batch_size = batch_tensors["input_ids"].shape[0]
    # indices = torch.randperm(rollout_batch_size)

    # for start in range(0, batch_size, micro_batch_size):
    #     batch_indices = indices[start : start + micro_batch_size]
    #     microbatch = {
    #         key: value[batch_indices]
    #         for key, value in batch_tensors.items()
    #     }
    #     yield microbatch
    for start in range(0, rollout_batch_size, micro_batch_size):
        microbatch = {
            key: value[start : start + micro_batch_size]
            for key, value in batch_tensors.items()
        }
        yield microbatch


def train_on_rollout_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_batch: dict[str, torch.Tensor],
    epochs_per_rollout_batch: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    cliprange: float,
    pad_token_id: int,
    device_train: str,
    grpo_step: int,
    num_train_steps_per_rollout: int,
    norm_type: Literal["constant", "mean", "normalize"],
    norm_constant: float,
    log,
) -> dict[str, float]:
    model.train()

    last_clip_fraction = 0.0
    total_loss = 0.0
    total_entropy = 0.0
    train_step = 0

    optimizer.zero_grad()

    for epoch in range(epochs_per_rollout_batch):
        microbatches = iter_microbatches(rollout_batch, micro_batch_size)
        n_microbatches = (rollout_batch["input_ids"].shape[0] + micro_batch_size - 1) // micro_batch_size
        step_loss = 0.0
        step_response_entropy = 0.0

        for micro_ind, microbatch in enumerate(microbatches, start=1):
            input_ids = microbatch["input_ids"].to(device_train)
            labels = microbatch["labels"].to(device_train)
            response_mask = microbatch["response_mask"].to(device_train)
            advantages = microbatch["advantages"].to(device_train)
            raw_rewards = microbatch["raw_rewards"].to(device_train)
            old_log_probs = None
            if loss_type == "grpo_clip":
                old_log_probs = microbatch["old_log_probs"] # already on gpu

            with torch.autocast(device_type=device_train, dtype=torch.bfloat16):
                scored = get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
            if "token_entropy" in scored:
                scored["token_entropy"] = masked_mean(scored["token_entropy"], response_mask, dim=None).detach().cpu()
                response_entropy = scored["token_entropy"]/ gradient_accumulation_steps
                step_response_entropy += float(response_entropy)

            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                norm_type=norm_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
                norm_constant=norm_constant
            )
            step_loss += float(loss.detach().cpu())

            clip_fraction_tensor = metadata.get("clip_fraction", None)
            if clip_fraction_tensor is None:
                last_clip_fraction = 0.0
            else:
                last_clip_fraction = float(clip_fraction_tensor.float().mean().detach().cpu().item())

            if micro_ind % gradient_accumulation_steps == 0 or micro_ind == n_microbatches:
                train_step += 1
                total_loss += step_loss
                total_entropy += step_response_entropy

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    .detach()
                    .cpu()
                    .item()
                )
                optimizer.step()
                optimizer.zero_grad()

                log(
                    f"rollout grpo_step {grpo_step} "
                    f"epoch {epoch + 1} "
                    f"step {train_step}] "
                    f"loss={step_loss:.6f} "
                    f"avg_response_entropy={step_response_entropy:.6f} "
                    f"last_clip_fraction={last_clip_fraction:.4f} "
                    f"grad_norm={grad_norm:.4f}"
                )
                wandb.log(
                    {
                        "rollout/step": grpo_step * num_train_steps_per_rollout + train_step,
                        "rollout/epoch": epoch + 1,
                        "rollout/grpostep": grpo_step,
                        "rollout/step_loss": step_loss,
                        "rollout/step_grad_norm": grad_norm,
                        "rollout/step_response_entropy": step_response_entropy,
                        "rollout/step_clip_fraction": last_clip_fraction,
                    }
                )
                step_loss = 0.0
                step_response_entropy = 0.0

    return {
        "loss": total_loss / max(train_step, 1),
        "clip_fraction": last_clip_fraction,
        "entropy": total_entropy / max(train_step, 1),
    }


def evaluate_model(
    model: torch.nn.Module,
    vllm_model: LLM,
    test_data: list[dict[str, Any]],
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    sampling_temperature: float,
    top_p: float,
    n_eval_examples: int,
    seed: int,
) -> dict[str, Any]:
    load_policy_into_vllm_instance(model, vllm_model)
    # eval_count = min(n_eval_examples, len(test_prompts))
    # todo
    test_data = random.sample(test_data, min(n_eval_examples, len(test_data)))
    eval_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        seed=seed,
        include_stop_str_in_output=True,
    )
    return evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=[item["prompt"] for item in test_data],
        ground_truths=[item["expected_answer"] for item in test_data],
        eval_sampling_params=eval_sampling_params,
    )


def run_grpo(config: GRPOConfig) -> None:
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=asdict(config),
    )
    config = apply_wandb_sweep_overrides(config)
    config.validate()
    set_seed(config.seed)

    run_name = config.wandb_run_name or "default"
    log, output_path = init_log_and_output_dir(config.output_dir, run_name)

    log(config)
    output_path = Path(output_path) # 存log和模型
    output_path.mkdir(parents=True, exist_ok=True)
    log(f"Artifacts/logs will be saved under: {output_path}")

    train_data = load_math_dataset_and_format(config.train_data_path)
    test_data = load_math_dataset_and_format(config.test_data_path)
    log(f"Number of training examples: {len(train_data)}")
    log(f"Number of test examples: {len(test_data)}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=config.device_train,
    )
    if config.use_torch_compile:
        model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    vllm_model = init_vllm(
        model_id=config.model_path,
        device=config.device_vllm,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    for grpo_step in range(1, config.n_grpo_steps + 1):

        # sample a batch of prompts for this rollout
        batch_size = min(config.n_prompts_per_rollout_batch, len(train_data))
        indices = random.sample(range(len(train_data)), batch_size)
        batch_prompts = [train_data[index]["prompt"] for index in indices]
        batch_ground_truths = [train_data[index]["expected_answer"] for index in indices]

        # compute rewards and advantages for the batch, and tokenize the rollout outputs for training
        load_policy_into_vllm_instance(model, vllm_model)
        repeated_prompts, rollout_responses, repeated_ground_truths = build_rollout_batch(
            vllm_model=vllm_model,
            prompts=batch_prompts,
            ground_truths=batch_ground_truths,
            group_size=config.group_size,
            sampling_min_tokens=config.sampling_min_tokens,
            sampling_max_tokens=config.sampling_max_tokens,
            sampling_temperature=config.sampling_temperature,
            top_p=config.top_p,
            seed=config.seed,
        )

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        rollout_batch = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "response_mask": tokenized["response_mask"],
            "advantages": advantages.float().unsqueeze(1),
            "raw_rewards": raw_rewards.float().unsqueeze(1),
        }

        if config.loss_type == "grpo_clip":
            rollout_batch["old_log_probs"] = compute_old_log_probs(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                device_train=config.device_train,
                micro_batch_size=config.micro_old_log_prob_batch_size,
            )
            assert rollout_batch["input_ids"].shape[0] == config.rollout_batch_size, (
                "rollout batch tensor size must equal rollout_batch_size"
            )

        train_metrics = train_on_rollout_batch(
            model=model,
            optimizer=optimizer,
            rollout_batch=rollout_batch,
            epochs_per_rollout_batch=config.epochs_per_rollout_batch,
            micro_batch_size=config.micro_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            loss_type=config.loss_type,
            norm_type=config.norm_type,
            cliprange=config.cliprange,
            pad_token_id=tokenizer.pad_token_id,
            device_train=config.device_train,
            grpo_step=grpo_step,
            num_train_steps_per_rollout=config.num_train_steps_per_rollout,
            norm_constant=config.norm_constant,
            log=log,
        )

        log(
            f"[grpo step {grpo_step}] "
            f"total_rewards={reward_metadata['total_rewards']:.4f} "
            f"format_rewards={reward_metadata['format_rewards']:.4f} "
            f"answer_rewards={reward_metadata['answer_rewards']:.4f} "
            f"loss={train_metrics['loss']:.6f} "
            f"clip_fraction={train_metrics['clip_fraction']:.4f} "
            f"entropy={train_metrics['entropy']:.4f}"
        )
        wandb.log(
            {
                "train/grpo_step": grpo_step,
                "train/total_rewards": reward_metadata["total_rewards"],
                "train/format_rewards": reward_metadata["format_rewards"],
                "train/answer_rewards": reward_metadata["answer_rewards"],
                "train/loss": train_metrics["loss"],
                "train/clip_fraction": train_metrics["clip_fraction"],
                "train/entropy": train_metrics["entropy"],
            },
            step=grpo_step,
        )

        if grpo_step % config.eval_every == 0 or grpo_step == config.n_grpo_steps:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eval_example_count = get_eval_example_count(
                grpo_step=grpo_step,
                n_grpo_steps=config.n_grpo_steps,
                final_n_eval_examples=config.n_eval_examples,
                first_n_eval_examples=config.n_first_eval_examples,
            )
            eval_result = evaluate_model(
                model=model,
                vllm_model=vllm_model,
                test_data=test_data,
                sampling_min_tokens=config.sampling_min_tokens,
                sampling_max_tokens=config.sampling_max_tokens,
                sampling_temperature=config.sampling_temperature,
                top_p=config.top_p,
                n_eval_examples=eval_example_count,
                seed=config.seed,
            )
            log(
                f"[===eval step {grpo_step}] "
                f"n_eval={eval_example_count} "
                f"accuracy={eval_result['accuracy']:.4f} "
                f"format_rate={eval_result['format_rate']:.4f}"
                f"avg_length={eval_result['avg_length']:.2f} "
            )
            wandb.log(
                {
                    "eval/step": grpo_step,
                    "eval/accuracy": eval_result["accuracy"],
                    "eval/format_rate": eval_result["format_rate"],
                    "eval/avg_length": eval_result["avg_length"],
                    "eval/avg_correct_length": eval_result["avg_correct_length"],
                    "eval/avg_incorrect_length": eval_result["avg_incorrect_length"],
                },
                step=grpo_step,
            )

            # checkpoint_dir = output_path / f"f{run_name}"
            # checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # model.save_pretrained(checkpoint_dir)
            # tokenizer.save_pretrained(checkpoint_dir)

    wandb.finish()


def main(
    config_path: str = typer.Option(
        "configs/grpo.yaml",
        help="Path to GRPO YAML config file.",
    ),
    wandb_project: Optional[str] = typer.Option("cs336-grpo", help="wandb project name."),
    wandb_run_name: Optional[str] = typer.Option(None, help="Optional override for wandb run name."),
    wandb_mode: Optional[str] = typer.Option("online", help="Optional override for wandb mode."),
    set_values: List[str] = typer.Option(
        [],
        "--set",
        help="Override config values from CLI, e.g. --set learning_rate=3e-6 --set n_grpo_steps=50",
    ),
) -> None:
    config = load_config_from_yaml(config_path)
    config = apply_cli_overrides_to_dataclass(config, set_values)
    if wandb_project is not None:
        config.wandb_project = wandb_project
    if wandb_run_name is not None:
        config.wandb_run_name = wandb_run_name
    if wandb_mode is not None:
        config.wandb_mode = wandb_mode
    run_grpo(config)


if __name__ == "__main__":
    typer.run(main)