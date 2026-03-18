"""GRPO training loop for GSM8K/MATH-style reasoning."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

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
    load_math_dataset_and_format)

DEFAULT_LOSS_TYPE: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
] = "reinforce_with_baseline"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "/root/autodl-tmp/qwen-math-1.5b/Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA_PATH = str(REPO_ROOT / "data" / "math" / "train.jsonl")
DEFAULT_TEST_DATA_PATH = str(REPO_ROOT / "data" / "math" / "test.jsonl")
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
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = DEFAULT_LOSS_TYPE
    use_std_normalization: bool = True
    seed: int = 69
    eval_every: int = 5
    n_eval_examples: int = 2048
    top_p: float = 1.0
    cliprange: float = 0.2
    wandb_project: str = "cs336-grpo"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_wandb_sweep_overrides(config: GRPOConfig) -> GRPOConfig:
    config_field_names = {f.name for f in fields(GRPOConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


def sample_question_batch(
    prompts: list[str],
    ground_truths: list[str],
    n_prompts_per_rollout_batch: int,
) -> tuple[list[str], list[str]]:
    ## todo
    batch_size = min(n_prompts_per_rollout_batch, len(prompts))
    indices = random.sample(range(len(prompts)), batch_size)
    batch_prompts = [prompts[index] for index in indices]
    batch_ground_truths = [ground_truths[index] for index in indices]
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
) -> tuple[list[str], list[str], list[str]]:
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
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


def compute_old_log_probs_in_microbatches(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device_train: str,
    micro_batch_size: int,
) -> torch.Tensor:
    old_log_prob_chunks: list[torch.Tensor] = []
    batch_size = input_ids.shape[0]

    with torch.no_grad():
        for start in range(0, batch_size, micro_batch_size):
            end = start + micro_batch_size
            old_log_probs = get_response_log_probs(
                model=model,
                input_ids=input_ids[start:end].to(device_train),
                labels=labels[start:end].to(device_train),
                return_token_entropy=False,
            )["log_probs"]
            old_log_prob_chunks.append(old_log_probs.detach().cpu())

    return torch.cat(old_log_prob_chunks, dim=0)


def iter_microbatches(
    batch_tensors: dict[str, torch.Tensor],
    micro_batch_size: int,
):
    batch_size = batch_tensors["input_ids"].shape[0]
    indices = torch.randperm(batch_size)

    for start in range(0, batch_size, micro_batch_size):
        batch_indices = indices[start : start + micro_batch_size]
        microbatch = {
            key: value[batch_indices]
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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    cliprange: float,
    pad_token_id: int,
    device_train: str,
    grpo_step: int,
    num_train_steps_per_rollout: int,
) -> dict[str, float]:
    model.train()

    last_loss = 0.0
    last_clip_fraction = 0.0
    total_loss = 0.0
    n_optimizer_step = 0

    optimizer.zero_grad()

    for epoch in range(epochs_per_rollout_batch):
        microbatches = iter_microbatches(rollout_batch, micro_batch_size)
        n_microbatches = (rollout_batch["input_ids"].shape[0] + micro_batch_size - 1) // micro_batch_size
        step_loss = 0.0
        step_entropy = 0.0
        step_response_entropy = 0.0

        for microbatch_index, microbatch in enumerate(microbatches, start=1):
            input_ids = microbatch["input_ids"].to(device_train)
            labels = microbatch["labels"].to(device_train)
            response_mask = microbatch["response_mask"].to(device_train)
            advantages = microbatch["advantages"].to(device_train)
            raw_rewards = microbatch["raw_rewards"].to(device_train)
            old_log_probs = None
            if loss_type == "grpo_clip":
                old_log_probs = microbatch["old_log_probs"].to(device_train).detach()

            scored = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )

            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
            )

            response_entropy = (
                masked_mean(scored["token_entropy"], response_mask).detach().cpu().item()
                / gradient_accumulation_steps
            )
            entropy = scored["token_entropy"].mean().detach().cpu().item() / gradient_accumulation_steps
            step_loss += float(loss.detach().cpu())
            step_entropy += entropy
            step_response_entropy += response_entropy

            last_loss = float(loss.detach().cpu())
            clip_fraction_tensor = metadata.get("clip_fraction", None)
            if clip_fraction_tensor is None:
                last_clip_fraction = 0.0
            else:
                last_clip_fraction = float(clip_fraction_tensor.float().mean().detach().cpu().item())

            should_step = microbatch_index % gradient_accumulation_steps == 0
            is_last = microbatch_index == n_microbatches

            if should_step or is_last:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    .detach()
                    .cpu()
                    .item()
                )
                optimizer.step()
                optimizer.zero_grad()

                n_optimizer_step += 1
                total_loss += step_loss
                print(
                    f"[train step {n_optimizer_step}] "
                    f"overall train step {grpo_step * num_train_steps_per_rollout + n_optimizer_step} "
                    f"loss={step_loss:.6f} "
                    f"last_loss={last_loss:.6f} avg_response_entropy={step_response_entropy:.6f} "
                    f"avg_entropy={step_entropy:.6f}"
                )
                wandb.log(
                    {
                        "train/step": grpo_step * num_train_steps_per_rollout + n_optimizer_step,
                        "train/rollout_epoch": epoch + 1,
                        "train/roll_step_loss": step_loss,
                        "train/roll_step_grad_norm": grad_norm,
                        "train/roll_step_token_entropy": step_entropy,
                        "train/roll_step_response_entropy": step_response_entropy,
                        "train/roll_step_clip_fraction": last_clip_fraction,
                    }
                )
                step_loss = 0.0
                step_entropy = 0.0
                step_response_entropy = 0.0

    return {
        "loss": total_loss / max(n_optimizer_step, 1),
        "clip_fraction": last_clip_fraction,
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
) -> dict[str, Any]:
    load_policy_into_vllm_instance(model, vllm_model)
    # eval_count = min(n_eval_examples, len(test_prompts))
    # todo
    eval_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
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
        config={k: v for k, v in vars(config).items()},
    )
    config = apply_wandb_sweep_overrides(config)
    config.validate()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_math_dataset_and_format(config.train_data_path)
    test_data = load_math_dataset_and_format(config.test_data_path)
    train_prompts = [item["prompt"] for item in train_data]
    train_ground_truths = [item["expected_answer"] for item in train_data]


    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=config.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    initial_eval = evaluate_model(
        model=model,
        vllm_model=vllm_model,
        test_data=test_data,
        sampling_min_tokens=config.sampling_min_tokens,
        sampling_max_tokens=config.sampling_max_tokens,
        sampling_temperature=config.sampling_temperature,
        top_p=config.top_p,
        n_eval_examples=config.n_eval_examples,
    )
    print(f"[step 0] accuracy={initial_eval['accuracy']:.4f}")
    wandb.log(
        {
            "eval/step": 0,
            "eval/accuracy": initial_eval["accuracy"],
            "eval/format_rate": initial_eval["format_rate"],
            "eval/avg_length": initial_eval["avg_length"],
        },
        step=0,
    )

    for grpo_step in range(1, config.n_grpo_steps + 1):
        print(f"\n{'=' * 50}")
        print(f"[GRPO Step {grpo_step}/{config.n_grpo_steps}]")

        prompt_batch, ground_truth_batch = sample_question_batch(
            prompts=train_prompts,
            ground_truths=train_ground_truths,
            n_prompts_per_rollout_batch=config.n_prompts_per_rollout_batch,
        )

        load_policy_into_vllm_instance(model, vllm_model)
        repeated_prompts, rollout_responses, repeated_ground_truths = build_rollout_batch(
            vllm_model=vllm_model,
            prompts=prompt_batch,
            ground_truths=ground_truth_batch,
            group_size=config.group_size,
            sampling_min_tokens=config.sampling_min_tokens,
            sampling_max_tokens=config.sampling_max_tokens,
            sampling_temperature=config.sampling_temperature,
            top_p=config.top_p,
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
            "advantages": advantages.float(),
            "raw_rewards": raw_rewards.float(),
        }

        if config.loss_type == "grpo_clip":
            rollout_batch["old_log_probs"] = compute_old_log_probs_in_microbatches(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                device_train=config.device_train,
                micro_batch_size=config.micro_train_batch_size,
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
            cliprange=config.cliprange,
            pad_token_id=tokenizer.pad_token_id,
            device_train=config.device_train,
            grpo_step=grpo_step,
            num_train_steps_per_rollout=config.num_train_steps_per_rollout,
        )

        checkpoint_dir = output_dir / f"{config.wandb_run_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        print(
            f"[train_step {grpo_step}] "
            f"total_rewards={reward_metadata['total_rewards']:.4f} "
            f"format_rewards={reward_metadata['format_rewards']:.4f} "
            f"answer_rewards={reward_metadata['answer_rewards']:.4f} "
            f"loss={train_metrics['loss']:.6f} "
            f"clip_fraction={train_metrics['clip_fraction']:.4f}"
        )
        wandb.log(
            {
                "train/grop_step": grpo_step,
                "train/total_rewards": reward_metadata["total_rewards"],
                "train/format_rewards": reward_metadata["format_rewards"],
                "train/answer_rewards": reward_metadata["answer_rewards"],
                "train/loss": train_metrics["loss"],
                "train/clip_fraction": train_metrics["clip_fraction"],
            },
            step=grpo_step,
        )

        if grpo_step % config.eval_every == 0:
            eval_result = evaluate_model(
                model=model,
                vllm_model=vllm_model,
                test_data=test_data,
                sampling_min_tokens=config.sampling_min_tokens,
                sampling_max_tokens=config.sampling_max_tokens,
                sampling_temperature=config.sampling_temperature,
                top_p=config.top_p,
                n_eval_examples=config.n_eval_examples,
            )
            print(
                f"[eval {grpo_step}] "
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

    wandb.finish()


def main(
    train_data_path: str = typer.Option(DEFAULT_TRAIN_DATA_PATH, help="Path to GRPO training data."),
    test_data_path: str = typer.Option(DEFAULT_TEST_DATA_PATH, help="Path to validation data."),
    model_path: str = typer.Option(DEFAULT_MODEL_PATH, help="Policy checkpoint or HF model path."),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory for step checkpoints."),
    device_train: str = typer.Option("cuda:0", help="Device used for policy training."),
    device_vllm: str = typer.Option("cuda:1", help="Device used for vLLM rollouts/eval."),
    n_grpo_steps: int = typer.Option(200, help="Number of rollout/update iterations."),
    learning_rate: float = typer.Option(1e-5, help="AdamW learning rate."),
    advantage_eps: float = typer.Option(1e-6, help="Epsilon used in reward normalization."),
    rollout_batch_size: int = typer.Option(256, help="Number of sampled completions per rollout batch."),
    group_size: int = typer.Option(8, help="Number of completions sampled per prompt."),
    sampling_temperature: float = typer.Option(1.0, help="Sampling temperature for rollouts/eval."),
    sampling_min_tokens: int = typer.Option(4, help="Minimum number of generated tokens."),
    sampling_max_tokens: int = typer.Option(1024, help="Maximum number of generated tokens."),
    epochs_per_rollout_batch: int = typer.Option(1, help="Gradient epochs per rollout batch."),
    train_batch_size: int = typer.Option(256, help="Number of rollout samples consumed per optimizer step."),
    gradient_accumulation_steps: int = typer.Option(128, help="Number of accumulation steps per optimizer step."),
    gpu_memory_utilization: float = typer.Option(0.85, help="vLLM GPU memory utilization target."),
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = typer.Option(
        DEFAULT_LOSS_TYPE,
        help="Policy gradient loss to optimize.",
    ),
    use_std_normalization: bool = typer.Option(True, help="Normalize group advantages by group std."),
    seed: int = typer.Option(69, help="Random seed."),
    eval_every: int = typer.Option(5, help="Run validation every N training steps."),
    n_eval_examples: int = typer.Option(1024, help="Number of validation examples to evaluate."),
    top_p: float = typer.Option(1.0, help="Top-p used for rollouts/eval."),
    cliprange: float = typer.Option(0.2, help="Clip range used only for grpo_clip."),
    wandb_project: str = typer.Option("cs336-grpo", help="wandb project name."),
    wandb_run_name: Optional[str] = typer.Option(None, help="wandb run name."),
    wandb_mode: str = typer.Option("online", help="wandb mode."),
) -> None:
    config = GRPOConfig(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        model_path=model_path,
        output_dir=output_dir,
        device_train=device_train,
        device_vllm=device_vllm,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        seed=seed,
        eval_every=eval_every,
        n_eval_examples=n_eval_examples,
        top_p=top_p,
        cliprange=cliprange,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )
    run_grpo(config)


if __name__ == "__main__":
    typer.run(main)