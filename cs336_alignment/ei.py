"""Expert Iteration training loop with Typer + Weights & Biases support."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import torch
import typer
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import sft_microbatch_train_step
from cs336_alignment.utils import (
    evaluate_vllm,
    get_response_log_probs,
    init_vllm,
    load_math_dataset_and_format,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
    init_log_and_output_dir,
    _unwrap_policy_model,
    set_seed
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "/root/autodl-tmp/qwen-math-1.5b/Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA_PATH = str(REPO_ROOT / "data" / "math" / "train.jsonl")
DEFAULT_TEST_DATA_PATH = str(REPO_ROOT / "data" / "math" / "val.jsonl")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "logs" / "ei_checkpoints")


@dataclass(slots=True)
class EIConfig:
    train_data_path: str = DEFAULT_TRAIN_DATA_PATH
    test_data_path: str = DEFAULT_TEST_DATA_PATH
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    device_train: str = "cuda:0"
    device_vllm: str = "cuda:1"
    seed: int = 69
    n_ei_steps: int = 5
    rollout_batch_size: int = 256
    n_rollouts_per_prompt: int = 8
    sft_epochs: int = 2
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    eval_every_ei_steps: int = 1
    num_eval_examples: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 1024
    gpu_memory_utilization: float = 0.9
    wandb_project: str = "cs336-ei"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

    @property
    def n_prompts_per_rollout_batch(self) -> int:
        return self.rollout_batch_size // self.n_rollouts_per_prompt

    def validate(self) -> None:
        assert self.rollout_batch_size > 0, "rollout_batch_size must be positive"
        assert self.n_rollouts_per_prompt > 0, "n_rollouts_per_prompt must be positive"
        assert self.rollout_batch_size % self.n_rollouts_per_prompt == 0, (
            "rollout_batch_size must be divisible by n_rollouts_per_prompt"
        )
        assert self.n_prompts_per_rollout_batch > 0, "n_prompts_per_rollout_batch must be positive"
        assert self.micro_batch_size > 0, "micro_batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.n_ei_steps > 0, "n_ei_steps must be positive"
        assert self.eval_every_ei_steps > 0, "eval_every_ei_steps must be positive"


def apply_wandb_sweep_overrides(config: EIConfig) -> EIConfig:
    config_field_names = {f.name for f in fields(EIConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


class PromptResponseDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        sample = self.samples[idx]
        return sample["prompt"], sample["response"]


def build_filtered_dataloader(
    filtered_samples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> DataLoader:
    dataset = PromptResponseDataset(filtered_samples)

    def collate_fn(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        prompts, responses = zip(*batch)
        return tokenize_prompt_and_output(
            prompt_strs=list(prompts),
            output_strs=list(responses),
            tokenizer=tokenizer,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def cycle_dataloader(dataloader: DataLoader) -> Any:
    while True:
        for batch in dataloader:
            yield batch


def rollout_and_filter(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    config: EIConfig,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    sampling_params = SamplingParams(
        n=config.n_rollouts_per_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = vllm_model.generate(prompts, sampling_params)

    filtered_samples: list[dict[str, str]] = []
    total_rollouts = 0
    correct_rollouts = 0

    for prompt, output_group, gt in zip(prompts, outputs, ground_truths):
        for completion in output_group.outputs:
            total_rollouts += 1
            response = completion.text.strip()
            reward_result = r1_zero_reward_fn(response, gt, fast=True)
            if reward_result.get("reward", 0.0) >= 1.0:
                correct_rollouts += 1
                filtered_samples.append({"prompt": prompt, "response": response})

    filter_rate = correct_rollouts / max(total_rollouts, 1)
    rollout_stats = {
        "total_rollouts": float(total_rollouts),
        "correct_rollouts": float(correct_rollouts),
        "filter_rate": filter_rate,
    }
    return filtered_samples, rollout_stats


def sft_on_filtered_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    filtered_samples: list[dict[str, str]],
    config: EIConfig,
    log: callable,
    ei_step: int,
    optimize_step: int = 0,
) -> dict[str, float]:
    dataloader = build_filtered_dataloader(
        filtered_samples=filtered_samples,
        tokenizer=tokenizer,
        batch_size=config.micro_batch_size,
    )
    iterator = cycle_dataloader(dataloader)
    optimizer.zero_grad()
    model.train()

    denom = config.micro_batch_size * config.gradient_accumulation_steps
    # todo: 改成epoch训练，目前是按照step训练，会有点点问题
    n_sft_steps = max(1, math.ceil((len(filtered_samples) * config.sft_epochs) / denom))

    last_loss = 0.0
    total_loss = 0.0
    for step in range(1, n_sft_steps + 1):
        step_loss = 0.0
        step_entropy = 0.0
        step_response_entropy = 0.0

        for _ in range(config.gradient_accumulation_steps):
            batch = next(iterator)
            input_ids = batch["input_ids"].to(config.device_train)
            labels = batch["labels"].to(config.device_train)
            response_mask = batch["response_mask"].to(config.device_train)

            # !!解决cudaoutofmemory If sequence length is too long, split the micro-batch into two halves to lower peak memory.
            split_chunks = [(input_ids, labels, response_mask)]
            if input_ids.shape[1] > 1024 and input_ids.shape[0] > 1:
                half = input_ids.shape[0] // 2
                split_chunks = [
                    (input_ids[:half], labels[:half], response_mask[:half]),
                    (input_ids[half:], labels[half:], response_mask[half:]),
                ]

            split_factor = len(split_chunks)
            grad_scale = config.gradient_accumulation_steps * split_factor
            for chunk_input_ids, chunk_labels, chunk_response_mask in split_chunks:
                with torch.autocast(device_type=config.device_train, dtype=torch.bfloat16):
                    scored = get_response_log_probs(
                        model=model,
                        input_ids=chunk_input_ids,
                        labels=chunk_labels,
                        return_token_entropy=True,
                    )
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=scored["log_probs"],
                    response_mask=chunk_response_mask,
                    gradient_accumulation_steps=grad_scale,
                    normalize_constant=1.0,
                )
                response_entropy = (
                    scored["token_entropy"][chunk_response_mask].mean().detach().cpu().item()
                    / grad_scale
                )
                entropy = (
                    scored["token_entropy"].mean().detach().cpu().item()
                    / grad_scale
                )
                step_loss += float(loss.detach().cpu())
                step_entropy += entropy
                step_response_entropy += response_entropy

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        optimize_step += 1
        last_loss = step_loss
        total_loss += step_loss

        log(
            f"[ei {ei_step} sft {step} step {optimize_step}] loss={step_loss:.6f} "
            f"avg_response_entropy={step_response_entropy:.6f} avg_entropy={step_entropy:.6f}"
        )
        wandb.log(
            {
                "sft/ei_step": ei_step,
                "sft/step": optimize_step,
                "sft/step_loss": step_loss,
                "sft/step_avg_entropy": step_entropy,
                "sft/step_avg_response_entropy": step_response_entropy,
            }
        )

    return {
        "sft_last_loss": last_loss,
        "sft_steps": float(n_sft_steps),
        "sft_total_loss": total_loss // max(1, n_sft_steps),
    }, optimize_step


def evaluate_model(
    vllm_model: LLM,
    test_data: list[dict[str, Any]],
    config: EIConfig,
) -> dict[str, Any]:
    test_data = random.sample(test_data, min(config.num_eval_examples, len(test_data)))
    eval_sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
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


def run_expert_iteration(config: EIConfig) -> None:
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=asdict(config),
    )
    try:
        config = apply_wandb_sweep_overrides(config)
        config.validate()
        set_seed(config.seed)

        run_name = config.wandb_run_name or "default"
        log, output_path = init_log_and_output_dir(config.output_dir, config.wandb_run_name)
        
        log(config)
        output_path = Path(output_path) # 存log和模型
        output_path.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=config.device_train,
        )
        model = torch.compile(model)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        vllm_model = init_vllm(
            model_id=config.model_path,
            device=config.device_vllm,
            seed=config.seed,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )

        train_data = load_math_dataset_and_format(config.train_data_path)
        test_data = load_math_dataset_and_format(config.test_data_path)

        load_policy_into_vllm_instance(model, vllm_model)
        initial_metrics = evaluate_model(vllm_model=vllm_model, test_data=test_data, config=config)
        log(
            f"[eval 0] accuracy={initial_metrics['accuracy']:.4f} "
            f"format_rate={initial_metrics['format_rate']:.4f} avg_length={initial_metrics['avg_length']:.1f}"
        )
        wandb.log(
            {
                "eval/ei_step": 0,
                "eval/accuracy": initial_metrics["accuracy"],
                "eval/format_rate": initial_metrics["format_rate"],
                "eval/avg_length": initial_metrics["avg_length"],
            }
        )

        optimize_step = 0
        for ei_step in range(1, config.n_ei_steps + 1):
            log(f"\n{'=' * 50}")
            log(f"[EI Step {ei_step}/{config.n_ei_steps}]")

            n_prompts = min(config.n_prompts_per_rollout_batch, len(train_data))
            prompt_indices = random.sample(range(len(train_data)), n_prompts)
            rollout_prompts =  [train_data[idx]["prompt"] for idx in prompt_indices]
            rollout_ground_truths = [train_data[idx]["expected_answer"] for idx in prompt_indices]

            load_policy_into_vllm_instance(model, vllm_model)
            filtered_samples, rollout_stats = rollout_and_filter(
                vllm_model=vllm_model,
                prompts=rollout_prompts,
                ground_truths=rollout_ground_truths,
                config=config,
            )
            log(
                f"ei_step{ei_step}  Rollout stats: {rollout_stats['correct_rollouts']}/{rollout_stats['total_rollouts']} correct "
                f"({rollout_stats['filter_rate'] * 100:.1f}%)"
            )
            wandb.log(
                {
                    "rollout/ei_step": ei_step,
                    "rollout/correct_rollouts": rollout_stats["correct_rollouts"],
                    "rollout/filter_rate": rollout_stats["filter_rate"],
                    "rollout/filtered_samples": float(len(filtered_samples)),
                }
            )

            if len(filtered_samples) == 0:
                log("  No correct rollouts, skipping SFT for this EI step.")
                continue

            sft_stats, optimize_step = sft_on_filtered_data(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                filtered_samples=filtered_samples,
                config=config,
                ei_step=ei_step,
                optimize_step=optimize_step,
                log=log
            )
            wandb.log(
                {
                    "sft/ei_step": ei_step,
                    "sft/last_loss": sft_stats["sft_last_loss"],
                    "sft/total_loss": sft_stats["sft_total_loss"],
                    "sft/steps": sft_stats["sft_steps"],
                }
            )

            if ei_step % config.eval_every_ei_steps == 0 or ei_step == config.n_ei_steps:
                load_policy_into_vllm_instance((model), vllm_model)
                metrics = evaluate_model(vllm_model=vllm_model, test_data=test_data, config=config)
                log(
                    f"[eval {ei_step}] accuracy={metrics['accuracy']:.4f} "
                    f"format_rate={metrics['format_rate']:.4f} avg_length={metrics['avg_length']:.1f}"
                )
                wandb.log(
                    {
                        "eval/ei_step": ei_step,
                        "eval/accuracy": metrics["accuracy"],
                        "eval/format_rate": metrics["format_rate"],
                        "eval/avg_length": metrics["avg_length"],
                        "eval/avg_correct_length": metrics["avg_correct_length"],
                        "eval/avg_incorrect_length": metrics["avg_incorrect_length"],
                        "eval/correct": metrics["correct"],
                        "eval/total": metrics["total"],
                    }
                )

            checkpoint_dir = output_path / f"{run_name}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _unwrap_policy_model(model).save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

        log("\nExpert Iteration complete.")
    finally:
        wandb.finish()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(
    train_data_path: str = typer.Option(DEFAULT_TRAIN_DATA_PATH, help="Path to EI train JSONL."),
    test_data_path: str = typer.Option(DEFAULT_TEST_DATA_PATH, help="Path to EI validation JSONL."),
    model_path: str = typer.Option(DEFAULT_MODEL_PATH, help="HF model path."),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory to save EI checkpoints."),
    device_train: str = typer.Option("cuda:0", help="Training device target."),
    device_vllm: str = typer.Option("cuda:1", help="vLLM device target."),
    seed: int = typer.Option(69, help="Random seed."),
    n_ei_steps: int = typer.Option(5, help="Number of EI rounds."),
    rollout_batch_size: int = typer.Option(256, help="Total rollouts per EI step."),
    n_rollouts_per_prompt: int = typer.Option(8, help="Rollouts generated per prompt."),
    sft_epochs: int = typer.Option(2, help="SFT epochs per EI step."),
    micro_batch_size: int = typer.Option(8, help="SFT micro-batch size."),
    gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps for SFT."),
    learning_rate: float = typer.Option(2e-5, help="SFT AdamW learning rate."),
    weight_decay: float = typer.Option(0.0, help="SFT AdamW weight decay."),
    eval_every_ei_steps: int = typer.Option(1, help="Evaluate every N EI steps."),
    num_eval_examples: int = typer.Option(1024, help="Evaluation sample size."),
    temperature: float = typer.Option(1.0, help="Generation temperature."),
    top_p: float = typer.Option(1.0, help="Generation top-p."),
    min_tokens: int = typer.Option(4, help="Generation min tokens."),
    max_tokens: int = typer.Option(1024, help="Generation max tokens."),
    gpu_memory_utilization: float = typer.Option(0.9, help="vLLM GPU memory utilization."),
    wandb_project: str = typer.Option("cs336-ei", help="wandb project name."),
    wandb_run_name: Optional[str] = typer.Option(None, help="wandb run name."),
    wandb_mode: str = typer.Option("online", help="wandb mode."),
) -> None:
    config = EIConfig(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        model_path=model_path,
        output_dir=output_dir,
        device_train=device_train,
        device_vllm=device_vllm,
        seed=seed,
        n_ei_steps=n_ei_steps,
        rollout_batch_size=rollout_batch_size,
        n_rollouts_per_prompt=n_rollouts_per_prompt,
        sft_epochs=sft_epochs,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every_ei_steps=eval_every_ei_steps,
        num_eval_examples=num_eval_examples,
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )
    run_expert_iteration(config)


if __name__ == "__main__":
    typer.run(main)