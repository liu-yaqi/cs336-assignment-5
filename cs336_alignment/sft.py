"""SFT (Supervised Fine-Tuning) utilities and training entrypoint."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb
import torch
import typer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    get_response_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
    masked_normalize,
    tokenize_prompt_and_output,
    evaluate_vllm,
    load_math_dataset_and_format
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen2.5-Math-1.5B")
DEFAULT_TRAIN_DATA_PATH = str(REPO_ROOT / "data" / "math" / "sft_gpt-oss-120b.jsonl")
DEFAULT_TEST_DATA_PATH = str(REPO_ROOT / "data" / "math" / "val.jsonl")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "model" / "sft_checkpoints")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_wandb_sweep_overrides(config: SFTConfig) -> SFTConfig:
    """Apply wandb sweep parameters to config when keys match dataclass fields."""
    config_field_names = {f.name for f in fields(SFTConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


@dataclass(slots=True)
class SFTConfig:
    train_data_path: str = DEFAULT_TRAIN_DATA_PATH
    test_data_path: str = DEFAULT_TEST_DATA_PATH
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    device_train: str = "cuda:0"
    device_vllm: str = "cuda:1"
    seed: int = 69
    n_sft_steps: int = 256
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    normalize_constant: float = 1.0
    eval_every: int = 16
    num_eval_examples: int = 1024
    log_generation_examples: int = 8
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.85
    wandb_project: str = "cs336-sft"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

    def validate(self) -> None:
        assert self.micro_batch_size > 0, "micro_batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.n_sft_steps > 0, "n_sft_steps must be positive"



class PromptResponseDataset(Dataset):
    def __init__(self, data: list[Dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        # return self.data[idx]['prompt'], self.data[idx]['response'], self.data[idx]['expected_answer']
        return self.data[idx]['prompt'], self.data[idx].get('response', self.data[idx].get('reasoning_trace', ""))


def build_sft_dataloader(
    train_data: list[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    add_eos_to_response: bool = False,
) -> DataLoader:
    dataset = PromptResponseDataset(train_data)

    def collate_fn(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        batch_prompts, batch_responses = zip(*batch)
        response_list = list(batch_responses)
        if add_eos_to_response and tokenizer.eos_token is not None:
            response_list = [
                response if response.endswith(tokenizer.eos_token) else response + tokenizer.eos_token
                for response in response_list
            ]

        tokenized = tokenize_prompt_and_output(
            prompt_strs=list(batch_prompts),
            output_strs=response_list,
            tokenizer=tokenizer,
        )
        return tokenized

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def cycle_dataloader(dataloader: DataLoader) -> Any:
    while True:
        for batch in dataloader:
            yield batch


def log_generations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    vllm_model: LLM,
    sampling_params: SamplingParams,
    data: list[Dict[str, str]],
    device_train: str,
) -> dict[str, Any]:
    prompts = [item["prompt"] for item in data]
    ground_truths = [item["expected_answer"] for item in data]
    response_list = [item.get("response", item.get("reasoning_trace", "")) for item in data]

    metrics, results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        return_output_results=True,
    )
    
    tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=response_list,
            tokenizer=tokenizer,
        )
    input_ids = tokenized["input_ids"].to(device_train)
    labels = tokenized["labels"].to(device_train)
    response_mask = tokenized["response_mask"].to(device_train)
    model.eval()
    with torch.no_grad():
        scored = get_response_log_probs(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=True,
                    )
    model.train()
    entropy = scored['token_entropy'].cpu()
    for i in range(len(entropy)):
        results[i]["avg_token_entropy"] = entropy[i].mean().item()
        results[i]["avg_response_entropy"] = entropy[i, response_mask[i]].mean().item()

    return {
        "metrics": metrics,
        "examples": results}


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute one SFT microbatch backward pass with gradient scaling."""
    masked_token_count = max(normalize_constant, response_mask.sum().item())
    per_token_loss = -policy_log_probs
    loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=masked_token_count,
    )
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "masked_token_count": masked_token_count,
    }
    return loss, metadata


def run_sft(config: SFTConfig) -> None:
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config={k: v for k, v in vars(config).items()},
    )
    config = apply_wandb_sweep_overrides(config)
    config.validate()
    set_seed(config.seed)

    print(config)
    output_path = Path(config.output_dir) # 存log和模型
    output_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=config.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    vllm_model = init_vllm(
        model_id=config.model_path,
        device=config.device_vllm,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    train_data = load_math_dataset_and_format(config.train_data_path)
    train_loader = build_sft_dataloader(
        train_data=train_data,
        tokenizer=tokenizer,
        batch_size=config.micro_batch_size,
    )

    test_data = load_math_dataset_and_format(config.test_data_path)
    log_test_data = random.sample(test_data, min(config.log_generation_examples, len(test_data)))

    optimizer.zero_grad()
    model.train()
    train_iterator = cycle_dataloader(train_loader)

    for step in range(1, config.n_sft_steps + 1):
        step_loss = 0.0
        step_entropy = 0.0
        step_response_entropy = 0.0

        for _ in range(config.gradient_accumulation_steps):
            batch = next(train_iterator)
            input_ids = batch["input_ids"].to(config.device_train)
            labels = batch["labels"].to(config.device_train)
            response_mask = batch["response_mask"].to(config.device_train)

            scored = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=config.normalize_constant,
            )
            response_entropy = scored["token_entropy"][response_mask].mean().detach().cpu().item() / config.gradient_accumulation_steps
            entropy = scored["token_entropy"].mean().detach().cpu().item() / config.gradient_accumulation_steps
            step_loss += float(loss.detach().cpu())
            step_entropy += entropy
            step_response_entropy += response_entropy

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        print(
            f"[step {step}] loss={step_loss:.6f} avg_response_entropy="
            f"{step_response_entropy:.6f} avg_entropy={step_entropy:.6f}"
        )
        wandb.log(
            {
                "train/step": step,
                "train/loss": step_loss,
                "train/avg_entropy": step_entropy,
                "train/avg_response_entropy": step_response_entropy,
            },
            step=step,
        )

        if step % config.eval_every == 0:
            load_policy_into_vllm_instance(model, vllm_model)
            metrics = evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=r1_zero_reward_fn,
                prompts=[item["prompt"] for item in test_data],
                ground_truths=[item["expected_answer"] for item in test_data],
                eval_sampling_params=sampling_params,
            )

            print(
                f"[eval step {step}] accuracy={metrics['accuracy']:.4f} "
                f"format_rate={metrics['format_rate']:.4f} avg_length={metrics['avg_length']:.1f}"
            )
            wandb.log(
                {
                    "eval/step": step,
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_length": metrics["avg_length"],
                    "eval/avg_correct_length": metrics["avg_correct_length"],
                    "eval/avg_incorrect_length": metrics["avg_incorrect_length"],
                    "eval/correct": metrics["correct"],
                    "eval/total": metrics["total"],
                },
                step=step,
            )

            generation_log = log_generations(
                model=model,
                tokenizer=tokenizer,
                vllm_model=vllm_model,
                sampling_params=sampling_params,
                data = log_test_data, 
                device_train=config.device_train
            )

            for example_index, example in enumerate(generation_log["examples"], start=1):
                print(f"[generation {example_index}] prompt={example['prompt']}")
                print(f"[generation {example_index}] response={example['model_output']}")
                print(f"[generation {example_index}] ground_truth={example['ground_truth']}")
                print(f"[generation {example_index}] reward={example['reward']}")
                print(
                    f"[generation {example_index}] avg_token_entropy="
                    f"{example['avg_token_entropy']:.4f} avg_response_entropy="
                    f"{example['avg_response_entropy']:.4f} response_length="
                    f"{example['response_length']:.0f}"
                )

            wandb.log(
                {
                    "eval_examples/accuracy": generation_log["metrics"]["accuracy"],
                    "eval_examples/format_rate": generation_log["metrics"]["format_rate"],
                    "eval_examples/avg_length": generation_log["metrics"]["avg_length"],
                },
                step=step,
            )

            checkpoint_dir = output_path / f"step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    wandb.finish()


def main(
    train_data_path: str = typer.Option(DEFAULT_TRAIN_DATA_PATH, help="Path to SFT train JSONL."),
    test_data_path: str = typer.Option(DEFAULT_TEST_DATA_PATH, help="Path to validation JSONL."),
    model_path: str = typer.Option(DEFAULT_MODEL_PATH, help="HF model path."),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints."),
    device_train: str = typer.Option("cuda:0", help="Training device map target."),
    device_vllm: str = typer.Option("cuda:1", help="vLLM device target."),
    seed: int = typer.Option(2026, help="Random seed."),
    n_sft_steps: int = typer.Option(256, help="Number of optimizer steps."),
    micro_batch_size: int = typer.Option(8, help="Per-microbatch size."),
    gradient_accumulation_steps: int = typer.Option(8, help="Microbatches per optimizer step."),
    learning_rate: float = typer.Option(2e-5, help="AdamW learning rate."),
    weight_decay: float = typer.Option(0.0, help="AdamW weight decay."),
    beta1: float = typer.Option(0.9, help="AdamW beta1."),
    beta2: float = typer.Option(0.95, help="AdamW beta2."),

    normalize_constant: float = typer.Option(1.0, help="Loss normalization constant."),

    eval_every: int = typer.Option(16, help="Run validation every N optimizer steps."),
    num_eval_examples: int = typer.Option(1024, help="Validation examples per eval."),
    log_generation_examples: int = typer.Option(8, help="Examples to log with generation."),
    max_tokens: int = typer.Option(1024, help="Generation max tokens."),
    temperature: float = typer.Option(1.0, help="Generation temperature."),
    top_p: float = typer.Option(1.0, help="Generation top-p."),
    gpu_memory_utilization: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    
    wandb_project: str = typer.Option("cs336-sft", help="wandb project name."),
    wandb_run_name: Optional[str] = typer.Option(None, help="wandb run name."),
    wandb_mode: str = typer.Option("online", help="wandb mode."),
) -> None:
    config = SFTConfig(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        model_path=model_path,
        output_dir=output_dir,
        device_train=device_train,
        device_vllm=device_vllm,
        seed=seed,
        n_sft_steps=n_sft_steps,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        normalize_constant=normalize_constant,
        eval_every=eval_every,
        num_eval_examples=num_eval_examples,
        log_generation_examples=log_generation_examples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        gpu_memory_utilization=gpu_memory_utilization,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )
    run_sft(config)


if __name__ == "__main__":
    typer.run(main)