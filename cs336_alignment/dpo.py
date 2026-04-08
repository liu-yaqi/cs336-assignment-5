"""Supplemental assignment DPO training entrypoint.

This file is independent from the main GRPO/SFT pipelines and is intended for
preference optimization in the safety/RLHF supplement.
"""

from __future__ import annotations

import os
import gzip
import importlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.dpo_helper import (
    compute_per_instance_dpo_loss_and_logps,
    response_logprob,
)
from cs336_alignment.utils import init_log_and_output_dir, save_unwrapped_pretrained, set_seed

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = typer.Typer(add_completion=False, no_args_is_help=True)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "/root/autodl-tmp/cs336-assignment-5/logs/instruct_sft_checkpoints/0406-093056-default/fdefault-best"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "logs" / "supplement_dpo_checkpoints")
DEFAULT_TRAIN_DATA_PATH = "/root/autodl-tmp/dataset"

_HUMAN_SPLIT_RE = re.compile(r"\n\nHuman:\s*")
_ASSISTANT_SPLIT_RE = re.compile(r"\n\nAssistant:\s*")


def apply_wandb_sweep_overrides(config: DPOConfig) -> DPOConfig:
    config_field_names = {f.name for f in fields(DPOConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


@dataclass(slots=True)
class DPOConfig:
    model_path: str
    preference_data_path: str
    output_dir: str = DEFAULT_OUTPUT_DIR

    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 64
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    learning_rate: float = 1e-6
    optimizer: str = "rmsprop"
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    beta: float = 0.1
    max_length: int = 512
    max_grad_norm: float = 1.0

    policy_device: str = "cuda:0"
    reference_device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    model_gradient_checkpoint: bool = False
    use_torch_compile: bool = True

    val_size: int = 200
    eval_interval: int = 100
    save_interval: int = 500

    wandb_project: str = "cs336-supplement-dpo"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"

    def validate(self) -> None:
        assert self.epochs > 0, "epochs must be positive"
        assert self.train_batch_size > 0, "train_batch_size must be positive"
        assert self.micro_batch_size > 0, "micro_batch_size must be positive"
        assert self.train_batch_size >= self.micro_batch_size, "train_batch_size must be >= micro_batch_size"
        self.gradient_accumulation_steps = max(1, math.ceil(self.train_batch_size / self.micro_batch_size))
        self.optimizer = self.optimizer.strip().lower()
        assert self.optimizer in {"rmsprop", "adamw8bit"}, "optimizer must be one of: rmsprop, adamw8bit"
        assert 0.0 <= self.warmup_ratio <= 1.0, "warmup_ratio must be in [0, 1]"
        assert self.eval_interval > 0, "eval_interval must be positive"
        assert self.save_interval > 0, "save_interval must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert self.val_size >= 0, "val_size must be non-negative"


def _create_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for lr scheduler")

    warmup_steps = 0
    if warmup_ratio > 0.0:
        warmup_steps = max(1, int(total_steps * warmup_ratio))
    warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(current_step: int) -> float:
        step = max(0, current_step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps == warmup_steps:
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _extract_first_turn(conversation: str) -> tuple[str, str, int]:
    text = conversation.strip()
    if not text:
        return "", "", 0

    if not text.startswith("Human:"):
        text = _HUMAN_SPLIT_RE.sub("\n\nHuman: ", "\n\n" + text, count=1).strip()

    human_parts = _HUMAN_SPLIT_RE.split("\n\n" + text)
    turns = [p for p in human_parts[1:] if p.strip()]
    num_human_turns = len(turns)
    if num_human_turns == 0:
        return "", "", 0

    first_turn = turns[0]
    assistant_parts = _ASSISTANT_SPLIT_RE.split(first_turn, maxsplit=1)
    if len(assistant_parts) != 2:
        return "", "", num_human_turns

    instruction = assistant_parts[0].strip()
    assistant_response = assistant_parts[1].strip()
    return instruction, assistant_response, num_human_turns


def load_anthropic_hh_dataset(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load Anthropic HH data with single-turn filtering.

    Returns a combined list from all files where each row contains
    instruction, chosen, rejected, and source_file.
    """
    processed: list[dict[str, Any]] = []
    filter_stats: dict[str, int] = {
        "total_rows": 0,
        "missing_chosen_or_rejected": 0,
        "non_single_turn": 0,
        "missing_instruction_or_response": 0,
        "instruction_mismatch": 0,
        "kept": 0,
    }

    def _preview(text: str, max_chars: int = 120) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    for file_path in file_paths:
        src_path = Path(file_path)
        source_file = src_path.name
        file_total = 0
        file_kept = 0

        with _open_text(src_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                filter_stats["total_rows"] += 1
                file_total += 1
                row = json.loads(line)

                chosen_text = str(row.get("chosen", ""))
                rejected_text = str(row.get("rejected", ""))
                if not chosen_text or not rejected_text:
                    filter_stats["missing_chosen_or_rejected"] += 1
                    continue

                instruction_c, chosen_resp, human_turns_c = _extract_first_turn(chosen_text)
                instruction_r, rejected_resp, human_turns_r = _extract_first_turn(rejected_text)

                if human_turns_c != 1 or human_turns_r != 1:
                    filter_stats["non_single_turn"] += 1
                    continue
                if not instruction_c or not chosen_resp or not rejected_resp:
                    filter_stats["missing_instruction_or_response"] += 1
                    continue
                if instruction_c != instruction_r:
                    filter_stats["instruction_mismatch"] += 1
                    continue

                processed.append(
                    {
                        "instruction": instruction_c,
                        "chosen": chosen_resp,
                        "rejected": rejected_resp,
                        "source_file": source_file,
                    }
                )
                filter_stats["kept"] += 1
                file_kept += 1

        typer.echo(
            f"[load_anthropic_hh_dataset] file={source_file} "
            f"raw_rows={file_total} kept={file_kept} filtered={file_total - file_kept}"
        )

    typer.echo("[load_anthropic_hh_dataset] filter summary:")
    typer.echo(f"  total_rows={filter_stats['total_rows']}")
    typer.echo(f"  missing_chosen_or_rejected={filter_stats['missing_chosen_or_rejected']}")
    typer.echo(f"  non_single_turn={filter_stats['non_single_turn']}")
    typer.echo(f"  missing_instruction_or_response={filter_stats['missing_instruction_or_response']}")
    typer.echo(f"  instruction_mismatch={filter_stats['instruction_mismatch']}")
    typer.echo(f"  kept={filter_stats['kept']}")

    if processed:
        sample_n = min(3, len(processed))
        typer.echo(f"[load_anthropic_hh_dataset] processed sample (first {sample_n}):")
        for i, row in enumerate(processed[:sample_n], start=1):
            typer.echo(
                f"  sample#{i} source={row['source_file']} "
                f"instruction='{_preview(str(row['instruction']))}' "
                f"chosen='{_preview(str(row['chosen']))}' "
                f"rejected='{_preview(str(row['rejected']))}'"
            )
    else:
        typer.echo("[load_anthropic_hh_dataset] processed sample: none (no valid rows after filtering)")

    return processed


@dataclass(slots=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    source_file: str


def _load_preference_examples(preference_data_path: str | Path) -> list[PreferenceExample]:
    hh_files = [Path( preference_data_path + "/harmless-base/train.jsonl.gz"), 
                Path(preference_data_path + "/helpful-base/train.jsonl.gz"), 
                Path(preference_data_path + "/helpful-online/train.jsonl.gz"), 
                Path(preference_data_path + "/helpful-rejection-sampled/train.jsonl.gz"),
                Path( preference_data_path + "/harmless-base/test.jsonl.gz"), 
                Path(preference_data_path + "/helpful-base/test.jsonl.gz"), 
                Path(preference_data_path + "/helpful-online/test.jsonl.gz"), 
                Path(preference_data_path + "/helpful-rejection-sampled/test.jsonl.gz"),
                ]
    if not hh_files:
        raise ValueError(
            "No HH files found. Provide a directory, a comma-separated list of files, "
            "or a single HH .jsonl/.jsonl.gz file path."
        )

    hh_rows = load_anthropic_hh_dataset(hh_files)
    return [
        PreferenceExample(
            prompt=str(row["instruction"]),
            chosen=str(row["chosen"]),
            rejected=str(row["rejected"]),
            source_file=str(row["source_file"]),
        )
        for row in hh_rows
    ]


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _eval_classification_accuracy(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    tokenizer: AutoTokenizer,
    val_data: list[PreferenceExample],
    beta: float,
    policy_device: str,
    reference_device: str,
    max_length: int,
) -> tuple[float, float, float]:
    if not val_data:
        return 0.0, 0.0

    was_training = policy.training
    correct = 0
    losses: list[float] = []
    margins: list[float] = []
    policy.eval()
    reference.eval()
    device_type = "cuda" if "cuda" in policy_device else "cpu"
    with torch.no_grad():
        for ex in val_data:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                loss, chosen_logp, rejected_logp, chosen_logp_ref, rejected_logp_ref = compute_per_instance_dpo_loss_and_logps(
                    lm=policy,
                    lm_ref=reference,
                    tokenizer=tokenizer,
                    beta=beta,
                    prompt=ex.prompt,
                    response_chosen=ex.chosen,
                    response_rejected=ex.rejected,
                    policy_device=policy_device,
                    reference_device=reference_device,
                    max_length=max_length,
                )
            
            chosen_reward =  (chosen_logp - chosen_logp_ref.to(chosen_logp.device))
            rejected_reward =  (rejected_logp - rejected_logp_ref.to(rejected_logp.device))
            
            if float(chosen_reward.detach().cpu()) > float(rejected_reward.detach().cpu()):
                correct += 1
            margin = float((rejected_reward - chosen_reward).detach().cpu().item())
            losses.append(loss.detach().cpu().item())
            margins.append(margin)
            
    if was_training:
        policy.train()
    else:
        policy.eval()
    return correct / len(val_data), sum(losses) / max(len(losses), 1), sum(margins) / max(len(margins), 1)


def _build_config_from_locals(local_vars: dict[str, Any]) -> DPOConfig:
    field_names = {f.name for f in fields(DPOConfig)}
    kwargs = {k: v for k, v in local_vars.items() if k in field_names}
    return DPOConfig(**kwargs)


def run_dpo(config: DPOConfig) -> None:
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
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pref_data = _load_preference_examples(config.preference_data_path)
    if not pref_data:
        raise ValueError("No valid preference data found (prompt/chosen/rejected).")

    random.shuffle(pref_data)
    val_count = min(config.val_size, max(1, len(pref_data) // 20)) if config.val_size > 0 else 0
    val_data = pref_data[:val_count]
    train_data = pref_data[val_count:]
    if not train_data:
        raise ValueError("Training split is empty. Reduce val_size or provide more data.")
    log(f"loaded preference examples={len(pref_data)} train={len(train_data)} val={len(val_data)}")

    policy = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    policy.to(config.policy_device)
    if config.model_gradient_checkpoint:
        policy.config.use_cache = False
        try:
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            policy.gradient_checkpointing_enable()
    if config.use_torch_compile:
        policy = torch.compile(policy)
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    reference.to(config.reference_device)
    reference.eval()
    reference.requires_grad_(False)

    # Current supplemental training path assumes policy and reference are colocated.
    if _get_model_device(policy) != _get_model_device(reference):
        raise ValueError(
            "supplement_dpo currently assumes policy/reference are on the same device. "
            "Set --reference-device equal to --policy-device."
        )

    if config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            policy.parameters(),
            lr=config.learning_rate,
            alpha=config.rmsprop_alpha,
            eps=config.rmsprop_eps,
            weight_decay=config.weight_decay,
        )
        log("optimizer=RMSprop")
    else:
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for optimizer=adamw8bit. Install it with `pip install bitsandbytes`."
            ) from exc
        optimizer = bnb.optim.AdamW8bit(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.99),
        )
        log("optimizer=AdamW8bit")
    optimizer.zero_grad()

    cursor = 0
    random.shuffle(train_data)
    device_type = "cuda" if "cuda" in config.policy_device else "cpu"

    effective_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
    optimizer_steps_per_epoch = math.ceil(len(train_data) / effective_batch_size)
    total_training_steps = config.epochs * optimizer_steps_per_epoch
    scheduler = _create_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=total_training_steps,
        warmup_ratio=config.warmup_ratio,
    )
    log(
        f"micro_batch_size={config.micro_batch_size}, train_batch_size={config.train_batch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}, effective_batch_size={effective_batch_size}"
    )
    log(f"epochs={config.epochs}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}, total_steps={total_training_steps}")
    log(f"lr_scheduler=cosine_with_warmup warmup_ratio={config.warmup_ratio:.4f}")

    global_step = 0
    for _ in range(config.epochs):
        for _ in range(optimizer_steps_per_epoch):
            global_step += 1
            step_loss = 0.0
            step_correct = 0
            step_total = 0
            step_margin_loss = 0.0

            for _ in range(config.gradient_accumulation_steps):
                for _ in range(config.micro_batch_size):
                    ex = train_data[cursor]
                    cursor += 1
                    if cursor >= len(train_data):
                        cursor = 0
                        random.shuffle(train_data)

                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                        loss, chosen_logp, rejected_logp, chosen_logp_ref, rejected_logp_ref = compute_per_instance_dpo_loss_and_logps(
                            lm=policy,
                            lm_ref=reference,
                            tokenizer=tokenizer,
                            beta=config.beta,
                            prompt=ex.prompt,
                            response_chosen=ex.chosen,
                            response_rejected=ex.rejected,
                            policy_device=config.policy_device,
                            reference_device=config.reference_device,
                            max_length=config.max_length,
                        )
                    loss = loss / (config.gradient_accumulation_steps * config.micro_batch_size)
                    loss.backward()

                    chosen_reward = config.beta * (chosen_logp - chosen_logp_ref.to(chosen_logp.device))
                    rejected_reward = config.beta * (rejected_logp - rejected_logp_ref.to(rejected_logp.device))
                    
                    if float(chosen_reward.detach().cpu()) > float(rejected_reward.detach().cpu()):
                        step_correct += 1
                    step_total += 1
                    step_margin_loss += float((rejected_reward - chosen_reward).detach().cpu())
                    step_loss += float(loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step_acc = step_correct / max(step_total, 1)
            step_margin_loss = step_margin_loss / max(step_total, 1)
            current_lr = scheduler.get_last_lr()[0]
            log(
                f"[dpo step {global_step}] loss={step_loss:.6f} "
                f"train_acc={step_acc:.4f} train_margin_loss={step_margin_loss:.6f} "
                f"lr={current_lr:.8e}"
            )
            wandb.log(
                {
                    "train/dpo_step": global_step,
                    "train/loss": step_loss,
                    "train/accuracy": step_acc,
                    "train/margin_loss": step_margin_loss,
                    "train/lr": current_lr,
                },
                step=global_step,
            )

            if global_step % config.eval_interval == 0 or global_step == total_training_steps:
                val_acc, val_loss, val_margin_loss = _eval_classification_accuracy(
                    policy=policy,
                    reference=reference,
                    tokenizer=tokenizer,
                    val_data=val_data,
                    beta=config.beta,
                    policy_device=config.policy_device,
                    reference_device=config.reference_device,
                    max_length=config.max_length,
                )
                log(
                    f"[===eval step {global_step}] "
                    f"val_classification_accuracy={val_acc:.4f} "
                    f"val_loss={val_loss:.6f} "
                    f"val_margin_loss={val_margin_loss:.6f}"
                )
                wandb.log(
                    {
                        "eval/step": global_step,
                        "eval/classification_accuracy": val_acc,
                        "eval/val_loss": val_loss,
                        "eval/val_margin_loss": val_margin_loss,
                    },
                    step=global_step,
                )

            if global_step % config.save_interval == 0 or global_step == total_training_steps:
                checkpoint_dir = output_path / f"best"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_unwrapped_pretrained(policy, checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                log(f"saved checkpoint to {checkpoint_dir}")

                torch.cuda.empty_cache()

    log("dpo training finished")
    wandb.finish()


@app.command()
def train(
    model_path: str = typer.Option(..., help="Instruction-finetuned model path."),
    preference_data_path: str = typer.Option(DEFAULT_TRAIN_DATA_PATH, help="HH preference dataset path (dir, .jsonl, .jsonl.gz, or comma-separated files)."),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory."),
    seed: int = typer.Option(69, help="Random seed."),
    epochs: int = typer.Option(1, help="Number of training epochs."),
    train_batch_size: int = typer.Option(64, help="Effective train batch size used to derive gradient accumulation."),
    micro_batch_size: int = typer.Option(1, help="Per-step micro batch size."),
    learning_rate: float = typer.Option(1e-6, help="RMSprop learning rate."),
    optimizer: str = typer.Option("rmsprop", help="Optimizer: rmsprop or adamw8bit."),
    rmsprop_alpha: float = typer.Option(0.99, help="RMSprop alpha."),
    rmsprop_eps: float = typer.Option(1e-8, help="RMSprop eps."),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio used by cosine LR scheduler."),
    weight_decay: float = typer.Option(0.0, help="Weight decay."),
    beta: float = typer.Option(0.1, help="DPO beta."),
    max_length: int = typer.Option(512, help="Max token length used in DPO log-prob computation."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clip norm."),
    policy_device: str = typer.Option("cuda:0", help="Device for trainable policy model."),
    reference_device: str = typer.Option("cuda:0", help="Device for frozen reference model."),
    torch_dtype: str = typer.Option("bfloat16", help="Model/autocast dtype: float32|float16|bfloat16."),
    model_gradient_checkpoint: bool = typer.Option(True, help="Enable policy model gradient checkpointing."),
    use_torch_compile: bool = typer.Option(False, help="Enable torch.compile for policy model."),
    val_size: int = typer.Option(200, help="Validation example count."),
    eval_interval: int = typer.Option(50, help="Evaluate every N optimizer steps."),
    save_interval: int = typer.Option(150, help="Save checkpoint every N optimizer steps."),
    wandb_project: str = typer.Option("cs336-supplement-dpo", help="wandb project name."),
    wandb_run_name: str | None = typer.Option("default", help="wandb run name."),
    wandb_mode: str = typer.Option("online", help="wandb mode."),
) -> None:
    config = _build_config_from_locals(locals())
    run_dpo(config)


if __name__ == "__main__":
    app()
