"""Supplemental assignment SFT training entrypoint.

This file is intentionally independent from the math-focused SFT pipeline.
It targets instruction tuning data used in the safety/RLHF supplement.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import random

import torch
import typer
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from cs336_alignment.instruct_data import PackedExample, PackedSFTDataset, get_packed_sft_dataset, iterate_batches
from cs336_alignment.utils import init_log_and_output_dir, save_unwrapped_pretrained, set_seed

app = typer.Typer(add_completion=False, no_args_is_help=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "logs" / "instruct_sft_checkpoints")
DEFAULT_PACKED_DATA_OUTPUT_DIR = str(REPO_ROOT / "data" / "instruct_sft_packed_data")
DEFAULT_TRAIN_DATA_PATH = str(REPO_ROOT / "data" / "sft-data" / "train.jsonl.gz")
DEFAULT_VALID_DATA_PATH = str(REPO_ROOT / "data" / "sft-data" / "test.jsonl.gz")


def apply_wandb_sweep_overrides(config: SFTConfig) -> SFTConfig:
    config_field_names = {f.name for f in fields(SFTConfig)}
    for key, value in dict(wandb.config).items():
        if key in config_field_names:
            setattr(config, key, value)
    return config


@dataclass(slots=True)
class SFTConfig:
    model_path: str
    train_data_path: str = DEFAULT_TRAIN_DATA_PATH
    valid_data_path: str = DEFAULT_VALID_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    packed_data_output_dir: str = DEFAULT_PACKED_DATA_OUTPUT_DIR

    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 32
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_seq_len: int = 512
    max_grad_norm: float = 1.0

    device_train: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    model_gradient_checkpoint: bool = True
    use_torch_compile: bool = True

    eval_interval: int = 100
    save_interval: int = 500
    log_every: int = 1

    wandb_project: str = "cs336-instruct-sft"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"

    def validate(self) -> None:
        assert self.epochs > 0, "epochs must be positive"
        assert self.train_batch_size > 0, "train_batch_size must be positive"
        assert self.micro_batch_size > 0, "micro_batch_size must be positive"
        assert self.train_batch_size >= self.micro_batch_size, "train_batch_size must be >= micro_batch_size"
        self.gradient_accumulation_steps = max(1, math.ceil(self.train_batch_size / self.micro_batch_size))
        assert 0.0 <= self.warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.log_every > 0, "log_every must be positive"
        assert self.eval_interval > 0, "eval_interval must be positive"
        assert self.save_interval > 0, "save_interval must be positive"


def cycle_dataloader(dataloader: DataLoader) -> Any:
    while True:
        for batch in dataloader:
            yield batch


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device_train: str,
    dtype: torch.dtype,
) -> float:
    device_type = "cuda" if "cuda" in device_train else "cpu"
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device_train)
            labels = batch["labels"].to(device_train)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
                out = model(input_ids=input_ids, labels=labels)
            losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def _save_packed_dataset(examples: list[PackedExample], save_path: Path, split: str) -> None:
    serialized_examples: list[dict[str, torch.Tensor]] = []
    for ex in examples:
        serialized_examples.append(
            {
                "input_ids": ex.input_ids,
                "labels": ex.labels,
            }
        )

    torch.save(
        {
            "split": split,
            "num_examples": len(serialized_examples),
            "examples": serialized_examples,
        },
        save_path,
    )


def _load_packed_examples(save_path: Path) -> list[PackedExample]:
    payload = torch.load(save_path, map_location="cpu")
    raw_examples = payload.get("examples", [])
    examples: list[PackedExample] = []
    for ex in raw_examples:
        examples.append(
            PackedExample(
                input_ids=ex["input_ids"].long(),
                labels=ex["labels"].long(),
            )
        )
    return examples


def run_sft(config: SFTConfig) -> None:
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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map=config.device_train,
    )
    if config.model_gradient_checkpoint:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if config.use_torch_compile:
        model = torch.compile(model)
    model.train()
    log(f"model loaded")

    packed_output_dir = Path(config.packed_data_output_dir) / f"f{run_name}"
    train_packed_path = packed_output_dir / "train_packed.pt"
    valid_packed_path = packed_output_dir / "valid_packed.pt"

    if train_packed_path.exists() and valid_packed_path.exists():
        train_examples = _load_packed_examples(train_packed_path)
        valid_examples = _load_packed_examples(valid_packed_path)
        dataset = PackedSFTDataset(train_examples)
        log(f"loaded packed train dataset from {train_packed_path}")
        log(f"loaded packed valid dataset from {valid_packed_path}")
    else:
        train_examples, dataset = get_packed_sft_dataset(
            tokenizer=tokenizer,
            dataset_path=config.train_data_path,
            seq_length=config.max_seq_len,
            shuffle=True,
        )
        valid_examples, valid_dataset = get_packed_sft_dataset(
            tokenizer=tokenizer,
            dataset_path=config.valid_data_path,
            seq_length=config.max_seq_len,
            shuffle=False,
        )
        packed_output_dir.mkdir(parents=True, exist_ok=True)
        _save_packed_dataset(train_examples, train_packed_path, "train")
        _save_packed_dataset(valid_examples, valid_packed_path, "valid")
        log(f"saved packed train dataset to {train_packed_path}")
        log(f"saved packed valid dataset to {valid_packed_path}")

    if len(dataset) == 0:
        raise ValueError("No valid packed training examples found from train_data_path.")
    # if len(valid_dataset) == 0:
    #     raise ValueError("No valid packed validation examples found from valid_data_path.")
    log(f"packed dataset size={len(dataset)} seq_length={config.max_seq_len}")
    # log(f"valid packed dataset size={len(valid_dataset)} seq_length={config.max_seq_len}")

    train_loader = iterate_batches(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
    )

    effective_batch_size = config.gradient_accumulation_steps * config.micro_batch_size
    log(
        f"micro_batch_size={config.micro_batch_size}, train_batch_size={config.train_batch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}, effective_batch_size={effective_batch_size}"
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_training_steps = config.epochs * optimizer_steps_per_epoch
    log(f"epochs={config.epochs}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}, total_steps={total_training_steps}")

    try:
        bnb = importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for 8-bit AdamW. Install it with `pip install bitsandbytes`."
        ) from exc

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    log("optimizer=AdamW8bit")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * config.warmup_ratio),
        num_training_steps=total_training_steps,
    )
    optimizer.zero_grad()

    device_type = "cuda" if "cuda" in config.device_train else "cpu"
    autocast_dtype = torch.bfloat16

    global_step = 0
    accum_count = 0
    accum_loss = 0.0
    best_eval_loss = float("inf")
    best_checkpoint_dir = output_path / f"f{run_name}-best"

    for epoch in range(1, config.epochs + 1):
        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(config.device_train)
            labels = batch["labels"].to(config.device_train)

            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=device_type == "cuda"):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / config.gradient_accumulation_steps

            loss.backward()
            accum_loss += float(loss.detach().cpu())
            accum_count += 1

            should_step = (accum_count == config.gradient_accumulation_steps) or (batch_idx == len(train_loader))
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % config.log_every == 0:
                log(f"[step {global_step}] loss={accum_loss:.6f}")
                wandb.log(
                    {
                        "train/step": global_step,
                        "train/loss": accum_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            should_eval = (global_step % config.eval_interval == 0) or (global_step == total_training_steps)
            should_save_check = (global_step % config.save_interval == 0) or (global_step == total_training_steps)

            if should_eval or global_step == 20:
                if global_step == total_training_steps:
                    valid_dataset = PackedSFTDataset(valid_examples)
                else:
                    sample_valid_examples = random.sample(valid_examples, k=1600)
                    valid_dataset = PackedSFTDataset(sample_valid_examples)
                valid_loader = iterate_batches(
                        valid_dataset,
                        batch_size=config.micro_batch_size,
                        shuffle=False,
                    )   
                eval_loss = _evaluate_loss(
                    model=model,
                    dataloader=valid_loader,
                    device_train=config.device_train,
                    dtype=autocast_dtype,
                )
                log(f"[eval step {global_step}] eval_loss={eval_loss:.6f}")
                wandb.log(
                    {
                        "eval/step": global_step,
                        "eval/loss": eval_loss,
                    },
                    step=global_step,
                )

                if should_save_check and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_unwrapped_pretrained(model, best_checkpoint_dir)
                    tokenizer.save_pretrained(best_checkpoint_dir)
                    log(f"saved best checkpoint (eval_loss={best_eval_loss:.6f}) to {best_checkpoint_dir}")

            accum_count = 0
            accum_loss = 0.0

    log("supplement sft training finished")
    wandb.finish()


def _build_config_from_locals(local_vars: dict[str, Any]) -> SFTConfig:
    # Keep CLI and config fields in sync automatically: only shared names are used.
    field_names = {f.name for f in fields(SFTConfig)}
    config_kwargs = {k: v for k, v in local_vars.items() if k in field_names}
    return SFTConfig(**config_kwargs)


@app.command()
def train(
    model_path: str = typer.Option(DEFAULT_MODEL_PATH, help="Base model path."),
    train_data_path: str = typer.Option(DEFAULT_TRAIN_DATA_PATH, help="Path to sft-data style JSON/JSONL with prompt/response fields."),
    valid_data_path: str = typer.Option(DEFAULT_VALID_DATA_PATH, help="Validation data path under sft-data."),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory for checkpoints."),
    packed_data_output_dir: str = typer.Option(DEFAULT_PACKED_DATA_OUTPUT_DIR, help="Output directory for serialized packed datasets."),
    seed: int = typer.Option(2026, help="Random seed."),
    epochs: int = typer.Option(1, help="Number of training epochs."),
    train_batch_size: int = typer.Option(32, help="Effective train batch size used to derive gradient accumulation."),
    micro_batch_size: int = typer.Option(1, help="Per-step micro batch size."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate."),
    weight_decay: float = typer.Option(0.0, help="Weight decay."),
    warmup_ratio: float = typer.Option(0.03, help="Warmup ratio for linear warmup + cosine decay schedule."),
    max_seq_len: int = typer.Option(512, help="Max sequence length."),
    max_grad_norm: float = typer.Option(1.0, help="Gradient clip norm."),
    device_train: str = typer.Option("cuda:0", help="Training device map target."),
    torch_dtype: str = typer.Option("bfloat16", help="Model/autocast dtype: float32|float16|bfloat16."),
    model_gradient_checkpoint: bool = typer.Option(False, help="Enable model gradient checkpointing."),
    use_torch_compile: bool = typer.Option(True, help="Enable torch.compile for the model."),
    eval_interval: int = typer.Option(100, help="Evaluate every N optimizer steps."),
    save_interval: int = typer.Option(200, help="Consider saving best checkpoint every N optimizer steps."),
    log_every: int = typer.Option(1, help="Log every N steps."),
    wandb_project: str = typer.Option("cs336-supplement-sft", help="wandb project name."),
    wandb_run_name: str | None = typer.Option("default", help="wandb run name."),
    wandb_mode: str = typer.Option("online", help="wandb mode."),
) -> None:
    config = _build_config_from_locals(locals())
    run_sft(config)


if __name__ == "__main__":
    app()
