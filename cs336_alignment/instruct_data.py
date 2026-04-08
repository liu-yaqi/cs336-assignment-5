from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


ALPACA_INSTRUCTION_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n"
    "\n"
    "### Instruction:\n"
    "{prompt}\n"
    "\n"
    "### Response:\n"
    "{response}"
)


@dataclass(slots=True)
class PackedExample:
    # One fixed-length LM training sample (next-token prediction).
    input_ids: torch.Tensor
    labels: torch.Tensor


class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[PackedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "input_ids": example.input_ids,
            "labels": example.labels,
        }


def _open_text(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def _load_sft_documents(dataset_path: str | Path) -> list[str]:
    # Each JSONL row is converted to one Alpaca-style instruction document.
    documents: list[str] = []
    with _open_text(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", ""))
            response = str(row.get("response", ""))
            documents.append(
                ALPACA_INSTRUCTION_TEMPLATE.format(
                    prompt=prompt,
                    response=response,
                )
            )
    return documents


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    documents = _load_sft_documents(dataset_path)
    if shuffle:
        random.shuffle(documents)

    # Concatenate all documents into one token stream.
    # We keep per-document special tokens and ensure docs are delimited by eos.
    token_stream: list[int] = []
    eos_id = tokenizer.eos_token_id
    for doc in documents:
        doc_ids = tokenizer.encode(doc, add_special_tokens=True)
        token_stream.extend(doc_ids)
        if eos_id is not None and (not doc_ids or doc_ids[-1] != eos_id):
            token_stream.append(eos_id)

    # Split into non-overlapping chunks of length seq_length.
    # labels are the next tokens for each input position.
    n_examples = max((len(token_stream) - 1) // seq_length, 0)
    examples: list[PackedExample] = []

    for i in range(n_examples):
        start = i * seq_length
        input_ids = torch.tensor(token_stream[start : start + seq_length], dtype=torch.long)
        labels = torch.tensor(token_stream[start + 1 : start + seq_length + 1], dtype=torch.long)
        examples.append(PackedExample(input_ids=input_ids, labels=labels))

    return examples, PackedSFTDataset(examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    # Keep last partial batch for evaluation/training parity with assignment tests.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    del mmlu_example
    if not model_output:
        return None

    text = model_output.strip().upper()
    # Parse explicit answer markers like "Answer: B" or "B) ...".
    explicit_patterns = [
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([ABCD])\)?\b",
        r"\b([ABCD])\s*[\)\.:]\s",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    if not model_output:
        return None

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None

    # GSM8K convention: use the last numeric value in the output as prediction.
    value = matches[-1].replace(",", "")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".")[0]
    return value
