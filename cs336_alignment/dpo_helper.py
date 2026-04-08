from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from .instruct_data import ALPACA_INSTRUCTION_TEMPLATE


def response_logprob(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    device: torch.device | str | None = None,
    max_length: int | None = None,
) -> torch.Tensor:
    # per-instance处理

    full_prompt = ALPACA_INSTRUCTION_TEMPLATE.format(prompt=prompt, response="")

    prompt_ids = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        full_prompt + response,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )["input_ids"]
    full_ids = torch.cat([full_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1)

    if device is not None:
        full_ids = full_ids.to(device)

    if full_ids.shape[1] <= 1:
        return torch.tensor(0.0, device=device)

    prompt_len = prompt_ids.shape[1]

    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]

    logits = model(input_ids=input_ids).logits
    vocab_size = logits.shape[-1]
    token_nll = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    token_log_probs = -token_nll

    response_start = max(prompt_len - 1, 0)
    if response_start >= token_log_probs.shape[1]:
        return torch.tensor(0.0, device=device)

    return token_log_probs[:, response_start:].sum(dim=-1).squeeze(0)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    policy_device: torch.device | str | None = None,
    reference_device: torch.device | str | None = None,
    max_length: int | None = None,
) -> torch.Tensor:
    loss, *_ = compute_per_instance_dpo_loss_and_logps(
        lm=lm,
        lm_ref=lm_ref,
        tokenizer=tokenizer,
        beta=beta,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
        policy_device=policy_device,
        reference_device=reference_device,
        max_length=max_length,
    )
    return loss


def compute_per_instance_dpo_loss_and_logps(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    policy_device: torch.device | str | None = None,
    reference_device: torch.device | str | None = None,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # per-instance处理
    chosen_logp = response_logprob(
        lm,
        tokenizer,
        prompt,
        response_chosen,
        device=policy_device,
        max_length=max_length,
    )
    rejected_logp = response_logprob(
        lm,
        tokenizer,
        prompt,
        response_rejected,
        device=policy_device,
        max_length=max_length,
    )

    with torch.no_grad():
        chosen_logp_ref = response_logprob(
            lm_ref,
            tokenizer,
            prompt,
            response_chosen,
            device=reference_device,
            max_length=max_length,
        )
        rejected_logp_ref = response_logprob(
            lm_ref,
            tokenizer,
            prompt,
            response_rejected,
            device=reference_device,
            max_length=max_length,
        )

    pi_logratio = chosen_logp - rejected_logp
    ref_logratio = chosen_logp_ref - rejected_logp_ref
    if policy_device != reference_device:
        ref_logratio = ref_logratio.to(pi_logratio.device)
    dpo_margin = pi_logratio - ref_logratio
    return -F.logsigmoid(beta * dpo_margin), chosen_logp, rejected_logp, chosen_logp_ref, rejected_logp_ref
