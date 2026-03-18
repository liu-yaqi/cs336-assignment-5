"""GRPO (Group Relative Policy Optimization) functions."""

from typing import Any, Callable, Literal

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths.
        rollout_responses: list[str], rollouts from the policy.
        repeated_ground_truths: list[str], the ground truths for the examples.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero.
        normalize_by_std: bool, whether to normalize the rewards by std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            normalized_rewards, raw_rewards, metadata
    """
    # Compute raw rewards
    raw_rewards = []
    format_rewards = []
    answer_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_result = reward_fn(response, ground_truth)
        raw_rewards.append(reward_result["reward"])
        format_rewards.append(reward_result["format_reward"])
        answer_rewards.append(reward_result["answer_reward"])

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float)

    # Group normalize
    n_groups = len(rollout_responses) // group_size
    normalized_rewards = torch.zeros_like(raw_rewards)

    raw_rewards = raw_rewards.reshape(n_groups, group_size)  # (n_groups, group_size)
    normalized_rewards = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True)  # (n_groups, group_size)

    if normalize_by_std:
        group_std = raw_rewards.std(dim=-1, keepdim=True)  # (n_groups, 1)
        normalized_rewards = normalized_rewards / (group_std + advantage_eps)
    normalized_rewards = normalized_rewards.flatten()
    raw_rewards = raw_rewards.flatten()
    
    metadata = {
        'total_rewards': float(torch.mean(raw_rewards)),
        'format_rewards': sum(format_rewards) / len(format_rewards),
        'answer_rewards': sum(answer_rewards) / len(answer_rewards)
    }

    return normalized_rewards, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    # Policy gradient loss: -reward * log_prob
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the GRPO-Clip per-token loss and metadata.
    """
    # # Expand advantages to match sequence length
    # advantages = advantages.unsqueeze(-1)  # (batch_size, 1, 1)
    # advantages = advantages.expand(-1, policy_log_probs.shape[1])  # (batch_size, seq_len)

    # Compute the ratio: exp(new_log_prob - old_log_prob)
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # Compute the clipped ratio
    ratio_clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    # Compute the loss using the clipped ratio
    # Loss = -min(ratio * advantage, clipped_ratio * advantage)
    v = ratio * advantages
    v_clipped = ratio_clipped * advantages

    # Take the minimum (for positive advantages, we want to clip from above;
    # for negative advantages, we want to clip from below)
    loss = -1 * torch.min(v, v_clipped) # batch_size, seq_len

    metadata = {"clip_fraction": v > v_clipped}

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    Select and compute the desired policy-gradient loss.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
    loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length). cliprange Required for "grpo_clip"; scalar ε used for clipping.
    
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss (batch_size, sequence_length), per-token loss.
    metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip": 
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor)) # (batch_size, seq_len)

    if dim is None:
        return torch.sum(masked_tensor) / (torch.sum(mask) + 1e-8)
    else:   
        sum_result = torch.sum(masked_tensor, dim=dim)
        mask_sum = torch.sum(mask, dim=dim)
        return sum_result / mask_sum 


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    # Compute the per-token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Apply response mask and sum
    loss = masked_mean(per_token_loss, response_mask)

    # Scale by gradient accumulation
    loss = loss / gradient_accumulation_steps

    # Compute gradients
    loss.backward()

    return loss, metadata
