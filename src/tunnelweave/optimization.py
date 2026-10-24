"""Audited parameter groups and optimizer-step cosine schedule."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .contracts import ContractError


def parameter_groups(model, learning_rates: dict[str, float], weight_decay: float = .01):
    if set(learning_rates) != {"lora", "new", "decoder"} or any(v <= 0 for v in learning_rates.values()):
        raise ContractError("learning_rates must provide positive lora/new/decoder values")
    buckets: dict[tuple[str, bool], list] = defaultdict(list)
    counts = defaultdict(int)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            family = "lora"
        elif name.startswith(("token_rows.", "fusion.", "main_projection.", "aux_projection.")):
            family = "new"
        elif name.startswith(("sam_model.sam_mask_decoder.", "aux_decoder.")):
            family = "decoder"
        else:
            raise ContractError(f"unexpected trainable parameter: {name}")
        no_decay = (name.endswith(".bias") or "norm" in name.lower() or
                    "token" in name.lower() or "embedding" in name.lower() or
                    param.ndim < 2)
        buckets[(family, no_decay)].append(param)
        counts[family] += param.numel()
    if not buckets:
        raise ContractError("no trainable parameters")
    groups = [{"params": params, "lr": learning_rates[family],
               "weight_decay": 0.0 if no_decay else weight_decay,
               "family": family, "no_decay": no_decay}
              for (family, no_decay), params in sorted(buckets.items())]
    return groups, dict(counts)


def cosine_multiplier(optimizer_step: int, total_steps: int,
                      warmup_fraction: float = .05, min_ratio: float = .1) -> float:
    if total_steps <= 0 or not 0 <= warmup_fraction < 1 or not 0 < min_ratio <= 1:
        raise ContractError("invalid schedule")
    warmup = max(1, round(total_steps * warmup_fraction))
    if optimizer_step < warmup:
        return (optimizer_step + 1) / warmup
    if optimizer_step >= total_steps:
        return min_ratio
    progress = (optimizer_step - warmup) / max(1, total_steps - warmup)
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2
