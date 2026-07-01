"""Structured phase-plan encoder used as cross-attention memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import torch
from torch import Tensor, nn


FIELDS = ("phase", "target_part", "hand_part", "motion")
FREE_TEXT_ROW = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?P<phase>[^-]+?)\s*-\s*"
    r"target_part:\s*(?P<target_part>[^-]+?)\s*-\s*"
    r"hand_part:\s*(?P<hand_part>[^-]+?)\s*-\s*"
    r"motion:\s*(?P<motion>.+?)\s*$"
)


def normalize_plan(plan: Any) -> list[dict[str, str]]:
    """Accept v1/v2 dictionaries or a raw list and validate task rows."""
    if isinstance(plan, Mapping):
        plan = plan.get("steps")
    if isinstance(plan, str):
        parsed = []
        for line in plan.splitlines():
            match = FREE_TEXT_ROW.match(line)
            if match:
                parsed.append(
                    {field: match.group(field).strip() for field in FIELDS}
                )
        if not parsed:
            raise ValueError("could not parse any decomposer rows from free text")
        plan = parsed
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise ValueError("task plan must be a list or a {'steps': [...]} dictionary")
    normalized = []
    for index, row in enumerate(plan):
        if not isinstance(row, Mapping):
            raise ValueError(f"task step {index} must be an object")
        missing = [field for field in FIELDS if field not in row]
        if missing:
            raise ValueError(f"task step {index} is missing: {', '.join(missing)}")
        normalized.append({field: str(row[field]) for field in FIELDS})
    if not normalized:
        raise ValueError("task plan cannot be empty")
    return normalized


def _bucket(value: str, num_buckets: int) -> int:
    """Map a string deterministically to an embedding-table row."""
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = (result ^ byte) * 16777619 & 0xFFFFFFFF
    return result % num_buckets


def encode_plan(plan: Any, num_buckets: int = 4096) -> Tensor:
    """Convert a structured plan to precomputed bucket IDs shaped [steps, fields]."""
    rows = normalize_plan(plan)
    return torch.tensor(
        [
            [_bucket(row[field], num_buckets) for field in FIELDS]
            for row in rows
        ],
        dtype=torch.long,
    )


class StructuredTaskEncoder(nn.Module):
    """Learnable, dependency-free encoder for decomposer slot values.

    Hash buckets allow new dataset vocabulary values without changing model
    shapes. Collisions are possible, so production training can replace this
    with a frozen language encoder while preserving the model interface.
    """

    def __init__(self, d_model: int, num_buckets: int = 4096) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embeddings = nn.ModuleDict(
            {field: nn.Embedding(num_buckets, d_model) for field in FIELDS}
        )
        self.field_scale = nn.Parameter(torch.ones(len(FIELDS)))
        self.norm = nn.LayerNorm(d_model)

    def _bucket(self, value: str) -> int:
        return _bucket(value, self.num_buckets)

    def forward(
        self,
        plans: Sequence[Any] | Tensor,
        device: torch.device,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(plans, Tensor):
            plan_ids = plans.to(device)
            if plan_ids.ndim != 3 or plan_ids.shape[-1] != len(FIELDS):
                raise ValueError("encoded plans must have shape [batch, steps, 4]")
            if padding_mask is None:
                raise ValueError("padding_mask is required for encoded plans")
            padding_mask = padding_mask.to(device)
        else:
            encoded = [encode_plan(plan, self.num_buckets) for plan in plans]
            max_steps = max(ids.shape[0] for ids in encoded)
            plan_ids = torch.zeros(
                len(encoded), max_steps, len(FIELDS), dtype=torch.long, device=device
            )
            padding_mask = torch.ones(
                len(encoded), max_steps, dtype=torch.bool, device=device
            )
            for batch_index, ids in enumerate(encoded):
                steps = ids.shape[0]
                plan_ids[batch_index, :steps] = ids.to(device)
                padding_mask[batch_index, :steps] = False

        tokens = torch.stack(
            [
                self.field_scale[field_index]
                * self.embeddings[field](plan_ids[:, :, field_index])
                for field_index, field in enumerate(FIELDS)
            ],
            dim=0,
        ).sum(dim=0)
        return self.norm(tokens), padding_mask
