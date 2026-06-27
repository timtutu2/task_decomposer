"""Structured phase-plan encoder used as cross-attention memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import torch
from torch import Tensor, nn


FIELDS = ("phase", "target_part", "contact", "motion")
FREE_TEXT_ROW = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?P<phase>[^-]+?)\s*-\s*"
    r"target_part:\s*(?P<target_part>[^-]+?)\s*-\s*"
    r"contact:\s*(?P<contact>[^-]+?)\s*-\s*"
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
        # FNV-1a is deterministic across processes (unlike Python's hash()).
        result = 2166136261
        for byte in value.encode("utf-8"):
            result = (result ^ byte) * 16777619 & 0xFFFFFFFF
        return result % self.num_buckets

    def forward(self, plans: Sequence[Any], device: torch.device) -> tuple[Tensor, Tensor]:
        rows = [normalize_plan(plan) for plan in plans]
        max_steps = max(len(plan) for plan in rows)
        memory = torch.zeros(len(rows), max_steps, self.norm.normalized_shape[0], device=device)
        padding_mask = torch.ones(len(rows), max_steps, dtype=torch.bool, device=device)
        for batch_index, plan in enumerate(rows):
            for step_index, step in enumerate(plan):
                token = memory.new_zeros(memory.shape[-1])
                for field_index, field in enumerate(FIELDS):
                    bucket = torch.tensor(self._bucket(step[field]), device=device)
                    token = token + self.field_scale[field_index] * self.embeddings[field](bucket)
                memory[batch_index, step_index] = self.norm(token)
                padding_mask[batch_index, step_index] = False
        return memory, padding_mask
