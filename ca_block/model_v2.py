"""Cross-attention pose refiner with pre-norm transformer decoder blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .pose_adapter import POSE_FEATURES
from .task_encoder import StructuredTaskEncoder


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer decoder block: self-attention + cross-attention + MLP."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        pose: Tensor,
        task_memory: Tensor,
        task_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normed = self.norms[0](pose)
        pose = pose + self.self_attention(normed, normed, normed, need_weights=False)[0]

        query = self.norms[1](pose)
        pose = pose + self.cross_attention(
            query, task_memory, task_memory,
            key_padding_mask=task_padding_mask,
            need_weights=False,
        )[0]

        return pose + self.mlp(self.norms[2](pose))


@dataclass
class ModelOutput:
    refined_pose_tokens: Tensor
    pose_delta: Tensor


class CrossAttentionAdaLNZero(nn.Module):
    """End-to-end refiner: stacked cross-attention decoder blocks over pose history and task plan."""

    def __init__(
        self,
        pose_features: int = POSE_FEATURES,
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.pose_features = pose_features
        self.pose_projection = nn.Linear(pose_features, d_model)
        self.task_encoder = StructuredTaskEncoder(d_model)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, num_heads) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, pose_features)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, pose_history: Tensor, task_plans: Sequence[Any]) -> ModelOutput:
        if pose_history.ndim != 3 or pose_history.shape[-1] != self.pose_features:
            raise ValueError(
                f"pose_history must have shape [batch, time, {self.pose_features}]"
            )
        if len(task_plans) != pose_history.shape[0]:
            raise ValueError("one task plan is required for each pose-history batch item")
        pose = self.pose_projection(pose_history)
        task_memory, task_mask = self.task_encoder(task_plans, pose.device)
        for block in self.blocks:
            pose = block(pose, task_memory, task_mask)
        delta = self.delta_head(self.output_norm(pose))
        return ModelOutput(refined_pose_tokens=pose, pose_delta=delta)
