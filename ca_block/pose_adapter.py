"""Adapters for the JSON produced by the current HOISDF export.

Each temporal token is
    21 hand joints * xyz + object axis-angle + object translation = 69 values.
The mesh vertices in ``pred_mano.json`` are intentionally ignored: they are
redundant with the joints for this first integration and are much larger.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


POSE_FEATURES = 21 * 3 + 3 + 3


def _rotation_matrix_to_axis_angle(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Convert a 3x3 rotation matrix to a stable axis-angle vector."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    vector = [
        matrix[2][1] - matrix[1][2],
        matrix[0][2] - matrix[2][0],
        matrix[1][0] - matrix[0][1],
    ]
    sin_angle = math.sin(angle)
    if angle < 1e-6:
        return [value / 2.0 for value in vector]
    if abs(sin_angle) < 1e-6:
        # Near pi, recover the unsigned axis from the diagonal.
        axis = [
            math.sqrt(max(0.0, (matrix[i][i] + 1.0) / 2.0))
            for i in range(3)
        ]
        axis[0] = math.copysign(axis[0], vector[0] or 1.0)
        axis[1] = math.copysign(axis[1], vector[1] or 1.0)
        axis[2] = math.copysign(axis[2], vector[2] or 1.0)
        norm = math.sqrt(sum(value * value for value in axis)) or 1.0
        return [angle * value / norm for value in axis]
    scale = angle / (2.0 * sin_angle)
    return [scale * value for value in vector]


def load_hoisdf_history(
    mano_path: str | Path,
    object_path: str | Path,
    *,
    horizon: int = 5,
    end_frame: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Load one causal history window as ``[1, horizon, 69]``.

    ``end_frame`` is exclusive. It defaults to the final common frame.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    with Path(mano_path).open(encoding="utf-8") as stream:
        mano = json.load(stream)
    with Path(object_path).open(encoding="utf-8") as stream:
        objects = json.load(stream)

    if not isinstance(mano, list) or len(mano) < 1:
        raise ValueError("MANO JSON must contain [joints, vertices]")
    joints = mano[0]
    frame_count = min(len(joints), len(objects))
    end = frame_count if end_frame is None else end_frame
    if not horizon <= end <= frame_count:
        raise ValueError(
            f"need 0 <= end_frame <= {frame_count} and at least {horizon} history frames"
        )

    rows: list[list[float]] = []
    for index in range(end - horizon, end):
        frame_joints = joints[index]
        if len(frame_joints) != 21 or any(len(joint) != 3 for joint in frame_joints):
            raise ValueError(f"frame {index}: expected 21x3 hand joints")
        obj = objects[index]
        rotation = obj.get("rot")
        translation = obj.get("trans")
        if (
            not isinstance(rotation, list)
            or len(rotation) != 3
            or any(len(row) != 3 for row in rotation)
            or not isinstance(translation, list)
            or len(translation) != 3
        ):
            raise ValueError(f"frame {index}: expected object rot[3][3] and trans[3]")
        hand = [coordinate for joint in frame_joints for coordinate in joint]
        rows.append(hand + _rotation_matrix_to_axis_angle(rotation) + translation)

    return torch.tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)
