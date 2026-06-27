import json

import pytest

torch = pytest.importorskip("torch")

from ca_block.model import CrossAttentionAdaLNZero
from ca_block.pose_adapter import load_hoisdf_history
from ca_block.task_encoder import normalize_plan


PLAN = {
    "steps": [
        {
            "phase": "approach",
            "target_part": "object",
            "contact": "none",
            "motion": "reach",
        },
        {
            "phase": "grasp",
            "target_part": "object",
            "contact": "full_grasp",
            "motion": "none",
        },
    ]
}


def test_identity_initialization_and_gradient_flow():
    model = CrossAttentionAdaLNZero(d_model=32, num_heads=4, num_layers=2)
    history = torch.randn(1, 5, 69)
    output = model(history, [PLAN])
    assert output.refined_pose_tokens.shape == (1, 5, 32)
    assert output.pose_delta.shape == history.shape
    assert torch.count_nonzero(output.pose_delta) == 0
    output.pose_delta.sum().backward()
    assert model.delta_head.weight.grad is not None


def test_load_hoisdf_history(tmp_path):
    joints = [[[[0.0, 0.0, 0.0] for _ in range(21)] for _ in range(3)]]
    objects = [
        {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "trans": [0, 0, 0]}
        for _ in range(3)
    ]
    mano = tmp_path / "mano.json"
    obj = tmp_path / "object.json"
    mano.write_text(json.dumps(joints))
    obj.write_text(json.dumps(objects))
    assert load_hoisdf_history(mano, obj, horizon=2).shape == (1, 2, 69)


def test_free_text_decomposer_output():
    plan = normalize_plan(
        "1. approach - target_part: object - contact: none - motion: reach\n"
        "2. grasp - target_part: object - contact: full_grasp - motion: none"
    )
    assert [row["phase"] for row in plan] == ["approach", "grasp"]
