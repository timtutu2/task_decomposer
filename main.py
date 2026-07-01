"""Integrate task decomposition with task-conditioned HOISDF pose refinement.

This entry point is deliberately inference-plumbing only. A new refiner is
identity-initialized (its deltas are zero) and must be trained from trajectory
supervision before its predictions become meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ca_block import CrossAttentionAdaLNZero, load_hoisdf_history


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "refer_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated pose-refinement pipeline.")
    parser.add_argument("task", nargs="?", default="pick the object and rotate the object")
    parser.add_argument("--mano", type=Path, default=REFERENCE / "pred_mano.json")
    parser.add_argument("--object", type=Path, default=REFERENCE / "pred_object.json")
    parser.add_argument("--plan-json", type=Path, help="Use a saved decomposer result; skip Ollama.")
    parser.add_argument("--checkpoint", type=Path, help="Trained refiner state_dict.")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_task_plan(task: str, plan_json: Path | None) -> dict:
    if plan_json is not None:
        with plan_json.open(encoding="utf-8") as stream:
            return json.load(stream)
    # Import lazily: loading saved plans should not require Ollama.
    # The decomposer currently supports direct-script imports (``from schema``),
    # so expose its directory while it is being migrated into a package.
    sys.path.insert(0, str(ROOT / "task_decomposer"))
    from decomposer import run_v2

    return run_v2(task)


def main() -> None:
    args = parse_args()
    plan = get_task_plan(args.task, args.plan_json)
    history = load_hoisdf_history(
        args.mano,
        args.object,
        horizon=args.horizon,
        end_frame=args.end_frame,
        device=args.device,
    )
    model = CrossAttentionAdaLNZero().to(args.device)
    if args.checkpoint:
        try:
            state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        except TypeError:  # PyTorch < 2.0, as used by some HOISDF environments.
            state = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        output = model(history, [plan])

    result = {
        "task": args.task,
        "plan": plan,
        "history_shape": list(history.shape),
        "pose_delta_shape": list(output.pose_delta.shape),
        "last_pose_delta": output.pose_delta[0, -1].cpu().tolist(),
        "warning": None if args.checkpoint else "untrained identity initialization: deltas are zero",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
