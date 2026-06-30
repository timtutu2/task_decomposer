#!/usr/bin/env python3
"""Evaluate a trained checkpoint against the HOISDF baseline on HO3D test sequences.

Object pose metrics (in mm, matching HOISDF paper conventions):
  OCE   – Object Center Error : ||pred_trans - gt_trans||
  MCE   – Mean Corner Error   : mean_i ||pred_corner_i - gt_corner_i||
  ADD-S – Symmetric avg closest-point distance over the 8 bbox corners

Reports HOISDF (no refinement) vs Refined (HOISDF + model delta).
Note: hand MJE is omitted because test-set GT only provides the wrist joint.

Usage:
    python eval.py checkpoints/final.pt
    python eval.py checkpoints/epoch_050.pt --horizon 5 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ca_block import CrossAttentionAdaLNZero
from ca_block.pose_adapter import _rotation_matrix_to_axis_angle

PREDS_DIR = ROOT / "preprocess" / "hoisdf_preds" / "ho3d" / "test"
PLANS_DIR = ROOT / "preprocess" / "plans"        / "ho3d" / "test"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot_matrix_from_aa(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) → rotation matrix (3, 3) via Rodrigues."""
    R, _ = cv2.Rodrigues(aa.reshape(3, 1))
    return R


def _transform_corners(
    rot: np.ndarray,   # (3, 3) rotation matrix
    trans: np.ndarray, # (3,)  translation
    rest: np.ndarray,  # (8, 3) corners in rest pose
) -> np.ndarray:
    """Apply rigid transform to rest-pose corners → (8, 3)."""
    return rest @ rot.T + trans


def _oce(pred_t: np.ndarray, gt_t: np.ndarray) -> float:
    return float(np.linalg.norm(pred_t - gt_t)) * 1000.0  # mm


def _mce(pred_c: np.ndarray, gt_c: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_c - gt_c, axis=1))) * 1000.0  # mm


def _adds(pred_c: np.ndarray, gt_c: np.ndarray) -> float:
    # for each predicted corner, find nearest GT corner
    dists = np.linalg.norm(pred_c[:, None, :] - gt_c[None, :, :], axis=2)
    return float(np.mean(dists.min(axis=1))) * 1000.0  # mm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pred_features(pred_dir: Path) -> np.ndarray:
    """Return (T, 69) float32: 63 hand joints + 3 obj axis-angle + 3 obj trans."""
    with open(pred_dir / "pred_mano.json") as f:
        mano = json.load(f)
    with open(pred_dir / "pred_object.json") as f:
        objects = json.load(f)

    joints_list = mano[0]
    T = min(len(joints_list), len(objects))
    rows = []
    for i in range(T):
        hand = [c for j in joints_list[i] for c in j]          # 63 values
        rot  = _rotation_matrix_to_axis_angle(objects[i]["rot"])  # 3 values
        trans = objects[i]["trans"]                               # 3 values
        rows.append(hand + rot + trans)
    return np.array(rows, dtype=np.float32)


def _load_gt_object(ho3d_test: Path, seq: str, frames: list[int]) -> list[dict]:
    """Load per-frame GT object data from HO3D test meta pkls."""
    gt = []
    for f in frames:
        ann = np.load(ho3d_test / seq / "meta" / f"{f:04d}.pkl", allow_pickle=True)
        R_gt = _rot_matrix_from_aa(np.array(ann["objRot"]).flatten())
        t_gt = np.array(ann["objTrans"], dtype=np.float64)
        rest  = np.array(ann["objCorners3DRest"], dtype=np.float64)
        gt_corners = np.array(ann["objCorners3D"], dtype=np.float64)
        gt.append({"R": R_gt, "t": t_gt, "rest": rest, "corners": gt_corners})
    return gt


# ---------------------------------------------------------------------------
# Per-segment evaluation
# ---------------------------------------------------------------------------

def _eval_segment(
    pred_np: np.ndarray,    # (T, 69)
    gt_list: list[dict],
    plan: dict,
    model: CrossAttentionAdaLNZero,
    device: str,
    horizon: int,
) -> dict[str, list[dict]]:
    T = min(len(pred_np), len(gt_list))
    pred_t = torch.from_numpy(pred_np[:T]).to(device)
    metrics: dict[str, list[dict]] = {"hoisdf": [], "refined": []}

    model.eval()
    with torch.no_grad():
        for t in range(horizon - 1, T):
            window = pred_t[t - horizon + 1 : t + 1].unsqueeze(0)  # (1, H, 69)
            output = model(window, [plan])
            delta  = output.pose_delta[0, -1].cpu().numpy()         # (69,)

            gt = gt_list[t]
            for name, pose in [("hoisdf", pred_np[t]), ("refined", pred_np[t] + delta)]:
                # pred stores object as axis-angle (63:66) + translation (66:69)
                pred_R = _rot_matrix_from_aa(pose[63:66].astype(np.float64))
                pred_t_obj = pose[66:69].astype(np.float64)
                pred_corners = _transform_corners(pred_R, pred_t_obj, gt["rest"])

                metrics[name].append({
                    "oce":  _oce(pred_t_obj, gt["t"]),
                    "mce":  _mce(pred_corners, gt["corners"]),
                    "adds": _adds(pred_corners, gt["corners"]),
                })

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = json.loads((ROOT / "dataset.json").read_text())
    default_ho3d_root = Path(cfg["ho3d"]["root"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trained state dict (.pt)")
    parser.add_argument("--ho3d-root", type=Path, default=default_ho3d_root)
    parser.add_argument("--horizon",   type=int,  default=5)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ho3d_test = args.ho3d_root / "test"

    model = CrossAttentionAdaLNZero().to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_metrics: dict[str, list[dict]] = {"hoisdf": [], "refined": []}
    seq_metrics: dict[str, dict[str, list[dict]]] = {}

    plan_files = sorted(PLANS_DIR.glob("*.json"))
    if not plan_files:
        raise SystemExit(f"No plan files found in {PLANS_DIR}")

    for plan_file in plan_files:
        plan_data = json.loads(plan_file.read_text())
        seq = plan_data["sequence"]
        seq_metrics[seq] = {"hoisdf": [], "refined": []}

        for seg in plan_data["segments"]:
            seg_idx  = seg["segment_index"]
            pred_dir = PREDS_DIR / seq / f"seg_{seg_idx}"
            if not (pred_dir / "pred_mano.json").exists():
                continue

            rgb_dir = args.ho3d_root / "test" / seq / "rgb"
            frames  = [
                i for i in range(seg["frame_start"], seg["frame_end"])
                if (rgb_dir / f"{i:04d}.png").exists()
            ]
            if len(frames) < args.horizon:
                continue

            pred_np = _load_pred_features(pred_dir)
            gt_list = _load_gt_object(ho3d_test, seq, frames)
            T = min(len(pred_np), len(gt_list))
            if T < args.horizon:
                continue

            seg_m = _eval_segment(
                pred_np[:T], gt_list[:T], seg["plan"],
                model, args.device, args.horizon,
            )
            n = len(seg_m["hoisdf"])
            print(f"  {seq}/seg_{seg_idx}: {n} frames evaluated")

            for k in ("hoisdf", "refined"):
                all_metrics[k].extend(seg_m[k])
                seq_metrics[seq][k].extend(seg_m[k])

    # ---------------------------------------------------------------------------
    # Per-sequence table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Sequence':<12}  {'OCE-H':>7} {'OCE-R':>7}  {'MCE-H':>7} {'MCE-R':>7}  {'ADDS-H':>7} {'ADDS-R':>7}")
    print("-" * 72)
    for seq in sorted(seq_metrics):
        for k in ("hoisdf", "refined"):
            if not seq_metrics[seq][k]:
                seq_metrics[seq][k] = [{"oce": float("nan"), "mce": float("nan"), "adds": float("nan")}]
        h = seq_metrics[seq]["hoisdf"]
        r = seq_metrics[seq]["refined"]
        h_oce  = np.nanmean([m["oce"]  for m in h])
        r_oce  = np.nanmean([m["oce"]  for m in r])
        h_mce  = np.nanmean([m["mce"]  for m in h])
        r_mce  = np.nanmean([m["mce"]  for m in r])
        h_adds = np.nanmean([m["adds"] for m in h])
        r_adds = np.nanmean([m["adds"] for m in r])
        print(f"{seq:<12}  {h_oce:>7.2f} {r_oce:>7.2f}  {h_mce:>7.2f} {r_mce:>7.2f}  {h_adds:>7.2f} {r_adds:>7.2f}")

    # ---------------------------------------------------------------------------
    # Overall table
    # ---------------------------------------------------------------------------
    print("=" * 72)
    total_frames = len(all_metrics["hoisdf"])
    print(f"\nTotal frames evaluated: {total_frames}")
    print(f"\n{'Metric':<8}  {'HOISDF':>10}  {'Refined':>10}  {'Delta':>10}")
    print("-" * 44)
    for metric in ("oce", "mce", "adds"):
        h_vals = [m[metric] for m in all_metrics["hoisdf"]]
        r_vals = [m[metric] for m in all_metrics["refined"]]
        h_mean = float(np.mean(h_vals))
        r_mean = float(np.mean(r_vals))
        print(f"{metric.upper():<8}  {h_mean:>10.2f}  {r_mean:>10.2f}  {r_mean - h_mean:>+10.2f}")
    print("\n(all values in mm; H = HOISDF baseline, R = Refined)")


if __name__ == "__main__":
    main()
