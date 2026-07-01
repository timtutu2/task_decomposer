#!/usr/bin/env python3
"""Train CrossAttentionAdaLNZero on HO3D segments.

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 64 --lr 3e-4
    python train.py --checkpoint checkpoints/v3_without_pool/epoch_100.pt  # resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ca_block import CrossAttentionAdaLNZero
from ca_block.pose_adapter import _rotation_matrix_to_axis_angle


def _aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Batch axis-angle (..., 3) → rotation matrix (..., 3, 3) via Rodrigues."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    axis  = aa / angle
    cos, sin = angle.cos(), angle.sin()
    x, y, z  = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    K = K.reshape(*aa.shape[:-1], 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    return cos[..., None] * I + sin[..., None] * K + (1 - cos)[..., None] * outer


def _geodesic_loss(pred_aa: torch.Tensor, gt_aa: torch.Tensor) -> torch.Tensor:
    """Mean geodesic rotation error between two sets of axis-angle vectors (..., 3).

    Uses atan2(sin θ, cos θ) instead of acos(cos θ) to avoid the infinite
    gradient of acos at ±1 (i.e. when predicted and GT rotations are identical
    or antipodal), which otherwise causes NaN loss from the very first batch.
    """
    R_pred = _aa_to_rotmat(pred_aa.float())
    R_gt   = _aa_to_rotmat(gt_aa.float())
    R_diff = R_pred.transpose(-1, -2) @ R_gt
    # cos θ = (trace − 1) / 2
    cos_a = ((R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    # sin θ = ‖R − Rᵀ‖_F / (2√2)  because R − Rᵀ = 2 sin θ · [axis]×  and ‖[axis]×‖_F = √2
    # ε inside sqrt keeps the gradient finite when the skew part is zero (identical rotations).
    diff  = R_diff - R_diff.transpose(-1, -2)
    sin_a = (diff.pow(2).sum(dim=(-2, -1)) + 1e-12).sqrt() / (2.0 * 2.0 ** 0.5)
    return torch.atan2(sin_a, cos_a).mean()

PLANS_DIR = ROOT / "preprocess" / "plans"       / "ho3d" / "train"
PREDS_DIR = ROOT / "preprocess" / "hoisdf_preds" / "ho3d" / "train"


def load_pred_features(pred_dir: Path) -> np.ndarray:
    """Load HOISDF predictions for a segment as (T, 69) float32 array."""
    with open(pred_dir / "pred_mano.json") as f:
        mano = json.load(f)
    with open(pred_dir / "pred_object.json") as f:
        objects = json.load(f)

    joints_list = mano[0]
    T = min(len(joints_list), len(objects))
    rows = []
    for i in range(T):
        hand  = [c for j in joints_list[i] for c in j]          # 63
        rot   = _rotation_matrix_to_axis_angle(objects[i]["rot"]) # 3
        trans = objects[i]["trans"]                               # 3
        rows.append(hand + rot + trans)
    return np.array(rows, dtype=np.float32)


def load_gt_features(ho3d_root: Path, seq_name: str, frames: list[int]) -> np.ndarray:
    """Load GT hand+object poses from HO3D pkls as (T, 69) float32 array.

    GT is in the same coordinate convention as the stored HOISDF predictions
    (HO3D space after COORD_CHANGE is applied in decode_batch).
    """
    rows = []
    for f in frames:
        ann      = np.load(ho3d_root / "train" / seq_name / "meta" / f"{f:04d}.pkl",
                           allow_pickle=True)
        hand     = np.array(ann["handJoints3D"], dtype=np.float32).flatten()  # 63
        obj_rot  = np.array(ann["objRot"],        dtype=np.float32).flatten() # 3 axis-angle
        obj_trans= np.array(ann["objTrans"],      dtype=np.float32).flatten() # 3
        rows.append(np.concatenate([hand, obj_rot, obj_trans]))
    return np.array(rows, dtype=np.float32)


class SegmentDataset(Dataset):
    def __init__(self, plans_dir: Path, preds_dir: Path, ho3d_root: Path, horizon: int = 5):
        self.horizon = horizon
        self._windows: list[tuple[torch.Tensor, torch.Tensor, dict]] = []

        plan_files = sorted(plans_dir.glob("*.json"))
        print(f"Loading {len(plan_files)} plan files…")

        for plan_file in plan_files:
            plan_data = json.loads(plan_file.read_text())
            seq_name  = plan_data["sequence"]

            for seg in plan_data["segments"]:
                seg_idx  = seg["segment_index"]
                pred_dir = preds_dir / seq_name / f"seg_{seg_idx}"
                if not (pred_dir / "pred_mano.json").exists():
                    continue

                # Reconstruct the frame list the same way run_hoisdf_inference.py did
                rgb_dir = ho3d_root / "train" / seq_name / "rgb"
                frames  = [
                    i for i in range(seg["frame_start"], seg["frame_end"])
                    if (rgb_dir / f"{i:04d}.png").exists()
                ]
                if not frames:
                    continue

                pred_np = load_pred_features(pred_dir)
                gt_np   = load_gt_features(ho3d_root, seq_name, frames)

                T = min(len(pred_np), len(gt_np))
                if T < horizon:
                    continue

                pred_t = torch.from_numpy(pred_np[:T])
                gt_t   = torch.from_numpy(gt_np[:T])

                for t in range(T - horizon + 1):
                    self._windows.append((
                        pred_t[t : t + horizon],
                        gt_t[t : t + horizon],
                        seg["plan"],
                    ))

        print(f"Dataset ready: {len(self._windows)} windows")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        return self._windows[idx]


def collate_fn(batch):
    preds = torch.stack([b[0] for b in batch])  # (B, H, 69)
    gts   = torch.stack([b[1] for b in batch])  # (B, H, 69)
    plans = [b[2] for b in batch]
    return preds, gts, plans


def main() -> None:
    dataset_cfg = json.loads((ROOT / "dataset.json").read_text())
    default_ho3d_root = Path(dataset_cfg["ho3d"]["root"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",     type=Path, default=ROOT / "train_config.json",
                        help="Training config JSON (default: train_config.json)")
    parser.add_argument("--ho3d-root",  type=Path, default=default_ho3d_root)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size from the training config")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    horizon    = cfg["horizon"]
    epochs     = cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    lr         = cfg["lr"]
    save_every = cfg["save_every"]
    output     = ROOT / "checkpoints" / cfg["output"]
    rot_weight   = cfg["rot_weight"]
    trans_weight = cfg["trans_weight"]

    print(f"Config: {args.config}")
    for k, v in cfg.items():
        if k == "batch_size":
            v = batch_size
        print(f"  {k}: {v}")

    output.mkdir(parents=True, exist_ok=True)

    dataset = SegmentDataset(PLANS_DIR, PREDS_DIR, args.ho3d_root, horizon)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # data is pre-loaded into RAM; workers would copy tensors
        collate_fn=collate_fn,
    )

    model = CrossAttentionAdaLNZero().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    best_loss   = float("inf")

    if args.checkpoint:
        ckpt_data = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if isinstance(ckpt_data, dict) and "model" in ckpt_data:
            model.load_state_dict(ckpt_data["model"])
            optimizer.load_state_dict(ckpt_data["optimizer"])
            start_epoch = ckpt_data["epoch"] + 1
            best_loss   = ckpt_data.get("best_loss", float("inf"))
            print(f"Resumed from {args.checkpoint} (epoch {ckpt_data['epoch']}, best_loss={best_loss:.6f})")
        else:
            # legacy checkpoint — state dict only
            model.load_state_dict(ckpt_data)
            print(f"Resumed from {args.checkpoint} (legacy format, epoch unknown — starting epoch counter at 1)")

    def pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hand_loss  = ((pred[..., :63]  - target[..., :63])  ** 2).mean()
        rot_loss   = _geodesic_loss(pred[..., 63:66], target[..., 63:66])
        trans_loss = ((pred[..., 66:69] - target[..., 66:69]) ** 2).mean()
        return hand_loss + rot_weight * rot_loss + trans_weight * trans_loss

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0

        for pred, gt, plans in loader:
            pred = pred.to(args.device)  # (B, H, 69)
            gt   = gt.to(args.device)    # (B, H, 69)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=pred.device.type == "cuda",
            ):
                model_out = model(pred, plans)
                refined   = pred + model_out.pose_delta
                loss = pose_loss(refined, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        is_best  = avg_loss < best_loss

        if is_best:
            best_loss = avg_loss

        print(f"Epoch {epoch:03d}/{epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}" + (" *" if is_best else ""))

        bundle = {
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "epoch":      epoch,
            "best_loss":  best_loss,
        }

        if is_best:
            torch.save(bundle, output / "best.pt")

        if epoch % save_every == 0:
            ckpt = output / f"epoch_{epoch:03d}.pt"
            torch.save(bundle, ckpt)
            print(f"  → saved {ckpt}")

    final = output / "final.pt"
    torch.save(bundle, final)
    print(f"Done. Final checkpoint → {final}  (best_loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
