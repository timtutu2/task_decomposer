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
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    horizon    = cfg["horizon"]
    epochs     = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr         = cfg["lr"]
    save_every = cfg["save_every"]
    output     = ROOT / "checkpoints" / cfg["output"]
    obj_weight = cfg["obj_weight"]

    print(f"Config: {args.config}")
    for k, v in cfg.items():
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
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {args.checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # obj_weight upweights dims 63:69 (axis-angle + translation) relative to
    # the 63 hand-joint dims, which otherwise dominate the MSE gradient.
    _dim_weights = torch.ones(69)
    _dim_weights[63:] = obj_weight

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = _dim_weights.to(pred.device)
        return ((pred - target) ** 2 * w).mean()

    for epoch in range(1, epochs + 1):
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
                loss = weighted_mse(refined, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d}/{epochs}  loss={avg_loss:.6f}")

        if epoch % save_every == 0:
            ckpt = output / f"epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    final = output / "final.pt"
    torch.save(model.state_dict(), final)
    print(f"Done. Final checkpoint → {final}")


if __name__ == "__main__":
    main()
