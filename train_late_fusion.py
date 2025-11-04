from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from data_fusion.association import associate_weighted_mahalanobis
from data_fusion.data_loading import load_nuscenes_gt, make_two_noise_levels
from data_fusion.late_fusion import LateFusionNet, compute_attention_features
from data_fusion.math_utils import quat_to_yaw
from data_fusion.models import Object3d


def object_to_vector(obj: Object3d) -> np.ndarray:
    return np.array(
        [
            obj.xdistance,
            obj.ydistance,
            obj.zdistance,
            obj.width,
            obj.depth,
            obj.height,
            quat_to_yaw(obj.orientation),
        ],
        dtype=np.float32,
    )


class LateFusionDataset(Dataset):
    def __init__(
        self,
        gt_by_sample: Dict[str, List[Object3d]],
        noisy_low_by_sample: Dict[str, List[Object3d]],
        noisy_high_by_sample: Dict[str, List[Object3d]],
        matches_by_sample: Dict[str, List[Tuple[int, int]]],
    ):
        self.features = []
        self.sensor1 = []
        self.sensor2 = []
        self.gt = []

        for sample_token, matches in matches_by_sample.items():
            gt_lookup = {obj.object_id: obj for obj in gt_by_sample.get(sample_token, [])}
            low = noisy_low_by_sample[sample_token]
            high = noisy_high_by_sample[sample_token]
            for idx_low, idx_high in matches:
                o1 = low[idx_low]
                o2 = high[idx_high]
                gt_obj = gt_lookup.get(o1.object_id)
                if gt_obj is None:
                    continue

                self.features.append(compute_attention_features(o1, o2))
                self.sensor1.append(object_to_vector(o1))
                self.sensor2.append(object_to_vector(o2))
                self.gt.append(object_to_vector(gt_obj))

        if not self.features:
            raise ValueError(
                "LateFusionDataset received no matched pairs. "
                "Check association thresholds or dataset fraction."
            )

        self.features = np.stack(self.features)
        self.sensor1 = np.stack(self.sensor1)
        self.sensor2 = np.stack(self.sensor2)
        self.gt = np.stack(self.gt)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.sensor1[idx]),
            torch.from_numpy(self.sensor2[idx]),
            torch.from_numpy(self.gt[idx]),
        )


def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def train_model(
    dataset: LateFusionDataset,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[LateFusionNet, dict]:
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two samples to create train/val splits.")

    val_size = max(1, int(len(dataset) * val_split))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = LateFusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for feats, s1, s2, gt in progress:
            feats = feats.to(device)
            s1 = s1.to(device)
            s2 = s2.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            gate = model(feats).squeeze(1)

            fused_lin = gate.unsqueeze(1) * s1[:, :6] + (1 - gate).unsqueeze(1) * s2[:, :6]
            pos_loss = torch.mean((fused_lin - gt[:, :6]) ** 2)

            yaw1 = s1[:, 6]
            yaw2 = s2[:, 6]
            gt_yaw = gt[:, 6]
            c_fused = gate * torch.cos(yaw1) + (1 - gate) * torch.cos(yaw2)
            s_fused = gate * torch.sin(yaw1) + (1 - gate) * torch.sin(yaw2)
            fused_yaw = torch.atan2(s_fused, c_fused)
            yaw_loss = torch.mean(angle_diff_torch(fused_yaw, gt_yaw) ** 2)

            loss = pos_loss + 0.5 * yaw_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
        progress.close()

        avg_train = running_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
            for feats, s1, s2, gt in progress_val:
                feats = feats.to(device)
                s1 = s1.to(device)
                s2 = s2.to(device)
                gt = gt.to(device)

                gate = model(feats).squeeze(1)
                fused_lin = gate.unsqueeze(1) * s1[:, :6] + (1 - gate).unsqueeze(1) * s2[:, :6]
                pos_loss = torch.mean((fused_lin - gt[:, :6]) ** 2)
                yaw1 = s1[:, 6]
                yaw2 = s2[:, 6]
                gt_yaw = gt[:, 6]
                c_fused = gate * torch.cos(yaw1) + (1 - gate) * torch.cos(yaw2)
                s_fused = gate * torch.sin(yaw1) + (1 - gate) * torch.sin(yaw2)
                fused_yaw = torch.atan2(s_fused, c_fused)
                yaw_loss = torch.mean(angle_diff_torch(fused_yaw, gt_yaw) ** 2)
                loss = pos_loss + 0.5 * yaw_loss
                val_loss += loss.item() * feats.size(0)
            progress_val.close()

        avg_val = val_loss / val_size
        print(f"Epoch {epoch:02d} | train={avg_train:.6f} | val={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                "input_dim": model.net[0].in_features,
                "hidden_dim": model.net[0].out_features,
                "state_dict": model.state_dict(),
                "val_loss": best_val,
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model.")
    model.load_state_dict(best_state["state_dict"])
    model.eval()
    return model, best_state


def prepare_dataset(
    ann_root: str,
    fraction: float,
    sigmas_low: dict,
    sigmas_high: dict,
) -> Tuple[LateFusionDataset, Dict[str, List[Object3d]], Dict[str, List[Object3d]], Dict[str, List[Object3d]], Dict[str, List[Tuple[int, int]]]]:
    gt_by_sample = load_nuscenes_gt(ann_root, prefer_channel="LIDAR_TOP", fraction=fraction)
    noisy_per_sample = make_two_noise_levels(gt_by_sample, sigmas_low, sigmas_high, seed=42)

    noisy_low_by_sample = {}
    noisy_high_by_sample = {}
    matches_weighted_by_sample = {}

    for sample_token, (noisy_low, noisy_high) in noisy_per_sample.items():
        noisy_low_by_sample[sample_token] = noisy_low
        noisy_high_by_sample[sample_token] = noisy_high
        matches_wls, _, _ = associate_weighted_mahalanobis(noisy_low, noisy_high, dist_thresh=3.0)
        matches_weighted_by_sample[sample_token] = matches_wls

    dataset = LateFusionDataset(gt_by_sample, noisy_low_by_sample, noisy_high_by_sample, matches_weighted_by_sample)
    return dataset, gt_by_sample, noisy_low_by_sample, noisy_high_by_sample, matches_weighted_by_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IA-inspired late fusion network.")
    parser.add_argument("--ann-root", type=str, default=None, help="Path to nuScenes annotations.")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of samples to use for speed.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="models/late_fusion.pth")
    args = parser.parse_args()

    default_ann_root = Path(__file__).resolve().parent / "dataset" / "nuscenes" / "v1.0-trainval"
    ann_root = args.ann_root or os.environ.get("NUSCENES_ANN_ROOT", str(default_ann_root))
    if not os.path.isdir(ann_root):
        raise FileNotFoundError(
            f"NuScenes annotations not found at '{ann_root}'. "
            "Set --ann-root or NUSCENES_ANN_ROOT to the correct location."
        )

    sigmas_low = dict(x=0.20, y=0.20, z=0.20, yaw=np.deg2rad(0.05), w=0.5, l=0.5, h=0.5)
    sigmas_high = dict(x=0.50, y=0.50, z=0.50, yaw=np.deg2rad(0.50), w=1.00, l=1.00, h=1.00)

    dataset, *_ = prepare_dataset(ann_root, args.fraction, sigmas_low, sigmas_high)
    if len(dataset) == 0:
        raise RuntimeError("No training pairs found; check dataset path or fraction.")
    print(f"Prepared {len(dataset)} training samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_state = train_model(dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"Saved trained late-fusion model to {output_path} (val_loss={best_state['val_loss']:.6f})")


if __name__ == "__main__":
    main()
