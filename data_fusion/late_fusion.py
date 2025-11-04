from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .math_utils import angle_diff, sigma_from_uncertainties, weighted_circular_mean, yaw_to_quat, quat_to_yaw
from .models import Object3d


def sensor_reliability(obj: Object3d) -> float:
    sigmas = [
        sigma_from_uncertainties(obj, 0, default=0.5),
        sigma_from_uncertainties(obj, 1, default=0.5),
        sigma_from_uncertainties(obj, 2, default=0.5),
        sigma_from_uncertainties(obj, 4, default=0.5),
        sigma_from_uncertainties(obj, 5, default=0.5),
        sigma_from_uncertainties(obj, 6, default=0.5),
        sigma_from_uncertainties(obj, 3, default=np.deg2rad(5.0)),
    ]
    return 1.0 / (np.mean([s * s for s in sigmas]) + 1e-6)


def compute_attention_features(o1: Object3d, o2: Object3d) -> np.ndarray:
    rel1 = sensor_reliability(o1)
    rel2 = sensor_reliability(o2)
    dx = o1.xdistance - o2.xdistance
    dy = o1.ydistance - o2.ydistance
    dz = o1.zdistance - o2.zdistance
    yaw_delta = abs(angle_diff(quat_to_yaw(o1.orientation), quat_to_yaw(o2.orientation)))
    return np.array(
        [
            rel1,
            rel2,
            rel1 - rel2,
            np.sqrt(dx * dx + dy * dy + dz * dz),
            yaw_delta,
            1.0,  # bias
        ],
        dtype=np.float32,
    )


class LateFusionNet(nn.Module):
    """Lightweight MLP producing a gating weight between 0 and 1."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


@dataclass
class LateFusionModel:
    net: LateFusionNet
    device: torch.device

    def gate(self, features: np.ndarray) -> float:
        tensor = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            gate = self.net(tensor.unsqueeze(0)).squeeze(0).item()
        return float(np.clip(gate, 0.05, 0.95))


def load_late_fusion_model(
    checkpoint_path: Optional[Path],
    device: Optional[torch.device] = None,
) -> Optional[LateFusionModel]:
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    device = device or torch.device("cpu")
    data = torch.load(checkpoint_path, map_location=device)
    model = LateFusionNet(input_dim=data["input_dim"], hidden_dim=data["hidden_dim"])
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    return LateFusionModel(net=model, device=device)


def fuse_attention_boxes(
    o1: Object3d,
    o2: Object3d,
    model: Optional[LateFusionModel] = None,
) -> Object3d:
    features = compute_attention_features(o1, o2)

    if model is not None:
        gate = model.gate(features)
    else:
        weights = np.array([1.5, -1.5, 1.0, -0.8, -0.6, 0.0], dtype=np.float32)
        gate = 1.0 / (1.0 + np.exp(-np.dot(weights, features)))
        gate = float(np.clip(gate, 0.05, 0.95))

    fused_x = gate * o1.xdistance + (1 - gate) * o2.xdistance
    fused_y = gate * o1.ydistance + (1 - gate) * o2.ydistance
    fused_z = gate * o1.zdistance + (1 - gate) * o2.zdistance
    fused_w = gate * o1.width + (1 - gate) * o2.width
    fused_l = gate * o1.depth + (1 - gate) * o2.depth
    fused_h = gate * o1.height + (1 - gate) * o2.height

    yaw1 = quat_to_yaw(o1.orientation)
    yaw2 = quat_to_yaw(o2.orientation)
    # Weighted circular mean using learned gate
    fused_yaw = weighted_circular_mean(yaw1, 1.0 / (gate + 1e-6), yaw2, 1.0 / (1 - gate + 1e-6))
    fused_orientation = yaw_to_quat(fused_yaw)

    if isinstance(o1.uncertainties, list) and isinstance(o2.uncertainties, list):
        len_min = min(len(o1.uncertainties), len(o2.uncertainties))
        fused_unc = [
            gate * o1.uncertainties[k] + (1 - gate) * o2.uncertainties[k]
            for k in range(len_min)
        ]
    else:
        fused_unc = o1.uncertainties or o2.uncertainties

    classification = o1.classification if sensor_reliability(o1) >= sensor_reliability(o2) else o2.classification

    return Object3d(
        id_frame=o1.id_fame,
        instance_token=o1.instance_token or o2.instance_token,
        object_id=o1.object_id or o2.object_id,
        timestamp=o1.timestamp,
        xdistance=fused_x,
        ydistance=fused_y,
        zdistance=fused_z,
        width=fused_w,
        depth=fused_l,
        height=fused_h,
        orientation=fused_orientation,
        classification=classification,
        uncertainties=fused_unc,
        dt=o1.dt,
    )
