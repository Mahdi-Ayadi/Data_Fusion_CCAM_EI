from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .math_utils import quat_to_yaw
from .models import Object3d


def plot_first_frames_global(
    gt_by_sample: Dict[str, List[Object3d]],
    noisy_low_by_sample: Dict[str, List[Object3d]],
    noisy_high_by_sample: Dict[str, List[Object3d]],
    fused_by_sample: Optional[Dict[str, List[Object3d]]] = None,
    num_frames: int = 5,
    dist_range: Optional[float] = None,
    output_path: Optional[str] = None,
):
    """Visualize detections for the first few frames in global coordinates."""

    def box_corners_global(obj: Object3d) -> np.ndarray:
        cx, cy = obj.xdistance, obj.ydistance
        w, l = obj.width, obj.depth
        yaw = quat_to_yaw(obj.orientation)
        x_c = np.array([l / 2, l / 2, -l / 2, -l / 2])
        y_c = np.array([w / 2, -w / 2, -w / 2, w / 2])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = np.stack([x_c, y_c], axis=1) @ R.T + np.array([cx, cy])
        return corners

    sample_tokens = list(gt_by_sample.keys())[:num_frames]
    n = len(sample_tokens)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    all_xy = []
    for token in sample_tokens:
        for objs in gt_by_sample[token]:
            all_xy.append([objs.xdistance, objs.ydistance])
    all_xy = np.array(all_xy)
    if dist_range is None and all_xy.size > 0:
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)
        pad = 10.0
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min - pad, y_max + pad)
    else:
        first_objs = gt_by_sample[sample_tokens[0]]
        if len(first_objs):
            cx = np.mean([o.xdistance for o in first_objs])
            cy = np.mean([o.ydistance for o in first_objs])
        else:
            cx = cy = 0.0
        dr = dist_range or 50.0
        xlim = (cx - dr, cx + dr)
        ylim = (cy - dr, cy + dr)

    for idx, token in enumerate(sample_tokens):
        ax = axes[idx]
        gt_objs = gt_by_sample.get(token, [])
        noisy_low = noisy_low_by_sample.get(token, [])
        noisy_high = noisy_high_by_sample.get(token, [])
        fused = fused_by_sample.get(token, []) if fused_by_sample else []

        for o in gt_objs:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False, edgecolor="blue", linewidth=1.0, label="GT"))
        for o in noisy_low:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False, edgecolor="green", linewidth=0.8, label="Low noise"))
        for o in noisy_high:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False, edgecolor="red", linewidth=0.8, label="High noise"))
        for o in fused:
            ax.add_patch(Polygon(box_corners_global(o), closed=True, fill=False, edgecolor="orange", linestyle="--", linewidth=1.2, label="Fused"))

        ax.set_title(f"Frame {idx + 1}\n{token[:6]}...", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Global X [m]")
        ax.set_ylabel("Global Y [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="upper right", ncol=4, frameon=False)
    fig.suptitle("Top-down global visualization (GT / Noisy / Fused)", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)
