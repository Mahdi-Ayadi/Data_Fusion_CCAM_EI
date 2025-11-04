from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from data_fusion.association import associate_greedy_euclidean, associate_weighted_mahalanobis
from data_fusion.data_loading import load_nuscenes_gt, make_two_noise_levels
from data_fusion.evaluation import (
    evaluate_against_gt,
    evaluate_nuscenes_like,
    print_metrics,
    print_nuscenes_metrics,
)
from data_fusion.fusion import fuse_average_boxes, fuse_matches, fuse_weighted_least_squares
from data_fusion.late_fusion import fuse_attention_boxes, load_late_fusion_model
from data_fusion.plotting import plot_first_frames_global
from data_fusion.tracking import run_kalman_fusion


def main() -> None:
    # Path to nuScenes json folder (contains sample.json, sample_annotation.json, sample_data.json)
    default_ann_root = Path(__file__).resolve().parent / "dataset" / "nuscenes" / "v1.0-trainval"
    ann_root = os.environ.get("NUSCENES_ANN_ROOT", str(default_ann_root))
    if not os.path.isdir(ann_root):
        raise FileNotFoundError(
            f"NuScenes annotations not found at '{ann_root}'. "
            "Set the NUSCENES_ANN_ROOT environment variable to the correct location."
        )

    # 1) Load GT
    print("#------------------------------------------")
    print("1 - Loading data from ground truth nuScenes")
    gt_by_sample = load_nuscenes_gt(
        ann_root,
        prefer_channel="LIDAR_TOP",
        fraction=0.05,  # subsample to keep the demo fast
    )

    # 2) Create two noisy datasets with different uncertainty levels
    print("#------------------------------------------")
    print("2 - Applying noise to simulate 2 sensors")
    sigmas_low = dict(x=0.20, y=0.20, z=0.20, yaw=np.deg2rad(0.05), w=0.5, l=0.5, h=0.5)
    sigmas_high = dict(x=0.50, y=0.50, z=0.50, yaw=np.deg2rad(0.50), w=1.00, l=1.00, h=1.00)

    noisy_per_sample = make_two_noise_levels(gt_by_sample, sigmas_low, sigmas_high, seed=42)

    # 3) For each sample, associate low vs high noise and fuse with multiple strategies
    print("#------------------------------------------")
    print("3 - Association and Fusion Algorithms")
    fused_baseline = {}
    fused_weighted = {}
    fused_late = {}
    matches_weighted_by_sample = {}
    noisy_low_by_sample = {}
    noisy_high_by_sample = {}

    sample_times = {}
    for token, objs in gt_by_sample.items():
        ts_list = [o.timestamp for o in objs if not np.isnan(o.timestamp)]
        sample_times[token] = min(ts_list) if ts_list else np.nan

    ordered_tokens = sorted(
        noisy_per_sample.keys(),
        key=lambda tok: sample_times.get(tok, float("inf")),
    )

    model_path = Path(__file__).resolve().parent / "models" / "late_fusion.pth"
    late_model = load_late_fusion_model(model_path)
    if late_model:
        print(f"Loaded trained late-fusion model from {model_path}")
    else:
        print("No trained late-fusion model found; using heuristic gating.")

    late_fuse_fn = (
        (lambda a, b: fuse_attention_boxes(a, b, model=late_model))
        if late_model
        else fuse_attention_boxes
    )

    for sample_token in ordered_tokens:
        noisy_low, noisy_high = noisy_per_sample[sample_token]
        noisy_low_by_sample[sample_token] = noisy_low
        noisy_high_by_sample[sample_token] = noisy_high

        matches_base, _, _ = associate_greedy_euclidean(noisy_low, noisy_high, dist_thresh=5.0)
        fused_baseline[sample_token] = fuse_matches(noisy_low, noisy_high, matches_base, fuse_average_boxes)

        matches_wls, _, _ = associate_weighted_mahalanobis(noisy_low, noisy_high, dist_thresh=3.0)
        matches_weighted_by_sample[sample_token] = matches_wls
        fused_weighted[sample_token] = fuse_matches(noisy_low, noisy_high, matches_wls, fuse_weighted_least_squares)
        fused_late[sample_token] = fuse_matches(noisy_low, noisy_high, matches_wls, late_fuse_fn)

    fused_kalman = run_kalman_fusion(
        ordered_tokens,
        noisy_low_by_sample,
        noisy_high_by_sample,
        matches_weighted_by_sample,
    )

    # 4) Evaluate and plot results
    print("#------------------------------------------")
    print("4 - Evaluate and plot results")

    baseline_metrics = evaluate_against_gt(fused_baseline, gt_by_sample)
    weighted_metrics = evaluate_against_gt(fused_weighted, gt_by_sample)
    kalman_metrics = evaluate_against_gt(fused_kalman, gt_by_sample)
    late_metrics = evaluate_against_gt(fused_late, gt_by_sample)

    print_metrics("Baseline (avg fusion)", baseline_metrics)
    print_metrics("Weighted LS fusion", weighted_metrics)
    print_metrics("Kalman fusion", kalman_metrics)
    print_metrics("Late fusion (attention)", late_metrics)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_first_frames_global(
        gt_by_sample=gt_by_sample,
        noisy_low_by_sample=noisy_low_by_sample,
        noisy_high_by_sample=noisy_high_by_sample,
        fused_by_sample=fused_baseline,
        num_frames=2,
        dist_range=100,
        output_path=str(output_dir / "fusion_baseline.png"),
    )
    plot_first_frames_global(
        gt_by_sample=gt_by_sample,
        noisy_low_by_sample=noisy_low_by_sample,
        noisy_high_by_sample=noisy_high_by_sample,
        fused_by_sample=fused_weighted,
        num_frames=2,
        dist_range=100,
        output_path=str(output_dir / "fusion_weighted.png"),
    )
    plot_first_frames_global(
        gt_by_sample=gt_by_sample,
        noisy_low_by_sample=noisy_low_by_sample,
        noisy_high_by_sample=noisy_high_by_sample,
        fused_by_sample=fused_kalman,
        num_frames=2,
        dist_range=100,
        output_path=str(output_dir / "fusion_kalman.png"),
    )
    plot_first_frames_global(
        gt_by_sample=gt_by_sample,
        noisy_low_by_sample=noisy_low_by_sample,
        noisy_high_by_sample=noisy_high_by_sample,
        fused_by_sample=fused_late,
        num_frames=2,
        dist_range=100,
        output_path=str(output_dir / "fusion_late.png"),
    )

    print("Plots exported to:")
    print(f"  - {output_dir / 'fusion_baseline.png'}")
    print(f"  - {output_dir / 'fusion_weighted.png'}")
    print(f"  - {output_dir / 'fusion_kalman.png'}")
    print(f"  - {output_dir / 'fusion_late.png'}")

    # 5) nuScenes-style metrics
    print("#------------------------------------------")
    print("5 - nuScenes-style metrics")

    metrics_baseline_ns = evaluate_nuscenes_like(
        gt_by_sample,
        fused_baseline,
        dist_thresholds=(0.5, 1.0, 2.0, 4.0),
    )
    metrics_weighted_ns = evaluate_nuscenes_like(
        gt_by_sample,
        fused_weighted,
        dist_thresholds=(0.5, 1.0, 2.0, 4.0),
    )
    metrics_kalman_ns = evaluate_nuscenes_like(
        gt_by_sample,
        fused_kalman,
        dist_thresholds=(0.5, 1.0, 2.0, 4.0),
    )
    metrics_late_ns = evaluate_nuscenes_like(
        gt_by_sample,
        fused_late,
        dist_thresholds=(0.5, 1.0, 2.0, 4.0),
    )

    print_nuscenes_metrics("Baseline (avg fusion)", metrics_baseline_ns)
    print_nuscenes_metrics("Weighted LS fusion", metrics_weighted_ns)
    print_nuscenes_metrics("Kalman fusion", metrics_kalman_ns)
    print_nuscenes_metrics("Late fusion (attention)", metrics_late_ns)


if __name__ == "__main__":
    main()
