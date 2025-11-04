"""Core package for data fusion experiments."""

from .models import Object3d, Quaternion
from .data_loading import load_nuscenes_gt, add_gaussian_noise, make_two_noise_levels
from .association import associate_greedy_euclidean, associate_weighted_mahalanobis
from .fusion import fuse_average_boxes, fuse_weighted_least_squares, fuse_matches
from .late_fusion import fuse_attention_boxes, load_late_fusion_model, compute_attention_features, LateFusionNet, LateFusionModel
from .tracking import run_kalman_fusion
from .evaluation import (
    evaluate_against_gt,
    evaluate_nuscenes_like,
    print_metrics,
    print_nuscenes_metrics,
)
from .plotting import plot_first_frames_global

__all__ = [
    "Object3d",
    "Quaternion",
    "load_nuscenes_gt",
    "add_gaussian_noise",
    "make_two_noise_levels",
    "associate_greedy_euclidean",
    "associate_weighted_mahalanobis",
    "fuse_average_boxes",
    "fuse_weighted_least_squares",
    "fuse_matches",
    "fuse_attention_boxes",
    "load_late_fusion_model",
    "LateFusionNet",
    "LateFusionModel",
    "compute_attention_features",
    "run_kalman_fusion",
    "evaluate_against_gt",
    "evaluate_nuscenes_like",
    "print_metrics",
    "print_nuscenes_metrics",
    "plot_first_frames_global",
]
