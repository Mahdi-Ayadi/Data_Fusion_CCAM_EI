from __future__ import annotations

from math import atan2, pi, sqrt

import numpy as np

from .models import Quaternion

MIN_VARIANCE = 1e-6


def quat_to_yaw(q: Quaternion) -> float:
    """Extract yaw (heading around Z) from a quaternion, in radians."""
    R = q.rotation_matrix
    yaw = atan2(R[1, 0], R[0, 0])
    if yaw > pi:
        yaw -= 2 * pi
    if yaw <= -pi:
        yaw += 2 * pi
    return yaw


def yaw_to_quat(yaw: float) -> Quaternion:
    """Build a quaternion representing a pure yaw about Z axis."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return Quaternion(cy, 0.0, 0.0, sy)


def circular_mean(yaws: np.ndarray) -> float:
    """Mean of angles in radians, robust to wrap-around."""
    s = np.mean(np.sin(yaws))
    c = np.mean(np.cos(yaws))
    return atan2(s, c)


def angle_diff(a: float, b: float) -> float:
    """Return the wrapped difference a-b in [-pi, pi)."""
    d = a - b
    while d >= pi:
        d -= 2 * pi
    while d < -pi:
        d += 2 * pi
    return d


def sigma_from_uncertainties(obj: "Object3d", index: int, default: float = 1.0) -> float:
    """Extract sigma value at provided index from object's uncertainties."""
    from .models import Object3d  # local import to avoid circular dependency

    if isinstance(obj.uncertainties, (list, tuple)) and len(obj.uncertainties) > index:
        try:
            sigma = float(obj.uncertainties[index])
        except (TypeError, ValueError):
            sigma = default
        if sigma <= 0.0:
            return default
        return sigma
    return default


def weighted_average(v1: float, var1: float, v2: float, var2: float) -> float:
    """Variance-weighted estimate (inverse-variance weighting)."""
    w1 = 0.0 if var1 <= MIN_VARIANCE else 1.0 / var1
    w2 = 0.0 if var2 <= MIN_VARIANCE else 1.0 / var2
    if w1 + w2 == 0.0:
        return 0.5 * (v1 + v2)
    return (w1 * v1 + w2 * v2) / (w1 + w2)


def fused_sigma(var1: float, var2: float) -> float:
    """Variance of fused measurement under inverse-variance weighting."""
    w1 = 0.0 if var1 <= MIN_VARIANCE else 1.0 / var1
    w2 = 0.0 if var2 <= MIN_VARIANCE else 1.0 / var2
    if w1 + w2 == 0.0:
        return float(sqrt(max(var1, var2, MIN_VARIANCE)))
    fused_var = 1.0 / (w1 + w2)
    return float(sqrt(fused_var))


def weighted_circular_mean(y1: float, var1: float, y2: float, var2: float) -> float:
    """Circular mean supporting inverse-variance weighting."""
    w1 = 0.0 if var1 <= MIN_VARIANCE else 1.0 / var1
    w2 = 0.0 if var2 <= MIN_VARIANCE else 1.0 / var2
    if w1 + w2 == 0.0:
        return circular_mean(np.array([y1, y2], dtype=float))
    s = w1 * np.sin(y1) + w2 * np.sin(y2)
    c = w1 * np.cos(y1) + w2 * np.cos(y2)
    return atan2(s, c)
