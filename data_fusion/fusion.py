from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np

from .math_utils import (
    MIN_VARIANCE,
    fused_sigma,
    quat_to_yaw,
    sigma_from_uncertainties,
    weighted_average,
    weighted_circular_mean,
    yaw_to_quat,
)
from .models import Object3d


def fuse_average_boxes(o1: Object3d, o2: Object3d) -> Object3d:
    """Baseline fusion that averages positions, sizes, and yaw (circular mean)."""
    y1 = quat_to_yaw(o1.orientation)
    y2 = quat_to_yaw(o2.orientation)
    y_mean = np.mean([y1, y2])

    fused_unc = None
    if isinstance(o1.uncertainties, list) and isinstance(o2.uncertainties, list) and \
       len(o1.uncertainties) == len(o2.uncertainties):
        fused_unc = list((np.array(o1.uncertainties) + np.array(o2.uncertainties)) / 2.0)
    else:
        fused_unc = o1.uncertainties or o2.uncertainties

    return Object3d(
        id_frame=o1.id_fame,
        instance_token=o1.instance_token or o2.instance_token,
        object_id=f"{o1.object_id}|{o2.object_id}",
        timestamp=o1.timestamp,
        xdistance=0.5 * (o1.xdistance + o2.xdistance),
        ydistance=0.5 * (o1.ydistance + o2.ydistance),
        zdistance=0.5 * (o1.zdistance + o2.zdistance),
        width=0.5 * (o1.width + o2.width),
        depth=0.5 * (o1.depth + o2.depth),
        height=0.5 * (o1.height + o2.height),
        orientation=yaw_to_quat(y_mean),
        classification=o1.classification or o2.classification,
        uncertainties=fused_unc,
        dt=o1.dt,
    )


def fuse_weighted_least_squares(o1: Object3d, o2: Object3d) -> Object3d:
    """Fuse two detections with inverse-variance weighting (Weighted Least Squares)."""
    var_x1 = sigma_from_uncertainties(o1, 0) ** 2
    var_x2 = sigma_from_uncertainties(o2, 0) ** 2
    var_y1 = sigma_from_uncertainties(o1, 1) ** 2
    var_y2 = sigma_from_uncertainties(o2, 1) ** 2
    var_z1 = sigma_from_uncertainties(o1, 2) ** 2
    var_z2 = sigma_from_uncertainties(o2, 2) ** 2
    var_yaw1 = sigma_from_uncertainties(o1, 3, default=np.deg2rad(5.0)) ** 2
    var_yaw2 = sigma_from_uncertainties(o2, 3, default=np.deg2rad(5.0)) ** 2
    var_w1 = sigma_from_uncertainties(o1, 4) ** 2
    var_w2 = sigma_from_uncertainties(o2, 4) ** 2
    var_l1 = sigma_from_uncertainties(o1, 5) ** 2
    var_l2 = sigma_from_uncertainties(o2, 5) ** 2
    var_h1 = sigma_from_uncertainties(o1, 6) ** 2
    var_h2 = sigma_from_uncertainties(o2, 6) ** 2

    fused_x = weighted_average(o1.xdistance, var_x1, o2.xdistance, var_x2)
    fused_y = weighted_average(o1.ydistance, var_y1, o2.ydistance, var_y2)
    fused_z = weighted_average(o1.zdistance, var_z1, o2.zdistance, var_z2)

    fused_w = weighted_average(o1.width, var_w1, o2.width, var_w2)
    fused_l = weighted_average(o1.depth, var_l1, o2.depth, var_l2)
    fused_h = weighted_average(o1.height, var_h1, o2.height, var_h2)

    yaw1 = quat_to_yaw(o1.orientation)
    yaw2 = quat_to_yaw(o2.orientation)
    fused_yaw = weighted_circular_mean(yaw1, var_yaw1, yaw2, var_yaw2)
    fused_orientation = yaw_to_quat(fused_yaw)

    fused_unc = [
        fused_sigma(var_x1, var_x2),
        fused_sigma(var_y1, var_y2),
        fused_sigma(var_z1, var_z2),
        fused_sigma(var_yaw1, var_yaw2),
        fused_sigma(var_w1, var_w2),
        fused_sigma(var_l1, var_l2),
        fused_sigma(var_h1, var_h2),
    ]

    info_o1 = sum(0.0 if v <= MIN_VARIANCE else 1.0 / v for v in (var_x1, var_y1, var_z1, var_w1, var_l1, var_h1))
    info_o2 = sum(0.0 if v <= MIN_VARIANCE else 1.0 / v for v in (var_x2, var_y2, var_z2, var_w2, var_l2, var_h2))
    fused_class = (o1.classification if info_o1 >= info_o2 else o2.classification) or o1.classification or o2.classification

    return Object3d(
        id_frame=o1.id_fame,
        instance_token=o1.instance_token or o2.instance_token,
        object_id=f"{o1.object_id}|{o2.object_id}",
        timestamp=o1.timestamp,
        xdistance=fused_x,
        ydistance=fused_y,
        zdistance=fused_z,
        width=fused_w,
        depth=fused_l,
        height=fused_h,
        orientation=fused_orientation,
        classification=fused_class,
        uncertainties=fused_unc,
        dt=o1.dt,
    )


def fuse_matches(
    setA: List[Object3d],
    setB: List[Object3d],
    matches: Iterable[Tuple[int, int]],
    fuse_fn: Callable[[Object3d, Object3d], Object3d] = fuse_average_boxes,
) -> List[Object3d]:
    """Fuse matched pairs using the provided fusion function."""
    fused: List[Object3d] = []
    for iA, iB in matches:
        fused.append(fuse_fn(setA[iA], setB[iB]))
    return fused
